"""Performance monitoring: per-query, per-agent and per-stage metrics.

Monitoring exists here for a specific reason beyond the requirement list: almost
every design decision in this system is a trade-off with an observable cost
(hybrid retrieval is slower than dense-only, wider candidate pools cost latency,
conflict detection is quadratic in claims). Without measurement those choices
are assertions.

Everything is in-process and dependency-free. A production deployment would
export the same records to a time-series backend; the shape of what is recorded
would not change.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class AgentRecord:
    """One domain agent's contribution to one query."""

    domain: str
    sub_query: str
    passages: int
    claims: int
    latency_ms: float
    abstained: bool = False
    failed: bool = False
    error: str = ""
    top_relevance: float = 0.0


@dataclass
class QueryRecord:
    """Everything measurable about one end-to-end query.

    Stage timings are kept separately from the total because the interesting
    question is never "was it slow" but "which stage was slow" -- and for this
    architecture the answer is usually either index building or conflict
    detection, which have very different fixes.
    """

    trace_id: str
    query: str
    routed_domains: Tuple[str, ...]
    expected_domains: Tuple[str, ...] = ()
    intent: str = ""
    complexity: float = 0.0
    routing_confidence: float = 0.0
    total_ms: float = 0.0
    stage_ms: Dict[str, float] = field(default_factory=dict)
    agents: List[AgentRecord] = field(default_factory=list)
    claims_used: int = 0
    citations: int = 0
    conflicts: int = 0
    unresolved_conflicts: int = 0
    answer_confidence: float = 0.0
    abstained: bool = False
    degraded: bool = False
    feedback: Optional[float] = None
    synthesizer: str = ""

    @property
    def routing_correct(self) -> Optional[bool]:
        """Exact-set routing correctness, or ``None`` with no ground truth."""
        if not self.expected_domains:
            return None
        return set(self.routed_domains) == set(self.expected_domains)

    @property
    def routing_precision(self) -> Optional[float]:
        if not self.expected_domains:
            return None
        if not self.routed_domains:
            return 0.0
        hits = len(set(self.routed_domains) & set(self.expected_domains))
        return hits / len(set(self.routed_domains))

    @property
    def routing_recall(self) -> Optional[float]:
        if not self.expected_domains:
            return None
        hits = len(set(self.routed_domains) & set(self.expected_domains))
        return hits / len(set(self.expected_domains))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the record for export or offline analysis."""
        payload = asdict(self)
        payload["routed_domains"] = list(self.routed_domains)
        payload["expected_domains"] = list(self.expected_domains)
        return payload


class MetricsRegistry:
    """Collects query records and aggregates them into a report.

    Thread-safe, because the orchestrator records agent results from worker
    threads.
    """

    def __init__(self) -> None:
        self._records: List[QueryRecord] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def record(self, record: QueryRecord) -> QueryRecord:
        """Append a completed query record."""
        with self._lock:
            self._records.append(record)
            return record

    @property
    def records(self) -> List[QueryRecord]:
        with self._lock:
            return list(self._records)

    def find(self, trace_id: str) -> Optional[QueryRecord]:
        """Return the record for one query, or ``None`` if unknown."""
        with self._lock:
            return next((r for r in self._records if r.trace_id == trace_id), None)

    def attach_feedback(self, trace_id: str, rating: float) -> bool:
        """Record a satisfaction rating in ``[0, 1]`` against a past query."""
        with self._lock:
            record = next((r for r in self._records if r.trace_id == trace_id), None)
            if record is None:
                return False
            record.feedback = float(max(0.0, min(1.0, rating)))
            return True

    def clear(self) -> None:
        """Drop every recorded query."""
        with self._lock:
            self._records.clear()

    # -- aggregation ------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics across every recorded query."""
        records = self.records
        if not records:
            return {"queries": 0}

        answered = [r for r in records if not r.abstained]
        labelled = [r for r in records if r.expected_domains]
        rated = [r for r in records if r.feedback is not None]
        latencies = [r.total_ms for r in records]

        stage_totals: Dict[str, List[float]] = defaultdict(list)
        for record in records:
            for stage, value in record.stage_ms.items():
                stage_totals[stage].append(value)

        return {
            "queries": len(records),
            "answered": len(answered),
            "abstained": sum(1 for r in records if r.abstained),
            "degraded": sum(1 for r in records if r.degraded),
            "answer_rate": round(len(answered) / len(records), 4),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "p95": round(self._percentile(latencies, 95), 2),
                "max": round(max(latencies), 2),
            },
            "stage_ms_mean": {
                stage: round(statistics.fmean(values), 2)
                for stage, values in sorted(stage_totals.items())
            },
            "mean_citations": round(
                statistics.fmean([r.citations for r in answered]), 2) if answered else 0.0,
            "mean_confidence": round(
                statistics.fmean([r.answer_confidence for r in answered]), 4) if answered else 0.0,
            "conflicts_detected": sum(r.conflicts for r in records),
            "conflicts_unresolved": sum(r.unresolved_conflicts for r in records),
            "routing": self._routing_summary(labelled),
            "feedback": {
                "rated_queries": len(rated),
                "mean_rating": round(statistics.fmean([r.feedback for r in rated]), 4)
                if rated else None,
            },
            "per_domain": self.per_domain(),
        }

    def _routing_summary(self, labelled: Sequence[QueryRecord]) -> Dict[str, Any]:
        if not labelled:
            return {"labelled_queries": 0}
        precisions = [r.routing_precision or 0.0 for r in labelled]
        recalls = [r.routing_recall or 0.0 for r in labelled]
        exact = sum(1 for r in labelled if r.routing_correct)
        precision = statistics.fmean(precisions)
        recall = statistics.fmean(recalls)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "labelled_queries": len(labelled),
            "exact_set_accuracy": round(exact / len(labelled), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def per_domain(self) -> Dict[str, Dict[str, Any]]:
        """Per-agent performance, which is what identifies a weak domain.

        A domain that abstains often has a coverage gap in its corpus; one that
        is slow has an index problem; one that is never routed to has a cue
        lexicon problem. The aggregate number hides all three.
        """
        buckets: Dict[str, List[AgentRecord]] = defaultdict(list)
        for record in self.records:
            for agent in record.agents:
                buckets[agent.domain].append(agent)

        report: Dict[str, Dict[str, Any]] = {}
        for domain, agents in sorted(buckets.items()):
            report[domain] = {
                "invocations": len(agents),
                "abstentions": sum(1 for a in agents if a.abstained),
                "failures": sum(1 for a in agents if a.failed),
                "mean_passages": round(statistics.fmean([a.passages for a in agents]), 2),
                "mean_claims": round(statistics.fmean([a.claims for a in agents]), 2),
                "mean_latency_ms": round(statistics.fmean([a.latency_ms for a in agents]), 2),
                "mean_top_relevance": round(
                    statistics.fmean([a.top_relevance for a in agents]), 4),
                "contribution_rate": round(
                    sum(1 for a in agents if not a.abstained and not a.failed) / len(agents), 4),
            }
        return report

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        """Nearest-rank percentile. Exact for the small samples used here."""
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1,
                    max(0, int(round((percentile / 100.0) * len(ordered) + 0.5)) - 1))
        return ordered[index]

    # -- presentation ----------------------------------------------------- #

    def report(self) -> str:
        """Render the summary as a fixed-width text table."""
        summary = self.summary()
        if not summary.get("queries"):
            return "No queries recorded."

        lines = ["Performance report", "=" * 68]
        lines.append(f"queries            {summary['queries']}  "
                     f"(answered {summary['answered']}, "
                     f"abstained {summary['abstained']}, "
                     f"degraded {summary['degraded']})")
        lat = summary["latency_ms"]
        lines.append(f"latency ms         mean {lat['mean']:<9} median {lat['median']:<9} "
                     f"p95 {lat['p95']:<9} max {lat['max']}")
        if summary["stage_ms_mean"]:
            lines.append("stage ms (mean)    " + "  ".join(
                f"{k}={v}" for k, v in summary["stage_ms_mean"].items()))
        lines.append(f"answer quality     mean confidence {summary['mean_confidence']}  "
                     f"mean citations {summary['mean_citations']}")
        lines.append(f"conflicts          {summary['conflicts_detected']} detected, "
                     f"{summary['conflicts_unresolved']} left unsettled")

        routing = summary["routing"]
        if routing.get("labelled_queries"):
            lines.append(f"routing            exact-set {routing['exact_set_accuracy']}  "
                         f"precision {routing['precision']}  recall {routing['recall']}  "
                         f"f1 {routing['f1']}  (n={routing['labelled_queries']})")
        feedback = summary["feedback"]
        if feedback.get("rated_queries"):
            lines.append(f"feedback           {feedback['rated_queries']} rated, "
                         f"mean rating {feedback['mean_rating']}")

        lines.append("")
        lines.append(f"{'domain':<12}{'calls':>6}{'absts':>7}{'fails':>7}"
                     f"{'passages':>10}{'claims':>8}{'ms':>9}{'top rel':>9}")
        lines.append("-" * 68)
        for domain, stats in summary["per_domain"].items():
            lines.append(
                f"{domain:<12}{stats['invocations']:>6}{stats['abstentions']:>7}"
                f"{stats['failures']:>7}{stats['mean_passages']:>10}"
                f"{stats['mean_claims']:>8}{stats['mean_latency_ms']:>9}"
                f"{stats['mean_top_relevance']:>9}")
        return "\n".join(lines)
