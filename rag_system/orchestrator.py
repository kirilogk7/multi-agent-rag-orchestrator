"""The orchestrator: planning, agent coordination, synthesis, learning.

Execution is deliberately *two-phase* rather than a flat fan-out, and that is
the single most consequential design decision in the file.

A flat fan-out is simpler and faster: dispatch every domain agent at once, wait,
merge. But agents that start simultaneously cannot learn anything from each
other, which reduces "multi-agent" to "several independent searches with a
shared output format". So instead:

* **Phase 1** runs the primary domain alone. It searches the user's own words
  with nothing appended, and publishes what it discovered to the blackboard.
* **Phase 2** fans out every supporting domain concurrently, each one reading
  phase-1 context and specialising its own retrieval before searching.

The cost is one extra round trip. The benefit is measurable: for the deployment
question, the compliance agent's top hit moves from a generic audit-evidence
passage to the production change-control policy that actually answers it, because
phase 1 established that the deployment targets production.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain_agents import AgentContribution, ContextExtractor, DomainAgent, build_agents
from .monitoring import MetricsRegistry, QueryRecord
from .protocol import Blackboard, CLASSIFIER, MessageBus, MessageType, ORCHESTRATOR
from .query_classifier import Classification, QueryClassifier, SubQuery
from .synthesis import (
    Answer,
    Claim,
    ClaimExtractor,
    ClaudeSynthesizer,
    Conflict,
    ConflictResolver,
    ExtractiveSynthesizer,
    Synthesizer,
    verify_citations,
)
from .utils import DOMAINS, Document, Stopwatch, get_logger, load_corpus, truncate
from .vector_store import KnowledgeBase

LOGGER = get_logger(__name__)

#: Query-derived context is attributed to this author so that domain agents can
#: tell it apart from a peer agent's genuine discovery. See
#: ``DomainAgent._read_context`` for why that distinction matters.
QUERY_SEED_AUTHOR = "query_seed"


@dataclass(frozen=True)
class ExecutionPlan:
    """What the orchestrator decided to do, before it did it.

    Materialised as its own object so a plan can be inspected, logged and
    asserted against in tests without executing it.
    """

    trace_id: str
    query: str
    classification: Classification
    sub_queries: Tuple[SubQuery, ...]
    seeded_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> Optional[SubQuery]:
        return self.sub_queries[0] if self.sub_queries else None

    @property
    def supporting(self) -> Tuple[SubQuery, ...]:
        return self.sub_queries[1:]

    def describe(self) -> str:
        """Full rendering of the plan: routing, phases and per-domain budgets."""
        lines = [f"Execution plan (trace {self.trace_id})", self.classification.describe()]
        if self.seeded_context:
            lines.append("seeded ctx : " + ", ".join(
                f"{k}={v}" for k, v in self.seeded_context.items()))
        if not self.sub_queries:
            lines.append("plan       : abstain -- no domain cleared the confidence floor")
            return "\n".join(lines)
        lines.append(f"plan       : phase 1 -> {self.primary.domain}; "
                     f"phase 2 -> {', '.join(s.domain for s in self.supporting) or '(none)'}")
        lines.append("sub-queries:")
        lines.extend("  " + s.describe() for s in self.sub_queries)
        return "\n".join(lines)


class RAGOrchestrator:
    """Coordinates classification, domain agents, conflict resolution, synthesis.

    Args:
        knowledge_base: Indexed corpus. Built from the bundled synthetic corpus
            when omitted.
        classifier: Query classifier. Fitted against the knowledge base when
            omitted.
        synthesizer: Answer synthesiser. Defaults to the deterministic
            extractive one; pass ``ClaudeSynthesizer()`` for generative prose.
        resolver: Conflict resolver.
        max_workers: Concurrency for phase 2.
        agent_timeout: Seconds a supporting agent may take before the
            orchestrator gives up on it and degrades the answer. A hung agent
            must not hang the query.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        *,
        classifier: Optional[QueryClassifier] = None,
        synthesizer: Optional[Synthesizer] = None,
        resolver: Optional[ConflictResolver] = None,
        agents: Optional[Dict[str, DomainAgent]] = None,
        max_workers: int = 3,
        agent_timeout: float = 20.0,
    ) -> None:
        self.knowledge_base = knowledge_base or KnowledgeBase(load_corpus())
        self.bus = MessageBus()
        self.blackboard = Blackboard()
        self.metrics = MetricsRegistry()
        self.context_extractor = ContextExtractor()
        self.resolver = resolver or ConflictResolver()
        self.synthesizer = synthesizer or ExtractiveSynthesizer()
        self.max_workers = max_workers
        self.agent_timeout = agent_timeout

        self.agents = agents or build_agents(
            self.knowledge_base, bus=self.bus, blackboard=self.blackboard
        )
        self.classifier = classifier or QueryClassifier().fit(self.knowledge_base)
        self.bus.register(ORCHESTRATOR)

    # -- planning --------------------------------------------------------- #

    def plan(self, query: str, *, trace_id: Optional[str] = None) -> ExecutionPlan:
        """Classify a query and decide which agents to run with what budget.

        Raises:
            ValueError: If the query is empty or whitespace only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        trace_id = trace_id or uuid.uuid4().hex[:12]
        classification = self.classifier.classify(query)
        sub_queries = self.classifier.decompose(classification)

        seeded = self.context_extractor.extract(query)
        if seeded:
            self.blackboard.write_many(trace_id, seeded,
                                       author=QUERY_SEED_AUTHOR, confidence=0.5)
        self.bus.send(CLASSIFIER, ORCHESTRATOR, MessageType.RESULT,
                      {"reason": f"routed to {', '.join(classification.domains) or 'none'}",
                       "intent": classification.intent.value,
                       "complexity": classification.complexity,
                       "confidence": round(classification.confidence, 3)},
                      trace_id=trace_id)

        return ExecutionPlan(
            trace_id=trace_id,
            query=query.strip(),
            classification=classification,
            sub_queries=tuple(sub_queries),
            seeded_context=seeded,
        )

    # -- execution -------------------------------------------------------- #

    def answer(
        self,
        query: str,
        *,
        expected_domains: Sequence[str] = (),
        trace_id: Optional[str] = None,
        plan: Optional[ExecutionPlan] = None,
    ) -> Answer:
        """Answer a query end to end.

        Args:
            query: The user's question.
            expected_domains: Optional ground-truth routing labels. Recorded for
                monitoring and used by ``apply_feedback``; they do not influence
                the answer.
            trace_id: Supply to correlate with an externally generated id.
            plan: A plan already produced by ``plan()``. Pass it when the plan
                was inspected first, so planning is not repeated -- re-planning
                would duplicate the classifier's protocol messages in the
                transcript and double-seed the blackboard.

        Returns:
            An ``Answer``, which may be an abstention. Abstaining is a success
            path, not a failure: it is what the system does instead of answering
            a question its corpus cannot support.

        Raises:
            ValueError: If the query is empty.
        """
        stages: Dict[str, float] = {}

        with Stopwatch() as total:
            with Stopwatch() as timer:
                if plan is None:
                    plan = self.plan(query, trace_id=trace_id)
            stages["classify"] = round(timer.elapsed_ms, 3)

            if not plan.sub_queries:
                answer = self._abstain(plan, stages, total)
                return answer

            with Stopwatch() as timer:
                contributions = self._run_agents(plan)
            stages["retrieve"] = round(timer.elapsed_ms, 3)

            claims = [c for contribution in contributions for c in contribution.claims]
            degraded = any(c.failed for c in contributions)
            notes = self._notes(contributions)

            with Stopwatch() as timer:
                conflicts = self.resolver.detect(claims)
            stages["resolve_conflicts"] = round(timer.elapsed_ms, 3)

            with Stopwatch() as timer:
                answer = self.synthesizer.synthesize(
                    plan.query, claims, conflicts,
                    domains=list(plan.classification.domains),
                    notes=notes, degraded=degraded,
                )
            stages["synthesize"] = round(timer.elapsed_ms, 3)

        answer.trace_id = plan.trace_id
        answer.timings_ms = {**stages, "total": round(total.elapsed_ms, 3)}
        self._record(plan, contributions, answer, expected_domains, stages,
                     total.elapsed_ms)
        return answer

    def _run_agents(self, plan: ExecutionPlan) -> List[AgentContribution]:
        """Phase 1 primary, then phase 2 supporting agents concurrently."""
        contributions: List[AgentContribution] = []

        primary = plan.primary
        assert primary is not None  # guarded by the caller
        request = self.bus.send(
            ORCHESTRATOR, primary.domain, MessageType.REQUEST,
            {"sub_query": primary.text, "phase": 1, "params": primary.params.describe()},
            trace_id=plan.trace_id,
        )
        contributions.append(
            self.agents[primary.domain].handle(
                primary, trace_id=plan.trace_id, parent_id=request.message_id)
        )

        supporting = plan.supporting
        if not supporting:
            return contributions

        requests = {
            sub.domain: self.bus.send(
                ORCHESTRATOR, sub.domain, MessageType.REQUEST,
                {"sub_query": sub.text, "phase": 2, "params": sub.params.describe()},
                trace_id=plan.trace_id,
            )
            for sub in supporting
        }

        # Collected in plan order rather than completion order: the answer must
        # not change shape because one agent happened to finish first.
        results: Dict[str, AgentContribution] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(supporting))) as pool:
            futures = {
                pool.submit(
                    self.agents[sub.domain].handle, sub,
                    trace_id=plan.trace_id, parent_id=requests[sub.domain].message_id,
                ): sub
                for sub in supporting
            }
            for future in as_completed(futures, timeout=None):
                sub = futures[future]
                try:
                    results[sub.domain] = future.result(timeout=self.agent_timeout)
                except Exception as exc:  # noqa: BLE001 - degrade, never abort
                    LOGGER.warning("supporting agent %s did not complete: %s",
                                   sub.domain, exc)
                    results[sub.domain] = AgentContribution(
                        domain=sub.domain, sub_query=sub.text, issued_query=sub.text,
                        params=sub.params, failed=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self.bus.send(sub.domain, ORCHESTRATOR, MessageType.FAILURE,
                                  {"reason": str(exc)}, trace_id=plan.trace_id)

        contributions.extend(results[sub.domain] for sub in supporting if sub.domain in results)
        return contributions

    @staticmethod
    def _notes(contributions: Sequence[AgentContribution]) -> List[str]:
        """Caveats that belong in the answer rather than only in the logs."""
        notes: List[str] = []
        for contribution in contributions:
            if contribution.failed:
                notes.append(f"the {contribution.domain} agent failed "
                             f"({contribution.error}); its domain is unrepresented")
            elif contribution.abstained:
                notes.append(f"the {contribution.domain} agent found no passage above its "
                             f"evidence floor and contributed nothing")
            elif contribution.expansion_terms:
                notes.append(f"the {contribution.domain} agent refined its search using "
                             f"context from another agent: "
                             f"{', '.join(contribution.expansion_terms)}")
        return notes

    def _abstain(self, plan: ExecutionPlan, stages: Dict[str, float],
                 total: Stopwatch) -> Answer:
        """Build an abstention for an out-of-domain query."""
        answer = self.synthesizer.synthesize(
            plan.query, [], [], domains=[],
            notes=["no domain cleared the routing confidence floor; "
                   "the question appears to fall outside the indexed domains"],
        )
        answer.trace_id = plan.trace_id
        answer.timings_ms = {**stages, "total": round(total.elapsed_ms, 3)}
        self.bus.send(ORCHESTRATOR, ORCHESTRATOR, MessageType.ABSTAIN,
                      {"reason": "out of domain"}, trace_id=plan.trace_id)
        self.metrics.record(QueryRecord(
            trace_id=plan.trace_id, query=plan.query, routed_domains=(),
            intent=plan.classification.intent.value,
            complexity=plan.classification.complexity,
            routing_confidence=plan.classification.confidence,
            total_ms=round(total.elapsed_ms, 3), stage_ms=dict(stages),
            abstained=True, synthesizer=self.synthesizer.name,
        ))
        return answer

    def _record(
        self,
        plan: ExecutionPlan,
        contributions: Sequence[AgentContribution],
        answer: Answer,
        expected_domains: Sequence[str],
        stages: Dict[str, float],
        total_ms: float,
    ) -> QueryRecord:
        return self.metrics.record(QueryRecord(
            trace_id=plan.trace_id,
            query=plan.query,
            routed_domains=plan.classification.domains,
            expected_domains=tuple(expected_domains),
            intent=plan.classification.intent.value,
            complexity=plan.classification.complexity,
            routing_confidence=plan.classification.confidence,
            total_ms=round(total_ms, 3),
            stage_ms=dict(stages),
            agents=[c.to_record() for c in contributions],
            claims_used=len(answer.claims),
            citations=len(answer.citations),
            conflicts=len(answer.conflicts),
            unresolved_conflicts=sum(1 for c in answer.conflicts if not c.decisive),
            answer_confidence=answer.confidence,
            abstained=answer.abstained,
            degraded=answer.degraded,
            synthesizer=answer.synthesizer,
        ))

    # -- knowledge updates ------------------------------------------------ #

    def ingest(self, document: Document, *, refit_classifier: bool = True) -> Dict[str, Any]:
        """Add a document to the live knowledge base.

        Args:
            document: The document to index.
            refit_classifier: Re-fit the classifier's domain centroids against
                the updated corpus. On by default and worth the cost: a new
                document changes what a domain *is*, so leaving the centroids
                stale means the routing reflects a corpus that no longer exists.

        Returns:
            The ingestion record, including which document it retired.
        """
        record = self.knowledge_base.ingest(document, rebuild=True)
        if refit_classifier:
            self.classifier.fit(self.knowledge_base)
        LOGGER.info("ingested %s v%s into %s (retired %s)", document.doc_id,
                    document.version, document.domain, record.get("retired") or "nothing")
        return record

    # -- learning --------------------------------------------------------- #

    def apply_feedback(
        self,
        trace_id: str,
        rating: float,
        *,
        expected_domains: Sequence[str] = (),
        learning_rate: float = 0.08,
    ) -> Dict[str, Any]:
        """Fold an outcome signal back into routing, trust and retrieval.

        Three parameters move, each for a different reason:

        * **Routing weights** -- corrected against ``expected_domains``, so a
          query routed to the wrong domain becomes less likely to repeat.
        * **Agent trust** -- agents that contributed to a well-rated answer gain
          trust; those that contributed to a poorly-rated one lose it. This
          feeds claim confidence and therefore conflict resolution.
        * **Retrieval alpha** -- a poor rating nudges the domain's lexical/dense
          balance, on the theory that a bad answer from confident routing is
          more likely a retrieval-mix problem than a routing problem.

        Every update is bounded and inspectable. That is a deliberate choice over
        something more powerful: with a handful of ratings, a transparent rule
        that can be reasoned about beats an opaque one that cannot be debugged.

        Args:
            trace_id: The query being rated.
            rating: Satisfaction in ``[0, 1]``.
            expected_domains: Ground-truth routing, if known.
            learning_rate: Step size.

        Returns:
            The resulting learned state.

        Raises:
            KeyError: If ``trace_id`` was never recorded.
        """
        record = self.metrics.find(trace_id)
        if record is None:
            raise KeyError(f"no recorded query with trace_id {trace_id!r}")

        rating = float(max(0.0, min(1.0, rating)))
        self.metrics.attach_feedback(trace_id, rating)
        signal = (rating - 0.5) * 2.0  # map [0,1] -> [-1,+1]

        labels = tuple(expected_domains) or record.expected_domains
        if labels:
            classification = self.classifier.classify(record.query)
            self.classifier.apply_feedback(classification, labels,
                                           learning_rate=learning_rate)

        for agent_record in record.agents:
            agent = self.agents.get(agent_record.domain)
            if agent is None or agent_record.failed:
                continue
            if agent_record.abstained:
                # Abstaining on a well-rated answer is correct behaviour, not a
                # shortcoming, so it is not penalised.
                continue
            agent.adjust_trust(signal * learning_rate)

        if signal < 0:
            for domain in record.routed_domains:
                self.classifier.policy.nudge_domain_alpha(domain, signal * learning_rate)

        self.bus.send(ORCHESTRATOR, ORCHESTRATOR, MessageType.FEEDBACK,
                      {"reason": f"rating {rating:.2f}", "signal": round(signal, 3)},
                      trace_id=trace_id)
        return self.learned_state()

    def replay_feedback(
        self,
        labelled: Sequence[Tuple[str, Sequence[str]]],
        *,
        rounds: int = 1,
        learning_rate: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """Train routing on labelled queries and report accuracy per round.

        Returns one record per round with routing accuracy measured *before* the
        round's updates are applied, so the sequence shows the learning curve
        rather than a single after-the-fact number.
        """
        history: List[Dict[str, Any]] = []
        for round_index in range(rounds):
            correct = 0
            for query, expected in labelled:
                classification = self.classifier.classify(query)
                if set(classification.domains) == set(expected):
                    correct += 1
                else:
                    self.classifier.apply_feedback(classification, expected,
                                                   learning_rate=learning_rate)
            history.append({
                "round": round_index + 1,
                "accuracy": round(correct / len(labelled), 4) if labelled else 0.0,
                "routing_weights": dict(self.classifier.routing_weights),
            })
        return history

    def learned_state(self) -> Dict[str, Any]:
        """Every parameter the learning mechanism can move."""
        return {
            "routing_weights": dict(self.classifier.routing_weights),
            "domain_alpha": dict(self.classifier.policy.domain_alpha),
            "agent_trust": {d: round(a.trust_weight, 4) for d, a in self.agents.items()},
        }

    def save_learned_state(self, path: Path) -> Path:
        """Persist learned state as inspectable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.learned_state(), indent=2), encoding="utf-8")
        return path

    # -- introspection ---------------------------------------------------- #

    def verify(self, answer: Answer) -> List[Dict[str, Any]]:
        """Re-check every citation in an answer against its source document."""
        return verify_citations(answer, self.knowledge_base.document)

    def explain(self, trace_id: str) -> str:
        """Render the full inter-agent transcript for one query."""
        return self.bus.render_trace(trace_id)

    def stats(self) -> Dict[str, Any]:
        """Corpus, agent and metrics statistics in one payload."""
        return {
            "knowledge_base": self.knowledge_base.stats(),
            "agents": {d: a.stats() for d, a in self.agents.items()},
            "synthesizer": self.synthesizer.name,
            "queries_recorded": len(self.metrics),
            "learned_state": self.learned_state(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        stats = self.knowledge_base.stats()
        return (f"RAGOrchestrator(domains={list(self.agents)}, "
                f"documents={stats['documents']}, passages={stats['passages']}, "
                f"synthesizer={self.synthesizer.name!r})")


def build_default_orchestrator(*, use_claude: bool = False) -> RAGOrchestrator:
    """Construct an orchestrator over the bundled synthetic corpus.

    Args:
        use_claude: Use the generative synthesiser when an API key is present.
            Falls back to the extractive one otherwise, so this is always safe
            to pass.
    """
    knowledge_base = KnowledgeBase(load_corpus())
    synthesizer: Synthesizer
    if use_claude and ClaudeSynthesizer.available():
        synthesizer = ClaudeSynthesizer()
    else:
        if use_claude:
            LOGGER.info("no Anthropic credential found; using the extractive synthesizer")
        synthesizer = ExtractiveSynthesizer()
    return RAGOrchestrator(knowledge_base, synthesizer=synthesizer)
