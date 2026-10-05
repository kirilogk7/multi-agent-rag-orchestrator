"""Domain-specific RAG agents and the context they share.

Each agent owns exactly one knowledge domain: its own vector-store namespace,
its own retrieval, its own claim extraction. It cannot see another domain's
documents, which is deliberate -- it means cross-domain coverage has to be
produced by coordination rather than by a lucky global search, and it keeps each
domain independently indexable and independently updatable.

The interesting behaviour is in ``_expand_query``. An agent reads the shared
blackboard before searching and folds the context other agents have discovered
into its own query. Concretely: asked "what's the process for deploying a
microservice", the technical agent retrieves the deployment runbook and notices
the deployment targets production. It writes ``environment=production`` to the
blackboard. The compliance agent then searches for *production* change control
rather than change control in general, and finds the CAB requirement that a
search on the user's original words would have ranked far lower.

That is the difference between three agents searching in parallel and three
agents actually cooperating, and it is why the orchestrator runs the primary
domain before the supporting ones rather than fanning all of them out at once.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .monitoring import AgentRecord
from .protocol import Blackboard, BROADCAST, Message, MessageBus, MessageType, ORCHESTRATOR
from .query_classifier import SubQuery
from .synthesis import Claim, ClaimExtractor
from .utils import RetrievedChunk, Stopwatch, get_logger, normalize_text, truncate
from .vector_store import KnowledgeBase, RetrievalParams

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Context extraction
# --------------------------------------------------------------------------- #

#: Context dimensions worth sharing between agents, as ``key -> {value: cues}``.
#:
#: These are the facts that change what a *different* domain should search for.
#: Deliberately a small, curated set rather than open-ended entity extraction:
#: a fact only earns a place on the blackboard if another agent would retrieve
#: differently because of it. "environment=production" changes which change-control
#: rules apply; "the service is written in Go" changes nothing for compliance.
CONTEXT_CUES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "environment": {
        "production": ("production", "prod ", "live traffic", "customer-facing"),
        "staging": ("staging", "pre-production", "test environment"),
    },
    "data_sensitivity": {
        "personal_data": ("personal data", "pii", "customer data", "data subject",
                          "special category", "gdpr"),
        "restricted": ("restricted", "payment credential", "authentication secret",
                       "credentials"),
        "internal": ("internal use", "internal only"),
    },
    "change_type": {
        "deployment": ("deploy", "deployment", "release", "rollout", "ship"),
        "new_workflow": ("new data processing", "new workflow", "implement a new",
                         "new pipeline"),
        "diagnosis": ("troubleshoot", "debug", "diagnose", "incident", "latency spike"),
        "procurement": ("vendor", "procure", "purchase", "contract", "subprocessor"),
    },
    "governing_body": {
        "change_advisory_board": ("change advisory board", "cab"),
        "data_governance_council": ("data governance council", "governance council"),
        "data_protection_officer": ("data protection officer", "dpo"),
        "security_team": ("security team",),
    },
}

#: Terms added to a sibling agent's query for a given context value. These are
#: the actual mechanism by which context sharing changes retrieval.
CONTEXT_EXPANSIONS: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("environment", "production"): ("production", "change control"),
    ("data_sensitivity", "personal_data"): ("personal data", "lawful basis", "retention"),
    ("data_sensitivity", "restricted"): ("restricted", "encryption", "access control"),
    ("change_type", "deployment"): ("change record", "approval", "release"),
    ("change_type", "new_workflow"): ("processing register", "impact assessment", "sign-off"),
    ("change_type", "diagnosis"): ("diagnostics", "logging", "payload"),
    ("change_type", "procurement"): ("due diligence", "data processing agreement"),
    ("governing_body", "change_advisory_board"): ("change advisory board approval",),
    ("governing_body", "data_governance_council"): ("council sign-off",),
}


class ContextExtractor:
    """Derives shareable context from a query and from retrieved passages.

    Cue-based rather than model-based, for the same reason the classifier uses a
    lexicon alongside its centroids: the extraction has to be inspectable, and
    a wrong fact on the blackboard silently misdirects every downstream agent.
    """

    def __init__(self, cues: Optional[Dict[str, Dict[str, Tuple[str, ...]]]] = None) -> None:
        self.cues = cues or CONTEXT_CUES

    def extract(self, text: str) -> Dict[str, str]:
        """Return the context dimensions present in ``text``.

        Where several values of one dimension match, the one whose cue appeared
        earliest wins, on the assumption that the first mention is the subject
        and later ones are qualifications.
        """
        normalized = normalize_text(text)
        found: Dict[str, str] = {}
        for dimension, values in self.cues.items():
            best: Optional[Tuple[int, str]] = None
            for value, cues in values.items():
                positions = [normalized.find(c) for c in cues]
                hits = [p for p in positions if p >= 0]
                if hits and (best is None or min(hits) < best[0]):
                    best = (min(hits), value)
            if best is not None:
                found[dimension] = best[1]
        return found

    def expansion_terms(self, context: Dict[str, Any], *, max_terms: int = 4) -> List[str]:
        """Translate blackboard context into a few extra query terms.

        Capped, because expansion is a recall boost bought with precision. An
        uncapped version measurably backfired: eight appended terms diluted the
        lexical signal enough to push the correct change-control policy out of
        the compliance agent's results altogether. ``context`` is expected in
        confidence order, so the cap keeps the best-supported facts.
        """
        terms: List[str] = []
        for dimension, value in context.items():
            for term in CONTEXT_EXPANSIONS.get((dimension, str(value)), ()):
                if term not in terms:
                    terms.append(term)
                if len(terms) >= max_terms:
                    return terms
        return terms


# --------------------------------------------------------------------------- #
# Agent contributions
# --------------------------------------------------------------------------- #

@dataclass
class AgentContribution:
    """What one domain agent produced for one sub-query.

    Carries enough detail to explain itself: the query as *actually issued*
    after context expansion, the retrieval parameters used, the passages found,
    the claims extracted, and which blackboard facts were read and written.
    """

    domain: str
    sub_query: str
    issued_query: str
    params: RetrievalParams
    passages: List[RetrievedChunk] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    latency_ms: float = 0.0
    abstained: bool = False
    failed: bool = False
    error: str = ""
    context_read: Dict[str, Any] = field(default_factory=dict)
    context_written: Dict[str, Any] = field(default_factory=dict)
    expansion_terms: Tuple[str, ...] = ()

    @property
    def confidence(self) -> float:
        """Best claim confidence, or zero when the agent contributed nothing."""
        return max((c.confidence for c in self.claims), default=0.0)

    @property
    def top_relevance(self) -> float:
        return max((p.relevance for p in self.passages), default=0.0)

    def to_record(self) -> AgentRecord:
        """Project into the monitoring record type."""
        return AgentRecord(
            domain=self.domain,
            sub_query=self.sub_query,
            passages=len(self.passages),
            claims=len(self.claims),
            latency_ms=round(self.latency_ms, 3),
            abstained=self.abstained,
            failed=self.failed,
            error=self.error,
            top_relevance=round(self.top_relevance, 4),
        )

    def describe(self) -> str:
        """Multi-line summary: what was searched, found, and shared."""
        if self.failed:
            return f"[{self.domain}] FAILED: {self.error}"
        if self.abstained:
            return (f"[{self.domain}] abstained -- no passage cleared the evidence floor "
                    f"({self.params.min_score:.2f})")
        lines = [
            f"[{self.domain}] {len(self.passages)} passage(s), {len(self.claims)} claim(s), "
            f"{self.latency_ms:.1f} ms",
            f"      issued: {truncate(self.issued_query, 100)}",
        ]
        if self.expansion_terms:
            lines.append(f"      expanded with shared context: "
                         f"{', '.join(self.expansion_terms)}")
        for passage in self.passages:
            lines.append(f"      - {passage.describe()}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Domain agent
# --------------------------------------------------------------------------- #

class DomainAgent:
    """A retrieval-augmented agent scoped to a single knowledge domain.

    Args:
        domain: The domain this agent owns. Must exist in the knowledge base.
        knowledge_base: Shared knowledge base; the agent only ever touches its
            own namespace.
        bus: Message bus for protocol traffic.
        blackboard: Shared context store.
        extractor: Claim extractor. Shared instances are safe -- it is stateless.
        context_extractor: Context extractor for blackboard writes.
        trust_weight: Learnable multiplier applied to this agent's claim
            confidences. Moved by orchestrator feedback; this is the mechanism by
            which a domain that keeps being wrong stops dominating synthesis.
    """

    def __init__(
        self,
        domain: str,
        knowledge_base: KnowledgeBase,
        *,
        bus: Optional[MessageBus] = None,
        blackboard: Optional[Blackboard] = None,
        extractor: Optional[ClaimExtractor] = None,
        context_extractor: Optional[ContextExtractor] = None,
        trust_weight: float = 1.0,
    ) -> None:
        knowledge_base.store(domain)  # fail fast on an unknown domain
        self.domain = domain
        self.knowledge_base = knowledge_base
        self.bus = bus
        self.blackboard = blackboard
        self.extractor = extractor or ClaimExtractor()
        self.context_extractor = context_extractor or ContextExtractor()
        self.trust_weight = trust_weight
        self._lock = threading.RLock()

        if bus is not None:
            bus.register(domain)

    @property
    def address(self) -> str:
        return self.domain

    # -- main entry point ------------------------------------------------- #

    def handle(
        self,
        sub_query: SubQuery,
        *,
        trace_id: str,
        parent_id: Optional[str] = None,
    ) -> AgentContribution:
        """Answer one sub-query, publishing context and protocol messages.

        Never raises on a retrieval failure. A crashed agent must degrade the
        answer, not destroy it -- the orchestrator needs the other agents'
        contributions far more than it needs a stack trace.
        """
        context = self._read_context(trace_id)
        terms = self.context_extractor.expansion_terms(context) if sub_query.params.expand_query else []
        issued = self._expand_query(sub_query.text, terms)

        contribution = AgentContribution(
            domain=self.domain,
            sub_query=sub_query.text,
            issued_query=issued,
            params=sub_query.params,
            context_read=context,
            expansion_terms=tuple(terms),
        )

        with Stopwatch() as timer:
            try:
                passages = self._retrieve(issued, sub_query.params)
                contribution.passages = list(passages)
                claims: List[Claim] = []
                for passage in passages:
                    claims.extend(self.extractor.extract(passage, issued))
                contribution.claims = self._apply_trust(claims)
                contribution.abstained = not contribution.claims
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                LOGGER.warning("agent %s failed: %s: %s", self.domain,
                               type(exc).__name__, exc)
                contribution.failed = True
                contribution.error = f"{type(exc).__name__}: {exc}"
        contribution.latency_ms = timer.elapsed_ms

        if not contribution.failed and contribution.passages:
            contribution.context_written = self._publish_context(contribution, trace_id)

        self._report(contribution, trace_id=trace_id, parent_id=parent_id)
        return contribution

    # -- steps ------------------------------------------------------------ #

    def _retrieve(self, query: str, params: RetrievalParams) -> List[RetrievedChunk]:
        """Search this agent's namespace.

        Isolated as its own method so tests can subclass and inject a failure
        without the production path carrying any test-only machinery.
        """
        return self.knowledge_base.search(query, self.domain, params)

    def _read_context(self, trace_id: str) -> Dict[str, Any]:
        """Read context contributed by *peer domain agents* only.

        The filter is the whole point, and it was added after measuring the
        alternative. Expanding on every fact on the board -- including the ones
        derived from the user's own query before any agent ran -- actively hurt
        the primary agent: for "what's the process for deploying a microservice",
        query-derived terms like "change record, approval, release" pushed the
        deployment runbook out of the technical agent's top results entirely and
        replaced it with secrets-management and database passages.

        That makes sense in hindsight. Terms re-derived from the query add no
        information the query did not already contain, while diluting the
        lexical signal that made the right document rank first. Only another
        agent's *discovery* is new information, so only that earns an expansion.

        A consequence worth noting: the first agent to run expands nothing and
        searches the user's words cleanly, which is why the orchestrator runs the
        primary domain before the supporting ones.
        """
        if self.blackboard is None:
            return {}
        from .utils import DOMAINS
        peers = {d for d in DOMAINS if d != self.domain}
        peer_facts = [f for f in self.blackboard.facts(trace_id) if f.author in peers]
        if not peer_facts:
            return {}
        # Highest-confidence fact per key, then keys ordered by that confidence,
        # so a capped expansion keeps the best-evidenced context.
        best: Dict[str, Any] = {}
        for fact in sorted(peer_facts, key=lambda f: f.confidence):
            best[fact.key] = (fact.confidence, fact.value)
        ordered = sorted(best.items(), key=lambda kv: -kv[1][0])
        return {key: value for key, (_, value) in ordered}

    @staticmethod
    def _expand_query(text: str, terms: Sequence[str]) -> str:
        """Append context terms not already present in the query.

        Appending rather than rewriting keeps the user's own wording dominant in
        the lexical ranking; the extra terms act as a recall boost, not a
        replacement query.
        """
        if not terms:
            return text
        normalized = normalize_text(text)
        extra = [t for t in terms if normalize_text(t) not in normalized]
        return f"{text} {' '.join(extra)}".strip() if extra else text

    def _apply_trust(self, claims: Sequence[Claim]) -> List[Claim]:
        """Scale claim confidence by this agent's learned trust weight."""
        if abs(self.trust_weight - 1.0) < 1e-9:
            return list(claims)
        from dataclasses import replace
        return [
            replace(claim, confidence=float(round(
                max(0.0, min(1.0, claim.confidence * self.trust_weight)), 4)))
            for claim in claims
        ]

    def _publish_context(self, contribution: AgentContribution,
                         trace_id: str) -> Dict[str, Any]:
        """Write context discovered in the retrieved passages to the blackboard.

        Confidence is tied to the strength of the evidence it was drawn from, so
        a fact inferred from a weak passage cannot outvote one drawn from a
        strong passage on the same dimension.
        """
        if self.blackboard is None:
            return {}
        # Context is only as trustworthy as the passage it was read from. An
        # agent's lower-ranked passages are frequently off-topic -- asserting
        # facts from them puts noise on the board that every peer then expands
        # on. The bar is deliberately high: strong passages only.
        evidence_passages = self._context_evidence(contribution.passages)
        if not evidence_passages:
            return {}
        evidence = " ".join(p.chunk.text for p in evidence_passages)
        discovered = self.context_extractor.extract(evidence)
        if not discovered:
            return {}

        confidence = min(1.0, 0.45 + 0.55 * contribution.top_relevance)
        self.blackboard.write_many(trace_id, discovered, author=self.domain,
                                   confidence=confidence)
        if self.bus is not None:
            self.bus.send(
                self.domain, BROADCAST, MessageType.PARTIAL,
                {"key": ", ".join(f"{k}={v}" for k, v in discovered.items()),
                 "context": discovered, "confidence": round(confidence, 3)},
                trace_id=trace_id,
            )
        return discovered

    @staticmethod
    def _context_evidence(
        passages: Sequence[RetrievedChunk],
        *,
        absolute_floor: float = 0.45,
        relative_floor: float = 0.80,
        limit: int = 2,
    ) -> List[RetrievedChunk]:
        """Select the passages strong enough to assert shared context from.

        Both floors matter. The absolute one stops a uniformly weak result set
        from asserting anything; the relative one stops a strong top hit from
        dragging its mediocre neighbours in with it.
        """
        if not passages:
            return []
        top = max(p.relevance for p in passages)
        floor = max(absolute_floor, relative_floor * top)
        return [p for p in passages if p.relevance >= floor][:limit]

    def _report(self, contribution: AgentContribution, *, trace_id: str,
                parent_id: Optional[str]) -> None:
        """Send the terminal protocol message for this contribution."""
        if self.bus is None:
            return
        if contribution.failed:
            kind, payload = MessageType.FAILURE, {"reason": contribution.error}
        elif contribution.abstained:
            kind, payload = MessageType.ABSTAIN, {
                "reason": f"no passage above relevance floor "
                          f"{contribution.params.min_score:.2f}"}
        else:
            kind, payload = MessageType.RESULT, {
                "reason": f"{len(contribution.passages)} passage(s), "
                          f"{len(contribution.claims)} claim(s)",
                "passages": [p.doc_id for p in contribution.passages],
                "confidence": round(contribution.confidence, 3),
            }
        self.bus.send(self.domain, ORCHESTRATOR, kind, payload,
                      trace_id=trace_id, parent_id=parent_id)

    # -- learning --------------------------------------------------------- #

    def adjust_trust(self, delta: float, *, lower: float = 0.7,
                     upper: float = 1.25) -> float:
        """Nudge this agent's trust weight, clamped to a sane band.

        Clamped because trust is a tie-breaker, not a veto: an agent that drifts
        to zero trust would be silently excluded from every future answer on the
        strength of a handful of ratings.
        """
        with self._lock:
            self.trust_weight = float(max(lower, min(upper, self.trust_weight + delta)))
            return self.trust_weight

    def stats(self) -> Dict[str, Any]:
        """Namespace statistics plus this agent's learned state."""
        payload = self.knowledge_base.store(self.domain).stats()
        payload["trust_weight"] = round(self.trust_weight, 4)
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"DomainAgent(domain={self.domain!r}, "
                f"documents={len(self.knowledge_base.documents(self.domain))}, "
                f"trust={self.trust_weight:.2f})")


def build_agents(
    knowledge_base: KnowledgeBase,
    *,
    bus: Optional[MessageBus] = None,
    blackboard: Optional[Blackboard] = None,
    domains: Optional[Sequence[str]] = None,
    extractor: Optional[ClaimExtractor] = None,
) -> Dict[str, DomainAgent]:
    """Construct one agent per domain, sharing bus, blackboard and extractor."""
    from .utils import DOMAINS
    shared_extractor = extractor or ClaimExtractor()
    shared_context = ContextExtractor()
    return {
        domain: DomainAgent(
            domain, knowledge_base, bus=bus, blackboard=blackboard,
            extractor=shared_extractor, context_extractor=shared_context,
        )
        for domain in (domains or DOMAINS)
    }
