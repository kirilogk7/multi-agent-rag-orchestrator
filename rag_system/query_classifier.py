"""Query understanding: domain routing, intent, complexity, decomposition.

The classifier answers four questions about an incoming query, and each answer
changes what the system does next:

1. **Which domains are needed?** Multi-label, not single-label. Every realistic
   question in this corpus spans two domains -- "how do I deploy *and* what
   compliance checks apply" is one question with two owners -- so a top-1
   classifier would be wrong on the assignment's own test scenarios.
2. **What kind of question is it?** A procedural "how do I" wants steps; a
   policy "what is required" wants exact figures and prohibitions. The intent
   changes the retrieval weighting and the shape of the synthesised answer.
3. **How complex is it?** Complexity drives retrieval depth. Spending a wide
   candidate pool on "what is our log retention" is waste; spending a narrow one
   on a three-part question loses evidence.
4. **Should it be split?** Decomposition turns one multi-part query into per-domain
   sub-queries, so each agent searches for what it actually owns rather than
   searching for the whole sentence and matching the irrelevant half.

Scoring combines two independent signals -- a *learned* centroid similarity from
the indexed corpus and a *hand-authored* cue lexicon prior. Neither alone is
enough: centroids drift when a domain's corpus is small, and lexicons are blind
to paraphrase. The blend is also what gives the learning mechanism something to
adjust, via per-domain routing weights.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import (
    DOMAINS,
    _stem as _stem_token,
    cosine_similarity,
    get_logger,
    normalize_text,
    tokenize,
    truncate,
)
from .vector_store import KnowledgeBase, RetrievalParams

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #

class QueryIntent(str, Enum):
    """What kind of answer the query is asking for."""

    PROCEDURAL = "procedural"      # "what's the process for...", "how do I deploy"
    DIAGNOSTIC = "diagnostic"      # "why is X slow", "how do I troubleshoot"
    REQUIREMENT = "requirement"    # "what is required", "what checks are needed"
    APPROVAL = "approval"          # "who approves", "what approvals are needed"
    DEFINITIONAL = "definitional"  # "what is a DPIA"
    COMPARATIVE = "comparative"    # "should we build or buy"


#: Surface cues per intent. Ordered by specificity: the first intent whose cues
#: fire wins, so "what approvals are needed to deploy" classifies as APPROVAL
#: rather than the broader REQUIREMENT.
INTENT_CUES: Tuple[Tuple[QueryIntent, Tuple[str, ...]], ...] = (
    (QueryIntent.APPROVAL, ("approval", "approve", "approver", "sign-off", "sign off",
                            "authorise", "authorize", "who signs", "who can approve")),
    (QueryIntent.DIAGNOSTIC, ("troubleshoot", "debug", "diagnose", "why is", "why are",
                              "root cause", "slow", "failing", "degraded", "investigate",
                              "latency spike", "outage")),
    (QueryIntent.COMPARATIVE, ("versus", "vs", "compare", "trade-off", "tradeoff",
                               "build or buy", "better than", "difference between")),
    (QueryIntent.PROCEDURAL, ("process for", "how do i", "how to", "steps", "procedure",
                              "runbook", "workflow for", "what's the process",
                              "what is the process", "implement", "deploy")),
    (QueryIntent.REQUIREMENT, ("required", "requirement", "needed", "must", "mandatory",
                               "checks", "obligation", "policy on", "rules for",
                               "am i allowed", "permitted")),
    (QueryIntent.DEFINITIONAL, ("what is", "what are", "define", "definition of",
                                "meaning of", "explain")),
)

#: Hand-authored routing prior. These are the terms a domain owner would name as
#: unambiguously theirs. Weights separate strong signals ("dpia" is only ever
#: compliance) from supporting ones ("service" leans technical but is generic).
DOMAIN_CUES: Dict[str, Dict[str, float]] = {
    "technical": {
        "deploy": 2.0, "deployment": 2.0, "microservice": 2.5, "canary": 2.0,
        "rollback": 1.8, "api": 1.8, "latency": 2.2, "performance": 1.8,
        "troubleshoot": 2.0, "debug": 1.8, "database": 1.8, "query": 1.2,
        "kubernetes": 2.2, "container": 1.8, "pipeline": 1.5, "cache": 1.8,
        "timeout": 1.8, "observability": 2.0, "trace": 1.5, "tracing": 1.8,
        "logs": 1.2, "incident": 1.5, "build": 1.2, "endpoint": 1.8,
        "throughput": 1.8, "scaling": 1.8, "replica": 1.8, "index": 1.5,
        "architecture": 1.3, "runbook": 1.3, "infrastructure": 1.5,
        "terraform": 2.0, "p99": 2.5, "schema": 1.3, "retry": 1.5,
        "connection": 1.5, "monitoring": 1.3, "service": 1.0, "code": 1.2,
    },
    "business": {
        "approval": 2.0, "approve": 2.0, "approver": 2.0, "budget": 2.2,
        "cost": 1.8, "spend": 2.0, "procurement": 2.5, "vendor": 1.8,
        "contract": 1.8, "headcount": 2.5, "hiring": 2.0, "roi": 2.2,
        "business": 1.8, "stakeholder": 2.0, "prioritisation": 2.0,
        "prioritization": 2.0, "intake": 1.8, "kpi": 2.2, "council": 1.8,
        "authority": 1.8, "delegation": 2.0, "finance": 2.2, "investment": 2.0,
        "sla": 1.8, "forecast": 2.0, "procure": 2.0, "purchase": 2.0,
        "sign-off": 2.0, "governance": 1.5, "eur": 1.5, "committee": 1.5,
        "benefit": 1.5, "capacity": 1.3, "roadmap": 1.8, "showback": 2.2,
    },
    "compliance": {
        "compliance": 2.5, "gdpr": 2.5, "privacy": 2.2, "dpia": 2.5,
        "retention": 2.0, "audit": 2.2, "policy": 1.6, "policies": 1.6,
        "security": 1.8, "regulation": 2.2, "regulatory": 2.2, "consent": 2.2,
        "breach": 2.2, "segregation": 2.2, "encryption": 2.0, "lawful": 2.5,
        "residency": 2.2, "subprocessor": 2.5, "classification": 1.8,
        "personal": 1.8, "pii": 2.5, "control": 1.5, "controls": 1.5,
        "evidence": 1.8, "cab": 2.0, "soc2": 2.5, "attestation": 2.2,
        "least privilege": 2.2, "access": 1.3, "confidential": 2.0,
        "restricted": 1.8, "anonymisation": 2.2, "data protection": 2.5,
        "change control": 2.2, "exception": 1.5, "risk": 1.3, "legal": 1.8,
        # Permission-seeking phrasing. A question of the form "can I do X" is a
        # question about what policy allows, even when every other word in it is
        # technical. Without these cues "Can I enable verbose request tracing in
        # production?" routed to technical alone and was answered with the runbook
        # that says how to do it -- never surfacing the security policy that
        # forbids it. That is precisely the failure conflict resolution exists to
        # prevent, so the routing has to reach compliance in the first place.
        # Weighted modestly: these are hints, not domain ownership.
        "am i allowed": 2.2, "allowed to": 1.6, "may i": 1.6,
        "is it ok": 1.6, "permitted": 1.8, "can i": 1.1,
    },
}

#: Phrases that mark a clause boundary in a compound question. Order matters:
#: longer patterns are tried first so "as well as" is not split on "as".
CLAUSE_SPLITTERS: Tuple[str, ...] = (
    r"\s+while\s+(?:also\s+)?", r"\s+as\s+well\s+as\s+", r"\s*,\s*and\s+",
    r"\s*;\s*", r"\s+and\s+(?:what|which|who|how|whether|any|the\s+\w+\s+)",
    r"\s+and\s+also\s+", r"\s*\?\s*(?=[A-Za-z])", r"\s+but\s+",
)
_CLAUSE_RE = re.compile("|".join(CLAUSE_SPLITTERS), flags=re.IGNORECASE)

_WH_RE = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how)\b", re.IGNORECASE)

#: Words that frame a question without naming its subject. Excluded when
#: anchoring a clause, because carrying "process" or "needed" across to a
#: sibling clause adds no retrievable signal.
META_TERMS: frozenset = frozenset({
    "process", "processes", "step", "steps", "procedure", "approach", "way",
    "thing", "things", "check", "checks", "need", "needed", "needs", "require",
    "required", "requirement", "requirements", "following", "follow", "also",
    "well", "place", "use", "used", "doing", "get", "getting", "make", "take",
    "what's", "whats", "how's", "it's", "there's", "let's",
})


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DomainScore:
    """Per-domain routing evidence, kept decomposed for explainability.

    A single blended number tells you a route was chosen but not why. Keeping
    the centroid and cue components separate means a bad route can be diagnosed
    as "the lexicon misfired" or "the corpus centroid is too broad" rather than
    just "the classifier was wrong".
    """

    domain: str
    score: float
    centroid_similarity: float
    cue_score: float
    matched_cues: Tuple[str, ...]
    routing_weight: float = 1.0

    def describe(self) -> str:
        """One-line rendering of this domain's routing evidence."""
        cues = ", ".join(self.matched_cues[:5]) or "-"
        return (f"{self.domain:<11} {self.score:.3f} "
                f"(centroid={self.centroid_similarity:.3f}, cues={self.cue_score:.3f} "
                f"[{cues}], w={self.routing_weight:.2f})")


@dataclass(frozen=True)
class Classification:
    """The classifier's full read on a query.

    Attributes:
        query: Original query text.
        selected: Domains judged necessary, best first.
        scores: Every domain's score, for inspection and for metrics.
        intent: Dominant question type.
        complexity: Normalised ``[0, 1]`` difficulty estimate.
        clauses: Clause segmentation used for decomposition.
        out_of_domain: True when no domain clears the confidence floor, which
            instructs the orchestrator to abstain rather than route to a
            best-of-bad-options agent.
    """

    query: str
    selected: Tuple[DomainScore, ...]
    scores: Tuple[DomainScore, ...]
    intent: QueryIntent
    complexity: float
    clauses: Tuple[str, ...]
    out_of_domain: bool = False

    @property
    def domains(self) -> Tuple[str, ...]:
        return tuple(d.domain for d in self.selected)

    @property
    def is_multi_domain(self) -> bool:
        return len(self.selected) > 1

    @property
    def confidence(self) -> float:
        """Confidence in the routing decision itself.

        Defined as the top score tempered by the margin over the first rejected
        domain. A high top score with a near-tied runner-up is *not* a confident
        route, and treating it as one is how multi-domain questions get answered
        by a single agent.
        """
        if not self.selected:
            return 0.0
        top = self.selected[0].score
        rejected = [s.score for s in self.scores if s.domain not in self.domains]
        if not rejected:
            return float(min(1.0, top))
        margin = max(0.0, top - max(rejected))
        return float(min(1.0, 0.65 * top + 0.35 * min(1.0, margin * 2.5)))

    def describe(self) -> str:
        """Full multi-line rendering of the classification, for plans and demos."""
        lines = [
            f'query      : "{truncate(self.query, 100)}"',
            f"intent     : {self.intent.value}",
            f"complexity : {self.complexity:.2f}",
            f"routing    : {', '.join(self.domains) or '(none -- out of domain)'} "
            f"(confidence {self.confidence:.2f})",
            "scores     :",
        ]
        lines.extend("    " + s.describe() for s in self.scores)
        if len(self.clauses) > 1:
            lines.append("clauses    :")
            lines.extend(f"    {i + 1}. {c}" for i, c in enumerate(self.clauses))
        return "\n".join(lines)


@dataclass(frozen=True)
class SubQuery:
    """One domain-scoped question derived from the original query.

    ``params`` travels with the sub-query rather than being looked up by the
    agent, so the plan is a complete, inspectable record of what was asked of
    whom and with what search budget.
    """

    domain: str
    text: str
    rationale: str
    params: RetrievalParams
    intent: str = ""
    source_clause: str = ""
    routing_score: float = 0.0

    def describe(self) -> str:
        """Render this sub-query with its rationale and search budget."""
        return (f"[{self.domain}] {truncate(self.text, 86)}\n"
                f"      why: {self.rationale}\n"
                f"      retrieval: {self.params.describe()}")


# --------------------------------------------------------------------------- #
# Dynamic retrieval policy
# --------------------------------------------------------------------------- #

class RetrievalPolicy:
    """Maps a classification onto per-sub-query retrieval parameters.

    This is the "dynamic retrieval strategy" requirement, isolated into one
    readable function so the behaviour can be audited and tuned rather than
    being scattered across agents as magic numbers.

    The rules, and why each exists:

    * **Depth follows complexity.** A compound three-part question needs more
      passages than a single lookup, and over-retrieving on simple questions
      just adds noise for synthesis to filter.
    * **Lexical weight follows intent.** Requirement and approval questions turn
      on exact tokens -- "two reviewers", "30 days", "CAB" -- where BM25 beats
      embeddings. Diagnostic questions are paraphrase-heavy, so dense wins.
    * **Diversity follows breadth.** Multi-domain questions lower ``mmr_lambda``
      to force cross-document coverage, because conflict detection needs two
      independent sources before it can notice a disagreement at all.
    * **The abstention floor follows routing confidence.** When routing is
      confident the floor drops, because a weak-but-real passage is worth
      surfacing. When routing is marginal the floor rises, so an uncertain route
      does not get to invent an answer out of noise.
    """

    def __init__(self, base: Optional[RetrievalParams] = None) -> None:
        self.base = base or RetrievalParams()
        #: Per-domain learned nudge on the dense/lexical balance, moved by feedback.
        self.domain_alpha: Dict[str, float] = {d: 1.0 for d in DOMAINS}

    def for_sub_query(
        self,
        domain: str,
        classification: Classification,
        *,
        is_primary: bool = True,
    ) -> RetrievalParams:
        """Compute retrieval parameters for one domain of one query."""
        complexity = classification.complexity
        confidence = classification.confidence
        intent = classification.intent

        top_k = self.base.top_k
        if complexity > 0.45:
            top_k += 1
        if complexity > 0.70:
            top_k += 1
        if not is_primary:
            # A supporting domain contributes constraints, not the main
            # narrative, so it gets a smaller budget.
            top_k = max(2, top_k - 1)
        top_k = max(2, min(6, top_k))

        pool = int(max(10, round(top_k * (3.0 + 2.0 * complexity))))

        dense_weight, lexical_weight = 1.0, 1.0
        if intent in (QueryIntent.REQUIREMENT, QueryIntent.APPROVAL):
            lexical_weight = 1.35   # exact figures and named bodies matter most
        elif intent is QueryIntent.DIAGNOSTIC:
            dense_weight = 1.30     # symptoms are described, not named
        elif intent is QueryIntent.DEFINITIONAL:
            dense_weight = 1.15
        lexical_weight *= self.domain_alpha.get(domain, 1.0)

        mmr_lambda = 0.85 if not classification.is_multi_domain else 0.68
        mmr_lambda -= 0.10 * complexity
        mmr_lambda = float(max(0.50, min(0.90, mmr_lambda)))

        # Confident routing earns a lower evidence floor; marginal routing pays
        # a higher one.
        min_score = self.base.min_score * (1.25 - 0.45 * confidence)

        return RetrievalParams(
            top_k=top_k,
            candidate_pool=pool,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            mmr_lambda=mmr_lambda,
            min_score=float(round(min_score, 4)),
            rrf_k=self.base.rrf_k,
            expand_query=classification.is_multi_domain,
        )

    def nudge_domain_alpha(self, domain: str, delta: float) -> float:
        """Adjust a domain's lexical emphasis, clamped to a sane band.

        Called by the learning mechanism. Clamping matters: an unbounded weight
        update driven by a handful of feedback signals will happily collapse the
        retriever into lexical-only or dense-only behaviour.
        """
        current = self.domain_alpha.get(domain, 1.0)
        updated = float(max(0.6, min(1.6, current + delta)))
        self.domain_alpha[domain] = updated
        return updated


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

class QueryClassifier:
    """Routes queries to domains, reads their intent, and decomposes them.

    Fit against the indexed corpus so that routing reflects what the knowledge
    base actually contains rather than an assumption about what it should
    contain.

    Args:
        selection_ratio: A domain is selected if its score is at least this
            fraction of the top domain's score. Lower values route more
            broadly; this is the main recall/precision dial for multi-label
            routing.
        min_domain_score: Absolute floor. If no domain clears it the query is
            out of domain and the system abstains. Raised relative to the
            raw cue scale because noisy-OR fusion lifts the whole score range.
        max_domains: Hard cap on fan-out, so a vague query cannot mobilise every
            agent for nothing.
    """

    def __init__(
        self,
        *,
        selection_ratio: float = 0.52,
        min_domain_score: float = 0.30,
        max_domains: int = 3,
    ) -> None:
        self.selection_ratio = selection_ratio
        self.min_domain_score = min_domain_score
        self.max_domains = max_domains
        self.policy = RetrievalPolicy()
        #: Learned multiplier per domain, moved by ``apply_feedback``.
        self.routing_weights: Dict[str, float] = {d: 1.0 for d in DOMAINS}
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._centroids: Dict[str, np.ndarray] = {}
        self._fitted = False

    # -- fitting ---------------------------------------------------------- #

    def fit(self, knowledge_base: KnowledgeBase) -> "QueryClassifier":
        """Learn a TF-IDF centroid per domain from the indexed passages."""
        per_domain = {d: knowledge_base.domain_texts(d) for d in DOMAINS}
        corpus = [t for texts in per_domain.values() for t in texts]
        if not corpus:
            raise ValueError("cannot fit QueryClassifier on an empty knowledge base")

        tokenised = [tokenize(t) for t in corpus]
        doc_freq: Counter = Counter()
        for tokens in tokenised:
            doc_freq.update(set(tokens))
        terms = sorted(doc_freq)
        self._vocab = {t: i for i, t in enumerate(terms)}
        n_docs = len(tokenised)
        df = np.array([doc_freq[t] for t in terms], dtype=np.float32)
        self._idf = (np.log((n_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

        # A centroid is the mean of its domain's L2-normalised passage vectors.
        # Normalising before averaging stops long passages dominating the
        # centroid purely by virtue of having more terms.
        for domain, texts in per_domain.items():
            if not texts:
                self._centroids[domain] = np.zeros(len(terms), dtype=np.float32)
                continue
            matrix = self._vectorize(texts)
            centroid = matrix.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            self._centroids[domain] = (centroid / norm) if norm else centroid

        self._fitted = True
        LOGGER.debug("fitted QueryClassifier on %d passages, %d terms", n_docs, len(terms))
        return self

    def _vectorize(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(t for t in tokenize(text) if t in self._vocab)
            for term, count in counts.items():
                matrix[row, self._vocab[term]] = 1.0 + math.log(count)
        matrix *= self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (matrix / norms).astype(np.float32)

    # -- classification --------------------------------------------------- #

    def classify(self, query: str) -> Classification:
        """Analyse a query: domains, intent, complexity, clause structure.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If the query is empty or whitespace only.
        """
        if not self._fitted:
            raise RuntimeError("QueryClassifier.classify called before fit")
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        clauses = self.split_clauses(query)
        intent = self.detect_intent(query)
        scores = self._score_domains(query, clauses)
        selected = self._select(scores)
        complexity = self.estimate_complexity(query, clauses, len(selected))

        return Classification(
            query=query.strip(),
            selected=tuple(selected),
            scores=tuple(scores),
            intent=intent,
            complexity=complexity,
            clauses=tuple(clauses),
            out_of_domain=not selected,
        )

    def _score_domains(self, query: str, clauses: Sequence[str]) -> List[DomainScore]:
        query_vec = self._vectorize([query])[0]
        raw: Dict[str, Tuple[float, float, Tuple[str, ...]]] = {}

        for domain in DOMAINS:
            centroid = self._centroids.get(domain)
            centroid_sim = 0.0
            if centroid is not None and centroid.size:
                centroid_sim = float(max(0.0, cosine_similarity(centroid[None, :], query_vec)[0]))
            cue_score, matched = self._cue_score(query, clauses, domain)
            raw[domain] = (centroid_sim, cue_score, matched)

        # Centroid similarities sit in a narrow absolute band because every domain
        # shares the same corporate vocabulary, so they are rescaled against the
        # strongest domain for this query. Ratio rescaling is used rather than
        # min-max: min-max forces the weakest domain to exactly zero and the
        # strongest to exactly one, discarding the magnitude difference that
        # distinguishes "clearly this domain" from "marginally this domain".
        centroid_values = [raw[d][0] for d in DOMAINS]
        ceiling = max(centroid_values)
        ratios = [(v / ceiling) if ceiling > 0 else 0.0 for v in centroid_values]

        results: List[DomainScore] = []
        for (domain, (centroid_sim, cue_score, matched)), ratio in zip(raw.items(), ratios):
            weight = self.routing_weights.get(domain, 1.0)
            blended = self._combine_evidence(cue_score, ratio) * weight
            results.append(
                DomainScore(
                    domain=domain,
                    score=float(round(min(1.0, blended), 4)),
                    centroid_similarity=float(round(centroid_sim, 4)),
                    cue_score=float(round(cue_score, 4)),
                    matched_cues=matched,
                    routing_weight=weight,
                )
            )
        return sorted(results, key=lambda s: -s.score)

    @staticmethod
    def _combine_evidence(cue_score: float, centroid_ratio: float,
                          centroid_reliability: float = 0.55) -> float:
        """Fuse the lexicon prior and the corpus centroid as independent evidence.

        A noisy-OR rather than a weighted sum. The two signals have genuinely
        different error profiles -- the lexicon is high-precision and blind to
        paraphrase, the centroid is high-recall and easily pulled around by
        shared corporate vocabulary -- so neither should be able to veto the
        other, which is exactly what a weighted sum lets the stronger-weighted
        term do.

        The concrete failure this fixes: for "troubleshoot API performance while
        following our security policies", the compliance centroid is the highest
        of the three, and under a weighted sum that outranked technical despite
        technical matching three strong cues to compliance's two. Noisy-OR keeps
        the cue evidence decisive while still letting the centroid pull in a
        domain that matched no cues at all.

        ``centroid_reliability`` discounts the centroid's contribution, encoding
        that it is the less trustworthy of the two signals.
        """
        cue = max(0.0, min(1.0, cue_score))
        centroid = max(0.0, min(1.0, centroid_ratio)) * centroid_reliability
        return float(1.0 - (1.0 - cue) * (1.0 - centroid))

    def _cue_score(self, query: str, clauses: Sequence[str],
                   domain: str) -> Tuple[float, Tuple[str, ...]]:
        """Lexicon prior, saturating so one domain cannot win on term count alone.

        Scored per clause and then averaged over the *best-matching* clauses
        rather than over the whole query, so that a two-clause question where
        only the second clause is about compliance still scores compliance
        strongly -- averaging over the full query would dilute it below the
        selection floor.
        """
        cues = DOMAIN_CUES[domain]
        matched: List[str] = []

        def clause_score(text: str) -> float:
            normalized = normalize_text(text)
            tokens = set(tokenize(text, stem=False))
            total = 0.0
            for cue, weight in cues.items():
                hit = (cue in tokens) if " " not in cue else (cue in normalized)
                if hit:
                    total += weight
                    if cue not in matched:
                        matched.append(cue)
            # Saturating transform: the first few cues carry most of the signal.
            return 1.0 - math.exp(-total / 3.2)

        per_clause = [clause_score(c) for c in clauses] or [clause_score(query)]
        whole = clause_score(query)
        # Take the strongest clause, softened by the whole-query reading, so a
        # single on-topic clause is decisive but a passing mention is not.
        best = max(per_clause)
        score = 0.7 * best + 0.3 * whole
        return float(min(1.0, score)), tuple(matched)

    def _select(self, scores: Sequence[DomainScore]) -> List[DomainScore]:
        if not scores:
            return []
        top = scores[0].score
        if top < self.min_domain_score:
            return []
        threshold = max(self.min_domain_score, top * self.selection_ratio)
        return [s for s in scores if s.score >= threshold][: self.max_domains]

    # -- intent and complexity -------------------------------------------- #

    @staticmethod
    def detect_intent(query: str) -> QueryIntent:
        """First matching cue group wins; falls back to REQUIREMENT."""
        text = normalize_text(query)
        for intent, cues in INTENT_CUES:
            if any(cue in text for cue in cues):
                return intent
        return QueryIntent.REQUIREMENT

    @staticmethod
    def split_clauses(query: str) -> List[str]:
        """Segment a compound question into clauses.

        Fragments shorter than four tokens are folded back into the previous
        clause, because "and what" style splits otherwise leave stubs that
        carry no retrievable signal.
        """
        parts = [p.strip(" ,;?") for p in _CLAUSE_RE.split(query) if p and p.strip(" ,;?")]
        if not parts:
            return [query.strip()]

        merged: List[str] = []
        for part in parts:
            if len(part.split()) < 4 and merged:
                merged[-1] = f"{merged[-1]} {part}".strip()
            else:
                merged.append(part)
        return merged or [query.strip()]

    @staticmethod
    def estimate_complexity(query: str, clauses: Sequence[str], n_domains: int) -> float:
        """Blend structural signals into a ``[0, 1]`` complexity estimate.

        Deliberately shallow and explainable: clause count, question-word count,
        length and domain breadth. A learned regressor would need labelled
        difficulty data that does not exist here, and would be harder to justify
        when it got a case wrong.
        """
        tokens = tokenize(query, stem=False)
        n_clauses = len(clauses)
        n_wh = len(_WH_RE.findall(query))
        length = len(tokens)

        clause_term = min(1.0, (n_clauses - 1) / 2.0)
        wh_term = min(1.0, max(0, n_wh - 1) / 2.0)
        length_term = min(1.0, max(0, length - 6) / 18.0)
        domain_term = min(1.0, max(0, n_domains - 1) / 2.0)

        score = 0.30 * clause_term + 0.20 * wh_term + 0.20 * length_term + 0.30 * domain_term
        return float(round(min(1.0, score), 3))

    # -- decomposition ---------------------------------------------------- #

    def decompose(self, classification: Classification) -> List[SubQuery]:
        """Turn a classification into per-domain sub-queries.

        Each selected domain is paired with the clause that drew it in, which is
        what makes the sub-query narrower than the original question. When a
        domain was selected on whole-query evidence rather than any single
        clause, it receives the full query -- a less precise search, but better
        than inventing a clause it did not match.

        Returns:
            One sub-query per selected domain, highest-scoring domain first.
        """
        if classification.out_of_domain:
            return []

        clauses = list(classification.clauses)
        sub_queries: List[SubQuery] = []

        for rank, domain_score in enumerate(classification.selected):
            domain = domain_score.domain
            clause, clause_score = self._best_clause(clauses, domain)
            is_primary = rank == 0

            if len(clauses) > 1 and clause_score > 0.25 and clause != classification.query:
                text = self.anchor_clause(clause, classification.query)
                anchored = text != clause
                rationale = (
                    f"clause {clauses.index(clause) + 1} of {len(clauses)} matched "
                    f"{domain} cues ({', '.join(domain_score.matched_cues[:3]) or 'centroid only'})"
                    + ("; anchored to the query subject because the clause alone "
                       "was not self-sufficient" if anchored else "")
                )
            else:
                text = classification.query
                rationale = (
                    f"no single clause isolated {domain}; routed on whole-query evidence "
                    f"(score {domain_score.score:.2f})"
                )

            params = self.policy.for_sub_query(domain, classification, is_primary=is_primary)
            sub_queries.append(
                SubQuery(
                    domain=domain,
                    text=text,
                    rationale=rationale,
                    params=params,
                    intent=classification.intent.value,
                    source_clause=clause,
                    routing_score=domain_score.score,
                )
            )
        return sub_queries

    @staticmethod
    def anchor_clause(clause: str, query: str, *, min_self_sufficient: int = 6,
                      max_anchor_terms: int = 4) -> str:
        """Carry the query's subject into a clause that lost it when split.

        Clause splitting is syntactic, so it happily produces clauses that are
        not semantically self-sufficient. "What's the process for deploying a new
        microservice and what compliance checks are needed?" splits into a first
        clause that names the subject and a second -- "compliance checks are
        needed" -- that names nothing retrievable at all.

        Searching the compliance corpus for that stub measurably failed: it
        returned generic audit-evidence and control-monitoring passages and
        missed the production change-control policy that actually answers the
        question. Anchoring it back to "deploying new microservice" fixes that,
        and costs nothing on clauses that were already complete.

        Args:
            clause: The clause to anchor.
            query: The full original query to draw subject terms from.
            min_self_sufficient: Content-token count above which a clause is left
                alone.
            max_anchor_terms: Cap on terms carried across.
        """
        clause_tokens = tokenize(clause, stem=False)
        if len(clause_tokens) >= min_self_sufficient:
            return clause

        present = set(tokenize(clause))
        anchors: List[str] = []
        for token in tokenize(query, stem=False):
            if token in META_TERMS or _stem_token(token) in present:
                continue
            if token in anchors:
                continue
            anchors.append(token)
            if len(anchors) >= max_anchor_terms:
                break
        return f"{clause} {' '.join(anchors)}".strip() if anchors else clause

    def _best_clause(self, clauses: Sequence[str], domain: str) -> Tuple[str, float]:
        """Pick the clause with the strongest evidence for ``domain``."""
        if not clauses:
            return "", 0.0
        scored = [(c, self._cue_score(c, [c], domain)[0]) for c in clauses]
        return max(scored, key=lambda pair: pair[1])

    # -- learning --------------------------------------------------------- #

    def apply_feedback(
        self,
        classification: Classification,
        expected_domains: Sequence[str],
        *,
        learning_rate: float = 0.08,
    ) -> Dict[str, float]:
        """Nudge routing weights towards the domains that should have been used.

        A deliberately simple multiplicative update -- reward domains that were
        expected, penalise domains that fired without being expected, leave the
        rest alone. It is the simplest thing that demonstrably moves routing in
        the right direction, and being transparent matters more here than being
        optimal: an opaque update on a handful of feedback signals would be
        impossible to debug.

        Weights are clamped to ``[0.6, 1.5]`` so no amount of feedback can
        switch a domain off entirely or let one dominate the others.

        Args:
            classification: The routing decision being corrected.
            expected_domains: Ground-truth domains for the query.
            learning_rate: Step size per update.

        Returns:
            The updated routing weights.
        """
        expected = {d for d in expected_domains if d in self.routing_weights}
        predicted = set(classification.domains)

        for domain in self.routing_weights:
            if domain in expected and domain not in predicted:
                delta = +learning_rate            # missed: should have routed here
            elif domain in predicted and domain not in expected:
                delta = -learning_rate            # false positive: over-routed
            elif domain in expected and domain in predicted:
                delta = +learning_rate * 0.25     # correct: reinforce gently
            else:
                continue
            self.routing_weights[domain] = float(
                max(0.6, min(1.5, self.routing_weights[domain] + delta))
            )
        return dict(self.routing_weights)

    # -- persistence ------------------------------------------------------ #

    def export_weights(self) -> Dict[str, Any]:
        """Serialise the learned state so it can be inspected or checkpointed."""
        return {
            "routing_weights": dict(self.routing_weights),
            "domain_alpha": dict(self.policy.domain_alpha),
            "selection_ratio": self.selection_ratio,
            "min_domain_score": self.min_domain_score,
        }

    def load_weights(self, payload: Dict[str, Any]) -> None:
        """Restore state previously produced by ``export_weights``."""
        for domain, value in (payload.get("routing_weights") or {}).items():
            if domain in self.routing_weights:
                self.routing_weights[domain] = float(value)
        for domain, value in (payload.get("domain_alpha") or {}).items():
            if domain in self.policy.domain_alpha:
                self.policy.domain_alpha[domain] = float(value)

    def save_weights(self, path: Path) -> Path:
        """Write learned weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export_weights(), indent=2), encoding="utf-8")
        return path
