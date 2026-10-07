"""Claim extraction, conflict resolution, citation tracking and answer synthesis.

This module turns retrieved passages into an answer, and it is where the
assignment's three "advanced" requirements actually live:

* **Conflict resolution.** Three generic detectors (numeric disagreement,
  polarity opposition, declared supersession) find contradictory claims, and a
  weighted confidence model adjudicates them. Crucially the loser is reported,
  not deleted -- an answer that silently drops the runbook an engineer has been
  following is worse than useless, because it leaves them unaware they are
  non-compliant.

* **Citation tracking.** Every claim carries the character span it came from, so
  a citation can be *verified* against the source document rather than merely
  rendered. ``verify_citations`` re-reads each span and reports mismatches.

* **Synthesis.** The default synthesiser is extractive and deterministic, so the
  demo is reproducible and needs no API key. ``ClaudeSynthesizer`` implements
  the same interface for when a real model is wanted, and falls back to the
  extractive path on any API failure rather than losing the answer.
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import (
    DOMAIN_PRECEDENCE,
    Document,
    Measurement,
    RetrievedChunk,
    extract_measurements,
    get_logger,
    jaccard,
    polarity,
    recency_weight,
    split_sentences,
    tokenize,
    truncate,
)

LOGGER = get_logger(__name__)


def version_key(version: str) -> Tuple[int, ...]:
    """Sort key ordering dotted version strings numerically.

    Non-numeric components sort as ``0``, which is enough for the ``major.minor``
    strings this corpus uses and degrades predictably on anything else.

    >>> sorted(["10.0", "2.0", "1.5"], key=version_key)
    ['1.5', '2.0', '10.0']
    """
    parts = []
    for component in str(version).split("."):
        parts.append(int(component) if component.isdigit() else 0)
    return tuple(parts)


#: Anchors too generic to establish that two quantities measure the same thing.
#:
#: These are the predicates that sit next to a number in almost every policy
#: sentence ("retained for 90 days", "requires two reviewers"). Sharing one of
#: them says the two claims are both rules with numbers in them, not that they
#: are rules about the same subject. When the only shared anchor is generic, the
#: numeric detector demands much stronger sentence-level overlap before it will
#: call a contradiction -- this is what stops "application logs are retained for
#: 90 days" being reported as contradicting "customer transaction records are
#: retained for seven years".
GENERIC_ANCHORS: frozenset = frozenset({
    "retain", "requir", "sampl", "keep", "set", "review", "complet", "approv",
    "exceed", "appli", "use", "includ", "provid", "must", "need", "allow",
    "expir", "raise", "report", "record", "document", "perform", "run", "take",
    "within", "least", "than", "after", "befor", "everi", "each",
    "default", "standard", "normal", "typic", "gener", "other",
})

#: Words marking a sentence as describing a *procedure* rather than a rule.
#:
#: These exist because rewarding only obligation language biases claim selection
#: toward policy prose and against instructions, which is exactly backwards for a
#: "how do I" question. Asked how to troubleshoot API latency, the definitive
#: sentence in the corpus -- "Begin API latency triage by confirming the symptom in
#: the golden-signal dashboard" -- scored lowest of all candidates, because it
#: contains no "must" and no quantity. Which marker class earns the bonus is
#: therefore chosen by query intent (see ``ClaimExtractor._salience``).
PROCEDURAL_MARKERS: frozenset = frozenset({
    "begin", "start", "first", "next", "then", "check", "verify", "confirm",
    "inspect", "identify", "decompose", "review", "run", "use", "open", "collect",
    "measure", "compare", "look", "trace", "triage", "diagnose", "reproduce",
    "enable", "disable", "restart", "roll", "rollback", "escalate", "follow",
    "ensure", "set", "configure", "apply", "raise", "lower", "increase",
})

#: Words that mark a sentence as stating a rule rather than describing context.
#: A claim-bearing sentence is what conflict detection and citation need; a
#: scene-setting sentence is noise in both.
OBLIGATION_MARKERS: frozenset = frozenset({
    "must", "required", "require", "requires", "shall", "mandatory", "prohibited",
    "forbidden", "may", "cannot", "need", "needs", "should", "obliged", "expected",
    "permitted", "allowed", "blocked", "rejected", "enforced", "triggers",
    "approved", "approval", "withdrawn", "banned",
})


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Claim:
    """A single assertion extracted from a retrieved passage.

    Claims, not passages, are the unit of synthesis. A passage can contain one
    useful rule and three sentences of background; quoting the whole passage
    dilutes the answer and makes contradiction detection operate on the wrong
    granularity.

    Attributes:
        char_start/char_end: Span within the *source document* text, which is
            what makes the citation verifiable.
        confidence: Blended trust score. See ``ClaimExtractor._confidence``.
        components: The individual factors behind ``confidence``, retained so a
            surprising ranking can be explained rather than argued about.
        measurements: Normalised quantities, used by numeric conflict detection.
        polarity: ``+1`` for an affirmative rule, ``-1`` for a prohibition.
    """

    claim_id: str
    text: str
    doc_id: str
    domain: str
    chunk_id: str
    char_start: int
    char_end: int
    confidence: float
    salience: float
    retrieval_relevance: float
    polarity: int
    document: Document
    measurements: Tuple[Measurement, ...] = ()
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def tokens(self) -> List[str]:
        return tokenize(self.text)

    @property
    def topicality(self) -> float:
        """How much this claim is *about the question*, in ``[0, 1]``.

        Deliberately distinct from ``confidence``, which answers a different
        question. Topicality blends passage relevance with sentence-level
        salience only; it ignores authority and recency entirely.

        The separation exists because conflating the two produced a real defect.
        Asked "what approvals are needed to deploy a microservice", the technical
        agent displayed "Retries use exponential backoff and are capped at three
        attempts" -- which scored higher than "a production deployment requires
        one approving reviewer" purely because the retry guidance lives in a
        recent *standard* while the approval rule lives in an older *runbook*.
        Authority and recency say nothing about whether a sentence answers the
        question asked, so they are excluded here and kept where they belong, in
        conflict adjudication and answer confidence.
        """
        return float(round(0.60 * self.retrieval_relevance + 0.40 * self.salience, 4))

    def describe(self) -> str:
        """One-line summary of this claim and its confidence."""
        return (f"[{self.domain}/{self.document.citation_label}] "
                f"conf={self.confidence:.3f} {truncate(self.text, 96)}")


class ClaimExtractor:
    """Pulls rule-bearing sentences out of retrieved passages and scores them.

    Scoring blends four independent factors rather than trusting retrieval
    alone, because retrieval relevance answers "did this match the query" and
    says nothing about "should this be believed". A stale wiki page can be the
    best lexical match in the corpus.

    Args:
        min_salience: Drop sentences below this salience. The floor exists to
            keep background prose out of the answer.
        max_claims_per_passage: Cap per passage, so one verbose document cannot
            crowd out a terser one that is equally relevant.
    """

    #: Confidence weights. Passage relevance and sentence-level salience together
    #: dominate, because they are the two signals that answer "is this about the
    #: question"; authority and recency answer "should this be believed" and
    #: together still outweigh either one alone, which is what lets a current
    #: policy beat a well-matching outdated runbook.
    #:
    #: Salience is weighted this highly after observing the alternative: at a
    #: lower weight, a recent high-authority *standard* on an adjacent topic
    #: (secrets management, on a question about deployment) outscored the
    #: deployment runbook's own rollout procedure, because authority and recency
    #: do not care what the question was.
    WEIGHTS: ClassVar[Dict[str, float]] = {
        "relevance": 0.36,
        "salience": 0.26,
        "authority": 0.22,
        "recency": 0.16,
    }

    def __init__(self, *, min_salience: float = 0.18,
                 max_claims_per_passage: int = 3) -> None:
        self.min_salience = min_salience
        self.max_claims_per_passage = max_claims_per_passage

    def extract(
        self,
        retrieved: RetrievedChunk,
        query: str,
        *,
        intent: str = "",
        now: Optional[date] = None,
    ) -> List[Claim]:
        """Extract scored claims from one retrieved passage.

        Character offsets are resolved against the parent document and verified;
        a sentence whose span cannot be located is still returned, but with its
        offsets collapsed to the chunk boundary so that citation verification
        reports it honestly rather than pointing at the wrong text.
        """
        document = retrieved.document
        query_tokens = set(tokenize(query))
        # The document's own title and section are strong topical evidence and are
        # free to use: a sentence from "API Performance Troubleshooting Guide"
        # deserves topical credit on an API-performance question even when that
        # sentence happens to share few words with the query itself.
        title_tokens = set(tokenize(f"{document.title} {document.section}"))
        claims: List[Claim] = []

        for sentence in split_sentences(retrieved.chunk.text):
            salience = self._salience(sentence, query_tokens, title_tokens, intent)
            if salience < self.min_salience:
                continue

            start = document.text.find(sentence)
            if start == -1:
                start, end = retrieved.chunk.char_start, retrieved.chunk.char_end
            else:
                end = start + len(sentence)

            components = {
                "relevance": float(max(0.0, min(1.0, retrieved.relevance))),
                "authority": document.authority_weight,
                "recency": recency_weight(document.effective, now=now),
                "salience": salience,
            }
            confidence = sum(self.WEIGHTS[k] * v for k, v in components.items())

            claims.append(
                Claim(
                    claim_id=uuid.uuid4().hex[:10],
                    text=sentence.strip(),
                    doc_id=document.doc_id,
                    domain=document.domain,
                    chunk_id=retrieved.chunk.chunk_id,
                    char_start=start,
                    char_end=end,
                    confidence=float(round(confidence, 4)),
                    salience=float(round(salience, 4)),
                    retrieval_relevance=float(round(retrieved.relevance, 4)),
                    polarity=polarity(sentence),
                    document=document,
                    measurements=tuple(extract_measurements(sentence)),
                    components={k: float(round(v, 4)) for k, v in components.items()},
                )
            )

        claims.sort(key=lambda c: -c.confidence)
        return claims[: self.max_claims_per_passage]

    #: Intents whose answers are procedures, so procedural language is the signal.
    PROCEDURAL_INTENTS: frozenset = frozenset({"diagnostic", "procedural"})

    def _salience(
        self,
        sentence: str,
        query_tokens: Iterable[str],
        title_tokens: Iterable[str] = (),
        intent: str = "",
    ) -> float:
        """How much this sentence looks like an answer to *this* query.

        Four signals:

        * **Lexical overlap** with the query, blended with overlap between the
          query and the parent document's title. The title term matters because a
          query shares few words with any single sentence, so sentence overlap
          alone barely separates candidates -- while "API Performance
          Troubleshooting Guide" is decisive evidence on an API-performance
          question.
        * **Marker class matched to intent.** A requirement question is answered
          by obligations ("must", "prohibited"); a diagnostic or procedural
          question is answered by instructions ("begin", "check", "compare").
          Rewarding only obligations made the system rank a data-store rule above
          the actual latency triage procedure when asked how to troubleshoot
          latency. The matching class earns the full bonus and the other a
          reduced one, since a sentence can legitimately be both.
        * **Presence of a concrete quantity**, which in policy prose usually
          marks the operative sentence.

        Args:
            sentence: The candidate sentence.
            query_tokens: Stemmed content tokens of the query.
            title_tokens: Stemmed tokens of the document title and section.
            intent: Query intent value, selecting which marker class is rewarded.
        """
        tokens = set(tokenize(sentence))
        if not tokens:
            return 0.0
        query_tokens = set(query_tokens)
        if not query_tokens:
            return 0.0

        sentence_overlap = len(tokens & query_tokens) / len(query_tokens)
        title_overlap = (len(set(title_tokens) & query_tokens) / len(query_tokens)
                         if title_tokens else 0.0)
        overlap = 0.72 * sentence_overlap + 0.28 * title_overlap

        raw = set(tokenize(sentence, stem=False))
        has_obligation = bool(raw & OBLIGATION_MARKERS)
        has_procedure = bool(raw & PROCEDURAL_MARKERS)
        has_quantity = bool(extract_measurements(sentence))

        wants_procedure = intent in self.PROCEDURAL_INTENTS
        primary = has_procedure if wants_procedure else has_obligation
        secondary = has_obligation if wants_procedure else has_procedure

        score = 0.55 * min(1.0, overlap * 2.2)
        score += 0.26 if primary else 0.0
        score += 0.10 if secondary else 0.0
        score += 0.15 if has_quantity else 0.0
        return float(round(min(1.0, score), 4))


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Conflict:
    """Two claims that cannot both be acted on, plus the adjudication.

    Attributes:
        kind: Which detector fired -- ``numeric``, ``polarity`` or ``supersession``.
        subject: Short human label for what is in dispute.
        winner/loser: The adopted and overruled claims.
        margin: Score gap between them. A small margin is a genuine
            near-tie and is reported as low confidence rather than hidden.
        basis: Component scores behind the decision, for explainability.
    """

    conflict_id: str
    kind: str
    subject: str
    winner: Claim
    loser: Claim
    margin: float
    basis: Dict[str, Dict[str, float]]
    explanation: str

    @property
    def decisive(self) -> bool:
        """Whether the resolution is clear-cut rather than a coin toss.

        A supersession is always decisive regardless of margin: the documents
        themselves declare which one replaced the other, so the heuristic score
        gap is beside the point. Reporting a declared version bump as a near tie
        would understate a certainty the corpus hands us outright.
        """
        if self.kind == "supersession":
            return True
        # Spelled out rather than chained as ``a == "superseded" != b``: that
        # reads like a typo even though Python evaluates it correctly.
        if (self.loser.document.status == "superseded"
                and self.winner.document.status != "superseded"):
            return True
        return self.margin >= 0.08

    def describe(self) -> str:
        """Multi-line rendering of the dispute and how it was resolved."""
        return (
            f"[{self.kind}] {self.subject}\n"
            f"    adopted  : {self.winner.document.citation_label} "
            f"({self.winner.domain}, {self.winner.document.authority}, "
            f"{self.winner.document.effective_date})\n"
            f"               \"{truncate(self.winner.text, 110)}\"\n"
            f"    overruled: {self.loser.document.citation_label} "
            f"({self.loser.domain}, {self.loser.document.authority}, "
            f"{self.loser.document.effective_date})\n"
            f"               \"{truncate(self.loser.text, 110)}\"\n"
            f"    basis    : {self.explanation} (margin {self.margin:.3f}"
            f"{'' if self.decisive else ', NEAR TIE'})"
        )


class ConflictResolver:
    """Finds contradictory claims and decides which one to act on.

    The detectors are deliberately *generic* -- none of them knows anything
    about this corpus. Hard-coding "the retention conflict" would demonstrate
    nothing; these rules fire on any pair of claims exhibiting the structural
    pattern, which is what makes the mechanism worth having:

    1. **Numeric** -- two claims quantify the same thing (same unit family, at
       least one shared subject anchor) with different values.
    2. **Polarity** -- two claims with high lexical overlap assert opposite
       permission ("enable verbose tracing" / "verbose tracing must not be
       enabled").
    3. **Supersession** -- one claim's document explicitly supersedes the
       other's, or is a later version of the same titled procedure. This is a
       *declared* conflict and is resolved without scoring.

    Resolution weights authority, domain precedence, recency and retrieval
    relevance. Retrieval relevance is deliberately the smallest term: being the
    best lexical match for the question is close to irrelevant to being right.

    Every detector is additionally gated on the two claims being *topically*
    related, which is what separates a contradiction from a coincidence. Without
    that gate the numeric detector reports "application logs are retained for 90
    days" as contradicting "customer transaction records are retained for seven
    years" -- both quantify a duration and both share the token "retained", but
    they govern different data categories and do not disagree about anything.

    Candidate conflicts are then deduplicated per document pair and dispute, so
    two documents that disagree once are reported once. Reporting it per claim
    pair instead turns a single real disagreement into a wall of near-identical
    findings that buries the signal.

    Args:
        overlap_threshold: Minimum token Jaccard for the polarity detector,
            which is also gated on the two claims having opposite polarity.
        numeric_tolerance: Relative difference below which two quantities are
            treated as agreeing, absorbing rounding and phrasing noise.
        topical_floor: Minimum sentence-level Jaccard for a numeric conflict, on
            top of the shared-anchor requirement.
        generic_anchor_floor: The stricter overlap required when the only shared
            anchor is a generic measurement predicate (see ``GENERIC_ANCHORS``).
        supersession_floor: Minimum sentence-level Jaccard before a declared
            version relationship counts as a conflict between two claims. A
            superseding document restates most of its predecessor; only the
            claims that actually changed are in dispute.
        max_conflicts: Cap on reported conflicts, so one pathological query
            cannot produce an unreadable answer.
    """

    WEIGHTS: ClassVar[Dict[str, float]] = {
        "authority": 0.34,
        "domain_precedence": 0.26,
        "recency": 0.22,
        "relevance": 0.18,
    }

    def __init__(
        self,
        *,
        overlap_threshold: float = 0.20,
        numeric_tolerance: float = 0.05,
        topical_floor: float = 0.08,
        generic_anchor_floor: float = 0.20,
        supersession_floor: float = 0.14,
        max_conflicts: int = 6,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.numeric_tolerance = numeric_tolerance
        self.topical_floor = topical_floor
        self.generic_anchor_floor = generic_anchor_floor
        self.supersession_floor = supersession_floor
        self.max_conflicts = max_conflicts

    # -- detection -------------------------------------------------------- #

    def detect(self, claims: Sequence[Claim], *, now: Optional[date] = None) -> List[Conflict]:
        """Find and resolve every conflict among a set of claims.

        Only cross-document pairs are considered: a document contradicting
        itself is a drafting problem in the source, not something this system
        should adjudicate at query time.
        """
        # Candidates are collected first and only then collapsed, because the
        # best representative of a dispute is not necessarily the first pair
        # encountered -- it is the pair with the strongest mutual evidence.
        candidates: Dict[Tuple[Any, ...], Tuple[float, str, str, Claim, Claim]] = {}

        for i, left in enumerate(claims):
            for right in claims[i + 1:]:
                if left.doc_id == right.doc_id:
                    continue

                detected = (
                    self._numeric(left, right)
                    or self._polarity(left, right)
                    or self._supersession(left, right)
                )
                if detected is None:
                    continue
                kind, subject, dispute_key = detected

                key = (frozenset((left.doc_id, right.doc_id)), kind, dispute_key)
                strength = min(left.confidence, right.confidence)
                existing = candidates.get(key)
                if existing is None or strength > existing[0]:
                    candidates[key] = (strength, kind, subject, left, right)

        resolved = [
            self._resolve(kind, subject, left, right, now=now)
            for _, kind, subject, left, right in sorted(
                candidates.values(), key=lambda item: -item[0]
            )
        ]
        return resolved[: self.max_conflicts]

    def _supersession(self, a: Claim, b: Claim) -> Optional[Tuple[str, str, str]]:
        """Detect a declared version relationship covering the same subject.

        The version link alone is not enough. A v2 procedure restates most of v1
        verbatim; flagging every claim pair across the two versions reports
        agreement as conflict. The topical gate restricts it to the claims whose
        subject matter actually overlaps.
        """
        da, db = a.document, b.document
        related = (
            da.supersedes == db.doc_id
            or db.supersedes == da.doc_id
            or (da.status != db.status and (da.title, da.section) == (db.title, db.section))
        )
        if not related:
            return None
        if jaccard(a.tokens, b.tokens) < self.supersession_floor:
            return None
        title = da.title or db.title
        # Ordered numerically, not lexically: as strings "10.0" sorts before
        # "2.0", so a corpus that ever reaches a double-digit major version would
        # report the supersession the wrong way round in the subject line.
        older, newer = sorted((da.version, db.version), key=version_key)
        return ("supersession",
                f"{title}: {older} superseded by {newer}",
                "version")

    def _numeric(self, a: Claim, b: Claim) -> Optional[Tuple[str, str, str]]:
        """Detect two claims quantifying the *same* subject differently.

        Two gates, both required. The shared anchor establishes that the
        quantities attach to a common noun; the sentence-level overlap floor
        establishes that the surrounding claims are about the same topic. The
        anchor alone is too weak, because a generic predicate such as "retained"
        is an anchor for every retention rule in the corpus regardless of what
        is being retained.
        """
        overlap = jaccard(a.tokens, b.tokens)
        if overlap < self.topical_floor:
            return None
        for ma in a.measurements:
            for mb in b.measurements:
                if ma.family != mb.family:
                    continue
                shared = set(ma.anchors) & set(mb.anchors)
                if not shared:
                    continue
                scale = max(abs(ma.base_value), abs(mb.base_value), 1e-9)
                if abs(ma.base_value - mb.base_value) / scale <= self.numeric_tolerance:
                    continue

                # A specific shared anchor ("log", "production") is itself good
                # evidence of a common subject. If every shared anchor is a
                # generic measurement predicate, the pair has to earn it on
                # sentence-level overlap instead.
                specific = sorted(shared - GENERIC_ANCHORS)
                if not specific and overlap < self.generic_anchor_floor:
                    continue

                anchor = specific[0] if specific else sorted(shared)[0]
                return ("numeric",
                        f"{anchor} ({ma.family}): {ma.raw} vs {mb.raw}",
                        f"{ma.family}:{anchor}")
        return None

    def _polarity(self, a: Claim, b: Claim) -> Optional[Tuple[str, str, str]]:
        """Detect opposite permission on substantially the same subject."""
        if a.polarity == b.polarity:
            return None
        overlap = jaccard(a.tokens, b.tokens)
        if overlap < self.overlap_threshold:
            return None
        # Longest first, then alphabetically. The alphabetical tiebreak is load
        # bearing: `key=len` alone is not a total order over a set, so two
        # equal-length tokens came out in set-iteration order and this label
        # changed between runs with different PYTHONHASHSEED values.
        shared = sorted(set(a.tokens) & set(b.tokens),
                        key=lambda t: (-len(t), t))[:3]
        subject = ", ".join(shared) if shared else "opposing guidance"
        return ("polarity",
                f"permitted vs prohibited -- {subject} (overlap {overlap:.2f})",
                "polarity")

    # -- resolution ------------------------------------------------------- #

    def _resolve(self, kind: str, subject: str, a: Claim, b: Claim, *,
                 now: Optional[date] = None) -> Conflict:
        """Score both claims and adopt the stronger one."""
        score_a, basis_a = self._score(a, now=now)
        score_b, basis_b = self._score(b, now=now)

        # A superseded document can never win, whatever it scores. Version
        # history is an explicit statement of intent and outranks any heuristic.
        if a.document.status == "superseded" and b.document.status != "superseded":
            winner, loser, score_w, score_l = b, a, score_b, score_a
        elif (b.document.status == "superseded"
                and a.document.status != "superseded"):
            winner, loser, score_w, score_l = a, b, score_a, score_b
        elif score_a >= score_b:
            winner, loser, score_w, score_l = a, b, score_a, score_b
        else:
            winner, loser, score_w, score_l = b, a, score_b, score_a

        return Conflict(
            conflict_id=uuid.uuid4().hex[:10],
            kind=kind,
            subject=subject,
            winner=winner,
            loser=loser,
            margin=float(round(abs(score_w - score_l), 4)),
            basis={
                winner.document.citation_label: basis_a if winner is a else basis_b,
                loser.document.citation_label: basis_b if winner is a else basis_a,
            },
            explanation=self._explain(kind, winner, loser),
        )

    def _score(self, claim: Claim, *, now: Optional[date] = None) -> Tuple[float, Dict[str, float]]:
        components = {
            "authority": claim.document.authority_weight,
            "domain_precedence": DOMAIN_PRECEDENCE.get(claim.domain, 0.7),
            "recency": recency_weight(claim.document.effective, now=now),
            "relevance": claim.retrieval_relevance,
        }
        total = sum(self.WEIGHTS[k] * v for k, v in components.items())
        return float(round(total, 4)), {k: float(round(v, 4)) for k, v in components.items()}

    @staticmethod
    def _explain(kind: str, winner: Claim, loser: Claim) -> str:
        """Name the factor that actually decided it, in plain language."""
        if kind == "supersession":
            return (f"{loser.document.citation_label} is superseded by "
                    f"{winner.document.citation_label}")
        reasons: List[str] = []
        if winner.document.authority_weight > loser.document.authority_weight:
            reasons.append(f"{winner.document.authority} outranks {loser.document.authority}")
        if DOMAIN_PRECEDENCE.get(winner.domain, 0) > DOMAIN_PRECEDENCE.get(loser.domain, 0):
            reasons.append(f"{winner.domain} governs over {loser.domain}")
        if winner.document.effective > loser.document.effective:
            reasons.append(f"newer ({winner.document.effective_date} > "
                           f"{loser.document.effective_date})")
        if not reasons:
            reasons.append("higher retrieval relevance with equal standing")
        return "; ".join(reasons)


# --------------------------------------------------------------------------- #
# Citations and answers
# --------------------------------------------------------------------------- #

#: Bands for turning a confidence score into a word. One definition, because the
#: answer body and the ``Answer`` object both label the same number and silently
#: disagreeing about where "high" starts would be worse than either boundary.
CONFIDENCE_BANDS: Tuple[Tuple[float, str], ...] = ((0.70, "high"), (0.45, "moderate"))


def confidence_label(confidence: float) -> str:
    """Name a confidence score: ``high``, ``moderate`` or ``low``.

    >>> [confidence_label(c) for c in (0.9, 0.5, 0.1)]
    ['high', 'moderate', 'low']
    """
    for floor, label in CONFIDENCE_BANDS:
        if confidence >= floor:
            return label
    return "low"


@dataclass(frozen=True)
class Citation:
    """A numbered reference back to an exact span of a source document."""

    marker: str
    doc_id: str
    domain: str
    title: str
    section: str
    version: str
    authority: str
    effective_date: str
    quote: str
    char_start: int
    char_end: int

    def render(self) -> str:
        """Render as a source-list entry, including the verifiable span."""
        section = f" / {self.section}" if self.section else ""
        return (f"{self.marker} {self.doc_id} v{self.version} -- {self.title}{section} "
                f"({self.domain}, {self.authority}, {self.effective_date}, "
                f"chars {self.char_start}-{self.char_end})")


@dataclass
class Answer:
    """The orchestrator's final output.

    Carries not just the text but everything needed to audit it: the claims
    used, the citations with their source spans, the conflicts and how they were
    resolved, and whether any agent failed on the way.
    """

    query: str
    text: str
    confidence: float
    domains: Tuple[str, ...] = ()
    citations: Tuple[Citation, ...] = ()
    claims: Tuple[Claim, ...] = ()
    conflicts: Tuple[Conflict, ...] = ()
    abstained: bool = False
    degraded: bool = False
    notes: Tuple[str, ...] = ()
    trace_id: str = ""
    synthesizer: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence_label(self) -> str:
        return confidence_label(self.confidence)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return self.text


def verify_citations(answer: Answer, resolve_document: Any) -> List[Dict[str, Any]]:
    """Re-read every citation's span from its source and confirm it matches.

    This is the guard against a confidently formatted answer whose citations
    point at the wrong text -- the failure mode that makes a RAG system
    *worse* than no system, because the citation makes the error look checked.

    Args:
        answer: The answer to audit.
        resolve_document: Callable mapping a ``doc_id`` to a ``Document`` or
            ``None`` (``KnowledgeBase.document`` satisfies this).

    Returns:
        One record per citation with an ``ok`` flag and, on failure, the reason.
    """
    report: List[Dict[str, Any]] = []
    for citation in answer.citations:
        document = resolve_document(citation.doc_id)
        if document is None:
            report.append({"marker": citation.marker, "doc_id": citation.doc_id,
                           "ok": False, "reason": "source document not found"})
            continue
        span = document.text[citation.char_start:citation.char_end]
        normalized_span = " ".join(span.split())
        normalized_quote = " ".join(citation.quote.split())
        if normalized_span == normalized_quote:
            report.append({"marker": citation.marker, "doc_id": citation.doc_id,
                           "ok": True, "reason": "span matches quoted text"})
        elif normalized_quote and normalized_quote in " ".join(document.text.split()):
            report.append({"marker": citation.marker, "doc_id": citation.doc_id,
                           "ok": False,
                           "reason": "quote present in document but span offsets are wrong"})
        else:
            report.append({"marker": citation.marker, "doc_id": citation.doc_id,
                           "ok": False, "reason": "quote not found in source document"})
    return report


# --------------------------------------------------------------------------- #
# Synthesizers
# --------------------------------------------------------------------------- #

class Synthesizer(ABC):
    """Interface for turning claims and conflicts into an answer."""

    name: str = "abstract"

    @abstractmethod
    def synthesize(
        self,
        query: str,
        claims: Sequence[Claim],
        conflicts: Sequence[Conflict],
        *,
        domains: Sequence[str],
        notes: Sequence[str] = (),
        degraded: bool = False,
    ) -> Answer:
        """Produce an answer. Must populate citations and confidence."""


class ExtractiveSynthesizer(Synthesizer):
    """Deterministic, template-driven synthesis. The default.

    Extractive rather than generative, which is a real trade-off and worth being
    explicit about. It costs fluency: the output reads as structured findings
    rather than prose. It buys three things that matter more for this brief --
    every sentence in the answer is verbatim from a source document so it cannot
    hallucinate, the output is byte-identical for a given corpus and date so the
    notebook and the tests are reproducible, and it needs no API key or network.
    (The date qualifier is real: recency decay reads ``date.today()``, so claim
    confidence drifts slowly as the corpus ages.)

    ``ClaudeSynthesizer`` is the generative counterpart for when fluency is
    worth the trade.
    """

    name = "extractive"

    def __init__(
        self,
        *,
        max_claims_per_domain: int = 4,
        dedupe_threshold: float = 0.72,
        topicality_floor: float = 0.40,
        relative_floor: float = 0.55,
    ) -> None:
        self.max_claims_per_domain = max_claims_per_domain
        self.dedupe_threshold = dedupe_threshold
        self.topicality_floor = topicality_floor
        self.relative_floor = relative_floor

    def synthesize(
        self,
        query: str,
        claims: Sequence[Claim],
        conflicts: Sequence[Conflict],
        *,
        domains: Sequence[str],
        notes: Sequence[str] = (),
        degraded: bool = False,
    ) -> Answer:
        """Assemble claims and conflicts into a cited answer.

        Assembly order matters and is not arbitrary:

        1. Claims that lost a conflict are removed from the main body, because
           presenting overruled guidance as current is the worst possible
           outcome. They are *not* discarded -- they reappear in the conflict
           section, quoted and cited.
        2. Surviving claims are deduplicated. Several documents restate the same
           rule, and repeating it three times with three citations reads as
           padding rather than thoroughness.
        3. Citation markers are assigned over everything that appears anywhere in
           the output, conflict losers included, so an overruled claim is exactly
           as traceable as an adopted one.
        4. Findings are grouped by domain in routing order, so the primary
           domain's answer leads.

        Args:
            query: The original question, echoed into the answer for context.
            claims: Every claim gathered from every agent.
            conflicts: Already-detected conflicts, with winners decided.
            domains: Selected domains in routing order; controls section order.
            notes: Caveats to surface to the reader (agent failures, abstentions,
                context sharing).
            degraded: Whether an agent failed, which lowers confidence and is
                stated explicitly rather than hidden.

        Returns:
            An ``Answer``. With no claims this is an abstention with confidence
            ``0.0`` -- a legitimate outcome, not an error.
        """
        if not claims:
            return Answer(
                query=query,
                text=(
                    "No confident answer available.\n\n"
                    "No passage in the indexed domains cleared the evidence threshold for "
                    "this question. Rather than assemble an answer from weak matches, the "
                    "system is abstaining.\n\n"
                    "Suggested next steps: rephrase using the vocabulary of the relevant "
                    "domain, or check whether the governing document has been ingested."
                ),
                confidence=0.0,
                domains=tuple(domains),
                abstained=True,
                degraded=degraded,
                notes=tuple(notes),
                synthesizer=self.name,
            )

        overruled = {c.loser.claim_id for c in conflicts}
        deduped = self._dedupe([c for c in claims if c.claim_id not in overruled])

        # Select what the answer will actually say before minting any citations.
        # Doing it in the other order produces a source list padded with
        # documents the reader never sees quoted.
        required = {c.winner.claim_id for c in conflicts}
        kept = self._select(deduped, domains, required=required)

        # Citations cover everything that appears anywhere in the output,
        # conflict losers included -- an overruled claim is quoted in the conflict
        # section and must be just as traceable as an adopted one.
        cited = kept + [c.loser for c in conflicts if c.loser.claim_id in overruled]
        citations, marker_of = self._build_citations(cited)

        sections: List[str] = [self._lead(query, kept, domains, conflicts)]
        sections.extend(self._domain_sections(kept, domains, marker_of))
        if conflicts:
            sections.append(self._conflict_section(conflicts, marker_of))
        confidence = self._confidence(kept, conflicts, degraded)
        sections.append(self._confidence_section(confidence, kept, conflicts, degraded, notes))
        sections.append("Sources\n" + "\n".join("  " + c.render() for c in citations))

        return Answer(
            query=query,
            text="\n\n".join(s for s in sections if s),
            confidence=confidence,
            domains=tuple(domains),
            citations=tuple(citations),
            claims=tuple(kept),
            conflicts=tuple(conflicts),
            degraded=degraded,
            notes=tuple(notes),
            synthesizer=self.name,
        )

    # -- assembly --------------------------------------------------------- #

    def _dedupe(self, claims: Sequence[Claim]) -> List[Claim]:
        """Drop near-duplicate claims, keeping the most trusted phrasing.

        Several documents restate the same rule; repeating it three times with
        three citations looks thorough and reads as padding.
        """
        kept: List[Claim] = []
        for claim in sorted(claims, key=lambda c: -c.confidence):
            if any(jaccard(claim.tokens, other.tokens) >= self.dedupe_threshold
                   for other in kept):
                continue
            kept.append(claim)
        return kept

    def _select(
        self,
        claims: Sequence[Claim],
        domains: Sequence[str],
        *,
        required: Optional[set] = None,
    ) -> List[Claim]:
        """Choose which claims the answer states, ranked by topicality.

        The rule: **each contributing domain shows its single best finding, plus
        any others that clear the topicality bar.** Both halves matter.

        Showing the best regardless of score guarantees a domain that was routed
        to is actually represented, so a weak-but-real contribution is visible
        rather than silently dropped. Gating the rest on topicality stops the
        padding that the previous confidence-ordered version produced: asked
        about deployment approvals, the technical agent filled its four slots
        with API gateway retry and timeout guidance, because those passages came
        from a recent high-authority document and nothing was checking whether
        they addressed the question.

        A domain whose corpus genuinely has little to say about a question should
        say little. Fewer, on-topic findings beat four slots filled.

        Args:
            claims: Deduplicated, non-overruled claims.
            domains: Selected domains, in routing order.
            required: Claim ids that must be included whatever they score --
                conflict winners, since the conflict section refers to them.

        Returns:
            Selected claims, ordered by domain then by descending topicality.
        """
        required = required or set()
        selected: List[Claim] = []

        for domain in domains:
            in_domain = sorted(
                (c for c in claims if c.domain == domain),
                key=lambda c: -c.topicality,
            )
            if not in_domain:
                continue
            best = in_domain[0]
            floor = max(self.topicality_floor, self.relative_floor * best.topicality)
            chosen = [
                c for c in in_domain
                if c is best or c.claim_id in required or c.topicality >= floor
            ]
            selected.extend(chosen[: self.max_claims_per_domain])

        # Any required claim from a domain that was not routed to (possible when
        # a conflict spans an unexpected domain) is appended rather than lost.
        chosen_ids = {c.claim_id for c in selected}
        selected.extend(c for c in claims
                        if c.claim_id in required and c.claim_id not in chosen_ids)
        return selected

    def _build_citations(self, claims: Sequence[Claim]) -> Tuple[List[Citation], Dict[str, str]]:
        """Assign markers in order of first appearance, one per claim span."""
        citations: List[Citation] = []
        marker_of: Dict[str, str] = {}
        for claim in claims:
            marker = f"[{len(citations) + 1}]"
            marker_of[claim.claim_id] = marker
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=claim.doc_id,
                    domain=claim.domain,
                    title=claim.document.title,
                    section=claim.document.section,
                    version=claim.document.version,
                    authority=claim.document.authority,
                    effective_date=claim.document.effective_date,
                    quote=claim.text,
                    char_start=claim.char_start,
                    char_end=claim.char_end,
                )
            )
        return citations, marker_of

    @staticmethod
    def _lead(query: str, claims: Sequence[Claim], domains: Sequence[str],
              conflicts: Sequence[Conflict]) -> str:
        active = [d for d in domains if any(c.domain == d for c in claims)]
        domain_phrase = (
            f"{active[0]} guidance" if len(active) == 1
            else " and ".join([", ".join(active[:-1]), active[-1]]) + " guidance"
        )
        docs = len({c.doc_id for c in claims})
        lead = (f'Question: "{truncate(query, 150)}"\n\n'
                f"Answered from {domain_phrase}: {len(claims)} findings across {docs} "
                f"source document(s).")
        if conflicts:
            lead += (f" {len(conflicts)} contradiction(s) between sources were detected "
                     f"and resolved -- see below.")
        return lead

    def _domain_sections(self, claims: Sequence[Claim], domains: Sequence[str],
                         marker_of: Dict[str, str]) -> List[str]:
        sections: List[str] = []
        for domain in domains:
            subset = sorted((c for c in claims if c.domain == domain),
                            key=lambda c: -c.topicality)
            if not subset:
                continue
            lines = [f"{domain.capitalize()} ({len(subset)} finding(s))"]
            for claim in subset:
                lines.append(f"  - {claim.text} {marker_of[claim.claim_id]}")
            sections.append("\n".join(lines))
        return sections

    @staticmethod
    def _conflict_section(conflicts: Sequence[Conflict],
                          marker_of: Dict[str, str]) -> str:
        lines = [f"Conflicting guidance ({len(conflicts)} resolved)"]
        for conflict in conflicts:
            adopted = marker_of.get(conflict.winner.claim_id, "")
            overruled = marker_of.get(conflict.loser.claim_id, "")
            lines.append(f"  ! {conflict.subject}")
            lines.append(f"    adopted   {adopted} {conflict.winner.document.citation_label} "
                         f"({conflict.winner.domain}, {conflict.winner.document.authority}, "
                         f"{conflict.winner.document.effective_date})")
            lines.append(f'              "{truncate(conflict.winner.text, 120)}"')
            lines.append(f"    overruled {overruled} {conflict.loser.document.citation_label} "
                         f"({conflict.loser.domain}, {conflict.loser.document.authority}, "
                         f"{conflict.loser.document.effective_date})")
            lines.append(f'              "{truncate(conflict.loser.text, 120)}"')
            lines.append(f"    basis     {conflict.explanation}; margin {conflict.margin:.3f}"
                         + ("" if conflict.decisive else " -- NEAR TIE, treat as unsettled"))
        return "\n".join(lines)

    def _confidence(self, claims: Sequence[Claim], conflicts: Sequence[Conflict],
                    degraded: bool) -> float:
        """Blend claim confidence with structural penalties.

        Agreement between independent documents raises confidence; an unresolved
        near-tie and a failed agent both lower it. A system that reports high
        confidence while one of its agents crashed is lying by omission.
        """
        if not claims:
            return 0.0
        weights = [c.confidence for c in claims]
        total = sum(weights)
        if total <= 0.0:
            # Reachable: an agent configured with ``trust_weight=0.0`` scales
            # every claim confidence to zero. Claims still exist, so the
            # abstention branch above does not catch it, and the weighted mean
            # below would divide by zero. Zero confidence is the honest answer.
            return 0.0
        base = sum(w * w for w in weights) / total  # confidence-weighted mean

        corroboration = min(1.0, len({c.doc_id for c in claims}) / 3.0)
        score = 0.82 * base + 0.18 * corroboration

        near_ties = sum(1 for c in conflicts if not c.decisive)
        score -= 0.08 * near_ties
        if degraded:
            score -= 0.12
        return float(round(max(0.0, min(1.0, score)), 4))

    @staticmethod
    def _confidence_section(confidence: float, claims: Sequence[Claim],
                            conflicts: Sequence[Conflict], degraded: bool,
                            notes: Sequence[str]) -> str:
        label = confidence_label(confidence)
        parts = [f"{len(claims)} finding(s) from {len({c.doc_id for c in claims})} document(s)"]
        if conflicts:
            decisive = sum(1 for c in conflicts if c.decisive)
            parts.append(f"{decisive}/{len(conflicts)} conflict(s) resolved decisively")
        if degraded:
            parts.append("one or more agents failed -- answer is partial")
        text = f"Confidence: {confidence:.2f} ({label}) -- " + "; ".join(parts) + "."
        if notes:
            text += "\n" + "\n".join(f"  note: {n}" for n in notes)
        return text


class ClaudeSynthesizer(Synthesizer):
    """Generative synthesis via the Anthropic API. Optional, off by default.

    Activates only when an API key is resolvable, so the default demo path stays
    offline and deterministic. The prompt hands Claude the numbered claims and
    the already-resolved conflicts and asks it to write prose over them -- the
    model does *not* get to re-adjudicate conflicts or introduce facts, because
    conflict resolution here is a governance decision with an auditable basis
    and should not be delegated to a sampled output.

    Any API failure falls back to the extractive synthesiser rather than
    propagating, so enabling this can degrade the prose but never lose the
    answer.

    Args:
        model: Model ID. Defaults to Claude Opus 5.
        max_tokens: Output cap. Deliberately modest -- the task is a bounded
            synthesis over supplied material, not open-ended generation.
    """

    name = "claude"

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        fallback: Optional[Synthesizer] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.fallback = fallback or ExtractiveSynthesizer()
        self._client: Any = None

    @staticmethod
    def available() -> bool:
        """Whether both the SDK and a credential are present."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = (anthropic.Anthropic(api_key=self.api_key) if self.api_key
                            else anthropic.Anthropic())
        return self._client

    SYSTEM_PROMPT = (
        "You are the synthesis stage of a multi-agent retrieval system for internal "
        "engineering, business and compliance documentation.\n\n"
        "You are given numbered findings retrieved and scored by domain agents, and a list "
        "of contradictions those agents already resolved. Write the answer to the user's "
        "question using only those findings.\n\n"
        "Rules:\n"
        "1. Use only the supplied findings. Introduce no facts of your own. If the findings "
        "do not answer part of the question, say so explicitly.\n"
        "2. Cite every factual sentence with the bracketed marker of the finding it rests "
        "on, e.g. [3]. Never cite a marker you were not given.\n"
        "3. Do not re-adjudicate the supplied conflicts. Report the adopted guidance as "
        "authoritative, state that the overruled guidance exists, and give the stated basis "
        "for the decision. A reader following the overruled guidance needs to know.\n"
        "4. Organise by what the reader must do, not by which agent found what.\n"
        "5. Be concise and concrete. Prefer the exact figures, named bodies and deadlines "
        "in the findings over paraphrase."
    )

    def synthesize(
        self,
        query: str,
        claims: Sequence[Claim],
        conflicts: Sequence[Conflict],
        *,
        domains: Sequence[str],
        notes: Sequence[str] = (),
        degraded: bool = False,
    ) -> Answer:
        """Generate prose over the supplied findings, falling back on any failure."""
        # The extractive pass is run regardless: it establishes the citation
        # numbering the model is told to use, and it is the fallback if the API
        # call fails. Sharing the numbering means a cited marker means the same
        # thing whichever synthesiser produced the text.
        baseline = self.fallback.synthesize(
            query, claims, conflicts, domains=domains, notes=notes, degraded=degraded
        )
        if baseline.abstained or not self.available():
            if not baseline.abstained:
                LOGGER.info("ClaudeSynthesizer unavailable (no SDK or credential); "
                            "using %s", self.fallback.name)
            return baseline

        try:
            client = self._ensure_client()
            # Deliberately the stable `messages.create` surface. An earlier
            # version passed `betas=[...]`/`fallbacks="default"` here to get a
            # server-side refusal fallback, but those arguments only exist on
            # `client.beta.messages.create`; on the stable method the SDK raises
            # TypeError before any request is sent, which the broad except below
            # then swallowed -- so the generative path silently never ran. The
            # server-side fallback is redundant anyway: a refusal, an error, or
            # an empty body all land on the deterministic answer below.
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._build_prompt(query, baseline)}],
            )
            if getattr(response, "stop_reason", None) == "refusal":
                LOGGER.warning("synthesis refused by the model; using %s", self.fallback.name)
                return baseline

            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            if not text:
                return baseline

            # Sources and conflict detail are appended from our own records
            # rather than trusting the model to reproduce them faithfully.
            appendix = "\n\nSources\n" + "\n".join("  " + c.render() for c in baseline.citations)
            return Answer(
                query=query,
                text=text + appendix,
                confidence=baseline.confidence,
                domains=tuple(domains),
                citations=baseline.citations,
                claims=baseline.claims,
                conflicts=baseline.conflicts,
                degraded=degraded,
                notes=(*notes, f"prose synthesised by {self.model}"),
                synthesizer=f"{self.name}:{self.model}",
            )
        except ImportError:
            return baseline
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            # Broad on purpose: every API failure mode (auth, rate limit, 5xx,
            # timeout, connection) has the same correct response here, which is
            # to serve the deterministic answer instead of raising into a demo.
            LOGGER.warning("Claude synthesis failed (%s: %s); using %s",
                           type(exc).__name__, exc, self.fallback.name)
            return baseline

    @staticmethod
    def _build_prompt(query: str, baseline: Answer) -> str:
        lines = [f"User question: {query}", "", "Findings:"]
        # Not strict, and deliberately so: ``citations`` also covers the
        # conflict losers quoted in the conflict section, so it is longer than
        # ``claims``. Truncating to the displayed claims is the intended
        # pairing -- markers line up because citations are minted over
        # ``kept + losers``, in that order.
        for citation, claim in zip(baseline.citations, baseline.claims, strict=False):
            lines.append(
                f"{citation.marker} ({claim.domain}, {claim.document.authority}, "
                f"effective {claim.document.effective_date}, confidence "
                f"{claim.confidence:.2f}) {claim.text}"
            )
        if baseline.conflicts:
            lines += ["", "Conflicts already resolved by the agents (do not re-decide):"]
            for conflict in baseline.conflicts:
                lines.append(
                    f"- {conflict.subject}\n"
                    f"  adopted: {conflict.winner.document.citation_label} -- "
                    f"{conflict.winner.text}\n"
                    f"  overruled: {conflict.loser.document.citation_label} -- "
                    f"{conflict.loser.text}\n"
                    f"  basis: {conflict.explanation}"
                )
        if baseline.degraded:
            lines += ["", "Note: at least one domain agent failed. Say that the answer "
                           "may be incomplete."]
        lines += ["", "Write the answer now. Do not include a Sources list; one is appended "
                      "automatically."]
        return "\n".join(lines)
