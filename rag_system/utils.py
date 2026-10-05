"""Shared primitives for the multi-agent RAG system.

This module deliberately holds only *generic* building blocks -- data models,
text processing, scoring helpers and small utilities. Anything that encodes a
policy decision (how to route, how to retrieve, how to resolve a conflict)
lives in the module that owns that decision.

Nothing here requires a network connection or a third-party model, which is
what allows the whole system to run deterministically offline.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The three mock knowledge domains served by dedicated RAG agents.
DOMAINS: Tuple[str, ...] = ("technical", "business", "compliance")

#: Source authority tiers, highest first. Used by conflict resolution: a policy
#: statement outranks a runbook describing current practice, because the runbook
#: may simply be out of date with respect to the policy.
AUTHORITY_WEIGHTS: Dict[str, float] = {
    "policy": 1.00,
    "standard": 0.85,
    "runbook": 0.65,
    "wiki": 0.40,
}
DEFAULT_AUTHORITY = "wiki"

#: Which domain's statement governs when two domains contradict each other.
#:
#: This encodes an organisational fact rather than a modelling convenience: a
#: compliance policy *defines* the rule, a business standard *allocates* the
#: decision, and a technical runbook *describes current practice*. When practice
#: and policy disagree, practice is the thing that is out of date. Retrieval
#: score deliberately does not decide this -- the runbook is usually the better
#: lexical match for an engineer's phrasing, and letting that win would invert
#: the correct answer.
DOMAIN_PRECEDENCE: Dict[str, float] = {
    "compliance": 1.00,
    "business": 0.85,
    "technical": 0.70,
}

STOPWORDS: frozenset = frozenset("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does
doesn doing don down during each few for from further had hadn has hasn have haven
having he her here hers herself him himself his how i if in into is isn it its
itself just ll me more most mustn my myself no nor not now o of off on once only or
other our ours ourselves out over own re s same shan she should shouldn so some such
t than that the their theirs them themselves then there these they this those through
to too under until up ve very was wasn we were weren what when where which while who
whom why will with won would wouldn you your yours yourself yourselves
""".split())

#: Markers that flip the polarity of a statement. Used by the polarity-based
#: contradiction detector.
#: Deliberately excludes "without" and "unless". Both read as negations but
#: scope a subordinate clause rather than the main assertion: "reload credentials
#: without a restart" is a requirement, not a prohibition, and "purged within 30
#: days unless a legal hold applies" is still a purge rule. Including them
#: produced false contradictions between two sentences that agreed.
NEGATION_MARKERS: frozenset = frozenset({
    "not", "never", "no", "cannot", "prohibited", "forbidden", "disallowed",
    "must-not", "may-not", "banned", "denied", "refuse", "reject",
    "rejected", "withdrawn",
})

#: Words that spell out small numbers. Needed because policy prose says
#: "two independent reviewers", not "2 reviewers".
NUMBER_WORDS: Dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}

#: Canonical unit families. Values within a family are comparable after being
#: converted to the family's base unit; values across families are not.
UNIT_FAMILIES: Dict[str, Tuple[str, float]] = {
    # token          (family,     multiplier to base unit)
    "second": ("duration", 1 / 86400), "seconds": ("duration", 1 / 86400),
    "minute": ("duration", 1 / 1440), "minutes": ("duration", 1 / 1440),
    "hour": ("duration", 1 / 24), "hours": ("duration", 1 / 24),
    "day": ("duration", 1.0), "days": ("duration", 1.0),
    "week": ("duration", 7.0), "weeks": ("duration", 7.0),
    "month": ("duration", 30.0), "months": ("duration", 30.0),
    "quarter": ("duration", 91.0), "quarters": ("duration", 91.0),
    "year": ("duration", 365.0), "years": ("duration", 365.0),
    "percent": ("ratio", 1.0),
    "eur": ("currency", 1.0), "usd": ("currency", 1.0),
    "reviewer": ("people", 1.0), "reviewers": ("people", 1.0),
    "approver": ("people", 1.0), "approvers": ("people", 1.0),
    "signatory": ("people", 1.0), "signatories": ("people", 1.0),
    "replica": ("count", 1.0), "replicas": ("count", 1.0),
    "attempt": ("count", 1.0), "attempts": ("count", 1.0),
    "retry": ("count", 1.0), "retries": ("count", 1.0),
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_NUMBER_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\b"
)


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Document:
    """A single source document in one of the mock knowledge domains.

    The metadata fields are not decoration: ``authority``, ``effective_date``,
    ``version``, ``status`` and ``supersedes`` are all consumed by conflict
    resolution to decide which of two contradictory statements to trust.
    """

    doc_id: str
    domain: str
    title: str
    text: str
    section: str = ""
    authority: str = DEFAULT_AUTHORITY
    version: str = "1.0"
    effective_date: str = "2025-01-01"
    supersedes: Optional[str] = None
    status: str = "active"
    tags: Tuple[str, ...] = ()

    @property
    def authority_weight(self) -> float:
        return AUTHORITY_WEIGHTS.get(self.authority, AUTHORITY_WEIGHTS[DEFAULT_AUTHORITY])

    @property
    def effective(self) -> date:
        return parse_date(self.effective_date)

    @property
    def citation_label(self) -> str:
        """Human-facing source label, e.g. ``COMP-CHG-002 v2.0``."""
        return f"{self.doc_id} v{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON Lines output."""
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Document":
        """Build a ``Document`` from a corpus record, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in payload.items() if k in known}
        missing = {"doc_id", "domain", "title", "text"} - data.keys()
        if missing:
            raise ValueError(f"document record missing required field(s): {sorted(missing)}")
        if "tags" in data and data["tags"] is not None:
            data["tags"] = tuple(data["tags"])
        else:
            data.pop("tags", None)
        return cls(**data)


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage. Chunks, not documents, are embedded and ranked.

    ``char_start``/``char_end`` are offsets into the parent document's text,
    which is what makes a citation verifiable rather than merely plausible.
    """

    chunk_id: str
    doc_id: str
    domain: str
    text: str
    char_start: int
    char_end: int
    ordinal: int = 0


@dataclass
class RetrievedChunk:
    """A chunk together with the scores that got it retrieved.

    Two score fields, because ranking and confidence are different jobs:

    * ``score`` is the fused reciprocal-rank score. It orders results well but
      is not interpretable in absolute terms -- RRF values cluster tightly
      around ``1/k`` by construction.
    * ``relevance`` is an absolute, calibrated blend of cosine similarity and
      saturated BM25 in ``[0, 1]``. This is the number an agent thresholds
      against to decide whether to answer or abstain, and the number that feeds
      downstream confidence scoring.

    Keeping the component scores as well (not just the totals) is what lets the
    notebook show *why* a change in retrieval policy changed the outcome.
    """

    chunk: Chunk
    document: Document
    score: float
    relevance: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    retrieved_by: str = ""

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    def describe(self) -> str:
        """One-line summary of this passage and the scores that retrieved it."""
        return (
            f"{self.document.citation_label} [{self.document.domain}] "
            f"rel={self.relevance:.3f} (dense={self.dense_score:.3f}, "
            f"bm25={self.lexical_score:.2f}) -- {self.document.title}"
        )


@dataclass
class Measurement:
    """A quantity extracted from prose, normalised for comparison.

    ``family`` groups comparable units ("days" and "months" are both duration);
    ``base_value`` is expressed in that family's base unit so that 30 days and
    1 month compare equal. ``anchors`` are the salient nouns near the quantity,
    used to decide whether two measurements are even talking about the same
    thing before declaring them contradictory.
    """

    raw: str
    value: float
    unit: str
    family: str
    base_value: float
    anchors: Tuple[str, ...]


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #

def normalize_text(text: str) -> str:
    """Casefold, strip accents and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _stem(token: str) -> str:
    """Crude suffix stripping.

    A real system would use a proper stemmer; this handles the plural/gerund
    variation that actually matters in this corpus ("deployment"/"deployments",
    "approve"/"approving") without pulling in a dependency.
    """
    for suffix, keep in (("ies", 3), ("sses", 2), ("ing", 3), ("ed", 2), ("s", 1)):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            base = token[: -keep] if suffix == "ies" else token[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            return base
    return token


def tokenize(text: str, *, stem: bool = True, keep_stopwords: bool = False) -> List[str]:
    """Split text into normalised content tokens.

    Args:
        text: Raw input.
        stem: Apply crude suffix stripping so morphological variants collide.
        keep_stopwords: Retain function words. Needed for polarity detection,
            where "must not" is the entire signal.
    """
    tokens = _WORD_RE.findall(normalize_text(text))
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [_stem(t) for t in tokens] if stem else tokens


def split_sentences(text: str) -> List[str]:
    """Split a passage into sentences.

    Regex sentence splitting is imperfect in general; this corpus is clean
    declarative prose, and claim-level conflict detection needs sentence
    granularity, so the trade-off is acceptable and the failure mode (an
    occasional merged sentence) is benign.
    """
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_document(
    doc: Document,
    *,
    target_tokens: int = 70,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    """Split a document into sentence-aligned, lightly overlapping chunks.

    Sentence alignment keeps claims intact -- splitting mid-sentence would
    produce passages that cite cleanly but read as fragments, and would break
    the numeric/polarity comparisons that conflict detection relies on.

    Args:
        doc: Document to chunk.
        target_tokens: Soft upper bound on chunk size in whitespace tokens.
        overlap_sentences: Trailing sentences repeated into the next chunk, so a
            claim spanning a boundary is still retrievable as a whole.

    Returns:
        Chunks with character offsets into ``doc.text``.
    """
    sentences = split_sentences(doc.text)
    if not sentences:
        return []

    # Locate each sentence in the source text so offsets survive chunking.
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for sent in sentences:
        start = doc.text.find(sent, cursor)
        if start == -1:  # defensive: splitter altered the text
            start = cursor
        end = start + len(sent)
        spans.append((sent, start, end))
        cursor = end

    chunks: List[Chunk] = []
    buffer: List[Tuple[str, int, int]] = []
    size = 0
    ordinal = 0

    def flush() -> None:
        nonlocal buffer, size, ordinal
        if not buffer:
            return
        text = " ".join(s for s, _, _ in buffer)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{ordinal}",
                doc_id=doc.doc_id,
                domain=doc.domain,
                text=text,
                char_start=buffer[0][1],
                char_end=buffer[-1][2],
                ordinal=ordinal,
            )
        )
        ordinal += 1
        tail = buffer[-overlap_sentences:] if overlap_sentences else []
        buffer = list(tail)
        size = sum(len(s.split()) for s, _, _ in buffer)

    for sent, start, end in spans:
        length = len(sent.split())
        if size and size + length > target_tokens:
            flush()
        buffer.append((sent, start, end))
        size += length

    # Final flush; suppress a trailing chunk that is pure overlap of the last one.
    if buffer and not (chunks and len(buffer) <= overlap_sentences):
        tail_only = buffer
        buffer = tail_only
        text = " ".join(s for s, _, _ in buffer)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{ordinal}",
                doc_id=doc.doc_id,
                domain=doc.domain,
                text=text,
                char_start=buffer[0][1],
                char_end=buffer[-1][2],
                ordinal=ordinal,
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Quantity extraction (consumed by conflict detection)
# --------------------------------------------------------------------------- #

def _to_number(token: str) -> Optional[float]:
    token = token.replace(",", "")
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])
    try:
        return float(token)
    except ValueError:
        return None


def extract_measurements(
    sentence: str,
    *,
    anchor_window: int = 6,
    subject_prefix: int = 3,
) -> List[Measurement]:
    """Pull comparable quantities out of a sentence.

    A quantity is only useful for contradiction detection if we also know what
    it measures, so each measurement carries the salient content words near it
    as ``anchors``. "logs retained for 90 days" and "logs purged within 30 days"
    share the anchor ``log`` and the duration family, which is what lets the
    resolver notice they disagree.

    Anchors come from two places, because a local window alone is not enough.
    English puts the thing a sentence is about in subject position, which is
    often further from the quantity than any reasonable window: in "Logs
    containing personal data must be purged within 30 days", the subject "logs"
    sits eight tokens before the number. A window wide enough to reach it would
    also swallow unrelated clauses, so the leading content tokens are added
    directly instead.

    Args:
        sentence: A single sentence of prose.
        anchor_window: Tokens either side of the quantity to scan for anchors.
        subject_prefix: Leading content tokens always treated as anchors, on the
            assumption they name the sentence's subject.

    Returns:
        One ``Measurement`` per recognised quantity+unit pair.
    """
    raw_tokens = _WORD_RE.findall(normalize_text(sentence))
    found: List[Measurement] = []

    def is_anchor_token(token: str) -> bool:
        return (
            token not in STOPWORDS
            and token not in UNIT_FAMILIES
            and len(token) > 2
            and _to_number(token) is None
        )

    subject_anchors = [t for t in raw_tokens if is_anchor_token(t)][:subject_prefix]

    for idx, token in enumerate(raw_tokens):
        value = _to_number(token)
        if value is None:
            continue

        # The unit is the next token, allowing one intervening modifier
        # ("90 calendar days", "two independent reviewers").
        unit_token = None
        for offset in (1, 2, 3):
            if idx + offset >= len(raw_tokens):
                break
            candidate = raw_tokens[idx + offset]
            if candidate in UNIT_FAMILIES:
                unit_token = candidate
                break
        if unit_token is None:
            continue

        family, multiplier = UNIT_FAMILIES[unit_token]
        lo = max(0, idx - anchor_window)
        hi = min(len(raw_tokens), idx + anchor_window + 1)
        local = [t for t in raw_tokens[lo:hi] if is_anchor_token(t)]
        anchors = tuple(sorted({_stem(t) for t in local + subject_anchors}))
        found.append(
            Measurement(
                raw=f"{token} {unit_token}",
                value=value,
                unit=unit_token,
                family=family,
                base_value=value * multiplier,
                anchors=anchors,
            )
        )
    return found


def polarity(sentence: str) -> int:
    """Return ``-1`` for a prohibitive/negated statement, ``+1`` otherwise.

    Multi-word prohibitions ("must not", "may not") are joined first so that a
    single scan over tokens catches them.
    """
    text = normalize_text(sentence)
    text = re.sub(r"\b(must|may|should|can|will|shall)\s+not\b", r"\1-not", text)
    text = re.sub(r"\bis\s+not\b|\bare\s+not\b|\bdo\s+not\b|\bdoes\s+not\b", "not", text)
    tokens = _WORD_RE.findall(text)
    hits = sum(1 for t in tokens if t in NEGATION_MARKERS)
    return -1 if hits % 2 == 1 else 1


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard overlap of two token collections. ``0.0`` when either is empty."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``matrix`` and ``vector``.

    Zero vectors yield ``0.0`` rather than ``nan``, which matters because an
    out-of-vocabulary query embeds to zero and must not poison the ranking.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    mat_norm = np.linalg.norm(matrix, axis=1)
    vec_norm = float(np.linalg.norm(vector))
    if vec_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    denom = np.where(mat_norm == 0.0, 1.0, mat_norm) * vec_norm
    sims = (matrix @ vector) / denom
    sims[mat_norm == 0.0] = 0.0
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def calibrated_relevance(
    dense_score: float,
    lexical_score: float,
    *,
    dense_weight: float = 0.6,
    lexical_saturation: float = 8.0,
) -> float:
    """Blend cosine similarity and BM25 into an absolute ``[0, 1]`` relevance.

    Fused rank scores are good for ordering and useless for thresholding. This
    gives an absolute figure instead, which is what abstention and confidence
    scoring actually need: a query with no real match in the corpus scores near
    zero here, whereas it would still produce a top-ranked RRF result.

    BM25 is squashed through ``min(1, score / saturation)`` because its upper
    bound is corpus-dependent -- without saturation, one rare-term match could
    swamp the semantic signal.
    """
    dense = max(0.0, min(1.0, dense_score))
    lexical = max(0.0, min(1.0, lexical_score / lexical_saturation)) if lexical_saturation else 0.0
    return float(dense_weight * dense + (1.0 - dense_weight) * lexical)


def minmax_normalize(scores: Sequence[float]) -> List[float]:
    """Scale scores to ``[0, 1]``. A flat input maps to all ones."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if math.isclose(hi, lo):
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Optional[Sequence[float]] = None,
    k: int = 60,
) -> Dict[str, float]:
    """Fuse several ranked ID lists into one score per ID.

    RRF is used instead of summing raw scores because dense cosine and BM25 live
    on incomparable scales; rank position is the common currency. ``k`` damps the
    influence of the very top ranks.

    Args:
        rankings: Ranked ID lists, best first.
        weights: Per-ranking weight. Defaults to uniform.
        k: Smoothing constant.

    Returns:
        Mapping of ID to fused score.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must be the same length")

    fused: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + position + 1)
    return fused


def recency_weight(effective: date, *, now: Optional[date] = None, half_life_days: float = 540.0) -> float:
    """Exponential recency decay in ``(0, 1]``.

    Used so that a newer document outranks an older one of equal authority.
    Documents dated in the future (a scheduled policy) score ``1.0`` rather than
    being penalised.
    """
    now = now or date.today()
    age_days = max(0.0, (now - effective).days)
    return float(0.5 ** (age_days / half_life_days))


def parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD``, falling back to a neutral epoch on bad input."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date(2000, 1, 1)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #

def default_data_dir() -> Path:
    """Path to the bundled synthetic corpus."""
    return Path(__file__).resolve().parent.parent / "data" / "synthetic"


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from a JSON Lines file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON -- {exc}") from exc


def load_corpus(data_dir: Optional[Path] = None, domains: Sequence[str] = DOMAINS) -> List[Document]:
    """Load the synthetic corpus for the requested domains.

    Args:
        data_dir: Directory holding ``<domain>.jsonl`` files. Defaults to the
            bundled corpus.
        domains: Which domain files to read.

    Returns:
        All documents, in file order.

    Raises:
        FileNotFoundError: If a requested domain file is absent.
    """
    base = Path(data_dir) if data_dir else default_data_dir()
    documents: List[Document] = []
    for domain in domains:
        path = base / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no corpus file for domain {domain!r} at {path}")
        for record in load_jsonl(path):
            record.setdefault("domain", domain)
            documents.append(Document.from_dict(record))
    return documents


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

class Stopwatch:
    """Context manager measuring wall-clock milliseconds.

    >>> with Stopwatch() as sw:
    ...     pass
    >>> sw.elapsed_ms >= 0.0
    True
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def get_logger(name: str = "rag_system", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a handler at most once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def truncate(text: str, limit: int = 160) -> str:
    """Shorten text for log lines and tables, on a word boundary where possible."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + "…"
