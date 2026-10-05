"""End-to-end scenario tests.

These cover the behaviour the assignment asks to be demonstrated, plus the
failure modes that matter more than the happy path: abstaining instead of
guessing, degrading instead of crashing, and citations that actually point at
the text they quote.

Everything runs offline against the bundled synthetic corpus in a few seconds.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Set

import pytest

from rag_system.domain_agents import DomainAgent
from rag_system.orchestrator import RAGOrchestrator
from rag_system.protocol import MessageType
from rag_system.query_classifier import QueryIntent
from rag_system.synthesis import ClaimExtractor, ConflictResolver
from rag_system.utils import Document, load_corpus
from rag_system.vector_store import KnowledgeBase, RetrievalParams

from .conftest import ASSIGNMENT_SCENARIOS


# --------------------------------------------------------------------------- #
# The assignment's named scenarios
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query,expected", ASSIGNMENT_SCENARIOS)
def test_assignment_scenario_routes_to_expected_domains(
    orchestrator: RAGOrchestrator, query: str, expected: Set[str]
) -> None:
    """Each named scenario must be recognised as spanning its two domains."""
    classification = orchestrator.classifier.classify(query)
    assert set(classification.domains) == expected, classification.describe()
    assert classification.is_multi_domain


@pytest.mark.parametrize("query,expected", ASSIGNMENT_SCENARIOS)
def test_assignment_scenario_produces_cited_answer(
    orchestrator: RAGOrchestrator, query: str, expected: Set[str]
) -> None:
    """Each scenario must yield a real answer with verifiable citations."""
    answer = orchestrator.answer(query, expected_domains=sorted(expected))

    assert not answer.abstained
    assert not answer.degraded
    assert answer.citations, "an answer with no citations is not traceable"
    assert answer.confidence > 0.3

    # Both domains must actually contribute; routing to a domain that then
    # contributes nothing is a silent failure.
    contributing = {claim.domain for claim in answer.claims}
    assert contributing == expected, f"contributing domains {contributing} != {expected}"

    # Every citation marker used in the text must be defined in the source list.
    for citation in answer.citations:
        assert citation.marker in answer.text


@pytest.mark.parametrize("query,expected", ASSIGNMENT_SCENARIOS)
def test_assignment_scenario_decomposes_per_domain(
    orchestrator: RAGOrchestrator, query: str, expected: Set[str]
) -> None:
    """Decomposition must produce one scoped sub-query per selected domain."""
    plan = orchestrator.plan(query)
    assert {s.domain for s in plan.sub_queries} == expected
    for sub_query in plan.sub_queries:
        assert sub_query.text.strip()
        assert sub_query.rationale.strip()
        assert 2 <= sub_query.params.top_k <= 6


# --------------------------------------------------------------------------- #
# Query classification
# --------------------------------------------------------------------------- #

def test_single_domain_query_does_not_fan_out(orchestrator: RAGOrchestrator) -> None:
    classification = orchestrator.classifier.classify(
        "What is the budget approval threshold for vendor spend?")
    assert classification.domains[0] == "business"


def test_out_of_domain_query_abstains(orchestrator: RAGOrchestrator) -> None:
    """An unanswerable question must abstain rather than answer from noise."""
    answer = orchestrator.answer("How do I bake a sourdough loaf at home?")
    assert answer.abstained
    assert answer.confidence == 0.0
    assert not answer.citations
    assert "abstaining" in answer.text.lower()


def test_empty_query_raises(orchestrator: RAGOrchestrator) -> None:
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            orchestrator.answer(bad)


def test_intent_detection(orchestrator: RAGOrchestrator) -> None:
    detect = orchestrator.classifier.detect_intent
    assert detect("How do I troubleshoot API latency?") is QueryIntent.DIAGNOSTIC
    assert detect("What approvals are required to ship this?") is QueryIntent.APPROVAL
    assert detect("What's the process for deploying a service?") is QueryIntent.PROCEDURAL


def test_complexity_increases_with_query_structure(orchestrator: RAGOrchestrator) -> None:
    simple = orchestrator.classifier.classify("What is our log retention period?")
    compound = orchestrator.classifier.classify(
        "What's the process for deploying a new microservice, and what compliance "
        "checks are needed, and who approves the spend?")
    assert compound.complexity > simple.complexity


def test_clause_anchoring_restores_a_stranded_subject(
    orchestrator: RAGOrchestrator
) -> None:
    """A clause that lost its subject must get it back before retrieval."""
    query = ("What's the process for deploying a new microservice and what "
             "compliance checks are needed?")
    anchored = orchestrator.classifier.anchor_clause("compliance checks are needed", query)
    assert "microservice" in anchored.lower()


# --------------------------------------------------------------------------- #
# Dynamic retrieval strategy
# --------------------------------------------------------------------------- #

def test_retrieval_params_respond_to_complexity(orchestrator: RAGOrchestrator) -> None:
    """A harder query must earn a larger search budget."""
    policy = orchestrator.classifier.policy
    simple = orchestrator.classifier.classify("What is our log retention period?")
    compound = orchestrator.classifier.classify(
        "What's the process for deploying a new microservice and what compliance "
        "checks are needed and what business approvals are required?")

    simple_params = policy.for_sub_query(simple.domains[0], simple)
    compound_params = policy.for_sub_query(compound.domains[0], compound)

    assert compound_params.top_k >= simple_params.top_k
    assert compound_params.candidate_pool > simple_params.candidate_pool
    # Broader questions need cross-document coverage, so diversity rises.
    assert compound_params.mmr_lambda < simple_params.mmr_lambda


def test_requirement_intent_favours_lexical_matching(
    orchestrator: RAGOrchestrator
) -> None:
    """Exact figures matter for requirements; paraphrase matters for diagnosis."""
    policy = orchestrator.classifier.policy
    requirement = orchestrator.classifier.classify(
        "What approvals are required before a production release?")
    diagnostic = orchestrator.classifier.classify(
        "Why is the checkout service suddenly slow?")

    req_params = policy.for_sub_query(requirement.domains[0], requirement)
    diag_params = policy.for_sub_query(diagnostic.domains[0], diagnostic)

    assert req_params.lexical_weight > req_params.dense_weight
    assert diag_params.dense_weight > diag_params.lexical_weight


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #

def test_namespaces_are_isolated(knowledge_base: KnowledgeBase) -> None:
    """A domain agent must not be able to reach another domain's documents."""
    for domain in ("technical", "business", "compliance"):
        hits = knowledge_base.search("approval policy deployment", domain,
                                     RetrievalParams(top_k=5))
        assert all(hit.document.domain == domain for hit in hits)


def test_hybrid_retrieval_beats_dense_only_on_rare_tokens(
    knowledge_base: KnowledgeBase
) -> None:
    """BM25 is in the mix precisely for exact identifiers like p99."""
    lexical_off = knowledge_base.search(
        "p99 latency regression", "technical",
        RetrievalParams(top_k=3, lexical_weight=0.0, min_score=0.0))
    hybrid = knowledge_base.search(
        "p99 latency regression", "technical", RetrievalParams(top_k=3, min_score=0.0))
    assert "TECH-API-004" in {hit.doc_id for hit in hybrid}
    assert hybrid[0].lexical_score > 0.0
    assert lexical_off  # dense-only still returns something, just less precisely


def test_search_below_threshold_returns_nothing(knowledge_base: KnowledgeBase) -> None:
    hits = knowledge_base.search("xylophone marmalade quarterly elephant", "technical",
                                 RetrievalParams(top_k=5))
    assert hits == []


def test_superseded_documents_are_excluded_by_default(
    knowledge_base: KnowledgeBase
) -> None:
    """Answers quote current policy; history is available only on request."""
    default = knowledge_base.search("data processing workflow approval cost", "business",
                                    RetrievalParams(top_k=8))
    assert "BIZ-APPR-002" not in {hit.doc_id for hit in default}

    with_history = knowledge_base.search(
        "data processing workflow approval cost", "business",
        RetrievalParams(top_k=8), include_superseded=True)
    assert "BIZ-APPR-002" in {hit.doc_id for hit in with_history}


def test_chunk_offsets_address_real_document_text(
    knowledge_base: KnowledgeBase
) -> None:
    """Chunk spans must index the parent document exactly."""
    for document in knowledge_base.documents():
        for chunk in knowledge_base.store(document.domain).chunks:
            if chunk.doc_id != document.doc_id:
                continue
            span = document.text[chunk.char_start:chunk.char_end]
            assert chunk.text.split()[0] in span


def test_duplicate_ingestion_is_idempotent() -> None:
    """Re-delivering a known document must not duplicate its passages."""
    corpus = load_corpus()
    kb = KnowledgeBase(corpus)
    before = kb.stats()["passages"]
    kb.ingest(corpus[0], rebuild=True)
    assert kb.stats()["passages"] == before


def test_ingesting_an_unknown_domain_raises(knowledge_base: KnowledgeBase) -> None:
    bad = Document(doc_id="X-1", domain="legal", title="t", text="some text")
    with pytest.raises(ValueError, match="unknown domain"):
        knowledge_base.ingest(bad)


def test_ingesting_empty_text_raises(knowledge_base: KnowledgeBase) -> None:
    bad = Document(doc_id="X-2", domain="technical", title="t", text="   ")
    with pytest.raises(ValueError, match="empty text"):
        knowledge_base.ingest(bad)


# --------------------------------------------------------------------------- #
# Agent communication
# --------------------------------------------------------------------------- #

def test_transcript_records_the_full_exchange(orchestrator: RAGOrchestrator) -> None:
    answer = orchestrator.answer(
        "What's the process for deploying a new microservice and what compliance "
        "checks are needed?")
    messages = orchestrator.bus.trace(answer.trace_id)

    kinds = {m.type for m in messages}
    assert MessageType.REQUEST in kinds
    assert MessageType.RESULT in kinds
    # Sequence numbers give a total order even though agents run concurrently.
    assert [m.sequence for m in messages] == sorted(m.sequence for m in messages)


def test_context_is_shared_between_agents(fresh_orchestrator: RAGOrchestrator) -> None:
    """The supporting agent must specialise its search using phase-1 findings.

    This is the test that distinguishes genuine coordination from three
    independent searches sharing an output format.
    """
    query = ("What's the process for deploying a new microservice and what "
             "compliance checks are needed?")
    plan = fresh_orchestrator.plan(query)
    fresh_orchestrator.answer(query, plan=plan)

    facts = fresh_orchestrator.blackboard.facts(plan.trace_id)
    authors = {f.author for f in facts}
    assert "technical" in authors, "the primary agent published no context"
    assert fresh_orchestrator.blackboard.best(plan.trace_id, "environment") == "production"

    broadcasts = [m for m in fresh_orchestrator.bus.trace(plan.trace_id)
                  if m.type is MessageType.PARTIAL]
    assert broadcasts, "no partial findings were broadcast"


def test_context_sharing_changes_the_issued_query(
    fresh_orchestrator: RAGOrchestrator
) -> None:
    """Shared context must reach the supporting agent's actual search string."""
    query = ("What's the process for deploying a new microservice and what "
             "compliance checks are needed?")
    plan = fresh_orchestrator.plan(query)
    primary = plan.primary
    supporting = plan.supporting[0]

    fresh_orchestrator.agents[primary.domain].handle(primary, trace_id=plan.trace_id)
    contribution = fresh_orchestrator.agents[supporting.domain].handle(
        supporting, trace_id=plan.trace_id)

    assert contribution.expansion_terms, "supporting agent expanded nothing"
    assert contribution.issued_query != contribution.sub_query


def test_primary_agent_searches_the_users_words_unexpanded(
    fresh_orchestrator: RAGOrchestrator
) -> None:
    """Phase 1 must not be diluted by terms re-derived from the query."""
    query = ("What's the process for deploying a new microservice and what "
             "compliance checks are needed?")
    plan = fresh_orchestrator.plan(query)
    contribution = fresh_orchestrator.agents[plan.primary.domain].handle(
        plan.primary, trace_id=plan.trace_id)
    assert contribution.expansion_terms == ()
    assert contribution.issued_query == plan.primary.text


# --------------------------------------------------------------------------- #
# Conflict resolution
# --------------------------------------------------------------------------- #

def _claims_for(kb: KnowledgeBase, query: str, domains: Sequence[str],
                include_superseded: bool = True) -> List:
    extractor = ClaimExtractor()
    claims = []
    for domain in domains:
        for hit in kb.search(query, domain, RetrievalParams(top_k=5),
                             include_superseded=include_superseded):
            claims.extend(extractor.extract(hit, query))
    return claims


def test_numeric_conflict_prefers_policy_over_runbook(
    knowledge_base: KnowledgeBase
) -> None:
    """A compliance policy must beat a technical runbook on approval counts."""
    claims = _claims_for(
        knowledge_base,
        "What approvals are needed to deploy a microservice that touches personal data?",
        ("technical", "compliance"))
    conflicts = ConflictResolver().detect(claims)

    numeric = [c for c in conflicts if c.kind == "numeric"]
    assert numeric, "the planted reviewer-count conflict was not detected"
    conflict = numeric[0]
    assert conflict.winner.doc_id == "COMP-CHG-002"
    assert conflict.loser.doc_id == "TECH-DEPLOY-001"
    assert conflict.decisive
    assert "policy outranks runbook" in conflict.explanation


def test_polarity_conflict_detected(knowledge_base: KnowledgeBase) -> None:
    """A prohibition must be recognised as contradicting a permission."""
    claims = _claims_for(
        knowledge_base,
        "Can I enable verbose request tracing in production to debug latency?",
        ("technical", "compliance"))
    conflicts = ConflictResolver().detect(claims)

    polarity = [c for c in conflicts if c.kind == "polarity"]
    assert polarity, "the planted verbose-tracing conflict was not detected"
    assert polarity[0].winner.domain == "compliance"


def test_polarity_conflict_is_reachable_end_to_end(
    orchestrator: RAGOrchestrator
) -> None:
    """The prohibition must surface through the full pipeline, not just the resolver.

    ``test_polarity_conflict_detected`` feeds the resolver directly with both
    domains forced, which proves the detector works but *not* that a user can
    reach it. This asserts the whole path: a permission question must route to
    compliance, retrieve the prohibition, and report the contradiction.

    Routing is the fragile link. "Can I enable verbose request tracing in
    production to debug latency?" routes to technical alone -- three strong
    technical cues (latency, debug, tracing) against the weak "can i" signal --
    and so is answered with the runbook that explains how, never surfacing the
    policy that forbids it. Explicit permission phrasing clears that bar. See the
    routing limitation noted in README and data/synthetic/README.md.
    """
    answer = orchestrator.answer(
        "Am I allowed to enable verbose request tracing in production?")

    assert "compliance" in answer.domains, (
        f"permission question did not reach compliance; routed to {answer.domains}")
    polarity = [c for c in answer.conflicts if c.kind == "polarity"]
    assert polarity, (
        "the verbose-tracing prohibition did not surface as a conflict end to end; "
        f"cited: {sorted({c.doc_id for c in answer.citations})}")
    assert polarity[0].winner.doc_id == "COMP-SEC-006"
    assert "prohibited" in answer.text or "overruled" in answer.text


def test_supersession_conflict_is_decisive(knowledge_base: KnowledgeBase) -> None:
    """A declared version bump resolves without relying on heuristic margins."""
    claims = _claims_for(knowledge_base,
                         "What approval is needed for a new data processing workflow?",
                         ("business", "compliance"))
    conflicts = ConflictResolver().detect(claims)

    supersession = [c for c in conflicts if c.kind == "supersession"]
    assert supersession, "the superseded approval procedure was not flagged"
    conflict = supersession[0]
    assert conflict.winner.doc_id == "BIZ-APPR-012"
    assert conflict.loser.doc_id == "BIZ-APPR-002"
    assert conflict.decisive


def test_unrelated_quantities_are_not_a_conflict(
    knowledge_base: KnowledgeBase
) -> None:
    """Two durations sharing only a generic verb must not be reported."""
    resolver = ConflictResolver()
    claims = _claims_for(knowledge_base, "retention periods", ("technical", "compliance"))
    for conflict in resolver.detect(claims):
        if conflict.kind != "numeric":
            continue
        # Any reported numeric conflict must name a substantive shared subject,
        # not merely "retained".
        assert not conflict.subject.startswith("retain ("), conflict.describe()


def test_overruled_claim_is_reported_not_deleted(
    orchestrator: RAGOrchestrator
) -> None:
    """A reader following the losing guidance must be told it was overruled."""
    answer = orchestrator.answer(
        "What approvals are needed to deploy a microservice that touches personal data?")
    assert answer.conflicts
    conflict = answer.conflicts[0]
    assert conflict.loser.text[:40] in answer.text
    assert "overruled" in answer.text


def test_clean_query_reports_no_conflicts(orchestrator: RAGOrchestrator) -> None:
    """Conflict detection must not invent disagreement where there is none."""
    answer = orchestrator.answer("What is the canary window for a deployment?")
    assert answer.conflicts == ()


# --------------------------------------------------------------------------- #
# Citation tracking
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("query,_expected", ASSIGNMENT_SCENARIOS)
def test_every_citation_verifies_against_its_source(
    orchestrator: RAGOrchestrator, query: str, _expected: Set[str]
) -> None:
    """Citation spans must re-read as the text they claim to quote."""
    answer = orchestrator.answer(query)
    audit = orchestrator.verify(answer)
    assert audit
    failures = [record for record in audit if not record["ok"]]
    assert not failures, f"unverifiable citations: {failures}"


def test_every_citation_is_referenced_in_the_answer(
    orchestrator: RAGOrchestrator
) -> None:
    """The source list must not carry entries the reader never sees quoted.

    Regression guard: citations were previously minted over every retrieved claim
    rather than over the selected ones, producing source lists with 15 entries for
    8 displayed findings.
    """
    for query, _expected in ASSIGNMENT_SCENARIOS:
        answer = orchestrator.answer(query)
        orphans = [c.marker for c in answer.citations if c.marker not in answer.text]
        assert not orphans, f"unreferenced citations {orphans} for {query!r}"


def test_findings_are_ranked_by_topicality_not_authority(
    orchestrator: RAGOrchestrator
) -> None:
    """Within a domain, findings appear in descending topicality order.

    Regression guard: ranking by blended confidence let a recent high-authority
    document outrank an on-topic older one, so a question about deployment
    approvals was answered with API gateway retry guidance.
    """
    answer = orchestrator.answer(
        "What approvals are needed to deploy a microservice that touches personal data?")
    for domain in answer.domains:
        scores = [c.topicality for c in answer.claims if c.domain == domain]
        assert scores == sorted(scores, reverse=True), f"{domain} findings misordered"


def test_diagnostic_query_surfaces_procedural_guidance(
    orchestrator: RAGOrchestrator
) -> None:
    """A "how do I" question must surface the procedure, not just the rules.

    Regression guard: salience rewarded only obligation markers, so for the
    troubleshooting scenario the corpus's own latency triage procedure scored
    lowest of all candidates and was dropped in favour of a data-store rule.
    """
    answer = orchestrator.answer(
        "How do I troubleshoot API performance issues while following our "
        "security policies?")
    cited = {c.doc_id for c in answer.citations}
    assert "TECH-API-004" in cited, (
        "the API performance troubleshooting guide was not used for a "
        f"troubleshooting question; cited instead: {sorted(cited)}")


def test_topicality_ignores_authority_and_recency(
    knowledge_base: KnowledgeBase
) -> None:
    """Topicality must depend only on relevance and salience.

    Asserted directly on the data model, because this separation is the whole
    point: authority and recency belong to conflict adjudication, not to deciding
    whether a sentence answers the question.
    """
    extractor = ClaimExtractor()
    query = "What is our log retention period?"
    hits = knowledge_base.search(query, "compliance", RetrievalParams(top_k=3))
    claims = [c for hit in hits for c in extractor.extract(hit, query)]
    assert claims
    for claim in claims:
        expected = round(0.60 * claim.components["relevance"]
                         + 0.40 * claim.components["salience"], 4)
        assert claim.topicality == pytest.approx(expected, abs=1e-4)


def test_citation_verification_catches_a_corrupted_span(
    orchestrator: RAGOrchestrator
) -> None:
    """The verifier must fail a citation whose offsets were tampered with.

    Without this, 'all citations verified' would only prove the verifier agrees
    with itself.
    """
    answer = orchestrator.answer("What is our log retention period?")
    assert answer.citations
    broken = replace(answer.citations[0], char_start=0, char_end=5)
    answer.citations = (broken,) + answer.citations[1:]

    audit = orchestrator.verify(answer)
    assert any(not record["ok"] for record in audit)


# --------------------------------------------------------------------------- #
# Knowledge updates
# --------------------------------------------------------------------------- #

def test_ingesting_a_new_policy_retires_the_one_it_supersedes(
    fresh_orchestrator: RAGOrchestrator, update_document: Document
) -> None:
    record = fresh_orchestrator.ingest(update_document)
    assert record["retired"] == "COMP-CHG-002"

    superseded = fresh_orchestrator.knowledge_base.document("COMP-CHG-002")
    assert superseded is not None, "the retired document must remain for audit"
    assert superseded.status == "superseded"


def test_knowledge_update_changes_the_answer(
    fresh_orchestrator: RAGOrchestrator, update_document: Document
) -> None:
    """The same question must yield different guidance after a policy change."""
    query = ("What approvals are needed to deploy a microservice that touches "
             "personal data?")
    before = fresh_orchestrator.answer(query)
    assert "COMP-CHG-002" in {c.doc_id for c in before.citations}

    fresh_orchestrator.ingest(update_document)
    after = fresh_orchestrator.answer(query)

    cited_after = {c.doc_id for c in after.citations}
    assert "COMP-CHG-009" in cited_after, "the new policy was not used"
    assert "COMP-CHG-002" not in cited_after, "the retired policy is still being quoted"
    assert before.text != after.text


# --------------------------------------------------------------------------- #
# Error handling and graceful degradation
# --------------------------------------------------------------------------- #

class _BrokenAgent(DomainAgent):
    """A domain agent whose retrieval always fails."""

    def _retrieve(self, query, params):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated vector store outage")


def test_failing_agent_degrades_rather_than_aborts() -> None:
    """One crashed agent must cost its domain, not the whole answer."""
    kb = KnowledgeBase(load_corpus())
    orchestrator = RAGOrchestrator(kb)
    orchestrator.agents["compliance"] = _BrokenAgent(
        "compliance", kb, bus=orchestrator.bus, blackboard=orchestrator.blackboard)

    answer = orchestrator.answer(
        "What's the process for deploying a new microservice and what compliance "
        "checks are needed?")

    assert not answer.abstained, "a single agent failure must not suppress the answer"
    assert answer.degraded
    assert answer.citations
    assert any("compliance" in note and "failed" in note for note in answer.notes)
    assert {claim.domain for claim in answer.claims} == {"technical"}


def test_degradation_lowers_reported_confidence() -> None:
    """A partial answer must not be presented with full confidence."""
    query = ("What's the process for deploying a new microservice and what "
             "compliance checks are needed?")
    healthy = RAGOrchestrator(KnowledgeBase(load_corpus())).answer(query)

    kb = KnowledgeBase(load_corpus())
    broken = RAGOrchestrator(kb)
    broken.agents["compliance"] = _BrokenAgent("compliance", kb, bus=broken.bus,
                                               blackboard=broken.blackboard)
    degraded = broken.answer(query)

    assert degraded.confidence < healthy.confidence


def test_failure_is_visible_in_the_transcript() -> None:
    kb = KnowledgeBase(load_corpus())
    orchestrator = RAGOrchestrator(kb)
    orchestrator.agents["business"] = _BrokenAgent(
        "business", kb, bus=orchestrator.bus, blackboard=orchestrator.blackboard)

    answer = orchestrator.answer(
        "What business approvals are required for implementing a new data "
        "processing workflow?")
    kinds = {m.type for m in orchestrator.bus.trace(answer.trace_id)}
    assert MessageType.FAILURE in kinds


def test_very_long_query_is_handled(orchestrator: RAGOrchestrator) -> None:
    query = ("What is the process for deploying a new microservice " * 40).strip() + "?"
    answer = orchestrator.answer(query)
    assert answer.text


def test_non_ascii_query_is_handled(orchestrator: RAGOrchestrator) -> None:
    answer = orchestrator.answer("Quelles approbations sont requises — déploiement?")
    assert answer.text  # may abstain; must not raise


# --------------------------------------------------------------------------- #
# Learning mechanism
# --------------------------------------------------------------------------- #

def test_feedback_moves_routing_weights(fresh_orchestrator: RAGOrchestrator) -> None:
    query = "What is our log retention period?"
    answer = fresh_orchestrator.answer(query, expected_domains=["compliance", "technical"])
    before = dict(fresh_orchestrator.classifier.routing_weights)

    fresh_orchestrator.apply_feedback(answer.trace_id, rating=0.1,
                                      expected_domains=["compliance", "technical"])
    after = fresh_orchestrator.classifier.routing_weights

    assert after != before
    # Technical was expected but not routed to, so its weight must rise.
    assert after["technical"] > before["technical"]


def test_positive_feedback_raises_agent_trust(
    fresh_orchestrator: RAGOrchestrator
) -> None:
    answer = fresh_orchestrator.answer("What is our log retention period?")
    before = fresh_orchestrator.agents["compliance"].trust_weight
    fresh_orchestrator.apply_feedback(answer.trace_id, rating=1.0)
    assert fresh_orchestrator.agents["compliance"].trust_weight > before


def test_feedback_replay_improves_routing_accuracy() -> None:
    """Replaying labelled feedback must not make routing worse."""
    orchestrator = RAGOrchestrator(KnowledgeBase(load_corpus()))
    labelled = [
        ("What is the budget approval threshold for vendor spend?", ["business"]),
        ("How do I tune a database connection pool?", ["technical"]),
        ("What is our log retention period?", ["compliance"]),
        ("What business approvals are required for a new data processing workflow?",
         ["business", "compliance"]),
    ]
    history = orchestrator.replay_feedback(labelled, rounds=3)
    assert len(history) == 3
    assert history[-1]["accuracy"] >= history[0]["accuracy"]


def test_weights_stay_bounded_under_repeated_feedback(
    fresh_orchestrator: RAGOrchestrator
) -> None:
    """No amount of feedback may switch a domain off or let one dominate."""
    answer = fresh_orchestrator.answer("What is our log retention period?",
                                       expected_domains=["business"])
    for _ in range(80):
        fresh_orchestrator.apply_feedback(answer.trace_id, rating=0.0,
                                          expected_domains=["business"])
    for weight in fresh_orchestrator.classifier.routing_weights.values():
        assert 0.6 <= weight <= 1.5
    for agent in fresh_orchestrator.agents.values():
        assert 0.7 <= agent.trust_weight <= 1.25


def test_feedback_for_unknown_trace_raises(orchestrator: RAGOrchestrator) -> None:
    with pytest.raises(KeyError):
        orchestrator.apply_feedback("does-not-exist", rating=1.0)


# --------------------------------------------------------------------------- #
# Performance monitoring
# --------------------------------------------------------------------------- #

def test_metrics_capture_per_stage_and_per_agent_detail() -> None:
    orchestrator = RAGOrchestrator(KnowledgeBase(load_corpus()))
    for query, expected in ASSIGNMENT_SCENARIOS:
        orchestrator.answer(query, expected_domains=sorted(expected))

    summary = orchestrator.metrics.summary()
    assert summary["queries"] == 3
    assert summary["answered"] == 3
    assert summary["latency_ms"]["mean"] > 0
    assert {"classify", "retrieve", "resolve_conflicts", "synthesize"} <= set(
        summary["stage_ms_mean"])

    routing = summary["routing"]
    assert routing["labelled_queries"] == 3
    assert routing["exact_set_accuracy"] == 1.0
    assert routing["f1"] == pytest.approx(1.0)

    per_domain = summary["per_domain"]
    assert {"technical", "business", "compliance"} >= set(per_domain)
    assert orchestrator.metrics.report().startswith("Performance report")


def test_abstention_is_recorded_separately() -> None:
    orchestrator = RAGOrchestrator(KnowledgeBase(load_corpus()))
    orchestrator.answer("How do I bake a sourdough loaf at home?")
    summary = orchestrator.metrics.summary()
    assert summary["abstained"] == 1
    assert summary["answer_rate"] == 0.0


def test_empty_registry_reports_cleanly() -> None:
    orchestrator = RAGOrchestrator(KnowledgeBase(load_corpus()))
    assert orchestrator.metrics.summary() == {"queries": 0}
    assert orchestrator.metrics.report() == "No queries recorded."
