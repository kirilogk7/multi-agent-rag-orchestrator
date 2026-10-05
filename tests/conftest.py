"""Shared fixtures.

The knowledge base and classifier are session-scoped because fitting them does
real work (chunking, SVD, centroids) that no test mutates. Tests that *do* mutate
state -- ingestion, feedback learning -- build their own instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from rag_system.orchestrator import RAGOrchestrator
from rag_system.query_classifier import QueryClassifier
from rag_system.utils import Document, default_data_dir, load_corpus, load_jsonl
from rag_system.vector_store import KnowledgeBase

#: The three scenarios named in the assignment, with their expected routing.
ASSIGNMENT_SCENARIOS = [
    (
        "What's the process for deploying a new microservice and what compliance "
        "checks are needed?",
        {"technical", "compliance"},
    ),
    (
        "How do I troubleshoot API performance issues while following our security "
        "policies?",
        {"technical", "compliance"},
    ),
    (
        "What business approvals are required for implementing a new data processing "
        "workflow?",
        {"business", "compliance"},
    ),
]


@pytest.fixture(scope="session")
def corpus() -> List[Document]:
    return load_corpus()


@pytest.fixture(scope="session")
def knowledge_base(corpus: List[Document]) -> KnowledgeBase:
    return KnowledgeBase(corpus)


@pytest.fixture(scope="session")
def classifier(knowledge_base: KnowledgeBase) -> QueryClassifier:
    return QueryClassifier().fit(knowledge_base)


@pytest.fixture(scope="session")
def orchestrator(knowledge_base: KnowledgeBase,
                 classifier: QueryClassifier) -> RAGOrchestrator:
    """Read-only orchestrator shared across tests that do not mutate state."""
    return RAGOrchestrator(knowledge_base, classifier=classifier)


@pytest.fixture
def fresh_orchestrator() -> RAGOrchestrator:
    """Isolated orchestrator for tests that ingest documents or apply feedback."""
    return RAGOrchestrator(KnowledgeBase(load_corpus()))


@pytest.fixture(scope="session")
def update_document() -> Document:
    """The progressive-delivery policy used by the knowledge-update tests."""
    path = default_data_dir() / "knowledge_update.jsonl"
    return Document.from_dict(next(iter(load_jsonl(path))))
