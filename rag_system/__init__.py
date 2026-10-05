"""Multi-agent RAG orchestration over three mock knowledge domains.

Quick start::

    from rag_system import build_default_orchestrator

    orchestrator = build_default_orchestrator()
    answer = orchestrator.answer(
        "What's the process for deploying a new microservice "
        "and what compliance checks are needed?"
    )
    print(answer.text)
    print(orchestrator.explain(answer.trace_id))   # inter-agent transcript

The package runs fully offline on the bundled synthetic corpus: no API key, no
model download, deterministic output.

Module map:

* ``utils``            -- data models, text processing, scoring helpers
* ``protocol``         -- agent message bus and shared blackboard
* ``vector_store``     -- embedding backends, BM25, hybrid retrieval, knowledge base
* ``query_classifier`` -- domain routing, intent, complexity, decomposition, retrieval policy
* ``domain_agents``    -- per-domain RAG agents and cross-agent context sharing
* ``synthesis``        -- claim extraction, conflict resolution, citations, synthesisers
* ``monitoring``       -- per-query and per-agent metrics
* ``orchestrator``     -- planning, two-phase execution, feedback learning
"""

from .domain_agents import AgentContribution, ContextExtractor, DomainAgent, build_agents
from .monitoring import AgentRecord, MetricsRegistry, QueryRecord
from .orchestrator import ExecutionPlan, RAGOrchestrator, build_default_orchestrator
from .protocol import Blackboard, Fact, Message, MessageBus, MessageType
from .query_classifier import (
    Classification,
    DomainScore,
    QueryClassifier,
    QueryIntent,
    RetrievalPolicy,
    SubQuery,
)
from .synthesis import (
    Answer,
    Citation,
    Claim,
    ClaimExtractor,
    ClaudeSynthesizer,
    Conflict,
    ConflictResolver,
    ExtractiveSynthesizer,
    Synthesizer,
    verify_citations,
)
from .utils import DOMAINS, Chunk, Document, RetrievedChunk, load_corpus
from .vector_store import (
    BM25Index,
    EmbeddingBackend,
    KnowledgeBase,
    RetrievalParams,
    SentenceTransformerBackend,
    TfidfSvdBackend,
    VectorStore,
)

__version__ = "1.0.0"

__all__ = [
    "AgentContribution", "AgentRecord", "Answer", "BM25Index", "Blackboard",
    "Chunk", "Citation", "Claim", "ClaimExtractor", "Classification",
    "ClaudeSynthesizer", "Conflict", "ConflictResolver", "ContextExtractor",
    "DOMAINS", "Document", "DomainAgent", "DomainScore", "EmbeddingBackend",
    "ExecutionPlan", "ExtractiveSynthesizer", "Fact", "KnowledgeBase",
    "Message", "MessageBus", "MessageType", "MetricsRegistry", "QueryClassifier",
    "QueryIntent", "QueryRecord", "RAGOrchestrator", "RetrievalParams",
    "RetrievalPolicy", "RetrievedChunk", "SentenceTransformerBackend", "SubQuery",
    "Synthesizer", "TfidfSvdBackend", "VectorStore", "build_agents",
    "build_default_orchestrator", "load_corpus", "verify_citations",
    "__version__",
]
