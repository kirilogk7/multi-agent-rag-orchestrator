"""Agent communication protocol: typed messages, a bus, and a blackboard.

Two mechanisms, because multi-agent coordination needs both:

* **Message passing** (``MessageBus``) is point-to-point and ordered. It carries
  requests and results, and every message is appended to an immutable
  transcript keyed by ``trace_id`` -- so any answer can be replayed and audited
  after the fact.
* **Shared context** (``Blackboard``) is broadcast and additive. An agent
  writes a fact it discovered while working; agents that run later read it and
  specialise their own retrieval. This is how the technical agent noticing
  "target = production, data = PII" causes the compliance agent to search for
  change-control rules it would otherwise have missed.

Both are thread-safe, because the orchestrator fans agents out concurrently.
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

# Well-known addresses. Agents are addressed by name so the bus stays agnostic
# about who is listening.
ORCHESTRATOR = "orchestrator"
CLASSIFIER = "query_classifier"
BROADCAST = "*"


class MessageType(str, Enum):
    """The verbs of the protocol.

    ``PARTIAL`` exists so an agent can publish an intermediate finding before
    it has finished, which is what makes cross-agent context sharing possible
    within a single query rather than only between queries.
    """

    REQUEST = "request"       # orchestrator -> agent: please answer this sub-query
    PARTIAL = "partial"       # agent -> all: an intermediate finding
    RESULT = "result"         # agent -> orchestrator: final contribution
    FAILURE = "failure"       # agent -> orchestrator: I could not answer, and why
    ABSTAIN = "abstain"       # agent -> orchestrator: nothing confident to say
    FEEDBACK = "feedback"     # orchestrator -> agent: outcome signal for learning


@dataclass(frozen=True)
class Message:
    """One immutable envelope on the bus.

    Attributes:
        message_id: Unique identifier for this envelope.
        trace_id: Groups every message belonging to one top-level query.
        parent_id: The message this one responds to, giving a causal chain.
        sender: Address of the sending agent.
        recipient: Address of the intended recipient, or ``BROADCAST``.
        type: Protocol verb.
        payload: Type-specific body. Kept as a plain dict so the transcript
            stays serialisable for audit export.
        timestamp: Unix timestamp at construction.
        sequence: Monotonic counter giving a total order across the bus, which
            wall-clock timestamps cannot guarantee under concurrency.
    """

    message_id: str
    trace_id: str
    sender: str
    recipient: str
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    def summary(self) -> str:
        """One-line rendering for transcripts and debug output."""
        subject = (
            self.payload.get("sub_query")
            or self.payload.get("reason")
            or self.payload.get("key")
            or ""
        )
        arrow = f"{self.sender} -> {self.recipient}"
        return f"#{self.sequence:03d} {self.type.value:<8} {arrow:<42} {str(subject)[:70]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


class MessageBus:
    """In-memory, thread-safe message bus with an append-only transcript.

    The transcript is the point. A production deployment would swap the queues
    for a real broker, but the audit property -- every inter-agent exchange for
    a given query is recoverable in causal order -- is what the design is
    actually providing, and it is preserved either way.

    Example:
        >>> bus = MessageBus()
        >>> _ = bus.send("orchestrator", "technical", MessageType.REQUEST,
        ...              {"sub_query": "how do I deploy?"}, trace_id="t1")
        >>> [m.type.value for m in bus.inbox("technical")]
        ['request']
        >>> len(bus.trace("t1"))
        1
    """

    def __init__(self) -> None:
        self._queues: Dict[str, Deque[Message]] = defaultdict(deque)
        self._transcript: List[Message] = []
        self._counter = itertools.count(1)
        self._lock = threading.RLock()

    # -- publishing ------------------------------------------------------- #

    def send(
        self,
        sender: str,
        recipient: str,
        type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
        *,
        trace_id: str,
        parent_id: Optional[str] = None,
    ) -> Message:
        """Enqueue a message for ``recipient`` and record it in the transcript.

        A ``BROADCAST`` recipient is delivered to every address that has a queue,
        excluding the sender, so an agent never receives its own broadcast.
        """
        with self._lock:
            message = Message(
                message_id=uuid.uuid4().hex[:12],
                trace_id=trace_id,
                parent_id=parent_id,
                sender=sender,
                recipient=recipient,
                type=type,
                payload=dict(payload or {}),
                sequence=next(self._counter),
            )
            self._transcript.append(message)
            if recipient == BROADCAST:
                for address, queue in self._queues.items():
                    if address != sender:
                        queue.append(message)
            else:
                self._queues[recipient].append(message)
            return message

    def register(self, address: str) -> None:
        """Create a queue for ``address`` so it can receive broadcasts.

        Agents must register before the first broadcast, otherwise they simply
        were not listening yet -- the same semantics as a real pub/sub topic.
        """
        with self._lock:
            self._queues.setdefault(address, deque())

    # -- consuming -------------------------------------------------------- #

    def inbox(self, address: str, *, drain: bool = False) -> List[Message]:
        """Return messages waiting for ``address``.

        Args:
            address: Recipient address.
            drain: Remove the returned messages from the queue.
        """
        with self._lock:
            queue = self._queues[address]
            messages = list(queue)
            if drain:
                queue.clear()
            return messages

    def trace(self, trace_id: str) -> List[Message]:
        """Every message for one query, in bus order."""
        with self._lock:
            return [m for m in self._transcript if m.trace_id == trace_id]

    def transcript(self, *, limit: Optional[int] = None) -> List[Message]:
        """The full ordered transcript, optionally only the most recent entries."""
        with self._lock:
            return list(self._transcript[-limit:] if limit else self._transcript)

    def causal_chain(self, message_id: str) -> List[Message]:
        """Walk ``parent_id`` links from a message back to its root cause."""
        with self._lock:
            by_id = {m.message_id: m for m in self._transcript}
            chain: List[Message] = []
            current = by_id.get(message_id)
            seen: set = set()
            while current and current.message_id not in seen:
                seen.add(current.message_id)
                chain.append(current)
                current = by_id.get(current.parent_id) if current.parent_id else None
            return list(reversed(chain))

    def render_trace(self, trace_id: str) -> str:
        """Human-readable transcript for one query."""
        messages = self.trace(trace_id)
        if not messages:
            return f"(no messages for trace {trace_id})"
        lines = [f"Transcript for trace {trace_id} ({len(messages)} messages)"]
        lines.extend("  " + m.summary() for m in messages)
        return "\n".join(lines)

    def clear(self) -> None:
        """Drop all queues and transcript history."""
        with self._lock:
            self._queues.clear()
            self._transcript.clear()


@dataclass(frozen=True)
class Fact:
    """A piece of shared context, with provenance.

    Provenance is not optional: a fact asserted by the compliance agent carries
    different weight than one guessed by the classifier, and downstream readers
    need to know which they are looking at.
    """

    key: str
    value: Any
    author: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class Blackboard:
    """Per-trace shared context that agents write to and read from.

    Scoped by ``trace_id`` so concurrent queries cannot contaminate each other.
    Writes are additive -- multiple agents may assert the same key -- because
    discarding a competing assertion here would hide a disagreement that
    conflict resolution is better placed to adjudicate.

    Example:
        >>> bb = Blackboard()
        >>> _ = bb.write("t1", "environment", "production", author="technical")
        >>> bb.best("t1", "environment")
        'production'
        >>> _ = bb.write("t1", "environment", "staging", author="business", confidence=0.2)
        >>> bb.best("t1", "environment")   # highest-confidence assertion wins
        'production'
    """

    def __init__(self) -> None:
        self._facts: Dict[str, List[Fact]] = defaultdict(list)
        self._lock = threading.RLock()

    def write(
        self,
        trace_id: str,
        key: str,
        value: Any,
        *,
        author: str,
        confidence: float = 1.0,
    ) -> Fact:
        """Assert ``key = value`` for one trace."""
        with self._lock:
            fact = Fact(key=key, value=value, author=author, confidence=confidence)
            self._facts[trace_id].append(fact)
            return fact

    def write_many(self, trace_id: str, items: Dict[str, Any], *, author: str,
                   confidence: float = 1.0) -> List[Fact]:
        """Assert several keys at once."""
        return [self.write(trace_id, k, v, author=author, confidence=confidence)
                for k, v in items.items()]

    def read(self, trace_id: str, key: str) -> List[Fact]:
        """All assertions of ``key``, newest last."""
        with self._lock:
            return [f for f in self._facts[trace_id] if f.key == key]

    def best(self, trace_id: str, key: str, default: Any = None) -> Any:
        """The highest-confidence value for ``key``, or ``default`` if unasserted.

        Ties break towards the later assertion, on the assumption that an agent
        writing after reading the board had more information available.
        """
        facts = self.read(trace_id, key)
        if not facts:
            return default
        return max(enumerate(facts), key=lambda pair: (pair[1].confidence, pair[0]))[1].value

    def snapshot(self, trace_id: str) -> Dict[str, Any]:
        """Flattened ``key -> best value`` view, for prompt construction."""
        with self._lock:
            keys = {f.key for f in self._facts[trace_id]}
        return {k: self.best(trace_id, k) for k in sorted(keys)}

    def facts(self, trace_id: str) -> List[Fact]:
        """Every fact on the board for one trace, in write order."""
        with self._lock:
            return list(self._facts[trace_id])

    def authors(self, trace_id: str) -> List[str]:
        """Distinct agents that contributed context to this trace."""
        with self._lock:
            seen: List[str] = []
            for fact in self._facts[trace_id]:
                if fact.author not in seen:
                    seen.append(fact.author)
            return seen

    def clear(self, trace_id: Optional[str] = None) -> None:
        """Drop one trace's context, or everything."""
        with self._lock:
            if trace_id is None:
                self._facts.clear()
            else:
                self._facts.pop(trace_id, None)
