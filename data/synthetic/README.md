# The synthetic corpus

49 documents across three mock knowledge domains, plus one fixture used to
demonstrate live knowledge updates.

## Provenance

**Every document here was written from scratch for this exercise.** Nothing is
copied, scraped or adapted from real internal documentation, and there is no real
company data in this repository.

There are also **no companies in the corpus at all** — no real ones and no
fictional placeholder. Documents refer to "the platform team", "the owning team",
"the Change Advisory Board" and, once, "the organisation". The only proper nouns
are three open-source tools (Terraform, OpenTelemetry, Kubernetes), standard
acronyms (API, TLS, JSON, DPIA, RACI, KPI), and "European" in a data-residency
clause.

The prose is **modeled on generic professional conventions**, which are concepts
rather than anyone's text:

| Domain | Conventions drawn on |
|---|---|
| `technical` | SRE golden signals, canary deployment and automated rollback, connection-pool and p99 latency troubleshooting, IaC and CI practice, incident severity ladders |
| `business` | Delegation-of-authority matrices, RACI, business-case and TCO practice, showback cost allocation, intake and prioritisation funnels |
| `compliance` | GDPR concepts (lawful basis, purpose limitation, DPIA, 72-hour breach notification), SOC 2-style control testing and evidence, ITIL-style change management with a Change Advisory Board, segregation of duties |

Everything specific is **invented**: all figures (90 days, 30 days, the
€10k/€25k/€250k approval thresholds, reviewer counts), all dates, all version
numbers, all document identifiers, and the authority assigned to each document.
None of it should be read as advice or as a description of any real organisation's
policy.

The assignment this corpus was built for calls for "synthetic data and mock
knowledge domains", so inventing it is the requirement rather than a shortcut.

## Files

| File | Contents |
|---|---|
| `technical.jsonl` | 17 documents, 35 passages |
| `business.jsonl` | 16 documents, 32 passages |
| `compliance.jsonl` | 16 documents, 34 passages |
| `knowledge_update.jsonl` | 1 document, not loaded by default — see [Knowledge updates](#knowledge-updates) |

Totals: **49 documents → 101 retrievable passages**, ~4,400 words, averaging ~90
words per document. Effective dates span 2024-06-01 to 2026-01-15. Documents are
chunked at load time into sentence-aligned passages with character offsets back
into the source text, which is what makes citations verifiable.

## Record schema

One JSON object per line. Every field except `tags` is consumed by the system —
this metadata is load-bearing, not decoration.

```json
{
  "doc_id": "COMP-CHG-002",
  "domain": "compliance",
  "title": "Production Change Control Standard for Regulated Systems",
  "section": "Approval requirements",
  "authority": "policy",
  "version": "2.0",
  "effective_date": "2025-11-02",
  "supersedes": null,
  "status": "active",
  "tags": ["change-control", "approval", "deployment", "personal-data", "cab"],
  "text": "Production changes that touch personal data require ..."
}
```

| Field | Consumed by |
|---|---|
| `doc_id` | Citation labels; supersession links. Prefixes `TECH-`/`BIZ-`/`COMP-` are invented |
| `domain` | Selects the vector-store namespace, and sets domain precedence in conflict resolution |
| `title`, `section` | Citation rendering; title/section overlap feeds claim salience |
| `authority` | Conflict resolution — the largest single term (weight 0.34) |
| `version` | Citation labels; supersession detection |
| `effective_date` | Recency decay in both claim confidence and conflict resolution |
| `supersedes` | Retires the named document on ingestion; a superseded document can never win a conflict |
| `status` | `superseded` documents are excluded from search by default but retained for audit |
| `tags` | Optional metadata filtering at retrieval time |
| `text` | Chunked, embedded, BM25-indexed, and split into claims |

### Authority tiers

Ordered, and the ordering is the point: a policy *defines* the rule, a standard
says how it is implemented, a runbook *describes current practice*. When practice
and policy disagree, practice is the thing that is out of date.

| Tier | Weight | Count |
|---|---|---|
| `policy` | 1.00 | 13 |
| `standard` | 0.85 | 23 |
| `runbook` | 0.65 | 9 |
| `wiki` | 0.40 | 4 |

## Planted conflicts

Four contradictions were designed **before** any detection code was written. The
contradiction *types* were chosen first — numeric, polarity, supersession — and
then document pairs were written to exhibit each one. That ordering matters: it
means the detectors were built against an independent specification rather than
reverse-engineered to pass on data that happened to be there.

| # | Dispute | Sources | Detector | Resolution |
|---|---|---|---|---|
| 1 | Approving reviewers for a production deploy: **1 vs 2** | `TECH-DEPLOY-001` (runbook, 2025-03-14) vs `COMP-CHG-002` (policy, 2025-11-02) | numeric | Policy wins — higher authority, compliance governs, newer |
| 2 | Log retention: **90 days vs purge within 30 days** | `TECH-OBS-003` (runbook) vs `COMP-DATA-004` (policy) | numeric | Policy wins |
| 3 | Verbose production tracing: **do it vs never do it** | `TECH-API-005` (runbook) vs `COMP-SEC-006` (policy) | polarity | Policy wins |
| 4 | Data-workflow approval: **team lead alone vs governance council** | `BIZ-APPR-002` v1.0 (2024-06-01) vs `BIZ-APPR-012` v2.0 (2026-01-15) | supersession | v2.0 wins — declared version bump |

Conflict 4 is the only `status: superseded` document in the corpus, which is why
it is excluded from search by default. It is still reachable with
`include_superseded=True`, and the system uses that deliberately so it can explain
why current guidance differs from what someone remembers.

### Triggering each conflict

A conflict only fires when **both** halves are retrieved in the same query, which
in turn requires the query to route to both domains. Verified phrasings:

```bash
# conflict 1 — numeric, reviewer counts
python -m rag_system "What approvals are needed to deploy a microservice that touches personal data?"
# conflict 2 — numeric, log retention
python -m rag_system "how long are logs containing personal data retained before deletion"
# conflict 3 — polarity, verbose tracing (note the phrasing; see the limitation below)
python -m rag_system "Am I allowed to enable verbose request tracing in production?"
```

**Conflict 4 does not surface through a plain query, by design.** Superseded
documents are excluded from retrieval by default, because an answer should quote
current policy — and `BIZ-APPR-002` was already retired at load time, since
`BIZ-APPR-012` declares `supersedes: BIZ-APPR-002`. So the supersession detector
never receives it on the default path.

It is reachable two ways:

```python
# (a) explicitly, with history included — what tests/test_scenarios.py does
kb.search("data processing workflow approval cost", "business",
          RetrievalParams(top_k=8), include_superseded=True)

# (b) through the knowledge-update flow, where retirement happens live
orchestrator.ingest(update)   # retires COMP-CHG-002; the next answer re-resolves
```

Worth stating as a genuine design gap rather than a quirk: the supersession
detector is implemented and tested, but **nothing in the default pipeline feeds it
superseded documents**, so in normal use it is inert. The capability it was built
for — telling a user "you may be thinking of the old rule, here is what changed" —
needs the orchestrator to run a second narrow pass with `include_superseded=True`
purely for conflict detection, without letting retired text be quoted as current
guidance. That pass is not implemented.

**A known routing limitation, worth being explicit about.** Conflict 3 is the
fragile one, and the reason is instructive. These two phrasings behave
differently:

| Query | Routes to | Conflict 3 fires |
|---|---|---|
| "**Am I allowed to** enable verbose request tracing in production?" | compliance, technical | yes |
| "**Is** verbose request tracing **permitted** in production?" | technical, compliance | yes |
| "**Can I** enable verbose request tracing in production **to debug latency**?" | technical only | **no** |

The third routes to technical alone because three strong technical cues (`latency`
2.2, `debug` 1.8, `tracing` 1.8) overwhelm the single weak permission signal
`can i` (1.1), and the selection threshold is relative to the leading domain's
score — so an unusually strong primary domain raises the bar for secondary ones.

The consequence is exactly the failure conflict resolution exists to prevent: the
user is told *how* to enable verbose tracing and never told that the security
policy forbids it.

This was left as a documented limitation rather than fixed, deliberately. The
available fixes were either raising the `can i` weight — which over-routes genuine
technical questions — or capping the relative threshold, which on measurement
fixed this single query at exactly one threshold value (0.42) while improving
nothing on an independent set of twelve harder queries, and broke a different case
at 0.40. That is a magic number tuned to one test, not an improvement. A real
system would learn per-cue weights from logged feedback instead of hand-tuning
them, which is the fix recorded in the architecture document's limitations.

`tests/test_scenarios.py::test_polarity_conflict_is_reachable_end_to_end` pins the
phrasings that do work, so the path cannot silently break.

## Knowledge updates

`knowledge_update.jsonl` is **not loaded by default**. It holds `COMP-CHG-009`
v1.0, a 2026-09-01 policy that declares `supersedes: COMP-CHG-002` and relaxes the
two-reviewer rule for services with automated rollback and no personal-data
access.

Ingesting it demonstrates that knowledge updates change answers rather than just
changing the index: the same question asked before and after returns different
guidance, stops citing the retired document, and re-resolves conflict 1. See
notebook §8 and `test_knowledge_update_changes_the_answer`.

```python
from rag_system import build_default_orchestrator, Document
from rag_system.utils import default_data_dir, load_jsonl

orchestrator = build_default_orchestrator()
update = Document.from_dict(next(iter(load_jsonl(
    default_data_dir() / "knowledge_update.jsonl"))))
orchestrator.ingest(update)        # retires COMP-CHG-002, refits the classifier
```

## Adding documents

Append a line to the relevant `<domain>.jsonl`. Only `doc_id`, `domain`, `title`
and `text` are required; the rest default to `section=""`, `authority="wiki"`,
`version="1.0"`, `effective_date="2025-01-01"`, `status="active"`.

Two things to know. Documents are loaded at startup and chunked automatically, so
a new file line needs no code change. And because the default embedding backend is
corpus-statistical (TF-IDF projected through a truncated SVD), adding a document
refits the namespace — intentionally, since a new document changes the IDF
landscape, and projecting into a stale latent space quietly degrades every
comparison against it.

If a new document is meant to contradict an existing one, check it is actually
detected — the detectors require a shared subject, not merely a shared unit or a
shared verb:

```bash
python -m rag_system "your query here" --explain
```
