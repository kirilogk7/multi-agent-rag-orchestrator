# The Self-Service Paradox

*A design for one automation platform that serves business users and power
users without becoming two platforms.*

## The core idea

Most low-code platforms ship an escape hatch to code, and it is almost always
one-way: the moment anyone uses it, that workflow leaves the business user's
reach for good. That is what two platforms wearing one logo looks like in
production, and it is the specific failure this design exists to avoid.

It happens because the premise is wrong. The brief frames this as a dial
between simplicity and power; it isn't one. Business users and power users
aren't asking for different *amounts* of the same thing — they're describing
the same workflow at different levels of detail. "Approve the invoice" and the
typed call that performs it are one step at two zoom levels, not two products.

So the real question isn't "how simple vs. how powerful" — it's **"can several
notations sit over one shared document?"** If yes, there's one platform. If no,
there are two.

The design here keeps that escape hatch open in both directions:

```
  canvas       code editor      wizard        API / SDK
(drag, drop)   (text, diff)  (guided form)  (power users,
     |              |              |          automation)
     |              |              |              |
     +--------------+------ typed patches --------+
                          |
                          v
                +----------------------+
                |     canonical IR     |   one workflow,
                |   (nodes + edges)    |   one document
                +----------------------+
                          |
                          v
                   execution engine
```

Canvas, code editor, wizard, and a plain REST API are four *views* of one
document, not four products. None of them can see state the others can't,
and none of them saves by overwriting the whole file — more on why that
matters in Section 3.

## 1. Architecture design

**One canonical IR; every surface is a projection over it.** A workflow is a
single document — typed nodes and edges, plus a UI-annotation layer the
canvas owns and the code editor never touches. The canvas renders it as
boxes, the code editor renders it as text, a wizard renders a filtered
subset of it as a form. None of them is the "real" version; the document is.

**Interface connections.** Every surface — including the API — talks to the
IR through the same patch endpoint; there is no separate "canvas service"
and "code service" with their own state to keep in sync. This is also the
answer to the brief's "direct API access" requirement for power users: the
API isn't a bolt-on for integrations, it's a first-class client of the same
document model the canvas uses, so a power user can create or edit a
workflow by script, CI pipeline, or CLI, and see the result on the canvas
immediately — because it's the same file, reached through the same patch
contract, not a parallel one.

**Control flow is declarative; custom logic is encapsulated, never inlined.**
Arbitrary code cannot be drawn as a flowchart, so this design doesn't pretend
otherwise. Branching, loops, and parallelism are always declarative — which
means always drawable. Anything that needs real code lives inside a typed
*block*: a node with named inputs, named outputs, an owner, and a test
badge. A business user can't read what's inside it, but they can always see
its shape, who owns it, and whether it's tested. That's the difference
between "representable" and "expandable," and it's what makes the canvas
trustworthy around code it can't render — it never needs to.

**Why this is one platform and not two with a shared login:** there's one
document schema, one execution engine, and four renderers. A power user's
change is visible on the canvas the moment they save, because it's the same
file. There's no sync step, no export/import, no "convert to code" button
that severs the connection.

**This is also what makes progressive disclosure structural rather than a UI
convention.** Because every rung reads and writes the same document through
the same patch contract, "disclosing more power" never means switching to a
different system with its own save format — it means the *same* document
accepting a more detailed patch than before. Section 2 walks through what a
user actually experiences climbing that ladder.

## 2. User experience strategy

**Escalation happens per node, not per user or per workflow.** A business
user doesn't "graduate" to power-user mode — they hit a wall on *one step* of
one workflow and escalate just that step, while the rest stays exactly as
simple as it was:

```
Six rungs, climbed per NODE, not per user or per workflow:

  1 template  2 wizard  3 canvas  4 expr  5 code  6 GitOps

One real workflow, mixing three of them at once:

  +----------+     +-------------+     +----------------+
  |  Invoice |---->|  Calc tax   |---->|  Needs review? |
  | (rung 1) |     |  (rung 5)   |     |   (rung 3)     |
  +----------+     +-------------+     +----------------+
   from a            engineer's           business user
   template          typed code           dragged this in

Descent is always available: a business user can open "Calc tax"
and see typed ports, an owner, and a test badge -- without reading
any code.
```

**Power users supply business users — that's the actual relationship.**
"Making it simpler" and "making it more powerful" sound like they trade off
because the brief treats the two audiences as adversaries. They're not —
they're a supply chain. An engineer's one-off script, published as a block,
becomes every business user's drag-and-drop node:

```
          business user hits a limit, asks for help
                          |
                          v
          power user writes code / a subgraph
                          |
                          v
                  "publish as block"
                          |
                          v
         +-------------------------------+
         |   typed ports, owner, tests   |   <- block registry
         +-------------------------------+
                          |
                          v
   appears as ONE drag-and-drop node, in the catalog,
       with a form, for every business user after

Each loop: the catalog gains a capability, the next business user
needs no help, and the power user's work is reused, not repeated.
```

**Onboarding follows the same ladder.** Everyone starts at rung 1 — a
template for the workflow they actually want ("approve an invoice," not "a
blank canvas"). Wizards cover the common parameterized cases. The canvas is
for free composition once someone outgrows templates. Code is never the
entry point; it's where one *node* ends up after a business user asks for
something the catalog doesn't have yet.

**Interface design: one screen, three panels, always visible together.**
The rung a node is on is never hidden behind a mode switch — a business
user sees the escalated node sitting in their canvas, not a separate
"advanced" screen:

```
+--------------------------------------------------------------+
| Invoice approval   * business-critical  [Test][Publish]      |
+-----------+----------------------------------+---------------+
| CATALOG   |  CANVAS                          | INSPECTOR     |
|           |                                  |               |
| [search]  |   +---------+                    | node: Calc tax|
|           |   | Invoice |                    |               |
| Suggested |   +---------+                    | effects:      |
|  Tax      |        |                         |   writes:fin. |
|  Approval |        v                         |               |
|  Payment  |   +-----------+                  | > view source |
|           |   | Calc tax  |<ok>              | > owner, tests|
| My team   |   +-----------+                  |               |
|  ...      |        |                         | ^ rung 5      |
+-----------+----------------------------------+---------------+
| Effects: writes:finance   cost: EUR 0.004/run                |
+--------------------------------------------------------------+
```

Catalog on the left (templates and published blocks, searchable); canvas in
the middle (the workflow itself); inspector on the right, which is where
escalation actually happens — selecting a node shows its rung, its declared
effects, and a one-click path to the next rung up or down. A business user
never has to know "rung 5" is code to work around a node that's on it; they
see effects, an owner, and a test badge, which is all the inspector promises
at any rung.

## 3. Technical implementation

**Key components — six pieces, each doing one job:**

| Component | Job |
|---|---|
| IR store | holds the canonical workflow document, versioned |
| Projection renderers | canvas, code editor, wizard, API — one per surface |
| Patch validator | accepts/rejects typed ops before they touch the IR |
| Block registry | typed ports, declared effects, owner, certification tier |
| Durable scheduler | runs the DAG, retries, backoff, timers, human waits |
| Run log | event-sourced; every step replayable for debugging |

Everything below follows from how these six talk to each other — never by
reaching into one another's internals, only through the IR and the patches
in front of it.

**The first data model: the IR is a keyed document, not a list.** Nodes and
edges are maps keyed by stable id, not arrays — so two people adding a node
each append at a different key instead of the same list position, which is
what makes concurrent edits mergeable instead of conflicting:

```yaml
schema: workflow/v1
id: wf_invoice_approval
version: 7               # publishing increments; immutable once run
owner: team:finance-ops
tier: business-critical  # drives governance, see Section 4

nodes:                   # keyed by id -- concurrent edits merge
  n_tax:
    block: calculate-tax@2.1   # pinned major version
    inputs: { amount: { ref: n_invoice.amount } }
  n_review:
    kind: decision
    condition: { expr: "n_invoice.amount > 10000" }

edges:
  e1: { from: n_invoice, to: n_tax }
  e2: { from: n_tax, to: n_review }

ui:                       # canvas-only; code editor never writes this
  nodes: { n_tax: { x: 120, y: 240 } }
```

**Surfaces write typed patches, never the whole document.** `addNode`,
`setInput`, `setSource`, `setUi` — each is a small, named operation. This is
the actual mechanism behind "the canvas can't corrupt code" in Section 1: a
drag on the canvas emits a `setUi` patch, which by construction cannot touch
a node's `source` field, so it cannot clobber a code author's comment even
by accident. A whole-document save couldn't make that guarantee.

**Blocks are the one extension mechanism, used by everyone.** A block
declares typed ports and its effects (`pure`, `network: [...]`,
`writes: [finance]`, ...). An engineer builds one with a code editor; the
platform team builds one the same way for a built-in action. There's no
separate "plugin SDK" — publishing a block *is* the extension model, for
internal authors and the platform itself alike.

**Most blocks are existing microservices, not new code.** The scenario
starts from a service estate that already exists, and the fastest way to
fill the catalog is to wrap it: a block is a typed envelope over an
endpoint — ports mapped to request and response fields, effects declared
by the owning team, credentials handled by the platform rather than the
block. Importing an OpenAPI spec drafts one block per operation, landing
as `uncertified` and owned by whoever imported it, exactly like any other
block. This matters more for adoption than for architecture: a catalog
that starts empty gets used by nobody, and "publish as block" only
becomes a flywheel once there is something there to compose against.

A block definition is the second data model, and the only other one a
user ever authors:

```yaml
schema: block/v1
id: calculate-tax
version: 2.1             # workflows pin the major; 2.x stays compatible
owner: team:finance-eng  # a team, never a person -- see Section 4
tier: verified

ports:
  in:  { amount: money, region: string }
  out: { tax: money, breakdown: json }

effects:                 # what governance reads -- see Section 4
  network: [tax-api.internal]
  writes:  []            # pure apart from the call
  cost:    EUR 0.004/run

impl:
  kind: service          # or `inline` for code written in the editor
  endpoint: POST https://tax-api.internal/v2/calculate
  identity: workflow     # the workflow's principal, not the caller's
```

**Execution is boring on purpose.** A durable scheduler runs the DAG,
retries failed nodes with backoff, and logs every step so a run can be
replayed. None of this is novel — it's standard workflow-engine practice —
and that's deliberate: the interesting problem here is the shared document
and the escalation ladder, not inventing a new execution model.

**What this has to hold, and what happens when it doesn't.** Numbers
first, because "it scales" is not a design. The shape this targets is
~10k workflows, ~50k runs/day with a ~10x month-end peak, and tens of
concurrent editors — not thousands, because authoring is a human activity
and the edit path is never the hot path. The two loads want opposite
things, so they are scaled separately:

| | Edit path | Run path |
|---|---|---|
| Load | tens of concurrent authors | 50k runs/day, bursty |
| Scales by | one store, read replicas | partition queue by workflow id |
| Bottleneck | patch validation (cheap) | block execution (the real work) |

The IR store stays a single consistent document store: workflow documents
are kilobytes and edits are rare, so there is nothing to gain by sharding
it and a merge property to lose. The scheduler is the part that scales
out, and it does so the ordinary way — partition by workflow id, add
workers.

**Partial failure is the normal case.** A run is a sequence of calls into
other teams' services, so some of them will fail halfway. Three rules,
all deliberately boring:

- Every execution is keyed by `(run_id, node_id, attempt)` and deduped on
  that key, so a retry after a timeout cannot pay an invoice twice.
- A block that is not safely retryable declares `at_most_once`; the
  scheduler then stops the run and waits for a human rather than guess.
- There is no automatic rollback, because money that has moved cannot be
  un-moved. Compensation is a step the author draws on the canvas, not a
  platform feature that pretends otherwise.

**What a business user sees when it breaks** is the run log drawn on the
canvas they already know: the failed node marked, its inputs and the
error beside it, every upstream output still inspectable, and one button
to resume from that node once the cause is fixed. No stack trace, and no
re-running the whole workflow to find out whether it worked this time.

## 4. Long-term maintainability

**Review is keyed on blast radius and declared effects, never on how a
workflow was built.** This is where the simplicity-versus-power framing
does the most damage: "drag-and-drop" and "needs no review" are not the
same axis.

| Built by | Low blast radius (e.g. renames a file) | High blast radius (e.g. moves money) |
|---|---|---|
| Dragging boxes | No review needed | Review required |
| Writing code | No review needed | Review required |

A workflow that pays invoices needs sign-off whether an engineer wrote it or
a business user dragged it together from certified blocks. A workflow that
renames files in someone's own folder needs none, regardless of how it was
built. Governance reads the *effects* a workflow declares, not its
authorship method — which also means a business user is never blocked from
shipping something safe just because the platform can't tell how it was
made.

**Effects say what a workflow may do; identity says who it does it as.**
The table above is only enforceable if the platform knows what a run is
allowed to touch, so a workflow executes as its own principal —
`workflow:wf_invoice_approval`, owned by `team:finance-ops` — and never
as whoever clicked Run. A business user triggering a payment run does not
borrow an engineer's access, and an engineer testing it does not borrow
theirs. Blocks never hold credentials at all: a block declares
`network: [tax-api.internal]`, and the platform injects a scoped,
short-lived token for that host at call time. Dragging a block whose
declared effects exceed the workflow's granted scope fails in the patch
validator, at edit time, on the canvas, naming the missing permission —
rather than at 3am on the first real run.

**Adding capabilities is the flywheel from Section 2, not a separate
process.** Every "publish as block" event adds a capability to the platform
without a release cycle — the catalog's growth rate is a measurable side
effect of ordinary usage, not something a roadmap has to schedule.

**Community governance: certification is a ladder, not a gate.** A new
block starts `uncertified`, visible only within its author's team — nothing
stops anyone from using it there. Promotion to `community` (visible
platform-wide) requires a declared owner and a passing test; `verified`
adds a platform-team review of its declared effects; `core` is reserved for
blocks the platform team owns outright. This only governs who else can
*find* a block in the shared catalog, never who can build one — so
publishing something half-finished to your own team never breaks anyone
else's workflow.

**Schema changes are additive, forever.** New fields are optional; an
existing field's meaning never changes within a major version; unknown
fields are preserved on read and write, so an old client can't silently
delete data a new client wrote. A `workflow/v1 -> v2` change ships with a
migration function the platform owns and runs automatically — not homework
assigned to every workflow's author, most of whom will have moved teams by
the time it matters.

**Blocks outlive the people who wrote them, so their lifecycle is the
platform's job, not the author's.** Pinning `calculate-tax@2.1` is what
stops an upstream change from silently altering a finance workflow — but
a pin nobody ever moves is just a slow leak. Three cases the platform
owns because the workflow's author cannot:

- **A new major ships.** The old one keeps working. Workflows still on it
  are listed on the block's page and their owners get one notification,
  not a deadline — nothing breaks on the platform's schedule.
- **A security fix ships.** The platform moves the pin within the major
  version without asking, because `2.1 -> 2.2` is compatible by
  definition, and tells the owners afterwards.
- **The owning team dissolves.** Ownership is on a team and never a
  person, for exactly this reason. An orphaned block keeps running
  everywhere it is already used and drops to `uncertified`, so it stops
  being discoverable for new work until someone adopts it.

The asymmetry is deliberate: a block is easy to publish and hard to
delete. A stale block in the catalog costs a bad search result; a deleted
one costs a broken payment run.

**The one metric that tells you if this actually worked: the descent
rate** — the share of power-user-authored blocks that business users go on
to edit from the canvas. Adoption numbers and satisfaction scores can both
look healthy while the two populations quietly diverge into two platforms
that happen to share a login page. The descent rate can't fake that. If it's
zero, the one-way door exists in practice no matter what the architecture
diagram claims.

## What this doesn't solve

- **Code that can't be represented as a block still can't be dragged in** —
  this design makes code *encapsulated*, not drawable. That's intentional,
  not a gap, but it means a canvas-only user can never fully read a
  complex block's logic, only its shape.
- **Real-time co-editing of the same node** (two people in the same code
  block at once) needs CRDT-level merge below the patch layer. Not designed
  here — patches solve concurrent edits to *different* parts of a workflow,
  not simultaneous edits to the same part.
- **Migrating an org that already runs two separate platforms** is a
  separate, harder project — this design is for building one platform from
  the start, not merging two existing ones.
- **Block quality still depends on someone reviewing effect declarations
  honestly.** A block that mis-declares its own effects is a self-reported
  lie the platform can't catch without sandboxing every execution, which
  this design doesn't attempt.
