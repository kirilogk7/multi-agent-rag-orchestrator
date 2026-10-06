# The Self-Service Paradox

*A design for one automation platform that serves business users and power
users without becoming two platforms.*

## The core idea

The brief frames this as a dial between simplicity and power. It isn't one.
Business users and power users aren't asking for different *amounts* of the
same thing — they're describing the same workflows at different levels of
detail. So the real question isn't "how simple vs. how powerful" — it's
**"can several notations sit over one shared document?"** If yes, there's one
platform. If no, there are two platforms wearing one logo, which is what most
low-code tools actually ship: the escape hatch to code is almost always
one-way, so the moment someone uses it, the workflow leaves the business
user's reach for good.

The design here keeps that door open in both directions:

```
   canvas             code editor         wizard
 (drag, drop)         (text, diff)     (guided form)
       \                  |                  /
        \                 |                 /
         +---------- typed patches --------+
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

Canvas, code editor, and wizard are three *views* of one document, not three
products. None of them can see state the others can't, and none of them
saves by overwriting the whole file — more on why that matters in Section 3.

## 1. Architecture design

**One canonical IR; every surface is a projection over it.** A workflow is a
single document — typed nodes and edges, plus a UI-annotation layer the
canvas owns and the code editor never touches. The canvas renders it as
boxes, the code editor renders it as text, a wizard renders a filtered
subset of it as a form. None of them is the "real" version; the document is.

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
document schema, one execution engine, and three renderers. A power user's
change is visible on the canvas the moment they save, because it's the same
file. There's no sync step, no export/import, no "convert to code" button
that severs the connection.

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

## 3. Technical implementation

**The IR is a keyed document, not a list.** Nodes and edges are maps keyed
by stable id, not arrays — so two people adding a node each append at a
different key instead of the same list position, which is what makes
concurrent edits mergeable instead of conflicting:

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

**Execution is boring on purpose.** A durable scheduler runs the DAG,
retries failed nodes with backoff, and logs every step so a run can be
replayed. None of this is novel — it's standard workflow-engine practice —
and that's deliberate: the interesting problem here is the shared document
and the escalation ladder, not inventing a new execution model.

## 4. Long-term maintainability

**Review is keyed on blast radius and declared effects, never on how a
workflow was built.** This is the point the brief's framing misses most
directly: "drag-and-drop" and "needs no review" are not the same axis.

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

**Schema changes are additive, forever.** New fields are optional; an
existing field's meaning never changes within a major version; unknown
fields are preserved on read and write, so an old client can't silently
delete data a new client wrote. A `workflow/v1 -> v2` change ships with a
migration function the platform owns and runs automatically — not homework
assigned to every workflow's author, most of whom will have moved teams by
the time it matters.

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
