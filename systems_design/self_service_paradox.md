# The Self-Service Paradox

**A design for one automation platform that serves business users and power users
without becoming two platforms.**

---

## Executive summary

**The paradox as stated is not the real one.** The brief frames simplicity against
power as if there were one dial the two audiences wanted in different positions. If
that were true, the honest answer would be "build two products". But business users
and power users are not asking for different amounts of the same thing — they are
describing the *same workflows at different levels of abstraction*. The answerable
question is therefore: can one system support several notations over a single shared
artifact?

**What actually breaks real platforms is the one-way door.** Almost every low-code
tool offers an escape hatch to code, and almost all of them make it irreversible —
Unreal's Blueprints compile to C++ and cannot be recovered; Power Automate and
Power Automate Desktop share a brand and little else. The moment that door shuts
behind a workflow, you have two platforms with a trapdoor between them, and every
symptom in the brief follows from that single mechanism: business users cannot
maintain anything a power user touched, power users must abandon the tooling to add
one expression, and the platform team ends up with two roadmaps.

**The design, in five moves:**

1. **One canonical IR.** A workflow is one text document. The canvas and the code
   editor are both *projections* over it; neither owns state the other cannot see.
   Surfaces emit typed patches rather than rewritten documents, which is why the
   canvas cannot corrupt a code author's comments even in principle.
2. **Declarative orchestration, encapsulated computation.** Arbitrary code cannot be
   drawn as a flowchart, so this design does not pretend otherwise. Control flow is
   *always* declarative and therefore always drawable; custom code lives inside
   typed blocks. An engineer's 200-line function appears on a business user's canvas
   as a node with typed ports, an owner and a test badge — always representable,
   just not always expandable. This is what makes bidirectionality achievable
   instead of a promise to write a decompiler.
3. **A six-rung ladder, escalated per node.** Template → wizard → canvas →
   expression → code block → GitOps. A workflow can sit at rung 2 and rung 4 at
   once. Descent is always available, enforced by invariant tests in CI rather than
   asserted in documentation.
4. **Power users supply business users.** "Publish as block" turns one engineer's
   custom logic into every business user's drag-and-drop node. The two audiences are
   a supply chain, not a conflict, and the platform's job is to shorten it.

5. **Governance keys on blast radius and declared effects, never on authoring
   method.** Dragging boxes to move money is still moving money. A workflow that
   renames files in one person's folder needs no review whether or not it contains
   code; one that pays invoices needs review whether or not it was built by
   dragging boxes.

**The measure of success is the descent rate** — business users continuing to edit,
on the canvas, workflows that contain power users' code. Adoption and satisfaction
can both look healthy while two platforms quietly diverge; the descent rate cannot.
If it is zero, the one-way door exists in practice whatever the architecture diagram
claims.

### Where to find each required area

| Required area | Section |
|---|---|
| Architecture design — system, interface connections, progressive disclosure | [§2](#2-architecture-design) |
| User experience strategy — transitions, interface design, onboarding | [§3](#3-user-experience-strategy) |
| Technical implementation — components, data models, extension mechanisms | [§4](#4-technical-implementation) |
| Long-term maintainability — evolution, adding capabilities, governance | [§5](#5-long-term-maintainability) |

Risks, the gaps this design does *not* solve, and a four-phase delivery sequence
follow in [§6](#6-risks-and-how-they-bite) and [§7](#7-delivery-sequence). §1 sets
out the reasoning behind the reframing above; a reader short of time can skip to §2.

---

## Contents

1. [The real problem](#1-the-real-problem)
2. [Architecture design](#2-architecture-design)
3. [User experience strategy](#3-user-experience-strategy)
4. [Technical implementation](#4-technical-implementation)
5. [Long-term maintainability](#5-long-term-maintainability)
6. [Risks and how they bite](#6-risks-and-how-they-bite)
7. [Delivery sequence](#7-delivery-sequence)
8. [Summary](#8-summary)

---

## 1. The real problem

### 1.1 The paradox as stated is not quite the paradox

The brief frames this as a tension between simplicity and power, as if there were
a single dial and the two audiences wanted it in different positions. If that were
true the problem would be genuinely unsolvable, and the honest answer would be
"build two products".

It isn't true. Business users and power users do not want different *amounts* of
the same thing — they want to operate at different **levels of abstraction** over
the same underlying system. A business user wiring an approval step and an
engineer writing a custom tax calculation are both describing a workflow. They
disagree about notation, not about substance.

Reframed, the question becomes answerable:

> Can one system support multiple notations over a single shared artifact, such
> that work done in any notation remains fully available in the others?

### 1.2 Why platforms fail at this: the one-way door

Nearly every low-code platform offers an escape hatch to code. Almost all of them
make it a **one-way door**:

```
   +-----------------+   "Convert to code"    +-----------------+
   |  Visual editor  |----------------------->|   Code editor   |
   |                 |                        |                 |
   |   drag & drop   |   X  no way back  X    |  full control   |
   +-----------------+                        +-----------------+
            v                                          v
     business users                               power users
      stranded here                              stranded here
```

Unreal Engine's Blueprints compile to C++ and cannot be recovered. Power Automate
and Power Automate Desktop share a brand and almost nothing else. Zapier's code
step is a black box inside an otherwise visual flow.

The moment that door closes behind a workflow, the platform *has* become two
platforms — with the additional indignity of a trapdoor between them. Every
consequence the brief worries about follows from this single mechanism:

- A business user cannot maintain anything a power user has touched, so power
  users become a permanent bottleneck on work they already finished.
- A power user who needs one custom expression must abandon the visual tooling
  entirely, losing templates, guardrails and the catalog.
- The two populations diverge into separate tool cultures, and the platform team
  ends up maintaining two roadmaps.

**So the primary architectural requirement is not "support both audiences". It is
"never close that door."** Everything else in this design is downstream of that
one constraint.

### 1.3 The precedent worth copying

The most successful progressive-disclosure tool ever built is the spreadsheet. The
same grid serves someone totalling a column and someone writing a pivot model, and
there is **no mode switch between them**. A novice clicks AutoSum; an analyst types
a formula into the same cell; both edit one document, and either can open the
other's file.

Three properties make that work, and this design adopts all three:

| Spreadsheet property | Why it matters | Adopted as |
|---|---|---|
| One artifact, many notations | Nobody converts a spreadsheet to "pro mode" | A single canonical workflow document (§2.2) |
| Escalation is **per cell**, not per file | One complex formula doesn't make the file a programming project | Per-node escalation (§3.2) |
| The formula bar is a *visible* next rung | Users discover depth by using the tool, not by reading docs | The escalation ladder (§2.4) |

It is just as important to copy where spreadsheets **failed**: no types, no tests,
no modularity, no review, no ownership. Those omissions produced a genuine
catastrophe literature — reconciliation errors in public finance, the Reinhart-Rogoff
coding error, repeated trading losses. A self-service automation platform that
executes real business processes has strictly more blast radius than a spreadsheet,
so the governance those tools lack is not optional here (§5).

### 1.4 Design principles

Everything below follows from five commitments:

1. **One artifact, many projections.** The visual editor and the code editor are
   both *views* over the same canonical document. Neither is primary; neither owns
   state the other cannot see.
2. **Escalation is local and reversible.** A user escalates one node, not the whole
   workflow, and can always descend again.
3. **Declarative orchestration, arbitrary computation.** Control flow is always
   declarative (therefore always renderable). Custom code is always encapsulated
   behind a typed interface (therefore always *representable*, even when its
   internals are not renderable). This is the structural trick that makes
   bidirectionality tractable rather than heroic — see §2.3.
4. **Power users are producers; business users are consumers.** The platform's job
   is to shorten the path from "an engineer wrote custom logic" to "it is a
   drag-and-drop node with a form". The two audiences are a supply chain, not a
   conflict (§5.2).
5. **Governance scales with blast radius, not with authoring method.** A workflow
   that renames files in someone's own folder needs no review regardless of whether
   it contains code. A workflow that pays invoices needs review regardless of
   whether it was built by dragging boxes.

---

## 2. Architecture design

### 2.1 System overview

```
+--------------------------------------------------------------------------------------------------+
| SURFACES                                           (all equal citizens -- none is "the real one")|
|                                                                                                  |
|  +----------+ +--------+ +--------+ +------------+ +--------+ +-----------+                      |
|  | Template | | Wizard | | Canvas | | Expression | |  Code  | |   CLI /   |                      |
|  | gallery  | | /forms | | editor | |    bar     | | editor | | SDK / API |                      |
|  |  rung 0  | | rung 1 | | rung 2 | |   rung 3   | | rung 4 | |  rung 5   |                      |
|  +----------+ +--------+ +--------+ +------------+ +--------+ +-----------+                      |
+--------------------------------------------------------------------------------------------------+

         ------------------------------+-------------------------------
                                       | every surface reads and writes
                                       | the SAME document, losslessly
+--------------------------------------------------------------------------------------------------+
| PROJECTION LAYER                                          <- the critical component; see 2.2, 4.2|
|  - render : IR -> view model (canvas graph / form schema / formatted text)                       |
|  - apply  : view edit -> typed IR patch (never a re-serialisation of the whole)                  |
|  - preserve: layout, comments, formatting, unknown fields (round-trip safety)                    |
+--------------------------------------------------------------------------------------------------+
                                       v
+--------------------------------------------------------------------------------------------------+
| CANONICAL WORKFLOW IR                                               <- the single source of truth|
|  typed DAG . stable node IDs . deterministic serialisation . text-diffable                       |
+--------------------------------------------------------------------------------------------------+
         |--------------------------------|--------------------------------|
         v                                v                                v
 +------------------------------+ +------------------------------+ +------------------------------+
 | VALIDATION & COMPILE         | | BLOCK REGISTRY               | | POLICY ENGINE                |
 |- port type-check             | |- catalog + search            | |- policy-as-code (OPA-style)  |
 |- effect aggregation          | |- semver versions             | |- static eval, pre-run,       |
 |- cost estimate               | |- ownership                   | |  from declared effects       |
 |                              | |- certification               | |- blast-radius tiering        |
 |                              | |- usage stats                 | |- approval requirements       |
 +------------------------------+ +------------------------------+ +------------------------------+
                 | immutable, validated workflow version
                 v
+--------------------------------------------------------------------------------------------------+
| EXECUTION PLANE                                                                                  |
|  Trigger router -> Planner -> Durable scheduler -> Step workers                                  |
|                                         |- declarative evaluator                                 |
|                                         |- expression sandbox                                    |
|                                         |- code sandbox (isolate)                                |
|                                         |- connector egress proxy                                |
|  state store: event-sourced run log                                                              |
+--------------------------------------------------------------------------------------------------+
                 v
+--------------------------------------------------------------------------------------------------+
| OBSERVABILITY                                                                                    |
|  per-run trace . per-step input/output inspection . replay . cost attribution                    |
|  ONE run view for every rung -- a business user and an engineer debug the                        |
|  same artifact with the same tool                                                                |
+--------------------------------------------------------------------------------------------------+
```

The shape to notice: **the surfaces are a wide, shallow layer, and everything below
the projection layer is shared.** There is exactly one execution engine, one
observability stack, one policy engine, one catalog. A business user's wizard-built
workflow and an engineer's Git-managed workflow are the same kind of object in the
same database, run by the same scheduler, debugged in the same trace view.

That is what "single unified platform" has to mean to be worth anything. Two UIs
over one backend is not unification if the two UIs produce artifacts that cannot
be opened by the other.

### 2.2 The canonical IR, and why it must exist first

Every surface is a projection over one document. The document is a typed DAG with
these properties, each of which exists to solve a specific failure:

| Property | Failure it prevents |
|---|---|
| **Stable node IDs**, never positional refs | Renaming or reordering a node silently rewires edges; diffs become unreadable |
| **Deterministic serialisation** (canonical key order, fixed formatting) | Dragging a node on canvas produces a 400-line diff and code review becomes impossible |
| **Text-first format** (YAML/JSON with a published JSON Schema) | Workflows can't live in Git, so power users can't use their own tools |
| **Non-semantic data preserved as annotations** (layout, comments, folds) | The visual editor destroys the code author's comments; the code editor scrambles the layout |
| **Unknown fields preserved, not dropped** | An older client silently deletes data written by a newer one |
| **Explicit schema version** | No migration path; every change is breaking |

The last three are what the industry usually gets wrong, and they are not
cosmetic. A visual editor that reformats the document on save makes the Git
workflow unusable, which drives power users off the platform, which recreates the
two-platform problem by a different route.

### 2.3 The structural trick: declarative orchestration, encapsulated computation

Here is the hard part of bidirectionality, stated plainly: **arbitrary code cannot
be rendered as a flowchart.** Any design that promises a general code↔visual round
trip is promising something unachievable, and will deliver it as a buggy
approximation that corrupts people's work.

So this design does not promise that. It draws a line:

```
+--------------------------------------------------------------------------------+
| ORCHESTRATION -- always declarative, therefore always renderable               |
|                                                                                |
| control flow . data flow . branching . looping over collections .              |
| error handling . retries . timeouts . parallelism . scheduling .               |
| approvals . waits                                                              |
|                                                                                |
| Expressed ONLY in the IR. There is no code path that can express               |
| orchestration -- the SDK (rung 5) emits IR too. One notation for               |
| structure means the canvas can always draw the structure.                      |
+--------------------------------------------------------------------------------+
                                         | nodes reference blocks
                                         v
+--------------------------------------------------------------------------------+
| COMPUTATION -- arbitrary, encapsulated behind a typed interface                |
|                                                                                |
| +----------------------------------------------------------------------------+ |
| | Block: calculate-tax @2.1                                                  | |
| | in  : { amount: number, jurisdiction: string }   <- typed ports            | |
| | out : { tax: number, breakdown: TaxLine[] }                                | |
| | effects: [ pure ]                                <- declared               | |
| | +------------------------------------------------------------------------+ | |
| | | impl: typescript   <- opaque to the canvas, and that is fine           | | |
| | | export default function(input) { /* 200 lines */ }                     | | |
| | +------------------------------------------------------------------------+ | |
| |                                                                            | |
| +----------------------------------------------------------------------------+ |
+--------------------------------------------------------------------------------+
```

A power user's 200-line tax function appears on a business user's canvas as a node
with two input ports, two output ports, a code badge, an owner and a test status.
The business user can **see it, wire it, type-check against it, run it, and debug
it** — they simply cannot read its internals, which they did not want to anyway.

This is the move that makes the whole design tractable. Bidirectionality is
achieved **structurally**, by confining unrenderable content inside typed boxes,
rather than by attempting a general-purpose decompiler. Nothing is ever
unrepresentable; only some things are un-*expandable*.

Degradation is graceful rather than fatal: if a future block kind cannot be
rendered richly, it still renders as an opaque typed box. The canvas never shows a
dead end.

### 2.4 Progressive disclosure: a ladder, not a switch

Six rungs. Every rung is **reversible**, **composable with every other rung in the
same workflow**, and **applied per node rather than per workflow**.

```
                     capability -------------------------------------------->

 rung 5  GitOps / SDK / API         | workflow-as-code in your own repo,
          ########################  | CI, code review, the full platform API
            ^                       | descend: `platform pull` opens the same workflow on canvas
            v                       |
 rung 4  Code block                 | TypeScript/Python in a sandbox, typed
          ####################      | ports, own tests, publishable as a block
            ^                       | descend: delete the node; the rest of the workflow is untouched
            v                       |
 rung 3  Expression                 | "{{ order.total * 1.2 }}" in any field,
          #############             | autocomplete from the upstream schema
            ^                       | descend: replace with a literal value
            v                       |
 rung 2  Canvas composition         | drag blocks, draw edges, branch, loop
          ###########
            ^                       | descend: no-op; canvas is the home view
            v                       |
 rung 1  Configure a template       | guided form, validated inputs
          ######
            ^                       | "customise" reveals the canvas beneath -- the template was always a workflow
            v                       |
 rung 0  Run a template             | zero authoring
          ##
```

Three properties distinguish this from the usual "beginner/advanced mode" toggle:

**Local, not global.** A workflow with eleven drag-and-drop nodes and one code
node is at rung 2 *and* rung 4 simultaneously. It appears in the business user's
canvas as twelve boxes. There is no sense in which the workflow "is now a code
project".

**Reversible at every rung.** The descent arrows above are real operations, not
aspirations. Because the IR is the only state, `platform pull` → edit on canvas →
`platform push` is a supported loop, and so is the reverse.

**Templates are not a separate artifact type.** A template *is* a workflow, with
some inputs marked as parameters and the rest marked as locked. "Customise this
template" unlocks it into the canvas. This matters enormously: the usual design
makes templates a distinct, non-editable thing, which means the most common
on-ramp for business users is a dead end. Here the on-ramp *is* the platform.

### 2.5 Interface connections

How the surfaces actually relate — the contract each one has with the IR:

| Surface | Reads | Writes | Round-trip guarantee |
|---|---|---|---|
| Template gallery | workflow + `parameters` manifest | run config only | n/a (no authoring) |
| Wizard / forms | `parameters` manifest → form schema | typed patches to parameter values | Exact: the form is generated from the schema, so it cannot express an invalid document |
| Canvas | full IR + `ui:` annotations | structural patches (add/remove node, rewire, set input) | Exact: patches are typed operations on the graph, never a reserialisation |
| Expression bar | upstream port schemas (for autocomplete) | one expression string into one field | Exact: a leaf value |
| Code editor | one block's source + its port schema | block source + schema | Exact: source is stored verbatim, byte for byte |
| CLI / SDK / API | full IR as text | full IR as text | Exact: the IR *is* the wire format; no translation occurs |

The pattern: **surfaces emit typed patches, not documents.** A canvas drag emits
`{op: "move", node: "n7", x: 240, y: 130}` against the `ui:` annotations, not a
rewritten YAML file. This is what makes concurrent editing, undo, audit and small
diffs all fall out of one mechanism — and it is why the canvas cannot corrupt a
power user's comments even in principle, because it never writes the regions where
comments live.

---

## 3. User experience strategy

### 3.1 One worked example, three rungs, one document

Everything above is abstract until you see the same workflow from both ends. Here
is an invoice approval workflow — the kind of thing a finance analyst builds and an
engineer later extends.

**What the business user sees (rung 2, canvas):**

```
                    +------------------+  trigger: webhook
                    | Invoice received |
                    +------------------+
                              | invoice
                              v
                    +------------------+  <- block: calculate-tax@2.1
                    |  Calculate tax   |     <code>  owner: @tax-eng  [ok] 14 tests
                    +------------------+     [pure]                   <- effects badge
                              | { tax, breakdown }
                              v
                    +-------------------+  <- decision
                    | Over EUR 10,000 ? |
                    +-------------------+
                  yes |---------------------| no
                      v                     v
            +------------------+  +-------------------+  <- approval blocks
            |   CFO approval   |  | Director approval |
            +------------------+  +-------------------+
                      |                     |
                      -----------|-----------
                                 v
                       +------------------+  <- block: pay-invoice@4.0
                       | Schedule payment |     [network, writes:finance]
                       +------------------+     tier: business-critical   <- blast-radius warning
```

**What the power user sees (rung 5, the same document in Git):**

```yaml
schema: workflow/v1                      # explicit version -> migratable
id: wf_invoice_approval
name: Invoice approval
owner: team:finance-ops
tier: business-critical                  # drives governance, not authoring method

triggers:
  - id: t1
    kind: webhook
    schema: { $ref: "schemas/invoice.json" }

nodes:                                   # keyed by id: concurrent inserts merge
  n_tax:
    block: calculate-tax@2.1             # pinned major version
    inputs:
      amount:       { ref: t1.body.total }
      jurisdiction: { ref: t1.body.billing_country }

  n_threshold:
    kind: decision
    cases:
      - when: "{{ t1.body.total > 10000 }}"   # rung 3 expression, inline
        then: n_cfo
      - else: n_director

  n_cfo:
    block: approval@1.3
    inputs: { approver: { role: cfo }, sla_hours: 48 }

  n_director:
    block: approval@1.3
    inputs: { approver: { role: director }, sla_hours: 24 }

  n_pay:
    block: pay-invoice@4.0
    inputs:
      amount: { expr: "{{ t1.body.total + n_tax.out.tax }}" }
      vendor: { ref: t1.body.vendor_id }
    retry: { attempts: 3, backoff: exponential }
    timeout: 30s

edges:
  e1: { from: t1,          to: n_tax }
  e2: { from: n_tax,       to: n_threshold }
  e3: { from: n_cfo,       to: n_pay }
  e4: { from: n_director,  to: n_pay }

ui:                                      # non-semantic; canvas owns this region,
  nodes:                                 # code editor never rewrites it
    n_tax:       { x: 120, y: 240 }
    n_threshold: { x: 120, y: 360 }
  notes:
    n_pay: "Finance asked for exponential backoff after the Q3 incident."
```

**What the engineer escalating one node sees (rung 4):**

```typescript
// block: calculate-tax@2.2   -- opened from the canvas, canvas stays visible
import { defineBlock } from "@platform/sdk";

export default defineBlock({
  inputs:  { amount: "number", jurisdiction: "string" },
  outputs: { tax: "number", breakdown: "TaxLine[]" },
  effects: ["pure"],                      // declared -> policy engine can verify
  async run({ amount, jurisdiction }) {
    const rules = await loadRules(jurisdiction);
    return applyRules(amount, rules);     // 200 lines elsewhere
  },
});
```

Three notations. One document. The YAML and the canvas are the *same object* — the
engineer's `retry` block and the finance analyst's dragged edge coexist, and either
person can open the other's view without converting anything.

Note what the engineer did *not* have to do: they did not rewrite the orchestration
in code. The branching, the approvals and the retry policy stay declarative, which
is precisely why the analyst can still see them.

### 3.2 Transitions between capabilities

Escalation and descent are the most important interactions in the product, so they
get explicit mechanics rather than being left to a modal dialog.

**Escalation is always additive and always in place.**

| From → To | Trigger | What happens |
|---|---|---|
| Template → Canvas | "Customise" | Parameters unlock; the canvas opens on the workflow that was always underneath |
| Field → Expression | Type `=` or click `fx` | Field becomes an expression editor with autocomplete from the upstream port schema |
| Expression → Code block | "Extract to code block" when an expression exceeds a complexity threshold | Platform scaffolds a block with correct port types inferred from the expression's inputs and outputs |
| Canvas → GitOps | `platform pull wf_invoice_approval` | Writes the IR to disk; canvas keeps working on the same workflow |
| Subgraph → Block | Select nodes → "Publish as block" | The single most valuable transition in the product (§5.2) |

**Descent is always available, and this is enforced, not hoped for.**

```
escalate ------------------------------------------------->
+----------+      +--------+      +------------+      +------+
| Template |<---->| Canvas |<---->| Expression |<---->| Code |
+----------+      +--------+      +------------+      +------+
      <----------------------------------- descend

 Enforcement: a CI gate generates random valid IR, renders it through every
 projection, applies a random edit, re-parses, and asserts the result is the
 intended document and nothing else changed (4.2). A projection that cannot
 round-trip does not ship.
```

Two rules about transitions that are easy to get wrong:

- **Never show a one-way warning dialog.** "Converting to code cannot be undone"
  is a confession that the architecture failed. If a user sees that string, the
  design has been violated.
- **Escalation must not relocate the user.** Opening a code editor navigates to a
  side panel with the canvas still on screen, not to a different application view.
  Users who feel they have *left* the familiar tool do not come back.

### 3.3 Interface design

**One canvas, layered detail.** The same canvas serves everyone; what differs is
how much it reveals.

```
+------------------------------------------------------------------------------------------+
| Invoice approval            * business-critical   [Test] [Diff] [Publish]                |
+----------------+------------------------------------------+------------------------------+
| CATALOG        |  CANVAS                                  | INSPECTOR                    |
|                |                                          | (selected: n_pay)            |
| [search]       |    +---------+                           |                              |
|                |    | Invoice |                           | Amount                       |
| Suggested      |    +---------+                           | +--------------------------+ |
|  Tax           |         v                                | | fx {{ total +            | |
|  Approval      |    +----------+ <code> [ok]              | |     n_tax.tax }}         | |
|  Payment       |    | Calc tax |                          | +--------------------------+ |
|                |    +----------+                          |  ^ rung 3, inline            |
| My team        |          v                               |                              |
|  ...           |    +-----------+                         | Vendor                       |
|                |    | Over 10k? |                         | [ invoice.vendor v ]         |
| Cmd+K commands |    +-----------+                         |                              |
|                |                                          | > Retry & timeout            |
|                |                                          | > Error handling             |
|                |                                          |  ^ collapsed by default      |
|                |                                          |                              |
+------------------------------------------------------------------------------------------+
| [!] This workflow moves money. Publishing requires finance approval.                     |
|   Effects: network, writes:finance, reads:pii     Est. cost: EUR 0.004/run               |
+------------------------------------------------------------------------------------------+
```

Specific decisions and their reasons:

| Decision | Reason |
|---|---|
| Advanced settings (retry, timeout, error handling) collapsed, never hidden | Hidden features are undiscoverable; always-visible features overwhelm. Collapsed-with-a-label is the only option that serves both |
| Effects and cost shown in the status bar at all times | A business user needs to know a workflow touches PII *before* publishing, not in a post-incident review |
| `fx` affordance in every field | This is the formula bar. It is the single most important discoverability element in the product, because it is how users find out depth exists |
| Block cards show owner, test status, usage count, effects | A business user's decision to trust an engineer's block is a real decision and needs real evidence |
| Command palette (`⌘K`) alongside drag-and-drop | Power users navigate by keyboard; forcing them to drag is the same insult as forcing a business user to type |
| One run/trace view for all rungs | Debugging is where the two-platform problem reappears if you let it. Same trace, same step inspector, same replay, whoever built the workflow |

**No role-based UI gating.** The interface is not configured by "user type". A
business user who learns expressions is not blocked by a permission, and an
engineer is not forced through a wizard. Capability gating by job title is how you
build two products inside one binary and prevent anyone from ever growing. What
*is* gated is publishing to high-blast-radius tiers — by review, not by role (§5.3).

### 3.4 Onboarding pathways

Three doors, converging deliberately on the same artifact:

```
    "I have a task"            "I'll build it"           "I write code"
            v                         v                         v
  +------------------+    +-----------------------+    +-----------------+
  | Template gallery |    |     Blank canvas      |    |  platform init  |
  |  run in < 5 min  |    |  guided first node,   |    | scaffold, local |
  |                  |    | sample data preloaded |    |   test, push    |
  +------------------+    +-----------------------+    +-----------------+
            |                         |                         |
            |  first successful run   |                         |
            -----------------------------------------------------
                                      v
                           +---------------------+
                           |                     |
                           |  the same workflow  |
                           |   the same canvas   |
                           | the same trace view |
                           +---------------------+
                                      |
                                      |
                    |-----------------------------------|
                    v                                   v
+----------------------------------+        +-----------------------------+
|      growth happens by USE       |        |   growth happens by NEED    |
| fx hints, "others also used...", |        | hit a limit -> the platform |
|    template internals visible    |        |   names the next rung and   |
|           on customise           |        |        scaffolds it         |
+----------------------------------+        +-----------------------------+
```

The guiding metric is **time to first successful run**, targeted under ten minutes
for a templated path. Nothing else about onboarding matters if a user's first
session ends without something working.

Two deliberate mechanisms for growth:

**Growth by use.** The `fx` affordance, visible template internals, and
"people who used this block also used…" expose the next rung passively. A user
discovers expressions because they saw `fx` in a field they were already editing —
not because they read documentation.

**Growth by need, with a scaffold.** When a user hits a genuine limit, the platform
names the escalation and does the mechanical part. An expression that grows past a
complexity threshold offers "extract to a code block" and generates the block with
correct port types already inferred. The user's next step is writing *their logic*,
not learning the block format. Capability ceilings are the natural teaching moment,
and most platforms waste them by returning a validation error instead.

**Explicit non-goal: a tutorial.** Nobody completes the tutorial. The template
gallery is the tutorial, because it produces a working artifact the user actually
wanted, which they can then open and inspect.

---

## 4. Technical implementation

### 4.1 Component inventory

| Component | Responsibility | Build or buy |
|---|---|---|
| **IR schema + validator** | Defines and enforces the canonical document | Build (it is the product); JSON Schema + a typed codec |
| **Projection layer** | Bidirectional render/apply for each surface | Build — highest-risk component (§4.2) |
| **Expression evaluator** | Pure, total expression language | Build on a sandboxed parser (CEL or a restricted JS subset); do **not** embed a general interpreter |
| **Block registry** | Catalog, versions, ownership, certification, usage stats | Build thin service over Postgres + an OCI registry for code artifacts |
| **Validation/compile** | Port type-check, effect aggregation, cost estimate | Build |
| **Policy engine** | Policy-as-code over declared effects and tier | Buy/adopt: OPA or Cedar. Do not write a policy DSL |
| **Durable scheduler** | At-least-once step execution, retries, waits, timers | Buy/adopt: Temporal (or equivalent). Writing a correct durable executor is a multi-year project and a tempting mistake |
| **Code sandbox** | Untrusted code execution | Adopt: V8 isolates for JS/TS, WASM or gVisor-class isolation for Python. Security boundary — not a place for cleverness |
| **Egress proxy** | Capability-scoped network access per declared effects | Build thin; deny by default, allowlist derived from the block manifest |
| **Run store** | Event-sourced run log, step I/O, replay | Build on Postgres + object storage for large payloads |
| **Observability** | Trace view, step inspector, replay, cost attribution | Build UI over the run store; export OpenTelemetry |

The build/buy split matters for credibility of the plan: the differentiated work is
the IR, the projection layer and the governance model. Durable execution, policy
evaluation and sandboxing are solved problems where adopting a mature implementation
is strictly better than a bespoke one.

### 4.2 The projection layer in detail

This is where the design either works or quietly fails, so it deserves specifics.

Each surface implements two functions:

```
render : (IR, viewContext) -> ViewModel
apply  : (ViewEdit)        -> Patch[]          // typed ops, never a whole document
```

Patches are the only way state changes:

```typescript
type Patch =
  | { op: "addNode";    node: Node }
  | { op: "removeNode"; id: NodeId }
  | { op: "setInput";   node: NodeId; port: string; value: InputBinding }
  | { op: "addEdge";    from: PortRef; to: PortRef }
  | { op: "setUi";      node: NodeId; ui: Partial<UiAnnotation> }   // canvas-only region
  | { op: "setSource";  block: BlockRef; source: string }           // verbatim bytes
  | ...
```

Four consequences, each of which solves a problem the naive design has:

1. **The canvas structurally cannot corrupt code comments**, because `setUi` only
   writes the `ui:` region and `setSource` stores bytes verbatim. This is an
   architectural guarantee, not a test result.
2. **Diffs are minimal.** Moving a node emits one `setUi` patch, so the Git diff is
   two lines. Code review of workflow changes stays feasible.
3. **Undo, audit log and concurrent editing all fall out of one mechanism.** The
   patch stream is the undo stack, the audit trail, and the CRDT/OT input.
4. **Merge conflicts are rare and legible** — but only because nodes and edges are
   *keyed maps* rather than lists (§4.3). With lists, two people each adding a node
   both append at the same position and conflict even though their work is
   unrelated, which would undo most of the benefit. Keyed by stable id, their edits
   touch disjoint regions of the file and merge. Deterministic serialisation then
   keeps the surviving diffs minimal. Genuine conflicts are confined to concurrent
   edits of the same field, which is the correct behaviour.

**Round-trip testing is a CI gate, not a test suite nicety.** The naive
formulation is a trap, though, and worth spelling out so nobody builds it:

```
# WRONG -- requires an oracle you cannot write
parse(serialise(apply(render(d, s), e))) == expected(d, e)
```

`expected(d, e)` is the entire difficulty. Computing the correct result of an
arbitrary edit means reimplementing the edit semantics in the test, and now you
have two implementations that can agree with each other and both be wrong. This is
the standard property-testing oracle problem, and it is why "we have round-trip
tests" is often worth less than it sounds.

The tractable version asserts **invariants**, each of which needs no oracle:

```
for all valid IR documents d, surfaces s, valid edits e:

  1. IDENTITY      parse(serialise(d)) == d                    # no edit at all
  2. LOCALITY      regions of d untouched by e are BYTE-identical after the edit
  3. REFLECTION    render(result, s) shows e applied            # the edit landed
  4. IDidempotence apply(e) twice == apply(e) once              # where e is idempotent
  5. STABILITY     serialise(parse(serialise(d))) == serialise(d)
  6. CROSS-SURFACE render(d, canvas) and render(d, code) agree on node set,
                   edges, and every input binding
```

Invariant 2 is the one that catches real bugs: it is how you discover that saving
from the canvas dropped an unknown field written by a newer client, reordered a
map, or normalised a quoted string. Invariant 6 is what stops the two surfaces
drifting into disagreement about what the document means — the failure that
produces two platforms. A generator over the IR grammar supplies `d` and `e`; a
projection that fails any invariant does not ship.

### 4.3 Data models

**Workflow** — the canonical artifact:

```yaml
schema: workflow/v1
id: wf_<slug>
name: string
owner: team:<slug> | user:<id>
tier: personal | team | business-critical      # -> governance requirements
version: 7                                     # immutable; publishing increments
state: draft | published | deprecated

parameters:                     # what a template exposes; what a wizard renders
  - { name: threshold, type: number, default: 10000, locked: false }

triggers: [ { id, kind: webhook|schedule|event|manual, config } ]

# Nodes and edges are KEYED MAPS, not lists. This is a merge property, not a
# style preference: two people each adding a node to a YAML *list* both append at
# the same position and conflict, which would break the clean-merge claim in 4.2.
# Keyed by stable id, concurrent insertions touch disjoint regions and merge.
nodes:
  <NodeId>:                                    # stable, never reused
    block: <name>@<major>          # OR kind: decision | loop | parallel | wait
    inputs:
      <port>: { literal: any } | { ref: PortRef } | { expr: string }
    retry:   { attempts, backoff, retry_on }
    timeout: duration
    on_error: fail | continue | route:<NodeId>

edges:                          # keyed by edge id for the same reason
  <EdgeId>: { from: PortRef, to: PortRef }

ui:                             # non-semantic. Canvas owns; code editor never writes
  nodes:
    <NodeId>: { x, y, collapsed }
  notes: { <NodeId>: string }
  # Annotations are keyed by node id and are garbage-collected when that node is
  # removed, so deleting a commented node does not leave an orphaned note behind.

x-unknown-preserved: {}         # forward compatibility: never dropped on save
```

**Block** — the unit of extension and the boundary of encapsulation:

```yaml
schema: block/v1
name: calculate-tax
version: 2.1.0                  # semver; workflows pin major
owner: team:tax-engineering
visibility: private | team | org
certification: core | verified | community | uncertified

ports:
  inputs:  { amount: {type: number, required: true}, jurisdiction: {type: string} }
  outputs: { tax: {type: number}, breakdown: {type: "TaxLine[]"} }

# Effects split by how far they can be trusted. Conflating the two would
# overstate what the platform can guarantee -- see the note below the schema.
effects:
  enforced:                     # mechanically verifiable at the sandbox boundary
    - pure                      #   no syscalls, no sockets, no filesystem
    - network: [api.vendor.com] #   egress proxy denies anything unlisted
    - filesystem: none
  attested:                     # author's assertion; NOT mechanically verifiable
    - reads: [pii]
    - writes: [finance]
    - cost: 0.004
    - human-in-the-loop: false
impl:
  kind: visual-subgraph | expression | code | connector
  runtime: node20 | python3.12          # when kind=code
  artifact: oci://registry/blocks/calculate-tax@sha256:...
  source_ref: git://...                  # for "view source" from the canvas

tests:   { count: 14, passing: true, coverage: 0.82 }
usage:   { workflows: 37, runs_30d: 12400, error_rate_30d: 0.0003 }
deprecation: { superseded_by: calculate-tax@3, sunset: 2027-06-30 }
```

The `effects` field is the keystone of the governance model. Because it is declared
on the block and **aggregated statically up the workflow graph**, the policy engine
can answer "does this workflow read PII and egress to the internet?" *before the
workflow has ever run*.

**But only half of it is enforceable, and the design must not pretend otherwise.**
A sandbox can guarantee that a block declaring `pure` cannot open a socket, and an
egress proxy can guarantee it reaches no host outside its allowlist — those are
boundary properties, checkable mechanically. `writes: [finance]` is a different
kind of claim entirely: no sandbox can inspect bytes leaving a block through a
declared, allowlisted connection and determine whether they constitute a financial
write. That is a semantic assertion by the author.

So the two classes get different treatment:

| | Guarantee | Consequence of a false declaration |
|---|---|---|
| `enforced` | Mechanical, at the sandbox and proxy boundary | The block simply fails -- it cannot do the undeclared thing |
| `attested` | None. Author's word, reviewed by a human | Caught by review, by audit sampling, or not at all |

Three mitigations, none of which makes an attestation a guarantee:

* **Certification tiers do the real work for attested effects** (§5.3). A block
  with network egress that claims `writes: none` is exactly what security review
  exists to examine, and `uncertified` blocks are barred from `business-critical`
  workflows.
* **Enforced effects bound the attested ones.** A block with
  `enforced: [pure]` cannot write anything anywhere, so `writes: none` is implied
  rather than trusted. Attestation only carries weight where egress is permitted.
* **Connector-mediated access makes some attestations enforceable.** If finance
  writes are only reachable through a platform connector rather than raw HTTP,
  `writes: [finance]` becomes observable at the connector and stops being a claim.
  Routing sensitive systems behind connectors is how you shrink the attested set
  over time.

This is still what lets a business user safely compose an engineer's code -- the
trust question is answered by the platform rather than by their judgment about code
they cannot read -- but it is answered with a mix of proof and accountability, not
proof alone.

**Run** — event-sourced for replay and debugging:

```yaml
run_id: string
workflow: { id, version }          # exact immutable version executed
trigger:  { kind, payload_ref }
status:   queued | running | waiting | succeeded | failed | cancelled
steps:
  - { node_id, attempt, status, started_at, ended_at,
      input_ref, output_ref,                 # object storage; redacted by data class
      error: { type, message, stack_ref } }
cost: { compute, external_calls, total }
```

Storing per-step inputs and outputs by reference is what makes one debugging
experience serve both audiences: a business user clicks a node and sees the data
that went in and came out; an engineer clicks the same node and reads the stack
trace. Redaction follows the data classes in the effects declaration, so the trace
view does not become a PII leak.

### 4.4 The expression language, and the discipline of keeping it small

Rung 3 is where scope creep is most dangerous. An expression language that grows
loops, mutable state and I/O becomes a badly designed programming language with no
debugger, no tests and no review path — and it will be where the worst production
incidents originate.

So the expression language is deliberately **pure and total**:

| Allowed | Forbidden | Escalation |
|---|---|---|
| Field access, arithmetic, comparison, boolean logic | Loops | `loop` orchestration node (rung 2) |
| String/date/number builtins | Mutable state | Block outputs |
| `map`/`filter` over collections with a pure lambda | I/O, network, filesystem | Code block (rung 4) |
| Null-safe navigation, defaults | Recursion, unbounded iteration | Code block |
| Type-checked against upstream port schemas | Untyped dynamic access | — |

Guaranteeing termination means expressions can be evaluated during editing to show
live previews with sample data, which is a significant UX win, and it means an
expression can never hang a workflow.

The rule enforced in product terms: **when a user wants something the expression
language forbids, the platform offers the next rung rather than growing the
language.** Every exception to that rule is a step toward rebuilding JavaScript
badly.

### 4.5 Execution model

```
trigger -> +----------------+  validated, immutable workflow version
           | Trigger router |
           +----------------+
                    v
           +---------+  topological plan; parallelism discovered from the DAG --
           | Planner |  a business user gets concurrency without knowing the word
           +---------+
                v
           +-------------------+  durable: survives worker loss, process restart and deploys;
           | Durable scheduler |  owns retries, backoff, timers, waits (human approvals are
           +-------------------+  just long waits)
                     |
              |----------------------|-----------------|-------------------|
              v                      v                 v                   v
  +-----------------------+  +--------------+  +--------------+  +------------------+
  | declarative evaluator |  | expr sandbox |  | code sandbox |  |    connector     |
  |                       |  |              |  |  (isolate)   |  | via egress proxy |
  +-----------------------+  +--------------+  +--------------+  +------------------+
              |                      |                 |                   |
              ------------------------------|-------------------------------
                                            v
                      event-sourced run log  -> trace view, replay, cost attribution
```

Design points worth stating:

- **Steps must be idempotent or declare that they are not.** The scheduler is
  at-least-once; a block that moves money declares `non_idempotent` and the platform
  requires an idempotency key, refusing to publish otherwise. This is the kind of
  correctness requirement a self-service platform must impose *for* its users,
  because business users will not reason about retry semantics.
- **Human approvals are durable waits, not a special subsystem.** Treating them as
  ordinary long-running steps means they inherit timeouts, escalation, audit and
  replay for free.
- **Parallelism is inferred from the graph.** Independent branches run concurrently
  without anyone requesting it.
- **The egress proxy is the enforcement point for declared network effects.** Deny
  by default; the allowlist is compiled from block manifests. A block cannot reach a
  host it did not declare.

### 4.6 Extension mechanisms

The rule that keeps the platform coherent: **everything that is not the IR, the
projection layer, or the runtime is a block.** The core never special-cases a
connector, a vendor or a department. If adding a capability requires changing core
code, the extension model is wrong and needs fixing rather than working around.

| Mechanism | Who uses it | What it adds |
|---|---|---|
| **Block SDK** | Power users, platform team | New computation: `defineBlock` with typed ports, declared effects, tests. Publish to the registry |
| **Promote subgraph to block** | Anyone | Select nodes on canvas → publish as a reusable block. **The flywheel — see §5.2** |
| **Connector framework** | Integration engineers | Third-party systems as blocks with managed auth and scoped egress |
| **Custom triggers** | Power users | New ways to start workflows (queue consumer, file watcher, domain event) |
| **Template packs** | Domain champions | Curated starting points per department, versioned like blocks |
| **Expression builtins** | Platform team only | Deliberately restricted: a governance decision, not an open extension point (§4.4) |
| **Policy bundles** | Security, compliance | Org rules as code; evaluated on publish, not bolted on afterwards |
| **Webhooks / events out** | Everyone | Workflows as participants in the wider event fabric |

Two notes on the SDK: it emits **IR, not orchestration code**. An engineer using the
SDK to build a whole workflow programmatically produces the same YAML a canvas user
would, which is why rung 5 does not fork the platform. And blocks are versioned
artifacts in an OCI registry with immutable digests, so a published workflow's
behaviour cannot change underneath it.

---

## 5. Long-term maintainability

### 5.1 Platform evolution

A self-service platform accumulates thousands of artifacts built by people who have
since changed teams or left. Evolution therefore has to happen *without* asking
authors to migrate their own work, because most of them are no longer available to
ask.

**Three rules make the IR evolvable:**

| Rule | Consequence |
|---|---|
| **Additive-only schema changes within a major version** | New optional fields; never a changed meaning for an existing field |
| **Unknown fields preserved on read and write** | An old client cannot silently delete data a new client wrote — the single most common corruption bug in multi-client systems |
| **Explicit `schema:` version on every document** | Migration is a function, and its applicability is decidable |

**Migrations are code the platform owns, not homework it assigns.** A `workflow/v1
→ v2` change ships with a migration function, runs lazily on read and in a backfill,
and is covered by the same round-trip property tests as the projections. Blocks get
codemods: `calculate-tax@2 → @3` ships with a transform for the mechanical part of
call-site updates, plus a report of what needs human judgment.

**Deprecation runs on a clock, publicly.** A deprecated block shows a sunset date on
every canvas that uses it, with the usage count and the replacement named. Sunset
removes it from search first, then blocks *new* usage, then finally fails on publish —
never on an existing production run without a long, visible notice period.

**Versions are immutable.** A published workflow version pins exact block digests, so
its behaviour cannot drift. Upgrading is an explicit act that produces a new version
with a reviewable diff. This is the difference between a platform teams trust with
payroll and one they don't.

### 5.2 The flywheel: power users supply business users

This is the part of the design that makes the original tension productive rather
than merely survivable. The two audiences are not competing for one dial — they are
a **supply chain**, and the platform's central job is to shorten it.

```
  +------------------+    hits a limit    +---------------------------+
  |  Business user   |------------------->|        Power user         |---- "publish as block"
  | builds on canvas |   asks for help    | writes code or a subgraph |   |
  +------------------+                    +---------------------------+   |
            ^                                                             |
            |                                                             |
            |                                                             |
            |                                                     +-------v--------+
            |                                                     | Block registry |
            |                                                     |  typed ports   |
            |                                                     |    effects     |
            |                                                     |  owner, tests  |
            |                                                     +----------------+
            |                                                              |
            |  appears as ONE drag-and-drop node,                          |
            |---------------------------------------------------------------
               with a form, in the catalog

Each loop: the catalog gains a capability, the next business user
needs no help, and the power user's work is leveraged, not consumed.
```

Contrast with what happens without this mechanism: a power user helps a colleague by
building a bespoke workflow, which then has exactly one user and one maintainer, and
the next colleague with the same need starts over. The engineering effort does not
compound — it is spent.

Three design requirements fall out of taking the flywheel seriously:

- **"Publish as block" must be a two-click operation on a canvas selection**, with
  port types inferred from the subgraph's boundary. If it requires a repo, a
  pipeline and a review, nobody does it and the flywheel never turns.
- **The catalog must surface evidence, not just names**: owner, usage count,
  observed 30-day error rate, test status, effects. A business user's decision to
  depend on someone else's block is a real engineering decision made by a
  non-engineer, and the platform owes them the data to make it.
- **Block authorship must be visible and credited.** Internal platforms live or die
  on whether contributing is rewarded. Usage counts on a leaderboard are cheap to
  build and change behaviour.

The health metric for the whole platform is the **reuse ratio** — blocks consumed
per block authored. A ratio near 1 means everyone is building bespoke things and the
platform is a worse IDE. A ratio in the tens means the flywheel is turning.

### 5.3 Governance: proportional to blast radius, not to authoring method

The instinct is to govern by *how* something was built — code gets review, drag-and-drop
doesn't. That instinct is exactly backwards, and acting on it is how low-code
platforms become shadow IT.

A workflow that pays invoices is dangerous whether it was assembled by dragging
boxes or written in TypeScript. A workflow that renames files in one person's folder
is harmless either way.

**So governance keys on two declared properties: tier and effects.**

| Tier | Example | Review to publish | Observability |
|---|---|---|---|
| **personal** | Reformat my exported report | None | Self-serve logs |
| **team** | Route support tickets, notify a channel | Team owner approval | Team dashboard |
| **business-critical** | Pay invoices, provision access, touch the ledger | Named approvers + security review when effects include `writes:finance`, `reads:pii` or unrestricted egress | Paged on-call, SLO, audit retention |

Tier is declared by the author and **validated against observed effects**: a workflow
declaring `personal` while aggregating `writes:finance` is rejected at publish with
the specific effect named. You cannot opt out of governance by mislabelling, and the
error message teaches rather than scolds.

**Block certification** is the parallel axis:

| Tier | Meaning | Requirements |
|---|---|---|
| `core` | Platform team owns it | Full review, SLA, migration guarantees |
| `verified` | Reviewed and endorsed | Tests, named owner, security review if effects are non-pure |
| `community` | Shared as-is | Declared effects, sandbox enforcement, visible usage stats |
| `uncertified` | Private/experimental | Cannot be used by a `business-critical` workflow |

Review effort is proportional to declared effects: a `pure` block needs tests and an
owner; a block with network egress and PII access needs security review. This is
enforceable precisely *because* effects are declared and sandbox-enforced (§4.3) —
without that, every review would have to assume the worst.

### 5.4 Community and ownership

**Orphaned workflows are the predictable failure mode.** A platform three years in
is full of business-critical automation whose author left. Concrete mitigations:

- **Ownership is a team, never a person**, for anything above `personal` tier.
- **Quarterly ownership recertification** for `business-critical` workflows; an
  unclaimed workflow escalates to the team's manager, then to a deprecation clock.
- **An unowned block with usage cannot simply be deleted** — it is adopted by the
  platform team as `core` or formally sunset with migration help.

**Domain champions over a central queue.** The platform team owns the IR, the
projection layer, the runtime and the core blocks. Each department has one or two
champions who own its template pack and domain blocks. This keeps the platform team
off the critical path for domain knowledge they do not have, which is the usual
reason internal platforms stall.

**An internal marketplace** with search, usage counts, reliability stats and
ratings. The important function is not discovery but **deduplication**: search-before-create
prompts, plus a periodic near-duplicate report, are what prevent the predictable
drift into four hundred slightly different "send Slack message" blocks.

### 5.5 How we would know it is working

Metrics chosen because they would actually falsify the design, not flatter it:

| Metric | Reads as healthy | Failure it detects |
|---|---|---|
| Time to first successful run | < 10 min (templated) | Onboarding is broken |
| % workflows authored by non-engineers | rising | Reverted to an engineering tool |
| **Escalation rate** (workflows using rung ≥ 3) | 15–40% | Too low: power users went elsewhere. Too high: the visual layer is inadequate |
| **Descent rate** (workflows edited on canvas *after* code was added) | **> 0, and growing** | **The single most important number here. If it is zero, the one-way door exists in practice whatever the architecture claims** |
| Block reuse ratio | > 10 | Flywheel not turning; effort is spent, not compounded |
| Round-trip property-test pass rate | 100%, gating | Projection drift |
| Orphaned business-critical workflows | → 0 | Governance decay |
| Support tickets per 100 workflows | falling | Platform not actually self-service |

The descent rate deserves the emphasis. Every claim in this document rests on
bidirectionality being real in practice rather than true on paper, and the only
evidence that counts is business users continuing to edit workflows that contain
power users' code.

### 5.6 Team topology

| Team | Owns | Does not own |
|---|---|---|
| **Platform core** | IR, projection layer, scheduler, sandboxes, policy engine, core blocks | Domain logic, department templates |
| **Domain champions** (embedded, part-time) | Template packs, domain blocks, departmental onboarding | Platform primitives |
| **Governance council** (small, cross-functional) | Certification decisions, tier policy, deprecation approvals | Day-to-day review (delegated to owners) |

The anti-pattern to avoid: a central team that reviews every workflow. It becomes
the bottleneck that self-service was supposed to remove, and it is why many internal
automation platforms end up with lower throughput than the ticket queue they
replaced.

---

## 6. Risks and how they bite

Stated honestly, because a design document that lists only strengths is not useful
for deciding whether to build this.

| Risk | Severity | Why it is plausible | Mitigation |
|---|---|---|---|
| **The projection layer is hard** | **Highest** | Bidirectional editing is genuinely difficult engineering and it is on the critical path for everything else | Patch-based edits (never whole-document rewrites); property-based round-trip tests as a release gate; §2.3's structural scoping so unrenderable content is *contained* rather than round-tripped |
| **Expression language scope creep** | High | Every quarter brings a reasonable request for one more feature; the sum is an unplanned language | Written charter that it stays pure and total; every request answered with the next rung, and the scaffolding to get there cheaply |
| **Sandbox escape** | High | Executing untrusted employee code is a real RCE surface | Adopt mature isolation (V8 isolates, WASM/gVisor); deny-by-default egress proxy; effects declaration enforced at runtime; treat it as a security boundary with a threat model and external review |
| **Shadow production** | High | Business-critical automation accumulates without anyone deciding it should | Blast-radius tiering validated against observed effects; promotion pipeline; ownership recertification |
| **Block sprawl** | Medium | Easy publishing is the point, and it has this cost | Search-before-create; near-duplicate reports; certification tiers; usage-based promotion and sunset |
| **Canvas does not scale to large graphs** | Medium | A 150-node workflow is unreadable however it is drawn | Subgraph blocks as the primary decomposition tool; collapse/expand; search and minimap. Partly mitigated by the flywheel, which rewards extraction |
| **Power users reject it anyway** | Medium | Engineers are rightly suspicious of low-code | Rung 5 is a first-class citizen: real Git, real CI, real code review, real local testing, a real API. If the CLI feels like a toy, they leave — and this is a product-quality risk, not an architectural one |
| **Migration debt** | Medium | Thousands of artifacts, few available authors | Platform-owned migrations and codemods; additive-only changes; preserved unknown fields |

### 6.1 What this design does not solve

Four gaps a reviewer should be able to find without being told, so they are stated
rather than left to be discovered.

**Phase 1 has a chicken-and-egg problem the roadmap creates.** Rungs 4-5 arrive in
months 6-9, so power users are underserved for half a year. But power users are
exactly who build the initial block catalog, and business users cannot compose
blocks that do not exist. Shipping phase 1 with 20 hand-built core blocks and no
contribution path means the catalog does not grow until phase 3. The mitigation is
unglamorous: the platform team writes the first 40-50 blocks itself, treating that
as seeding cost rather than product work, and recruits a handful of power users as
design partners with direct SDK access ahead of general availability. Both cost
real headcount, and the plan should say so.

**Template divergence is unaddressed.** Templates are workflows with locked
parameters, and "customise" unlocks them — which forks the instance. When the
template author ships a fix, forked instances do not receive it, and there is no
merge path back. This is the classic fork problem and it has no clean answer here.
A partial one: keep a `derived_from: <template>@<version>` link, surface "the
template you started from has updated" with a diff, and let the owner adopt changes
node by node. That is a real feature with real cost, not a footnote.

**Canvas scalability has a circular answer.** §6 says a 150-node workflow is
unreadable and that subgraph blocks are the decomposition tool — but decomposing
requires the author to notice the need and do the work, which is precisely what
someone who built a 150-node graph did not do. A platform that only rewards
decomposition after the fact has already lost. Better: warn at a node threshold,
suggest extraction candidates automatically by finding weakly-connected subgraphs,
and make "extract these 12 nodes to a block" a one-click operation on that
suggestion.

**The attested half of the effects model is only as good as review** (§4.3). A block
with legitimate network egress can carry any `writes:` claim its author chooses,
and no sandbox will catch a false one. Certification tiers and audit sampling are
accountability mechanisms, not guarantees. Shrinking the attested set by routing
sensitive systems behind platform connectors is the structural fix, and it is
ongoing work rather than a launch property.

**The honest summary of the risk profile:** this design concentrates nearly all of
its technical risk in one component, the projection layer, and buys a great deal of
simplicity elsewhere in exchange. That is a deliberate trade — one hard problem with
a clear correctness property that can be mechanically tested is more tractable than
a diffuse mess of partial integrations between two half-platforms. But if the
projection layer cannot be made to round-trip reliably, the design does not degrade
gracefully; it degrades into the two-platform outcome it was built to avoid. That is
the thing to prototype first and the thing to kill the project over.

---

## 7. Delivery sequence

Four phases. The sequencing argument matters more than the dates.

```
 Phase 1 -- 0-3 months -- "the spine"
+------------------------------------------------------------------------------+
| IR + schema + validator . deterministic serialiser . durable scheduler       |
| canvas (rungs 0-2) . ~20 core blocks . one run/trace view                    |
| round-trip property tests in CI from day one                                 |
|                                                                              |
| <- Ship visual-only. But the IR is already text-serialisable, already        |
|    has stable IDs, already preserves unknown fields, already has a           |
|    ui: region. No user can see any of that yet. It is still the most         |
|    important work in the project.                                            |
+------------------------------------------------------------------------------+

 Phase 2 -- 3-6 months -- "depth"
+------------------------------------------------------------------------------+
| expressions (rung 3) with live preview . block SDK . registry                |
| policy engine + tiering . observability: step I/O inspection, replay         |
+------------------------------------------------------------------------------+

 Phase 3 -- 6-9 months -- "the escape hatch, in both directions"
+------------------------------------------------------------------------------+
| code blocks (rung 4) + sandbox + egress proxy . promote-subgraph-to-block    |
| <- the flywheel starts turning here; descent rate becomes measurable         |
+------------------------------------------------------------------------------+

 Phase 4 -- 9-12 months -- "the ecosystem"
+------------------------------------------------------------------------------+
| CLI / SDK / GitOps (rung 5) . marketplace . certification . migrations       |
| domain champions onboarded . deprecation tooling                             |
+------------------------------------------------------------------------------+
```

**The sequencing insight, which is the main practical claim of this section:** build
the IR and the projection discipline in phase 1, *before any user can benefit from
them*. It is tempting to ship a visual builder whose state is whatever the React
tree happens to hold, and add an export later. That is precisely how platforms
arrive at the two-product problem — by the time anyone wants a code path, the
authoritative state lives in UI components, the serialiser is lossy, and
"bidirectional" has become an unaffordable rewrite.

The architecture has to be correct before the features that depend on it exist. In
exchange, phases 3 and 4 are mostly additive: new surfaces over a document model
that already supports them.

**What phase 1 deliberately omits:** code blocks, CLI, marketplace, certification.
Power users are underserved for roughly six months. That is a real cost, and the
alternative — shipping a code path over an immature IR — costs more, because it
bakes the divergence in permanently.

---

## 8. Summary

The paradox dissolves once you stop treating simplicity and power as positions on
one dial and start treating them as **notations over one artifact**.

Five load-bearing decisions:

1. **One canonical IR; every surface is a projection over it.** No surface owns
   state another cannot see. This is the whole design in one sentence.
2. **Declarative orchestration, encapsulated computation.** Structure is always
   declarative and therefore always drawable; arbitrary code lives inside typed
   blocks and is therefore always *representable* even when not expandable. This is
   what makes bidirectionality achievable instead of aspirational.
3. **Escalation is local and reversible.** Per node, not per workflow; descent
   always available, enforced by a property test in CI rather than promised in
   documentation.
4. **Power users supply business users.** "Publish as block" converts one
   engineer's custom logic into every business user's drag-and-drop node. The
   audiences are a supply chain, and the platform's job is to shorten it.
5. **Governance keys on blast radius and declared effects, never on authoring
   method.** Dragging boxes to move money is still moving money.

The measure of success is not adoption or satisfaction, both of which a
two-platform compromise can achieve for a while. It is the **descent rate**: business
users continuing to edit, on the canvas, workflows that contain power users' code.
If that number is positive and growing, the door stayed open and there is genuinely
one platform. If it is zero, there are two platforms wearing one brand — whatever
the architecture diagram says.
