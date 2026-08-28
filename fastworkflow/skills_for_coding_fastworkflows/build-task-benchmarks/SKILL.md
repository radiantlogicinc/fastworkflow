---
name: build-task-benchmarks
description: >-
  Build a two-tier conversation benchmark for a fastWorkflow workflow: short single-errand
  conversations that act as unit tests and reusable blocks, then long real-world tasks of 20-100
  steps composed from those blocks. Covers turn and conversation structure, seeding objects with
  enough structure to sustain a chain, reading handles out of rendered responses, the verdict
  vocabulary that keeps a suite honest, crossing contexts with the framework navigation commands,
  static context planning, and scoring execution and intent routing as two independent axes so
  compounding failure over long tasks becomes visible. Use when a workflow has no end-to-end
  coverage, when per-command smoke tests pass but real tasks fail, when asked how well a workflow
  handles multi-step or real-world work, or when building an agent benchmark over a command tree.
---

# Building task benchmarks

A per-command smoke test proves a command reaches its backend. It cannot tell you whether a user
can get a job done, because a job is many commands in sequence, and everything that makes
sequences hard — navigation, handles passed between turns, the context a turn lands in — is
exactly what a per-command test removes.

Build the benchmark in two tiers. Never skip the first one.

| Tier | Shape | Answers |
|---|---|---|
| **Conversations** | 3-15 turns, one errand, usually one workspace | Does each capability work in its natural setting? |
| **Tasks** | 20-100 steps, several conversations chained, many contexts | Does a real job complete? |

The tiers are not two views of the same thing. Tier 1 is your unit-test layer **and** the block
library tier 2 is assembled from. That is what makes a tier-2 failure attributable: every block is
independently green, so a step that only fails at length is a length problem, not a command
problem. If you write tier 2 first you will spend your time debugging whether the command or the
chain is broken.

## Tier 1 — conversations as unit tests

### Structure

A conversation is a name, a purpose, and an ordered list of turns. A turn is what the user
**says**, the command that should handle it, its parameters, and optionally what to capture from
the response and which context it should land in. Full dataclass and runner skeleton:
[reference.md](reference.md).

Four rules make the difference between a real harness and a loop over command calls:

1. **Navigation is real.** The context a turn lands in is the context the next turn runs in. Do not
   reset between turns. A conversation that only works because each turn was handed a fresh context
   is not testing anything a user will do.
2. **Handles come out of the rendered response.** Parse the uid a later turn needs from the *text*
   an earlier turn produced, not from internal state. The rendered response is all an agent ever
   sees, so if a listing does not expose its handles legibly, that is a defect this suite should
   catch.
3. **Run the validation hook.** `Signature.validate_extracted_parameters` is part of the runtime
   path, and normalising rules write back onto the parameter object the command body then reads.
   Skipping it tests a call the runtime never makes. Return a validation failure as a soft outcome,
   not an exception — the runtime hands the message back for correction, so a bad call is still a
   conversation.
4. **Writes are opt-in and paired.** Every write turn gets a restore turn, so a default run is
   net-neutral. Anything genuinely destructive is *validated* (its input object is constructed) and
   never sent.

### Verdicts

One boolean per turn destroys the signal. Distinguish, at minimum:

| Verdict | Meaning |
|---|---|
| `OK` | Ran and answered with rows, or entered a context |
| `EMPTY` | Ran correctly; this dataset holds nothing to show |
| `SOFT` | Deliberate not-found or rejected-parameters answer |
| `STUB` | Command declares itself unimplemented and says so |
| `SKIP` | Not run in this mode (write / unsafe); parameters still validated |
| `BLOCKED` | An earlier turn failed, so the chain could not continue |
| `FAIL` | The command raised, or landed in the wrong context |

`EMPTY` versus `FAIL` is the distinction that keeps a suite honest. A suite that scores an empty
dataset as failure gets muted; one that scores it as success hides real regressions. Keep them
apart and report both.

### Seeding

Do not seed from the first page of a picker. Seed objects with **enough structure hanging off them
to sustain a whole chain** — a person who has rights *and* accounts *and* group memberships, a
container that has children, a record that has a parent. Rare shapes are exactly the ones that
break chains and exactly the ones missing from page one.

- Query with explicit bounds. An unbounded scan of a realtime view can take the backend down.
- Where a shape is rare, walk the catalogue rather than the first page, and say so in a comment.
- Seed **per variant** when a command's answer is typed. If findings are typed by entity family,
  one seed per family, and ask each only for the family it has — otherwise you record an empty
  result as a command defect.
- Fail loudly at seed time if the dataset cannot support the suite. A missing seed should stop the
  run, not silently produce twenty `EMPTY`s.

### Coverage

Enumerate the routable command set from the routing definition and diff it against the commands
your conversations exercised. Print the uncovered ones. Without this, a new command is
untested-by-default and nobody notices.

## Tier 2 — real-world tasks

### Composition, not copy-paste

Write each tier-1 errand as a **function returning a list of turns**, then build tasks by
concatenating them. Record on each task which conversations it combines. Copy-pasted steps drift
from the blocks they came from and destroy attributability.

Pick task subjects a practitioner would recognise as a unit of work — an investigation, a
recertification, an onboarding or offboarding sweep, a periodic review, an audit of one object
across every system it touches. The test of a good subject is that you can state its purpose in
two sentences without naming a command.

### The ladder is the experiment

Build roughly six or seven tasks spanning **20 to 100 steps**, each about 1.4× the last. The ladder
is not padding — it is how you separate two hypotheses that a single long task cannot:

- *Long tasks are harder per step* (per-step quality falls as the task grows), versus
- *Long tasks have more chances to fail* (per-step quality is flat; only exposure grows).

In practice it is the second, and that changes the fix entirely.

### Crossing contexts

A task that stays in one workspace is a tier-1 conversation wearing a costume. Real tasks move
between them, and movement is a **command**, not a harness operation.

fastWorkflow puts navigation commands in every context — reset to root, go up to parent, list what
is available here, report the current context. They live in the framework's metadata-extraction
workflow rather than the workflow's own command folder, and they read the app workflow off *their*
workflow's context. So a harness must run the real command bodies through a small shim object that
carries `{"app_workflow": workflow}`, never reimplement them. Reimplementation is how you end up
testing your own parent-chain logic instead of the framework's.

Treat these verbs as first-class steps: they get a `say`, they get routed, and they are scored like
everything else. They are usually the least-trained commands in a workflow and the most
load-bearing in a long task.

### Static context planning

Because navigation is declared — a turn either names the context it enters, or is a go-up, or is a
reset — the context every step runs in is computable **before running anything**. Maintain a
current context and a stack; a navigation turn pushes, a go-up pops, a reset clears.

This pays twice: it is what lets you score routing with no backend at all, and in live mode it is a
free assertion that the workflow's real navigation agrees with the declared plan.

## Score two axes, never blended

| Axis | Needs | Answers |
|---|---|---|
| **Execution** | live backend | Does the data layer survive the length? |
| **Routing** | trained models only — no backend, no LLM | Would an agent have picked this command? |

Run them from one harness over one task definition so the two numbers describe the same steps.
Keeping routing backend-free matters: it is the half that can run in CI.

Expect these to diverge sharply, and the divergence is the finding. A workflow can execute every
step of a 100-step task perfectly and still be unusable, because the harness names commands
explicitly while a user types English.

### Routing outcome taxonomy

Scoring routing as right/wrong loses the two outcomes that matter most:

| Outcome | Meaning | Cost |
|---|---|---|
| `direct` | The classifier named the command outright | None — the only clean route |
| `escalated` | Command is not on this surface; a **lone** escalation label correctly sent it up the parent chain | None — this is a feature, not a near-miss |
| `ambiguous` | Right command returned among several candidates | Runtime stops to ask. Survivable once, corrosive over eighty steps |
| `misrouted` | A different command would have fired | Task derails |
| `lost` | Escalated, but nothing up the chain answers either | Task derails |

Score escalation by walking the ancestor stack exactly as the framework's wildcard handler does. An
escalation label returned *beside* local candidates takes the ambiguity branch at runtime and never
escalates, so it is `ambiguous`, not `escalated`.

### Report compounding explicitly

This is the number the benchmark exists to produce. With a per-step clean rate `p` over `n` steps,
unattended completion is roughly `p^n`. At `p = 0.95`, a 20-step task completes 36% of the time and
a 100-step task 0.6% of the time. At `p = 0.55` nothing above twenty steps is viable.

Report per-task: steps, per-step clean rate, `p^n`, and **the index of the first derailing step**.
The first-derail index is what a reader acts on; the percentage is what they quote.

State the independence caveat honestly — a command that derails derails every time it appears, so
real runs cluster their failures rather than sampling independently. It does not change the
direction of the conclusion.

## Phrasing discipline

The routing half is only as honest as its wording.

- **Hold the phrasings out of the training seeds**, and make the harness *refuse to run* if any
  leaks in. Compare on a normalised form so dropping a full stop does not evade the check.
- **Rotate three or four phrasings per command** across tasks, deterministically, so no task's
  score rests on one lucky wording and two runs stay comparable.
- **Hold one business vocabulary steady across a whole session.** A real practitioner says
  "credential" or "login" consistently, not a different synonym each time. This scores lower than
  per-case varied wording and is the more representative number — a consistent vocabulary the
  seeds never trained on will fail consistently, which is precisely the failure users hit.
- Offline runs have no seeded ids; substitute readable placeholders, and carry the *labels* from
  the last live run so searches are worded as they really would be.

## What long tasks surface that short ones cannot

Look for these; they are the recurring findings.

- **Navigation verbs are load-bearing and undertrained.** The reset-to-root verb typically ships
  with a couple of global phrasings and must be recognised from inside every entity context. When
  it misroutes, the session is stranded rather than merely one step poorer.
- **Dual-purpose commands are the largest single routing cost.** A command that both lists rows and
  opens one — same label, distinguished only by whether a uid was supplied — has to absorb
  "show me the accounts" and "step into that account". The second reads as navigation and there is
  no navigation label to receive it. Splitting the label is usually the highest-value fix.
- **Weight contexts by dwell time.** Rank contexts by steps executed in them, not by command count.
  A context a task passes through twice matters far less than one it works inside forty times.
- **Deployment gaps only a long task reaches.** Writes that depend on a vocabulary no deployed
  command can create; a command empty on every object reached from four different directions. Both
  are invisible to a suite that touches each command once.

## Anti-patterns

- Blending execution and routing into one score. They have different causes and different fixes.
- Letting a long task be the only coverage of a command. Verify every command appears in at least
  one tier-1 conversation.
- Making routing require the backend. Half your benchmark then cannot run in CI.
- Inventing working data the deployed surface cannot create. Check what exists before writing a
  step that labels, tags or classifies something.
- Reporting a percentage without the first-derail index.

## Checklist

```
- [ ] Tier 1 exists and covers every routable command; uncovered commands are printed
- [ ] Tasks are composed from tier-1 block functions, and record which they combine
- [ ] Step ladder spans ~20 to ~100 with no large gap
- [ ] Context transitions use the framework navigation commands, run through a shim
- [ ] Static context plan agrees with the live run
- [ ] Handles are parsed from rendered responses, not internal state
- [ ] Verdicts separate EMPTY from FAIL from STUB from SKIP
- [ ] Writes are paired with restores; destructive commands validated, never sent
- [ ] Routing runs with no backend and no LLM
- [ ] Escalation scored by walking the ancestor stack, lone label only
- [ ] Phrasings held out from seeds, enforced by a run-blocking check
- [ ] Report gives per-step rate, p^n, and first-derail index per task
```

## Related

- Reading a training run's held-out routing numbers, and the single-utterance
  `intent_benchmark.json` that complements these multi-turn tasks: the `evaluate-intent-routing`
  skill.
- Turning a routing failure found here into a structural fix: the `design-context-models` and
  `detect-duplicate-capabilities` skills.
- Making a handle-consuming step reachable by declaring its producer: the
  `declare-parameter-producers` skill.
- Harness dataclasses, runner, context planner and routing scorer: [reference.md](reference.md).
