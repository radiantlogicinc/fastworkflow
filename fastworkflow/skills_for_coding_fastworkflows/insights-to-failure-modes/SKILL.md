---
name: insights-to-failure-modes
description: >-
  Cluster distilled insights (fastworkflow distill output - execution_agent_anti_patterns.md
  and planning_agent_insights.md, from teacher-student, self or pool distillation, for any
  workflow) into a failure-mode taxonomy YAML that says, for every mode, whether and how it
  can become a deterministic policy verifier rule. Use when asked to cluster, group or make a
  taxonomy of insights or anti-patterns, to turn distillation output into failure modes or
  verifier rules, to extend a taxonomy with a new distillation round, or to decide which
  insights should become policy rows. The whole loop is in distill-policy-verifiers.
---

# Insights -> failure modes

Distillation produces a flat list of prose rules ("insights"). Many repeat each other, some
are wrong, and some are about one task only. A verifier table needs the opposite: a few
**workflow-level failure modes**, each with a check a program can run. This skill is the step
in between. Its output is a taxonomy file that a person or agent can compile into policy rows
without re-reading the insights. It is one step of
[distill-policy-verifiers](../distill-policy-verifiers/SKILL.md).

It applies to any workflow. The examples below come from different domains on purpose.
Nothing here assumes a particular set of commands; learn the workflow's commands and rules
from its own files.

The quality bar: every rule later compiled from your taxonomy will block the agent at run
time. A mode you get wrong becomes a rule that fires on good behaviour. Be conservative about
what you mark compilable, and explicit about why you left things out.

## Inputs to gather first

1. **The corpus**: the two insight files `distill` wrote (`<workflow>/Insights/<workflow>/`).
   Snapshot them before a later round appends to them, e.g. into `<workflow>/_policy/corpus/`.
   - Number the items by position: execution items `E1..En`, planning items `P1..Pn`.
   - Use a different prefix per distillation source (e.g. `SE`/`SP` for self-distilled,
     `CE`/`CP` for pool), so ids never clash across taxonomies.
   - A cumulative corpus (a later round appended to an earlier one) keeps its numbering:
     item 7 keeps its id in every round.
2. **Provenance**: which runs, tasks and trial pairs produced the corpus, and which side was
   the reference (a teacher run, or a passing trial of the same model).
   - `fastworkflow distill ... --provenance FILE` writes this. It maps every pair to the
     insights it produced and its `reference_source`; match insights to it by text to tag
     each mode's `source`.
   - Otherwise use the distill report, or the experiment's annotations in the debug UI.
3. **The workflow's commands**: names, parameters, and what each returns (`<workflow>/_commands/`,
   or `what_can_i_do`). A check can only be phrased over what the workflow exposes.
4. **The domain policy**: whatever document states the rules the agent must follow (the
   workflow's policy text, its description, the business rules it was built from). Check
   every insight against it.
5. **Existing taxonomies** for this workflow (`<workflow>/_policy/*.yaml`). Reuse their mode
   ids and record overlap instead of inventing near-duplicates.
6. **What actually failed**, when available: the failing trials' grading (missed actions,
   failed assertions). This separates a causal insight from a side remark.

Read all of it before clustering. Clustering from the insight text alone is how wrong
insights become rules.

## Procedure

### 1. Read each insight as a claim about a defect

For each insight, note (in scratch, not output) the defect it describes, the commands
involved, and the evidence it cites. Then check four things. Each has turned a distilled
insight into a harmful rule before:

- **Task-specific content.** Record ids, amounts, dates, names ("cancel order #A123",
  "refund $41.20", "rebook flight XY12 on the 3rd"). The defect may be general even when the
  wording is not: keep the defect, strip the values. A mode never names one task's data.
- **One user's preference dressed as a workflow rule.** "This command cannot remove a single
  line item, so cancel the whole order" joins a workflow fact (the command cannot remove) to
  what one user chose (cancelling as their fallback). Only the fact is workflow-level.
  Compiled whole, the rule pushes every agent toward the destructive fallback. Such a rule
  has led an agent to cancel an order the user only wanted changed.
- **Contradiction with the domain policy.** An insight learnt from turn counts ("don't
  re-ask for confirmation the user already gave") can contradict a policy that demands
  explicit confirmation before every write. One learnt from a single run ("never tell the
  user an email will be sent") can contradict a policy that says one is. Exclude such
  insights, quoting the policy line.
- **Judge errors.** The judge can misattribute (blame the failing side for what the
  reference did), miscompute (a wrong total), or over-generalise. Where an insight cites
  something checkable, check it against the trace or the grading.

### 2. Group into modes by defect, not by command

Two insights belong together when the same check would catch both. Name the mode for the
defect (`entity-left-unprocessed`, `write-without-confirmation`), not the command
(`cancel-issues`). A planning insight and an execution insight about the same defect share
a mode; list both.

**Symptoms vs causes.** Some insights describe a symptom of another defect. "The answer
claimed a change that never ran" is the visible end of "a requested change was never
executed". If an existing mode's check already catches the symptom from the trajectory, put
the insight in that mode and say so in its notes. Don't create a second rule that matches
the answer's wording to catch the same thing.

Every insight lands in exactly one mode, including those you will not compile. An excluded
insight gets a mode with `compiled: false`, so the reason is on record. `check_taxonomy.py`
enforces this.

Watch for **collisions**: insights that push the agent in opposite directions ("prefer
modifying over cancelling" vs "for a removal, cancel instead"). Usually they cover disjoint
cases that only separate once the user's intent is known. Record the collision and how to
tell the cases apart; a rule set containing both without that distinction can deadlock the
agent.

### 3. Classify the signal each mode needs

`signal` says what a verifier must observe to decide the case:

| signal | decidable from | examples |
|---|---|---|
| `trajectory` | commands run and their outputs | argument outside an accepted set; a write before every candidate record was read; acting on a record whose status forbids it |
| `+intent` | trajectory plus what the user asked for (a model maps each message onto the workflow's declared operations) | "asked for *all* of X"; "asked to be told a figure" |
| `+user_text` | the user's literal words | a value the user stated |
| `+question-judge` | a model's reading of a question the agent is about to ask | "asks for what it could look up"; "repeats a question with nothing new" |
| `none` | nothing mechanical | semantic matching of descriptions; tone; ordering advice for a planner |

Prefer the weakest signal that decides the case. A `trajectory` check is reproducible; each
model-backed signal adds a failure mode and latency.

### 4. Decide compile / don't compile, and say why

Mark a mode `compiled: <rule id>` only if all of these hold:

- **Positive evidence.** The rule fires on something observed (a value present and wrong, a
  record read in one state and never acted on), never on the *absence* of an argument name
  or a phrasing. Agents spell the same argument many ways; absence-based rules misfire on
  correct calls.
- **Workflow-level.** The check holds for every task in the workflow. State it as a
  *principle* over the workflow's own vocabulary (its operations, the commands that perform
  them, each command's preconditions), never as the list of situations the insights happened
  to describe.
  - "Every request is carried out for every record it covers" is a principle.
  - "All pending orders were cancelled, and orders left at an old address were asked about"
    is two training tasks written down.
  - A list misses every other case and fires on look-alikes. If the insights only support
    the narrow version, compile the principle they are instances of, or don't compile.
- **Not already covered.** When another rule already enforces the principle behind an
  insight, don't compile a second, narrower one: point to the rule that covers it.
- **No contradiction** with the domain policy.
- **Outcome-relevant.** A failing trial lost reward to it, or the tool would reject the call
  anyway (pre-empting a doomed call is safe).

Otherwise mark it `compiled: false`, with `notes` that start with one of these reasons, then
the specifics:

- `contradicts-policy`
- `lexical-only`: detectable only by matching wording, so brittle.
- `no-deterministic-signal`
- `not-outcome-relevant`
- `needs-judge`
- `task-specific`
- `covered-by`: another rule enforces the same principle; name it.

Good advice that cannot be checked is prompt material, not verifier material: say so.

### 5. Record overlap and gaps

- `overlap`: the mode (and rule) in another taxonomy that covers the same defect. Recurrence
  across distillation sources is evidence the mode is real.
- `gaps`: a failure cause visible in the grading that no insight mentions. Record it; do not
  compile a rule for it. The table follows the corpus. A rule invented from your own reading
  of a trace is not a distilled rule, and mixing the two makes results uninterpretable.

### 6. Write the taxonomy and validate it

Write `<workflow>/_policy/taxonomy[_<corpus>].yaml` following
[taxonomy-schema.md](taxonomy-schema.md) (read it now if you have not). For an incremental
round:

- append new modes under a `# --- round N ---` comment;
- add the round to `corpus.rounds`, and update `coverage`;
- leave earlier modes alone unless a new insight changes them.

Then validate (the script needs PyYAML, which fastworkflow's dependencies install):

```bash
python fastworkflow/skills_for_coding_fastworkflows/insights-to-failure-modes/check_taxonomy.py \
  <taxonomy.yaml> <execution_agent_anti_patterns.md> <planning_agent_insights.md> \
  [--exec-prefix E --plan-prefix P] [--forbid REGEX ...]
```

(From an installed package, the skill folder is under
`python -c "import fastworkflow, os; print(os.path.dirname(fastworkflow.__file__))"`.)

It reports these problems:

- insights assigned to no mode, or to several;
- ids that do not exist;
- missing fields;
- compiled modes without a `check` or `hook`;
- uncompiled modes without a reason;
- task-data-looking literals in `check` fields. By default it flags long digit runs and
  currency amounts; pass `--forbid` with the workflow's own id shapes.

Fix everything it reports.

### 7. Hand-off summary

Finish with a short table for whoever compiles the rules: mode -> insights -> signal ->
compile decision -> overlap. Then list excluded insights with their one-line reasons, and the
gaps. The summary is what gets reviewed; the YAML is what gets compiled.

## Example

`fastworkflow/examples/retail_workflow/_policy/` holds a complete round:

- the corpus with its provenance (four cumulative rounds, self and pool distillation);
- `taxonomy.yaml` (37 modes over 117 insights, 13 compiled, the rest with their reasons);
- the `table.py` compiled from it.

Read it before starting and match its style. Apply it to *your* workflow's commands and
policy, not retail's.
