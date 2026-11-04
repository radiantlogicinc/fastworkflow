---
name: distill-policy-verifiers
description: >-
  Turn a workflow agent's recorded failures into deterministic policy verifiers: record trials,
  distil failing trials against passing ones (fastworkflow distill: teacher/student, --self or
  --pool-students), cluster the insights, compile a policy table, audit it with fastworkflow
  policy replay, ship it as <workflow>/_policy/table.py (which turns it on), and compare it
  with a control arm. Use when asked to make distilled insights enforceable, to build or extend a
  workflow's policy table, to add or debug a verifier rule, or to run the
  distill-verify-evaluate loop for any workflow.
---

# Distil policy verifiers

Distillation writes insights as prose, and prose in a prompt is advice the agent may ignore.
A **policy verifier** is a rule the framework checks while the agent works. A rule fires:

- **before a command runs**, on its real name and validated parameters;
- **before a question goes to the user**;
- **before the answer is given**.

When a rule fires, the agent gets the rule's text as the step's observation instead of the
step's result, and it carries on from there. The rules are a versioned table of predicates
over the conversation, so every decision names the rule and table version that produced it.
Where a rule needs to know what the user asked for, a model reads the user's words into the
workflow's own operation names. The rule still decides.

The engine is generic (`fastworkflow.policy_decision`, `policy_runtime`, `policy_intent`).
The rules are workflow content. A complete worked example is in
`fastworkflow/examples/retail_workflow/_policy/`; read it first.

## The loop

### 1. Record trials

- Run the workflow's benchmark as an experiment with observability on, several trials per
  task, and the provider response cache off (`FW_LM_CACHE=0`; otherwise repeated trials are
  identical).
- Every attempt needs a graded outcome (`outcome` pass/fail, or a `reward`): distillation
  pairs failing trials with passing ones.
- Start every run under a new experiment id. A resumed id mixes trials from different code.

**Set hold-out tasks aside now.** Before the first distillation, draw a random share of the
tasks (say a fifth, with a recorded seed) and write the list to `<workflow>/_policy/holdout.txt`.
Hold-out tasks are recorded and graded like the rest, but:

- they never appear in a distillation pair;
- they are never used to shape a rule: not its predicate, not its rewrite text, not the
  wording of `FACTS.intent_operations`;
- they are where a rule is checked: the replay audit (step 6) and the control comparison
  (step 8) report hold-out tasks separately.

A rule checked only on the tasks it was distilled from proves only that it fits them. When you
tune a rule on a task (for example, sharpening an operation's meaning after the reader misread
a request), that task becomes a training task: check the change on hold-out tasks before you
keep it.

### 2. Distil

Pick the mode by where passing trials exist:

| command | compares | use when |
|---|---|---|
| `fastworkflow distill <wf> --teacher A --student B` | each task's student attempt with the teacher's | a stronger model solves the tasks |
| `... --self RUN` | each failing trial with the first passing trial of the same task, same run | the model sometimes passes |
| `... --pool-students R1 R2 --pool-teachers T1` | each failing student trial with a passing student trial, else a passing teacher trial | mixed; the default for a real round |

Add `--provenance FILE` (pair -> insights -> reference source) and, for pools,
`--max-pairs-per-task N`. Later pairs of a task mostly repeat earlier insights.

**Distil a random subset, not every failing task.** When many tasks fail, a full pass costs a
judge call per pair (there is no dry run), and most pairs repeat what earlier ones found.
Instead:

1. From the failing, non-hold-out tasks that have a passing reference, draw a random subset
   (5 to 15 tasks, recorded seed). Pass each as `--task`, with `--max-pairs-per-task 1`.
   Write the drawn list into the round's corpus folder.
2. Run the round, cluster, compile, test and replay (steps 3 to 6).
3. Run the benchmark again with the table on (or replay it, if no live run is possible).
4. If tasks still fail, draw another subset from the tasks not yet distilled, or combine a
   new draw with the tasks that still fail, and repeat from 2.

Stop when the failing tasks pass, or when a round yields no new compilable mode (its insights
all land in existing modes or in uncompiled ones). Then the remaining losses need prompt work,
a judge or an engine hook, not more pairs. "All pass" on distilled tasks is a fit; the
hold-out tasks say whether it generalises.

The judge sees every insight already in the Insights folder and skips duplicates, so rounds
build on each other.

Where the output goes:

- Insights are appended to `<workflow>/Insights/<workflow>/`.
- The judge's notes are written as annotations on the failing trials. They are visible in
  the debug UI (`fastworkflow run_chatbot`).
- Snapshot the insight files into `<workflow>/_policy/corpus/` (one folder per round) before
  the next round appends to them.

### 3. Cluster

Use [insights-to-failure-modes](../insights-to-failure-modes/SKILL.md). Output:
`<workflow>/_policy/taxonomy.yaml`, where every insight is in exactly one mode and every mode
is compiled or carries a reason it was not.

### 4. Compile the table

Write the candidate as `<workflow>/_policy/table_next.py`, exposing `TABLE` and `FACTS`.
Not `table.py` yet: that name is the live table, enforced as soon as it exists (step 7).
[table-authoring.md](table-authoring.md) has the API and a skeleton. Rules that have held up:

- **One principle per row, never a list of situations.** "Every request is carried out for
  every record it covers", not the two cases the training tasks showed.
- **Positive evidence only.** Fire on a value that is present and wrong, or a record read
  and never acted on. Never on a missing argument name or a phrasing.
- **Facts are declared, not inferred.** The table says which commands write, what each
  operation's command is, what state each command requires, and which values are accepted.
  Predicates read those facts and the trajectory.
- **Requested work is two questions.** First, does this request need a command at all
  (operations with no command, fallbacks and questions do not)? Then, has that command run
  successfully on every record the request covers?
- **Never read targets from the intent reader's free text.** The intent's `detail` says
  where things come from as often as what to change.
- **The rewrite says what to do.** Name the commands and records, say why, say what to do
  next. "Read every record first" is weak; "records R2, R3 are not read yet; call get_record
  on each" works.
- **Exempt the honest cases.** A finish rule must not fire on an answer that is waiting for
  the user ("shall I go ahead?"). A write rule must not fire on a write the user explicitly
  approved.
- **Never contradict the domain policy, and never encode one user's preference.**
- **Order matters (first match wins).** Calls that cannot work go first, then pointless ones,
  then premature ones, then finish checks. Bump `TABLE_VERSION` on every change.

### 5. Test it on the trials it came from

For each row, add a test built from the recorded failing trial: the row fires at the step
where the trial went wrong. Add one built from a passing trial: the row stays silent. Tests
are plain Python over `CommandRecord` lists; see the example's
`tests/test_retail_policy_example.py`.

### 6. Audit it on whole recorded runs

```bash
fastworkflow policy replay <workflow> --experiment <id> --table _policy/table_next.py [--live]
```

The replay drives the same policy object as a live run, with the same valve and finish
bounds, over every recorded step.

- **Required:** zero replayed blocks on passing attempts. Every block on a failing attempt
  must point at a real mistake. Read each one.
- Report hold-out tasks separately. Blocks on hold-out passes are the false alarms that matter
  most, because nothing in the table was fitted to those tasks.
- **`--live`:** re-reads intent and judges questions with the policy model. Without it, a
  rule that needs intent sees only what the run recorded.
- **Marks:** `LIFTED` is a block the recording made that the new table would not.
  `ADDED` is a new block.

### 7. Turn it on

Rename the audited candidate to `<workflow>/_policy/table.py`. There is no setting: a
workflow with that file enforces it, and one without it is the unchanged agent. A table
file that is malformed fails at startup, so rules are never skipped silently.

The only optional setting is the model for the readings: `LLM_POLICY_INTENT` (with
`LITELLM_API_KEY_POLICY_INTENT`), else `LLM_AGENT`.

The guard checks only steps the model chose:

- a command the user typed is not checked, and neither is a command a skill's code runs;
- assistant mode and MCP direct actions are not checked.

### 8. Evaluate against a control

- Run the same held-out tasks, same models and same trial count in two arms: the workflow
  without `_policy/table.py` (control) and with it. Use two copies of the workflow folder
  rather than adding and removing the file between runs. Compare pass rates.
- Read every fire in the debug UI: a step a rule rewrote shows a **Policy** section with the
  rule, table version and what the intent reader saw. Count right fires and wrong fires.
- Report the table version and the arms. A verifier is a disclosed change to the agent, not
  to the benchmark.

### 9. Iterate

Distil the enforced run's failures as the next round, against a new corpus folder, drawing a
new random subset as in step 2 and never from the hold-out tasks. Extend the taxonomy under
`# --- round N ---`, bump the table version, and replay the new table on the old runs as well
as the new ones.

## When a rule misbehaves

- **It fires on correct behaviour.** The predicate reads the wrong evidence. Find the step in
  `policy replay` output, write a failing test from it, fix the predicate. Do not raise the
  valve.
- **The agent loops on a block.** The rewrite asks for something the agent cannot do, or the
  rule cannot be satisfied in this situation. The engine lets a step through after two blocks
  by the same rule, and pushes a final answer back at most three times per turn (only after
  progress). Treat hitting either bound as a defect in the rule.
- **A command's failure counted as success.** The tool reports errors as text. Set
  `FACTS.failure_pattern`.
- **Intent rules are silent.** Check that `FACTS.intent_operations` is set and that its
  descriptions separate look-alike operations. A trace step shows `policy_requests`.
- **A slow turn.** Each user message costs one intent reading (25 s limit, fails open).
  Question judging is off unless `FACTS.judge_questions` is set.
