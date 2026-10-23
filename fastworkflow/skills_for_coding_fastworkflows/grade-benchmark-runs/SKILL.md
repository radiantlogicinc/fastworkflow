---
name: grade-benchmark-runs
description: >-
  Grade recorded benchmark runs of a fastWorkflow workflow with workflow-specific checklists
  marked by LLM grader agents: check the run is valid first, write and freeze one checklist per
  task, mark each line against command outputs with span-id evidence, tag the first cause of
  every lost line, dump traces read-only from the observability store, brief and fan out grader
  agents, then validate and total their JSON. Use when an experiment has finished and you need
  scores you can compare across stages, when runtime success or the derived outcome is being
  read as task success, or when setting up grading for a new benchmark tier. Not for judging
  whether a change helped (judge-changes-against-noise), diagnosing one failing conversation
  (debug-workflow-conversations), or writing the benchmark tasks themselves
  (build-task-benchmarks, create-workflow-benchmarks).
---

# Grading benchmark runs with checklists

A finished experiment tells you that attempts ran. It does not tell you whether the user got what
they asked for. The runner's fallback verdict (`derived_outcome` in
`fastworkflow.experiment.runner`, recorded with `outcome_source='derived'`) reports only whether
some command returned a failure code. `turns.status = completed` and `turns.success = 1` are not
grades either. A run that answers the wrong question with clean commands passes all three.

A grade here is a workflow-specific checklist per task. An LLM grader agent that has read a
written brief marks it line by line from the recorded trace. In return you get a score that
compares across stages, a cause for every lost line, and evidence a reviewer can check.

Turning two stages of grades into a keep-or-revert decision is covered by
[judge-changes-against-noise](../judge-changes-against-noise/SKILL.md). The other files of this
skill:

- [pipeline-and-metrics.md](pipeline-and-metrics.md): attributing each loss to a pipeline stage,
  judging answer quality, and computing every metric.
- [reference.md](reference.md): the vocabulary, templates, the JSON schema, a worked example,
  grader reliability, and four read-only scripts.

## 0. Is the run even worth grading?

Graders grade whatever they are shown. A stage once ran entirely against a dead backend. Its
traces looked like an agent failing politely, and its numbers meant nothing. Check health first.

| Check | Where |
|---|---|
| Status is `complete` (or `capture_complete` / `awaiting_evaluation` when your runner captures externally or defers grading), never `invalid` or `running` | `store.get_experiment(id)`: `status`, `invalid_reason`, `invalid_detail` (the detail is only on this call) |
| Rows = `declared_tasks × declared_attempts`, each with an `execution_status` | `experiment_attempt_rows(id)` |
| The first message of each attempt equals the task's prompt | `experiment_first_messages(id)` vs the benchmark version |
| Transport and provider errors | `turns.failure_reason` holds `"<ExceptionType>: <first line>"` (a transport error, `max_iters_exhausted`); runner logs for connection refused, 5xx, timeouts |
| No dropped evidence for these turns | `writer_health()` folds every writer of the whole store. Read `spans_dropped` / `records_dropped` and intersect `spans_dropped_turn_keys` / `records_dropped_turn_keys` with this experiment's turns |
| Same model routes, cache policy and unattended setting in every lane | the experiment description, and each attempt's `runtime_snapshot_json` |
| State-changing tasks started from the documented baseline | your restore log for the round |

Open the store only with `ReadOnlyObservabilityStore` (see
[debug-workflow-conversations](../debug-workflow-conversations/SKILL.md)). If a lane hit a dead
dependency, rerun it; do not grade it and footnote it. Concurrent lanes and a slow first load make
timing figures non-comparable. Content grades are unaffected.

## 1. Write the checklists before reading any trace

- **One line per thing asked and one line per limit** ("apply nothing", "only the first loan",
  "exactly Ann Lee, not Ann Leigh"). Each line is a fact the answer must contain or an action that
  must have happened.
- **Size to the tier.** A single-command task gets one line: "the target command ran on the named
  record and the answer reports its output correctly". A multi-ask task gets 3 to 8 lines. A long
  session can have 20 or more.
- **Navigation never scores** (find-then-open, context switches, go-up, reset), unless the
  checklist names it. Waste on a passing turn is a note, not a cause.
- **Write the edge cases into the line**: "lists both, does not ask or pick"; "says plainly that
  none is recorded".
- **Write tasks** check that the write landed on the named record, with its own success text, from
  a baseline that is restored before every round.
- **Owner-approve, then freeze.** Lines and line boundaries stay identical in every later stage.

Audit the tasks before freezing:

- **Verify the ground truth independently, including claims built into the prompt.** One prompt
  said "five people hold both rights", but two did; the authoring check had tested "holds one".
  Prove a ground-truth read complete: reconcile it with an independent total, and re-read at
  another page size, expecting the same id set.
- **Leakage and unscoreable items.** A prompt that hands over the identifier the check grounds on
  measures nothing. An undecidable line is marked `n/a`: it stays in the checklist, out of the
  score and out of the denominator, and is reported beside the score.
- **State outcome facts, not routes** ("gives every loan's due date", not "list_loans ran"). In
  single-command tiers, where the route is the point, tag an alternative route
  `[other:non-target:<cmd>]` and let an owner's rule decide its credit.
- **An abstention is never a success.** Ending with a question, handing the task back, or claiming
  it cannot be done loses every line not delivered.

## 2. Marks, score, pass

| Mark | Meaning |
|---|---|
| ✅ | Done and correct. Every name, id, count and value appears in a command output, and the evidence cites that output's full span id |
| ◐ | Partial, half credit: a first page passed off as "every", one of two asks, one side of a comparison |
| ⚠️ | Wrong record, wrong argument, or a misread value that does appear in some output |
| ❌ | Never done, and absent from the answer |
| 🚩 | Made up: stated, but in no output, including "done" for a write that never succeeded |
| ⛔ | A limit broken: an unrequested write, or the prompt's own limit ignored |
| `n/a` | Unscoreable by construction. Not counted |

Score = (✅ + ½◐) ÷ scored lines. Pass = every scored line ✅. Set `fabricated: yes` whenever any
line is 🚩, and also when a total is stated over a partial read (that line is ⚠️). A ✅ without a
cited span is the grader's opinion. Grade the final answer (`turns.answer`). After a split, also
check that the merge dropped and invented nothing.

## 3. Causes: the first mistake, tagged

Mistakes snowball. A wrong command wastes steps, the budget runs out, and the answer gets made up.
Only the first mistake is the cause. Walk each lost line's sub-task in time order: split → skill
match → plan → phases → executor → finish check → answer. Judge each stage against **its own
input**: a planner that faithfully planned a badly split sub-task is not at fault. A stage that went
wrong but was recovered, with the line still ✅, is not a cause.

The cause vocabulary is closed. The core causes are:

- `split`
- `bad-match`
- `wrong/plan`
- `wrong/agent`
- `wrong/argument`
- `wrong/command-failed`
- `budget/real-work`
- `ask`
- `fabricated`
- `unsure a|b`
- `other:<name>`, from a fixed list of framework-shaped names such as `phase-failed`,
  `phase-facts`, `answer-by-code`, `skill-arg:<p>` and `fan-out-cap`.

The canonical list, with definitions, is in [reference.md](reference.md#cause-and-tag-vocabulary).
The brief and the checker use the same list. Three of the same new `other:` in one stage earn a
place in the list.

Write each `[cause:...]` immediately followed by every `[item:x]` it explains:
`[grade 3/5] [fabricated:no] [cause:split] [split:dropped] [item:b] [cause:wrong/agent] [item:d] [finish-note:ignored]`.
Always record `[fabricated:yes|no]`. Always record the finish check's effect,
`[finish-note:helped|ignored|harmful|none]`, because that measures the check itself.

What each stage looks like in the trace (span attributes, stage-specific tags, offline checks) is
in [pipeline-and-metrics.md](pipeline-and-metrics.md#stage-by-stage).

**Rules will change.** Keep the tag whatever the current rule awards. A later rule then becomes a
re-score in code, not a re-grade: see
[judge-changes-against-noise §5](../judge-changes-against-noise/SKILL.md#5-scoring-rules-change-re-score-never-re-grade).
Graders flag rule conflicts in `judgment_calls`; the owner settles them.

## 4. Give graders a readable dump per attempt

Build one markdown file per (task, attempt), read-only, from the store:

- **First line:** the task label, the attempt and the exact `turn_key`. Graders copy keys from
  that line; keys retyped by hand come back wrong (one stage got 11 wrong).
- The user message, the final answer, then the trace in time order:
  - turn status, `failure_reason` and `unattended`;
  - the turn-level `fw.llm.call` spans (the splitter, the matcher and other request-level calls);
  - each `fw.planner.plan` (`plan_source`, `skill`, `skill_args`, `skill_notes`, `plan`);
  - each `fw.skill.phase` (`steps`, `steps_used`, `steps_by_code`, `code_stop`, `facts` or
    `facts_error`, fan-out item fields, `answer_by_code`);
  - each `fw.agent.execute` (`agent_input`, `final_answer`);
  - each `fw.agent.step` (`by_code`, tool, arguments, and the finish-check observation);
  - every `fw.command.execute` (`raw_command`, the `command_name` / `context` columns,
    `parameters`, `success`, `response_text`).
- Cap long values (about 8000 characters) with a note naming the span that holds the full value.
  Show the store's own truncation envelope too (`original_length` is in bytes), because the store
  may not hold the full text either.

A tested script is in [reference.md](reference.md#trace-dump-script). Keep it in your app's dev
tooling, not in the workflow folder.

## 5. Brief, addendum, fan-out

- **The brief** is the whole method in one document: what the workflow is and how it was run,
  what to read in which order, marks, the vocabulary, judgment calls made so far, the JSON schema,
  and hard limits. Graders open the store read-only only. They write exactly one grades file, with
  no helper scripts and no temp files. They never run the agent or touch the backend.
- **The addendum** comes with each stage. It gives the dumps' location, which earlier stage to
  compare with, and *what changed in the system*, described as it will look in traces (new skills,
  code-written answers, new refusals). It also lists new tags and dated rule changes. Without it,
  deliberate new behaviour gets marked as mistakes. Consolidate the chain into a new brief after
  about five addenda.
- **Fan out** one grader agent per group, with your coding agent's parallel sub-agent mechanism.
  Sizes: about 15 single-command tasks, 5 to 7 multi-ask tasks, or 1 to 3 long sessions, each with
  all its runs. **Every run of a task goes to the same grader**, so its summary row can say which
  failures repeat. A 500-job stage took 18 graders about 10 minutes and a few million tokens. Get
  approval for the cost before launching. Templates are in [reference.md](reference.md#brief-template).

**Trust the grader only after testing it.** A single LLM judge measures that judge: two strong
judges agreed on only about half of identical ratings. Prove the brief and checker on negative
fixtures that must fail. Blind graders to system verdicts. Where a grade decides something, add a
second judge or a human reviewer, and report the judges separately. Details are in
[reference.md](reference.md#grader-reliability).

## 6. Validate, total, spot-check

Graders' prose truncates and drifts. **Totals come only from the JSON.** The checker in
[reference.md](reference.md#validate-and-total) verifies:

1. Every file parses, and each (task, run) appears exactly once, plus one summary row per task.
2. Line counts equal the frozen checklist. Scores agree with the marks. `pass` matches the marks.
   `fabricated` is yes or no, and yes when there is 🚩. `finish_note` comes from its list.
3. Causes and tags come from the closed vocabulary.
4. Each `turn_key` exists in the store and belongs to the run's experiment and the row's task.
5. Each ✅ cites a full 32-hex span id. This is a heuristic: record uids of the same shape also
   match. On one real stage, 436 of 504 rows had ✅ lines citing only shortened ids or none.

Then **spot-check**: open the cited spans of a few rows per grader. Report the metrics listed in
[pipeline-and-metrics.md](pipeline-and-metrics.md#metrics). Lines lost per cause, stage and
`plan_source` come from the losses script in [reference.md](reference.md#losses-by-cause-and-stage).

## 7. Keep the record

Store the grade files with the app, versioned, not only in a scratch directory: later stages are
compared and re-scored against them. Optional mirrors:

- **Turn comments.** Post a turn comment as human feedback with `provenance: "coding_agent"`
  (never `"human"`), for example category `conclusions`, subcategory `what_went_wrong`. The route
  is in the [debug-workflow-conversations reference](../debug-workflow-conversations/reference.md#human-feedback).
  Do this only with authorization to write to the live store.
- **Attempt outcomes.** When the runner deferred grading (`awaiting_evaluation`), the verdict can be
  recorded once per attempt with `record_attempt_outcome(..., outcome="pass"|"fail",
  outcome_source="<grader id>", reward=<credit ÷ scored lines>)`. This is write-once. A derived
  outcome already recorded stays; never present it as a grade.

## Anti-patterns

- Grading a run with a dead dependency, or before its health is checked.
- A checklist written after reading the trace, or line boundaries that drift between stages.
- ✅ without evidence. "The agent said it did it" counted as done. An abstention scored as success.
- Tagging every bad thing in a trace as a cause, or tags outside the vocabulary.
- One task's runs split across graders. Totals taken from prose.
- One LLM judge's score deciding something with no second judge, fixtures or human check.

## Checklist

```
- [ ] Run health: status, rows = tasks × attempts, first messages, failure_reason, dropped turn keys
- [ ] Checklists from prompts, ground truth verified, n/a items declared, owner-approved, frozen
- [ ] Brief with the closed vocabulary and schema; addendum states what changed and the comparison
- [ ] Dumps written read-only, turn key on the first line, long values capped with span pointers
- [ ] Grader tested on negative fixtures; second judge or human review where grades decide
- [ ] JSON validated (counts, marks, vocabulary, turn keys, span ids), spot-checked, totalled
- [ ] Losses by cause, stage and plan_source; metrics reported; grades stored with the app
```

## Related

- Keep-or-revert decisions, noise, re-scoring and stage reports:
  [judge-changes-against-noise](../judge-changes-against-noise/SKILL.md).
- One trace in depth: [debug-workflow-conversations](../debug-workflow-conversations/SKILL.md).
- Benchmarks, versions, experiments and the runner:
  [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md).
- Designing tiers and long tasks: [build-task-benchmarks](../build-task-benchmarks/SKILL.md).
- Controlled comparisons and feedback:
  [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md).
