# Grading every stage of the pipeline, answer quality, and metrics

Companion to [SKILL.md](SKILL.md). A checklist grades the final answer. A lost line is only useful
once it is pinned to the **first stage that went wrong**, because that is the stage a fix has to
change. This page covers three things:

- each stage of a fastWorkflow turn: what to read in the trace, the cause tag for a loss there,
  and how to tell a stage's own failure from a failure it inherited;
- how to judge answer quality separately from task completion;
- the metrics to report and how to compute each one.

Span and attribute names below come from `fastworkflow/tracing.py` (`SPAN_CONTRACTS`, span contract
23) and `fastworkflow/skill_phases.py`. Check them against the installed version, because these
contracts are versioned. The full attribute catalogue is in the
[debug-workflow-conversations reference](../debug-workflow-conversations/reference.md#span-catalog).

## Contents

- [The order to walk a lost line](#the-order-to-walk-a-lost-line)
- [Stage by stage](#stage-by-stage)
- [Answer quality is not task completion](#answer-quality-is-not-task-completion)
- [Metrics](#metrics)

## The order to walk a lost line

```
request split / merge → skill match → plan (skill call or text) → phases (facts, when, fan-out)
   → executor (code steps, agent steps, paging, hand-offs) → finish check → answer
```

Start from the sub-task that should have produced the line. Walk it left to right, and stop at the
first stage whose **output** was wrong given **its input**. That stage is the cause. Everything
after it is a consequence: mention it in the comment, but do not tag it as a cause.

Two rules settle most disputes:

- **Judge each stage against what it was given, not against what the user wanted.** A planner that
  correctly planned a wrongly split sub-task did its job, so the cause is `split`. An agent that
  faithfully ran a wrong plan did its job, so the cause is `wrong/plan`.
- **Recovery cancels a cause.** A stage that went wrong but was repaired later in the turn, with
  the line still ✅, is not a cause. At most it is a waste note. The finish check is the exception:
  record its effect either way (`finish-note:*`), because that measures the check itself.

## Stage by stage

| Stage | Read in the trace | Typical tags |
|---|---|---|
| Split / merge | the planner call's `user_query`; the last phase's `agent_input`, or `agent_input` in unphased runs; the turn-level `fw.llm.call` (what was proposed) | `split` + `[split:dropped\|joined\|invented\|referent\|header]` |
| Skill match | `skill` on `fw.planner.plan` | `bad-match` |
| Plan | `plan_source`, `skill_args`, `skill_notes`, `skill_fallback`, `plan` | `wrong/plan`, `other:skill-arg:<p>`, `other:skill-gap`, `other:skill-fallback` |
| Phases | `fw.skill.phase`: `facts` / `facts_error`, `skipped`, `item_count` vs `items_total` | `other:phase-failed`, `other:phase-facts`, `other:phase-when`, `other:fan-out-cap` |
| Executor | `fw.agent.step` (`by_code`), `fw.command.execute`, `code_stop`, `exhausted` | `wrong/agent`, `wrong/argument`, `wrong/command-failed`, `budget/real-work`, `ask` |
| Finish check | the `finish` step's `observation` | `finish-note:*`, `other:finish-note` |
| Answer | `final_answer`, `answer_only`, `answer_by_code`, the merged `turns.answer` | `other:answer-phase`, `other:answer-by-code`, `other:premise-repeated`, `fabricated` |

### 1. Request split and merge

The splitter and its code merge (`fastworkflow/split_merge.py`, which is code only and makes no
model call) record no span of their own. You see them two ways:

- **What actually ran.** Each sub-task gets its own planner call. The `fw.llm.call` under each
  `fw.planner.plan` carries the sub-task text in its `module_input` as `user_query`. In an unphased
  run, the sub-task's `fw.agent.execute` `agent_input` starts with that text. In a phased run, it
  does not: every phase except the last gets only its own steps, framed as "This is part k of N
  (…) of a larger job" (`skill_phases.phase_query`). Only the **last** phase's `agent_input`
  starts with the sub-task. Judge the split from these sources, never from the splitter's raw
  output, because the code merge changes it.
- **What the model proposed** is the turn-level `fw.llm.call` (parent `fw.turn`) whose
  `module_input` carries the request and whose `module_output` carries the sub-task list. The
  module names are generic, so identify the call by its input and output fields.

| Loss | Tag |
|---|---|
| A job asked for is in no sub-task | `[cause:split] [split:dropped]` |
| Two independent jobs joined, so one was half done | `[split:joined]` |
| A sub-task asks for something the user never wrote (copied skill wording, invented values) | `[split:invented]` |
| A later sub-task's "that one" / "the riskier one" left unresolved, or written in as the wrong record | `[split:referent]` |
| A summary or "header" sentence made into its own job | `[split:header]` |
| A qualifier dropped ("first", "every", "exactly", "do not apply") | `[split:dropped]`, and name the qualifier |

An over-split that still got everything right is `[split:harmless]`. That is a note, not a cause.
The `split` field of the row records the overall verdict.

### 2. Skill match

The matched skill appears as `skill` on `fw.planner.plan`. It is null when no skill matched, and
always null on `fw.planner.replan` spans, so read it from the first plan. The matcher's own model
call is a turn-level `fw.llm.call`.

- **Wrong skill, faithful sub-task:** `bad-match`. Name both skills in the comment
  (`[skill:<picked>]`).
- **No skill where one fits, and the text plan then fails:** `bad-match`, noting "no skill".
- **No skill where none was needed:** not a cause.

When a new skill ships, tag every row that used it (`[skill:<name>]`). You can then measure both its
own tasks and where it was picked by mistake.

### 3. Planning

`fw.planner.plan` / `fw.planner.replan` carry:

- `plan_source`: `skill_call` (typed arguments, steps expanded by code), `text` (a model-written
  plan), `none`, or the historical values `structured` / `text_fallback`;
- `skill_args`: the typed arguments;
- `skill_notes`: arguments that a guard or verify verdict replaced, and extra steps that were
  dropped;
- `skill_fallback`: why a skill call fell back to the text planner;
- `plan`: the numbered step list;
- `replan_trigger` on replans.

The planner can add steps of its own around a skill (`steps_before` / `steps_after` in
`skill_commands.py`), and code filters them. Where they run (`skill_phases.py`):

- `steps_before` runs in a phase of its own before the first phase. Its span has `code_stop`
  `off`.
- `steps_after` goes after the last phase's steps.
- When the last phase is answer-only, extra command steps instead run in a phase of their own
  (`extra_steps_phase`) just before it, so that the last phase stays answer-only.

| Loss | Tag |
|---|---|
| Skill call with a wrong argument (record kind, a flag, a name cut short) | `[cause:wrong/plan] [other:skill-arg:<param>]` |
| The skill's expansion lacks a step the request needs | `[cause:wrong/plan] [other:skill-gap]` |
| A guard or verify step replaced a correct argument with a wrong one | `wrong/plan`, citing the `skill_notes` entry |
| A text plan missing or misordering a step | `wrong/plan` |
| A fallback to text that then failed | `[cause:wrong/plan] [other:skill-fallback]`, citing `skill_fallback` |

### 4. Skill phases

Each phase run is one `fw.skill.phase` span with:

- `skill`, `phase`, `phase_index` / `phase_count`;
- `steps`, `steps_used`, `step_budget` (`FW_SKILL_PHASE_MAX_STEPS`, default 15), `exhausted`;
- `record` (what its facts were filled on), and `facts` or `facts_error` (status `error` when the
  phase failed);
- `skipped` (its `when` was false);
- for a fan-out item: `item`, `item_index` of `item_count`, and `items_total` (more than
  `item_count` means the cap cut items: `max_items`, else `FW_SKILL_FANOUT_MAX_ITEMS`, default 10),
  plus `stop_condition_met` when `until` ended it;
- `answer_only` and `answer_by_code` for the last phase.

| Loss | Tag |
|---|---|
| A phase failed and stopped the job (`facts_error`). Find why: its commands, a missing source run | `[cause:other:phase-failed]`, naming the phase and the root cause |
| Facts filled from the wrong row or record, or left empty, so later phases worked on the wrong thing | `[other:phase-facts]` |
| A phase skipped (`skipped`) that the request needed: its `when` read the input wrongly | `wrong/plan` if a skill argument drove it; otherwise `other:phase-when` |
| Fan-out cut by the cap, so items were never visited | `[other:fan-out-cap]`, not budget |
| Phases ran fine, facts complete, but the answer phase wrote it poorly | `[other:answer-phase]` (see stage 7) |

A lost line inside a phase whose own commands were wrong goes to the executor's cause (stage 5),
with the phase named in `phase_cause`.

### 5. Executor: code steps, agent steps, paging, hand-offs

Each `fw.agent.step` has `step_index`, `thought`, `tool_name`, `tool_args`, `observation`, and
`by_code` (true when code ran the step without the model). Each `fw.command.execute` has
`raw_command`, `parameters`, `success` and `response_text`, plus the `command_name` / `context`
columns, which say what ran and where. A phase's `steps_by_code` and `code_stop` say how far code
got before handing over: `done`, `open step N`, `step N failed`, `no path for step N`, `more pages
at step N`, `stopped`, or `off`.

| Loss | Tag |
|---|---|
| Plan right, agent did something else, stopped early, or skipped a step | `wrong/agent` |
| Right command, wrong record, uid or option (a look-alike name, a miscopied uid) | `wrong/argument` |
| Stopped paging where "every" was asked (`code_stop` "more pages at step N", no further pages run) | `wrong/agent` on an `every` line |
| The right command errored and nothing recovered | `wrong/command-failed` |
| Out of steps doing real work (`exhausted`, or `failure_reason = max_iters_exhausted`) | `budget/real-work` |
| Out of steps through wandering (repeats, back-and-forth) | The cause of the waste, usually `wrong/agent` |
| Code ran a step exactly as planned where an earlier agent improvised a fix | `[other:plan-followed]` |
| Loss inside a flat (single-phase) skill run | `[other:flat-phase]`, saying whether code or the agent ran the failing step |
| Parameter extraction bound a wrong or cut value (`fw.nlu.param_extraction`) | `wrong/argument`, with the `extraction` field filled |
| Asked instead of acting (an `ask_user` refused in an unattended run, or an answer ending in a question) | `ask` |

**Budgets differ by run type.**

- **Unphased run:** the executor's own step cap applies. A sub-task that reaches it can be continued
  in a further `fw.agent.execute`, at most `MAX_HANDOFFS` (2) times per sub-task.
- **Phased run:** there is no hand-off. A phase that reaches its budget (`exhausted`) simply ends.

Either way, before tagging `budget/real-work`, classify where the steps went: navigation, failed
commands, repeats, or productive reads. A bigger step cap is a hypothesis to test, not a fix. In
one programme, raising the cap used no extra steps and recovered none of the missing rows.

**What the agent saw is in the step spans.** Each step's `observation`, together with any
observation offloading (where an output was stored aside and the agent got a pointer), is the
ground truth for what the agent saw. Prompt envelopes are not: they are capped and may be cut.
Before blaming the agent for missing a value, check that its observation actually contained it.

### 6. Finish check

When the agent picks finish, code checks that every command step of the run's plan ran
(`fastworkflow/observation_offloading/finish_check.py`). If one did not, the agent is sent back once
with a note that begins "Finish check: not every plan step is done yet." In the trace this is the
`observation` of the `fw.agent.step` whose `tool_name` is `finish`, followed by more steps.

Tag every row: `[finish-note:none]` when no note fired, `helped` when the agent ran what the note
named and that fixed a line, `ignored` when it finished again without doing it, and `harmful` when
the note sent it into a wrong context, a duplicate write, or the step cap. A harmful note that is
the first mistake is `[cause:other:finish-note]`.

**"Executed" is not "satisfied."** The check confirms that each planned command ran. It does not
confirm that the output held what the step asked for, or that the answer reported it. In one
review, 24 of 33 turns that the check certified as "every step executed" were judged incomplete.
Never read a quiet finish check as a passed line. To evaluate the check itself, measure two things
separately:

- its detection offline: precision and recall of its notes against graded outcomes, replayed over
  stored turns;
- its live effect: how often it fired, and how often its notes were heeded (`helped` / `ignored` /
  `harmful` counts).

False notes also happen. One example is an equivalent command, counted as "not run".

The check also exempts some steps, so gaps there are invisible to it: conditional steps ("only
when …", "if needed"), negated steps, answer steps ("present …", "say …") and navigation-only
steps (`finish_check.py`). Grade those lines from the trace yourself.

### 7. Answer generation

There are three kinds of answer, and each has its own failures:

- **Model-written** (the agent's `final_answer`, or an answer-only phase with `answer_by_code` false).
  Failures: fabrication, copying errors (a dropped digit, a merged name), a count reported as the
  list, a value read from the wrong field.
- **Code-written** (`answer_by_code` true). Literal, one line per step, possibly long. Grade the
  content, not the style. A loss means the step text did not carry the fact:
  `[other:answer-by-code]`, saying what the step text lacked.
- **Merged** (after a split, the final `turns.answer` joins the sub-task answers). Check it against
  each `fw.agent.execute`'s `final_answer`. Anything dropped or invented in the merge is its own
  loss.

A loss here is a cause only when every stage before it was right: the facts or outputs held the
information, and the answer lost or bent it.

### Evaluating a stage offline

At runtime there is no reference for a stage's output. Offline there can be. Before a live stage,
replay the cheap stages over stored inputs with reference outputs:

- **Splitter:** score each benchmark prompt's split by failure category: over-split, under-split,
  dropped item, invented item, name changed or truncated, a dependent part split off, copied
  description, background sentence made into a job. Check names with word boundaries, because an
  exact-substring check misses a name cut short at its end. Score the production path faithfully,
  including the code merge and any single-part shortcut.
- **Matcher:** set a bar on faithfully split sub-tasks, such as "no confident wrong skill", and
  watch matches near the threshold. A skill-description edit moves both matching and splitting.
- **Planner:** score plans in code against reference plans: needed steps kept, "when asked" steps
  only when asked, request values used, inapplicable branches dropped.
- **Finish check / answer checks:** replay over stored turns and measure precision and recall
  against graded outcomes. Declare the bar in advance.

All of these are LLM-noisy at temperature 0. Repetitions and cache checks are covered in
[judge-changes-against-noise](../judge-changes-against-noise/SKILL.md#6-offline-component-replays-useful-and-easy-to-fool-yourself-with).
A template written after reading the reference plans contaminates the comparison: rerun the
baseline under the same conditions.

## Answer quality is not task completion

A run can complete every step and still answer wrongly. It can also answer everything and be
wrong on some of it. Grade both dimensions, line by line:

| Dimension | Question | Rule |
|---|---|---|
| **Accuracy** | Is every claim in the answer in some output? | Every name, id, count and value traced to a span. Not found: 🚩. Found but misread: ⚠️ |
| **Coverage** | Does the answer name everything the line asks for? | "Every" lines need the full set or an honest "K of N". A first page passed off as all of them is ◐, and ⚠️ if a total is then stated |
| **Count vs list** | Asked for a list, it gave a count (or the reverse)? | ◐, since one half of the line is missing |
| **Completeness of a read** | Did the commands read everything the answer claims to cover? | A "none" or a total over a partial read is ⚠️, plus `fabricated:yes` |
| **Made-up facts** | Anything stated that no command returned? | 🚩, plus `fabricated:yes`, whatever the first cause was |
| **Honest gaps** | Says plainly what is unavailable ("no owner recorded")? | ✅ when an output backs the absence |

Coverage without accuracy is the trap: an answer that names every record can still be wrong about
each one. In one re-score, a "names present" metric reproduced at 57 of 59, but accuracy on the same
answers was 0 of 59, and the ranking of the compared arms flipped at the top. Check the claims, not
the count of names.

Several more distinctions are worth recording:

- **Retrieval loss vs writer loss.** Was the missing value present in the run's own outputs? If it
  was, the answer step lost it: a count given instead of the requested list, a pointer such as "see
  the output above", or "unavailable" for a query that never ran. If it was not, a stage before the
  answer lost it. This one test splits most losses cleanly.
- **Repeated premise, flagged premise, invented value.** When the prompt carries a false claim,
  repeating it is deference, not fabrication: tag the repeat `other:premise-repeated`. Flagging
  the false claim is better than asked, so credit it in `judgment_calls`. Inventing a value is 🚩.
- **Claims bounded by what ran.** "All 40 members checked" with one member-opening command in the
  trace is ⚠️ and `fabricated: yes`, even when no output contradicts it.
- **Cut by the harness, not the agent.** An answer cut at the model's `max_tokens` (the
  `fw.llm.call` output stops at the limit) is censored. Exclude it from quality ratings and count it
  separately.

"Ran to completion" (`turns.status`), command success (`turns.success`) and the runner's derived
outcome measure execution. Report them, but never as quality.

## Metrics

Compute everything from the validated grades JSON and the store. Never use graders' prose.

| Metric | How |
|---|---|
| **Lines per tier per run** | Sum of credit (✅ = 1, ◐ = ½) over the tier's rows for that run; report against the frozen line total |
| **Mean over runs** | Mean of the per-run tier totals, with the per-run values beside it (the spread is your noise) |
| **Pass count / pass rate** | Rows with `pass: true`, per tier per run |
| **pass@1** | Passing rows ÷ all run rows of the tier. The store's `experiment_scores()` computes it from recorded attempt outcomes over the declared denominator. When each run is its own experiment, compute it from the grades |
| **pass^k** | With k = all runs: tasks whose every run passed ÷ tasks. That is what `experiment_scores()` computes, over the declared tasks. For k below the run count, use the unbiased estimator below. Report it beside pass@1; the gap between them is unreliability |
| **Consistency** | Per task: same pass/fail in every run, or not. List the tasks that flip. The summary rows' `consistency` gives the reason |
| **Made-up count** | Rows with `fabricated: yes`, per tier per run |
| **Lines lost per cause** | For each non-✅ line: 1 (or ½ for ◐) to the cause that explains its item. Group by cause and by task. Report the number of distinct tasks per cause too, because 8 tasks beat 1 task × 8 |
| **Lines lost per stage** | The same, grouped by pipeline stage (split / match / plan / phase / executor / finish / answer) |
| **Lines lost per plan source** | The same, split by the turn's `plan_source` (`skill_call` vs `text`, or mixed). This shows whether losses sit in typed skill calls or in model-written plans. The losses script in [reference.md](reference.md#losses-by-cause-and-stage) computes all three |
| **Target-command rate** (single-command tiers) | Rows with `target_ran: yes` ÷ rows. Kept separate from the score, so a relaxed scoring rule does not hide routing drift |
| **Skill share** | Per new skill: the rows that used it, its tasks' score, and where it was picked by mistake (`[skill:<name>]` on rows outside its tasks) |
| **LLM calls / tokens / cost per job** | `fastworkflow.observability.comparison.usage_rollup(store.get_spans(turn_key))`. It returns `calls`, `tokens` (prompt / completion / total, plus how many calls recorded none) and `cost`. It folds re-emitted records and wrapper calls, so do not count `fw.llm.call` rows yourself: one call can appear more than once. Missing usage is unknown, not zero |
| **Steps used / steps by code** | Sum `steps_used` and `steps_by_code` over the turn's `fw.skill.phase` spans (model steps = used − by code); for unphased runs, count `fw.agent.step` spans and their `by_code` |
| **Where code stopped** | Distribution of `code_stop` values across phases |
| **Time per job** | `completed_at − started_at − suspended_ms` from `turns`. Leave out stalled first jobs, or say they are in. Concurrent lanes make times comparable only within one stage design |

Rules for these numbers:

- **Put intervals on pass rates.** A task set of a few dozen gives wide intervals. Report a
  confidence interval (for example a bootstrap over tasks) next to pass@1 and pass^k, and do not
  rank configurations whose intervals overlap heavily.
- **pass^k with an unbiased estimator.** For a task with n runs of which c passed, use
  C(c, k) / C(n, k) and average it over tasks. The naive (c/n)^k overstates it: for n = 5, c = 3,
  k = 2 the naive value is 0.36 and the unbiased one is 0.30. Compare the decay ratio
  pass^k / pass@1 between stages, so that a higher pass@1 cannot hide a steeper decay.
- **Unscored is null, never 0 or false.** A metric with a zero denominator is null. Coercing "not
  scored" to "failed" makes an unrated set look like a bad one.
- **Two scorecards.** Process (routing, steps, code share, paging, exhaustion, calls, cost, time)
  and outcome (lines, passes, made-up, premise handling) are reported separately. A change can win
  one and not the other, so say which.
- **Never blend axes that trade off**: abstention vs fabrication, coverage vs accuracy, process vs
  answer. Enforcing "never ask" once moved every traded failure into fabrication and budget
  exhaustion. Fix floors for each axis before the treatment runs.
- **Measure the mechanism as well as the outcome.** For a change that adds an intervention (a check,
  a note, a guard), report how often it fired, how often it was heeded, and whether it fired where
  it should. In one stage the finish check fired on 21 of 60 turns and was ignored on 11. A rule
  that never fired says nothing about its value.
- **Blame by stage, validated.** Before relying on an automatic stage classifier, run it on correct
  rows and check that it finds the values present there.
- **Audit simulated users like judges.** If an LLM answers the agent's questions (an `ask_user`
  responder or a user simulator), it is part of the grader's environment. One such operator never
  supplied a correct novel fact, and its refusals licensed the agent to improvise. Count what its
  replies authorised, not only what they asserted. Prefer a constant, scoped reply, or unattended
  runs.
- **Consistency is descriptive.** It is not correctness. With one run per task, report
  "insufficient repeats", not "consistent".

Report per tier per run, the mean, and the comparison with the previous stage. Re-score the
previous stage under the current rules first; see
[judge-changes-against-noise](../judge-changes-against-noise/SKILL.md).
