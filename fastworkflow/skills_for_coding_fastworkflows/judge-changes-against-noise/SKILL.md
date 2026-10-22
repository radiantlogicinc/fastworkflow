---
name: judge-changes-against-noise
description: >-
  Decide whether changes to a fastWorkflow workflow or the framework actually helped, from
  checklist grades of repeated benchmark runs: design multi-run stages, measure each change on
  the tasks it targets against run-to-run noise, keep or revert it, land changes as separate
  commits so each can be reverted alone, re-apply changed scoring rules to stored grades from
  their tags, use offline component replays without fooling yourself (LLM non-determinism at
  temperature 0, a cache that is not really off, scorer blind spots), and report a stage. Use after
  a graded stage when choosing what to keep, when planning the next stage's runs, or when one
  run's numbers look like a regression or a win. Not for producing the grades
  (grade-benchmark-runs) or for diagnosing one failure (debug-workflow-conversations).
---

# Judging changes against noise

An LLM workflow does not give the same answer twice, even at temperature 0. One run per stage
measures the change plus a draw of noise that can be just as large. In one programme, a
single-run stage looked like a regression and was not. In another, a change whose offline gain
looked clear made no difference live. This skill turns graded stages into keep-or-revert
decisions that hold up.

It assumes you already have validated checklist grades per (task, run) from
[grade-benchmark-runs](../grade-benchmark-runs/SKILL.md). For general comparison hygiene (frozen
inputs, matched configuration, experiment identities, comparison refusals), follow
[optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md). This page adds what
repeated LLM runs require on top of that. Code and templates are in [reference.md](reference.md).

## 1. Design the stage before running it

- **Three runs per task, minimum.** Run each round as its own set of experiments on the same
  benchmark versions, or as one experiment with `runs_per_task` = 3. Graders need all of a task's
  runs together either way.
- **One configuration per stage, pinned and proven.** Keep these identical across the stage:
  - model routes in every role;
  - the unattended setting;
  - the cache, off;
  - the lane and worker layout;
  - env and budget values;
  - the benchmark version digest;
  - framework and app commits, taken from clean trees.

  Write all of it into each experiment description before a runner binds the registration.
  Afterwards edits are refused (`update_experiment_description`). Recompute the fingerprint
  before comparing: a working checkout once moved three commits between a baseline and its rerun.
- **Restore state between rounds.** Tasks that write need the data put back to a documented
  baseline before each round. Log the restore so graders can check it.
- **List each change's target tasks before the run**: the tasks the change was built to fix.
  Choosing them after seeing the scores is how noise gets kept.
- **Check run health before grading.** See §0 of
  [grade-benchmark-runs](../grade-benchmark-runs/SKILL.md).
- **Pre-register the decision.** Before the first run, write down each change's hypothesis, its
  target tasks, the metric, and the keep/revert rule (§4). Building the decision instrument
  before the data lands catches wrong pairing assumptions. "Cannot decide" is a different verdict
  from "do not keep".
- **Measure the noise floor once.** Run the identical configuration repeatedly and read every
  later delta against that spread. Distinct tasks buy precision faster than more repeats of the
  same few tasks.
- **Make repeats independent, then check that they were.** Count the distinct answers per task
  before trusting n. One programme found 310 attempts carrying 54 distinct answers, because a
  response cache made the "repeats" copies, and every variance estimate was void.
- **Same-day, interleaved controls.** The backend and providers drift. When a control arm runs
  beside a treatment, run both on the same day against the same backend state, interleaved by
  task. Before blaming a treatment, re-run a few control tasks and check that they still score in
  their earlier range.
- **Evidence-grade capture.** Writer health must show zero dropped turn records. Retention
  pruning evicts oldest-first when the store reaches `FW_OBS_DB_MAX_BYTES`, and it can take a run's
  own early spans; suppress it for the run with `FW_OBS_SUPPRESS_PRUNE` or
  `observability.store.suppress_pruning()`. Archive a store through its own snapshot path
  (`archive_to`, which uses a sqlite backup and `VACUUM INTO`), never with a file copy: the WAL
  holds the newest rows.
- **Cost-gate the runs.** Every paid stage gets an owner go-ahead with the exact command and a
  ceiling. Settle what you can first at no cost over stored evidence: re-scores, blame analysis,
  replays.

## 2. Land each risky change as its own commit

"The commit is the switch." A change that can be reverted alone can be judged alone. Runtime flags
look cheaper, but flags multiply configurations, linger after the decision, and let an untested
combination ship. Put one change in one commit, with a message that names its target tasks. Then
revert the commit, or keep it.

When several changes ship in one stage, they are measured together. You can still judge each one
on its own target tasks (§3), provided the target sets barely overlap. When two changes target
the same tasks, run them in separate stages or say that the verdict is joint.

## 3. Measure each change on its target tasks, per run

For each change:

1. **Target score per run.** Sum the target tasks' lines in each new run and in each run of the
   previous stage, under the **current** scoring rules (§5).
2. **Compare the runs as sets, not as means.** Write them as "old R1/R2/R3 → new R1/R2/R3", for
   example `10 / 8.5 / 9 → 12.5 / 13.5 / 13`.
3. **Check collateral damage through the cause tags.** A change can help its targets and hurt
   elsewhere. For example, new skill descriptions that the splitter also reads can add split
   losses on other tiers. Compare lines lost per cause outside the target set, old vs new.
4. **Check the side effects that are not scores:** LLM calls per job, time per job, and fan-out
   caps hit.

## 4. The decision rule

| Verdict | When |
|---|---|
| **Keep** | Every new run beats every old run on the target tasks. Or the gain is consistent and larger than the per-run spread (max − min) of both stages |
| **Keep (bug fix)** | It fixes a defect seen in the trace, even when the line count barely moves, provided it hurts nothing |
| **Revert** | It hurts the targets or elsewhere, makes no difference, or helps only within noise |
| **Revert (cost)** | It helps within noise at a real cost, for example 3× the LLM calls for a stage that sampled the splitter three times and voted |
| **Undecided** | A thin sample (one task, one flipped line). Say so, and re-measure in the next stage rather than claim it |

If two judges graded the stage, keep a change only where both judges see the gain. Movement that
only one judge sees is judge noise.

Retry only infrastructure failures that happened before the agent ran. Exhaustion, refusals and
degenerate answers are outcomes, and rerunning them until they pass is selection. Never fix the
system under test to raise the *baseline's* score: record the defect as a finding and fix it in the
next arm.

Say plainly when a verdict rests on a thin sample. A revert of "+0.5 lines on one task" is a
judgment call, not a measurement.

**Attribute carefully.** One dramatic event in one run can dominate a cause total. A 5-line
session collapsing on one bad split once outweighed every other split loss in the stage. Before
you credit or blame a change for a cause total, list the events behind the total. If one event
drives it, the verdict follows the event's own cause, not the total.

After a stage, write the decisions down with their numbers (§7). Make the reverts as separate
commits, so the kept configuration is a known commit pair. Say whether the kept configuration was
itself measured, or is expected to score within noise of the measured one.

## 5. Scoring rules change: re-score, never re-grade

Owners refine rules partway through a programme. One example: a right answer through a non-target
command first scored half, and later full. Because the grades carry the tag
(`[other:non-target:<cmd>]`), the old stages can be re-scored in code. The re-scorer in
[reference.md](reference.md#re-score-stored-grades-under-a-new-rule) copies rows and never edits the
stored file. On a real stage it moved 41 lines and reproduced the published re-scored total
exactly.

- Treat a change in what a grade means as a version: record the new rule, its date and who
  approved it. Old results are re-scored or withdrawn under it, never silently reinterpreted.
- Re-score **every** stage you compare against, including the baseline, before comparing.
- Keep the original files. The re-score is a view.
- A rule without a tag cannot be re-applied. Add the tag in the addendum the moment a rule is
  discussed, before it is decided.

## 6. Offline component replays: useful, and easy to fool yourself with

Before spending a full stage, replay the cheap LLM components (the request splitter, the skill
matcher) on every benchmark prompt. Score them with small checkers, such as an expected job list
per prompt or an expected skill per task. This catches routing drift early. It does not replace a
stage, because a replay's gain did not always survive live runs.

Five pitfalls, each of which has produced a false conclusion:

1. **Temperature 0 is not deterministic.** Identical prompts gave the same first answer for only
   about 9 of 49 multi-job requests. Use **at least 3 repetitions** per prompt and compare
   distributions, such as per-prompt pass counts out of k. Never compare single samples. Do not
   conclude causation from 2-run controls.
2. **Make sure the cache is really off.** If repeats are cache hits, they look perfectly stable and
   every comparison is meaningless. `FW_LM_CACHE=0` disables the DSPy response cache
   (`fastworkflow/utils/dspy_utils.py`). It is read through `fastworkflow.get_env_var`, which looks
   at the env files the workflow loaded. When the caller passes a default, as this read does,
   `os.environ` is never consulted. So setting it in `os.environ` inside a harness does nothing.
   Put it in the env file the harness passes to `fastworkflow.init`, the way `ExperimentHarness`
   sets it in the loaded values. Then **verify** it. `cache_hit` is often absent on recorded
   `fw.llm.call` spans, and absent means unknown, not a miss. Use evidence that does not depend
   on the flag:
   - repeats of a nontrivial prompt should sometimes differ, so count the distinct answers;
   - `usage` shows real token counts where it is recorded;
   - `usage_rollup(spans)["cache"]` reports hit, miss and unknown counts
     (`fastworkflow.observability.comparison`).
3. **Scorers have blind spots.** A lenient match accepted a stuttered name ("North Branch Library
   Library" matched "north branch library"). Add checks for repeated name parts, unresolved references
   ("each of them", "that X") left in a sub-task, names that belong to another job, and dropped
   kind words. Before trusting a checker, run it on deliberately broken outputs and confirm that
   each one fails.
4. **Hold data out and spend it once.** Rules or prompts revised after seeing failures, or tuned
   by an optimiser, score near-perfect in-sample. One reached 74 of 75 in-sample and was no better
   than the current version on held-out cases. Declare the tuning and held-out sets before
   collecting. A held-out set used for a decision is spent.
5. **Offline gain is not live gain, and it has a price.** Report the replay's improvement together
   with its cost in calls. "3 of 147 prompts better at 3× the calls" was reverted live.

## 7. Report the stage

Publish one results page per stage. Keep the grades JSON as the record. The page holds:

- a headline: the mean lines per run of the total, and the change against the previous stage and
  the baseline, both re-scored under the current rules;
- per tier: lines, R1/R2/R3, mean, passes per run, the previous stage's mean, and the made-up count;
- **a keep/revert table**: each change, its target tasks' old runs → new runs, the per-run delta,
  and the verdict with a one-line reason;
- lost lines per run, grouped by cause (and stage) with the tasks behind each group;
- **next fixes**, each with its "lines at stake": the lines lost per run on the tasks it targets.
  That number is a ceiling, not a forecast, and the page says so;
- what ran: the commits, models, cache, unattended setting, lanes, the state restores, and the
  grading setup.

Results are immutable. A correction is a new version of the page or file, and it leaves the
original in place. The template is in [reference.md](reference.md#results-page-outline). Record the conclusion in the
experiment notes as well, and the cross-stage synthesis in the benchmark analysis (see
[optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md)).

## Anti-patterns

- Judging from one run, or from the mean alone without the per-run values.
- Picking target tasks after looking at the scores.
- Comparing against a baseline graded under an older rule.
- Re-grading by hand to apply a rule.
- Bundling changes so that none can be reverted alone. Leaving runtime flags in place of commits.
- Crediting a cause total that one event drives.
- Trusting a replay whose repeats are identical, before checking the cache.
- Calling "lines at stake" a forecast.

## Checklist

```
- [ ] Hypotheses, target tasks, metrics and the keep/revert rule pre-registered
- [ ] Three or more runs per task; one configuration; commits in the experiment descriptions
- [ ] Repeats checked for distinct answers; noise floor known; pins recomputed; no dropped spans
- [ ] Target tasks listed per change before the run
- [ ] Each risky change is its own commit
- [ ] All compared stages re-scored under the current rules from tags
- [ ] Per change: old runs → new runs on target tasks, plus collateral by cause tag
- [ ] Verdict by the rule; thin samples and single-event totals called out
- [ ] Reverts committed separately; the kept configuration named
- [ ] Offline replays: ≥3 repetitions, cache verified off, checkers tested on broken outputs
- [ ] Results page: tiers, keep/revert table, losses by cause, next fixes with lines at stake
```

## Related

- Producing the grades, checklists, cause tags and metrics:
  [grade-benchmark-runs](../grade-benchmark-runs/SKILL.md).
- Paired comparisons, experiment notes and human feedback:
  [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md).
- Benchmark versions, experiment pins and the runner:
  [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md).
- Routing-model comparisons (a different kind of noise: holdouts and seeds):
  [evaluate-intent-routing](../evaluate-intent-routing/SKILL.md).
