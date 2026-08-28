---
name: evaluate-intent-routing
description: >-
  Read fastWorkflow's held-out routing and escalation evaluation honestly, and decide from it
  whether a change to a workflow's commands, seeds or context models actually helped. Covers why
  the training F1 measures memorisation, the whole-persona holdout and what it does and does not
  remove, which of the reported numbers is comparable across runs, keeping `intent_benchmark.json`
  disjoint from the seeds, leak calibration, and the paired-build discipline a claim requires. Use
  after `fastworkflow train` prints its evaluation table, when comparing two training runs, when
  writing or fixing benchmark cases, or when a reported accuracy looks too good.
---

# Evaluating intent routing

## The number that has misled everyone

Training reports a weighted F1 computed on a random split of the **same** synthetic utterances the
model trained on. Every utterance for a command comes from a handful of personas expanding one
seed list, so the "test" rows are near duplicates of the training rows. That number measures
**memorisation**. The measured gap on a 160-command workflow: ~0.94 reported F1 against 46.2%
held-out top-1.

It is still reported, named `in_distribution_f1` so it can no longer be mistaken for a
generalisation measure, and printed with a footer saying so. Judge models on top-1, in-list and
escalation recall. Never quote `in_distribution_f1` as accuracy.

## Two axes, never blended

Routing and escalation trade against each other, so one blended score would hide the trade.

**Routing** — did the classifier name the right command?

- **top-1** — the expected label came back as the **single, confident** answer. This is the only
  outcome that is a correct route.
- **in-list** — the expected label appears anywhere in the returned candidates. At runtime that is
  a clarification prompt, not a route. It is a real outcome worth tracking, and it is not a win.

**Escalation** — did the classifier correctly say "this command lives upstairs"? Scored as recall
only. Correct **only** when the escalation label comes back **alone and confident**, because only
a lone escalation label makes the runtime walk the parent chain. An escalation label returned
beside local candidates takes the ambiguity branch and the signal is silently discarded, so
counting it would report behaviour the runtime does not have.

## Two populations, and only one is comparable across runs

| Population | Source | Comparable across runs? |
|---|---|---|
| persona holdout (`routing`, `holdout_escalation`) | whole personas reserved from the generated utterances | **No.** The split is re-drawn every run, so two runs score different cases. |
| benchmark (`benchmark_routing`, `escalation`) | `<workflow>/intent_benchmark.json` | **Yes.** The file is fixed and its cases pair by construction. |

This is the single most common misreading of the report. A persona-holdout top-1 that moved three
points between runs may mean nothing at all — a context is never scored on the same phrasings
twice. **Compare on the benchmark number.** If a workflow has no benchmark file, it has no
cross-run measurement, only a snapshot.

A ratio with a zero denominator is `null`, never `0.0`. `"escalation_recall": null` means no
escalation cases were scored, which is a different fact from a measured 0%.

## What the persona holdout does and does not remove

Whole personas are reserved, never a random sample of rows. Holding out 25% of the rows of a
persona whose other rows are in training is the defect, not the fix. Consequences:

- Hand-written seed utterances (persona `__seed__`) always train and are never held out.
- An utterance attributed to several personas is held out only when **every** contributing persona
  is held out.
- A label that the split would leave below the trainer's floor of 2 training rows has its held-out
  rows **returned to training wholesale**, rather than being left partially trained to buy a
  metric. Those labels then have **no held-out coverage at all**, and a note in the report names
  them. A command with few utterances can therefore be invisible in this report rather than
  scoring badly in it — check the note before reading a clean result as coverage.

What it does not remove is the **wording**. Two personas asked the same question in one prompt
often produce near-identical sentences, so removing the author leaves the phrasing behind. That is
what `leak_calibration` quantifies: the median maximum token overlap between held-out rows and
their own label's training rows, and the fraction overlapping at 0.8 or more. High overlap means
the routing number is closer to recall than to generalisation.

`exact_duplicates` should be **0** — the split drops exact leaks. A non-zero value is a bug
signal, not a data observation.

Two more limits worth stating before quoting any number:

- The evaluated model **is** the shipped model, with held-out personas removed from training, so
  the numbers are **lower bounds**.
- The holdout is in-generator: held-out personas came from the same generator, so it measures
  generalisation across personas, not across real users.

## The benchmark file is the instrument; keep it clean

`<workflow>/intent_benchmark.json` is a hand-written held-out test set. Full schema, the two case
kinds, label-qualification rules and a worked example are in `docs/intent_benchmark_format.md` in
the fastWorkflow repository. What matters here is the discipline around it:

**Never paste a failing benchmark case into a seed list to "fix" it.** That is the mistake the
whole design defends against, and it is caught: benchmark utterances are enforced disjoint from
the seed table, and an overlap raises `BenchmarkLeakError` and **fails the run**. Comparison is on
a normalised form — NFKC, smart quotes folded, whitespace collapsed, surrounding punctuation
stripped, casefolded — so dropping the trailing period does not evade it. Merely *similar*
utterances are reported as warnings and do not fail.

The check runs in preflight, **before** any paid generation, so a leak costs you nothing but the
time to fix it.

**`Benchmark defect:` warnings are defects in the benchmark, not in the model.** A routing case
whose expected label is absent from the tested context's label space fails forever. The
authoritative label list is the `contexts` map in `___command_info/routing_definition.json`, or
the `classes_` of a context's `label_encoder.pkl`. Framework commands keep their
`IntentDetection/` prefix in every context including `"*"`; `wildcard` and `parameter_value` are
reserved labels and are never qualified.

Two independent phrasings per command is a reasonable floor. The reference workflow used 446
routing cases and 37 escalation cases.

## Reading the report

Printed at the end of training and written to `___command_info/heldout_evaluation.json`
(schema 2), with a `metric_notes` block restating what each metric does and does not mean.

Per context: `in_distribution_f1`, `routing`, `holdout_escalation`, `leak_calibration`,
`benchmark_routing`, `escalation`, plus the `seed` and the `heldout_personas` list.
`routing.per_command` is where a single bad command hides inside a healthy context average — read
it before concluding a context is fine.

**After a selective retrain, some rows were not measured this run.** Carried-forward contexts are
re-inserted from the previous report and tagged `carried_forward` with the version they came from.
Those numbers still describe the models this version ships, but a diff across two runs will show
them as unchanged because nothing re-measured them, not because nothing moved.

## Deciding whether a change helped

The noise floor depends entirely on whether generation was cached:

| Condition | Measured floor | What one paired build proves |
|---|---|---|
| both caches warm (no seed change, no `--regenerate-utterances`) | two full retrains produced **zero** verdict changes across 446 routing and 37 escalation cases, with byte-identical models | a one-case difference is real |
| generation re-ran | **20.6%** of routing verdicts (92 of 446) changed between two otherwise-identical runs | nothing; a reliability claim needs at least five paired builds, all reported |

So the first question about any comparison is not "did the number move" but "did the caches stay
warm". A change that forces regeneration — a persona change, a model change, a framework upgrade —
puts you in the second row whether you wanted it or not. See the `train-and-publish-models` skill
for what forces regeneration.

Then read cases, not percentages:

1. Compare **benchmark** verdicts, case by case, before and after.
2. Keep only the cases the change can actually explain. A change touching one command's seeds
   cannot move an unrelated framework command; that movement is churn.
3. Report the causal set as a **trade** — "N cases gained here against M lost there" — because a
   well-targeted fix usually does cost something at the boundary it moved.
4. Check escalation recall separately. Routing gains bought by suppressing escalation are not
   gains.

## Scoring outside a training run

The module never loads a model. Scoring takes a caller-supplied
`predict_fn(utterance) -> list[str]` returning ranked candidates, top-1 first — exactly what
`CommandRouter.predict` returns — so evaluation is importable and runnable without torch and
against any model you can wrap in that signature.

## Checklist

```
- [ ] in_distribution_f1 is not being quoted as accuracy
- [ ] The comparison is on benchmark_routing, not on the re-drawn persona holdout
- [ ] Caches were warm on both runs, or five paired builds exist
- [ ] Routing and escalation are reported separately
- [ ] top-1 and in-list are not conflated
- [ ] leak_calibration was read; exact_duplicates is 0
- [ ] No benchmark utterance has migrated into a seed list
- [ ] `Benchmark defect:` warnings were fixed as benchmark bugs, not chased as model bugs
- [ ] Carried-forward contexts are excluded from the diff, or noted as unmeasured
- [ ] The claim is a case-level trade, not a percentage delta
```

## Related

- Benchmark file schema, case kinds and label qualification: `docs/intent_benchmark_format.md` in
  the fastWorkflow repository.
- What forces regeneration, and therefore which noise floor applies: the
  `train-and-publish-models` skill.
- Turning a routing failure into a design fix: the `design-context-models` and
  `detect-duplicate-capabilities` skills.
