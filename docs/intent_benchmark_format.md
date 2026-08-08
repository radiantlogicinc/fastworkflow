# Intent benchmark file format

A benchmark file is a hand-written, held-out test set for a workflow's intent models. It is
the only artifact that reliably tells you whether intent detection actually works: the F1
reported during training is computed on a random split of the *same* synthetic utterances
the model trained on, so it measures memorisation (finding F1 in
`docs/intent_training_improvements_spec.md`).

Consumed by `fastworkflow/train/heldout_evaluation.py`.

## Default path

```
<workflow>/intent_benchmark.json
```

The file is optional. When it is absent, held-out evaluation falls back to the whole-persona
holdout of the generated utterances. The orchestrator should expose a CLI flag to point at
a different path.

## Schema

```jsonc
{
  "schema_version": 1,
  "cases": [ /* BenchmarkCase objects */ ]
}
```

A bare JSON array of cases is also accepted, so a quick hand-written file does not need the
wrapper.

### `BenchmarkCase`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `context` | string | yes | Which context's classifier this case tests. Use the context name as it appears in `___command_info/` — `"*"` for the global context. |
| `utterance` | string | yes | The phrasing to send. Must NOT appear in any command's seed utterances (see "Disjointness" below). |
| `kind` | `"routing"` \| `"escalation"` | no (default `"routing"`) | Which axis this case scores. |
| `expected_label` | string | routing cases only | The fully-qualified label the classifier should return, e.g. `"Account/close_account"`. Must match a label in that context's `label_encoder.pkl` exactly — comparison only trims whitespace (`normalize_label`), so there is no near-match. See "Label qualification" below. |
| `expected_ancestor_command` | string | escalation cases only | The command that is absent from `context`'s label space and present in one of its ancestors. |

## The two case kinds

### Routing

Scored on two numbers, reported separately:

- **top-1** — the classifier returned `expected_label` as its single, confident answer.
- **in-list** — `expected_label` appears anywhere in the returned candidates. At runtime an
  in-list-but-not-top-1 result is a clarification prompt, not a correct route, which is why
  it is never blended with top-1.

### Escalation

An escalation case is defined **structurally**, not by intent: the phrasing aims at a
command that is *provably absent* from the tested context's label space and *present in one
of that context's ancestors*. The only correct answer is a lone, confident escalation label
(`wildcard` by default), because that is what makes the runtime walk up the parent chain
(`fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py:100-104`).

A `wildcard` returned *alongside* local candidates does **not** count as a pass. That case
takes the ambiguity branch at runtime and the escalation signal is silently dropped
(finding F7), so counting it would report a behaviour the runtime does not have.

`validate_escalation_cases()` checks both halves of the definition and reports any case
where the expected command is in the tested context (that is a routing case) or in no
ancestor (that is a typo). Escalation recall is always reported as its own number and is
never folded into routing accuracy — the two trade against each other and a blended score
hides the trade (decision D2).

If a workflow introduces a second non-routable label alongside `wildcard`, pass it to
`score_escalation(..., escalation_labels={"wildcard", "<other>"})`.

### Out of scope is not supported

Out-of-scope scoring does not ship in held-out evaluation. The loader explicitly rejects
`kind: "out_of_scope"` and points to `fix-d28`, where the runtime behavior and acceptance
design must be settled first. A future implementation must measure the full runtime shadow
outcome—not only `CommandRouter.predict`—because the user-visible path also includes
exact/fuzzy/cache routing, ancestor traversal, NLU stages, and CME actions.

## Disjointness from the seed table

`assert_benchmark_disjoint_from_seeds()` fails the run if any benchmark utterance is also a
seed utterance. Without that check a benchmark decays into a memorisation test the first
time someone pastes a failing case into their seeds to "fix" it.

Comparison is on a **normalised** form, because an exact string match is too weak to catch
the realistic version of that mistake. Normalisation is: NFKC, smart quotes folded to ASCII,
internal whitespace collapsed, surrounding whitespace and punctuation (`"'`*.,;:!?…()[]{}`)
stripped from both ends, then casefolded. So `"Close the account."` collides with
`"close the account"`.

Utterances that are merely *similar* to a seed are reported by
`find_near_duplicate_benchmark_cases()` as warnings; they do not fail the run.

## Worked example

`fastworkflow/examples/my_workflow/intent_benchmark.json`:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "context": "Account",
      "utterance": "wind this account down for me",
      "expected_label": "Account/close_account",
      "kind": "routing"
    },
    {
      "context": "Account",
      "utterance": "who is the owner on record here",
      "expected_label": "Account/get_account_owner",
      "kind": "routing"
    },
    {
      "context": "ReviewTicket",
      "utterance": "approve everything from this app at once",
      "kind": "escalation",
      "expected_ancestor_command": "AccessReviewWorkspace/bulk_decide"
    },
    {
      "context": "*",
      "utterance": "what am I able to do right now",
      "expected_label": "IntentDetection/what_can_i_do",
      "kind": "routing"
    }
  ]
}
```

The escalation case above is a real measured example: in context `ReviewTicket` this phrasing
returned `['wildcard', 'ReviewTicket/certify_approve', 'ReviewTicket/show_review_item']` and
produced a two-option prompt, when the correct behaviour was to escalate to
`AccessReviewWorkspace/bulk_decide`. It scores as a failure, correctly.

## Label qualification

A command's label is its path under `_commands/` with `/` separators, so a command written
in a context folder is `Account/close_account` while one written at the `_commands/` root is
bare (`get_order_details`). Framework commands are injected into **every** context's label
space from the internal workflow's `_commands/IntentDetection/`, so they keep that prefix
everywhere, including in `"*"`: the last case above is `IntentDetection/what_can_i_do`, never
`what_can_i_do`. `wildcard` and `parameter_value` are reserved NLU labels rather than
commands and are never qualified.

Getting this wrong does not produce a near miss, it produces a case that fails forever, so
it is worth checking against the label space rather than guessing. `validate_routing_cases()`
reports every routing case whose expected label is absent from the tested context's label
space, and the trainer logs each one as a `Benchmark defect:` warning before the run starts —
read those warnings as benchmark defects, not as model defects. The authoritative list of a
context's labels is the `contexts` map in `___command_info/routing_definition.json`, or the
`classes_` of its `label_encoder.pkl` under `___command_info/` (the folder for `"*"` is named
`global`, since `*` is not a directory name).

## How many cases

The reference workflow used 446 routing cases (two per command, 160 commands plus core
commands) and 37 escalation cases. Two independent phrasings per command is a reasonable
floor. Note the measured noise floor, which depends on whether generation is cached. With
the utterance and parameter-example caches warm, two independent full retrains of a
32-context workflow produced zero verdict changes across 446 routing and 37 escalation
cases, with byte-identical models — so a single paired build is interpretable and a
one-case difference is real. When generation is not cached, because seeds changed,
`--regenerate-utterances` was passed, or the cache was evicted, the LLM re-draws the
training data and the historical floor applies: 20.6% of routing verdicts (92 of 446)
changed between two otherwise-identical pre-cache runs. Reliability claims under
regeneration still require at least five paired builds with all runs reported.

## Output

Held-out evaluation writes `<workflow>/___command_info/heldout_evaluation.json`:

```jsonc
{
  "schema_version": 2,
  "generated_at": "2026-08-02T14:31:07.123456+00:00",
  "metric_notes": { /* what each metric does and does not mean */ },
  "totals": {
    // The persona holdout. Re-drawn every run, so these score DIFFERENT cases
    // each time and must not be compared between runs.
    "routing_total": 446,
    "routing_top1": 0.462,
    "routing_in_list": 0.639,
    // The fixed benchmark from this file. The only cross-run-comparable number.
    "benchmark_routing_total": 446,
    "benchmark_routing_top1": 0.508,
    "benchmark_routing_in_list": 0.671,
    // Escalation cases from this file.
    "escalation_total": 37,
    "escalation_recall": 0.919,
    // Persona-held-out rows whose expected label is an escalation class. Scored
    // on lone-escalation semantics and never summed into routing.
    "holdout_escalation_total": 88,
    "holdout_escalation_recall": 0.83,
    "mean_in_distribution_f1": 0.94
  },
  "contexts": [
    {
      "context": "Account",
      "in_distribution_f1": 0.94,
      "routing": { "total": 42, "top1_correct": 20, "in_list_correct": 27,
                   "top1": 0.476, "in_list": 0.643, "per_command": { } },
      "benchmark_routing": { "total": 42, "top1_correct": 22, "in_list_correct": 29,
                             "top1": 0.524, "in_list": 0.690, "per_command": { } },
      "escalation": { "total": 4, "correct": 3, "recall": 0.75, "failures": [ ] },
      "holdout_escalation": { "total": 9, "correct": 7, "recall": 0.778,
                              "failures": [ ] },
      "seed": 42,
      "heldout_personas": ["persona_03", "persona_06"],
      "notes": [ ]
    }
  ]
}
```

A ratio whose denominator is zero is `null`, never `0.0` — `"escalation_recall": null`
means "no escalation cases were scored", which is a different fact from a measured 0%.
The human table renders both as `-`, but a JSON consumer diffing two runs would read a
missing measurement as a total failure.

`in_distribution_f1` is the legacy training-split score. It is kept so runs stay comparable
with older ones, and named so it can no longer be mistaken for a generalisation measure.
