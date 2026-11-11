# Intent-model training: full walkthrough

Companion to `../SKILL.md` §2. All line numbers refer to
`fastworkflow/model_pipeline_training.py` unless stated. Facts verified 2026-07-09
against v2.22.2 (commit c33b9a5).

## Who calls what

```
fastworkflow train <wf> [env] [passwords]        (cli.py:229-247)
  └─ train/__main__.py::train_workflow
       ├─ recursively trains child workflows under _workflows/ first (:48-57)
       ├─ if `datasets` missing: regenerate command_directory/routing JSON only, skip
       │  model training AND DSPy example generation (:64-72, slim-Docker design)
       ├─ model_pipeline_training.train(workflow)   (:803)  ← the fine-tuning loop
       ├─ _generate_dspy_examples_helper (:91-131)  ← few-shot corpus, see sibling ref
       └─ _prune_stale_artifacts (:133-180)         ← only AFTER success, so a failed
          retrain leaves the previous artifacts runnable
```

`train_main` also trains the bundled command_metadata_extraction (CME) workflow first
when the target path doesn't contain the substring `fastworkflow` and
`is_fast_workflow_trained` is False (`train/__main__.py:281-289`).

Open oddity (from discovery, **unverified by running a train — confirm with Dhar before
relying on it**): `is_fast_workflow_trained` (`train/__main__.py:225-256`) checks
`___command_info/ErrorCorrection/largemodel.pth`, but since v2.7.0 `train()` excludes the
ErrorCorrection context (`model_pipeline_training.py:821` — note the duplicate literal
`{'ErrorCorrection', 'ErrorCorrection'}`), so that path is never produced and the check
may always fail → CME possibly retrains on every external `fastworkflow train`.

## Building the per-context training set (:820-877)

For each context to train (workflow contexts − internal CME contexts, + `'*'` → folder
`global`):

1. **Labels** = context's commands + core commands (`cmd_dir.core_command_names`).
2. **Utterances per command** = output of the command's `generate_utterances` (its own, or the
   framework default `default_generate_utterances`), i.e. humanized command name +
   `generate_diverse_utterances(seeds, name)`
   (which itself prepends the raw command name and seeds). Commands lacking
   `Signature.Input` return `[]` and are skipped (:741-766).
3. **Negative class**: label `wildcard` gets (ancestor-context utterances − this
   context's own utterances) ∪ the `wildcard` command's seed utterances (:869-877).
   The wildcard seeds are intentionally junk-shaped ("3", "france", "id=3636", ... —
   `_workflows/command_metadata_extraction/_commands/wildcard.py:10-18`) so bare
   parameter values route to out-of-scope handling. **Consequence: your context
   hierarchy design directly shapes classifier accuracy** — ancestors' utterances are
   the out-of-scope examples.
4. `LabelEncoder.fit_transform(y)` (:910), `train_test_split(test_size=0.25,
   random_state=42)` (:914). Note: NOT stratified; tiny classes can be split unevenly.

## The two fine-tuning loops

Standard supervised fine-tuning of `AutoModelForSequenceClassification` (a pretrained
transformer with a fresh classification head sized to `num_labels`):

| Step | Tiny (:899-1001) | Distil (:1004-1054) |
|---|---|---|
| Checkpoint | `INTENT_DETECTION_TINY_MODEL`, default `google/bert_uncased_L-4_H-128_A-2` (:892-893) | `INTENT_DETECTION_LARGE_MODEL`, default `distilbert-base-uncased` (:894-895) |
| Tokenize | padding, truncation, max_length=128 (:924-930) | same, own tokenizer |
| Optimize | AdamW lr=1e-4 (:956) | AdamW lr=5e-5 (:1019) |
| Epochs | 12 (:957) | 5 (:1020, re-assigned :1026) |
| Batch | DataLoader batch_size=10, shuffle train (:934-946) | same (:1005-1017) |
| Loss | model's built-in cross-entropy (labels passed to forward) | same |
| Eval per epoch | `evaluate_model` → weighted F1, NDCG@k, test loss (:687-717) | same |

No early stopping, no LR schedule, no gradient clipping, no best-checkpoint selection
(`best_ndcg`/`best_f1` are initialized but never used — :962-963, :1024-1025); the LAST
epoch's weights are saved via `save_pretrained` (:679-684). Device = CUDA if available
else CPU (:24). Defaults chosen for transformers 4.48+/5.x compatibility after the
v2.18.0 breakage (old `prajjwal1/bert-tiny` lacked `model_type`/fast tokenizer).

Why these hyperparameters? **Provenance unknown.** Nothing documents tuning runs. The
epoch asymmetry (12 vs 5) is consistent with "smaller model needs more steps", but treat
all six numbers as empirical defaults that survived.

## Metrics, precisely

- **Weighted F1** (`sklearn.metrics.f1_score(average='weighted')`): per-class F1 averaged
  weighted by true-class frequency. With the wildcard class often dominant, weighted F1
  can look good while a rare command is broken — check per-class if debugging.
- **NDCG@3** (:393-414 pipeline version; :719-730 per-model version): for each sample,
  relevance is 1 only at the true label's rank in the top-k; DCG = 1/log2(rank+1); IDCG =
  1 (single relevant item), so per-sample NDCG = 1.0 (rank 1), 0.63 (rank 2), 0.5
  (rank 3), 0 (absent). It scores the *clarification list* quality.
- **distil_percentage** (`ModelPipeline.evaluate` :510-560): fraction of test samples
  where tiny confidence fell below the threshold and DistilBERT was consulted — the
  latency/accuracy trade dial.

## Threshold tuning, step by step

After both models train (per context):

1. `analyze_model_confidence` (:166-220) runs the test set through each model, splitting
   softmax confidences into "successful" (correct) and "failed" (wrong) populations, with
   min/max/mean/median for each.
2. `find_optimal_threshold(tiny_stats, tiny_test_loader, pipeline)` (:222-258):
   - Sweep = `np.linspace(mean failed tiny confidence, mean successful tiny confidence, 20)`.
   - For each candidate, set `pipeline.confidence_threshold` and run the full two-tier
     `evaluate`, recording f1, ndcg, distil_usage.
   - Winner maximizes `f1 * ndcg * (1 - 0.15 * distil_usage/100)` (:255-256).
   - Degenerate case: if either mean is missing (e.g., tiny made zero mistakes → no
     failed mean), returns threshold **-1** with -1 metrics (:228-241) and that gets
     written to threshold.json. A -1 threshold means tiny is always "confident enough".
3. Ambiguous thresholds (:1174-1190): `tiny_ambiguous_threshold` = mean confidence of
   tiny's misclassified test samples (0.0 if none); same for large. Rationale: if the
   answering model's confidence is below the typical confidence *of its own mistakes*,
   don't trust a single label — show top-k.
4. Written artifacts: `threshold.json`, `tiny_ambiguous_threshold.json`,
   `large_ambiguous_threshold.json` — each `{"confidence_threshold": <float>}`.

Worked real example (CME workflow, committed in-repo, `global` context):

| File | Value | Runtime meaning |
|---|---|---|
| threshold.json | 0.4129 | tiny confidence < 0.4129 → ask DistilBERT |
| tiny_ambiguous_threshold.json | 0.4010 | tiny answered and confidence ≤ 0.4010 → return top-k |
| large_ambiguous_threshold.json | 0.5468 | distil answered and confidence ≤ 0.5468 → return top-k |

(For the CME `IntentDetection` context, large_ambiguous is 0.0 — DistilBERT never
misclassified on that tiny test set, so a single label is always returned when distil
answers.)

**Calibration caveat for the tau2 program**: these are decision thresholds tuned on a
~25% split of ~20-25 utterances/command — noisy. Softmax confidences of fine-tuned BERTs
are known to be overconfident; nothing here recalibrates them. The thresholds are the
only "calibrated uncertainty" in the whole system (planning/memory have none — see the
tau2 plan's E23). Confidence may schedule escalation but must never authorize an
irreversible action.

## Runtime loading and caching

- `CommandRouter` per artifacts-folder is a path-keyed singleton (`__new__` cache,
  :271-287); `ModelPipeline` is keyed on (tiny_path, distil_path, threshold, device)
  (:337-354). Retraining on disk does NOT refresh an already-loaded process — restart or
  clear `CommandRouter._instances_cache`.
- `predict_single_sentence` (:564-599) loads `label_encoder.pkl` **on every call** via a
  module-global (:579-581) — a known inefficiency and a thread-safety smell (module-global
  `label_encoder` shared).
- Model path at runtime: `<app_workflow>/___command_info/<context_name>` with `'*'` →
  `global` (`intent_detection.py:39`, `CommandRouter.__init__` :293-294).
- Fail-fast: `is_workflow_trained` (:624-674) requires `threshold.json` for every
  non-internal context + `'*'` before a chat session starts (added v2.19.0 after
  crash-on-first-command bug).
- Fingerprint invalidation (v2.22.1) covers `command_directory.json` /
  `routing_definition.json` only — **model files have no fingerprint**; editing a
  command's utterances without retraining leaves stale models serving silently.

## Discipline rules that bind here

- NEVER train into `fastworkflow/examples/*/___command_info` from tests or experiments —
  copy the workflow to a temp dir first (fix-0hb incident, commit fa97b48; pattern in
  `tests/test_train_modern_stack.py:129-139`).
- `___command_info` contents are never shipped in the wheel (pyproject.toml:23-28);
  production images copy trained models from dev and regenerate only the JSONs.
- Local training recipe (uses repo-local real keys, NOT the bundled placeholder):
  `fastworkflow train ./fastworkflow/examples/hello_world ./env/.env ./passwords/.env`

## Re-verification

```bash
grep -n "num_epochs\|lr=1e-4\|lr=5e-5\|test_size=0.25\|batch_size=10" fastworkflow/model_pipeline_training.py
grep -n "linspace\|alpha = 0.15\|failed'\]\['mean'\]" fastworkflow/model_pipeline_training.py
grep -n "ErrorCorrection" fastworkflow/model_pipeline_training.py fastworkflow/train/__main__.py
python .claude/skills/fastworkflow-nlu-pipeline-reference/scripts/show_intent_thresholds.py \
  fastworkflow/_workflows/command_metadata_extraction
```
