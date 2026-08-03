---
name: fastworkflow-nlu-pipeline-reference
description: >
  Domain reference for fastWorkflow's NLU stack AS IMPLEMENTED IN THIS REPO: two-tier
  TinyBERT/DistilBERT intent classifiers, confidence thresholds, synthetic utterance
  generation (litellm + PersonaHub), DSPy parameter extraction, runtime matching layers
  (fuzzy/embedding-cache/classifier), and per-role litellm model routing. Load this when
  you hear: "wrong command predicted", "ambiguous intent", "intent detection", "parameter
  extraction returned NOT_FOUND", "threshold.json", "confidence threshold", "utterance
  generation", "DSPy signature/cache", "LabeledFewShot", "litellm_proxy", "why did the
  classifier pick X", "what does wildcard mean", "wildcard vs parameter_value", "reserved
  label", "escalation label", "why is there no wildcard class in this context",
  "in_distribution_f1", "___command_info/versions", or when modifying anything under
  model_pipeline_training.py, intent_detection.py, nlu_labels.py, signatures.py,
  generate_synthetic.py, heldout_evaluation.py, or cache_matching.py.
  Do NOT load for step-by-step failure triage (use fastworkflow-debugging-playbook),
  for deciding whether one training run beat another or how many seeds to add (use
  fastworkflow-intent-training-convergence), the full env-var catalog
  (fastworkflow-config-and-flags), or tau-bench benchmark mechanics
  (fastworkflow-taubench-reference).
---

# fastWorkflow NLU Pipeline Reference

Everything below is verified against source at v2.22.2 (commit c33b9a5), 2026-07-09,
**except** the passages marked "2026-08-02", which were re-verified against the working
tree after wave 1 of epic `fix-551` (held-out evaluation, determinism, artifact
versioning, and the reserved-label split) landed. File paths are repo-relative. **Trust
this document over CLAUDE.md and README where they disagree** — known doc rot is listed at
the end.

> **2026-08-02 — the biggest change since this skill was written.** `wildcard` is no
> longer a catch-all. `fastworkflow/nlu_labels.py` splits it into two reserved labels with
> different meanings and different training rules; §2 has the detail. Anything you
> remember about "the wildcard class" is at best half right now.
>
> `fastworkflow/model_pipeline_training.py` is under active edit as R1 is wired in, so the
> line numbers below drift. Prefer the symbol names; `grep -n` for them.

## When to use / when NOT to use

| You need... | Use |
|---|---|
| How intent detection / parameter extraction actually work here; what a threshold means; where a magic number lives | **This skill** |
| A symptom-to-fix triage table for a live failure | `fastworkflow-debugging-playbook` |
| Every env var with defaults and consumers | `fastworkflow-config-and-flags` |
| tau-bench / tau2-bench harness, pass^k, simulator mechanics | `fastworkflow-taubench-reference` |
| Why the architecture is shaped this way; invariants | `fastworkflow-architecture-contract` |
| Running train/build/CLI commands operationally | `fastworkflow-run-and-operate` |
| Measuring model quality instead of eyeballing | `fastworkflow-diagnostics-and-tooling` |
| Growing seeds/personas and judging whether run B beat run A | `fastworkflow-intent-training-convergence` |
| Variance/pass^k math, calibration analysis recipes | `fastworkflow-proof-and-analysis-toolkit` |

## Glossary (one line each, used throughout)

| Term | Meaning here |
|---|---|
| fine-tuning | Continuing training of a pretrained model's weights on your small labeled dataset |
| logits | Raw per-class scores from the classifier head, before normalization |
| softmax confidence | `max(softmax(logits))` — the top class's probability; this repo's only "confidence" signal |
| calibration | How well confidence tracks actual correctness. This repo tunes *decision thresholds* on held-out data but never calibrates the probabilities themselves (no temperature scaling) |
| embedding | Fixed-length vector representing text; here the DistilBERT `[CLS]` token's last hidden state (`cache_matching.py:51`) |
| cosine similarity | Angle-based vector similarity in [~-1, 1]; 1.0 = same direction |
| Levenshtein distance | Minimum single-character edits between strings; "normalized" = divided by the longer length, so 0.0 = identical |
| LabelEncoder | sklearn utility mapping label strings ↔ integer class ids (the ONLY load-bearing sklearn use in intent detection) |
| weighted F1 | Per-class harmonic mean of precision/recall, averaged weighted by class frequency |
| NDCG@3 | Ranking score: 1.0 if the true label is ranked 1st, discounted by 1/log2(rank+1) if 2nd/3rd, 0 if absent from top-3 |
| DSPy | Framework that compiles typed "Signatures" (input/output field specs) into LLM prompts via modules like `ChainOfThought` |
| ChainOfThought | DSPy module that makes the LLM emit reasoning before the output fields |
| LabeledFewShot | DSPy optimizer that stuffs labeled examples into the prompt (few-shot = examples-in-prompt) |
| litellm | Client library exposing one API over many LLM providers via `provider/model` strings |
| NOT_FOUND | String sentinel (env var `NOT_FOUND`, value `"NOT_FOUND"`) marking an unextracted parameter |

## The 30-second mental model

Every user turn in deterministic mode runs a pseudo-command called `wildcard` on an
internal "command_metadata_extraction" (CME) workflow. That one command
(`fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py`) drives the
whole NLU state machine:

```
user text
  └─> Stage 1  Intent detection   (intent_detection.py — 4-layer matching ladder)
  └─> Stage 2  Parameter extraction (parameter_extraction.py + utils/signatures.py — regex/DSPy)
  └─> Stage 3  Command execution  (your command's ResponseGenerator — outside NLU)
```

Any gap (ambiguous intent, missing/invalid parameter) produces a **clarification turn**,
never a guess — that is the framework's core reliability contract.

## 1. Intent detection: the 4-layer matching ladder

Order, from `_workflows/command_metadata_extraction/intent_detection.py` (`predict`, :36-167):

| # | Layer | Mechanism | Threshold | Code |
|---|---|---|---|---|
| 1 | Exact first-token | First whitespace/`(`-delimited token, lowercased, looked up in valid command names | exact | :100-105 |
| 2 | Fuzzy match | Normalized Levenshtein **distance** vs command names (input truncated-prefix compare) | distance ≤ 0.3 | :108-114 → `utils/fuzzy_match.py:17` |
| 3 | Embedding cache | Cosine similarity vs previously-clarified utterances in `<app_workflow>/___convo_info/cache.db` | ≥ 0.85 | :118 → `cache_matching.py:131` |
| 4 | Transformer classifier | `CommandRouter.predict` (two-tier BERT, next section) | trained thresholds | :121 |

Sharp edges, all verified:

- Layer 2's threshold is a **distance** (lower = closer), not a similarity. `find_best_matches`
  also compares against each candidate *truncated to the input's length* (`fuzzy_match.py:49`)
  — effectively prefix-biased matching.
- Layer 3: the `cache_match` function *default* is 0.90 (`cache_matching.py:131`) but the only
  call site passes **0.85** (`intent_detection.py:118`). The 0.90 default is never used.
- Layer 3 is an O(n) linear scan opening the RocksDB (`speedict.Rdict`) per message, one
  `cosine_similarity` call per cached entry (`cache_matching.py:168-179`). Embeddings are
  memoized in-process via `lru_cache(256)` keyed on `(id(pipeline), text)` (:18-27).
- **Learning loop**: after a successful ambiguity/misunderstanding clarification, the ORIGINAL
  utterance (from `cme_workflow.context["command"]`) is stored with the resolved label and its
  DistilBERT embedding (`intent_detection.py:151-162` → `store_utterance_cache`), so the next
  similar utterance short-circuits at layer 3. Multiple labels per utterance are resolved by
  highest frequency, tie-broken by most recent feedback date (`cache_matching.py:189-208`).
  This is the implementation behind README's "1-shot adaptation" claim.
- Special commands bypass the classifier entirely: `ErrorCorrection/abort` and
  `ErrorCorrection/you_misunderstood` (and `what_can_i_do` during ambiguity clarification) are
  matched only via layers 1-2 against their `plain_utterances` (:69-97).
- `majority_vote_predictions` is **dead code** — the call is commented out (`:180`) and its
  own TODOs admit predictions are deterministic, so voting is pointless without a
  temperature mechanism. Status: stalled reliability idea, relevant to tau2 work.
- **2026-08-02:** layer 4's outcome now goes through the reserved-label helpers, not a
  string compare. A single prediction routes (`:182-183`); a top-k list containing an
  escalation label follows fixed conservative behaviour: the signal is **discarded** and
  the user is prompted with the local candidates only (finding F7), while a warning names
  the suppressed labels. A top-k `parameter_value`
  deliberately never escalates.

## 2. Two-tier intent models (TinyBERT + DistilBERT)

**Not sklearn.** Despite CLAUDE.md's wording, intent models are two HuggingFace
`AutoModelForSequenceClassification` checkpoints fine-tuned with torch
(`fastworkflow/model_pipeline_training.py` — note: repo root of the package, NOT under
`train/`). sklearn contributes only `LabelEncoder`, `train_test_split`, `f1_score`
(the `PCA` import at :5 is unused).

| | Tier 1 "tiny" | Tier 2 "large" |
|---|---|---|
| Default checkpoint | `google/bert_uncased_L-4_H-128_A-2` | `distilbert-base-uncased` |
| Env override | `INTENT_DETECTION_TINY_MODEL` | `INTENT_DETECTION_LARGE_MODEL` (:892-895; env-FILE only — see §6 quirk) |
| Optimizer / LR | AdamW, 1e-4 (:956) | AdamW, 5e-5 (:1019) |
| Epochs | 12 (:957) | 5 (:1020) |
| Batch size | 10 (:936) | 10 |
| Max token length | 128 (train and inference) | 128 |
| Split | `train_test_split(test_size=0.25, random_state=42)` (:914, now :1053 — see note below) | same data |

One model **pair per command context**, trained by `train()`. Contexts trained =
workflow's contexts minus the internal CME contexts, plus `'*'` (mapped to folder `global`
— `GLOBAL_CONTEXT_FOLDER`, `context_set_for_training`). Commands without `Signature.Input`
are excluded from training entirely — they are `perform_action`-only. Labels are
fully-qualified (`Context/command`); the runtime reduces them with
`nlu_labels.label_of` (`prediction.split("/")[-1]`).

**Reserved labels — updated 2026-08-02, this changed.** A context's label space is its own
commands + core commands + up to two *reserved* labels that name no command. Until wave 1
of `fix-551` there was one, `wildcard`, doing two unrelated jobs in two different NLU
stages, which meant the escalation classifier was trained on the literal `"france"`.
`fastworkflow/nlu_labels.py` (standard-library only, so the trainer and the runtime can
both import it) now owns the vocabulary:

| Label | Stage | Meaning | Trained in which contexts | Members |
|---|---|---|---|---|
| `WILDCARD_LABEL` = `wildcard` | INTENT_DETECTION | **escalation signal** — "an ancestor context can serve this" | only where `ancestor_utterances - context_utterances` is non-empty; **a context with no ancestors gets no escalation class at all** | ancestor-context utterances not also valid locally, plus the CME `wildcard` command's humanised name |
| `PARAMETER_VALUE_LABEL` = `parameter_value` | PARAMETER_EXTRACTION | **bare-value catcher** — "this is a value, not a local command" | every context | the seven contentless literals `PARAMETER_VALUE_PLACEHOLDERS` (`"3"`, `"france"`, `"id=3636"`, …) |

Both are in `NON_ROUTABLE_LABELS`: a prediction of either resolves to
`command_name=None` (`intent_detection.py:368-380`,
`resolve_fully_qualified_command_name`) and neither is ever displayed to the user as a
choosable command (`:410-418`). Only `wildcard` is in `ESCALATION_LABELS` — a bare value
carries no evidence that an ancestor can help. Use `is_escalation` / `is_non_routable`
rather than string-comparing to `"wildcard"`.

Why the root-context drop is safe rather than a lost out-of-scope sink: in a root context
the escalation class would collapse to a single row, the split is not stratified, and the
ambiguous thresholds are the *mean confidence of misclassified test samples* — so that one
row would move a live runtime threshold. And a root context's parent walk terminates
immediately at `you_misunderstood`, which an unconfident classifier already reaches. See
the "no out-of-scope rejection" note in §5.

Assembly lives in `train()` — `grep -n "WILDCARD_LABEL\|PARAMETER_VALUE_LABEL"
fastworkflow/model_pipeline_training.py`.

**Runtime tiering** (`ModelPipeline.predict_batch` :416-508, `CommandRouter.predict` :321-334):

1. TinyBERT predicts. If softmax confidence ≥ `confidence_threshold` (from
   `threshold.json`), use its answer; else DistilBERT re-predicts (that sample only).
2. Whichever model answered: if its confidence > that model's *ambiguous* threshold
   (`tiny_ambiguous_threshold.json` / `large_ambiguous_threshold.json`), return ONE label;
   otherwise return top-k labels (k = 3, or 2 when only 2 classes; :581-582, :887-888)
   → triggers the ambiguity-clarification flow.

**Threshold tuning** (all per context, written at train time):

- `confidence_threshold` ← `find_optimal_threshold` (:222-258): sweeps 20 `linspace` points
  between TinyBERT's mean-failed-confidence and mean-successful-confidence, maximizing
  `f1 * ndcg * (1 - 0.15 * distil_usage%/100)` (alpha=0.15 at :255) — i.e., accuracy
  discounted by how often the slower model is consulted. If a context has zero
  misclassifications (no failed-confidence mean), it writes threshold **-1** (:229-241) —
  every input then goes to whichever branch `-1` implies (tiny always confident).
- `tiny/large_ambiguous_threshold` ← simply the **mean confidence of that model's
  misclassified test samples** (:1174-1190), 0.0 if it never misclassified.
- Real values (CME workflow `global` context, committed in-repo): confidence 0.4129,
  tiny-ambiguous 0.4010, large-ambiguous 0.5468. Inspect any workflow with
  `scripts/show_intent_thresholds.py` (in this skill).
- `find_optimal_confidence_threshold` (:50-162) with its magic `min_threshold=0.5129` and
  `max_top3_usage=0.3` is **DEAD CODE** — nothing calls it; the live path uses
  `find_optimal_threshold`. Provenance of 0.5129: unknown, presumed leftover experiment.
- A legacy `ambiguous_threshold.json` lingers in trained folders; `CommandRouter.__init__`
  (:298-311) reads only `threshold.json` + the `tiny_`/`large_` variants.

**What F1 / NDCG@3 mean here**: F1 (weighted) scores hard top-1 classification on the
25% split-off utterances. NDCG@3 credits partially-correct ranking — it is the quality
measure for the *top-k clarification list* the user sees when the model is unsure
(implementation :393-414 and :719-730). High F1 + low NDCG@3 would mean confident answers
are fine but clarification lists are bad.

**That split is over the SAME synthetic set the model trained on** (finding F1 of
`docs/intent_training_improvements_spec.md`), and every utterance for a command comes from
a handful of personas expanding one seed list, so the "test" rows are near duplicates of
the train rows. The number measures memorisation: ~0.94 reported F1 against 46.2%
real held-out top-1 on a 160-command workflow. It is **retained** because it is what
calibrates the ambiguity thresholds; it is only its use as a *quality* metric that is
unsound. As of 2026-08-02 it is reported under the name `in_distribution_f1` alongside
real held-out metrics (`fastworkflow/train/heldout_evaluation.py`, whole-persona holdout
plus an optional `<workflow>/intent_benchmark.json`, written to
`___command_info/heldout_evaluation.json`). For how to act on those numbers see
`fastworkflow-intent-training-convergence`; for the file schema, `docs/intent_benchmark_format.md`.

Artifacts land in `<workflow>/___command_info/versions/<version_id>/<context>/`, with a
`current` pointer file (plus a convenience `current` symlink) selecting the live set —
`fastworkflow/train/artifact_versioning.py`. This is an internal atomic-publication layout:
training retains only current and one previous recovery point. The pre-2026-08 flat layout
`___command_info/<context>/` is still read and is migrated on the next train. Per context:
`tinymodel.pth/`, `largemodel.pth/` (HF `save_pretrained` dirs), `label_encoder.pkl`,
`threshold.json`, `tiny_ambiguous_threshold.json`, `large_ambiguous_threshold.json` —
roughly 275 MB. Missing `threshold.json` = untrained context; `is_workflow_trained`
(:624-674, now :672) fail-fast checks exactly this before chat starts. **NEVER let
tests/experiments write into `fastworkflow/examples/*/___command_info` — train into temp
copies (fix-0hb incident, commit fa97b48).**

Full training walkthrough + metrics math: [references/intent-model-training.md](references/intent-model-training.md).

## 3. Synthetic utterance generation (train-time)

`fastworkflow/train/generate_synthetic.py::generate_diverse_utterances`. Called from each
command file's generated `generate_utterances` staticmethod (template:
`build/command_file_template.py:112-115`). Per command:

- Samples `SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS` (template default 4) personas from
  HuggingFace `proj-persona/PersonaHub` `persona.jsonl`, batches
  `PERSONAS_PER_BATCH` (1) × `UTTERANCES_PER_PERSONA` (5) through `litellm.completion` on
  `LLM_SYNDATA_GEN` (max_tokens=1000, temperature=1.0, top_p=0.9).
- Returns `[command_name] + seeds + generated` → default economics ≈ name + seeds + 4×5=20
  synthetic utterances per command. These env dials are your cost/quality knobs.
- **2026-08-02:** persona choice is no longer an unseeded `random.sample`. It is
  `select_persona_indices(n, num_personas, seed, command_name)`, derived from
  the trainer's fixed seed (`fastworkflow/train/determinism.py`), and each utterance's producing
  persona is recorded in `___command_info/training_provenance.json` — which is what makes
  the whole-persona held-out split possible. Provenance schema v2 separates the one
  generation record per fully-qualified command from each `(context, command)` labelled-row
  use, including explicit no-input/no-utterance skips and fallback use. The training report
  shows aggregate and per-context counts and applies row floors per context. Legacy flat
  provenance remains read-only compatible. Seeding alone still does **not** make a run
  reproducible: the LLM regenerates different text every time (spec §11 M4 — 0 of 5
  commands matched across two runs at the same seed); R6 reuse is what pins the draw.

**Traps (verified):**

| Trap | Location | Consequence |
|---|---|---|
| ~~`RateLimitError` → returns `[]`, dropping even the SEED utterances~~ **FIXED 2026-08-02** | `_with_retries`, `generate_diverse_utterances_with_provenance` | Now retries with a fixed five-attempt exponential-backoff policy and on terminal failure returns `[command_name] + seeds`, recording `fell_back` / `fallback_reason` per command in `training_provenance.json`. A starved command is visible now, but it is still seeds-only |
| LLM reply parsed by splitting on `[` / `]` persona headers | :368-377 | Format drift in the model's reply silently loses utterances |
| The 3 `SYNTHETIC_UTTERANCE_GEN_*` env vars are read at **module import** with `int` coercion, no defaults | :37-39 | Importing before `fastworkflow.init()` or with a sparse env file yields `None` constants, failing later and obscurely |
| No `datasets` package → seeds-only, warning logged | :446-454 | Deliberate slim-image degradation (`pip install fastworkflow[training]` to fix); also recorded as `fell_back` |

**How many seed utterances per command (`Signature.plain_utterances`)?**
Seed count is the largest lever measured anywhere in this pipeline — larger than the
persona dials above, larger than anything in the training loop. One workflow's curve, all
else held constant: 3.2 seeds/command → 46.2% held-out routing top-1, 8.0 → 70.4%, 9.3 →
73.8%, with returns flattening past roughly eight. **That is an observation from a single
160-command workflow, not a universal constant** — the shape (steep early, flattening) is
the transferable part; re-derive the number on your own workflow. Nothing in the training
output tells a developer this yet; the per-command report that would (R3b, `fix-551.4`) is
not implemented. Count what you have with `seed_utterance_count` in
`___command_info/training_provenance.json`. Vary the phrasing *family* — imperative,
question, colloquial, terse, synonym-heavy, value-bearing — rather than adding paraphrases
of the seed you already wrote.

## 4. DSPy — three distinct uses, don't conflate them

| Phase | What DSPy does | Where |
|---|---|---|
| `fastworkflow refine` | 5 `dspy.Signature` classes each in `ChainOfThought` generate field metadata, utterances, docstrings, workflow description; applied as additive-only LibCST edits | `build/genai_postprocessor.py:37-108` (signatures), :123/:151/:180-181/:217 (modules), LM = `get_lm("LLM_COMMAND_METADATA_GEN", "LITELLM_API_KEY_COMMANDMETADATA_GEN", max_tokens=2000)` :239 |
| `fastworkflow train` | NOT dspy calls — `litellm.completion` directly generates `dspy.Example(...)` literals as few-shot corpus for parameter extraction | `utils/generate_param_examples.py:312`, called with `num_examples=15, validation_threshold=0.3` from `train/__main__.py:110-114`; output `___command_info/<cmd>_param_labeled.json` |
| Run time | Dynamic Signature from the command's Pydantic `Input` model + `ChainOfThought` + `LabeledFewShot(k=len(trainset))` + `JSONAdapter` + `BestOfN(N=3, threshold=1.0)` | `utils/signatures.py:239-313` |

Run-time parameter extraction detail (`utils/signatures.py`):

- The signature docstring is generated per command, embedding field descriptions, enum
  values, `examples=[...]`, Required/Optional status, and defaults (:156-237). **This is why
  improving `Field(description=, examples=, pattern=)` metadata is THE lever for extraction
  quality** — it changes the prompt directly.
- `BestOfN`'s reward (`basic_checks` :285-292) returns 0.0 if any extracted value equals one
  of that field's `examples` — anti-parroting. Same rejection exists on the agent-mode
  regex path (`parameter_extraction.py:330-337`).
- Results are built with `model_construct(**param_dict)` — **no Pydantic validation** —
  with `NOT_FOUND` sentinels for gaps (:309-313); validation happens separately in
  `validate_parameters` (:315-670) which coerces types, checks regex `pattern`s, and runs
  `db_lookup`/`validate_extracted_parameters` hooks.
- Agent mode tries **regex XML extraction first** (`<field>value</field>` per
  `parameter_extraction.py:71-80, 296-357`); the LLM is only called if regex fails. All
  fields must be present or it falls back.
- `signatures.py:255` instantiates `dspy.LM(LLM_PARAM_EXTRACTION, ...)` directly — a legacy
  exception; every OTHER call site routes through `dspy_utils.get_lm` (so litellm_proxy
  routing works everywhere EXCEPT it also works here only if the raw model string resolves;
  see §6).
- Agent-loop `AdapterParseError` (DSPy failed to parse the LLM's structured reply) is
  retried up to **2 attempts total** at `workflow_execution_context.py:696-707`.
- The intent-clarification agent is a tool-free `ChainOfThought` over
  `IntentClarificationAgentSignature` (`intent_clarification_agent.py:11-54`).

**Parameter-example validation — fixed 2026-08-02, with narrower limitations remaining.**
Previously `generate_dspy_examples` accumulated `validated_examples` and then transformed
and returned **all** parsed examples anyway. Worse, Pydantic v2 rendered a string annotation
as `"<class 'str'>"` while the regex parser recognized only `"str"`, so validation saw no
string values and the observed zero-rejection rate was an instrumentation failure. It now
normalizes primitive/optional types and transforms only validated source examples.

A real-LLM audit over five `messaging_app_4` commands parsed all 75 requested examples and
found no obvious hallucination among 103 accepted string values. It also exposed two false
rejections of valid long broadcast messages: the fuzzy matcher considered at most five
words. The matcher now retains its 1–5-word windows for short identifiers and additionally
compares windows within two words of a long value's own length. Revalidating all 105 strings
from the audit yielded zero invalid values without relaxing threshold 0.3 for short IDs.

`fix-hru` subsequently replaced the regex field parser with an AST + `literal_eval` parser;
no model-provided code is executed. It validates primitives, lists, dictionaries,
unions/optionals, literals and enums against the Pydantic annotation, rejects malformed
values and missing required fields, and preserves omitted fields that have defaults.
Tuples, sets and arbitrary custom objects remain deliberately unsupported because they do
not round-trip through the runtime JSON-shaped example schema. Numeric grounding remains
unchanged (numbers are type-checked but accepted without proving they occur in the command).
Rejected details are returned, written with the command's accepted examples and cached; the
old process-global `rejected_examples.json` debug side effect was removed because each command
overwrote the previous file and left debris in the launch directory. Treat this file as
substantially hardened, not a general serializer for arbitrary Python objects.

**DSPy caching**: DSPy memoizes LLM calls on disk+memory. If refine/agent outputs seem
frozen after a prompt change, clear it:
`python -c "import dspy; dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)"`
in-process, or `rm -rf ~/.dspy_cache/ ./.dspy_cache/` (dir verified via
`python -m fastworkflow.utils.dspy_cache_utils status`, which reports `~/.dspy_cache` —
there is no `~/.cache/dspy`). `docs/DSPY_CACHE_GUIDE.md` documents this but has
doc rot: it references `fastworkflow.run_agent.agent_module` (module no longer exists) and
a root-level `dspy_cache_utils.py` (actually at `fastworkflow/utils/dspy_cache_utils.py`).
Cached calls also don't appear in `dspy.inspect_history()`.

More detail on all three phases: [references/dspy-and-synthetic-data.md](references/dspy-and-synthetic-data.md).

## 5. Ambiguity clarification state machine

`NLUPipelineStage` enum (`fastworkflow/__init__.py:13-18`), stored in the CME workflow's
context dict:

```
INTENT_DETECTION ──ambiguous (top-k)──> INTENT_AMBIGUITY_CLARIFICATION
INTENT_DETECTION ──out-of-scope*──────> INTENT_MISUNDERSTANDING_CLARIFICATION
(any) ──command resolved──> PARAMETER_EXTRACTION ──valid──> execute, reset to INTENT_DETECTION
```

\* out-of-scope: `wildcard.py:99-124` first walks PARENT contexts re-running prediction
before declaring misunderstanding. The walk is triggered by `command_name=None`, which
**either** reserved label produces — but only `wildcard` carries evidence that an ancestor
can help.

**There is no out-of-scope rejection anywhere in this pipeline (measured 2026-08-02).**
Probing the committed `examples/retail_workflow` `global` context directly through
`CommandRouter.predict`: out-of-scope prose predicted `wildcard` in **0 of 6** cases, and
two of the six routed *confidently* to a real command (`'book me a flight to Tokyo'` →
`IntentDetection/what_can_i_do`). `wildcard` was predicted for 3 of 4 bare values, which is
the `parameter_value` job, not rejection. Nothing is trained to recognise "this belongs to
no command at all". The only brake is the ambiguity threshold, and for retail it sits at
0.2395 / 0.3392 — anything clearing ~24–34% confidence executes. Filed as `fix-d28`; full
probe write-up in `docs/intent_training_improvements_spec.md` §11 (M1/M2).

- Suggested commands for the constrained re-selection are persisted in the app workflow's
  `___convo_info/cache.db` Rdict (`intent_detection.py:189-217`).
- In agent mode, ambiguity is delegated to the tool-free intent-clarification agent; if it
  can't resolve, it executes `abort` to reset the stage and tells the outer agent to
  `ask_user` (`workflow_agent.py:105-119, 229-259`).
- Successful clarification feeds the layer-3 learning cache (§1).
- Stale-state warning: `'command'`, `'stored_parameters'`, `'NLU_Pipeline_Stage'` in the CME
  context are reset by `Workflow.end_command_processing` (`workflow.py:291-303`); a leaked
  flag forces the wrong mode next turn.

## 6. litellm routing (per-role model selection)

One LLM role = one model env var + one key env var, resolved through
`utils/dspy_utils.py::get_lm` (:8-69). Template values are all
`mistral/mistral-small-latest` (`fastworkflow/examples/fastworkflow.env:6-11`).

| Role var | Key var | Consumer (verified call site) |
|---|---|---|
| `LLM_AGENT` | `LITELLM_API_KEY_AGENT` | `workflow_agent.py:270,309`; `workflow_execution_context.py:691` |
| `LLM_PLANNER` | `LITELLM_API_KEY_PLANNER` | `workflow_agent.py:507`; `workflow_execution_context.py:1019` |
| `LLM_PARAM_EXTRACTION` | `LITELLM_API_KEY_PARAM_EXTRACTION` | `utils/signatures.py:252-255` (direct `dspy.LM`, legacy) |
| `LLM_SYNDATA_GEN` | `LITELLM_API_KEY_SYNDATA_GEN` | `train/generate_synthetic.py:35-36`; `utils/generate_param_examples.py:333-334` |
| `LLM_COMMAND_METADATA_GEN` | `LITELLM_API_KEY_COMMANDMETADATA_GEN` (no underscore in COMMANDMETADATA) | `build/genai_postprocessor.py:239` — used by `build`/`refine`, **absent from the packaged env template**, and build/refine load NO env files (must be in OS env) |
| `LLM_CONVERSATION_STORE` | `LITELLM_API_KEY_CONVERSATION_STORE` | `run_fastapi_mcp/conversation_store.py:331` |
| `LLM_RESPONSE_GEN` | `LITELLM_API_KEY_RESPONSE_GEN` | **DEAD CONFIG** — templated and documented, zero code consumers (verified by grep) |

- **Proxy routing**: model value prefixed `litellm_proxy/` → routed via
  `LITELLM_PROXY_API_BASE` (mandatory, ValueError if unset) with optional
  `LITELLM_PROXY_API_KEY`; the per-role key var is IGNORED for proxied models (:48-65).
- **Env precedence quirk** (`fastworkflow/__init__.py:211-219`): `get_env_var` returns a
  supplied code default WITHOUT consulting `os.environ`. So `INTENT_DETECTION_TINY_MODEL`
  and `INTENT_DETECTION_LARGE_MODEL` can only be overridden via the workflow's **env
  file**, never the shell. Training seed, retry policy, report floors, and mixed top-k
  escalation behaviour are fixed trainer policy and have no environment override.
- New LLM call sites MUST use `get_lm(model_var, key_var)`, never bare `dspy.LM`.

## 7. Magic numbers: provenance is UNDOCUMENTED

No design doc, commit message, or comment justifies any of the following. **Treat every
one as empirical — "provenance unknown" — until the tau2 program (E-cards) revisits them.**
Top offenders (full table with all ~25 entries:
[references/magic-numbers.md](references/magic-numbers.md)):

| Value | What it gates | file:line |
|---|---|---|
| 0.3 | Fuzzy command-name match (max normalized Levenshtein distance) | `_workflows/command_metadata_extraction/intent_detection.py:111` |
| 0.85 | Embedding-cache cosine threshold (call site; fn default 0.90 unused) | `intent_detection.py:118` / `cache_matching.py:131` |
| 0.65 | `ModelPipeline` default confidence threshold (overwritten by trained value at load) | `model_pipeline_training.py:346,360,1059` |
| 0.15 | alpha: DistilBERT-usage penalty in threshold scoring | `model_pipeline_training.py:255` |
| 20 | linspace points swept in threshold search | `model_pipeline_training.py:227` |
| 12 / 5 | tiny / distil fine-tuning epochs | `model_pipeline_training.py:957,1020` |
| 1e-4 / 5e-5 | tiny / distil learning rates | `model_pipeline_training.py:956,1019` |
| 10 | training batch size | `model_pipeline_training.py:936,942` |
| 0.25 / 42 | test split fraction / random seed | `model_pipeline_training.py:914` |
| 0.5129 | `min_threshold` in DEAD `find_optimal_confidence_threshold` | `model_pipeline_training.py:50` |
| 15 / 0.3 | DSPy examples per command / fuzzy validation threshold (validation currently non-filtering, §4) | `train/__main__.py:113-114` |
| 0.9 / 4000 | temperature / max_tokens for DSPy example generation | `utils/generate_param_examples.py:335,407` |
| 1.0 / 0.9 / 1000 | temperature / top_p / max_tokens for utterance generation | `train/generate_synthetic.py:112-119` |
| 3 / 1.0 | BestOfN attempts / reward threshold in param extraction | `utils/signatures.py:295-299` |
| 2 | agent-call retries on AdapterParseError | `workflow_execution_context.py:700` |
| 0.2 / 0.7 | `DatabaseValidator.fuzzy_match` difflib cutoff / Levenshtein threshold | `utils/signatures.py:70,86` |
| 256 | embedding lru_cache size | `cache_matching.py:18` |

## 8. Known doc rot (trust code, cite this when correcting docs)

| Doc claim | Reality | Evidence |
|---|---|---|
| CLAUDE.md: intent models "DistilBERT/BERT via scikit-learn" | torch/transformers fine-tuning; sklearn = LabelEncoder/split/f1 only | `model_pipeline_training.py:3-17,956-957` |
| (Surprise, not rot) train-time code implies training lives in `fastworkflow/train/` | The actual training loop `model_pipeline_training.py` sits at the `fastworkflow/` package root (it is also a run-time dependency: `CommandRouter`/`ModelPipeline`) | `ls fastworkflow/model_pipeline_training.py` |
| `docs/DSPY_CACHE_GUIDE.md`: `fastworkflow.run_agent.agent_module`, root `dspy_cache_utils.py` | module gone; utility at `fastworkflow/utils/dspy_cache_utils.py` | grep |
| env template documents `LLM_RESPONSE_GEN` | zero consumers | grep `--include=*.py` |
| pyproject.toml:41 comment references `model_pipeline_training._load_tokenizer` | function no longer exists | grep |

## Provenance and maintenance

Facts verified 2026-07-09 against v2.22.2 (commit c33b9a5); passages marked 2026-08-02
re-verified against the working tree after wave 1 of epic `fix-551`. Re-verify volatile
facts:

```bash
# 2026-08-02: the two reserved labels and which one escalates
sed -n '46,77p' fastworkflow/nlu_labels.py
grep -rn "WILDCARD_LABEL\|PARAMETER_VALUE_LABEL" fastworkflow/ --include=*.py

# 2026-08-02: the escalation class is dropped where there are no ancestors
grep -n "if net_ancestor_utterances" -B 12 fastworkflow/model_pipeline_training.py

# 2026-08-02: fixed mixed top-k escalation behaviour
grep -n "Top-k escalation signal discarded" fastworkflow/_workflows/command_metadata_extraction/intent_detection.py

# 2026-08-02: held-out evaluation and internal artifact publication
grep -n "heldout_evaluation\." fastworkflow/model_pipeline_training.py
fastworkflow train --help

# 2026-08-02: rate limiting falls back to seeds rather than returning []
grep -n "fell_back\|_with_retries" fastworkflow/train/generate_synthetic.py

# Training hyperparameters (epochs/lr/batch/split) and threshold logic
grep -n "num_epochs\|lr=\|batch_size=10\|test_size" fastworkflow/model_pipeline_training.py
# Dead 0.5129 function still dead? (should only show the def, no call sites)
grep -rn "find_optimal_confidence_threshold" fastworkflow/
# Matching-ladder thresholds
grep -n "threshold=0.3\|0.85" fastworkflow/_workflows/command_metadata_extraction/intent_detection.py
grep -n "threshold=0.90" fastworkflow/cache_matching.py
# Model defaults
grep -n "INTENT_DETECTION" fastworkflow/model_pipeline_training.py
# DSPy example-gen numbers and the non-filtering return
grep -n "num_examples=15\|validation_threshold=0.3" fastworkflow/train/__main__.py
sed -n '604,613p' fastworkflow/utils/generate_param_examples.py
# Per-role LLM consumers
grep -rn "get_lm(" fastworkflow/ | grep -v "def get_lm"
# LLM_RESPONSE_GEN still dead?
grep -rn "LLM_RESPONSE_GEN" fastworkflow/ --include=*.py
# BestOfN / LabeledFewShot runtime extraction
grep -n "BestOfN\|LabeledFewShot\|JSONAdapter" fastworkflow/utils/signatures.py
# Live thresholds of any trained workflow
python .claude/skills/fastworkflow-nlu-pipeline-reference/scripts/show_intent_thresholds.py <workflow_dir>
```
