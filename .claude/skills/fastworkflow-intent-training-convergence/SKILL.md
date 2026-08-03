---
name: fastworkflow-intent-training-convergence
description: Load when tuning a workflow's intent-detection quality or judging whether one training change beat another — trigger phrases include "intent detection is weak", "should I add more utterances", "how many seed utterances per command", "did more seeds help", "training F1 is high but routing is bad", "what is in_distribution_f1", "how do I read heldout_evaluation.json", "top-1 vs in-list", "escalation recall dropped", "is this training run better than the last one", or "when do I stop retraining". Do NOT load for the statistics recipes themselves (use fastworkflow-proof-and-analysis-toolkit), for the idea-to-accepted-result lifecycle (use fastworkflow-research-methodology), for how the NLU stack is built or what a threshold means (use fastworkflow-nlu-pipeline-reference), for benchmark-file schema details (read docs/intent_benchmark_format.md), or for debugging a training crash (use fastworkflow-debugging-playbook).
---

# fastWorkflow Intent Training Convergence

Growing intent-detection quality is a measurement problem before it is a data problem.
Training reports an F1 around 0.94 on workflows whose real held-out routing accuracy is
under 50%, and two "identical" runs disagree on a large fraction of held-out cases. Both
facts have to be handled before a single utterance is added, or you will spend hours
tuning against noise.

The package now measures the right things — `fastworkflow/train/heldout_evaluation.py`
scores routing and escalation on held-out data and `fastworkflow train` prints and writes
the result. Training uses a fixed seed and reuses fingerprinted utterances and DSPy
examples, so an unchanged run does not redraw its data. Read Phase 0 before believing a
delta, and always score the same hand-written benchmark cases.

This skill is the loop: establish the floor, grow the sample, test paired, stop when the
gain drops under the floor. Statistics conventions belong to
`fastworkflow-proof-and-analysis-toolkit`; this skill is the application of them to
utterance sizing.

## When to use / when NOT to use

| Situation | Skill |
|---|---|
| Deciding whether to add seeds/personas, and how to tell if it worked | **this skill** |
| The McNemar / CI / pass^k math itself, pre-registration format | `fastworkflow-proof-and-analysis-toolkit` |
| Turning a hunch into an accepted or retired result | `fastworkflow-research-methodology` |
| How two-tier BERT, thresholds, and synthetic generation actually work | `fastworkflow-nlu-pipeline-reference` |
| The benchmark file's schema, field by field | `docs/intent_benchmark_format.md` |
| Training crashed, artifacts missing, env broken | `fastworkflow-debugging-playbook` |
| What `fastworkflow train` does operationally | `fastworkflow-run-and-operate` |

## What the package gives you now (and what it still does not)

This skill was originally written against an app-side harness, because the package
offered nothing to measure with. Wave 1 of epic `fix-551` moved most of it in-package.
Verified against the working tree 2026-08-02.

| Job | Then (external harness) | Now |
|---|---|---|
| Held-out routing + escalation score | `tests/eval_intent_benchmark.py`, app-side | `fastworkflow/train/heldout_evaluation.py`, run automatically at the end of `fastworkflow train`; printed as a table and written to `<workflow>/___command_info/heldout_evaluation.json` |
| Held-out data | hand-built benchmark only | whole-persona holdout of the generated utterances (automatic) **plus** an optional hand-written `<workflow>/intent_benchmark.json` |
| Seed/benchmark leak check | app-side assertion | run automatically before model fitting; exact collisions fail training |
| Keeping the previous run's artifacts | `mv` 8.6 GB by hand | internal atomic publication retains only `current` and `previous`; older and incomplete runs are pruned automatically |
| Seeding | nothing was seeded | fixed seed 42 inside the trainer seeds `random`/numpy/torch/transformers; it is not workflow configuration |
| Per-command provenance | none | `<workflow>/___command_info/training_provenance.json` — per command: seed, persona ids, seed-utterance count, generated count, final count, `fell_back` |
| Paired scoring of two runs | `tests/compare_runs.py`, app-side | **still not in the package.** Use `scripts/score_benchmark.py` in this skill directory |

**A module's presence under `fastworkflow/train/` is not evidence that it runs.** This was
the single most misleading thing about wave 1: five complete, unit-tested modules had no
call site anywhere outside their own file and `tests/`, so the suite was green while the
feature did not exist. Check with
`rg -n "<module>" -g '*.py' fastworkflow/ | grep -v train/<module>.py` before relying on one.

Status as of end of wave 3 (`fix-551`):

| module | wired? | where |
|---|---|---|
| `heldout_evaluation.py` (R1a/R1b) | yes | end of `train()`, plus pre-flight leak check |
| `determinism.py` (R2) | yes | `train_workflow` |
| `artifact_versioning.py` (R4) | yes | `train_workflow` |
| `utterance_cache.py` (R6) | yes | `train_workflow`, via module-level handle |
| `param_example_cache.py` (`fix-czb`) | yes | `train_workflow`, around DSPy example generation |
| `training_report.py` (R3b) | yes | `train_workflow`; structural failures prevent publication |
| `class_balance.py` (R7.2) | yes | `train()`, escalation rows only |
| `duplicate_detection.py` (R9b/F14) | yes | automatic preflight in `train_workflow`; duplicates fail before LLM generation |
| `selective_training.py` (R5) | yes | automatic in `train_workflow`; falls back to full training whenever carry-forward is unsafe |

Three traps this table will not save you from. A cache reached through a **module-level
handle** is installed by the caller, so `grep` finding the import proves nothing about
whether it is active on the path you care about — and the determinism test suites install
those handles *themselves*, so they pass whether or not `train_workflow` does.
`tests/test_cache_installation_wiring.py` is the guard for that specific blind spot.
`class_balance` is wired for `wildcard` only; its budget never binds on `parameter_value`,
which is a fixed row count that does not grow with workflow size. Selective training is an
implementation detail: unchanged contexts are reused, while missing baselines, global
changes, regeneration, or incomplete artifacts force a full retrain automatically.

Five gaps you must work around, all verified rather than assumed:

- **`fastworkflow train` has no benchmark flag.** The trainer looks for the benchmark at the fixed default path
  `<workflow>/intent_benchmark.json` (`heldout_evaluation.default_benchmark_path`). Put it
  there or it is not read.
- **Escalation and routing are scored separately.** The trainer passes the required
  benchmark kind explicitly and never blends the two axes.
- **The JSON report does not record per-case verdicts for routing** — only totals and
  per-command counts. Per-command counts cannot reconstruct which cases flipped, so you
  cannot derive an exact McNemar test from two reports. That is the other reason to use
  the script below.
- **The benchmark validators now run (`fix-eia`, fixed).** All four —
  `assert_benchmark_disjoint_from_seeds`, `find_near_duplicate_benchmark_cases`,
  `validate_routing_cases` and `validate_escalation_cases` — are called from
  `model_pipeline_training.py` before training starts. Disjointness **raises**
  `BenchmarkLeakError` and aborts the run, as `docs/intent_benchmark_format.md` always
  claimed; the other three warn. A seed/benchmark leak is still the most destructive
  mistake available in this loop, but it now fails fast instead of silently producing a
  flattering score.
- **Utterance reuse is on.** `train_workflow` installs the cache, so a fingerprint hit
  skips the LLM entirely.
  The fingerprint covers the seeds, the generation dials, the persona source, the model and
  the generator's own source, so *any* seed edit legitimately invalidates that command's
  entry and regenerates it — which is correct, and is why an edit to one command costs one
  command's worth of generation rather than the whole workflow's. Force a full redraw with
  `fastworkflow train --regenerate-utterances`, which refreshes the DSPy parameter-example
  cache too.
- **Reproducibility took two caches, not one.** Seeding alone changed nothing: 0/5 commands
  produced identical utterance sets at the fixed seed, because the LLM redraws the
  data every run. Fixing utterances was necessary but not sufficient — `generate_dspy_examples`
  is a **second** LLM path and needed its own cache (`fix-czb`). Fine-tuning itself was
  already reproducible and needed no change. When you next add an LLM call to the training
  path, assume it breaks reproducibility until measured.
- **Measure reproducibility on more than `hello_world`.** Its parameter fields are already
  in alphabetical order, so it structurally cannot detect key-ordering divergence: an early
  version of the parameter cache scored 5/5 there while leaving 3/5 artifacts differing on
  `messaging_app_4`. Use at least one workflow with non-alphabetical fields.

## Preconditions — hard gates

Do not start the loop until all four hold. Each one, skipped, invalidates every number
the loop produces.

**1. A hand-written benchmark file, not just the persona holdout.**
Persona holdout removes within-persona near-duplicates and nothing else: every persona
for a command is prompted with the same seed list, the same keyword bag and the same
command name, so both sides of the split are draws from one conditional distribution
(spec §10, AR1). It is strictly better than the random row split it replaces and it is
worth having, but it is an *in-generator* holdout. The developer-supplied benchmark is
the generalisation instrument. Write it to `<workflow>/intent_benchmark.json`; the schema
and a worked example are in `docs/intent_benchmark_format.md`. Two independent phrasings
per command is a reasonable floor.

Keep the benchmark disjoint from the seed table. Without that the benchmark decays into a
memorisation test the first time somebody pastes a failing case into their seeds to "fix"
it. The package ships the check but does not yet run it (`fix-eia`), so enforce it
yourself — `heldout_evaluation.assert_benchmark_disjoint_from_seeds(cases,
{command: seeds})` raises `BenchmarkLeakError` listing every collision, comparing on a
normalised form so `"Close the account."` collides with `"close the account"`.

**2. Both axes measured, not just routing.**
Intent detection has two jobs and they trade against each other.

- **Routing** picks the right command in the current context. Scored `top-1` (the single
  confident answer was right) and `in-list` (the right label was somewhere in the returned
  candidates). In-list-but-not-top-1 is a clarification prompt at runtime, not a correct
  route, which is why the two are never blended.
- **Escalation** returns a *lone, confident* `wildcard`, which is what sets
  `command_name=None` and drives the parent-chain walk in the CME `wildcard` command. A
  `wildcard` returned *alongside* local candidates does not escalate: it takes the ambiguity
  branch, is filtered out of the message the user sees, and is logged for diagnostics
  (finding F7). This is fixed product behaviour, not workflow configuration.

Anything that shrinks the escalation class buys routing accuracy with escalation recall.
Measure both or you will "improve" the workflow by breaking hierarchy navigation.

**3. `wildcard` no longer means "catch-all" — check which reserved label you are reasoning about.**
`fastworkflow/nlu_labels.py` now defines two reserved labels, and conflating them will
make you misread every number here.

| Label | Stage it belongs to | What it means | Trained where |
|---|---|---|---|
| `wildcard` | INTENT_DETECTION | *escalation*: "an ancestor context can serve this" | only in contexts that have ancestor utterances the context does not already own — **in a context with no ancestors the class is deliberately not emitted at all** |
| `parameter_value` | PARAMETER_EXTRACTION | *bare-value catcher*: the seven contentless literals (`"3"`, `"france"`, `"id=3636"`, …) | every context |

Both resolve to `command_name=None` and neither is ever shown to a user as a choosable
command (`NON_ROUTABLE_LABELS`). Only `wildcard` is in `ESCALATION_LABELS`, and only it
counts as a correct escalation. Consequences for this loop: a root context has no
escalation axis to measure; escalation cases in your benchmark file must target a context
that actually has ancestors (`validate_escalation_cases` checks this, but you have to call
it — see the gaps above); and artifacts trained before the split still carry the old merged
`wildcard` class, so they are not comparable to post-split artifacts on either axis.

**4. Hold the training inputs fixed.**
The fixed production seed and the two caches make unchanged inputs reusable. See Phase 0.

## Phase 0 — Establish the noise floor

**Do this before evaluating any change.** Score the current model against the hand-written
benchmark, make one change, train, and score the candidate against the *same cases*.
`fastworkflow train` reuses unchanged utterances and parameter examples and automatically
reuses unchanged contexts. If an input changed globally or reuse cannot be proven safe, it
falls back to a full retrain.

The historical warning still matters: before the caches landed, two back-to-back runs at
the same seed produced identical utterance sets for **0 of 5 commands**. The fixed seed
cannot control an LLM response. That is why deleting or bypassing either cache invalidates
a paired comparison. The often-quoted **20.6% verdict churn (92 of 446 routing cases)** was
measured before this machinery existed; treat it as historical evidence for persistence,
not the current noise floor.

The floor determines what you are able to detect at all. With `n` discordant pairs, an
exact two-sided McNemar test at α = 0.05 needs the split to be at least:

| Discordant pairs | Max losses | Min gains | Net cases | Net on a 446-case benchmark |
|---|---|---|---|---|
| 20 | 5 | 15 | 10 | 2.2% |
| 40 | 13 | 27 | 14 | 3.1% |
| 60 | 21 | 39 | 18 | 4.0% |
| 92 | 36 | 56 | 20 | 4.5% |
| 160 | 67 | 93 | 26 | 5.8% |

(Pure arithmetic, workflow-independent; every row re-derived 2026-08-02 with
`scripts/score_benchmark.py`'s `exact_mcnemar_p`, cross-checked against
`fastworkflow-proof-and-analysis-toolkit/scripts/passk_math.py mcnemar`. One more loss in
any row pushes p above 0.05.)

Read that as a budget. At 92 discordant pairs you cannot detect anything smaller than a
net 20 cases, so a change worth 10 cases is invisible no matter how many times you look at
the summary percentages.

**Small workflows are worse, not better.** On `examples/hello_world` the five commands
carried 23–38 utterances each (spec §11), so the whole training set is on the order of a
hundred rows and a 25% persona holdout leaves a few dozen cases in one context. One
flipped case is several percentage points. A benchmark of four cases cannot reach
significance at all — with 4 discordant pairs the smallest possible p is 0.125.

**Lowering the floor is usually cheaper than chasing a bigger effect.** Keep the caches,
score identical cases, or grow the benchmark. Every point of churn removed is sensitivity
gained.

## Phase 1 — Pre-register

Before the run, write down the change, the predicted direction and size on **both** axes,
and the decision you will make at each outcome. Follow the pre-registration format in
`fastworkflow-proof-and-analysis-toolkit`. The failure mode this prevents is real and
happened on the reference workflow: a per-context number moved 62.5% → 12.5%, a mechanism
was constructed to explain it, and the next run showed the context back at 50% — it had
been noise from the start.

Record the hypothesis and benchmark result outside the workflow tree. The trainer retains
only the current and immediately previous artifact sets for the duration of the comparison.

## Phase 2 — Grow the sample, one axis at a time

Ordered by measured leverage:

1. **Hand-written seed utterances per command (`Signature.plain_utterances`). The dominant
   input.** One workflow's measured curve, everything else held constant: 3.2 seeds/command
   → 46.2% routing top-1, 8.0 → 70.4%, 9.3 → 73.8%, with returns flattening past roughly
   eight. **That flattening point is an observation from a single 160-command workflow, not
   a constant** — treat it as the order of magnitude to aim for and re-derive it on your own
   workflow rather than as a target to stop at. Vary the phrasing family deliberately —
   imperative, question, colloquial, terse, synonym-heavy, value-bearing — rather than
   adding paraphrases of the seed you already have. Count what you actually have per
   command with `seed_utterance_count` in `___command_info/training_provenance.json`.
   (`training_report.py` carries the same eight as a fixed advisory,
   `DEFAULT_MIN_SEED_UTTERANCES`, and `fastworkflow train` prints thin commands.)
2. **Personas** (`SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS`) and **utterances per persona**
   (`SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA`). Cheaper than seeds and lower yield,
   because personas are still sampled from PersonaHub with no conditioning on your domain
   (spec F12; domain conditioning is R9, unimplemented). Note the interaction with
   evaluation: whole-persona holdout needs at least two personas per label to hold anything
   out, and `split_by_persona` reports labels below that in its notes.
3. **Class balance.** Where a context has ancestors, the `wildcard` class carries ancestor
   utterances not already local. The trainer caps it with coverage-preserving round-robin
   selection (R7.2, `fix-551.12`). When capping was tested it did **not** produce a
   significant accuracy change
   (p = 0.25 routing, p = 0.38 escalation) and the escalation point estimate moved the
   wrong way; what it did produce was a 44% training-time reduction. Treat imbalance as a
   cost lever, not an accuracy lever. Class weighting (R7.3) likewise has **no measured
   accuracy benefit** in this evidence base and is exploratory (spec §10, AR8).

Change **one** axis per run. With a floor this size you cannot attribute a joint change.

Do not quote the "3× the average real command class" sizing rule or the reference
workflow's `Exception`-context before/after figures: the adversarial review withdrew the
first as a fitted constant normalised against the wrong statistic (AR4) and marked the
second not-citable pending re-derivation (AR6).

## Phase 3 — Score paired, not aggregate

Comparing two summary percentages is how the reference workflow nearly shipped a "+2.7
point improvement" that was p = 0.25. Two further traps specific to the current package:

- **Do not pair the persona-holdout numbers from two `heldout_evaluation.json` files.**
  The held-out set is re-drawn each run, from freshly generated utterances. The two runs
  were scored on different cases, so there is nothing to pair. Those numbers are a
  within-run health check, not a comparison instrument.
- **Only the benchmark file gives identical cases across runs.** That is what makes a
  paired test possible at all.

The recipe, using the script in this skill directory:

```bash
S=.claude/skills/fastworkflow-intent-training-convergence/scripts/score_benchmark.py
WF=path/to/workflow
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

.venv/bin/python $S score --workflow $WF --benchmark $WF/intent_benchmark.json \
    --version previous --out "$TMP/before.json"
.venv/bin/python $S score --workflow $WF --benchmark $WF/intent_benchmark.json \
    --version current --out "$TMP/after.json"

.venv/bin/python $S compare --before "$TMP/before.json" --after "$TMP/after.json"
```

`score` loads each version's `CommandRouter` — the real runtime path, thresholds and all —
and records a per-case verdict. `compare` joins on the case, reports **baseline rate,
candidate rate, cases fixed, cases broken, and an exact two-sided McNemar p** per axis, and
always prints the broken cases. The fixed/broken pair is the informative part; the delta of
two percentages hides it. A change that nets positive while breaking a coherent family of
phrasings is usually a regression wearing a disguise.

Omit `--version` to score current. The temporary verdict files are deleted by the shell
trap; do not write experiment output into the repository.

## Phase 4 — Stop rule

Stop when any of these holds:

- The last step's net gain is smaller than the minimum detectable effect for your current
  discordant count. You are no longer measuring anything.
- Two consecutive steps fail to reach significance on either axis.
- Routing gains are being paid for entirely out of escalation recall.
- The remaining misses are concentrated in commands that are genuinely duplicate
  capabilities — two commands answering the same question cannot be separated by
  utterances, and the fix is to merge or alias them in the workflow.

When you stop, record the final numbers on both axes, the noise floor you measured, and
the changes that did *not* reach significance. The negative results are what stop the next
person repeating the loop.

## Worked example — reference workflow, 160 application commands, 33 contexts

Measured app-side before any of wave 1 existed, so the artifacts predate the
`wildcard`/`parameter_value` split and the in-package metric. The shape of the story is
what transfers, not the numbers.

| Run | Seeds/command | Routing top-1 | In-list | Escalation | Train time |
|---|---|---|---|---|---|
| baseline | 3.2 | 46.2% | 63.9% | not measured | — |
| v2 | 8.0 | 70.4% | 77.6% | not measured | — |
| v3 | 9.3 | 73.8% | 82.7% | 91.9% | 3h35m |
| v4 (wildcard capped 3×) | 9.3 | 76.5% | 84.5% | 83.8% | 2h00m |

Baseline → v2 was a large, obviously real effect: +24 points, far above any plausible
floor, no paired test needed to believe it. v2 → v3 was +3.4, already inside the floor and
never properly established. v3 → v4 was measured properly and came back **52 fixed / 40
broken, p = 0.25** on routing and **1 fixed / 4 broken, p = 0.38** on escalation — no
detectable change on either axis, and a 44% training-time reduction that was worth keeping
on its own merits.

The lesson the reference team paid nine hours of retraining for: the first seed expansion
was worth doing and everything after it was measured too imprecisely to judge. Establish
the floor first.

## Pitfalls

- **Reading per-command rates.** With two phrasings per command, a command's score is
  0%, 50%, or 100%. Commands that were never touched routinely swing 100% → 0% between
  runs. Per-command numbers are for finding candidates to investigate, never for
  concluding.
- **Pasting benchmark failures into your seeds.** The most tempting and most destructive
  move available, and right now nothing stops you: the disjointness check exists but is
  not called from the training path (`fix-eia`). Run
  `assert_benchmark_disjoint_from_seeds` yourself, in CI if you can.
- **Trusting the training F1.** It is now labelled `in_distribution_f1` in the report
  precisely so it cannot be misread, but it will still look excellent throughout. It is a
  random split over the same synthetic utterances the model trained on. It is retained
  because it is what calibrates the ambiguity thresholds — that use is fine; using it as a
  quality metric is not.
- **Assuming a fallen-back command trained on nothing.** It used to: rate limiting returned
  `[]` and the label never entered the classifier. That is fixed — generation now retries
  with exponential backoff and, on terminal failure, returns `[command_name] + seeds`, with
  `fell_back` and `fallback_reason` recorded per command in `training_provenance.json`.
  Check that file rather than grepping the log. A fallen-back command is not absent, but it
  is seeds-only, so its row count and its held-out score are both suspect.
- **Editing seeds for one command and comparing "everything else unchanged".** Ancestor
  utterances form descendant contexts' escalation class, so changing command `X`'s seeds
  changes the training data of other contexts too — and the closure is wider than the
  parent chain: ancestor commands are collected through the `base`-resolved `commands()`,
  so a command reaches a context's `wildcard` class through *any* ancestor that inherits
  it, whether or not its home context is on that context's parent chain (spec §10, AR2).
- **Comparing artifacts from either side of the `wildcard` split.** Pre-split artifacts
  train the seven bare-value literals into the escalation class. Their escalation numbers
  are not comparable to post-split ones.
- **Expecting `wildcard` to reject out-of-scope input.** It does not, and nothing else
  does either. Probing the committed retail `global` context, out-of-scope prose predicted
  `wildcard` in 0 of 6 cases and two of them routed confidently to a real command (spec
  §11, M1/M2; filed as `fix-d28`). Do not write "should be rejected" benchmark cases and
  score them as escalation — they are testing a behaviour the runtime does not have.
- **Leaving experiment debris in the repository.** Use a temporary workflow copy and a
  temporary output directory. Do not create repo-level measurement scripts, benchmark
  copies, or result JSON. The package retains only current and previous artifacts while a
  comparison is active.

## Phase 5 — Mandatory cleanup

After accepting the candidate, remove its comparison baseline and every temporary file.
The reusable scorer stays inside this skill; run-specific scripts and results do not.

```bash
rm -rf "$TMP"
.venv/bin/python - "$WF" <<'PY'
import sys
from fastworkflow.train import artifact_versioning

workflow = sys.argv[1]
artifact_versioning.retain_current_and_previous(workflow, previous_version=None)
PY
```

If the candidate is rejected, publish `previous` internally, delete the rejected candidate,
then run the same cleanup. Never leave `docs/experiments/`, repo-level `scripts/`, copied
benchmarks, or score JSON behind.

## Provenance and maintenance

Facts verified 2026-08-02 against the working tree after wave 1 of epic `fix-551` landed
and was integrated. The measured numbers in the worked example and in Phase 2 come from an
external 160-command workflow and are recorded in
`docs/intent_training_improvements_spec.md` §2; the post-integration measurements (M1–M4)
are in that document's §11, and the adversarial review that withdrew several previously
quoted figures is §10.

**Explicitly not verified here**, and therefore not claimed above: whether the historical
20.6% verdict-churn figure persists on a different machine or model stack, and whether the
whole-persona holdout's residual leak is small. Use the hand-written benchmark for claims.

Re-verify volatile claims:

```bash
# Held-out evaluation is wired into the trainer, and where the report is written
grep -n "heldout_evaluation\." fastworkflow/model_pipeline_training.py

# Training uses a fixed seed, and personas are no longer an unseeded random.sample
grep -n "def seed_everything\|def get_training_seed" fastworkflow/train/determinism.py
grep -n "select_persona_indices\|random.sample" fastworkflow/train/generate_synthetic.py

# Rate limiting falls back to seeds instead of returning []
grep -n "fell_back\|fallback_reason" fastworkflow/train/generate_synthetic.py

# The two reserved labels and which one escalates
sed -n '46,77p' fastworkflow/nlu_labels.py

# The escalation class is dropped where there are no ancestors
grep -n "if net_ancestor_utterances" -B 12 fastworkflow/model_pipeline_training.py

# A mixed top-k escalation label follows fixed conservative behaviour
grep -n "Top-k escalation signal discarded" \
     fastworkflow/_workflows/command_metadata_extraction/intent_detection.py

# Train owns validation, selective/full planning, publication and retention
fastworkflow train --help

# The reported F1 is still a split over the training distribution
grep -n "train_test_split" fastworkflow/model_pipeline_training.py

# Which modules are actually wired in (empty output = present but inert).
# Everything below should print a production call site.
for m in utterance_cache param_example_cache training_report class_balance \
         duplicate_detection selective_training; do
  echo "== $m"; rg -n "$m" -g '*.py' fastworkflow/ | grep -v "train/$m.py"
done

# A module-level handle can be imported and still never installed on the path you care
# about, and the determinism suites install these themselves -- so grep is not enough.
.venv/bin/python -m pytest tests/test_cache_installation_wiring.py -q
```
