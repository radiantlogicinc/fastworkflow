# Out-of-Scope Utterance Rejection: Design Proposal

> **Status:** UNRESOLVED DESIGN PROPOSAL; `fix-d28` REMAINS OPEN. The soft fallback
> wording is implemented, but no satisfactory out-of-scope solution has been established.
> `ErrorCorrection/you_misunderstood.py` now presents the capability list without
> presupposing that one listed command must be correct, and offers rephrasing as well as
> aborting. No out-of-scope detector, negative class, confidence gate, or routing change is
> implemented. House lifecycle stage 1 of 7 (`fastworkflow-docs-and-positioning` §6); this
> proposal has **not** been adversarially reviewed and carries no supremacy clause — where
> this document and the code disagree, the code wins.
> **Correction A1 (2026-08-02) is load-bearing.** A current-tree audit plus an isolated
> live retail turn disproved §1.3's account of the ambiguous first turn and found that the
> proposed "end of parent walk" gate cannot see confident misroutes. Section 10 preserves
> the original proposal as design history and states the corrected minimal contract. No
> detector should be implemented from §§3.1, 6.1, or 7.2 without applying A1.
> **Correction A2 (2026-08-02) tightens the evidence bar further.** The local legacy model
> artifacts are gitignored and not reconstructible from the pinned commit; the threshold
> sweep reused its evaluation cases for calibration; and the advertised direct-rephrase
> recovery path does not exist. Section 11 supersedes A1's "next code slice" and requires a
> runtime shadow measurement on fresh post-split versioned artifacts before any gate.
> **Correction A3 (2026-08-02) reframes the problem.** Fresh command-specific probes show
> that neither the global plain-utterance bank nor a predicted-command variant separates
> well enough. Section 12 retires "detect the complement of the workflow" as the primary
> design goal and proposes proof-carrying selective execution, deterministic risk controls,
> and negative-feedback learning. These are candidates, not accepted solutions.
> **Tracks:** beads `fix-d28`. Related: `fix-551.11` (R7.1 label split), `fix-2m5`
> (`parameter_value` competition), `fix-551.5` (R1a held-out evaluation).
> **Pinned against:** v2.23.0, commit `23dbe35`, with the committed `___command_info`
> artifacts for `fastworkflow/examples/retail_workflow` and
> `fastworkflow/examples/hello_world`.
> **Citation convention:** `file:line` refers to **`git show 23dbe35:<file>`**, not the
> working tree. The tree is dirty with in-flight R1/R2/R3/R7 work and
> `model_pipeline_training.py` and `intent_detection.py` are offset by hundreds of lines.
> Two exceptions, stated where they occur: `fastworkflow/cache_matching.py` and
> `fastworkflow/utils/signatures.py` are clean (identical either way), and
> `fastworkflow/train/heldout_evaluation.py` is untracked, so its line numbers are
> working-tree only. `_commands/wildcard.py`'s cited ranges (65–90, 99–124, 140–165) were
> checked line-by-line and are identical in both.
> **Measurement validity under the dirty tree:** every diff hunk in
> `model_pipeline_training.py` falls in imports, `get_artifact_path`,
> `cache_ancestor_utterances`, or `train()`. `CommandRouter`, `ModelPipeline`,
> `predict_single_sentence` and `find_optimal_threshold` — the entire inference path
> measured below — are byte-identical to `23dbe35`.
> **The measured artifacts predate the R7.1 label split — added by the integrator after
> this document was written.** `examples/retail_workflow/___command_info` contains the
> label `wildcard` 11 times and `parameter_value` zero times, so every probe below ran
> against the OLD single-overloaded-label model. This does **not** undermine the AUC
> results in §1, which are properties of the confidence distribution rather than of label
> names, and it is the direct explanation for "11 of 12 bare values predicted `wildcard`":
> under current code those route to `parameter_value` instead. But any *label-specific*
> claim here must be re-checked against freshly trained artifacts before it is relied on,
> and the AUC numbers should be re-measured post-split before the gate is built, since the
> split changes what the reserved classes absorb.
> **Not committed.** Per the never-commit rule, this file requires Dhar's explicit
> same-turn request before `git add`.

---

## Contents

1. [Problem statement, with measurements](#1-problem-statement-with-measurements)
2. [The design space](#2-the-design-space)
3. [Recommendation, and what would falsify it](#3-recommendation-and-what-would-falsify-it)
4. [The false-positive cost](#4-the-false-positive-cost)
5. [How this would be measured](#5-how-this-would-be-measured)
6. [Interaction with the existing reserved labels](#6-interaction-with-the-existing-reserved-labels)
7. [Scoping estimate](#7-scoping-estimate)
8. [Explicitly not established](#8-explicitly-not-established)
9. [Provenance and re-verification](#9-provenance-and-re-verification)
10. [Amendment A1 — correct the runtime contract before implementation](#10-amendment-a1--correct-the-runtime-contract-before-implementation)
11. [Amendment A2 — no-go until runtime shadow evidence exists](#11-amendment-a2--no-go-until-runtime-shadow-evidence-exists)
12. [Amendment A3 — proof-carrying selective execution](#12-amendment-a3--proof-carrying-selective-execution)
- [Appendix A — the 50 out-of-scope utterances, with measured outcomes](#appendix-a--the-50-out-of-scope-utterances-with-measured-outcomes)

---

## 1. Problem statement, with measurements

### 1.1 The claim under test

`fix-d28` asserts: *irrelevant utterances route confidently and silently to real commands,
and there is no mechanism to reject them.* The mechanism half of that is true by
inspection — nothing in `fastworkflow/` is trained on, or tests for, "this utterance
belongs to no command anywhere". The severity half needed measuring, and the measurement
changes the picture materially.

### 1.2 What was measured

Two probe runs against the **already-committed** trained artifacts, read-only, calling
`CommandRouter.predict` and `ModelPipeline.predict_batch` directly. No training was run.

| | `examples/retail_workflow` | `examples/hello_world` |
|---|---|---|
| Context probed | `global` (root, no ancestors) | `global` (root, no ancestors) |
| Label space | 20 labels: 15 app commands, 4 `IntentDetection/*` meta-commands, `wildcard` | 6 labels: 1 app command, 4 `IntentDetection/*`, `wildcard` |
| `threshold.json` (tiny→distil escalation) | 0.2395 | 0.5216 |
| `tiny_ambiguous_threshold.json` | 0.2395 | 0.5216 |
| `large_ambiguous_threshold.json` | 0.3392 | 0.6365 |

Note both committed artifact sets **predate** the `fix-551.11` root-context ruling: their
label space still contains `wildcard`. That is the favourable case for the "did we lose a
safety net" question — it lets us measure the pre-split state directly.

**In-scope set (retail, n=72):** a hand-written, seed-disjoint E-PV1 benchmark in the
`intent_benchmark.json` schema: four cases per app command, three per meta-command.
Programmatic disjointness check against the seed table: **0 leaks**. The temporary
benchmark copy was removed after the experiment; durable results remain in this document.

**Out-of-scope set (n=50):** written for this probe, across seven categories — general
knowledge (8), weather/time (4), other-domain tasks (9), coding (3), chit-chat (7), advice
(4), and *near-domain* (15): questions a real user might plausibly put to a retail
assistant that no command in this workflow serves ("what are your store opening hours",
"do you price match with your competitors", "is my data being sold to advertisers"). The
near-domain block is the adversarially hard half and is deliberately over-represented. All
50, with their measured outcomes, are in [Appendix A](#appendix-a--the-50-out-of-scope-utterances-with-measured-outcomes).

### 1.3 Result: current behaviour

```
out-of-scope routed confidently: 20/50 = 40%   95% CI [28%, 54%]
in-scope routed confidently:     63/72 = 88%
in-scope benchmark top-1:        45/72 = 62.5%
in-scope benchmark in-list:      51/72 = 70.8%
out-of-scope predicted 'wildcard': 0/50
```

`fix-d28`'s 2-of-6 figure generalises: **40% of out-of-scope utterances produce a single
confident prediction that `intent_detection.py:124` routes with no prompt.**

The other 60% fall below the ambiguity threshold, and what happens to them **in a root
context** is worth stating precisely, because it is not what the ambiguity machinery's
name suggests. `intent_detection.py` builds the three-option "The command is ambiguous.
Please choose…" message, stores the candidates, and returns `command_name=None`. In
`_commands/wildcard.py:99-124` that enters the parent walk; retail's
`context_inheritance_model.json` is empty, so `is_command_context_for_response_generation_root`
is true, the walk body never executes, and control reaches line 111 — `# out of scope
commands` — which runs `ErrorCorrection/you_misunderstood`. The three-option message is
discarded on this turn; what the user actually sees is `you_misunderstood`'s **full command
list**: "Please select the correct command name from the list below: … or type 'abort' to
cancel."

That is established by code trace, not by a runtime transcript, and should be confirmed
against a live turn before it is relied on. If it holds, the practical consequence is
that the ambiguous 60% already receive something close to the right out-of-scope
response — a capability list — and only the wording is wrong: it asserts that one of the
listed commands *is* what the user meant.

Where they land, by prediction:

| Landing label | All 50 | Of the 20 confident |
|---|---|---|
| `IntentDetection/what_can_i_do` | 10 | 6 |
| `IntentDetection/what_is_current_context` | 9 | 3 |
| `IntentDetection/reset_context` | 3 | 2 |
| `IntentDetection/go_up` | 4 | 1 |
| `list_all_product_types` | 6 | 2 |
| `get_user_details` | 2 | 2 |
| `modify_pending_order_items` | 3 | 1 |
| `transfer_to_human_agents` | 3 | 1 |
| `modify_pending_order_payment` | 2 | 1 |
| `calculate` | 1 | 1 |
| everything else | 7 | 0 |

**26 of 50 out-of-scope utterances are absorbed by the four CME meta-commands.** On
`hello_world` it is **26 of 26** — every single one.

### 1.4 The finding that changes the severity: blast radius

`fix-d28` says irrelevant utterances "route confidently and silently to real commands."
They route confidently. What happens next is not what the wording implies.

Two structural facts gate execution:

- A CME command (`is_cme_command=True`, which is every `IntentDetection/*`) is executed
  **immediately**, with no parameter stage —
  `_commands/wildcard.py:77-89`.
- An application command goes through `ParameterExtraction.extract()`
  (`_commands/wildcard.py:150-164`). A field whose extracted value is `NOT_FOUND` is
  reported missing by `utils/signatures.py:337-340`, clean and the turn becomes a parameter prompt,
  not an execution.

Classifying the 20 confident mis-routes by that gate:

| Outcome | Count | Commands |
|---|---|---|
| CME meta-command, executes immediately, read-only or navigational | 12 | `what_can_i_do` ×6, `what_is_current_context` ×3, `reset_context` ×2, `go_up` ×1 |
| App command with a **zero-field** `Input`, executes immediately, read-only | 2 | `list_all_product_types` ×2 |
| App command with ≥1 required field; utterance contains no extractable value, so the turn becomes a parameter prompt | 6 | `get_user_details` ×2 (1 field), `calculate` (1), `modify_pending_order_payment` (2), `modify_pending_order_items` (4), `transfer_to_human_agents` (1, default `NOT_FOUND`) |

**Zero of 50 out-of-scope utterances cause a state mutation.** Not one reaches a write.
The single most common outcome — 6 of 20 confident cases, 10 of 50 overall — is
`what_can_i_do`, which replies with the workflow's capability list. That is, arguably,
the *correct* response to an out-of-scope utterance, arrived at by accident.

The genuinely bad outcomes in the confident set are `reset_context` ×2 (the French
Revolution case from `fix-d28`) and `go_up` ×1: navigation side effects that are cosmetic
at the root but would be disorienting deeper in a context hierarchy. Everything else is
either a capability listing, a read, or a parameter prompt.

This does not make the gap acceptable — a retail assistant that answers "what does a
segmentation fault mean" with a product catalogue is still broken, and in agent mode a
wrong tool result enters the trajectory and can be reasoned from. But it is a **UX and
trajectory-quality defect, not a safety defect**, and it should be priced accordingly.
See §3.3 for the conditions under which that reclassification is wrong.

### 1.5 The decision-relevant question: does confidence already separate?

This is the question the whole design hangs on. If the classifier's confidence already
separates in-scope from out-of-scope, the fix is a threshold and costs almost nothing.

It does not. Measured as AUC — the probability that a randomly chosen in-scope utterance
scores higher than a randomly chosen out-of-scope one; 0.5 is coin-flipping, 1.0 is
perfect separation:

| Score | retail (n=72/50) | hello_world (n=7/26) |
|---|---|---|
| classifier max softmax probability | **0.788** | **0.533** |
| classifier top1−top2 margin | 0.772 | 0.571 |
| max cosine to the seed-utterance embedding bank | **0.911** | 0.604 |

Normalised entropy was tested too, on an earlier smaller probe (24 in-scope / 26
out-of-scope): AUC 0.718 on retail and 0.495 on hello_world. No better than max
probability; not carried forward.

On retail the two distributions overlap heavily: in-scope confidence runs
`min 0.183 / median 0.450 / max 0.896`, out-of-scope runs `median 0.265 / max 0.715`.
**Only 15 of 72 in-scope cases (21%) score above the highest out-of-scope confidence**;
the other 79% sit inside the out-of-scope range. The highest out-of-scope score, 0.7152,
belongs to "what is your carbon footprint policy" → `what_can_i_do`, and it beats 57 of
the 72 valid utterances. The best in-sample operating point on
`max_prob` with zero observed false positives is `t = 0.183`, which rejects **9 of 50 =
18%** of out-of-scope utterances — barely better than the shipped threshold already
achieves as a side effect. Buying more costs real false positives:

```
budget fp<= 0.0%: t=0.1830  rejects  9/50 = 18% [95% CI 10%,31%]  fp= 0/72
budget fp<= 5.6%: t=0.2327  rejects 18/50 = 36% [95% CI 24%,50%]  fp= 4/72
budget fp<=10.0%: t=0.2593  rejects 25/50 = 50% [95% CI 37%,63%]  fp= 6/72
```

On `hello_world` there is **no signal at all** (AUC 0.533, i.e. indistinguishable from
chance) and no threshold rejects anything without rejecting in-scope traffic first.

**Answer: no. Confidence does not already separate in-scope from out-of-scope. A
threshold-only fix is not available.**

### 1.6 Why `hello_world` is degenerate, and what that teaches

`hello_world` is not simply "a smaller version of retail". Its label space is 83%
meta-commands (4 of 6 non-reserved labels are `IntentDetection/*`), and its seed bank is
3 application utterances against 24 meta-command utterances. Splitting the embedding bank
by origin makes the mechanism visible:

| Seed bank | retail AUC | hello_world AUC |
|---|---|---|
| all commands | 0.911 | 0.604 |
| application commands only | 0.856 | 0.615 |
| CME meta-commands only | **0.348** | 0.604 |

The CME-only figure is *below* 0.5 — **inverted**. Out-of-scope prose is, on average, more
similar to `what can you do? / now what? / where am I?` than genuine in-scope retail
requests are. That is the mechanism behind §1.3's landing table stated as a number, and it
is a permanent structural property, not a training artifact: the meta-commands are
domain-free conversational English, and so is most out-of-scope input.

Two consequences carried forward:

- Out-of-scope detection gets **structurally harder as a workflow gets smaller**, because
  the fixed four meta-commands come to dominate the label space. Any proposal validated
  only on a large workflow will look better than it is.
- The `hello_world` probe also showed that all three held-out `add_two_numbers`
  paraphrases were predicted as `go_up`; the workflow's only application command scores
  0/3 top-1. That is an instance of the cross-run flakiness already recorded on `fix-2m5`,
  not a new defect — but it means `hello_world` cannot serve as a test bed for this
  feature. **All acceptance measurement must use `retail_workflow` or larger.**

### 1.7 Independent confirmation that `wildcard` was never an out-of-scope sink

The history in `fix-551.11` and spec §11 (M1/M2) says `wildcard` is an escalation signal
and never functioned as an out-of-scope detector, based on 0-of-6. Re-measured
independently here against the same pre-split artifacts, at larger n and on two workflows:

```
retail:      out-of-scope prose predicted 'wildcard':   0/50
hello_world: out-of-scope prose predicted 'wildcard':   0/26
retail:      bare values predicted 'wildcard':          5/6   (all but "3", which went to `calculate`)
hello_world: bare values predicted 'wildcard':          6/6
```

**0 of 76.** The claim holds and is stronger than originally stated. Dropping
`WILDCARD_LABEL` in root contexts removed nothing that was working, and "just put wildcard
back" is not an option — it would restore a class whose measured behaviour is a bare-value
catcher, which is precisely what `PARAMETER_VALUE_LABEL` now is.

### 1.8 One layer ruled out

`intent_detection.py:108` runs a fuzzy Levenshtein pre-match against command names at
threshold 0.3 *before* the classifier. It was a plausible additional source of
mis-routing. It fired on **0 of 95** probe utterances in either workflow: normalised
edit distance between prose and a command name is always far above 0.3. It is not a
contributor and no proposal here needs to touch it.

---

## 2. The design space

Five options. A and B are the ones `fix-d28` proposes; C is the one this document
recommends; D and E exist to keep the comparison honest.

### Option A — calibrate or raise the ambiguity threshold

Reject when `max_prob` falls below a threshold chosen as a real operating point rather
than, as today, the mean confidence of misclassified test rows
(`model_pipeline_training.py:1174-1190`).

**What it costs:** almost nothing. The artifact
(`tiny_ambiguous_threshold.json` / `large_ambiguous_threshold.json`) and the reader
(`CommandRouter.__init__`, `model_pipeline_training.py:298-312`) already exist. No new class, no new model, no retrain
of the classifier itself.

**Why it is not sufficient:** §1.5. AUC 0.788 on retail and 0.533 on hello_world. The
zero-false-positive operating point buys 18% detection, and the shipped threshold already
sits within 0.06 of it. There is no free detection left in this signal.

**What is still worth doing:** `fix-d28`'s complaint that the thresholds are a descriptive
statistic rather than a decision boundary is correct and is a real defect *for its own
reasons* — it makes the ambiguity prompt rate uncontrolled and un-comparable across runs.
That fix belongs to `fix-551.5`/R1a and should be pursued **separately, and not sold as
out-of-scope rejection.** Conflating the two is the main risk this option carries: it
looks like progress and delivers 18%.

### Option B — a dedicated negative class

Add a third reserved label, `out_of_scope`, trained on out-of-domain utterances, alongside
`WILDCARD_LABEL` and `PARAMETER_VALUE_LABEL`. This is `fix-d28`'s direction 1 and the
natural extension of R7's taxonomy.

**The attraction is real.** It needs no new runtime component: the label falls out of the
existing softmax, `NON_ROUTABLE_LABELS` already exists to keep it out of command lookups,
and `is_non_routable()` already routes it to `command_name=None`. If it worked it would be
the cleanest fit to the architecture.

**Cost 1 — where the negatives come from.** `generate_synthetic.py` can produce them via
the existing PersonaHub path, but the prompt has to say *what to generate*, and there are
only two shapes available. Generic out-of-domain prose lands exactly where the CME
meta-commands live (§1.6: CME-bank AUC 0.348) — the class would be carved out of the
region `what_can_i_do` occupies, and would cannibalise it. Near-domain negatives
("questions a user might ask this assistant that it cannot answer") are the ones that
matter, but generating them requires the LLM to reason about the complement of the command
set, and every generation error is an in-scope utterance labelled `out_of_scope` — a
training-time false positive, permanently baked into the model, in the direction §4 says
is the expensive one.

**Cost 2 — `fix-2m5`.** That issue records `PARAMETER_VALUE_LABEL`, backed by 7 contentless
literals, out-competing real commands in small workflows: `add_two_numbers` lost to it on
2 of 5 seeds. Reserved classes compete for probability mass with real commands. A third
reserved class, trained on a *large* and *diverse* corpus (an out-of-scope class needs
breadth to generalise), competes harder than a 7-row class does. `fix-2m5` is open and
unfixed; adding a bigger competitor before it is resolved is the wrong order.

**Cost 3 — the definition does not compose.** In a context with ancestors, "out of scope"
is not the complement of the local label space — that is what `WILDCARD_LABEL` means.
It is the complement of the union over the whole ancestor chain, which is not knowable at
the point where the per-context classifier is trained. Either the class means different
things in root and non-root contexts (the exact defect F6/R7.1 just finished removing from
`wildcard`), or the decision has to be deferred to the end of the parent walk — at which
point it is no longer a classifier label. This is the deepest objection and it is
structural, not empirical.

**Cost 4 — measurement.** Adding a class changes every context's label space, so every
number in every prior training run becomes incomparable, against a documented noise floor
of 20.6% verdict churn between identical runs (spec §2).

**Not rejected forever.** If Option C's gate proves insufficient *and* `fix-2m5` is fixed
*and* a near-domain negative corpus exists with human-verified labels, B becomes the right
answer for root contexts specifically. It is not the right first move.

### Option C — open-set gate on distance to the seed manifold (recommended)

Score an utterance by its maximum cosine similarity to the embeddings of the
context's seed utterances. Reject below a per-context threshold. This is a post-hoc gate
*outside* the softmax: it adds no class and changes no label space.

**Measured on retail** (72 in-scope benchmark cases, 50 out-of-scope; embedding is
`cache_matching.get_embedding`, the fine-tuned DistilBERT `[CLS]` vector the runtime
already computes):

```
AUC = 0.911    in-scope min=0.5715 median=0.8091
               out-of-scope median=0.6122 max=0.8090

budget fp<= 0.0%: t=0.5715  rejects 17/50 = 34% [95% CI 22%,48%]  fp=0/72
budget fp<= 1.4%: t=0.6146  rejects 26/50 = 52% [95% CI 39%,65%]  fp=1/72
budget fp<=10.0%: t=0.6493  rejects 30/50 = 60% [95% CI 46%,72%]  fp=7/72

leave-one-out (threshold refit on the other 71 in-scope cases):
  false positives 1/72 = 1.4% [95% CI 0%, 7%];  out-of-scope rejection mean 34%
```

**The marginal runtime cost is close to zero, and this is the strongest practical argument
for it.** `intent_detection.py:118` already calls `cache_match(...)`, which calls
`get_embedding(command, modelpipeline)` — the same DistilBERT `[CLS]` vector the gate
needs — on every intent-detection turn *whose utterance cache is non-empty*
(`cache_matching.py:154-158`, clean returns early on an empty cache, so a never-corrected
workflow does not yet pay it). Where it is already paid the gate is free, thanks to the
existing LRU (`_cached_embedding`, maxsize 256); where it is not, the worst case is one
extra DistilBERT forward pass, alongside the one-to-two transformer passes
`CommandRouter.predict` already performs on the same turn. On top of that: one cosine
against ~100 stored vectors, and a `float32[N, 768]` artifact per context — roughly
350 KB for retail's 117 seeds, against the ~276 MB per context the package already
writes (spec F11).

**What it costs honestly:**
- Another per-context artifact to write, version (`train/artifact_versioning.py`),
  invalidate, and keep in sync with the seed table. A stale bank silently degrades the
  gate; the fingerprinting from `fix-551.9`/R6 is the right mechanism and already exists.
- A per-context threshold, which has to be calibrated against labelled negatives — the
  same benchmark-provenance problem as Option B (§5), but at *evaluation* time only, where
  a labelling error is visible and fixable, rather than at training time where it is baked
  into weights.
- It fails on meta-command-dominated workflows (hello_world AUC 0.604). It must ship
  **off by default**, opt-in per workflow, with the gate refusing to arm unless calibration
  demonstrates separation.
- 34–52% detection is not "solved". It halves the problem at best.

### Option D — do nothing, and document the recovery path

Given §1.4 — zero mutations, 12 of 20 confident mis-routes answered by a meta-command,
6 stopped at a parameter prompt — the status quo's cost is much lower than `fix-d28`'s
framing implies. Doing nothing costs zero engineering and zero false-positive risk.

**Why not:** it leaves the agent-mode trajectory-poisoning question unanswered (a wrong
`list_all_product_types` result *is* consumed by the planner), and it leaves the 60% of
out-of-scope utterances that reach `you_misunderstood` being told to "select the correct
command name" from a list that does not contain what they asked for (§1.3). But D is the
correct baseline to beat, and any proposal that cannot beat it on measured numbers should
not ship. Note that most of D's remaining sting is removed by Option E alone.

### Option E — make the accidental behaviour deliberate

The system's *de facto* out-of-scope response is already "here is what I can do":
`what_can_i_do` is the modal landing label, 10 of 50 overall and 6 of 20 confident. E makes
that explicit — when nothing routes, answer with the capability list framed as a
non-answer, rather than with `ErrorCorrection/you_misunderstood`'s clarification loop.

This is nearly free and it is **complementary, not alternative**: it is the *action* half
of any detection mechanism, and §4 argues it is the action that makes a detector shippable
at all. It is folded into the recommendation below rather than competing with it.

---

## 3. Recommendation, and what would falsify it

### 3.1 The recommendation

**Three separable pieces, in this order.**

1. **Reclassify `fix-d28` from p1 to p2** and rewrite its severity paragraph with §1.4's
   blast-radius evidence. It is a UX and trajectory-quality defect. Nothing about it
   should block a release.

2. **Adopt Option E now (soft action), independent of any detector.** The parent walk
   already ends at `ErrorCorrection/you_misunderstood` with a full command list
   (`_commands/wildcard.py:109-124`, already commented `# out of scope commands`), so this
   is a **wording change, not a new path**: "Please select the correct command name from
   the list below" presupposes that one of them is what the user meant. It should be able
   to say instead "I don't think that's something I can help with here — what I can do
   is…". Small, no false-positive risk, and it is the exact response an armed detector
   would need in piece 3, so building it first de-risks that work.

3. **Prototype Option C behind an opt-in flag, defaulted off, and gate the decision on
   measurement.** Build the seed-embedding bank as a training artifact, add the gate at the
   *end of the parent walk*, and calibrate the threshold on a benchmark that includes
   out-of-scope cases (§5). Arm it for a workflow only when calibration shows the gate
   clears a pre-registered bar (§5.3).

**Explicitly not recommended now:** Option B. Revisit only when `fix-2m5` is closed and
§2-B cost 3 (the composition problem) has an answer.

### 3.2 Why C over B

Three reasons, in decreasing order of weight:

- **It adds no class.** `fix-2m5` is an open, unfixed demonstration that reserved classes
  cannibalise real commands. C sits outside the softmax and cannot.
- **The composition problem dissolves.** A distance gate evaluated once, at the end of the
  parent walk, is exactly "no context in the chain recognises this" — the definition
  Option B cannot express as a per-context label.
- **It is measured better and cheaper.** 0.911 vs 0.788 AUC on the same data; the
  embedding is already computed; a bad threshold is one JSON edit, a bad negative class is
  a retrain.

### 3.3 What would change my mind

Stated as falsifiable conditions, with the direction each pushes:

| Evidence | Consequence |
|---|---|
| A workflow with a **zero-parameter mutating command** (`delete_all_drafts()`, `submit_order()`) reachable from a root context | §1.4's blast-radius argument collapses for that workflow. `fix-d28` returns to p1 or p0. A conservative gate becomes mandatory, not opt-in. This is the single most likely way I am wrong, and it is a property of the *user's* workflow, not of retail. |
| Agent-mode measurement showing a wrong tool result materially degrades task success | Raises priority; strengthens C; does not change the choice between C and B. |
| Option C AUC below ~0.80 on a second realistic (non-`hello_world`) workflow | C is not generalising. Fall back to D + E, and re-open B. |
| Option C false-positive rate above ~2% on a 200+-case benchmark | C is unshippable at any detection rate (§4). |
| `fix-2m5` closed *and* a human-verified near-domain negative corpus exists | B becomes viable for root contexts; C and B could be combined (gate ∧ class). |
| A second run of the same experiment moves the numbers by more than ~20 points | Everything in §1 is inside the documented noise floor and none of it is interpretable. See §8. |

---

## 4. The false-positive cost

**A rejected valid utterance is worse than a misrouted one, and the asymmetry is large.**

A misroute is recoverable through machinery that already exists. Route to the wrong
command and the user gets a parameter prompt, a clarification prompt, or a wrong-but-
readable answer, and can restate; `ErrorCorrection/you_misunderstood` and the ambiguity
loop are built for exactly this. §1.4 measured that even the *confident* misroutes bottom
out at a prompt or a read. A rejection is a dead end: the user asked for something the
system can do, was told it cannot, and has no affordance to appeal. Repeat that twice and
the user stops trusting the system on requests it *can* serve — which costs more than the
original misroute did.

**What that bias costs in detection rate, measured.** On retail, forcing zero observed
false positives on 72 in-scope cases drops Option C from 52% detection to 34%:

```
fp = 0/72 : rejects 17/50 = 34%
fp = 1/72 : rejects 26/50 = 52%
```

That is 18 percentage points of detection, more than a third of the achievable total,
bought with a single false positive. The trade is steep and there is no operating point
that is both conservative and effective.

**And zero observed false positives is not zero.** With 0 events in 72 trials the
rule-of-three 95% upper bound on the true rate is 4.2%; leave-one-out refitting produces
1 false positive in 72 (1.4%, 95% CI [0%, 7%]). A 1–4% rejection rate on valid traffic is
**not acceptable for a hard reject**, whatever the detection rate.

**The design consequence is that the action must be soft, not hard.** Do not refuse.
On rejection, answer with the capability list framed as a non-answer (Option E) and leave
the turn open. Then:

- a **true** positive gets the best available out-of-scope response — the same one the
  classifier already produces by accident 20% of the time;
- a **false** positive costs the user one extra turn, because the capability list names the
  command they wanted and they can restate. That is the same cost as an ambiguity prompt,
  which the system already imposes on 12% of valid in-scope traffic (9/72 in-scope cases
  fall below the ambiguity threshold today).

With a soft action the false-positive cost collapses from "dead end" to "one wasted turn",
and a 1–4% false-positive rate becomes tolerable. **This is what makes the feature
shippable, and it should be treated as a precondition of shipping it, not a refinement.**
If a future variant proposes a hard refusal, the false-positive bar goes back up by an
order of magnitude and none of the numbers in §2-C clear it.

---

## 5. How this would be measured

### 5.1 Build on `heldout_evaluation.py`, do not reinvent

`fastworkflow/train/heldout_evaluation.py` already provides the report model
(`HeldoutReport`), two scorers (`score_routing`, `score_escalation`), the benchmark loader
(`load_benchmark_file`), the leak check (`assert_benchmark_disjoint_from_seeds`,
`find_near_duplicate_benchmark_cases`), case validators, and the writer
(`write_report` → `___command_info/heldout_evaluation.json`). The benchmark schema is
documented in `docs/intent_benchmark_format.md`.

The extension is deliberately minimal:

- **`BenchmarkCase.kind` gains `"out_of_scope"`.** Such a case carries `context` and
  `utterance` and **no** `expected_label` — there is nothing to expect.
- **A new `score_out_of_scope(cases, predict_fn)`** returning
  `OutOfScopeScore{total, rejected, detection_rate, failures}`. A case passes only when the
  turn ends in the out-of-scope response, mirroring `score_escalation`'s strictness: a
  correct-by-accident ambiguity prompt is **not** a pass, because the runtime behaviour
  differs.
- **A new `false_rejection_rate`** computed over the *existing* routing cases: the fraction
  of valid held-out utterances the gate would reject. This is the number that governs
  (§4) and it is free — it reuses cases that already exist.
- **`validate_out_of_scope_cases()`**, analogous to `validate_escalation_cases()`: fail the
  run if an "out-of-scope" utterance is in fact served by a command in the context or any
  ancestor. Without it the benchmark decays into a test of the author's imagination.
- **Never blended.** Report `detection_rate` and `false_rejection_rate` as two numbers,
  following decision D2's precedent for escalation vs routing. They trade against each
  other and a single figure hides the trade — which is the whole subject of §4.

### 5.2 Where the negative cases come from — the actual hard part

There is no free source of out-of-scope cases. Three, each with a bias:

| Source | Cost | Bias |
|---|---|---|
| **(a) Hand-written by the workflow author** | High per workflow; does not scale | Authors write *easy* negatives (weather, jokes). The near-domain cases that matter are the ones they do not think of, because near-domain is exactly the boundary they have internalised. |
| **(b) Borrowed: routing cases from a *different* workflow's benchmark** | Nearly free, and self-maintaining | Realistic "wrong domain" but structurally *far* domain. Systematically overstates detection. Also needs a cross-check that the borrowed utterance really is unserved here — "check my order status" is in scope for two different workflows. |
| **(c) LLM-generated near-domain negatives** | Moderate, reuses `generate_synthetic.py` | Highest value and highest risk: the generator is asked for the complement of the command set and will sometimes emit in-scope utterances. Every such error is a *labelling* error in the direction §4 calls expensive. Requires human verification, which puts it back near (a)'s cost. |

**Recommendation: (a) + (b), with (c) only under human review.** Require the near-domain
proportion to be **declared in the benchmark file**, because a benchmark that is 90% "tell
me a joke" will report a detection rate that says nothing about deployed behaviour. The 50
cases in §1.2 are 30% near-domain by construction; the detection rates in §2-C should be
read as conditional on that mix and would drop on a near-domain-only set.

Adopt the existing disjointness discipline unchanged: an out-of-scope case that collides
with a seed utterance is a benchmark bug, and `assert_benchmark_disjoint_from_seeds`
already catches it under the same normalisation.

### 5.3 The acceptance bar, pre-registered

To arm the gate for a workflow, on that workflow's benchmark, with **≥200 routing cases and
≥100 out-of-scope cases of which ≥40% are near-domain**:

- `false_rejection_rate ≤ 2%`, with the 95% upper bound ≤ 5%; **and**
- `detection_rate ≥ 40%`, 95% lower bound above the Option-A baseline at the same
  false-rejection rate; **and**
- both measured on **two independent training runs**, reported as a paired comparison.

That last clause is not optional. The documented noise floor is **20.6% of routing verdicts
changing between two identical training runs** (spec §2), and `fix-551.9`/M4 established
that training is not reproducible at a fixed seed because the LLM regenerates the training
data. A single-run number for this feature is uninterpretable, and the recipe for the
paired comparison is in `fastworkflow-proof-and-analysis-toolkit`.

### 5.4 What §1's numbers are and are not

They are a **feasibility probe**, n=72 in-scope and n=50 out-of-scope, on one workflow, from
one training run, with the out-of-scope set written by the same person who read the
results. They are enough to answer "does confidence already separate?" (no — §1.5) and to
rank the options. They are **not** an acceptance measurement and must not be quoted as one.

---

## 6. Interaction with the existing reserved labels

### 6.1 `WILDCARD_LABEL` — orthogonal, but the ordering is load-bearing

`WILDCARD_LABEL` means *"an ancestor may serve this"*. Out-of-scope means *"nothing in the
chain serves this"*. They are different propositions and, critically, **decidable at
different points**.

`_commands/wildcard.py:99-124` already walks the parent chain when `command_name is None`
and falls through to `ErrorCorrection/you_misunderstood` only when every ancestor has
declined — with the comment `# out of scope commands` at line 111 already naming that
path. **That fall-through is where the out-of-scope gate belongs.** Two consequences:

- **The gate must run once, after the walk, not per context.** A gate that fires inside a
  child context would suppress a legitimate escalation to a parent that *can* serve the
  utterance — converting a recoverable escalation into a rejection, the exact failure mode
  §4 says is most expensive.
- **A gate placed there is automatically correct in root contexts**, where the walk is
  empty and the gate is the only thing between the utterance and `you_misunderstood`.
  That is precisely the case `fix-551.11` left uncovered.

The fixed mixed-top-k behaviour is worth flagging: a top-k containing `wildcard`
alongside local candidates takes the ambiguity branch and the escalation signal is dropped
(finding F7). An utterance in that state never reaches the parent walk, so it never reaches
the gate. An out-of-scope gate
therefore **cannot** improve the 60% of cases that produce ambiguity prompts unless it is
also consulted on the ambiguity branch — which is a second, separate decision, with its own
false-positive exposure, and should be deferred to a later slice.

### 6.2 `PARAMETER_VALUE_LABEL` and `fix-2m5` — the decisive argument for Option C

`fix-2m5` records that `PARAMETER_VALUE_LABEL`, backed by 7 contentless literals, confidently
beats `add_two_numbers` on 2 of 5 training seeds. The mechanism is generic: **every reserved
class competes with every real command for the same probability mass**, and small workflows
have the least mass to spare.

- **Option B makes this strictly worse.** A third reserved class, trained on a broad
  negative corpus, is a stronger competitor than a 7-row class. And the region it must
  occupy is measured, §1.6: out-of-scope prose is closer to the CME meta-command seeds than
  in-scope requests are (CME-bank AUC 0.348), so the class would be carved out from under
  `what_can_i_do` — the very command that produces the best available out-of-scope response
  today.
- **Option C is neutral by construction.** It adds no label, does not change the label
  space, does not change the softmax, and cannot alter the relative probability of
  `parameter_value` and any real command. Its artifacts are additive and its verdict is
  computed after the classifier has spoken. **This is the single strongest argument for C
  over B**, and it is the argument `fix-2m5` was filed to make.

There is one real interaction. Bare values score *low* on the seed-manifold gate — measured
retail max cosine 0.43–0.64 for the six probes, i.e. mostly below the recommended
`t = 0.6146`. A gate placed naively at the top of intent detection would reject bare values
as out-of-scope and break parameter answering. Placing it after the parent walk avoids this
entirely: a `parameter_value` prediction is non-routable, so the walk runs, but during the
`PARAMETER_EXTRACTION` stage the code path never reaches the gate. **The gate must be
stage-scoped to `INTENT_DETECTION` and must never see a parameter answer.** This is a
correctness requirement, not a tuning detail.

### 6.3 If a third label is added later

`nlu_labels.py` is already the right home and its structure anticipates this:
`NON_ROUTABLE_LABELS` would gain the new label; `ESCALATION_LABELS` would **not**, for the
same reason `PARAMETER_VALUE_LABEL` is excluded — "nothing serves this" carries no evidence
that an ancestor can help. `score_escalation(..., escalation_labels=...)` is already
parameterised for exactly this (`heldout_evaluation.py:517-534`, working tree).

---

## 7. Scoping estimate

### 7.1 Recommendation piece 2 (Option E, the soft action) — small

| File | Change |
|---|---|
| `fastworkflow/_workflows/command_metadata_extraction/_commands/ErrorCorrection/you_misunderstood.py` | Response wording: replace "Please select the correct command name from the list below" with a non-answer preamble. The list itself is already correct |
| `fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py` | Only if the wording needs to differ between "ambiguous among local candidates" and "nothing matched" — that distinction is available at line 109-124 but is not currently passed down |
| `tests/` | One integration test per `.cursor/rules/testing_rules.mdc` — real workflow, no mocks. **First job: confirm the §1.3 code trace against a live turn** |

Roughly one day. Main risk: `you_misunderstood` participates in the
`INTENT_MISUNDERSTANDING_CLARIFICATION` stage machinery and in `is_cme_command` handling;
changing what that path returns can perturb agent-mode transcripts and the CLI keep-alive
loop. Behaviour change must be flag-guarded.

### 7.2 Recommendation piece 3 (Option C, the gate) — medium

| File | Change | Risk |
|---|---|---|
| `fastworkflow/model_pipeline_training.py` | Write `seed_embeddings.npz` (+ a manifest of source utterances and command names) per context at train time, next to `label_encoder.pkl` | Low. Additive; `get_artifact_path` already handles placement |
| `fastworkflow/train/artifact_versioning.py` | Include the new artifact in versioned assembly | Low |
| `fastworkflow/train/selective_training.py` | Bank invalidation on seed-table change, via the corrected upward-closure rule (`close_dirty_contexts`, AR2) | **Medium.** A stale bank degrades the gate silently — the failure mode is invisible |
| `fastworkflow/cache_matching.py` | Extract/expose a `max_similarity_to_bank` helper; reuse `get_embedding`'s LRU | Low |
| `fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py` | Consult the gate at the end of the parent walk; stage-scoped to `INTENT_DETECTION` (§6.2) | **Medium.** Placement is the whole design; a misplacement suppresses escalation or breaks parameter answering |
| `fastworkflow/train/heldout_evaluation.py` | `kind="out_of_scope"`, `score_out_of_scope`, `false_rejection_rate`, `validate_out_of_scope_cases` | Low. Well-factored module |
| `docs/intent_benchmark_format.md` | Document the third case kind | Low |
| `fastworkflow/train/__main__.py`, `cli.py` | Calibration subcommand; report the two rates | Low |
| New env var, e.g. `INTENT_DETECTION_OUT_OF_SCOPE_POLICY` | Off / soft / (never hard). Follow `TOPK_WILDCARD_POLICY`'s pattern — `get_env_var` with no code default so shell overrides work, `lru_cache`d, unknown values warn and fall back | Low |
| Threshold artifact `out_of_scope_threshold.json` | Written by calibration, **absent by default** so the gate is inert on every existing workflow | Low |
| `tests/` | Integration tests against `tests/example_workflow` / `tests/todo_list_workflow` | Low |

Roughly one to two weeks of engineering, plus benchmark authoring (§5.2), which is the
larger and less predictable cost and is *per workflow*, not one-off.

### 7.3 What could go wrong

- **The artifact goes stale and nobody notices.** The gate has no self-check: a bank built
  from an old seed table still returns plausible cosines. Mitigation: fingerprint the bank
  with the same provenance hash as the utterance cache (`fix-551.9`) and refuse to arm on
  mismatch — fail loud, not quiet.
- **Threshold calibrated on an easy benchmark.** §5.2's bias table. A benchmark of jokes
  and weather yields a threshold that rejects nothing real. Mitigation: require the
  near-domain proportion to be declared and enforce a floor.
- **Gate placed too early.** Suppresses escalation (§6.1) or eats bare values (§6.2). Both
  are silent regressions in features that are currently unmeasured — escalation recall is
  finding F10, still unmeasured on any in-repo workflow.
- **The two-tier pipeline is not accounted for.** `CommandRouter.predict` uses TinyBERT or
  DistilBERT depending on `threshold.json`, and the gate's embedding comes from the
  DistilBERT body regardless. Any confidence-based *combination* with the gate has to know
  which model produced the confidence; a distance-only gate sidesteps this and should stay
  distance-only for that reason.
- **Collision with E-PV1.** The retail benchmark and probe sets belonged to the
  `parameter_value` trade-off experiment. Both efforts wanted to modify
  `retail_workflow`'s label space and benchmark, so they had to be sequenced rather than
  run against the same artifacts. Temporary inputs were removed after reconciliation.
- **The evidence base evaporates on a second run.** 20.6% verdict churn (spec §2). Every
  number here could move. §5.3's two-run requirement exists for this reason.

---

## 8. Explicitly not established

- **That Option C generalises.** Measured on one workflow, one training run. AUC 0.911 on
  retail and 0.604 on hello_world is a two-point sample whose spread is larger than its
  mean is useful.
- **That 34–52% detection is enough to be worth the machinery.** Nobody has stated a target.
  §5.3 proposes 40% as a bar; that number is a placeholder pending an owner decision.
- **That agent mode is harmed by out-of-scope misroutes.** Asserted in §1.4 as a plausible
  cost. Unmeasured. It is the main reason the blast-radius argument might understate
  severity.
- **That the out-of-scope set in §1.2 is representative.** It was written by one person
  who then read the results. It is not adversarially sourced, not user-derived, and its
  category mix was chosen, not sampled.
- **Any number here at better than ~20 points of resolution.** The documented run-to-run
  churn dominates.
- **§1.3's account of what the ambiguous 60% actually see.** Established by reading
  `intent_detection.py` and `_commands/wildcard.py:99-124` plus retail's empty
  `context_inheritance_model.json`, not by capturing a live turn. It is the one claim in
  this document that a single integration test would settle, and §7.1 makes that the first
  task.
- **That `fix-d28` should be p2.** That is a recommendation (§3.1) contingent on §3.3's
  first row — the existence of zero-parameter mutating commands in real user workflows,
  which has not been surveyed.

---

## 9. Provenance and re-verification

All measurements were produced read-only against committed artifacts. No production code
was modified. No training was run. The throwaway probe scripts live **outside the repo**
in `/tmp/fix-d28-probe/`:

| Script | What it produces |
|---|---|
| `probe.py` | Per-utterance routing decision + softmax statistics for one workflow |
| `analyse.py` | The per-utterance tables and the threshold sweep |
| `auc.py` | Separability AUCs for max-prob / margin / entropy |
| `embed_probe.py`, `embed_probe2.py` | Seed-bank cosine scores; the all/app/CME bank split |
| `stats.py` | Wilson intervals and leave-one-out threshold stability |
| `final_probe.py` | **The headline run**: 72-case benchmark vs 50 out-of-scope, all gates |

Re-verification one-liners:

```bash
# Thresholds are the mean confidence of misclassified test rows, not an operating point
git show 23dbe35:fastworkflow/model_pipeline_training.py | sed -n '1174,1190p'
cat fastworkflow/examples/retail_workflow/___command_info/global/*threshold*.json

# The only rejection mechanism today
git show 23dbe35:fastworkflow/model_pipeline_training.py | sed -n '321,334p'

# CME commands execute with no parameter stage; app commands do not
sed -n '65,90p'  fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py
sed -n '140,165p' fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py

# The fall-through already labelled "out of scope commands" — the gate's home
sed -n '99,124p' fastworkflow/_workflows/command_metadata_extraction/_commands/wildcard.py

# NOT_FOUND is treated as missing, which is what stops the mutating commands
sed -n '337,341p' fastworkflow/utils/signatures.py

# The embedding is already computed on every intent-detection turn
git show 23dbe35:fastworkflow/_workflows/command_metadata_extraction/intent_detection.py | sed -n '116,126p'
grep -n "def get_embedding" -A 4 fastworkflow/cache_matching.py

# Evaluation machinery to extend
grep -n "def score_routing\|def score_escalation\|class BenchmarkCase\|def assert_benchmark_disjoint" \
  fastworkflow/train/heldout_evaluation.py

# The in-scope benchmark summary and measured outcomes are preserved in §2 and Appendix A.
```

---

## 10. Amendment A1 — correct the runtime contract before implementation

**Status:** targeted current-tree audit and live-turn correction, not a completed house
adversarial review. This amendment supersedes the runtime claims in §§1.3, 3.1 piece 3,
6.1, 7.2, and 8 where they conflict. The feasibility measurements remain historical
evidence against the pre-R7.1 artifacts; their acceptance limitations in §5.4 still apply.

### A1.1 The ambiguous first turn does not reach the fallback

The original §1.3 code trace skipped an earlier return. On a low-confidence top-k result:

1. `CommandNamePrediction.predict` returns `Output(error_msg=...)`.
2. `_commands/wildcard.py:42-57` sets
   `NLU_Pipeline_Stage=INTENT_AMBIGUITY_CLARIFICATION`.
3. The method returns the ambiguity error immediately.
4. The parent walk at `wildcard.py:99-107` and the `you_misunderstood` dispatch at
   lines 109-124 are not reached on that turn.

This was verified live on 2026-08-02 against an isolated temporary copy of the retail
workflow with its committed `global` model artifacts linked read-only. For
`"what is the weather in Paris"` the observed stage was
`INTENT_AMBIGUITY_CLARIFICATION`, `success=False`, and the first response line was
`"Ambiguous intent error for command 'what is the weather in Paris'"`. The neutral
`you_misunderstood` wording was absent.

Consequences:

- The wording fix improves genuine walk-exhaustion and misunderstanding recovery, but it
  does **not** change the first-turn response for the ambiguous 60% measured in §1.
- Consulting a gate on ambiguous top-k output remains a separate later decision. A1 does
  not add it.
- The direct `you_misunderstood` test proves the response text, not the original §1.3
  end-to-end trace. The trace is now disproved rather than merely unverified.

### A1.2 An end-of-walk-only gate misses the targeted failure

The measured 40% failure is a **single-label classifier commitment**. Its current path is:

1. `CommandRouter.predict(command)` returns one label.
2. `intent_detection.py` assigns it to `command_name`.
3. A CME command executes immediately at `wildcard.py:65-89`, or an application command
   proceeds to parameter extraction at lines 135-164.
4. Because `command_name` is not `None`, the parent-walk block is skipped.

Therefore a gate added only at `wildcard.py:109-124` cannot observe or reject any of those
confident commitments. It would instrument walk exhaustion, not fix the 40% mechanism.

### A1.3 Corrected minimal runtime contract

The first implementation slice, if the evidence gate below is eventually cleared, is a
**classifier-commit validator**, not a general front-door filter:

1. Consult it only in `CommandNamePrediction.predict`, inside
   `NLU_PipelineStage.INTENT_DETECTION`, after `CommandRouter.predict` returns exactly one
   label and before assigning that label to `command_name`.
2. Consult it only for a routable classifier label. `wildcard` and `parameter_value`
   remain non-routable and continue into the existing parent walk unchanged.
3. If the seed-manifold score is below the calibrated threshold, leave `command_name`
   unset. The existing parent walk then tries the next ancestor. If every context
   declines, the existing soft `you_misunderstood` response is used.
4. Validate each ancestor's classifier commitment against that ancestor's own seed bank.
   Sequential "reject locally, then try parent" composes to the whole context chain
   without building or scoring a separate union bank.
5. Do **not** gate exact command-name matches, fuzzy command-name matches, or the learned
   clarification cache in this slice. They are explicit or user-corrected routes, and
   none is part of the measured failure mechanism. This keeps the change attributable.
6. Do **not** gate low-confidence top-k ambiguity in this slice.
7. Do **not** gate `PARAMETER_EXTRACTION`; that branch does not call the classifier and
   bare values are known to score low against the seed manifold.

This placement catches the targeted confident classifier misroutes before either CME
execution or parameter extraction, preserves ancestor escalation, and requires no
reordering of the wildcard state machine.

### A1.4 Artifact contract remains additive and default-off

The measured candidate still requires a per-context bank generated from the hand-written
`plain_utterances` for that context's application and CME labels, embedded with the same
fine-tuned DistilBERT body as the context's `largemodel.pth`.

- Store the bank and its source manifest beside the context's model artifacts so artifact
  version publication and selective-training carry-forward keep model, bank, and
  fingerprint atomic.
- A threshold artifact is the opt-in. If the bank, threshold, or matching fingerprint is
  absent, the gate is inert and existing workflows behave exactly as before.
- Keep the bank optional in the first additive artifact format so legacy and
  carried-forward pre-gate contexts remain loadable.
- Reuse `cache_matching.get_embedding` for the query vector; do not add another model
  loader or embedding implementation.

### A1.5 Evidence gate before runtime code

The current evidence cannot authorize arming the detector:

- the routing benchmark has 72 cases, below §5.3's 200;
- the out-of-scope probe has 50 cases, below §5.3's 100 and exists only in this document;
- the committed retail model predates the `wildcard` / `parameter_value` split;
- no second realistic workflow demonstrates that the 0.911 AUC generalises;
- no paired post-split replicate exists, while the recorded routing-verdict churn is
  about 20%.

The next code slice is therefore measurement plumbing only: add an `out_of_scope`
benchmark case kind and separately report detection and false rejection through the
existing held-out evaluator. Reproduce the read-only feasibility result on the fixed
72/50 cases, then remeasure post-split on temporary workflow copies. Runtime arming
remains blocked until the pre-registered §5.3 bar is met or the owner explicitly revises
that bar.

### A1.6 Regression obligations for the eventual gate

Before any workflow enables the gate, verification must show:

- gate-off output is byte-for-byte behaviorally identical on legacy artifacts;
- confident classifier out-of-scope commitments are converted to the soft fallback;
- valid routing false-rejection rate and confidence interval meet §5.3;
- lone `wildcard` still reaches and succeeds in an ancestor context;
- `parameter_value` replies never enter the gate;
- exact, fuzzy, and clarification-cache routes bypass this first slice;
- low-confidence top-k behavior is unchanged;
- missing, stale, or mismatched gate artifacts fail open with a loud diagnostic.

---

## 11. Amendment A2 — no-go until runtime shadow evidence exists

**Status:** independent adversarial review of the proposal and A1 against the current
working tree. This amendment supersedes A1.5's next-step recommendation and §§5.3, 9, and
the status header where they conflict. **Option C remains a no-go.**

### A2.1 The measured artifacts are local legacy state, not committed evidence

The repeated description of the retail and hello-world model artifacts as "committed" is
false. Every `___command_info` model directory is gitignored; the retail directory is an
unversioned legacy layout with no manifest or `current.json`; and the pinned commit cannot
reconstruct its exact weights, generated utterances, or training inputs.

The 72/50 probe is freshly repeatable on this machine and useful for mechanism discovery.
It is not reproducible from commit `23dbe35`, so it cannot support a release or public
claim. Read "committed artifacts" throughout the historical sections as **local legacy
artifacts present on the measurement machine**.

### A2.2 Pre-split AUC must be discarded for arming decisions

The statement that the R7.1 label split cannot affect AUC because AUC concerns confidence
rather than label names is also false. The candidate embedding is the `[CLS]` vector from
the **fine-tuned** DistilBERT body. Retraining under a changed label space updates that body
as well as the classifier head, so both softmax confidence and embedding geometry can move.

The measured AUC 0.911 ranks candidates on the local pre-split model only. A gate threshold
must be calibrated and evaluated again on fresh, immutable, versioned post-split artifacts.

### A2.3 Define the candidate honestly

The probe's "seed manifold" is narrower than the model's training distribution. It is the
117 hand-written literal `plain_utterances` found in the retail global commands and CME
IntentDetection commands. It excludes command-name tokens, generated persona utterances,
template utterances, and post-holdout training rows.

Until a different bank wins a locked comparison, call this candidate the
**plain-utterance embedding bank**, not the training manifold. Production collection must
use hydrated command metadata for every command in the tested context; regex-scraping
top-level Python files is probe-only and does not generalise to nested contexts.

### A2.4 Calibration and evaluation must be disjoint

The feasibility sweep selected `t=0.5715` and `t=0.6146` on the same 72 positive and 50
negative cases used to report performance. Leave-one-out removed one positive case while
reusing every negative. Those rates are therefore in-sample operating points, not
generalisation estimates.

Before arming:

1. Split human-reviewed routing and out-of-scope cases into disjoint calibration and locked
   evaluation corpora before fitting any threshold.
2. Run at least **k=5 paired artifact builds**, not the two runs proposed in §5.3.
3. Report routing top-1/in-list, escalation recall, parameter-value handling, exact/fuzzy/
   correction-cache routes, CME routes, agent-mode outcomes, and latency.
4. Stratify detection by application vs CME intents and near-domain vs far-domain
   negatives. A single micro-AUC can hide the measured inverted CME-only signal.
5. Keep semantic out-of-scope labels under independent human review.
   `validate_out_of_scope_cases()` checks only structural metadata and deliberately cannot
   infer whether a command serves a natural-language request.

### A2.5 Classifier-only scores cannot prove the user-visible result

The held-out evaluator's new `score_out_of_scope` and `score_false_rejection` functions are
valid metric containers, but their boolean `reject_fn` must eventually be supplied by a
full runtime shadow path. `CommandRouter.predict` alone does not execute exact/fuzzy/cache
routing, parent traversal, NLU stage transitions, CME actions, or the final response.

The next implementation may observe and report a gate decision, but it must not alter
`command_name`, the parent walk, or execution. Shadow records must include route origin,
contexts visited, per-context similarity, proposed decision, actual command, final NLU
stage, and response type. Only that path can show both targeted interception and
non-regression.

### A2.6 The soft recovery affordance must describe real behavior

In `INTENT_MISUNDERSTANDING_CLARIFICATION`, an unmatched message defaults to
`what_can_i_do`; it does not re-enter intent classification. The earlier wording
`"rephrase your request or type 'abort'"` therefore advertised a direct recovery path the
state machine does not provide.

The minimal correction is truthful sequencing: **type `abort` to reset command processing,
then rephrase on the next turn**. The response and integration test now state and verify
that sequence. This does not solve false rejection of an ancestor command: the capability
list is built from CME plus the current context, not the ancestor-union. That is another
reason the detector remains shadow-only.

### A2.7 Runtime and artifact costs are not negligible by default

When the adaptive correction cache is empty, `cache_match` returns before computing an
embedding. A TinyBERT-only route also need not run DistilBERT. The gate can therefore add a
DistilBERT forward pass to turns that previously avoided one; latency must be measured.

Likewise, calibration artifacts must never be edited in place inside a published version:
selective carry-forward may hardlink files across versions. Bank, manifest, threshold, and
model must be assembled in a fresh version and published atomically, or remain external
shadow evidence.

### A2.8 Smallest safe next slice

1. Keep the neutral fallback wording, with the truthful abort-then-rephrase sequence.
2. Finish and retain the measurement schema and 50-case feasibility corpus, explicitly
   below acceptance size and mix.
3. Train fresh post-R7 artifacts into immutable versions without touching bundled example
   artifacts in place.
4. Add runtime-level **shadow-only** plain-utterance-bank scoring with locked logs and no
   routing effect.
5. Calibrate and evaluate on disjoint human-reviewed corpora over k≥5 paired builds.
6. Return for a gate/no-gate decision only after those results clear the false-rejection,
   detection, regression, and latency bars.

---

## 12. Amendment A3 — proof-carrying selective execution

**Status:** candidate direction after A2's no-go, not a ratified design. This amendment does
not authorize implementation or arming. `fix-d28` stays open until a solution demonstrates
useful out-of-scope handling, controlled false rejection, runtime recovery, and no
regression on fresh reproducible artifacts.

### A3.1 Reframe the question

The original question was:

> Can this workflow recognise the complement of everything it knows how to do?

That complement is open-ended, changes with every command and ancestor, and is impossible
to enumerate. A closed-set classifier is structurally forced to choose one of its labels,
so asking it for an absolute "none of the above" judgement after training it only on
positive intent classes is the wrong abstraction.

The narrower question is:

> Does the proposed command carry enough command-specific evidence to justify this
> execution under the command's risk policy?

This turns an unbounded domain-recognition problem into selective prediction over one
concrete candidate. A route becomes a proposal with evidence, not an authority to execute.

### A3.2 Fresh evidence: changing the cosine target does not rescue Option C

A read-only follow-up used the same local legacy retail model, 72 routing cases, 50
out-of-scope cases, and 117 literal `plain_utterances`. It compared the global bank with
command-specific alternatives. These remain in-sample feasibility numbers with every A2
limitation.

For the 63 confident in-scope commitments and 20 confident out-of-scope commitments:

- global maximum cosine: AUC 0.902; at zero observed false rejection, catches 4/20 OOS;
- cosine to the **predicted command's** seeds: AUC 0.885; catches 4/20 at zero false
  rejection, or 6/20 at 1/63 false rejection;
- classifier / nearest-seed-command agreement: catches 7/20, but falsely rejects 7/63
  valid commitments;
- predicted-command similarity margin: AUC 0.793 and catches 0/20 at zero or one false
  rejection.

Several semantically absurd routes have strong command-specific evidence under the current
fine-tuned embedding: "please summarise the French revolution" is nearest to
`reset_context`; "renew my car insurance policy" is nearest to
`modify_pending_order_payment`; "what is your child labour policy" is strongly nearest to
`what_can_i_do`.

Conclusion: **retire the current fine-tuned-CLS/plain-utterance gate as the presumptive
solution.** A different threshold target does not repair its geometry. Keep it only as a
documented baseline for future shadow comparisons.

### A3.3 Proposed architecture: a proof-carrying selective router

The existing classifier remains a cheap candidate generator. It must also report route
provenance: exact command token, fuzzy command-name match, positive correction-cache hit,
or classifier prediction.

Only implicit classifier commitments need new proof in the first design:

1. The classifier proposes a fully-qualified command.
2. A separate command verifier scores the pair
   `(user utterance, proposed command capability card)`.
3. The verifier returns support evidence and a calibrated abstention value.
4. The execution policy combines route provenance, support, and command risk.
5. Unsupported local candidates do not execute; a parent context may propose and verify
   its own candidate. If every context abstains, use the neutral fallback.

Exact command invocation and confirmed correction-cache entries are higher-authority
signals and may bypass the semantic verifier. Fuzzy matching needs its own review because
it is inferred rather than explicitly confirmed.

This composes through context ancestry without defining a per-context negative class:
every context independently proves the candidate it proposes. "No command in the chain
was able to prove support" is the resulting soft out-of-scope condition.

### A3.4 Capability cards and the verifier

A **capability card** is a versioned, reviewable contract for one command:

- fully-qualified name and concise purpose;
- hand-written positive utterances;
- Pydantic parameter names, types, descriptions, and required entities;
- declared effect: read, navigation, write, external side effect, or unknown;
- human-reviewed exclusions and near-miss commands;
- source fingerprint and schema version.

The leading verifier candidate is a global pairwise semantic matcher trained across
workflows:

- positive pair: an utterance with its true command card;
- hard negative pair: the same utterance with the nearest wrong command card;
- cross-workflow pairs: realistic unsupported requests without defining one monolithic
  out-of-scope class;
- human-reviewed near-domain negatives for the rare capability boundary.

At runtime it scores only the proposed command (or a very small candidate set), rather
than cross-encoding every command. Its encoder is frozen and versioned independently from
the per-context intent classifiers, so an intent-label retrain cannot silently move the
verifier's geometry.

This is not yet proven. A global verifier may fail to transfer between workflows, command
cards may omit load-bearing distinctions, and cross-workflow negatives can be mislabeled
when two workflows share a capability.

### A3.5 Conformal abstention, with the guarantee stated narrowly

Held-out valid pairs can calibrate a command-specific or pooled nonconformity score. The
runtime verifier then emits a p-value or prediction set:

- supported singleton: candidate may proceed under its risk policy;
- weak or empty support: abstain softly;
- multiple supported candidates: clarify.

Under exchangeability, split-conformal calibration can bound **marginal false abstention on
valid requests** at the chosen level. It does not guarantee out-of-scope detection,
conditional per-command coverage on rare classes, or robustness under deployment drift.
Those remain measured outcomes. Commands with too few independent calibration examples
must pool conservatively or remain unarmed.

### A3.6 Deterministic risk firewall: solve consequence before recognition

The measured retail failures show why safety should not wait for perfect OOS detection:
wrong routes have radically different consequences.

Execution policy should be explicit:

- explicit command or confirmed correction: execute normally;
- implicit classifier route to a read-only command: verifier policy may permit it;
- implicit classifier route to navigation, write, external effect, or unknown effect:
  require confirmation unless command-specific evidence clears a stricter reviewed bar;
- unannotated zero-parameter commands default to confirmation, not execution.

The built-in control plane deserves separate treatment. Commands such as `reset_context`
and `go_up` should not execute from an unsupported semantic guess. Harmless
`what_can_i_do` may remain permissive. This deterministic firewall can prevent accidental
state/navigation changes even if semantic OOS detection never becomes satisfactory.

This is an immediate safety design, not a claim that the UX problem is solved: an
out-of-scope utterance may still produce an irrelevant read or confirmation.

### A3.7 Separate control plane from application intent

The four CME meta-commands are domain-free conversational English and contaminate scope
signals; the measured CME-only embedding AUC is inverted at 0.348. A fixed set of generic
control commands should not compete indiscriminately with application intents.

Candidate decomposition:

1. explicit/exact handling for navigation and reset commands;
2. a narrow control-plane matcher for genuine meta requests;
3. application routing and command verification over application capabilities;
4. the neutral capability response as a fallback action, not a classifier class.

This could remove the modal accidental landing on `what_can_i_do` while retaining it as the
deliberate response. It also risks regressing natural paraphrases of legitimate meta
commands, so it must be shadowed and evaluated as its own ablation.

### A3.8 Negative feedback as a boundary-learning flywheel

The current correction cache records positive mappings after clarification. It should also
be possible to record:

> This utterance does **not** mean candidate X.

A negative correction is a deterministic veto for repeated or highly similar mistakes and
a training pair for the global verifier. An abstention response can ask a concrete,
recoverable question: "I can perform X; did you mean that?" A "no" answer records a
negative pair; a confirmed alternative records the existing positive mapping.

Requirements before this is viable:

- negative records are scoped by workflow, command fingerprint, and model/verifier version;
- contradictions between positive and negative feedback are surfaced, never last-write-wins;
- privacy and retention rules are explicit;
- feedback cannot silently authorize a write;
- stale command cards invalidate dependent feedback.

### A3.9 Alternative model ablation: outlier exposure without an OOS class

If the current per-context classifier is retained, test **outlier exposure** rather than a
third reserved label. During training:

- normal in-scope rows receive cross-entropy on their intent;
- vetted outlier rows receive a uniform-softmax or energy-margin objective;
- ancestor intents remain `wildcard` positives in child contexts, not outliers;
- outliers exclude every capability available in the local-to-root chain.

This teaches low support without allocating probability mass to a competing
`out_of_scope` class. Cross-workflow utterances are a cheap source of far-domain outliers;
human-reviewed near-domain cases remain necessary. This is an experiment arm, not the
default recommendation, because uniform uncertainty can damage calibration and CME
commands still create overlap.

### A3.10 Shadow experiment ladder

No candidate changes routing until this ladder produces satisfactory evidence:

1. **Freeze artifacts.** Train fresh post-R7 models into immutable versions; preserve exact
   provenance and never touch bundled example artifacts in place.
2. **Instrument route provenance.** Record exact/fuzzy/cache/classifier origin, contexts
   visited, proposed command, actual command, stage, response type, duration, and risk.
3. **Shadow the deterministic firewall.** Report which current executions would require
   confirmation, with special attention to valid control-plane commands.
4. **Shadow verifier arms on identical turns:**
   - current global plain-utterance cosine baseline;
   - predicted-command cosine baseline;
   - frozen pairwise capability-card verifier;
   - conformal abstention over verifier scores;
   - outlier-exposed classifier.
5. **Use disjoint data.** Separate training, calibration, and locked evaluation sets.
   Require independent human review of out-of-scope labels and at least 40% near-domain
   negatives in the OOS evaluation set.
6. **Run k≥5 paired builds.** Report routing top-1/in-list, OOS detection, false abstention,
   escalation, control-plane accuracy, parameter-value handling, confirmation burden,
   agent-mode task effect, and p50/p95 latency, stratified by route origin, command risk,
   application/CME, and near/far negatives.
7. **Evaluate recovery.** Measure whether false abstentions recover through confirmation or
   abort-then-rephrase, not merely whether a soft response string was emitted.

The risk firewall and semantic verifier are separate ablations. Combining them before each
is measured would recreate the bundled-mitigation failure recorded in the research
methodology.

### A3.11 Decision and kill criteria

The issue remains open. There is no satisfactory solution yet.

Do not implement an enforcing gate unless a candidate:

- meets the pre-registered false-abstention bound with its confidence interval;
- materially detects locked near-domain OOS cases beyond the current ambiguity baseline;
- does not regress ancestor escalation, parameter replies, control-plane commands, or
  agent-mode task success;
- provides a tested recovery path after false abstention;
- stays inside an accepted latency and confirmation-burden budget;
- reproduces over k≥5 fresh artifact builds and a second realistic workflow.

Retire the pairwise-verifier direction if it cannot beat the global-cosine baseline on
locked near-domain detection at the same false-abstention rate. Retire outlier exposure if
it improves OOS metrics by reducing valid routing or escalation beyond the pre-registered
non-regression bound. If no semantic candidate clears the bar, keep the deterministic risk
firewall and neutral fallback as the honest partial solution rather than shipping a weak
detector.

---

## Appendix A — the 50 out-of-scope utterances, with measured outcomes

`routes? = yes` means `CommandRouter.predict` returned a single label, which
`intent_detection.py:124` executes with no prompt. `seed cos` is the max cosine to the
117-utterance seed bank; the recommended Option-C threshold is 0.6146.

| # | category | utterance | routes? | predicted label | max prob | seed cos |
|---|---|---|---|---|---|---|
| 1 | general_knowledge | what is the capital of Australia | no (prompt) | `IntentDetection/what_is_current_context` | 0.2562 | 0.5546 |
| 2 | general_knowledge | who won the world cup in 2018 | no (prompt) | `IntentDetection/go_up` | 0.1788 | 0.4765 |
| 3 | general_knowledge | please summarise the french revolution | **yes** | `IntentDetection/reset_context` | 0.4517 | 0.6957 |
| 4 | general_knowledge | how many moons does jupiter have | **yes** | `list_all_product_types` | 0.2481 | 0.6338 |
| 5 | general_knowledge | explain photosynthesis to a ten year old | no (prompt) | `IntentDetection/reset_context` | 0.2390 | 0.6497 |
| 6 | general_knowledge | when did the berlin wall come down | no (prompt) | `IntentDetection/go_up` | 0.2580 | 0.6023 |
| 7 | general_knowledge | what language do they speak in brazil | no (prompt) | `IntentDetection/what_is_current_context` | 0.3314 | 0.5825 |
| 8 | general_knowledge | who wrote pride and prejudice | **yes** | `IntentDetection/reset_context` | 0.2457 | 0.6661 |
| 9 | weather_time | what is the weather in Paris | no (prompt) | `IntentDetection/what_is_current_context` | 0.1262 | 0.4515 |
| 10 | weather_time | what time is it in Tokyo right now | no (prompt) | `IntentDetection/what_is_current_context` | 0.2108 | 0.5721 |
| 11 | weather_time | will it rain tomorrow in Seattle | no (prompt) | `exchange_delivered_order_items` | 0.0979 | 0.4557 |
| 12 | weather_time | how many days until christmas | no (prompt) | `exchange_delivered_order_items` | 0.1094 | 0.5732 |
| 13 | other_domain_task | book me a flight to Tokyo | **yes** | `IntentDetection/what_can_i_do` | 0.2443 | 0.5344 |
| 14 | other_domain_task | reserve a table for two at seven tonight | **yes** | `calculate` | 0.3906 | 0.6878 |
| 15 | other_domain_task | schedule a meeting with my manager on friday | no (prompt) | `get_order_details` | 0.2287 | 0.5578 |
| 16 | other_domain_task | transfer five hundred dollars from savings to checking | no (prompt) | `modify_pending_order_items` | 0.2651 | 0.5351 |
| 17 | other_domain_task | play some jazz music | no (prompt) | `IntentDetection/go_up` | 0.2004 | 0.5675 |
| 18 | other_domain_task | send an email to my accountant about the invoice | **yes** | `get_user_details` | 0.5036 | 0.7485 |
| 19 | other_domain_task | translate this paragraph into spanish | **yes** | `IntentDetection/what_is_current_context` | 0.3484 | 0.6678 |
| 20 | other_domain_task | renew my car insurance policy | **yes** | `modify_pending_order_payment` | 0.4075 | 0.7227 |
| 21 | other_domain_task | find me a plumber near my house | no (prompt) | `find_user_id_by_name_zip` | 0.1805 | 0.5708 |
| 22 | coding | write me a python script that reverses a string | **yes** | `IntentDetection/go_up` | 0.4178 | 0.6859 |
| 23 | coding | why is my docker container exiting immediately | no (prompt) | `modify_pending_order_address` | 0.2485 | 0.6513 |
| 24 | coding | what does a segmentation fault mean | no (prompt) | `IntentDetection/what_can_i_do` | 0.3277 | 0.7341 |
| 25 | chitchat | tell me a joke about penguins | no (prompt) | `IntentDetection/what_is_current_context` | 0.1752 | 0.6720 |
| 26 | chitchat | how are you doing today | **yes** | `IntentDetection/what_can_i_do` | 0.4017 | 0.6879 |
| 27 | chitchat | what is the meaning of life | **yes** | `IntentDetection/what_is_current_context` | 0.5500 | 0.7422 |
| 28 | chitchat | good morning | no (prompt) | `transfer_to_human_agents` | 0.1214 | 0.4592 |
| 29 | chitchat | are you a robot | no (prompt) | `IntentDetection/what_is_current_context` | 0.2073 | 0.7094 |
| 30 | chitchat | thanks, that was helpful | no (prompt) | `transfer_to_human_agents` | 0.2099 | 0.5516 |
| 31 | chitchat | i am so bored right now | **yes** | `IntentDetection/what_is_current_context` | 0.4813 | 0.6979 |
| 32 | advice | i have a headache what should i take | no (prompt) | `IntentDetection/what_can_i_do` | 0.3345 | 0.6514 |
| 33 | advice | can i sue my landlord for not fixing the heating | **yes** | `modify_pending_order_items` | 0.4336 | 0.6007 |
| 34 | advice | should i invest in index funds or property | no (prompt) | `IntentDetection/what_can_i_do` | 0.3004 | 0.6087 |
| 35 | advice | how do i get my toddler to sleep through the night | **yes** | `transfer_to_human_agents` | 0.3831 | 0.6809 |
| 36 | near_domain | who is the chief executive of this company | no (prompt) | `list_all_product_types` | 0.1634 | 0.4630 |
| 37 | near_domain | what are your store opening hours | no (prompt) | `list_all_product_types` | 0.2738 | 0.6417 |
| 38 | near_domain | do you sponsor work visas for new hires | **yes** | `IntentDetection/what_can_i_do` | 0.2766 | 0.5712 |
| 39 | near_domain | are you hiring warehouse staff at the moment | no (prompt) | `IntentDetection/what_can_i_do` | 0.2774 | 0.6122 |
| 40 | near_domain | what is your carbon footprint policy | **yes** | `IntentDetection/what_can_i_do` | 0.7152 | 0.7741 |
| 41 | near_domain | which courier do you use for deliveries | no (prompt) | `modify_pending_order_address` | 0.1788 | 0.6002 |
| 42 | near_domain | do you price match with your competitors | no (prompt) | `list_all_product_types` | 0.2310 | 0.5868 |
| 43 | near_domain | can i get a job application form | **yes** | `IntentDetection/what_can_i_do` | 0.2722 | 0.5169 |
| 44 | near_domain | where are your warehouses located | no (prompt) | `get_product_details` | 0.1994 | 0.5644 |
| 45 | near_domain | what is your annual revenue | no (prompt) | `list_all_product_types` | 0.2083 | 0.5412 |
| 46 | near_domain | do you offer student discounts | **yes** | `list_all_product_types` | 0.3940 | 0.6442 |
| 47 | near_domain | is my data being sold to advertisers | **yes** | `get_user_details` | 0.4065 | 0.6496 |
| 48 | near_domain | why was my credit application declined | no (prompt) | `modify_pending_order_payment` | 0.3084 | 0.6375 |
| 49 | near_domain | can i speak to your legal department about a trademark | no (prompt) | `modify_pending_order_items` | 0.2142 | 0.4949 |
| 50 | near_domain | what is your policy on child labour in the supply chain | **yes** | `IntentDetection/what_can_i_do` | 0.7071 | 0.8090 |
