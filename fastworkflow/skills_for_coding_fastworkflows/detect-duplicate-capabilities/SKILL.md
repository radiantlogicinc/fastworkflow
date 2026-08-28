---
name: detect-duplicate-capabilities
description: >-
  Read and act on fastWorkflow's near-duplicate capability diagnostics — the pre-flight lexical
  separability scan and the post-training router-confusion scan — to decide whether two commands
  should be merged, re-seeded, or recorded as a deliberate near-duplicate. Covers what
  separability actually measures, the reporting bands and their thresholds, the
  `accepted_duplicates.json` accept-list, and the blind spots of each instrument. Use when
  `fastworkflow train` reports DUPLICATE CAPABILITIES, OVERLAPPING or MODEL CONFUSION, when a
  workflow keeps routing one command's phrasings to another, or when reviewing a command set for
  redundancy.
---

# Detecting duplicate capabilities

Some workflows expose the same capability twice: `ControlsMonitor/list_findings` and
`Directory/search_control_findings` answer the same question. Others hold legitimate neighbours
or opposites whose seed lists do not yet express the distinction. **Both shapes present
identically** — as benchmark failures — and the fix for one is the opposite of the fix for the
other. These two scans exist to tell them apart.

Neither scan changes what is trained, which labels exist, or what any model predicts. Neither
blocks a run. They report.

## The two instruments

| | Pre-flight lexical scan | Post-training router scan |
|---|---|---|
| Runs | before anything costs money | after models are trained and carried forward |
| Input | hand-written **seed** utterances | seed utterances + the trained `CommandRouter.predict` |
| Needs | no LLM key, no network | the models this run just produced |
| Asks | "does the training data separate these?" | "does the model actually separate these?" |
| Blind to | a duplicate pair sharing no distinctive vocabulary | nothing lexical, but only sees pairs it was run on |
| Output | `___command_info/duplicate_capabilities.json` + printed report | printed `MODEL CONFUSION` warning, merged into the same report |

They are complements, not a cheap version and a good version. Run and read both.

## What separability actually measures

"Near-duplicate" is defined as a property of the **training data**, not of the two commands'
meanings, because the training data is the only thing the classifier ever sees:

> Two commands are near-duplicates when a classifier restricted to that pair, trained on their
> own utterances, cannot tell them apart.

Made concrete as leave-one-out, balanced, nearest-centroid accuracy over the pair's utterances in
TF-IDF space. Three choices in that sentence carry the whole result:

- **Leave-one-out**, because an utterance sitting inside its own centroid classifies itself.
- **Balanced** (the mean of the two per-command recalls), so chance is 0.5 regardless of how many
  utterances each side has.
- **Document frequency computed across every command in the workflow**, so shared vocabulary —
  "order", "my", "please", the workflow's own subject matter — is discounted automatically and
  only distinguishing terms carry weight. This is what stops a workflow full of
  `modify_pending_order_address` / `_items` / `_payment` from producing a wall of false
  positives: those share their boilerplate and differ in exactly the terms IDF promotes.

Ties go to the *other* command. A tie means the classifier has no information, and scoring it as
a success would inflate separability precisely on the pairs the scan exists to find.

## The bands

| Band | Condition | What it means |
|---|---|---|
| `DUPLICATE CAPABILITIES` | separability **≤ 0.50** | at or below the coin-flip line; the pair appears truly indistinguishable |
| `OVERLAPPING` | separability ≤ 0.65 **and** centroid similarity ≥ 0.50 | separable, but only just; usually not a capability defect |
| `MODEL CONFUSION` | symmetric misroute rate **≥ 0.40** | the trained router sends these commands' own utterances to each other |
| skipped | either side has fewer than 3 utterances | counted and reported, never silently scored |
| not examined | centroid similarity < 0.10 | cost pre-filter only; such pairs share almost no vocabulary |

The 0.10 pre-filter is a performance measure, not part of the definition — a 160-command workflow
has 12,720 pairs and the separability computation is quadratic in utterances. Raising it makes
the scan faster and can in principle hide a finding.

`MODEL CONFUSION` measured on utterances the model **trained on** is a strong signal: it failed
to separate the pair with the answer in its training set.

## Reading one finding

Four fields decide your response. Read them together:

- **`separability`** — how badly they collide. Below the duplicate line, no amount of seed
  rewriting will save the pair.
- **`shared_contexts`** — the contexts whose label space contains **both** commands. A pair that
  co-occurs is a live classifier conflict. A pair in disjoint contexts is a design ambiguity that
  surfaces through the escalation class and the parent walk instead, so the responses differ and
  the report does not merge them into one verdict. Empty does **not** mean safe: they can still
  collide through an ancestor's wildcard set, which this field does not show.
- **`shared_terms`** — what the seeds must stop relying on.
- **`terms_only_in_a` / `terms_only_in_b`** — what the seeds must lean on. Empty on one side is
  the diagnosis by itself: that side's seeds say nothing the other side's do not.

## The three responses

**Merge.** The pair is one capability. Delete the redundant command and accept the longer
navigation path: the extra turn is paid once per task, while the classifier confusion is paid in
every context where both labels are visible. Keep both only when the survivor genuinely cannot
serve the case — for example when a listing must prove parent-to-child membership by paging a
parent-scoped view, which a global search by id cannot do.

**Re-seed.** The commands mean different things and the seeds do not say so. Decide which token
class carries the decision — the entity noun for "same question asked of different things", the
verb for "different action on the same thing" — and make every seed on both sides carry it. The
`design-context-models` skill covers which side to anchor and why the fix is usually
one-directional.

**Accept.** The pair is a deliberate near-duplicate. Record it, or the same warning appears on
every run forever, and a permanent warning is one nobody reads.

## The accept-list

`<workflow>/accepted_duplicates.json`:

```json
{
  "schema_version": 1,
  "accepted": [
    {"commands": ["Order/list_items", "Customer/list_purchases"],
     "reason": "Different scopes; the customer view is required for the returns flow."}
  ]
}
```

- It lives at the **workflow root**, not in `___command_info`. It records a developer's decision,
  so it is input like `intent_benchmark.json` — not output. Inside `___command_info` it would sit
  in territory the trainer prunes and that developers are told to delete to force a rebuild, and
  losing the decision would silently resurrect a warning its author already answered.
- Command names must match **exactly** as printed, fully qualified. Matching on a bare leaf name
  would let one entry silently suppress a same-named command in an unrelated context.
- Order within a pair does not matter. A bare `["a", "b"]` pair is accepted, but the object form
  is the one to write: it is the only one with somewhere to record **why**.
- One entry suppresses the pair on **both** instruments. Having judged two commands to be the
  same capability, you should not be asked again after training.
- Accepted pairs are still scanned and still listed, under `ACCEPTED`, with your reason. The tool
  never hides what it found.
- **Nothing rewrites this file.** It is yours.

Two report sections exist to keep it honest:

- `ACCEPT-LIST PROBLEMS` — a malformed entry never raises (a typo must not stop an expensive
  run), so it simply suppresses nothing, and is reported separately from routine notes because a
  suppression you believe is in place is not.
- `STALE ACCEPT-LIST ENTRIES` — pairs accepted but no longer reported, usually because the seeds
  have since been made distinctive. Delete them.

## Two blind spots worth stating

1. **The lexical scan is a lexical instrument.** Synonymous *verbs* do not defeat it — a control
   pair saying "list/pull/give" against "search/find/look/get" still scores 0.00, because the
   weight sits on the shared domain nouns. What defeats it is a duplicate pair whose vocabulary
   is disjoint end to end: two commands that mean the same thing and share no distinctive term.
   Nothing lexical can see that. The router scan is what covers it.
2. **Leave-one-out is biased downward on small utterance sets.** Removing an utterance from its
   own centroid shrinks self-similarity while the comparison centroid keeps all its mass, so the
   0.5 cut is more permissive than a true chance line. On the shipped retail workflow the worst
   pair scores 0.600 against a 0.5 threshold — that is the empirical headroom.

Neither blind spot is a reason to distrust a finding. Both are reasons not to read a **clean**
report as proof there are no duplicates.

## Running it outside a training run

```python
from fastworkflow.train import duplicate_detection

report = duplicate_detection.scan_workflow(workflow_folderpath)   # seeds + accept-list
print(duplicate_detection.format_report(report))
```

`scan_workflow` collects seeds and contexts, scans, and applies the accept-list. It needs no LLM
key and no network, so it is cheap enough to run on every seed edit rather than waiting for a
training run to tell you.

Generated utterances work too and give a sharper estimate, since they are what is actually
trained on — pass them to `find_duplicate_capabilities` directly. Seeds are the useful default
because generation is conditioned on them, making a seed-level finding an early warning about
vocabulary generation will amplify.

## Triage checklist

```
- [ ] Read separability first: below 0.50 means re-seeding cannot fix it
- [ ] Check shared_contexts to decide whether this is a live conflict or a design ambiguity
- [ ] Check whether one side's terms_only_in_* list is empty — that is the diagnosis
- [ ] Choose merge / re-seed / accept, and act on all three lists, not just DUPLICATES
- [ ] Record accepted pairs with a reason, at the workflow root
- [ ] Delete stale accept-list entries the report names
- [ ] Do not read a clean report as proof of no duplicates — check MODEL CONFUSION too
- [ ] After re-seeding, retrain and compare cases, not percentages
```

## Related

- Which token should discriminate a colliding pair, and which side to anchor: the
  `design-context-models` skill.
- Confirming a re-seed actually helped, without being fooled by training variance: the
  `evaluate-intent-routing` skill.
- Where in a training run each scan happens, and why the router scan needs carried-forward
  contexts: the `train-and-publish-models` skill.
