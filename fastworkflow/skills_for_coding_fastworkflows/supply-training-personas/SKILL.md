---
name: supply-training-personas
description: >-
  Decide who writes a fastWorkflow's synthetic training utterances, by adding a `personas.json`
  that either supplies the personas outright or conditions the PersonaHub draw on the
  application's domain keywords. Covers the file schema and where it is discovered, choosing
  between the two modes, the persona id rules that keep held-out evaluation honest, sizing a
  domain-filtered pool, and the full retrain a persona change forces. Use when generated
  utterances read nothing like the workflow's real users, when a workflow must train without
  downloading PersonaHub, or when authoring or reviewing a `personas.json`.
---

# Supplying training personas

## The mechanism being changed

Training utterances are written by an LLM role-playing a persona. By default that persona is
drawn uniformly from the whole PersonaHub corpus (~200k rows) and conditioned on **nothing** — a
retail workflow and an identity-governance workflow get the same population of astrophysicists,
sommeliers and marine biologists. Most drawn personas have no relationship to the vocabulary the
application's users actually employ.

`personas.json` is how an application states who its users are. It changes the *pool*; the
sampling itself is unchanged and stays deterministic.

## Two modes, one file

| `personas.json` contains | Source | Effect |
|---|---|---|
| a non-empty `personas` list | `app_supplied` | Your personas are used verbatim. **PersonaHub is never consulted or downloaded.** |
| `domain_keywords` (or `domain`) and no `personas` | `domain_conditioned` | PersonaHub is filtered to rows mentioning those keywords; the usual deterministic sample is taken from the survivors. |
| neither | — | error, not a fallback |

An explicit `personas` list **wins** over `domain_keywords` — supplying personas is the stronger
statement, and honouring the keywords as well would mix generic rows into a curated set.

## The file

Discovered at `<workflow>/personas.json`. Both forms parse:

```json
{
  "schema_version": 1,
  "domain": "retail order management",
  "domain_keywords": ["retail", "e-commerce", "shopper", "customer service"],
  "personas": [
    {"id": "impatient-shopper", "persona": "A shopper who orders often, tracks every delivery, and types in short fragments."},
    {"id": "support-agent", "persona": "A support agent handling returns and refunds all day; uses internal jargon."}
  ]
}
```

- A **bare JSON array** of personas is accepted, so a quick file needs no wrapper.
- An entry may be a **bare string** instead of an object; its id becomes its position.
- An object entry takes its text from `persona` or `text`, and its id from `id` or its position.
- `schema_version` must be `1`.
- `domain` prose with no `domain_keywords` is usable on its own — its own words (longer than two
  characters, lowercased) become the keywords. That is the least-effort form of the feature.

`SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE` points at a persona file outside the workflow folder, and
is the only route that needs no change to the workflow at all. Set it in `fastworkflow.env`, not
as a shell `export`.

Discovery order: `<workflow>/personas.json` → `SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE` → plain
PersonaHub.

## Rule 1 — a broken file fails the run, a broken env path does not

A `personas.json` that exists but cannot be used raises `PersonaConfigError` and stops training.
That is deliberate: an application that ships a persona file has stated an intent about its
training data, and silently training on the generic draw instead would produce a model whose
provenance record does not describe how it was built.

The env-var route is weaker. A `SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE` pointing at a path that
does not exist is treated as unset and the run falls back to PersonaHub **silently**. If you use
the env route, confirm the file was actually read by looking for the `Persona source:` line in
the training log.

An empty file — no personas and no keywords — is an error too, with the fix stated in the
message: delete the file to use the default draw. An empty file is more likely a mistake than an
intent.

## Rule 2 — ids must not collide with the holdout's vocabulary

Your ids are namespaced to `app:<id>` automatically, so PersonaHub row 42 and your persona `42`
stay distinguishable in a provenance record. Two constraints survive that namespacing:

- **No `+` in an id.** `+` joins the contributors of a composite persona id, and
  `heldout_evaluation.expand_persona_id` splits on it to recover the atomic personas behind one
  utterance. An id containing `+` is silently split into fragments and the whole-persona holdout
  leaks — which is the exact defect that holdout exists to prevent. Rejected at load time.
- **Ids must be unique.** Otherwise the provenance record cannot say which persona wrote what.

The reserved ids `__seed__` (hand-written seed utterances, never held out) and the
`__unresolved__:` prefix are also rejected, though the `app:` prefix means a hand-written file
cannot reach them.

## Rule 3 — a domain-filtered pool has a floor, and padding is reported

Filtering is a whole-word match on the lowercased persona text; multi-word keywords match as
substrings; a row matching any keyword survives. Surviving rows keep their original PersonaHub
ids, so a provenance record stays comparable with an unfiltered run.

The pool needs to be at least `4 x` the per-command persona count
(`SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS`), because a smaller pool gives every command nearly the
same handful of rows — which reintroduces the near-duplication the whole-persona holdout exists
to detect. Three outcomes, each reported once per run as a `Persona source:` log line:

| Matched rows | What happens | What to do |
|---|---|---|
| enough | the pool is exactly the matched rows | nothing |
| some, but under the floor | topped up from the unfiltered corpus; **matched rows are still drawn first**, so a command asking for more personas than matched gets unconditioned ones for the remainder | add keywords, or supply `personas` |
| none | falls back to the full corpus; "Domain conditioning had no effect on this run" | fix the keywords — they are almost certainly misspelt or too specific |

Padding rather than failing is deliberate: a mistyped keyword must not stop a multi-hour training
run at its first command. That makes the log line the only thing that tells you the feature did
nothing, so read it.

An app-supplied set of fewer than 4 personas gets its own warning for the same reason: every
command draws from nearly the same set, the utterances share a voice, and the holdout has little
to hold out.

## Rule 4 — adding or editing this file forces a full retrain

The active persona source contributes to two different keys, and a change moves both:

1. **The utterance-cache key.** Every cached generated utterance for the workflow is invalidated,
   so generation runs again and **costs money**.
2. **The training signature's global header.** A global-header change is one of the conditions
   that makes `fastworkflow train` choose a **full** retrain over a selective one — every
   context, not just the ones whose commands changed.

Three things independently move that key, and the cache-miss report names which:

- the source **name** — app-supplied and domain-conditioned are different variants;
- the **content** — the persona text, or the keyword list, so editing either invalidates the
  utterances they wrote;
- the **pool code** — the framework's own row-filtering and padding logic, so tuning it cannot
  serve you the old personas back.

The default PersonaHub draw contributes **nothing** to either key. That is why adding this
feature invalidated no existing workflow, and it is also why removing your `personas.json` is not
free: it is another change of source.

Plan for this. Do not add a `personas.json` in the same change as a seed-utterance edit you want
to measure — you will not be able to attribute the result.

## What this does not do

It does not improve intent detection. Domain-conditioned personas are a **hypothesis** —
plausibly a larger generalisation lever than anything in the training loop, but unmeasured in
this repository, and currently not measurable from a single paired run, because the utterances
come from a live LLM and two runs at the same seed produce different training data. The feature
makes that experiment possible. Nothing in the code or in this skill reports a result.

So treat a persona change as a change of training data, not as a fix. If you want to claim it
helped, you need the paired-build discipline in the `evaluate-intent-routing` skill.

## Verifying it took effect

```
- [ ] The training log shows a `Persona source:` line naming the source you configured
- [ ] For a domain-conditioned pool, that line reports a match count, not a fallback or a top-up
- [ ] `___command_info/training_provenance.json` records the source name for each command
- [ ] The utterance cache regenerated rather than reusing (the cache summary line at the end of
      the run distinguishes reused from generated)
```

## Authoring checklist

```
- [ ] personas.json is at the workflow root, next to intent_benchmark.json
- [ ] schema_version is 1
- [ ] Exactly one mode is expressed: a personas list, or domain keywords — not both by accident
- [ ] No id contains '+'; ids are unique
- [ ] Each persona describes how someone TALKS, not just who they are
- [ ] At least 4 personas if supplying them outright
- [ ] Domain keywords are common words that appear in persona descriptions, not product names
- [ ] The retrain this forces is budgeted for, and isolated from other changes being measured
```

## Related

- The retrain this triggers, and why it is full rather than selective: the
  `train-and-publish-models` skill.
- Measuring whether the resulting models are any good: the `evaluate-intent-routing` skill.
- If the problem is two commands that no persona could phrase apart, personas will not fix it —
  see the `detect-duplicate-capabilities` skill.
