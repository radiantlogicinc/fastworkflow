---
name: design-context-models
description: >-
  Design and tune the context inheritance (`base`) and context hierarchy (`parent`) models of a
  fastWorkflow command tree so intent routing stays accurate as the tree grows and a task takes the
  fewest commands. Covers deriving the containment graph from code, sizing base contexts to the
  parameter producers commands actually declare, removing duplicate commands, writing seed
  utterances that separate colliding labels, and verifying a retrain without being fooled by
  training variance. Use when routing accuracy is poor, a context carries too many commands, two
  commands are near-duplicates, wildcard escalation misfires, or a new context, workspace or base
  is being added to a workflow.
---

# Designing context models for routing

## The mechanism being optimized

Two files, two unrelated effects. Confusing them makes every later decision guesswork.

| File | Key | Effect |
|---|---|---|
| `_commands/context_inheritance_model.json` | `base` | Adds the base context's commands to the child's **callable surface**. Controls reachability. |
| `context_hierarchy_model.json` | `parent` | **No runtime navigation effect at all.** Read at training time only. |

A context's classifier does not choose among its surface. Its label set is:

```
own commands
+ inherited commands (transitive over `base`)
+ framework core commands + `wildcard`
+ every ancestor's surface, trained as `wildcard`     <- this comes from the `parent` axis
```

`go_up` and escalation both walk the **live object's** parent — the `Context.get_parent` callback
returning `obj.parent` — never the JSON. The `parent` axis exists to teach a context which
utterances are not its own, so it answers `wildcard`, which makes the runtime climb the live chain
and re-predict there.

Three consequences drive everything below:

- A `parent` edge no command can produce is **pure cost**. It can never be walked at runtime, and it
  still enlarges the wildcard class.
- A missing `parent` edge that *is* produced at runtime is **worse**. The child was never taught to
  reject that ancestor's utterances, so it confidently misroutes instead of escalating.
- Only a **lone** confident `wildcard` escalates. `wildcard` returned beside real candidates raises
  an ambiguity prompt instead, so "escalation" and "routing" trade against each other and must be
  scored separately.

## Rule 1 — derive the hierarchy from code, never hand-maintain it

The true `parent` graph is whatever each command constructs and assigns as the current context. Scan
for the assignment to `current_command_context` and record the class being entered; that set *is*
the containment graph. Diff it against the declared JSON and fix both directions: drop edges nothing
produces, add edges something produces.

Hand-maintained hierarchies drift silently, because a stale edge breaks nothing — it only costs
accuracy. Re-derive it whenever commands are added or deleted.

## Rule 2 — cutting the `parent` axis is free; cutting `base` is not

This is what resolves the tension between a small surface and a short path.

- **`parent` trimming costs no path length.** Reachability does not come from this file. Trim it to
  exactly the derived graph and the only thing that changes is that each classifier stops learning
  to reject things that could never reach it.
- **`base` trimming costs a real round trip** — but only sometimes. A command removed from a
  context's surface is still reachable *in place*, with no extra turn, if it lives in a **live
  ancestor**, because escalation will find it. It becomes genuinely unreachable if it only lives in
  a **sibling**.

So the safe cut depends on depth:

| Context | Parent at runtime | Cutting an inherited command |
|---|---|---|
| Entity nested under a workspace | that workspace | free — escalation still serves it in place |
| Top-level workspace | the global root | costs a real round trip out and back |

Before cutting anything from a `base`, confirm which case applies.

## Rule 3 — size each base context to its consumers' declared producers

Inheritance is all-or-nothing per base context, so a context that needs one command from a base must
take all of them. When a fat base is inherited by several contexts, split it.

The evidence for what a context needs is **not intuition** — it is the parameter-producer hints its
own commands declare (`available_from`). Collect, per context, the producers named by its commands;
that is the minimum base it needs.

Then chain the bases so nothing regresses:

```
FooLookup   (the narrow subset)  <- inherited by the workspaces that need only that
BarCatalog  (another subset)     <- inherited by the workspace that needs only that
Wide        base: [FooLookup, BarCatalog] + its own remainder
                                 <- inherited by the one context that needs everything
```

The wide base still resolves to the full set, so its consumer is untouched while the narrow
consumers shrink.

**A command's qualified name is its folder.** Moving a command between base contexts renames it, so
every producer hint, doc, test and routing artifact moves with it, and a retrain is mandatory.

## Rule 4 — delete a command that is the same query on another surface

Two commands that issue the same backend query with the same parameters, differing only in which
context they hang from, are one command. Delete the redundant one and accept the longer path: the
extra navigation turn is paid once per task, while the classifier confusion is paid in every context
where both labels are visible.

Keep both only when the surviving one cannot serve the case — for example when a listing must prove
parent-to-child membership by paging a parent-scoped view, which a global search by id cannot do.

## Rule 5 — know which token is supposed to discriminate

Near-duplicate labels are the dominant source of routing failure. For each colliding pair, decide
which token class carries the decision, then make **every seed on both sides** carry it.

| Pair shape | Discriminator | Example |
|---|---|---|
| Same question asked of different things | the **entity noun** | `Order/list_items` vs `Customer/list_purchases` — the head noun ("what they have") cannot divide them |
| Different action on the same thing | the **verb** | a global `open_reporting_workspace` vs `ReportingWorkspace/list_reports` |

If the seeds on one side name no entity at all ("list what it has", "their items please"), that side
cannot be separated no matter how the other side is worded.

## Rule 6 — collisions are directional; anchor the ancestor side

Two commands only compete when both land in the **same** model. That happens two ways: both on one
surface, or one of them arriving as an ancestor's `wildcard` negative. Work out which:

> `X` competes inside `Y` if `X` is on `Y`'s surface, or `X` belongs to an ancestor of `Y`.

Parent-child pairs are therefore **one-directional**. If `Parent/list_things` is a wildcard negative
down in `Child`, but `Child/list_things` never reaches `Parent`, then:

- Anchor the **ancestor** side to its entity noun in every seed.
- Let the **descendant** keep bare pronouns ("its items please", "what has it got"). Inside the
  child context a pronoun is unambiguous, and this preserves coverage of terse input.

Anchoring both sides is unnecessary and throws away the low-content register for no gain.

## Rule 7 — a global command lands in every context's wildcard class

Root-level navigation verbs are ancestors of everything, so their seeds are trained as `wildcard`
almost everywhere. A global verb whose seeds carry the vocabulary of the workspace it opens will
make that workspace's own commands lose to `wildcard`.

Keep global navigation seeds **strictly navigational** — where to go, not what to do once there. A
seed like "which reports am i receiving" on an `open_reports` verb is the workspace's listing
command asked verbatim; "sign up for a report" is its subscribe command.

## Rule 8 — shrinking a surface raises the wildcard share

Ancestor rows do not shrink when a surface does. Halve a context's commands and `wildcard` can
become several times the size of any single command class. That imbalance is tolerable when the
context's commands are lexically distant from its ancestors', and dangerous when they are close — so
after narrowing a base, check the narrowed context for over-escalation (its own commands losing to a
lone `wildcard`) before assuming the change was free.

## Procedure

```
- [ ] 1. Derive the containment graph from the command bodies
- [ ] 2. Diff it against context_hierarchy_model.json; fix edges in both directions
- [ ] 3. Per context, list surface vs. the producers its own commands declare
- [ ] 4. For each unused inherited command, check Rule 2 depth before cutting
- [ ] 5. Split fat base contexts; chain the wide base onto the narrow ones
- [ ] 6. Run the near-duplicate report; classify each pair by Rule 6 direction
- [ ] 7. Rewrite seeds per Rules 5-7
- [ ] 8. Retrain
- [ ] 9. Verify by case-level diff, not by percentages
```

Steps 1-5 are mechanical and safe. Steps 6-7 are where the accuracy is won: in practice a single
colliding pair can cost a context more than its entire label count does.

## Verification discipline

Retraining is mandatory after any of this: the label set is baked into each context's artifact, and
at inference a predicted label is looked up in a dict built from the *current* surface, so a stale
label fails hard rather than degrading.

Then read the results carefully. Four traps, all of which will otherwise produce a false conclusion:

1. **A re-drawn holdout is not a comparison.** If the evaluation redraws its fold each run, a context
   is never scored on the same phrasings twice — one run may score three of its labels and the next
   six. Per-context deltas are directional only.
2. **A shrinking label space shrinks the denominator.** Totals across runs are not like-for-like when
   the tree changed.
3. **Training variance is larger than it looks.** Expect whole-benchmark totals to move by a few
   points between runs with no input change, and individual framework commands to swing wildly.
4. **Only a fixed, hand-written case set compares across runs** — and only if the case set itself was
   not edited. When labels are renamed, the case set changes too, so treat it as an absolute reading.

The reliable signal is the **case-level diff**: list the failing cases before and after, and keep
only the ones the change can actually explain. A change touching one command's seeds cannot move an
unrelated framework command; that movement is churn. Report the causal set as a trade — "N cases
gained here against M lost there" — because a well-targeted seed fix usually does cost something at
the boundary it moved.

## Additional resources

- Diagnostic recipes, a failure-mode catalogue mapping symptom to cause to fix, and the exact
  computations for label space and surface: see [reference.md](reference.md)

## Related

- Step 6's near-duplicate report — what separability measures, the reporting bands, and the
  accept-list for a pair you have decided is deliberate: the `detect-duplicate-capabilities`
  skill.
- Step 8's retrain — what a rename invalidates, why some changes retrain everything, and how to
  roll back: the `train-and-publish-models` skill.
- Step 9's verification — which reported number is comparable across runs and which is not: the
  `evaluate-intent-routing` skill.
