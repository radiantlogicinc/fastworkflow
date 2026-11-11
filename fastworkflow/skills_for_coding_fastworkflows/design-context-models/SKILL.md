---
name: design-context-models
description: >-
  Design and tune the context inheritance (`base`) model of a fastWorkflow command tree, and the
  enclosing record each context's `get_parent` returns, so intent routing stays accurate as the tree
  grows and a task takes the
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

| File or source | Key | Effect |
|---|---|---|
| `_commands/context_inheritance_model.json` | `base` | Adds the base context's commands to the child's **callable surface**. Controls reachability. |
| `get_parent` in `_commands/<Context>/_<Context>.py` | the enclosing object | **The only parent source.** Each call reads the live object chain at run time. |

There is no `context_hierarchy_model.json`. A workflow that still has one loads, logs a warning that
the file is ignored and can be deleted, and routes the same way. `fastworkflow build` does not
generate it.

A context's classifier does not choose among its surface. Its label set is:

```
own commands
+ inherited commands (transitive over `base`)
+ framework core commands
+ `wildcard`, the reserved "not here" class
```

The "not here" class is built from **one hand-written seed of every command that is not one of the
context's own or inherited labels**, planning commands included. Ancestor commands are not labels of
their own; they are foreign commands, so each contributes one seed to the class. The internal CME
workflow has no "not here" class. Each head records which seed it used for each foreign command, in
its training signature.

Enclosing commands are reached at run time by walking the **live object's** parents: the current
context and its bases, then each enclosing record and its bases, then the global commands. A
"not here" answer moves the walk to the next scope; the nearest scope that recognises the request wins.
A command found in an enclosing scope runs on that record and is filed under it.

Consequences that drive everything below:

- A `get_parent` that returns the wrong object, or `None` where an enclosing record exists, hides
  every command of that record from the walk. Check it first when an enclosing command never runs.
- Only a **lone** confident `wildcard` escalates. `wildcard` returned beside real candidates raises
  an ambiguity prompt instead, so "escalation" and "routing" trade against each other and must be
  scored separately.
- The "not here" class grows with the number of commands in the workflow, one seed each, not with the
  depth of the context tree.

## Planning commands follow the same visibility

A planning command is visible exactly where a command of the same folder is. A file
`_commands/<Context>/<name>.py` is the planning command `<Context>/<name>`, and it is matched in the
context it sits in and wherever that context's commands are reachable, on the same lookup path as an
execution command:

1. the active context;
2. its bases, from `context_inheritance_model.json` `base`, transitively;
3. each enclosing record on the live chain, and that record's bases;
4. the global commands, in the `_commands/` root.

For a natural-language request the global head is asked first: a confident global command wins and
runs bound to the root object; otherwise the current context's head is asked and the climb continues.
An exact command name skips this step and takes the nearest scope. So a planning command in a base
context reaches every context that inherits the base, and one in a child context is not offered from a
sibling. Moving or renaming a planning command
changes its qualified name, exactly as for a command, so its `from_command_output` edges and previews move with it.

## Rule 1 — keep `get_parent` honest; there is no second copy to maintain

The parent of a context is what its `get_parent` returns for the live object. There is no JSON copy
to drift from it. Check each `get_parent` against the object it should return: a record's parent is
the record that holds it, and the root returns `None`. The scan that the earlier design used
(recording which record each command assigns to `workflow.record`) still shows which contexts
a command can enter, which is what `descends_to` needs, and what the shortlist will need when it is on.

## Rule 1a — navigation is declared once, with `descends_to`, on the command

A command that moves the workflow into another context declares it on its own class:
`@descends_to("<Context>")` on its `ResponseGenerator` or `PlanGenerator` (with `when_parameter="<param>"`
when a parameter picks the entry). Do not write
`enter_command` or `enter_commands` on a context class. That is a load error, and it names the command
and the fix. A command that moves the workflow into a context it does not declare logs one warning per
command and target. Planning and the run's listing read only the `descends_to` declarations.

## Rule 1b — `@user_approval_for_execution` is a command declaration too, and it is a gate

Next to `@descends_to`, a command's class may carry `@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)`
or `@user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)` (on its `ResponseGenerator`, or a
planning command's `PlanGenerator`). Absent means the agent decides, with no gate. It is validated at load and
shown in the listings with a marker. A `REQUIRED_ALWAYS` command asks before every run, including a run the
user typed; a `REQUIRED_UNTIL_DELEGATED` command asks until a standing approval exists. The reply is `yes`,
`no` or `always`; `always` records the delegation for that user, workflow and command. The approval is bound
to the exact parameters, is single-use and expires. It does not change navigation, and it is not read/write
gating: an easy-to-undo write needs no annotation. See `docs/planning_command_file_layout.md`,
"Approval annotation". Writing is declared separately, with `@writes` on the `ResponseGenerator` of every
command that changes data: inside a planning command's phase such a command runs only when a step of the phase
names it (the `author-planning-commands` skill, "Commands that write").

## Naming a record: `get_displayname` and `instance_label_attr`

A context class declares two things about the record it manages, and they answer different questions. Keep
them apart:

| | `get_displayname(cls, record)` | `instance_label_attr = "<attribute or property>"` |
|---|---|---|
| What it is | a classmethod that returns any free-form display string | a class attribute that names one attribute (or property) of the record |
| Example value | `"Book: Dune (b12)"` | `"uid"`, whose value on the record is `b12` |
| What it answers | "how should this record look to a person?" | "what is this record's identity?" |
| Who reads it | `workflow.record_displayname` (for the record the command acts on), the CLI and chat context line, observability | `context_identity` (listing tags, `search_memory`, the extract step, the approval binding) |
| When absent | optional; without it the context name is shown | nothing is guessed; the record has no identity in those readers |
| Derived from the other? | no: it may use the identity, but `instance_label_attr` is never read from it | no: it names an attribute, it does not compute one |

`workflow.record_displayname` reads `get_displayname` for the record the command acts on. It is a reader of
that declaration, not a third one. A computed identity is a property on the record, which `instance_label_attr`
names. The `instance_label` classmethod is removed (a context that still defines one fails the load).


## Rule 2 — cutting the `parent` axis is free; cutting `base` is not

This is what resolves the tension between a small surface and a short path.

- **Reachability comes from `base` and from `get_parent`, never from a second file.** A command
  removed from a context's surface is still reachable *in place*, with no extra turn, if it lives in
  a **live enclosing record**, because the walk climbs to it. It becomes genuinely unreachable if it
  only lives in a **sibling**.

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
own commands declare (`from_command_output` edges, which `command preview --check` resolves). Collect, per context, the producers named by its commands;
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
surface, or one of them arriving as a seed of the other head's `wildcard` ("not here") class. Work out
which:

> `X` competes inside `Y` if `X` is on `Y`'s surface, or `X` is a foreign command of `Y` (one seed in
> `Y`'s "not here" class).

Parent-child pairs are therefore **one-directional**. If `Parent/list_things` is a "not here" seed
down in `Child`, but `Child/list_things` never reaches `Parent`, then:

- Anchor the **ancestor** side to its entity noun in every seed.
- Let the **descendant** keep bare pronouns ("its items please", "what has it got"). Inside the
  child context a pronoun is unambiguous, and this preserves coverage of terse input.

Anchoring both sides is unnecessary and throws away the low-content register for no gain.

## Rule 7 — a global command lands in every context's wildcard class

Root-level navigation verbs are foreign to every context, so their seeds are trained as `wildcard`
almost everywhere: each global command contributes one seed to every head's "not here" class. A global verb whose seeds carry the vocabulary of the workspace it opens will
make that workspace's own commands lose to `wildcard`.

Keep global navigation seeds **strictly navigational** — where to go, not what to do once there. A
seed like "which reports am i receiving" on an `open_reports` verb is the workspace's listing
command asked verbatim; "sign up for a report" is its subscribe command.

## Rule 8 — shrinking a surface raises the wildcard share

The "not here" class holds one seed per foreign command. Halving a context's commands raises its
share of foreign seeds, so `wildcard` can become several times the size of any single command class.
That imbalance is tolerable when the context's commands are lexically distant from the foreign ones,
and dangerous when they are close — so after narrowing a base, check the narrowed context for
over-escalation (its own commands losing to a lone `wildcard`) before assuming the change was free.

## Procedure

```
- [ ] 1. Check each context's get_parent against the live object it should return
- [ ] 2. Check which contexts each command enters (`workflow.record` assignments)
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
label fails hard rather than degrading. An artifact trained before scoped lookup refuses to load and
says to retrain with `fastworkflow train`; the first train after that is a full retrain. After that,
editing one command's seeds retrains only the heads that hold it (plus any head that recorded its
seed), and adding or removing a command retrains every head.

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
