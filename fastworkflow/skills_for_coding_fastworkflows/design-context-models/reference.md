# Reference: diagnostics and failure modes

Companion to [SKILL.md](SKILL.md). Read when a specific number or a specific symptom is needed.

## Computing the numbers

### Surface of a context

Own commands (the `.py` files in `_commands/<Context>/`, excluding leading-underscore files) plus
the transitive closure over `base`. Files directly under `_commands/` are the global `*` context.

```python
def surface(ctx):
    out = set(own.get(ctx, []))
    for b in inheritance.get(ctx, {}).get("base", []):
        out |= surface(b)
    return out
```

Cross-check against the framework's own loader rather than trusting the hand computation — it
validates the JSON at the same time:

```python
from fastworkflow.command_context_model import CommandContextModel
m = CommandContextModel.load(workflow_path)
m.commands(ctx)                  # effective surface, qualified names
m.get_ancestor_contexts(ctx)     # transitive `parent` closure
```

### Label space of a context

What its classifier must actually separate:

```
len(surface(ctx))
+ len(union(surface(a) for a in ancestors(ctx)) - surface(ctx))
+ 4 core commands
+ 1 for `wildcard`
```

Track the sum over enterable contexts as the headline size metric. Base-tier contexts are not
enterable — nothing navigates into them — so exclude them from that sum. Note that they are still
trained, which is wasted work worth flagging if the framework does not skip them.

### Deriving the containment graph

Scan each command module for the object assigned to the current context:

```python
pat_ctx = re.compile(r"current_command_context\s*=\s*(\w+)")
pat_new = re.compile(r"(\w+)\s*=\s*(\w+)\(")   # resolve the local back to its class
```

The result maps `Context/command -> entered class`. Inverting it gives the true `parent` graph.
Anything in `context_hierarchy_model.json` that is not in this inversion is a stale edge.

### Which producers a context needs

Collect the parameter-producer hints declared by each context's own commands:

```python
re.finditer(r"'available_from':\s*\[([^\]]*)\]", src)
```

Group by owning context. That is the minimum set of producers the context needs on its surface, and
therefore the minimum base it can inherit.

## Failure-mode catalogue

### Own commands lose to a lone `wildcard`

**Symptom.** Several of a context's own commands come back as `wildcard`, so the runtime escalates
and the user is told the request cannot be served here.

**Cause.** The wildcard class is disproportionate — usually after a surface was narrowed without the
ancestor rows shrinking — *and* something in an ancestor is lexically close. The commonest offender
is a global navigation verb whose seeds describe what the user wants done rather than where they
want to be.

**Fix.** Make the ancestor's seeds strictly navigational (Rule 7). Verify by checking that the
context's own commands recover, not just that the totals moved.

### A command absorbs phrasings belonging to a parent

**Symptom.** A descendant command fires on utterances that name the parent's entity, and the
escalation report shows those cases as "escalation label present but not alone".

**Cause.** The parent's command arrives as a wildcard negative and its seeds are unanchored, so the
descendant's own seeds are the closer match for anything vague.

**Fix.** Anchor the ancestor side to its entity noun (Rule 6). Leave the descendant's pronouns alone.

### Two same-shaped commands in one context

**Symptom.** A near-duplicate report flags a pair with `shared_contexts` naming a real context, and
that context's top-1 is well below its in-list.

**Cause.** Both labels are on one surface and differ only by a noun the seeds do not emphasise.

**Fix.** Either delete one (Rule 4) or make the differing noun the only thing the seeds vary. A
family of `open_<entity>_by_uid` commands is the classic case: they share `open`, `by`, `uid` and
differ in one word.

**Do not** reflexively merge such a family into one command with an enum parameter. That trades an
intent-detection problem for a parameter slot-filling problem, and slot filling may have no
clarification path equivalent to the one intent detection has. Check whether the framework can
clarify an enum parameter value before proposing the merge.

### A context routes well but never escalates

**Symptom.** High routing accuracy, low escalation recall.

**Cause.** Missing `parent` edges — the context was never trained to reject its ancestors' commands.

**Fix.** Re-derive the hierarchy (Rule 1). This is the failure mode that a hierarchy audit catches
and that no amount of seed work will.

### Accuracy drops after an unrelated change

**Symptom.** A context nobody touched moves several points; a framework command such as `go_up`
swings from near-perfect to near-zero.

**Cause.** Training variance.

**Fix.** Nothing. Confirm by checking whether the moved cases are reachable from the change at all,
and report them as churn.

## Near-duplicate separability

A pairwise separability score below the report's overlap threshold means a classifier restricted to
just those two commands barely beats chance — the pair is unresolvable by any amount of context.
Treat that report as the primary design signal, ahead of accuracy: accuracy tells you a context is
struggling, separability tells you which two labels to fix.

Read three fields together:

- `separability` — how badly they collide.
- `shared_contexts` — whether they collide on a surface. Empty does **not** mean safe: they may still
  collide through an ancestor's wildcard set, which this field does not show. Check the hierarchy.
- `terms_only_in_a` / `terms_only_in_b` / `shared_terms` — the shared terms are what the seeds must
  stop relying on; the exclusive terms are what they must lean on.

## What to update when a command moves between contexts

A command's qualified name is `<Folder>/<command>`, so moving it is a rename. Update:

- the module's location on disk, and the scaffold or generator that emits it
- every `available_from` producer hint naming it
- the seed-utterance table keyed by `(context, command)`
- the inheritance model, and any `inherits:` prose in generated docs
- benchmark and end-to-end case tables
- navigation docs and command maps

Then retrain. Cached generated utterances are fingerprinted on the seed list *and* the command
name, so a rename invalidates exactly the moved commands and reuses the rest — prefer a plain
retrain over forcing full regeneration, which is slower and discards reproducibility for
untouched commands. The full list of fingerprint inputs, and what forces a full retrain anyway, is
in the `train-and-publish-models` skill.

## Reporting template

```markdown
## What changed
[the edits, grouped by rule]

## Measured effect
[totals, with the comparability caveat stated]

## Case-level diff
[cases fixed vs cases broken, restricted to what the change can explain]

## Left on the table
[flagged pairs not addressed, and why]
```
