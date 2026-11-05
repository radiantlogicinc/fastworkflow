---
name: declare-parameter-producers
description: >-
  Declare producers on fastWorkflow command parameters with `from_` so the planner can chain commands
  into a working sequence and so a missing parameter tells the caller which command supplies it.
  Covers the `from_` form (command names and `<command>.<field>` references), how the planner, the
  parameter listing and the missing-parameter message consume it, choosing a producer the caller can
  reach, matching handle types between producer and consumer, and what `command preview --check`
  verifies. Use when a command takes an id, uid, code or other opaque handle it cannot invent, when an
  agent cannot work out the order to call commands in, or when authoring or reviewing any command
  `Signature.Input`.
---

# Declaring parameter producers with `from_`

A command that takes an opaque handle (an id, uid, code, ticket number) cannot be called until
something produces that handle. `from_` records which command that is, turning a flat list of
commands into a graph the planner can walk.

## The declaration

On the `Input` field, with `from_` in place of `Field`:

```python
from fastworkflow.planning import from_   # the same helper planning commands import

class Signature:
    class Input(BaseModel):
        order_id: str = from_(
            "CustomerDesk/find_order.order_ids",
            description="The order to fetch",
            examples=["#W0000000"],
        )
```

`from_(source, **field_kwargs)` is `Field(**field_kwargs)` with the source recorded as the producer
edge (`planning_phases.py` `from_`). The source takes three forms:

- `"<Context>/<command>.<field>"`: a field of the producer's `Output`. Prefer this; it says which
  value the consumer needs.
- `"<Context>/<command>"` or `"<command>"`: a bare command. Accepted, but `command preview --check`
  reports it at info level and asks for a field.
- a list of those, when several commands can supply the handle: `from_(["A/x.id", "B/y.id"])`.

`from_` is for an `Input` field. An `Output` field cannot declare it (build the value in
`__result__`, for a planning command), and a facts model field is typed with a command's `Output` or `Input`, not `from_` (and not a copy: `fact_from` is removed).

`available_from` is no longer written by hand. A hand-written `json_schema_extra={"available_from":
...}` is an error that `command preview --check` reports ("available_from is written by from_"):
convert it to `from_`.

## Who reads it

Three consumers, and they behave differently:

1. **The planner.** Its instruction is to walk the graph of producer edges to build the command
   sequence (`workflow_agent.py` `PLANNER_INSTRUCTIONS`), and the agent's command display renders each
   resolved edge as `available_from: [...]` beside the parameter. The runtime text keeps that name; you
   still write `from_`. The edges are the only structured statement of "you
   must call X before Y", so an unhinted handle parameter is a parameter the planner has to guess its
   way to.
2. **The missing-parameter message.** When validation finds the field absent, it appends
   `use the <producers> command(s) to get <field> information` (`utils/signatures.py`). A human caller
   is told to `abort and use ...`; an agent (`run_as_agent` set on the workflow context) is told to
   just use it, because the agent can chain without abandoning the turn.
3. **`command preview --check`.** It resolves every `from_` edge and reports the ones that do not hold
   (see "What the check verifies" below).

Only `Input` fields are read. The edge is surfaced **only when the field is missing**, so it never
substitutes for `description`; write both.

**Above the listing size.** A workflow too large to list whole can show the planner a shortlist
instead: the commands that match the request, plus the producers of their inputs. A producer edge is
what brings a producer onto that shortlist, and a `descends_to` brings in the way to a context. The
shortlist is built but **off** in this release, so the planner sees the full command list and the
edges work as described above. Write the edges as if the shortlist were on, because it is expected to
be switched on with the plan review round.

## Rule 1: name a producer the caller can actually reach

A hint naming a command the caller cannot invoke is worse than no hint: it sends the planner down a
path that dead-ends. Reachable means either

- on the calling context's own surface, own or inherited, or
- on a **live ancestor's** surface, since a `wildcard` prediction escalates up the runtime parent
  chain and serves the command there without a navigation turn.

A **sibling** context is not reachable. Verify against the resolved routing definition, not against
the folder layout.

## Rule 2: the producer must emit the handle the consumer accepts

Two commands can both return "an order id" and still not be interchangeable.

- **Parent-scoped listings accept only their own children.** If `Project/list_tasks` verifies
  membership by paging a parent-scoped view, a global `find_task` returns handles it will reject.
  Name the sibling listing instead.
- **When the handle parameter is optional, the command is its own producer.** A listing that opens a
  child when given a uid and lists them when not is the only thing that yields uids it will accept:

```python
task_id: Optional[str] = from_(
    "Project/list_tasks.task_ids",                  # itself
    default=None,
    description="Task to open. Omit to list them instead.",
)
```

This is not a cycle. It reads as "call me bare first, then call me again with one of the handles I
returned." (A planning command's own `Input` may not name one of the commands its steps call; that is
a `--check` error.)

- **An edge is authoritative, not additive.** Where a default by-name convention would attach a
  global search, an explicit edge replaces it. Do not add the global search to a parent-scoped
  producer "just in case": every handle it yields is one the consumer rejects.

## Rule 3: describe the handle at both ends

The planner has to connect a producer's output to a consumer's input by reading prose. Make that
mechanical: have the producing command's `Output` name the field the consumer asks for.

```python
class Output(BaseModel):
    task_ids: list[str] = Field(
        description="The task_id of every task listed, in the order shown. "
                    "Pass one to a command that asks for a task_id.")
    labels: list[str] = Field(
        description="Human-readable name of each task, aligned by index with task_ids.")
```

When a listing returns two different handles, say so explicitly. An agent that confuses them will
pass the wrong one and get a not-found:

```python
    account_ids: list[str] = Field(
        description="The account each entitlement is held through, aligned by index. "
                    "The two are different handles and open different things.")
```

## Rule 4: keep the edges true as the tree changes

`from_` names a command by its qualified name, so moving a command between contexts, or deleting it,
invalidates every edge pointing at it. `command preview --check` reports an edge whose producer no
longer resolves, which catches a deleted or renamed command. It does not re-check reachability for a
qualified name that still exists, so re-check reachability whenever a context's inherited surface
changes (Rule 1).

## What `command preview --check` verifies

`fastworkflow command preview <workflow_folder> --check` runs `planning_commands.resolve_field_sources`
over every execution and planning command and reports each `error`. For each `from_` edge it checks:

- the reference is well formed (`[<Context>/]<command>[.<field>]`);
- the producer exists. A qualified name must exist; an unqualified name must be on the consumer's
  static lookup path (its own context, then its bases, then the global commands), and be unique there;
- a planning command's `Input` does not name a command its own steps call;
- with a field named, the producer's `Output` has that field, which is not an `Input` field, and its
  type matches the consumer's (equal, equal modulo `Optional`, or a list of the consumer's type);
- a hand-written `available_from` is reported.

It does **not** check Rules 2 and 3: whether the producer emits the handle the consumer accepts
(parent-scoped membership, the self-producing listing), whether the descriptions say what the
handles are, or whether an edge's producer is reachable from a live ancestor. Those stay authoring
guidance, as the rules above say.

## Authoring checklist

```
- [ ] Every non-inventable handle parameter carries a from_ edge
- [ ] The edge is context-qualified (Context/command.field)
- [ ] Each named producer is reachable from the consuming context (surface or live ancestor)
- [ ] Each named producer emits handles this consumer will accept (Rule 2)
- [ ] Parent-scoped consumers name the sibling listing, not a global search
- [ ] Optional-handle listings name themselves
- [ ] The producer's Output field names the consumer's input field in its description
- [ ] `description` is written independently: the edge only appears when the field is missing
- [ ] `command preview --check` passes
```

## Related

- `from_` answers "the parameter never arrived". Its counterpart for "the parameter arrived
  unusable" is the `validate-command-parameters` skill, and for "the parameter arrived nearly right"
  the `resolve-parameter-values` skill.
- Reachability is decided by the context models; see the `design-context-models` skill.
- Writing a planning command that uses these edges: the `author-planning-commands` skill.
