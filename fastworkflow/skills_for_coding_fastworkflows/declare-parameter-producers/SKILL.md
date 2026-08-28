---
name: declare-parameter-producers
description: >-
  Declare `available_from` producer hints on fastWorkflow command parameters so the planner can
  chain commands into a working sequence and so a missing parameter tells the caller which command
  supplies it. Covers the hint format, how the planner and the missing-parameter message consume it,
  choosing a producer the caller can actually reach, and matching handle types between producer and
  consumer. Use when a command takes an id, uid, code or other opaque handle it cannot invent, when
  an agent cannot work out the order to call commands in, or when authoring or reviewing any command
  `Signature.Input`.
---

# Declaring parameter producers with `available_from`

A command that takes an opaque handle — an id, uid, code, ticket number — cannot be called until
something produces that handle. `available_from` records which command that is, turning a flat list
of commands into a graph the planner can walk.

## The hint

On the `Input` field, context-qualified:

```python
class Signature:
    class Input(BaseModel):
        order_id: str = Field(
            description="The order to fetch",
            examples=["#W0000000"],
            json_schema_extra={'available_from': ['CustomerDesk/find_order']},
        )
```

The value is a **list** — name every command that can legitimately supply the handle.

## Who reads it

Two consumers, and they behave differently:

1. **The planner.** Its instruction is literally to walk the graph of `available_from` hints to build
   the command sequence. The hints are the only structured statement of "you must call X before Y",
   so an unhinted handle parameter is a parameter the planner has to guess its way to.
2. **The missing-parameter message.** When validation finds the field absent, it appends
   `use the <producers> command(s) to get <field> information`. A human caller is told to
   `abort and use ...`; an agent (`run_as_agent` set on the workflow context) is told to just use it,
   because the agent can chain without abandoning the turn.

Only `Input` fields are read. Putting the hint on an `Output` field does nothing.

The hint is surfaced **only when the field is missing**, so it never substitutes for `description` —
write both.

## Rule 1 — name a producer the caller can actually reach

A hint naming a command the caller cannot invoke is worse than no hint: it sends the planner down a
path that dead-ends. Reachable means either

- on the calling context's own surface, own or inherited, or
- on a **live ancestor's** surface, since a `wildcard` prediction escalates up the runtime parent
  chain and serves the command there without a navigation turn.

A **sibling** context is not reachable. Verify against the resolved routing definition, not against
the folder layout.

## Rule 2 — the producer must emit the handle the consumer accepts

Two commands can both return "an order id" and still not be interchangeable.

- **Parent-scoped listings accept only their own children.** If `Project/list_tasks` verifies
  membership by paging a parent-scoped view, a global `find_task` returns handles it will reject.
  Name the sibling listing instead.
- **When the handle parameter is optional, the command is its own producer.** A listing that opens a
  child when given a uid and lists them when not is the only thing that yields uids it will accept:

```python
task_id: Optional[str] = Field(
    default=None,
    description="Task to open. Omit to list them instead.",
    json_schema_extra={'available_from': ['Project/list_tasks']},   # itself
)
```

This is not a cycle. It reads as "call me bare first, then call me again with one of the handles I
returned."

- **A hint is authoritative, not additive.** Where a default by-name convention would attach a global
  search, an explicit hint replaces it. Do not append the global search to a parent-scoped
  producer "just in case" — every handle it yields is one the consumer rejects.

## Rule 3 — describe the handle at both ends

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

When a listing returns two different handles, say so explicitly — an agent that confuses them will
pass the wrong one and get a not-found:

```python
    account_ids: list[str] = Field(
        description="The account each entitlement is held through, aligned by index. "
                    "The two are different handles and open different things.")
```

## Rule 4 — keep the hints true as the tree changes

`available_from` names a command by its qualified name, so moving a command between contexts, or
deleting it, silently invalidates every hint pointing at it. Grep for the old qualified name whenever
a command is renamed, moved or removed, and re-check reachability whenever a context's inherited
surface changes.

## Authoring checklist

```
- [ ] Every non-inventable handle parameter carries a hint
- [ ] The hint is context-qualified
- [ ] Each named producer is reachable from the consuming context (surface or live ancestor)
- [ ] Each named producer emits handles this consumer will accept
- [ ] Parent-scoped consumers name the sibling listing, not a global search
- [ ] Optional-handle listings name themselves
- [ ] The producer's Output field names the consumer's input field in its description
- [ ] `description` is written independently — the hint only appears when the field is missing
```

## Related

- `available_from` answers "the parameter never arrived". Its counterpart for "the parameter arrived
  unusable" is the `validate-command-parameters` skill, and for "the parameter arrived nearly right"
  the `resolve-parameter-values` skill.
- Reachability is decided by the context models; see the `design-context-models` skill.
