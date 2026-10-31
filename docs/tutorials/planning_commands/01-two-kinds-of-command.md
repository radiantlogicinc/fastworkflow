# 1. Two kinds of command

**Goal:** when someone asks how the library is doing, the agent should list the branches,
then count the overdue loans and say how many there are. The library already has a command
for each step. This page writes the job that runs them in order, and shows the two kinds of
command side by side.

## An execution command

`count_overdue` is an execution command: it runs code.

<!-- file: fastworkflow/examples/library/_commands/Library/count_overdue.py -->
```python
"""count_overdue: how many loans are overdue, in one branch or in all."""
import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library


class Signature:
    """Count the overdue loans, in one branch or across the library."""

    class Input(BaseModel):
        branch: str = Field(default="", description="one branch's name; empty for every branch")

    class Output(BaseModel):
        total: int = Field(description="how many loans are overdue")

    plain_utterances = ["how many loans are overdue", "count the overdue loans at Central"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        library: Library = workflow.command_context_for_response_generation
        return Signature.Output(total=library.overdue_count(command_parameters.branch))
```

- **It lives in `_commands/Library/`**, so it belongs to the `Library` context: it runs on
  the library object, which `workflow.command_context_for_response_generation` hands it.
- **`class Signature`** says what it does (the docstring), what it takes (`Input`), what it
  returns (`Output`) and how people ask for it (`plain_utterances`).
- **`class ResponseGenerator`** makes it an execution command. Every `__call__` takes
  `(workflow, command, command_parameters)`: `command_parameters` is the filled `Input`, or
  `None` for a command that has no `Input` (its annotation then says `None`). Adding an
  `Input` later changes nothing else.
- **`__call__` returns the command's `Output`.** The framework does the rest: it writes the
  text the agent reads (the `Output` as JSON), sets `success`, and builds the result that
  callers and API clients receive. A command that has no `Output` returns its text, a `str`.
  To write the text yourself, or to send the client extra data (`artifacts`), return a
  `CommandResponse(response=..., output=...)`. To decline, raise
  `fastworkflow.Refused(message)` (page 5 has one).

## A planning command

`overdue_summary` is a planning command: it writes a plan, and the agent runs it.

<!-- file: fastworkflow/examples/library/_commands/Library/overdue_summary.py -->
```python
"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and no inputs: branches lists the branches; overdue counts the overdue loans
across the library, and its agent writes the answer.
"""
from fastworkflow.planning import run


class Signature:
    """List the library's branches, then say how many loans are overdue across the library."""

    plain_utterances = ["give me the library overview", "how is the library doing"]


class PlanGenerator:
    def branches(self):
        return [run("list_branches")]

    def overdue(self):
        return [run("count_overdue"),
                "say how many loans are overdue across the library"]
```

## Line by line

- **The module docstring** says the job in a few sentences, for the next person who reads
  the file. The framework does not read it.
- **It lives beside `count_overdue`**, in `_commands/Library/`. A planning command belongs
  to a context exactly as an execution command does: it is found while the workflow stands
  on the library, or on a record inside it (page 14). The qualified name is the file's path:
  `Library/overdue_summary`.
- **`class Signature`** is written exactly as an execution command's. Its **docstring is
  the description** the agent reads in its list of commands, as it plans and as it runs,
  and the one a request is matched by; a `Signature` without a docstring fails the load. Write it for a
  person: what the command does, starting with a verb. `plain_utterances` are example
  requests, written the way people ask.
- **No `Input`.** This command takes nothing from the request. Leaving `Input` out and
  writing an empty `class Input(BaseModel): pass` mean the same.
- **`class PlanGenerator`** makes it a planning command. The class name is the whole switch:
  a file defines exactly one of `ResponseGenerator` and `PlanGenerator`, and both in one
  file fails the load.
- **Each public method is a phase**, run in the order written: `branches`, then `overdue`.
  A method whose name starts with `_` is a helper, not a phase. A dunder is a hook of the
  framework: `__result__` returns the call's result in code, its answer, its `Output` or
  both, as `__call__` does for an execution command (pages 3 and 11), and `__phases__`
  chooses the phases (page 6). `__call__` belongs to `ResponseGenerator` alone.
- **A phase is one executor run** with its own step budget: the agent of `branches` is told
  to do only its steps, as part 1 of 2 of a larger job. Short, separate runs keep a long job
  from drifting.
- **Each method returns its steps**, one per line of the plan, exactly the words the agent
  will read. A method builds text only: it never calls the app or reads state. With no
  inputs, the methods take only `self`.
- **`run("list_branches")`** writes a step that names a command, which code runs itself. It
  comes from `fastworkflow.planning`, the module planning commands import their helpers from.
  Every step code acts on is written with a helper: `run` for a command, and page 7 has the
  rest. Every command a step names is checked against the workflow at load, and a typo
  keeps the planning command from loading (page 15).
- **A plain string is the agent's.** `"say how many ..."` names no command, so the agent
  does it: here, writing the answer.

## What the planner plans from

Before the agent runs a request, the planner writes a plan for it (page 14). It writes the
plan from its list of commands, which shows this command's *template*, every phase's steps
joined, beside its `Signature`. With `JEV_API_KEY` set, a request the command covers is a
typed call of it, and one it covers in part is that call with other steps. Without it there
is no match, and the planner plans from the template, adapting its steps when the request
needs more.

<!-- template: overdue_summary -->
```text
list_branches
count_overdue
say how many loans are overdue across the library
```

`fastworkflow command preview` renders one markdown page per planning command:

```sh
fastworkflow command preview fastworkflow/examples/library --out /tmp/library_previews
```

Without `--out` it writes next to the workflow, in `<workflow>/../docs/planning_command_previews`;
add `--check` in CI to fail when a committed preview is stale. This command's preview, below
its title:

```markdown
List the library's branches, then say how many loans are overdue across the library.

Source: `_commands/Library/overdue_summary.py`.

Calls: `Library/count_overdue`, `Library/list_branches`.
Depth: 1 (lower bound; open steps name no command)

## Inputs

None: every call runs the same plan.

## Example requests

- give me the library overview
- how is the library doing

## Phases

| phase | output | runs | fans out |
|---|---|---|---|
| `branches` | none | always | no |
| `overdue` | none | always | no |

## Facts

What each phase's output holds, filled in code from the commands the phase ran.

No phase passes facts on.

## Plan

One for every call: the planning command takes no inputs.

### Phase 1: `branches`

Reads no earlier facts. Output: none.

1. `list_branches`

### Phase 2: `overdue`

Reads no earlier facts. Output: none.

1. `count_overdue`
2. `say how many loans are overdue across the library`
```

`Calls:` lists the commands its steps name, and `Depth:` is the longest chain of planning
commands from it, itself counted, with an execution command counting 0. A step written in open
words (`say who wrote it`) names no command, so the depth is only a lower bound. The preview is
the one place the depth is shown; nothing in the runtime depends on it.

## At run time

When a request is one this command covers, and a `JEV_API_KEY` makes the call typed (page 14),
the plan for it (or for the part of it the command covers) is a call of it, and the agent runs
its steps. With no inputs there is nothing to fill, so every call runs the steps above:

```text
Phase branches
  1. list_branches
  code runs: list_branches
Phase overdue
  1. count_overdue
  2. say how many loans are overdue across the library
  code runs: count_overdue
  the agent writes the answer from these runs
```

Each step `run` wrote, with every required value given, is one **code runs** itself, before
the agent takes a step. The answer is the **last phase's**: its agent writes it from that
phase's runs. Nothing else passes from one phase to the next: the branches the first phase
listed are not in the answer. To say them too, the first phase must hand them on as a
**fact**: page 3.

This trace is what the tutorial's dry run prints for the call: it runs every step code can
run, against the sample commands, with no model, and stops where the agent would take over.

Planning commands run only in agent mode. In assistant mode (`fastworkflow run --assistant`,
or a message typed with a leading `/`), naming one gets `Library/overdue_summary is a
planning command; it runs in agent mode.` Execution commands run in both modes.

## Which kind to write

Write an execution command when code can decide every step, and a planning command when the
agent should run, judge or recover part of the job.

`overdue_summary` is a planning command only because its last step is the agent's: it words
the answer. Most of the planning commands on the next pages need no agent at all once their
arguments are known: code runs every step, and `__result__` writes the answer. They are
planning commands here so that each page can show one idea of the kind. In your own
workflow, a job like that is shorter as an execution command that calls your app (page 3
shows one); it runs in assistant mode and through direct API and MCP calls as well, where a
planning command is refused (page 14), and it returns its `Output` the same way.

A planning command earns its place when the agent has work in it: a value to find, a
judgement to make, a step to recover when it goes wrong, or a question for the user. On
these pages that is a row no code can pick (page 4), a step only the agent can decide
(page 7), and a missing argument (page 9).

## What you learned

A command is an execution command (`ResponseGenerator`: your code runs and returns its
`Output`) or a planning command (`PlanGenerator`: the agent runs its plan), in the same
`_commands/<Context>/` tree, with the same `Signature`. Each public method of a
`PlanGenerator` is a phase, one executor run, in the order written; it returns its steps. A
step code runs is written with `run`; a plain string is the agent's. Write a planning command
when the agent should run, judge or recover part of the job; otherwise, an execution command.
