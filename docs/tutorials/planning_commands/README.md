# Writing commands at every granularity: a tutorial

> **Status.** This tutorial describes planning commands as fastWorkflow implements them: one
> `_commands/` tree, `from_`, `__result__`, check options on any command's `Input` fields,
> nested planning commands, scoped lookup, commands that return their `Output`, facts typed by
> the commands that produce them, steps written with helpers, adapting a planning command to
> a request, and code checking the plan. Its sample workflow,
> [`fastworkflow/examples/library`](../../../fastworkflow/examples/library), is written for
> that release and does not load on earlier ones.

A fastWorkflow app is a set of **commands**, and a command is one of two kinds:

- an **execution command** runs your code. Its file has a `ResponseGenerator`;
- a **planning command** is a job the agent runs from a plan: the same commands, in the same
  order, with code running every step it can and the agent the rest. Its file has a
  `PlanGenerator`, which writes the plan.

Write an execution command when code can decide every step, and a planning command when the
agent should run, judge or recover part of the job (page 1 says more).

Both kinds live in the same folder tree, `_commands/<Context>/<name>.py`, and both start with
the same `Signature`: a docstring that says what the command does, an `Input` it takes and an
`Output` it returns. The class name is the only switch between them. Everything else on these
pages works the same for both kinds: where a command lives, how a request finds it, how its
arguments are filled, checked (the options of its `Input` fields) and asked for when one is
missing, how a field names another command's `Output` field (`from_`), what a command returns
(a `str`, its `Output`, or both), and that every command leaves the workflow where it ends.

These pages build planning commands step by step, one new idea per page, on a small library
workflow. Every file shown on a page is a real file of the sample workflow, quoted whole.

## The sample workflow

[`fastworkflow/examples/library`](../../../fastworkflow/examples/library) is a library with
two branches, four books (Dune, Dune Messiah, Emma, Persuasion), three members (Ann Lee, Raj
Patel, Mia Chen) and their loans. Fetch your own copy with
`fastworkflow examples fetch library`.

Its contexts, and what each one's object is:

| context | the object | its parent (`get_parent()`) | its bases |
|---|---|---|---|
| global (`_commands/` root) | none: these commands run from anywhere | | |
| `Library` | the library, the workflow's root object | none | |
| `Book` | one book | the library | `Record` |
| `Member` | one member | the library | `Record` |
| `Record` | never an object: the commands every record has | | |

The workflow starts on the library (the `startup` command makes a `Library` the root
object). `open_book` and `open_member` move it onto a record, and `go_up` moves it back.

Its execution commands, each of which returns its `Output`:

| context | command | takes | its `Output` |
|---|---|---|---|
| global | `library_hours` | `branch` (optional) | `branches`, `hours` |
| `Library` | `list_branches` | nothing | `names`, `total` |
| `Library` | `count_overdue` | `branch` (optional) | `total` |
| `Library` | `find_book` | `query`, `start` (a page) | `book_uids`, `titles`, `total` |
| `Library` | `open_book` | `book_uid` | `book_uid`, `title`, `author`, `copies` |
| `Library` | `find_member` | `query` | `member_uids`, `names`, `total` |
| `Library` | `open_member` | `member_uid` | `member_uid`, `name`, `on_loan` |
| `Book` | `show_book` | nothing | `book_uid`, `title`, `author`, `branch`, `copies`, `on_loan` |
| `Book` | `list_loans` | `start` | `borrowers`, `due_dates`, `total` |
| `Member` | `show_member` | nothing | `member_uid`, `name`, `on_loan` |
| `Member` | `list_member_loans` | `start` | `titles`, `due_dates`, `total` |
| `Record` | `show_notes` | nothing | `notes`, `total` |
| `Record` | `add_note` | `text` | `total` |

The planning commands these pages write are in the same tree, beside the execution commands
of their context: eight in `_commands/Library/` and two in `_commands/Book/`.

## The pages

1. [Two kinds of command](01-two-kinds-of-command.md): `ResponseGenerator` returning its
   `Output`, and `PlanGenerator`, each method a phase, steps written with `run`, the
   preview, a run, and which kind to write.
2. [An input](02-an-input.md): `Input` fields, how their values are filled, the
   `in_request` option, a call with one phase.
3. [A fact passed on, and an answer written in code](03-a-fact.md): a phase that hands on a
   command's `Output`, reading a phase by its name, `__result__`.
4. [Skipping a phase, picking a row](04-skipping-and-picking.md): `return []`, a skipped
   phase as `None`, `pick(...)` from an earlier phase's listing, a `_` helper.
5. [Where a command leaves the workflow](05-where-a-command-leaves-the-workflow.md): a
   navigation command and its `descends_to`, a producer `from_`, the record a call ends on.
6. [The shape of a call](06-the-shape-of-a-call.md): `__phases__`.
7. [Words the runtime reads](07-answer-wording.md): the step helpers (`run`, `full_list`,
   `pick`, `when`, `do_not_run`), notes, paging, and the answer helpers (`listed`, `count`,
   `num`).
8. [Once per item](08-fan-out.md): `@phase(over=...)`.
9. [Refusing a call](09-refusing-a-call.md): the check options, `validate_extracted_parameters`,
   `fastworkflow.Refused`, and a missing argument.
10. [Facts from more than one command](10-a-facts-model-per-input.md): a union of `Output`s,
    and a facts model with one field per command.
11. [A typed Output](11-a-typed-output.md): `Signature.Output`, returned by `__result__` with
    the answer, and who reads it.
12. [Calling a planning command](12-calling-a-planning-command.md): a step that names a
    planning command, the cycle guard, the step budget.
13. [Reuse without a call](13-reuse.md): `_` helpers, a base class in a `_` module, and when
    to call instead.
14. [Where a request finds its command](14-where-a-request-finds-its-command.md): contexts,
    the lookup path, how the agent plans a request and calls, extends or adapts a planning
    command, the checks on a plan, what the agent is shown, assistant mode.
15. [Troubleshooting](15-troubleshooting.md): the load errors and warnings, what they mean,
    how to fix them.

## Who reads a planning command

A planning command is read by code, by the planner and by the agent. The pages say which
reading each line is for:

- **the planner** (a model). Every part of a request gets a plan, which the planner writes
  before the agent runs it. With `JEV_API_KEY` set, a command that fits the part is a typed
  call: the planner fills its `Input`, code checks the values and writes its steps, and the
  planner adds steps before or after it for what the request asks beyond it. Otherwise the
  planner plans from its list of commands, where it reads the command's `Signature` and its
  *template*, the steps written with every value unknown, and calls the command or adapts a
  copy of its steps (page 14). A step that calls the command has its arguments filled the way
  any command's are, by the parameter extractor (page 9).
- **code** (the framework, no model) runs every step a helper wrote that it can run
  mechanically, fills each phase's facts from the commands' outputs, and runs `__result__`.
- **the agent, as it runs** the command's phases: it does what code cannot, a plain string, a
  judgement, an answer that needs writing.

An execution command has one reader at run time, code: it is your Python.

## Running these yourself

From the repository root:

```sh
fastworkflow command preview fastworkflow/examples/library --out /tmp/library_previews
fastworkflow command run-code fastworkflow/examples/library book_loans --input '{"title": "Dune"}'
fastworkflow train fastworkflow/examples/library
fastworkflow run fastworkflow/examples/library
```

`command preview` renders one markdown page per planning command, which is the quickest way
to see what the agent will plan from; add `--check` in CI to fail when a committed preview is
stale or a `from_` does not resolve. Planning commands run only in agent mode, the default
of `fastworkflow run`: `--assistant` runs deterministic mode and refuses a planning command
(page 14). `JEV_API_KEY` in `fastworkflow.passwords.env` is optional: without it no request
matches a planning command, so the planner plans from each one's template, and the `verify`
option checks nothing (pages 9 and 14).

The *runs* shown on these pages are what `fastworkflow command run-code` prints for the call,
run with no model: the same text the tutorial's test gets from `fastworkflow.planning.run_code`
(`tests/test_planning_command_tutorials.py`), which checks the pages against the code. A run shows
each phase's steps as the runtime writes them, the steps code runs against the sample commands
(on a real session of the sample workflow, so their effects are real: run it on a copy of the
workflow, not on your app's data), each move onto a record (`now in:`), the facts filled from
those runs, and where code hands over to the agent. A phase's facts are printed as the model
that holds them: `Output(...)` is the phase's command `Output`, handed on as it is. A step that
names another planning command runs that call indented. `run-code` exits 0 when the run completes
or stops at the first step code cannot run, and 1 when the command does not load or a step or a
hook fails. `--start-with 'open_book book_uid=b1'` puts the workflow on a record first, for a
command of that record's context (`shelf_report` is one).

The reference for everything here: [planning_command_file_layout.md](../../planning_command_file_layout.md)
(the one-page guide) and [planning_command_phases.md](../../planning_command_phases.md)
(phases at run time).
