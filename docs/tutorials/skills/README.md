# Writing commands at every granularity: a tutorial

> **Status.** This tutorial describes fastWorkflow once
> [unified_command_architecture](../../unified_command_architecture/) is implemented,
> seventh amendment included: one `_commands/` tree, `from_`, `__result__`, check options on
> any command's `Input` fields, nested planning commands, scoped lookup, one path for a
> command's arguments, commands that return their `Output`, facts typed by the commands that
> produce them, steps written with helpers, an agent that plans each request as the first
> step of its run, adapting a planning command to it, with code checking the plan, and
> `fastworkflow command run-code`. Its sample workflow,
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
15. [Troubleshooting](15-troubleshooting.md): `run-code`, the load errors and warnings, what
    they mean, how to fix them.

## Who reads a planning command

A planning command is read by code and by the agent, and the agent reads it at two moments.
The pages say which reading each line is for:

- **the agent, as it plans** (a model). Every part of a request gets a run of the agent, and
  the run's first step is a plan the agent writes, from its list of commands. On that list it
  reads the command's `Signature` and its *template*: the steps written with every value
  unknown. For a request the command covers, its plan is the one step that calls it; for one
  it covers in part, the call with steps before or after it; and when the request needs some
  of its steps dropped or changed, an adapted copy of them. Code checks the plan against what
  the workflow declares, and the agent reads the notes (page 14). The command's arguments are
  then filled the way any command's are, by the parameter extractor, from the step that calls
  it (page 9).
- **code** (the framework, no model) runs every step a helper wrote that it can run
  mechanically, fills each phase's facts from the commands' outputs, and runs `__result__`.
- **the agent, as it runs** the command's phases: it does what code cannot, a plain string, a
  judgement, an answer that needs writing.

An execution command has one reader at run time, code: it is your Python.

## Running these yourself

From the repository root:

```sh
fastworkflow command preview fastworkflow/examples/library --out /tmp/library_previews
fastworkflow command run-code fastworkflow/examples/library Library/book_by_title --input '{"title": "Dune"}'
fastworkflow train fastworkflow/examples/library
fastworkflow run fastworkflow/examples/library
```

`command preview` renders one markdown page per planning command, which is the quickest way
to see what the agent will plan from; add `--check` in CI to fail when a committed preview is
stale or a `from_` does not resolve. `command run-code` runs one call with no model and
prints what code does, phase by phase. Planning commands run only in agent mode, the default
of `fastworkflow run`: `--assistant` runs one execution command per message and refuses a
planning command (page 14). `JEV_API_KEY` in `fastworkflow.passwords.env` is optional: without
it, the agent finds the planning command that fits a request itself, as it plans, and the
`verify` option checks nothing (pages 9 and 14).

The *runs* shown on these pages are what `fastworkflow command run-code` prints for a call:
each phase's steps as the runtime writes them, the steps code runs against the sample
commands, the record the workflow moved to (`now in:`), the facts filled from those runs, and
the answer `__result__` writes. A phase's facts are printed as the model that holds them:
`find_book.Output(...)` is `find_book`'s `Output`, handed on as it is. Where code would hand
over to the agent, the run says so and stops.

The reference for everything here: [unified_command_architecture](../../unified_command_architecture/)
(the design), [skill_file_layout.md](../../skill_file_layout.md) (the one-page guide) and
[skill_phases.md](../../skill_phases.md) (phases at run time).
