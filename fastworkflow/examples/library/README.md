# library: the sample workflow of the planning-command tutorial

A small library (two branches, four books, three members and their loans) whose commands are
the worked examples of [docs/tutorials/planning_commands](../../../docs/tutorials/planning_commands/README.md).

> **Status.** Written for the release that implements the unified command architecture,
> sixth amendment included: planning commands in `_commands/`, `from_`, `__result__`, check
> options such as `in_request` on any command's `Input` fields, nested planning commands,
> scoped lookup, commands that return their `Output`, facts typed by the commands that
> produce them, steps written with helpers, and `pick` reading the listing wherever the
> call read it. It does not load on earlier releases, which have no `fastworkflow.from_`
> and no `fastworkflow.planning`.

## Layout

```
library/
  startup_action.json            # runs the startup command
  application/library.py         # the in-memory library the commands read
  _commands/
    context_inheritance_model.json   # Book and Member have the base Record
    startup.py                   # makes a Library the root object
    library_hours.py             # global: found from every context
    Library/                     # the library, the root object
      _Library.py                # the context's class
      _wording.py                # wording the planning commands share; never a command
      list_branches.py  count_overdue.py  find_book.py  open_book.py
      find_member.py    open_member.py
      overdue_summary.py  note_member.py  count_books.py  book_by_title.py
      member_report.py    author_books.py find_record.py  book_loans.py
    Book/                        # one book; its parent is the library
      _Book.py  show_book.py  list_loans.py  loan_report.py  shelf_report.py
    Member/                      # one member; its parent is the library
      _Member.py  show_member.py  list_member_loans.py
    Record/                      # a base of Book and Member; never an object
      _Record.py  show_notes.py  add_note.py
```

In each context folder, a file with a `ResponseGenerator` is an execution command and a file
with a `PlanGenerator` is a planning command. Every execution command returns its
`Signature.Output`, except `startup`, which has none and returns text.

| planning command | tutorial page | what it shows |
|---|---|---|
| `Library/overdue_summary` | 1 | two phases, no inputs, steps written with `run` |
| `Library/note_member` | 2 | inputs, `in_request`, `pick`, a call with one phase |
| `Library/count_books` | 3 | a phase that hands on a command's `Output`, an answer `__result__` writes |
| `Library/book_by_title` | 4, 11 | skipping, `pick` from an earlier phase's listing, a `_` helper, a typed `Output` that `__result__` returns with the answer |
| `Library/member_report` | 6, 9 | `__phases__`, `full_list`, refusing a call |
| `Library/author_books` | 8 | a fan-out, bound to the records it opens |
| `Library/find_record` | 10 | a union of `Output`s: whichever search ran |
| `Book/shelf_report` | 10 | a facts model with one field per command |
| `Library/book_loans` | 12 | calling a planning command as a step |
| `Book/loan_report` | 14 | a planning command of a context |

Several of these cover similar requests, because each one teaches one idea. A real workflow
would keep the most general of them. Most of them also need no agent once their arguments
are known: code runs every step and `__result__` writes the answer. In a real workflow such
a job is shorter as an execution command; a planning command is for a job the agent should
run, judge or recover part of (page 1 of the tutorial).

## Running it

```sh
fastworkflow examples fetch library
fastworkflow train ./examples/library
fastworkflow run ./examples/library
```

Then try "who wrote Dune?", "which branch holds it?", "who has this book?", "go back", "what
has Raj Patel borrowed, and any notes on his record?" and "who has borrowed Emma?".
Planning commands run in agent mode, the default; `--assistant` runs execution commands only.

The planner plans every request, and sees each planning command with its template: it calls
the command for a request the command covers, calls it with other steps for one it covers in
part, or adapts its steps. `JEV_API_KEY`, in `fastworkflow.passwords.env`, is optional: with
it, the Jev decision model tells the planner which planning command fits first, and sends the
request and the planning commands' descriptions to TypeSafe.

The preview of these planning commands, which shows the template the planner sees for each one,
comes with a later release. `fastworkflow command preview` still reads planning commands from
`_skills/`, and this library has none.
