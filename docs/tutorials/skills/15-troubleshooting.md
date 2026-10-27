# 15. Troubleshooting

A command file that does not load is skipped, and the workflow's other commands still load.
`fastworkflow run` prints the error before its first prompt, `fastworkflow train` stops on
it, and `fastworkflow command preview --check` fails naming the file and its error. Each
snippet below is a file saved in a copy of the sample workflow, at the path its message
names.

A call that loads but does not do what you meant is easiest to see with no model in the
way. `fastworkflow command run-code` runs one call, prints every step code runs, the facts
each phase hands on and what the hooks return, and stops where the agent would take over:

```sh
fastworkflow command run-code fastworkflow/examples/library Library/book_by_title --input '{"title": "Dune"}'
fastworkflow command run-code fastworkflow/examples/library Book/shelf_report --start-with "open_book book_uid=b1"
```

`--start-with` steps run first, to put the workflow where the command starts. It runs the
workflow's real execution commands, so use it on a development instance of your app.

## Where commands live

**Both classes in one file.** A file defines exactly one of `ResponseGenerator` and
`PlanGenerator`:

```text
_commands/Library/find_or_open.py: defines both ResponseGenerator and PlanGenerator; a command file defines one
```

**Fix:** split it into an execution command and a planning command that calls it.

**The same qualified name twice.** A planning command and an execution command cannot share
`<Context>/<name>`; a planning command named like a command of another context is fine.

**A `_skills/` folder.** It is no longer read, and a workflow that has one fails to load
with:

```text
_skills/ is no longer read; move these files into _commands/ (first make sure any script that regenerates _commands/ preserves them; see docs/unified_command_architecture/ 3.8)
```

**Fix:** move each root file to the `_commands/` root and each context folder into the
matching `_commands/<Context>/`, `_` helpers with the files that import them. A file that
moves one folder deeper needs one more dot in its relative imports (`from ._facts` becomes
`from .._facts`). If a script of yours regenerates `_commands/`, first make it keep every
file that defines a top-level `PlanGenerator`, and the `_` files they import.
`fastworkflow build` already never overwrites one.

**`__call__` on a `PlanGenerator`.** The phases a call runs are `__phases__`; `__call__`
belongs to `ResponseGenerator`:

```text
_commands/Library/member_report.py: PlanGenerator defines __call__, which is not read; the phases a call runs are __phases__(self, input)
```

**Fix:** rename the method `__phases__`.

**`enter_command` on a context class.** How a context is entered is declared once, on the
command that enters it:

```text
_commands/Book/_Book.py: Context.enter_command is no longer read; declare descends_to = "Book" on open_book's ResponseGenerator
```

**Fix:** move it to the command as `descends_to` (page 5), and delete it from the context
class.

## Outputs

**A paged command without `total`.** A command whose `Input` has an integer `start`,
`offset` or `skip` is paged, and the runtime reads its pages from its `Output` (page 7):

```text
_commands/Book/list_loans.py: Input.start makes list_loans a paged command, but its Output declares no total; declare total: int, how many rows there are on every page, or rename start if it is not a page
```

## Checks on the arguments

**A check option on a type it does not apply to.** Each option of an `Input` field applies to
some types only (page 9):

```python
class Input(BaseModel):
    loans: bool = Field(default=False, description="whether to name the books on loan",
                        json_schema_extra={"in_request": True})
```

```text
_commands/Library/member_report.py: Input.loans: in_request applies to a str or an int, and loans is a bool; to have the request decide a bool, write verify
```

**Fix:** drop the option, or use one that applies: `verify` decides a `bool` again from the
request (with `JEV_API_KEY`).

## `from_`

**A hand-written `available_from`.** It was the producer hint before `from_`:

```python
order_id: str = Field(json_schema_extra={"available_from": ["find_order"]})
```

```text
_commands/Library/get_order.py: Input.order_id: available_from is written by from_; write from_("find_order") or from_("find_order.<field>")
```

**Fix:** `order_id: str = from_("find_order.order_ids")`. Prose that used to sit in the hint
("in the current context") belongs in the field's description.

**A `from_` anywhere but an `Input` field.** An `Output` is built by code, and a facts
model's fields are typed with the commands they come from (page 10), so neither carries one:

```text
_commands/Library/book_by_title.py: Signature.Output.author: an Output field cannot declare from_; build the value in __result__
```

```text
_commands/Book/shelf_report.py: Shelf.branch (from 'show_book.branch'): from_ is only for a command's Input field; type the field with the command's Output instead
```

**Fix:** for a facts model, `book: ShowBook.Output`, and read `shelf.book.branch`.

**A `from_` on an `Input` field that names an `Input` field.** `from_` names the `Output`
field a value comes from. It does not borrow another command's description:

```python
query: str = from_("find_book.query")       # in _commands/Library/count_books.py
```

```text
_commands/Library/count_books.py: Input.query (from 'find_book.query'): find_book's query is an Input field; from_ names the Output field a value comes from. To describe the value, write Field(description=...); the preview shows find_book.query's description beside it
```

**An unqualified name off the path.** An `Input` field's unqualified name is looked up in the
declaring context, its bases, then the global commands. A producer in a context that encloses
this one is qualified, since which records enclose a record is known only at run time:

```python
member_uid: str = from_("find_member.member_uids")   # in _commands/Book/reserve_for.py
```

```text
_commands/Book/reserve_for.py: Input.member_uid (from 'find_member.member_uids'): no command find_member in Book, its bases or the global commands; qualify it (Library/find_member)
```

**Fix:** `from_("Library/find_member.member_uids")`. When the name matches commands in two
scopes the message lists both, qualified; pick one.

**A type that does not match.** A producer's type must equal its target's, `Optional` aside,
or be one item of a list target:

```text
_commands/Library/open_book.py: Input.book_uid (from 'find_book.book_uids'): int does not match find_book's Output.book_uids: list[str]
```

**A field the target does not have.**

```text
_commands/Library/open_book.py: Input.book_uid (from 'find_book.uids'): Library/find_book has no Output field 'uids'
```

**A producer hint to a command you call.** A producer hint tells the agent to run its target
before your command. If your own steps run that target, the hint is backwards:

```text
_commands/Library/book_loans.py: Input.title (from 'book_by_title.title') tells the agent to run book_by_title first, but this command's own steps call book_by_title. To describe the value, write Field(description=...) instead
```

## Facts

**A phase that hands on an `Output` its steps never run.** A phase annotated with a command's
`Output`, or with a facts model whose fields are, must run that command:

```text
_commands/Library/count_books.py: phase 'find' hands on find_book's Output, but none of its steps runs find_book
```

**A facts field of another type.** Every field of a facts model is typed with a command's
`Output` or `Input`:

```python
class Shelf(BaseModel):
    book: ShowBook.Output
    borrowers: list[str]
```

```text
_commands/Book/shelf_report.py: Shelf.borrowers is list[str]; a facts field is typed with a command's Output or Input. Compute other values in a property or method of Shelf
```

**Fix:** `loans: ListLoans.Output`, and, to keep the name, a property that returns
`self.loans.borrowers`.

## The answer and the `Output`

**An answer written as `say that` lines.** Code no longer copies them into the answer:

```python
def answer(self, input, book):
    return [f"say that {book.title} is by {book.author}"]
```

```text
_commands/Library/book_by_title.py: phase 'answer' writes the answer as 'say that' lines, which code no longer copies; return the text from __result__(self, input, ...) instead
```

**Fix:** rename the method `__result__`, and return the text:
`return f"{book.title} is by {book.author}."`.

**A hook parameter that names no phase.** `__result__` reads `input` and the phases by name:

```python
def __result__(self, input, found):
    ...
```

```text
_commands/Library/book_by_title.py: __result__ takes 'found', which is not a phase of this command
```

**A `Signature.Output` that nothing builds.** A planning command that declares an `Output`
returns it from `__result__` (page 11):

```text
_commands/Library/book_by_title.py: Signature.Output is declared, but nothing builds it; return it from __result__(self, input, ...)
```

**A `PlanGenerator` that subclasses another command's.** Reuse its whole job by calling it,
or share phases through a base class in a `_` module (page 13):

```text
_commands/Library/book_and_loans.py: PlanGenerator inherits from book_by_title's PlanGenerator; to reuse its whole job, call book_by_title in a step; to share phases, move them to a base class in a _ module
```

## Steps

**A step naming an unknown command or parameter.** Every command step of every input
combination is checked against the workflow's commands and each command's `Input`:

```python
def search(self, input):
    return [run("find_books", query=input.title),
            run("find_book", title=input.title)]
```

```text
_commands/Library/bad_command.py: a step runs 'find_books', which is not a command of the workflow
_commands/Library/bad_command.py: a step passes title to 'find_book', whose Signature.Input has no such field
```

**Fix:** use the command's name and its `Input` fields. Text for the agent goes in a note,
`run("find_book", query=input.title).note("the first page")`, which is never sent.

**A note that holds a condition.** Code never reads a note, so a condition written in one does
not stop the step:

```python
def note(self, input):
    return [run("add_note", text=input.text).note("only when the member asked")]
```

```text
_commands/Library/bad_note.py: phase 'note', step 'add_note text=<text> (only when the member asked)': the note holds a condition, but code never reads a note, so add_note would run every time. Decide it in code with if, or write when("the member asked", "add_note text=<text>") for the agent to decide
```

**A step written by hand.** It loads and runs, and `preview --check` warns without failing:

```python
def find(self, input) -> FindBook.Output:
    return [f"find_book query={input.title}"]
```

```text
_commands/Library/count_books.py: phase 'find', step 'find_book query=<title>': written by hand; write run(...) so that code reads it as you mean
```

**Fix:** `run("find_book", query=input.title)`. A value from the request can hold what the
step parser reads otherwise: written by hand, `add_note text=prefers email (not phone)` adds
the note "prefers email". The same warning names `full_list`, `when`, `do_not_run` or `pick`
for their words typed by hand.

**`pick` with the wrong parameter.** `pick` takes a `named=` name, and at most one
`..._uid=` parameter, for a uid your own rule chose:

```text
_commands/Library/bad_pick.py: phase 'book' fails while its template is built (every fact unknown): StepFormError: pick('open_book'): named= is the name the row is picked by
```

**A phase decided at run time.** While the template is built every fact is a placeholder,
which stands in for most of what code does with a value: attributes, indexes, tests,
iteration and arithmetic all work (`<find.total - 1>`). A phase that does something a
placeholder cannot, such as looking a fact up in a table, still loads and runs on the real
facts; the preview, and the agent reading the template as it plans, show it as decided at run time,
`preview --check` warns, and its steps go unchecked at load. A `__result__` that builds an
`Output` from the facts is shown the same way (page 4):

```python
def notes(self, book) -> ShowNotes.Output:
    floor = FLOORS[book.branch]            # FLOORS = {"Central": "the first floor", ...}
    return [run("show_notes").note(f"the book is on {floor}")]
```

```text
_commands/Book/bad_lookup.py: phase 'notes' is decided at run time: while the template is built (every fact unknown) it raised KeyError: '<book.branch>'
```

**`None` written into a step**, for a value of an enum with no entry in a table:

```text
_commands/Library/bad_none.py: phase 'find' writes None into a step ('None query=<name>'): a value it formats is None for some input
```

**A parameter that names no earlier phase:**

```text
_commands/Library/bad_binding.py: plan(): phase 'loans' takes 'found', which is not an earlier phase of this command
```

**A subclass that adds a phase without saying where** (page 13):

```text
_commands/Library/bad_order.py: PlanGenerator adds the phase(s) overdue to those of bad_order.Branches: write __phases__(self, input) returning the phases a call runs, in order (from: self.branches, self.overdue)
```

## Calls between planning commands

**A cycle.** If `a`'s steps name `b` and `b`'s name `a`, both fail the load, and the preview
names the cycle. A step written in open words can still name a running command at run
time; it is refused with `a is already running; call its steps directly.`

**The step budget.** `The step budget for nested commands is spent; continue with the
remaining steps directly.` means the nested calls of one turn used `FW_TURN_MAX_STEPS`
model steps (400 by default). Raise it in `fastworkflow.env` if your heaviest planning
command needs more; the steps of a planning command that is the whole plan for a request do
not count, the calls it makes do, and so do the steps of one the agent's plan called with
other steps (page 14).

## At run time, with no error

These are warnings, and one note, each logged once per command (or once per workflow) per
process.

**A command returned no `Output`.** `book_by_title declares an Output but returned none;
phases and from_ references cannot read it` means a command that declares a
`Signature.Output` returned only text. Return the `Output` on every path: an execution
command returns it from `__call__` (page 1), and a planning command from `__result__`, saying
"not found" in a field when it found nothing (page 11).

**A hook that raised.** A `__result__` that raises, or returns something other than a `str`,
its command's `Output` or a `CommandResponse`, keeps the call: a model writes the answer from
the phases' runs, `output` is `None`, and the warning names the command. Run the call with
`fastworkflow command run-code` to see the exception.

**`verify` checks nothing.** `library: planning commands are chosen and adapted by the
agent as it plans; set JEV_API_KEY to add Jev's choice and the verify checks of
<fields>.`, logged once at load, means the workflow has no `JEV_API_KEY`: the agent chooses
planning commands by itself as it plans, and the fields the line names carry a `verify`
option that checks nothing without the key (page 9).

**A navigation command without `descends_to`.** `open_shelf moved the workflow into Shelf but
declares no descends_to; declare descends_to = "Shelf" so plans can reach Shelf's commands`
means a command set `workflow.current_command_context` to a record of a context it does not
name. Declare it on the command (page 5).

**A planning command is refused.** `member_report is a planning command; it runs in agent
mode.` means it was named in assistant mode, in a `/`-prefixed message, or by a direct
action or MCP call; run it in agent mode.

**The workflow will not load after an upgrade.** "Retrain with `fastworkflow train`" means its
classifiers were trained before scoped lookup. A warning that `context_hierarchy_model.json`
is ignored means the file can be deleted.

## A fact named like a `str` method

While the template is built, a phase's facts are one placeholder, which is a `str`. A fact
named like a public `str` method (`title`, `count`, `format`, `split`, ...) still reads as the
fact, not the method, so `book.title` is `<book.title>`. Only a name the placeholder means
itself (`index`, the one item's place) keeps that meaning.
