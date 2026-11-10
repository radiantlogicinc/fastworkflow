# 15. Troubleshooting

A command file that does not load is skipped, and the workflow's other commands still load.
`fastworkflow run` prints the error before its first prompt, `fastworkflow train` stops on
it before any training, `run_fastapi_mcp` logs it at startup, and
`fastworkflow command preview --check` fails naming the file and its error. Each
snippet below is a file saved in a copy of the sample workflow, at the path its message
names.

A call that loads but does not do what you meant is easiest to see with no model in the
way: `fastworkflow command run-code <workflow> <command> --input '<json>'` runs one call phase
by phase, as the runtime does, and stops at the first step code cannot run (exit 0). Its output
is the runs the pages show, and a hook that raises exits 1 naming the hook (for example the
`__result__()` of `Library/book_loans`). Use `--start-with` for a command of a record's context
(`shelf_report` in the sample workflow needs a book first: `--start-with 'open_book book_uid=b1'`). The tests check the pages' runs against the code.

## Where commands live

**Both classes in one file.** A file defines exactly one of `ResponseGenerator` and
`PlanGenerator`:

```text
_commands/Library/find_or_open.py: defines both ResponseGenerator and PlanGenerator
```

**Fix:** split it into an execution command and a planning command that calls it.

**The same qualified name twice.** A planning command and an execution command cannot share
`<Context>/<name>`; a planning command named like a command of another context is fine.

**A `_skills/` folder.** It is no longer read, and a workflow that has one fails to load
with:

```text
_skills/ is no longer read; move these files into _commands/ (first make sure any script that regenerates _commands/ preserves them; see docs/migrating_skills_to_commands.md)
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
_commands/Library/member_report.py: PlanGenerator defines __call__; rename it to __phases__
```

**Fix:** rename the method `__phases__`.

**`enter_command` on a context class.** How a context is entered is declared once, on the
command that enters it:

```text
_commands/Book/_Book.py: Context.enter_command is no longer read; declare @descends_to("Book") on open_book's ResponseGenerator
```

**Fix:** move it to the command as `@descends_to` (page 5), and delete it from the context
class.

## Outputs

**A paged command without `total`.** A command whose `Input` has an integer `start`
is paged, and the runtime reads its pages from its `Output` (page 7):

```text
_commands/Book/list_loans.py: Input.start makes list_loans a paged command, but its Output declares no total; declare total: int, how many rows there are on every page, or rename start if it is not a page
```

## Checks on the arguments

**A check option on a type it does not apply to.** `copied_from_command_text` applies to a `str`,
an `int` or a `list[str]` only (page 9); a `bool` takes `classifier_confirms` instead:

```python
class Input(BaseModel):
    loans: bool = Field(default=False, description="whether to name the books on loan",
                        json_schema_extra={"copied_from_command_text": True})
```

```text
_commands/Library/member_report.py: option copied_from_command_text does not apply to field 'loans' of type bool (it takes a str, an int or a list[str])
```

**Fix:** drop the option, or use one that applies: `classifier_confirms` decides a `bool` again from the
command text (with `JEV_API_KEY`).

**A retired option.** The spellings `in_request`, `verify`, `positional`, `from`, `subject`, `means`
and `strip` are gone, and so are the helpers and attributes below. Each fails the load and names its
replacement, and each entry shows the load error the author gets. Write
`record_named_in_command_text: True` for a record name, `classifier_confirms: {"meanings": {value: meaning, ...}}`
for an enum's meanings, and `copied_from_command_text: {"ignore": [...]}` for the words `strip` removed
from a value's ends (page 9).

<!-- broken: retired_subject -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"subject": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_subject.py: field 'title': subject: True is removed: write "record_named_in_command_text": True
```

**`in_request`.** `in_request: True` is `copied_from_command_text: True` (page 2), and the dict form
`{"ignore": [...]}` keeps its words as `copied_from_command_text: {"ignore": [...]}`:

<!-- broken: retired_in_request -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_in_request.py: field 'title': in_request: True is removed: write "copied_from_command_text": True
```

**`in_request: "subject"`** is `record_named_in_command_text: True`. A `subject` beside an `ignore` list
is `record_named_in_command_text: {"ignore": [...]}`:

<!-- broken: retired_in_request_subject -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"in_request": "subject"})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_in_request_subject.py: field 'title': in_request: "subject" is removed: write "record_named_in_command_text": True
```

**`verify`.** `verify: True` is `classifier_confirms: True` on a `bool`, a `Literal`, or a `str` with
`record_named_in_command_text` alongside. `verify: "alone"` is `classifier_decides: True` on a `bool`.
`verify: {value: meaning}` is `classifier_confirms: {"meanings": {value: meaning}}`:

<!-- broken: retired_verify -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        loans: bool = Field(default=False, description="whether to name the loans",
                            json_schema_extra={"verify": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_verify.py: field 'loans': verify: True is removed: write "classifier_confirms": True
```

**`positional`** is `command_text_after_command_name: True`, at most one on an execution command's `Input`:

<!-- broken: retired_positional -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"positional": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_positional.py: field 'title': positional: True is removed: write "command_text_after_command_name": True
```

**`from`**, the producer key, is `from_command_output`, with the same `"cmd.field"` value (page 5):

<!-- broken: retired_from_key -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        book_uid: str = Field(description="the uid", json_schema_extra={"from": "find_book.book_uids"})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_from_key.py: field 'book_uid': from: "find_book.book_uids" is removed: write "from_command_output": "find_book.book_uids"
```

**`from_`**, the helper import. Its producer edge is the `from_command_output` key above. Importing the
helper fails the load with the message that names its replacement:

<!-- broken: retired_from_import -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import from_


class Signature:
    """Show the book."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("find_book", title=from_("find_book.book_uids"))]
```

<!-- load-error -->
```text
ImportError: Could not import module from path: _commands/retired_from_import.py. Error: from_ is removed: write json_schema_extra={"from_command_output": "<command>.<field>"} on the Input field
```

**`validate_extracted_parameters`** is removed. Its checks are a `@model_validator(mode="after")` on the
`Input` (page 9), which reads `info.context` (the command text, the workflow and the command):

<!-- broken: retired_validate_hook -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})

    @staticmethod
    def validate_extracted_parameters(workflow, command, input, command_text=None):
        return None


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_validate_hook.py: validate_extracted_parameters is removed: write @model_validator(mode="after") on Input (read info.context["command_text"], ["workflow"], ["command"])
```

**`approval`** on an execution command's class is a decorator now. A file that sets it fails the load
with the message below. The decorator is `@user_approval_for_execution` (see the
[planning command file layout](../../planning_command_file_layout.md), "Approval annotation"):

```text
CommandDeclarationError: _commands/retired_approval.py: retired_approval: approval = "mandatory" is removed: write @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)
```

**`writes = True`** is `@writes` on the `ResponseGenerator` (execution commands only):

```text
CommandDeclarationError: _commands/retired_writes.py: retired_writes: writes = True is removed: write @writes on the ResponseGenerator (execution commands only)
```

**`descends_to = "Book"`** is `@descends_to("Book")` on the `ResponseGenerator` (page 5), and
**`descend_parameter = "title"`** is `when_parameter` on the same decorator:

```text
CommandDeclarationError: _commands/retired_descends_to.py: retired_descends_to: descends_to = "Book" is removed: write @descends_to("Book") on the ResponseGenerator
CommandDeclarationError: _commands/descend_parameter_only.py: descend_parameter_only: descend_parameter = "title" is removed: write @descends_to("<Context>", when_parameter="title") on the ResponseGenerator
```

**A private attribute written by hand**, such as `__fw_writes__ = True`, fails the load: the decorators
store the mark themselves, so declare the decorator instead:

```text
CommandDeclarationError: _commands/retired_private_mark.py: retired_private_mark: __fw_writes__ is written by hand; declare it with @writes instead
```

**`pick`, `full_list`, `listed` and `num`** are removed helpers. Importing one fails the load with the
message that names its replacement. `pick` is `run(command, named=...)` (page 4):

<!-- broken: retired_pick_import -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import pick


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [pick("list_branches", named=input.title)]
```

<!-- load-error -->
```text
ImportError: Could not import module from path: _commands/retired_pick_import.py. Error: pick is removed: write run(command, named=<name>, <id input>=<value>), e.g. run("open_book", named=input.title)
```

`full_list` is `run(command, ..., all_pages=True)` (page 7):

<!-- broken: retired_full_list_import -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import full_list, run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
ImportError: Could not import module from path: _commands/retired_full_list_import.py. Error: full_list is removed: write run(command, ..., all_pages=True)
```

`listed` is `count_and_names_read(names, total, noun)` (page 7):

<!-- broken: retired_listed_import -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import listed, run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
ImportError: Could not import module from path: _commands/retired_listed_import.py. Error: listed is removed: write count_and_names_read(names, total, noun)
```

`num` is removed: write the number inline (page 7):

<!-- broken: retired_num_import -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import num, run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
ImportError: Could not import module from path: _commands/retired_num_import.py. Error: num is removed: write the number inline, e.g. f"{value}" (see the planning command docs)
```

**`named` and `all_pages` as `Input` field names.** Both are keywords of `run`, so no command can be
passed a field of either name, and the load refuses the field:

<!-- broken: retired_named_field -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        named: str = Field(description="the name", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_named_field.py: Input field 'named' is reserved: run() takes named= as its own keyword, so no command call can pass a field of that name; rename the field
```

The same rule holds for `all_pages`:

<!-- broken: retired_all_pages_field -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        all_pages: bool = Field(default=False, description="every page")


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_all_pages_field.py: Input field 'all_pages' is reserved: run() takes all_pages= as its own keyword, so no command call can pass a field of that name; rename the field
```

**`run` with `named=` and `all_pages=True` together.** `named=` opens one row and `all_pages=True` reads
every page, so a step with both fails while the template is built. Choose one (page 7):

<!-- broken: retired_named_all_pages -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input):
        return [run("list_loans", named=input.title, all_pages=True)]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/retired_named_all_pages.py: steps() fails on placeholder inputs (StepFormError: run('list_loans'): named= opens one row and all_pages=True reads every page; give one of them); declare Signature.template
```

**`named=` on a command with no row id.** A `named=` step needs an id input to open the row with: the
field declared with `from_command_output`, else the command's one required `str` field. A command with
neither fails the load:

```text
_commands/named_no_row.py: a step opens a row of 'list_branches' by named=, but 'list_branches' declares no row id input (a field from_command_output, or its one required str field): pass the id input explicitly
```

**`command_context_object`**, a parameter of a `Context` classmethod, is `record` (page 5):

```python
import fastworkflow


class Context:
    @classmethod
    def get_parent(cls, command_context_object):
        return None
```

```text
CommandContextModelError: context 'Book' does not load: get_parent: parameter "command_context_object" is removed: name it "record"
```

**`instance_label`**, a classmethod of a `Context`, is removed. A computed identity is a property of the
record, named by `instance_label_attr`:

```python
class Context:
    @classmethod
    def instance_label(cls, record):
        return record.uid
```

```text
CommandContextModelError: context 'Book' does not load: instance_label is removed: make the label a property on the record object and name it with instance_label_attr = "<property name>" (e.g. "instance_label")
```

**`command_context_for_response_generation`, `current_command_context` and `root_command_context`** are
gone from the `Workflow`. Reading or assigning one raises an `AttributeError` that names the replacement:

```text
AttributeError: Workflow.command_context_for_response_generation is removed: use workflow.record
AttributeError: Workflow.current_command_context is removed: use workflow.record (assigning it moves the workflow)
AttributeError: Workflow.root_command_context is removed: use workflow.root
```

**An `Input` built directly without a context.** A validator that reads `info.context` needs the context the
framework passes. Built as `Signature.Input(title="Dune")` in a test or a script, the error below names the
validator. Build it with `Signature.Input.model_validate(values, context={"command_text": ..., "workflow": None,
"command": ...})`, or with `Signature.Input.model_construct(...)` when the check is not the point:

```text
Value error, _commands/needs_context.py: Input.must_be_in_text needs the validation context (command_text, workflow, command): validate through the framework, or build the Input with Signature.Input.model_construct(...) in tests (it raised TypeError: 'NoneType' object is not subscriptable)
```

**`info.context["request"]`** is `info.context["command_text"]`. A validator that still reads the old key (or calls
`.get("request")`, which raises too) fails the call with the removed-key message, which names the replacement:

```text
validate_extracted_parameters raised KeyError: "'request' is removed from the validation context: read info.context['command_text']"
```


## `from_command_output`

**A hand-written `available_from`.** It was the producer hint before `from_command_output`:

```python
order_id: str = Field(json_schema_extra={"available_from": ["find_order"]})
```

```text
_commands/Library/get_order.py: Input.order_id: available_from is written by from_command_output; write "from_command_output": "find_order" or "from_command_output": "find_order.<field>"
```

**Fix:** `order_id: str = Field(json_schema_extra={"from_command_output": "find_order.order_ids"}, ...)`. Prose that used to sit in the hint
("in the current context") belongs in the field's description.

**A `from_command_output` anywhere but an `Input` field.** An `Output` is built by code, and a facts
model's fields are typed with the commands they come from (page 10), so neither carries one:

```text
_commands/Library/book_by_title.py: Signature.Output.author: an Output field cannot declare from_command_output; build the value in __result__
```

```text
_commands/Book/shelf_report.py: Shelf.branch (from_command_output 'show_book.branch'): from_command_output is only for a command's Input field; type the field with the command's Output instead
```

**Fix:** for a facts model, `book: ShowBook.Output`, and read `shelf.book.branch`.

**A `from_command_output` on an `Input` field that names an `Input` field.** It names the `Output`
field a value comes from. It does not borrow another command's description:

```python
query: str = Field(json_schema_extra={"from_command_output": "find_book.query"})       # in _commands/Library/count_books.py
```

```text
_commands/Library/count_books.py: Input.query (from 'find_book.query'): find_book's query is an Input field; from_command_output names the Output field a value comes from. To describe the value, write Field(description=...); the preview shows find_book.query's description beside it
```

**An unqualified name off the path.** An `Input` field's unqualified name is looked up in the
declaring context, its bases, then the global commands. A producer in a context that encloses
this one is qualified, since which records enclose a record is known only at run time:

```python
member_uid: str = Field(json_schema_extra={"from_command_output": "find_member.member_uids"})   # in _commands/Book/reserve_for.py
```

```text
_commands/Book/reserve_for.py: Input.member_uid (from 'find_member.member_uids'): no command find_member in Book, its bases or the global commands; qualify it (Library/find_member)
```

**Fix:** `"from_command_output": "Library/find_member.member_uids"`. When the name matches commands in two
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
_commands/Library/book_loans.py: Input.title (from 'book_by_title.title') tells the agent to run Library/book_by_title first, but this command's own steps call Library/book_by_title. To describe the value, write Field(description=...) instead
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
_commands/Library/book_and_loans.py: PlanGenerator inherits from Library/book_by_title's PlanGenerator; to reuse its whole job, call book_by_title in a step; to share phases, move them to a base class in a _ module
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
`run("find_book", query=input.title).note("the first page")`, which code never reads.

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
the note "prefers email". The same warning names `run(..., all_pages=True)`, `run(..., named=...)`,
`when(...)` or `do_not_run(...)` for their words typed by hand.

**A note that holds a condition.** It loads and runs; `preview --check` warns without
failing. Code never reads a note, so a condition written in one does not stop the step.
Bare `only` in a note (`first page only`) is not treated as a condition word; `only if` and
`only when` are.

```python
def note(self, input):
    return [run("add_note", text=input.text).note("only when the member asked")]
```

```text
_commands/Library/bad_note.py: phase 'note', step 'add_note text=<text> (only when the member asked)': the note holds a condition, but code never reads a note, so add_note would run every time. Decide it in code with if, or write when("the member asked", "add_note text=<text>") for the agent to decide
```

**Fix:** `when("the member asked", run("add_note", text=input.text))`, or decide in code
with `if`.

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
_commands/Library/bad_binding.py: plan(): phase 'loans' takes 'found', which is not an earlier phase of this call; give the parameter a default or build the phase without it
```

**A subclass that adds a phase without saying where** (page 13):

```text
_commands/Library/bad_order.py: PlanGenerator adds the phase(s) overdue to those of bad_order.Branches: write __phases__(self, input) returning the phases a call runs, in order (from: self.branches, self.overdue)
```

## Calls between planning commands

**A cycle.** If `a`'s steps name `b` and `b`'s name `a`, each command on the cycle gets a load
error that names the path. The commands still load, and `fastworkflow command preview --check`
fails with the same lines. For `Library/book_loans` and `Library/book_by_title` calling each
other, the load reports:

```text
_commands/Library/book_by_title.py: a planning command cannot call itself, even through others: Library/book_by_title -> Library/book_loans -> Library/book_by_title
_commands/Library/book_loans.py: a planning command cannot call itself, even through others: Library/book_loans -> Library/book_by_title -> Library/book_loans
```

A command whose steps name itself directly gets the one-step form,
`a planning command cannot call itself: Library/count_books -> Library/count_books`. A step
written in open words can still name a running command at run time; it is refused with
`Library/a is already running; call its steps directly.`

**The step budget.** `The step budget for nested commands is spent; continue with the
remaining steps directly.` means the nested calls of one turn used `FW_TURN_MAX_STEPS`
model steps (400 by default). Raise it in `fastworkflow.env` if your heaviest planning
command needs more; the steps of a planning command matched as a typed call do not count,
the calls it makes do, and so do the steps of one a plan calls as a step (page 14).

## At run time, with no error

These are warnings, and one note, each logged once per command (or once per workflow) per
process.

**A command returned no `Output`.** `book_by_title declares an Output but returned none;
phases and from_command_output references cannot read it` means a command that declares a
`Signature.Output` returned only text. Return the `Output` on every path: an execution
command returns it from `__call__` (page 1), and a planning command from `__result__`, saying
"not found" in a field when it found nothing (page 11).

**A hook that raised.** A `__result__` that raises, or returns something other than a `str`,
its command's `Output` or a `CommandResponse`, keeps the call: a model writes the answer from
the phases' runs, `output` is `None`, and the warning names the command and the exception.

**`classifier_confirms` checks nothing.** `library: planning commands are matched by intent detection; without
JEV_API_KEY, Jev's planning-match candidates and the classifier checks of <fields> are off.`, logged
once at load, means the workflow has no `JEV_API_KEY`. Planning commands are ordinary intent labels,
so intent detection still matches them. Without the key, the Jev matcher does not run, and the
fields the line names carry a `classifier_confirms` option that checks nothing (page 9).

**A navigation command without `@descends_to`.** `open_shelf moved the workflow into Shelf but
declares no @descends_to; declare @descends_to("Shelf") so plans can reach Shelf's commands`
means a command moved the workflow (by setting `workflow.record`) into a context it does not
name. Declare it on the command (page 5).

**A planning command is refused.** `member_report is a planning command; it runs in agent
mode.` means it was called by a direct API or MCP call, or it was named in assistant mode and
needs the agent. A refusal in assistant mode lists the reasons, each naming the step and what to
change. Run the request in agent mode, or change the step the reason names.

**An approval prompt does not run the command.** A command decorated with
`@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)` asks
before it runs. Reply `yes` to run it, `no` to decline, or `always` to approve it from now on for
that user, workflow and command. An approval expires after 2 turns or 15 minutes, and any change to
its parameters cancels it. `/approvals` lists the standing approvals, `/revoke_approval <command>`
revokes one, and `/cancel_pending` cancels the approval that is waiting (see
`docs/planning_command_file_layout.md`, "Approval annotation").

**The workflow will not load after an upgrade.** "Retrain with `fastworkflow train`" means its
classifiers were trained by an earlier release. Retrain it. The first train after the upgrade is a
full retrain, and for a workflow with planning commands it generates their training utterances,
which costs LLM calls. A warning that `context_hierarchy_model.json` is ignored means the file is no
longer read and can be deleted (page 14 says where parents come from).

## A fact named like a `str` method

While the template is built, a phase's facts are one placeholder, which is a `str`. A fact
named like a public `str` method (`title`, `count`, `format`, `split`, ...) still reads as the
fact, not the method, so `book.title` is `<book.title>`. Only a name the placeholder means
itself (`index`, the one item's place) keeps that meaning.

## Examples that do not load

Each file below is saved as `_commands/<name>.py` in a copy of the tutorial workflow. The message under it is what the load (or the check) reports, with the folder's path left out.

<!-- broken: no_docstring -->
```python
from fastworkflow.planning import run


class Signature:
    plain_utterances = ["show the thing"]


class PlanGenerator:
    def find(self):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/no_docstring.py: Signature needs a docstring (the planning command's description)
```

<!-- broken: call_defined -->
```python
from fastworkflow.planning import run


class Signature:
    """Show the thing."""


class PlanGenerator:
    def __call__(self):
        return [run("list_branches")]

    def find(self):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/call_defined.py: PlanGenerator defines __call__; rename it to __phases__
```

<!-- broken: result_param -->
```python
from fastworkflow.planning import run


class Signature:
    """Show the thing."""


class PlanGenerator:
    def find(self):
        return [run("list_branches")]

    def __result__(self, input, other):
        return "done"
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/result_param.py: __result__ takes 'other', which is not a phase of this command
```

<!-- broken: bad_option -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra={"strip_words": "the"})


class PlanGenerator:
    def find(self, input):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/bad_option.py: field 'title': unknown option 'strip_words'; its options are copied_from_command_text, record_named_in_command_text, classifier_confirms, classifier_decides, command_text_after_command_name, from_command_output, converted_from_command_text
```

<!-- broken: fan_out_param -->
```python
from fastworkflow.planning import phase, run


class Signature:
    """Show the thing."""


class PlanGenerator:
    def find(self):
        return [run("list_branches")]

    @phase(over="find.uids")
    def item(self, find, extra):
        return [run("list_branches")]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/fan_out_param.py: plan(): phase 'item': over='find.uids' names 'find', which is not an earlier phase with an output
```

<!-- broken: fact_source -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the thing."""


class PlanGenerator:
    class Found(BaseModel):
        uid: str = Field(json_schema_extra={"from": "list_shelves.uid"})

    def find(self) -> Found:
        return [run("list_branches")]
```

<!-- check-error -->
```text
_commands/fact_source.py: Found.uid is str; a facts field is typed with a command's Output or Input. Compute other values in a property or method of Found; copies are no longer read (fact_from and "from" are removed)
```


<!-- broken: say_that_last -->
```python
from pydantic import BaseModel, Field
from fastworkflow.planning import run


class Signature:
    """Show the branches."""


class PlanGenerator:
    class Branches(BaseModel):
        names: list[str] = Field(default_factory=list, description="the branch names")

    def find(self) -> Branches:
        return [run("list_branches")]

    def answer(self, find):
        return [f"say that {len(find.names)} branches"]
```

<!-- load-error -->
```text
PlanningCommandFormatError: _commands/say_that_last.py: phase 'answer' writes the answer as 'say that' lines, which code no longer copies; return the text from __result__(self, input, ...) instead
```
