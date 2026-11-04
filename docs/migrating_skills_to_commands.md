# Migrating an app's `_skills/` to planning commands

This guide is for an app workflow that keeps its planning commands in `_skills/`. Before this
release a planning command was a "skill" file in `_skills/`. Now it is a command file in
`_commands/`, with a `PlanGenerator` class, like any other command. A workflow that still has a
`_skills/` folder does not load.

The worked example is a production app workflow, `acme_workflow`. Its names are made up. The
steps are the ones the release requires; nothing here is specific to one app.

Read first:

- [planning_command_file_layout.md](planning_command_file_layout.md): the file form, one page.
- [planning_command_phases.md](planning_command_phases.md): how phases run.
- [planning_command_phases_code_run_steps.md](planning_command_phases_code_run_steps.md): the step helpers and code-run steps.
- [tutorials/planning_commands/README.md](tutorials/planning_commands/README.md): a step-by-step tutorial on a toy workflow.

Each step below says what the code checks and what it prints. Where a claim is about a message
or a flag, it comes from the code on this branch.

## 1. Stop the app's generator from overwriting planning commands

If the app generates files into `_commands/` (from its own code, as `fastworkflow build` does),
the generator must leave planning commands alone. A planning command is a file with a top-level
`class PlanGenerator`. Find them with `ast`, as fastworkflow does. The check looks only at
top-level class definitions, and it never imports the file:

```python
import ast
from pathlib import Path


def is_planning_command(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(node, ast.ClassDef) and node.name == "PlanGenerator"
               for node in tree.body)
```

Your generator must:

- skip every file under `_commands/` that `is_planning_command` returns true for, and never write
  to that path, even with an overwrite option;
- keep the `_` helper modules those files import (for example `_book_phases.py`);
- for a file that does not parse, keep it too. fastworkflow's own check falls back to a text
  search for `class PlanGenerator` when `ast` fails, so a half-edited planning command is not
  overwritten.

fastworkflow's own generator follows this rule. `fastworkflow build` refuses to write over a
planning command even with `--overwrite`, and it logs a warning that names the file: "it is a
planning command (defines PlanGenerator)". Generated files start with this header line:

```python
# fastworkflow:generated — regenerate with fastworkflow build; edits may be overwritten
```

The header is a note for people. It does not protect anything. A file without the header is
still overwritten by an execution-command generator, so the `PlanGenerator` check is the rule
that matters.

## 2. Move the files

Move each planning command from `_skills/` to `_commands/`:

| Before | After |
|---|---|
| `_skills/<name>.py` | `_commands/<name>.py` |
| `_skills/<Context>/<name>.py` | `_commands/<Context>/<name>.py` |
| `_skills/_<helper>.py` | `_commands/_<helper>.py` (or into the context folder that imports it) |
| `_skills/_split.py` | `_commands/_split.py` |

A planning command in a context folder moves with its folder: `_skills/<Context>/<name>.py`
becomes `_commands/<Context>/<name>.py`, and the command is named `<Context>/<name>`. Phase 1
added context folders under `_skills/`, so an app that has them keeps them as they are.

Files whose names start with `_` are not loaded as commands. A `_` helper moves with the
commands that import it. If two planning commands in different folders use the same helper,
keep one copy in a `_` module that both can import from.

### The split settings file

`_commands/_split.py` is the workflow's split settings. Every attribute in it is optional:

```python
import re

identifier_patterns = [re.compile(r"\bacct_[0-9]+")]
subject_hints = "An account is a record. Its id looks like acct_<number>."
```

It may also have a `class Signature` whose docstring replaces the splitter's instructions.
Its `Signature.Input` and `Signature.Output` fields can give the splitter's fields a description.
A field the splitter does not have fails the load.

The file may also define plain functions. A planning command can import one from it:

```python
from ._split import order_kind
```

No test in the repository covers a command that imports from `_split.py`. Check this in your
workflow with a load and `fastworkflow command preview --check` before you rely on it.

### A workflow that still has `_skills/` fails to load

fastworkflow refuses to load a workflow with a `_skills/` folder. The message is:

```
_skills/ is no longer read; move these files into _commands/ (first make sure any script that regenerates _commands/ preserves them; see docs/migrating_skills_to_commands.md)
```

The first sentence is the instruction to follow. The message is the constant
`PLANNING_COMMANDS_FOLDER_RETIRED` in `fastworkflow/command_directory.py`.
The check runs for the workflow and for each base workflow it inherits from.

## 3. Fix the imports

Relative imports depend on where the file sits. Moving a file changes its depth in two cases.

**A root file moves into a context folder.** Each leading dot adds one level. Change `from ._x`
to `from .._x`, and `from ..application` to `from ...application`:

```python
# _skills/book_report.py (root)           _commands/Book/book_report.py (context folder)
from ._phase_facts import Facts            from .._phase_facts import Facts
from ..application import library          from ...application import library
```

**A move to the `_commands/` root keeps every relative import.** A root skill moved to the root of
`_commands/` needs no changes.

A file that moves from `_skills/<Context>/` to `_commands/<Context>/` keeps its depth, because
both folders are one level below the workflow.

Check each changed import by loading the workflow. A wrong depth fails with an import error that
names the file.

## 4. Regenerate the pinned previews

Each planning command has a preview file that shows what a call runs. The previews are pinned
in the repository. Write them again after the move, then check them:

```bash
fastworkflow command preview <workflow_folder> --out <dir>
fastworkflow command preview <workflow_folder> --out <dir> --check
```

- `<workflow_folder>` is the workflow directory (the one that holds `_commands/`).
- `--out` is the folder for the previews. Without it, the previews go to `docs/planning_command_previews`
  next to the workflow folder.
- Without `--check`, the command writes the previews and prints how many were written, unchanged
  and removed.
- With `--check`, the command writes nothing. It exits 1 when a preview is stale, missing or
  extra, or when a planning command does not load.

`--check` also prints warnings about steps (section 8). Warnings are printed, but they do not
change the exit code. Only stale, missing or extra previews do.

## 5. Write the field options correctly

### `available_from` becomes `from_`

A hand-written `available_from` in `json_schema_extra` is now an error on a planning command's
`Input` field. The load reports it as: `available_from is written by from_; write from_("<command>") or from_("<command>.<field>")`.
Rewrite it with `from_`, and move any prose into the field's `description`:

```python
# before
class Input(BaseModel):
    book_uid: str = Field(
        description="the uid of the book, from the find step",
        json_schema_extra={"available_from": ["find_book.book_uids"]})

# after
from fastworkflow.planning import from_

class Input(BaseModel):
    book_uid: str = from_("find_book.book_uids",
                          description="the uid of the book to open")
```

`from_` takes the same keyword arguments as `Field`.

### Descriptions are written for people

A field's `description` says what the value is, for a person. Remove instructions to the model.
The framework adds the rule sentence for each option the field carries. For `in_request` on a
`str`, the model is told "Copy it exactly as the request writes it." once, so do not repeat it.

```python
# before
description="Copy the member's name exactly as written. Never guess a name."

# after
description="the member's name",
json_schema_extra={"in_request": True}
```

### `in_request` on each `str` field that relied on the implied check

On this branch, a planning command's `str` argument is still checked against the request
(it must be written in the request). The check is implied by the type. Add `in_request` to every
`str` field of a planning command that relies on that check and has no `in_request: "subject"`. The option makes
the check explicit, and it is the check you keep when the implied one goes:

```python
class Input(BaseModel):
    title: str = Field(description="the book's title",
                       json_schema_extra={"in_request": True})
```

A `str` field that carries `in_request: "subject"` already has the check, so leave it as it is.
`in_request` takes a `str`, an `int` or a `list[str]`; `"subject"` takes a `str`. On any other type it
is a load error.

### `PlanGenerator.__call__` is now `__phases__`

The method that returned the plan was `__call__`. It is now `__phases__`. A `PlanGenerator`
that still defines `__call__` does not load. The message is `PlanGenerator defines __call__; rename it to __phases__`.

```python
# before
class PlanGenerator:
    def __call__(self, input):
        return [self.find, self.answer]

# after
class PlanGenerator:
    def __phases__(self, input):
        return [self.find, self.answer]
```

`__phases__` is optional. Without it, every phase method runs in the order written.

## 6. Use the step helpers

Each step is written with a helper from `fastworkflow.planning`. `run` is the one most steps need.
It takes the command name and its values as keyword arguments, and it quotes a value that needs
quoting:

```python
from fastworkflow.planning import pick, run


class PlanGenerator:
    def find(self, input) -> Match:
        return [run("find_book", query=input.title)]        # not "find_book query=..."
```

Use `run` for every step that passes a value. A plain string still works: the step is read as
written. But a string that a helper would write gets a warning from `--check` (section 8).

### `pick` without a uid

`pick` opens one row of a listing. If the listing ran in an earlier phase of the same call, drop
the uid. Code then finds the row whose name matches, from the rows that call already read:

```python
# before
def book(self, input, find) -> Book:
    if not find.book_uids:
        return []
    return [pick("open_book", book_uid=find.only_uid(input.title), named=input.title)]

# after
def book(self, input, find) -> Book:
    if not find.book_uids:
        return []
    return [pick("open_book", named=input.title)]
```

Keep the uid when the phase writes it into another step. A `run` step that takes the uid as a
value, such as `run("list_loans", book_uid=find.only_uid(input.title))`, must keep it, because the
uid is what that step passes on. `named` is required and must not be empty.

## 7. Replace a subclassed `PlanGenerator`

A `PlanGenerator` can no longer subclass another command's `PlanGenerator`. The load fails with
`PlanGenerator inherits from <command>'s PlanGenerator; to reuse its whole job, call <command> in a step; to share phases, move them to a base class in a _ module`.
Choose one of two replacements.

**To reuse another command's whole job, call it in a step.** Use a nested `run`:

```python
class PlanGenerator:
    def answer(self, input, find):
        return [run("book_by_title", title=input.title)]
```

**To share phases between two commands, move them to a base class in a `_` module.** The shared
module keeps its leading underscore, so the loader does not treat it as a command. The base
class is not a planning command. Each command subclasses it and keeps its own `__phases__` when it
adds phases:

```python
# _commands/_book_phases.py
class BookPhases:
    def find(self, input) -> Found:
        return [run("find_book", query=input.title)]

    def answer(self, input, book=None):
        if book is None:
            return [f"say that no book matches {input.title}"]
        return [f"say that {book.name} is by {book.author}"]


# _commands/book_loans.py
from ._book_phases import BookPhases

class PlanGenerator(BookPhases):
    def __phases__(self, input):
        return [self.find, self.book, self.loans, self.answer]
```

The base module is in the repository as a working example: [tests/planning_tutorial_workflow/_commands/_book_phases.py](../tests/planning_tutorial_workflow/_commands/_book_phases.py).

## 8. Read the `--check` warnings

`fastworkflow command preview --check` prints two kinds of warning. Neither changes the exit code.

- **A step written by hand.** A step that a helper would write is written as a string. The
  warning names the file, the phase and the step, and says which helper to use (`run`, `pick`,
  `full_list`, `when` or `do_not_run`).
- **A note that holds a condition.** A command step whose note says when it runs (such as
  `(only when ...)`) gets a warning. Write the condition with `when` instead.

Fix each warning by using the helper it names. The code then reads the step as you mean it.

## 9. Names that changed

Apps write or import these names. Check the app for each one.

| Before | After |
|---|---|
| `fastworkflow.skill` (module) | `fastworkflow.planning`. The old module is removed with no alias: `import fastworkflow.skill` fails with `ModuleNotFoundError`. |
| `SkillFormatError`, `SkillInputError`, `SkillCallError` | `PlanningCommandFormatError`, `PlanningCommandInputError`, `PlanningCommandCallError`. Only an app that catches them is affected. |
| `load_skills` | `load_planning_commands` (only an app that calls it) |
| `PlanGenerator.__call__` | `PlanGenerator.__phases__` (section 5) |
| `FW_SKILL_PHASE_MAX_STEPS`, `FW_SKILL_FANOUT_MAX_ITEMS`, `FW_SKILL_FACTS_MAX_LIST_ITEMS` | Module constants, `PHASE_MAX_STEPS` (15), `FANOUT_MAX_ITEMS` (10) and `FACTS_MAX_LIST_ITEMS` (100) in `fastworkflow.planning_phases`. The environment settings are no longer read. Remove them from the app's env files. |

The `fastworkflow.planning` module exports `count`, `do_not_run`, `fact_from`, `from_`, `full_list`,
`listed`, `num`, `phase`, `pick`, `run` and `when`. Import from there.

## 10. Declare which command enters a context

A command that enters a context declares it with `descends_to` on its `ResponseGenerator` or
`PlanGenerator` class. The library example does this on `open_book`:

```python
class ResponseGenerator:
    descends_to = "Book"
```

A context class that still declares `enter_command` or `enter_commands` is a load error in this
release. The error names the file, the command and the fix. Delete the attribute from the context
class, and declare `descends_to` on each command that enters that context. A command that moves the
workflow into a context it does not declare logs one warning per command and target. Commands in the
app that do not enter a context need no change.

## 11. Retrain, and review approvals and return values

- **Retrain** each workflow that has planning commands with `fastworkflow train`. The first train
  after the upgrade generates training utterances for each planning command, which costs LLM calls.
  Until you retrain, the workflow's artifacts refuse to load and say to retrain.
- **Approvals.** A command with `approval = "mandatory"` now asks before it runs, and the user
  replies `yes`, `no` or `always`. See `docs/planning_command_file_layout.md`, "Approval annotation".
- **Return values.** A command returns its own `Signature.Output`, a `str`, a `CommandResponse` or a
  `CommandOutput`. Any other type fails that call with a `TypeError` that names the command.

## Coming in later releases

These items are not in this release. Do them when the release that lands them comes out.

- **Facts models as typed fields.** A group of `fact_from` copy fields that reads one command's
  output will become one field typed with that output. Copy fields will stop loading.
- **Answers in `__result__`.** A last phase written as `say that` lines will stop loading. Write the
  answer in `__result__`, which already works on this branch (see the library example
  [book_by_title.py](../fastworkflow/examples/library/_commands/Library/book_by_title.py)).
