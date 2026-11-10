# Planning command file layout

A planning command file (`_commands/<name>.py`; reference: the `fastworkflow.planning` module
documentation; the runtime: [planning_command_phases.md](planning_command_phases.md)) is a command file for a
whole job. Where an execution command has a `ResponseGenerator`, a planning command has a
`PlanGenerator`. It is read far more often than it is written, so every planning command file
has one form, the same few idioms as a command file, read top to bottom in the same order.
Planning commands live in `_commands/`, beside the execution commands; a `_skills/` folder is a
load error. `fastworkflow command preview` (below) renders what each call of it runs. A
step-by-step tutorial, on a sample workflow:
[tutorials/planning_commands](tutorials/planning_commands/README.md).

## How to write a planning command (one page)

```python
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_item import Signature as FindItem
from .list_parts import Signature as ListParts
from .open_item_by_uid import Signature as OpenItemByUid


class Signature:                      # exactly a command's Signature
    """What the planning command does (the planner chooses it by this)."""
    plain_utterances = ["what parts does the Alpha item have"]
    class Input(BaseModel):
        name: str = Field(description="the item's name")

class PlanGenerator:                  # a command generates a response; a planning command, a plan
    def find(self, input) -> FindItem.Output:             # phase 1; it fills find_item's Output
        return [run("open_catalog"),
                run("find_item", query=input.name)]

    class Item(BaseModel):                                # its facts: typed by commands' Outputs
        opened: OpenItemByUid.Output
        parts_list: ListParts.Output

        @property                                         # other values: computed, not copied
        def uid(self) -> str:
            return self.opened.item_uid

        @property
        def parts(self) -> list[str]:
            return self.parts_list.labels

    def item(self, input, find) -> Item:                  # phase 2 reads find's facts
        if not find.item_uids:
            return []                                     # skipped, decided when reached
        return [run("open_item_by_uid", named=input.name),
                run("list_parts", all_pages=True)]

    def __result__(self, input, item=None):               # the answer; None: item was skipped
        if item is None:
            return f"No item named {input.name} was found."
        return f"{item.uid} has parts {self._parts(item)}."

    def _parts(self, item):                               # _name: a helper, not a phase
        return ", ".join(item.parts) or "none"
```

**The rules.** (1) Each public method of `PlanGenerator` is a phase (one executor run), in
the order written. (2) Its parameters name what it reads: `input`, and earlier phases by name
(their facts; `=None` when that phase may have been skipped). (3) Its return annotation is what it
fills: a command's `Output`, or a facts model whose fields are typed with commands' `Output`s or `Input`s;
any other value is a property or method of it (copies, `fact_from` and `"from"`, are not read). (4) It
returns its steps, exactly the lines the agent reads. `return []` skips it: decided when the
phase is reached, on the real facts. A planning command with no `Input` fields may have phases too
(methods without `input`; every call has the one plan). (5) `__result__` is the one method
that is not a phase: it returns the call's answer from the phases' facts, and its parameters
name `input` and the phases (`__result__(self, input, item=None)`). It may return `None` to let
the agent answer, as a planning command without `__result__` does.

**Sharing phases.** A base class in a `_` module (`_commands/_book_phases.py`, `class BookPhases`,
as in `tests/planning_tutorial_workflow`) holds phases two commands share; `class PlanGenerator(BookPhases)`
keeps them and adds its own, and a method of the same name overrides one. A `PlanGenerator` may not
subclass another command's `PlanGenerator` (a load error): to reuse that command's whole job, call it in
a `run()` step. With `__phases__` the order is explicit: `return [self.find, self.book, self.loans,
self.answer]` (default: every phase, in the order written).

**Steps.** `run("command", k=v)` writes a step code runs when every value is given:
`find_item query=Dune`. `run("command", all_pages=True)` writes one that code reads through every page
(`(full list)`). `run("command", named=name)` opens the row a find listed, the one whose
label matches `name` (`run("command", item_id=uid, named=name)` when the row id is known; `item_id`
is the command's own input name). The reserved input names are `named` and `all_pages`.
`when("condition", step)` writes a step the agent decides (`only when ...`), and
`do_not_run("command")` a guard code passes over. `say that ...` is a verbatim line that writes no
answer by itself in a phase before the last (only `__result__` can use its text); `apply no change`
and `do not run X` are guards; any other line is the agent's. A last phase of only `say that` lines
does not load: return the answer from `__result__(self, input, <phase facts...>)`. A note in parentheses
after a step is never sent. Text helpers: `count(n, "login")` (use it rather than an inline `n == 1`, which the preview
cannot render), and `count_and_names_read(names, total, "group")` for a counted list. The retired helpers
`pick`, `full_list`, `listed` and `num` fail the import, each naming its replacement.

**The shape of a call.** By default a call runs every phase. When the input changes it, write
`def __phases__(self, input)` returning the phases, in order: `return [self.find, self.answer]`
or `[self.quick]`. A call of one phase that reads and fills no facts runs flat, as one plan.
A phase repeated for several cases is one method each, calling one shared module function.

**Options** (`@phase`, the one decorator): `over="find.uids"` runs the phase once per item
(its one parameter that is neither `input` nor a phase gets the item), with `max_items=`,
`until=`, `noun=`; `optional=True`: the job goes on when it fails; `output=` a facts model
that depends on the input. An option written `lambda input: ...` is decided per call.

**What the planner sees.** Each phase written once per input with every fact unknown:
`<find.total>`, `<find.uids[0]>`, `<find.labels[*]>`; a helper called on facts reads as its
name (`<parts>`); `pick` reads `... with the uid of the NAME match`. Every test on a fact
reads as true and is noted: the phase's lines end `(only when find.uids)`. So write a skip
as `if not <what the phase needs>: return []`, and put wording that tests facts in helpers.

**Checked at load** (else the planning command is not loaded): a method without `self`, a parameter
that names no earlier phase, an annotation that is not a model, a method that fails on
unknown facts or writes `None` into a step, a step naming a command the workflow lacks or a
parameter its `Signature.Input` lacks. `fastworkflow command preview <workflow>` shows every
call's phases, facts, steps and who writes the answer; `--check` fails when one is stale.

## When to write a planning command

A command is one of two kinds, in the same `_commands/` tree and with the same `Signature`. An
**execution command** (`ResponseGenerator`) runs your code and returns its `Output`; it runs in
agent and assistant mode and through direct API and MCP calls. A **planning command**
(`PlanGenerator`) is a job the agent runs from a plan: code runs every step it can, and the agent
does the rest. Write an execution command when code can decide every step. Write a planning
command when the agent should run, judge or recover part of the job: a value to find, a judgement
to make, a step to recover when it goes wrong, or a question for the user. A planning command runs
in agent mode. In assistant mode (`fastworkflow run --assistant`, or a message with a leading `/`)
an exactly named planning command runs when it is mechanical: code runs every step and nothing needs
the agent. One that needs the agent is refused with the reasons, which name the step and what to
change. Direct API and MCP calls still refuse a planning command, with "... is a planning command;
it runs in agent mode."

## The order

Every planning command file has these parts, in this order: the module docstring (the job, what it
reads, what it answers, the conditions that change it, in two to five sentences); the
imports; `Signature`; the facts models shared by several phases (a model one phase fills is
nested right above that method); `PlanGenerator`, its `__phases__` first when it has one, then
the phases in run order, then its `_helpers`; then module-level tables and shared functions.
A reader meets what the planning command does, what it takes, how a call is shaped, then each
phase with its facts, then the wording. Lookup tables and phrases shared by several planning
commands go in a helper module (`_commands/_<name>.py`, data only); everything that decides which
steps a planning command writes stays in its file.

## Rules on the arguments

- A rule that ties fields together, or a field to the request, is a pydantic validator on
  `Input` (`model_validator(mode="after")`, or `field_validator` for one field), written below the fields
  in `Signature`. It is the one place for a cross-field rule. It corrects by assigning `self.<field>` and
  returning `self`, or refuses with `raise fastworkflow.Refused(message, field=...)`. A plain `ValueError`
  refuses with no field. Its message tells the planner what to fix, naming the field and the value that
  would fit: `"record_a and record_b are both 'Alpha'; record_b is the other record the request names"`.
- The validator reads the context from `info.context`: `"command_text"` (the request; `""` while the template
  renders), `"workflow"` (`None` while the template renders) and `"command"` (the qualified name). It is
  always present, so a rule on the command text is silent in the template.
- Refuse only on a clear conflict. A refusal is shown to the planner, which fills the call
  once more. A second refusal is asked of the user as a question when the turn may ask
  (`refusal_question`); otherwise the planner plans from the template.
- Several model validators run in declaration order. **Upgrade step:** when a hook is moved into a validator,
  declare the moved validator first, before the validators the `Input` already has. Pydantic runs them in
  declaration order, so an existing check otherwise sees the value before the correction.

```python
class Input(BaseModel):
    first: str = Field(description="the first item's name")
    second: str = Field(description="the second item's name")

    @model_validator(mode="after")
    def _two_items(self, info):
        if self.first.casefold() == self.second.casefold():
            raise fastworkflow.Refused(f"first and second are both {self.first!r}; second is the other "
                                       "item the request names", field="second")
        return self
```

## Field options

An `Input` field takes options in `json_schema_extra`: a key per behaviour, its value `True` or a dict of that
key's own options. Combine keys by adding them. Plain pydantic `Field(...)`; there are no helper functions. An
unknown key, a retired spelling, a value that is neither `True` nor a dict of known options, or two keys that
conflict fails the load, naming the key to write instead. They work on planning commands today;
`fastworkflow.planning_commands` has the full rules, and `fastworkflow.FieldOptions` (and its TypedDicts) gives IDE
completion.

- `copied_from_command_text: True` (a `str`, an `int` or a `list[str]`): a `str` must appear in the request as a
  verbatim substring (per item for a `list[str]`, which keeps the items the request writes); an
  `int` must be a number the request writes. Otherwise the one value the request writes is taken,
  or the default when that is written; with several candidates the call is refused.
  `copied_from_command_text` checks the text the user wrote; it does not interpret the description.
- `record_named_in_command_text: True` (a `str`): the argument names a record of the request (a person, a group,
  a product, a code). It checks names already written in the command text; it does not interpret the
  description. The parameter extractor (`LLM_PARAM_EXTRACTION`) proposes the value from the
  field's description. The description says what the field holds. The framework appends "Copy it
  exactly as the request writes it. It must name a record the request names." The check then matches
  that proposal against names the request already writes: one of the request's subjects, a name-like
  word, or a name written in lower case. One cut short at the start or end of a single subject
  becomes that subject. A model chooses among those names again only when the field also has
  `classifier_confirms` alongside, and only when `JEV_API_KEY` is set.
- `classifier_confirms: True` (a `bool`, an enum, or a `str` with `record_named_in_command_text`): a classifier model
  decides the argument again from the request alone, and its answer replaces the planner's when it
  is sure enough. It needs `JEV_API_KEY`: without it nothing is checked, and the load logs one INFO
  line that says so. On an enum, `classifier_confirms` may instead be a dict of each value's meaning,
  `classifier_confirms: {"meanings": {"group": "a group of people", "unit": "an organisation unit"}}`, which the check
  reads. Every key must be one of the enum's values.
- `classifier_decides: True` (a `bool`): the classifier alone decides the field from the request. The typed
  planning call and the parameter extractor are not asked for it, and it keeps its default when the
  classifier is not sure or there is no `JEV_API_KEY`. Use it for a part of the job the request may
  leave out, worded as the exclusion with a false default, and read it in `__phases__`:

  ```python
  leave_out_risk: bool = Field(
      default=False,
      description="the request says to leave out the risk score, or asks only for other parts",
      json_schema_extra={"classifier_decides": True})
  ```

  Word it as the exclusion, not as "the request asks for the risk score": asked of a part the
  request does not mention, that question would drop it.
- `copied_from_command_text: {"ignore": [words]}` (a `str`; `record_named_in_command_text` takes the same `ignore`):
  the named words are removed from either end of the value before the check, so a kind word the model appends to
  a name ("Payroll group" becomes "Payroll") is cleaned as the request writes it. This is the cleanup the retired
  `strip` option did. A field with no check is cleaned in code instead: `strip_words(text, words)` from
  `fastworkflow.planning_commands`, called in the field's `field_validator`.
- `command_text_after_command_name: True` (a `str`, at most one per `Input`; execution commands only): takes the text
  after the command name. `from_command_output: "<command>.<field>"` (or a list) names the command that supplies the
  value (see `declare-parameter-producers`). `converted_from_command_text: True` (execution commands) marks a value
  converted from the command text, such as a date written "next Monday".

An `Input` validator's returned value is written back to the arguments: a validator that cleans a
value changes the argument, and a refusing one still refuses. The option checks run first, then the
model validators (in declaration order), then the roles.

The retired options fail the load and name the new key: `in_request` (write `copied_from_command_text: True`,
or `record_named_in_command_text: True` for `"subject"`), `verify` (write `classifier_confirms: True`, or
`classifier_decides: True` for `"alone"`), `means` (write `classifier_confirms: {"meanings": {value: meaning}}`),
`subject` (write `record_named_in_command_text: True`), `strip` (write `copied_from_command_text: {"ignore": [...]}`,
or call `strip_words(...)` for a field with no check), `positional` (write `command_text_after_command_name: True`),
and `from` (write `from_command_output`). The `validate_extracted_parameters` hook fails the load and names the
model validator form.


## Naming a record: `get_displayname` and `instance_label_attr`

A context class declares two things about the record it manages. They answer different questions, so keep them apart:

| | `get_displayname(cls, record)` | `instance_label_attr = "<attribute or property>"` |
|---|---|---|
| What it is | a classmethod returning any free-form display string | a class attribute naming one attribute (or property) of the record |
| Example | `"Book: Dune (b12)"` | `"uid"`, whose value on the record is `b12` |
| Answers | how the record should look to a person | what the record's identity is |
| Read by | `workflow.record_displayname` (for the record the command acts on), the CLI and chat context line, observability | `context_identity`: listing tags, `search_memory`, the extract step, the approval binding |
| When absent | optional; without it the context name is shown | nothing is guessed; the record has no identity in those readers |
| Derived from the other? | no: it may use the identity, but `instance_label_attr` is never read from it | no: it names an attribute, it does not compute one |

`workflow.record_displayname` reads `get_displayname` for the record the command acts on. It is a reader of that
declaration, not a third one. A computed identity is a property on the record, and `instance_label_attr` names
that property. The `instance_label` classmethod is removed: a context that still defines one fails the load.

## Approval annotation (`@user_approval_for_execution`)

A command may carry `@user_approval_for_execution(...)` on its class: `ResponseGenerator` for an execution command,
`PlanGenerator` for a planning command. It takes one of two values from `UserApproval`, and nothing else, and is
imported from `fastworkflow` (or `fastworkflow.planning`):

```python
@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)   # the action cannot be undone, e.g. sending a critical email
class ResponseGenerator:
    ...
```

- `UserApproval.REQUIRED_ALWAYS`: the framework asks before every run, and the run waits for the reply.
- `UserApproval.REQUIRED_UNTIL_DELEGATED`: the framework asks until a standing approval for the user exists.
- Absent: the agent decides, with no gate. Most commands are like this.

The annotation is not read/write gating. An easy-to-undo write needs no annotation, and only the two
values exist. The decorator is validated when the routing definition loads. The stored value is
`"required_always"` or `"required_until_delegated"` in `command_directory.json`. The old `approval = "mandatory"`
and `approval = "delegatable"` attributes fail the load and name the decorator to write. A value that is not a
`UserApproval` member fails the load and names the command, its file and the two allowed values. The listings show a marker after the name: `(needs your approval before
it runs)` for `REQUIRED_ALWAYS`, `(may need your approval)` for `REQUIRED_UNTIL_DELEGATED`. A workflow with no
annotation gets no marker.


## Write declaration (`@writes`)

An execution command that changes the application's data carries `@writes` on its
`ResponseGenerator`. A planning command does not declare it: it writes when one of its steps (directly,
or through a planning command it calls) runs a command that declares it, and it is read-only otherwise.
`@writes` on a `PlanGenerator` is a load error naming the command and its file (`fastworkflow.command_writes`);
the old `writes = True` attribute fails the load and names the decorator.

Inside a planning command's phase, a command that writes runs only when a step of that phase names it.
The phase's command listing leaves out the other write commands, and a call of one is refused unrun:
the agent reads that the command changes the application's data and no step of this part of the job
runs it. Outside a planning command's phase nothing changes. `@writes` is independent of `UserApproval`: an
easy-to-undo write needs no approval, and still carries `@writes`.



1. the planning command's name, its description and its kind (flat, phased, or fixed steps without
   inputs);
2. the inputs table: name, type, default, the sample value it is rendered with, the
   `classifier_confirms` / `classifier_decides` column, the `copied_from_command_text` /
   `record_named_in_command_text` column and the description, then the classifier meanings (the
   `classifier_confirms` dict);
3. the example requests (`plain_utterances`);
4. for a phased planning command, the phases (output model, when each runs, and how it fans out)
   and the facts each output model holds, with the command each one is typed by. When some inputs
   make a call run flat, one line above the table says when it runs as phases, and the
   table is about the phased calls. A phase's condition reads "only when <the tests its
   method made on the facts>" (`find.uids`);
   the "fans out" column gives `over`, `item`, `max_items` and
   `bind=False`. A cell that differs by input (the output model, the `when`, the fan-out)
   lists each value with the inputs that give it ("when kind = crate: `CrateMatch`");
5. the steps for the sample inputs: phase by phase for a phased planning command, the `steps()` list
   for a flat one. An answer-only phase (its steps name no command, read with the
   workflow's command names as a run reads them) ends with who writes its answer, decided
   by the same wording rules a run uses (`planning_phases.answer_by_code`,
   `answer_step_kind`): "Answer: written by code, word for word (steps 1-2 verbatim; step
   3 dropped as a guard)." or "Answer: written by the model (step 2 needs judgement)." A
   phase that runs commands has no such line: the agent writes its answer. Steps on a
   paged command (a paging parameter such as `start`) that write no paging value are
   followed by how many pages each reads, by the rule code pages by
   (`planning_phases.step_paging`): "Pages: step 2 reads up to 20 pages, then the agent reads
   on; steps 4 and 5 read page 1 only." Only a note that asks for the full list
   (`list_parts (full list)`, see `FULL_LIST_PHRASES` in `docs/planning_command_phases.md`) reads
   later pages, which code reads up to `planning_phases.CODE_MAX_PAGES` (20), then the agent
   reads on from the next page; any other note
   ("(all of them)") or none reads page 1 only. The paging parameters come from the
   workflow's commands, so a folder without its `_commands` shows no "Pages:" line. Steps
   a run adds (the planner's
   `steps_after`) are not in the preview.

**Sample inputs** come from the `Input` model: a str takes its default, else its first
`examples` value (`Field(examples=[...])`), else the placeholder `<name>`; a bool is false;
an int its default, else its first example, else 1, moved into its `ge`/`le` bounds (a
default of 0 under `ge=1` is rendered as 1); an enum its default, else its first value.

**Conditions shown both ways.** The inputs that may change the steps are the bools, the
enums, the optional strs whose default is empty (given, or left empty) and the ints (the
sample, and the first of its examples, 0 and 3 that differs from it and is in bounds).
Every combination of them is tried, up to 4096 (`MAX_GRID`); past that, the sample inputs
and each one or two changes of them (every two inputs' values together, the others at their
sample). The planning command's template (when it declares none) is rendered from the same grid, from
the defaults; a planning command past 4096 logs one warning at load naming its combination count, and
its template's conditions may then read "only for some of the arguments", so declare
`Signature.template` or fewer plan-changing inputs if the planner needs more. An input that never changes the
rendered steps stays at its sample value. The combinations of the others are grouped by the steps they give,
and each different plan is rendered once, titled by the fewest inputs that tell it apart
from the others ("Plan when compare_rights = true", "Plan unless kind = kit", "Plan when
scope = every, people = 3"). The "runs" column is written the same way, so it names the
input that decides a phase rather than every combination it holds for. With more than 16
different plans, the preview renders those that a pairwise cover of the inputs reaches
(every two inputs' values together at least once) and one plan for each other phase
layout, so a plan that needs two inputs changed is still shown; the phases table still
covers every plan. A combination that the `Input`'s model validators change or refuse
never reaches the `PlanGenerator`, so it is listed, not rendered ("every other combination" when all
of them have one reason). In a phased plan, a phase that reads the same as in the first
plan shown is one line ("As in the plan when ...").

**Earlier facts as placeholders.** A phase function is called with placeholder outputs of
the earlier phases, so each value reads as where it comes from: `<find.item_uids[0]>`,
`<item.label>`. A list fact holds one placeholder item; an int fact is 1 and a float
999999 in arithmetic, and each prints as its placeholder. Code that converts a number with
`int()` or `float()` loses the placeholder text and prints 999999 (999,999), a number no
real fact is mistaken for. A str placeholder is never cut by `split` or `partition`, so a
planning command that takes the part of a name before `"_"` keeps the whole placeholder. Facts that
are one value at run time share one placeholder: a uid fact (its source field `*_uid`) is
the first entry of the latest earlier list of those uids (`*_uids`), so code that looks
the uid up in that list finds it and the matched path is rendered. A fan-out phase is
rendered per item: over a list of items, each item (an empty list says the phase runs
nothing); over an earlier phase's list, the first item and one later item (that list's
model holds two placeholder entries), and each item's output takes the item as its uid
and the list's parallel entries (its label). A branch on facts is still rendered along one
path only, which is one more reason to keep conditions on inputs in `__phases__`. A phase whose
function reads earlier facts is marked "Steps below are rendered with placeholder facts:
...; values computed from facts (counts, comparisons, matches) are not real." A fan-out
phase's output reaches later phases as a list of one placeholder per item rendered
(`<items[0].uid>`). When a phase method fails on these sample facts, or writes no steps
with them, the phase is rendered as the template has it (every fact unknown) and the preview
says so. A call that runs flat is rendered as one list of steps under a line saying so. Any `Phase` setting other than `name`, `steps_fn`, `output` and `when` that is not
its default is listed with the phase, and so is a callable `when`; a callable setting is
shown by its source.

**Byte-stable.** The output holds no time and no absolute path; inputs, phases and facts
come in declaration order and combinations in a fixed order, so the same files render the
same bytes. `--check` writes nothing and exits 1 when a preview in `<dir>` differs from a
fresh render, is missing, or is an extra `.md` file no planning command renders, or when a phase
output's facts or a `from_command_output` fail the load (`command_preview.check_previews`). Run it in a test or in CI to keep committed
previews current. Both writing and `--check` fail, naming the file and its error, when a
planning command file does not load (the loader logs and skips it at run time;
`command_preview.render_previews`). Writing removes a stale preview (an `.md` file
whose first line is the generated marker) and keeps any other file, which `--check` then
reports as extra.
