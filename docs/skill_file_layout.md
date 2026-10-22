# Skill file layout

A skill command file (`_skills/<name>.py`; format: the `fastworkflow.skill_commands` module
documentation; phases: [skill_phases.md](skill_phases.md)) is read far more often than it is
written: by the developer who changes it, by a reviewer, and by whoever debugs a turn that
ran it. This convention fixes where each part of the file goes and how it is written, so
that any skill file reads top to bottom in the same order. `fastworkflow skill preview`
(below) renders each skill in the same order, from the code.

The convention is about layout only. It changes nothing the loader reads: a file laid out
this way loads and expands exactly as before.

## The order

Every skill file has these parts, in this order:

1. **Module docstring.** What the skill does, in two to five sentences: the job, what it
   reads, what it answers, and the conditions that change it. Not the history of the file.
2. **Imports.**
3. **`Signature`.** The description (its docstring, which the matcher and the planner read),
   `plain_utterances`, `Input` with its fields (and its validators, below the fields), and
   `validate_extracted_parameters` when there is one.
4. **Fact models** (a phased skill only). The `output` models of the phases, in the order
   of the phases that fill them, each field written `fact_from("<command>.<field>")`.
5. **`plan(input)` or `steps(input)`.** The table of contents. A phased skill has
   `plan()`, listing the phases, and no `steps()`: the loader gives it
   `concatenated_steps(plan(input))` (`skill_commands.command_from_module`). It writes
   `steps()` only when `plan()` returns `None` for some inputs (the flat steps of those
   calls), right after `plan()`. A flat skill has only `steps()`, written as a list of
   calls to its step functions.
6. **The step functions** (or phase functions), in the order `plan()` / `steps()` calls
   them. Each returns its steps.

Why this order: a reader meets the file in the order of the questions they ask. What does
it do (docstring)? What does it take (`Signature`)? What passes from one part to the next
(fact models: the data contract between phases, and the types the step functions are
annotated with)? In what order, and under which conditions, does it run (`plan()`)? Only
then the wording of each part. The table of contents sits above the functions it names
because a reader wants the outline before the detail; Python resolves those names when
`plan()` is called, not when it is defined, so this order costs nothing. The fact models
sit above `plan()` and the functions because both refer to them, so a reader never meets a
type before its definition.

## Naming

- **Phases**: short snake_case names that say what the phase yields or does: `find`,
  `item`, `check`, `report`. A later phase receives an earlier phase's output through a
  parameter of the same name, so name a phase for the fact it hands on (`item` hands on
  `ItemFacts`, read as `item.uid`). Phases that repeat a job for several inputs share a
  stem and add a suffix: `find_a`, `find_b`.
- **Phase functions**: `<phase>_steps` (`find_steps`, `item_steps`). One function shared
  by several phases is named for their stem (`find_steps` for `find_a` and `find_b`).
- **Step functions of a flat skill**: `<part>_steps`, one per part of the job
  (`lookup_steps`, `detail_steps`, `close_steps`).
- **Fact models**: CamelCase nouns ending in what they hold: `ItemMatch` (a search's hits),
  `ItemFacts` (a record as read).
- **Parameters of a phase function**: the input values it prints (`name`), then the
  earlier phases it reads, by their phase names, each `Optional[...] = None`: the
  function gets `None` when the flat steps and the template are built
  (`concatenated_steps` passes `None` for every parameter without a default) and when a
  phase it reads was skipped at run time. Which earlier phases a phase reads in a given
  plan is decided in `plan()` (below), not by these defaults.

## Writing the steps

- Each step function returns a **list literal of plain strings or f-strings**, one step
  per line, so the steps read as they will be numbered. Build a step in one place: not by
  concatenating fragments returned by other functions.
- A phrase several steps share may be a module-level `UPPER_CASE` constant, placed directly
  above the first function that uses it. A lookup table (`FIND = {"group": "find_group",
  ...}`) is a module-level dict placed the same way.
- A phase function called with no facts (every fact parameter `None`) returns the generic
  wording: that is what the flat steps and the template show, and what the planner sees. With
  facts it names the real values. Write both branches next to each other, the no-facts
  branch first.

## No hidden logic

- Everything that decides which steps a skill writes is in the skill file. A reader must
  never need to open another module to know what a step says or when it is written.
- A helper module (`_skills/_<name>.py`) holds data shared by several skills (lookup
  tables, shared phrases), never code that builds or chooses steps.
- A computation on facts (which of two items scores higher, what two lists share) is
  either written in the phase function that uses it, or is a short method of the fact model
  it reads. Name it for its result (`riskier`, `shared_labels`) and give it a one-line
  docstring.

## Conditions visible in `plan()`

- A condition on an input is written in `plan()` (or in a flat skill's `steps()`), not
  inside a step function: `Phase("check", ..., when=input.check)`, or an `if input.check:`
  block in `plan()`. A step function takes the values it prints, not the flags that decide
  whether it runs.
- When a later phase is built with or without the output of a phase the input leaves
  out, write both `Phase(...)` calls in `plan()`, under the same `if` as that phase, so
  the reader sees both shapes of the plan. Only a phase skipped at run time (a callable
  `when`, an `optional` phase) is read through a `None` default in the `steps_fn`
  (`lambda item, checks=None: ...`).
- A branch on facts, which exist only at run time (a search with no single match, two
  values that tie), lives in the phase function, at its top, with a comment saying what
  case it handles.
- Any other setting of a phase is given at the `Phase(...)` call in `plan()`, never
  computed in a function elsewhere, so the table of contents shows how every phase runs.
  This holds for settings that later versions of `Phase` add.

## Rules on the arguments

- A rule that ties fields together, or a field to the request, is a pydantic validator on
  `Input` (`model_validator(mode="after")`, or `field_validator`), written below the fields
  in `Signature`. It raises `ValueError` with a message that tells the planner what to fix,
  naming the field and the value that would fit: `"record_a and record_b are both 'Alpha';
  record_b is the other record the request names"`.
- The validator reads the request (and the workflow) from the validation context: declare
  `info: ValidationInfo` and read `(info.context or {}).get("request", "")`. The request is
  `""` when the template is rendered, so a rule on the request is silent there.
- Refuse only on a clear conflict. A refusal is shown to the planner, which fills the call
  once more; a second refusal drops the skill call and the planner plans from the template.
- `validate_extracted_parameters` stays for normalising an argument that only means
  something with others (resetting it to its default); its `(False, message)` is a refusal
  as above.

```python
class Input(BaseModel):
    first: str = Field(description="the first item's name exactly as the request writes it")
    second: str = Field(description="the second item's name exactly as the request writes it")

    @model_validator(mode="after")
    def _two_items(self):
        if self.first.casefold() == self.second.casefold():
            raise ValueError(f"first and second are both {self.first!r}; second is the other "
                             "item the request names")
        return self
```

## Example

A phased skill with a conditional phase:

```python
"""show_item: one named item, its parts and, when asked, its open checks.

Finds the item by name, opens the first match and reads its parts. When the request asks
about checks (`input.checks`), a phase counts them. The last phase writes the answer from
the facts the earlier phases filled.
"""
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase, fact_from


class Signature:
    """Show one named item and its parts, and its open checks when asked. One job per item."""

    plain_utterances = ["what parts does the Alpha item have", "show item Beta and its checks"]

    class Input(BaseModel):
        name: str = Field(description="the item's name exactly as the request writes it",
                          json_schema_extra={"subject": True})
        checks: bool = Field(default=False, description="the request asks about its open checks",
                             json_schema_extra={"verify": True})


class ItemMatch(BaseModel):
    item_uids: list[str] = fact_from("find_item.item_uids", min_length=1)


class ItemFacts(BaseModel):
    uid: str = fact_from("open_item_by_uid.item_uid")
    part_labels: list[str] = fact_from("list_parts.labels")


class CheckFacts(BaseModel):
    total: int = fact_from("count_checks.total")


def plan(input: Signature.Input) -> list[Phase]:
    phases = [
        Phase("find", lambda: find_steps(input.name), output=ItemMatch),
        Phase("item", lambda find: item_steps(input.name, find), output=ItemFacts),
        Phase("checks", lambda item: checks_steps(item), output=CheckFacts, when=input.checks),
    ]
    if input.checks:
        phases.append(Phase("report", lambda item, checks: report_steps(item, checks)))
    else:
        phases.append(Phase("report", lambda item: report_steps(item)))
    return phases


def find_steps(name):
    return ["open_catalog", f"find_item query={name}"]


def item_steps(name, find: Optional[ItemMatch] = None):
    uid = find.item_uids[0] if find else f"<the uid of the {name} match>"
    return [f"open_item_by_uid item_uid={uid}", "list_parts (full list)"]


def checks_steps(item: Optional[ItemFacts] = None):
    return [f"count_checks on {item.uid if item else 'the item'}"]


def report_steps(item: Optional[ItemFacts] = None, checks: Optional[CheckFacts] = None):
    if item is None:
        return ["say which parts the item has, and how many open checks when they were counted"]
    counted = f"; open checks: {checks.total}" if checks else ""
    return [f"say that {item.uid} has the parts {', '.join(item.part_labels) or 'none'}{counted}"]
```

A flat skill follows the same order without fact models and `plan()`: docstring,
`Signature`, `steps()` as the list of parts with its conditions, then the step functions:

```python
def steps(input: Signature.Input) -> list[str]:
    out = lookup_steps(input.name)
    if input.detail:
        out += detail_steps()
    return out + close_steps(input.name)
```

## Previews

`fastworkflow skill preview <workflow_folder> [--out <dir>] [--check]` writes one markdown
file per skill, `<dir>/<skill>.md`, generated from the skill files with the skill loader
(`fastworkflow.skill_preview`). The default `<dir>` is `docs/skill_previews` next to the
workflow folder (`<workflow_folder>/../docs/skill_previews`), which for a workflow package
inside its project is the project's `docs/skill_previews`.

A preview follows the order above:

1. the skill's name, its description and its kind (flat, phased, or fixed steps without
   inputs);
2. the inputs table: name, type, default, the sample value it is rendered with, the
   `verify` and `subject` flags and the description, then the other options (`strip`,
   `means`, `in_request`);
3. the example requests (`plain_utterances`);
4. for a phased skill, the phases (output model, when each runs, and how it fans out)
   and the facts each output model holds, with their `"from"` sources. When some inputs
   make `plan()` return no phases, one line above the table says when it returns phases,
   and the table is about the phased calls. A callable `when` reads "only when <its
   source>": a lambda's own text, on one line, with the closure variables, `input.<field>`
   reads and parameter defaults it uses written as their values when they are short and
   plain (`find.pick('<right>')`, not `find.pick(right_name)`), else the function's name;
   the "fans out" column gives `over`, `item`, `max_items`, `until` (by its source) and
   `bind=False`. A cell that differs by input (the output model, the `when`, the fan-out)
   lists each value with the inputs that give it ("when kind = crate: `CrateMatch`");
5. the steps for the sample inputs: phase by phase for a phased skill, the `steps()` list
   for a flat one.

**Sample inputs** come from the `Input` model: a str takes its default, else its first
`examples` value (`Field(examples=[...])`), else the placeholder `<name>`; a bool is false;
an int its default, else its first example, else 1, moved into its `ge`/`le` bounds (a
default of 0 under `ge=1` is rendered as 1); an enum its default, else its first value.

**Conditions shown both ways.** The inputs that may change the steps are the bools, the
enums, the optional strs whose default is empty (given, or left empty) and the ints (the
sample, and the first of its examples, 0 and 3 that differs from it and is in bounds).
Every combination of them is tried, up to 4096 (`MAX_GRID`); past that, the sample inputs
and each one or two changes of them (every two inputs' values together, the others at their
sample). The skill's template (when it declares none) is rendered from the same grid, from
the defaults; a skill past 4096 logs one warning at load naming its combination count, and
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
covers every plan. A combination that `validate_extracted_parameters` changes or refuses
never reaches `steps()`, so it is listed, not rendered ("every other combination" when all
of them have one reason). In a phased plan, a phase that reads the same as in the first
plan shown is one line ("As in the plan when ...").

**Earlier facts as placeholders.** A phase function is called with placeholder outputs of
the earlier phases, so each value reads as where it comes from: `<find.item_uids[0]>`,
`<item.label>`. A list fact holds one placeholder item; an int fact is 1 and a float
999999 in arithmetic, and each prints as its placeholder. Code that converts a number with
`int()` or `float()` loses the placeholder text and prints 999999 (999,999), a number no
real fact is mistaken for. A str placeholder is never cut by `split` or `partition`, so a
skill that takes the part of a name before `"_"` keeps the whole placeholder. Facts that
are one value at run time share one placeholder: a uid fact (its source field `*_uid`) is
the first entry of the latest earlier list of those uids (`*_uids`), so code that looks
the uid up in that list finds it and the matched path is rendered. A fan-out phase is
rendered per item: over a list of items, each item (an empty list says the phase runs
nothing); over an earlier phase's list, the first item and one later item (that list's
model holds two placeholder entries), and each item's output takes the item as its uid
and the list's parallel entries (its label). A branch on facts is still rendered along one
path only, which is one more reason to keep conditions on inputs in `plan()`. A phase whose
function reads earlier facts is marked "Steps below are rendered with placeholder facts:
...; values computed from facts (counts, comparisons, matches) are not real." A fan-out
phase's output reaches later phases as a list of one placeholder per item rendered
(`<items[0].uid>`). When a phase function fails on placeholders, the phase is rendered
with no earlier facts (its generic wording) and the preview says so. For inputs `plan()`
returns no phases for, the flat `steps()` are rendered under a line saying the call runs
flat. Any `Phase` setting other than `name`, `steps_fn`, `output` and `when` that is not
its default is listed with the phase, and so is a callable `when`; a callable setting is
shown by its source.

**Byte-stable.** The output holds no time and no absolute path; inputs, phases and facts
come in declaration order and combinations in a fixed order, so the same files render the
same bytes. `--check` writes nothing and exits 1 when a preview in `<dir>` differs from a
fresh render, is missing, or is an extra `.md` file no skill renders, or when a phase
output's `"from"` names a command or field the workflow does not have (at load only a
warning; `skill_preview.check_previews`); run it in a test or in CI to keep committed
previews current. Both writing and `--check` fail, naming the file and its error, when a
skill file does not load (the loader logs and skips it at run time;
`skill_preview.render_previews`). Writing removes a stale preview (an `.md` file
whose first line is the generated marker) and keeps any other file, which `--check` then
reports as extra.
