# Skill file layout

A skill file (`_skills/<name>.py`; reference: the `fastworkflow.skill` module documentation;
the runtime: [skill_phases.md](skill_phases.md)) is a command file for a whole job. It is read
far more often than it is written, so every skill file has one form, the same few idioms as a
command file, read top to bottom in the same order. `fastworkflow skill preview` (below)
renders what each call of it runs. A step-by-step tutorial, on a toy workflow:
[tutorials/skills](tutorials/skills/README.md).

## How to write a skill (one page)

```python
from fastworkflow.skill import fact_from, phase, pick

class Signature:                      # exactly a command's Signature
    """What the skill does (the planner chooses skills by this)."""
    plain_utterances = ["what parts does the Alpha item have"]
    class Input(BaseModel):
        name: str = Field(description="the item's name exactly as the request writes it")

class PlanGenerator:                  # a command generates a response; a skill, a plan
    def find(self, input) -> Match:                       # phase 1; it fills a Match
        return ["open_catalog",
                f"find_item query={input.name}"]

    class Item(BaseModel):                                # its facts, like an Output
        uid: str = fact_from("open_item_by_uid.item_uid")
        parts: list[str] = fact_from("list_parts.labels")

    def item(self, input, find) -> Item:                  # phase 2 reads find's facts
        if not find.uids:
            return []                                     # skipped, decided when reached
        return [pick("open_item_by_uid", item_uid=find.only_uid(input.name),
                     named=input.name),
                "list_parts (full list)"]

    def answer(self, input, item=None):                   # None: item was skipped
        if item is None:
            return [f"say that no item named {input.name} was found", "apply no change"]
        return [f"say that {item.uid} has parts {self._parts(item)}", "apply no change"]

    def _parts(self, item):                               # _name: a helper, not a phase
        return ", ".join(item.parts) or "none"
```

**The rules.** (1) Each public method of `PlanGenerator` is a phase (one executor run), in
the order written. (2) Its parameters name what it reads: `input`, and earlier phases by name
(their facts; `=None` when that phase may have been skipped). (3) Its return annotation is the
facts model it fills: each field says its source, `fact_from("command.field")`. (4) It
returns its steps, exactly the lines the agent reads. `return []` skips it: decided when the
phase is reached, on the real facts. A skill with no `Input` fields may have phases too
(methods without `input`; every call has the one plan). A `PlanGenerator` that subclasses another skill's
inherits its phases, first and in their order; a method of the same name overrides one in
its place, and its new phases follow.

**Steps.** `command k=v (note)` runs a command: code runs it when every value is written, the
note is never sent, a listing read whole says `full list`. `pick(command, x_uid=uid,
named=name)` opens the row a find listed. `say that ...` is an answer line code copies word
for word; `apply no change` and `do not run X` are guards; any other line is the agent's. A
last phase of only `say that` lines and guards is answered by code. Text helpers: `num(x)`,
`count(n, "login")`, `listed(names, total, "group")`.

**The shape of a call.** By default a call runs every phase. When the input changes it, write
`def __call__(self, input)` returning the phases, in order: `return [self.find, self.answer]`
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

**Checked at load** (else the skill is not loaded): a method without `self`, a parameter
that names no earlier phase, an annotation that is not a model, a method that fails on
unknown facts or writes `None` into a step, a step naming a command the workflow lacks or a
parameter its `Signature.Input` lacks. `fastworkflow skill preview <workflow>` shows every
call's phases, facts, steps and who writes the answer; `--check` fails when one is stale.

## The order

Every skill file has these parts, in this order: the module docstring (the job, what it
reads, what it answers, the conditions that change it, in two to five sentences); the
imports; `Signature`; the facts models shared by several phases (a model one phase fills is
nested right above that method); `PlanGenerator`, its `__call__` first when it has one, then
the phases in run order, then its `_helpers`; then module-level tables and shared functions.
A reader meets what the skill does, what it takes, how a call is shaped, then each phase with
its facts, then the wording. Lookup tables and phrases shared by several skills go in a helper
module (`_skills/_<name>.py`, data only); everything that decides which steps a skill writes
stays in the skill file.

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
   make a call run flat, one line above the table says when it runs as phases, and the
   table is about the phased calls. A phase's condition reads "only when <the tests its
   method made on the facts>" (`find.uids`); a callable option (`until`) reads as its source;
   the "fans out" column gives `over`, `item`, `max_items`, `until` (by its source) and
   `bind=False`. A cell that differs by input (the output model, the `when`, the fan-out)
   lists each value with the inputs that give it ("when kind = crate: `CrateMatch`");
5. the steps for the sample inputs: phase by phase for a phased skill, the `steps()` list
   for a flat one. An answer-only phase (its steps name no command, read with the
   workflow's command names as a run reads them) ends with who writes its answer, decided
   by the same wording rules a run uses (`skill_phases.answer_by_code`,
   `answer_step_kind`): "Answer: written by code, word for word (steps 1-2 verbatim; step
   3 dropped as a guard)." or "Answer: written by the model (step 2 needs judgement)." A
   phase that runs commands has no such line: the agent writes its answer. Steps on a
   paged command (a paging parameter such as `start`) that write no paging value are
   followed by how many pages each reads, by the rule code pages by
   (`skill_phases.step_paging`): "Pages: step 2 reads up to 20 pages; steps 4 and 5
   read page 1 only." Only a note that asks for the full list (`list_parts (full list)`,
   see `FULL_LIST_PHRASES` in `docs/skill_phases.md`) reads later pages, which code reads
   up to `skill_phases.CODE_MAX_PAGES` (20); any other note
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
never reaches the `PlanGenerator`, so it is listed, not rendered ("every other combination" when all
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
path only, which is one more reason to keep conditions on inputs in `__call__`. A phase whose
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
fresh render, is missing, or is an extra `.md` file no skill renders, or when a phase
output's `"from"` names a command or field the workflow does not have (at load only a
warning; `skill_preview.check_previews`). Run it in a test or in CI to keep committed
previews current. Both writing and `--check` fail, naming the file and its error, when a
skill file does not load (the loader logs and skips it at run time;
`skill_preview.render_previews`). Writing removes a stale preview (an `.md` file
whose first line is the generated marker) and keeps any other file, which `--check` then
reports as extra.
