---
name: author-planning-commands
description: >-
  Write fastWorkflow planning commands (`_commands/<name>.py`, or `_commands/<Context>/<name>.py`):
  parameterized multi-command jobs whose steps are written in code and become the agent's numbered
  plan, flat or as phases that pass facts to each other. Covers when to write a planning command
  instead of a command, the file shape (`Signature` + a top-level `PlanGenerator`), `Input` and
  `from_`, phases and facts typed by their commands, the step helpers (`run`, `full_list`, `when`,
  `do_not_run`, `pick`), nesting with its cycle guard and step budget, `__phases__` and `__result__`,
  `in_request`, reuse from `_` modules only, the `command preview` / `--check` loop, the step wording
  the framework reads in code, and descriptions written for people. Use when adding, reviewing or
  debugging a planning command, or when its steps are run, checked or answered differently than
  intended.
---

# Writing planning commands

A planning command is one Python file, `_commands/<name>.py`, or `_commands/<Context>/<name>.py` for
a planning command of a context (named `<Context>/<name>`). It is written like a command file: its
`Signature` describes a job and its `Input` declares the job's typed arguments, exactly as a
command's does. Where a command has a `ResponseGenerator`, a planning command has a top-level
`PlanGenerator`: each public method is a phase that returns its steps as text.

How a matched planning command runs (`workflow_agent.match_planning_command` matches it;
`planning_commands.typed_planning_call` fills it):

1. A planning command is a label of the classifiers whose scope holds it, trained from its
   `plain_utterances`, so a request can reach it as it reaches an execution command. With
   `JEV_API_KEY` the matcher picks the planning command for a sub-task from its description and
   `plain_utterances` first. Without the key the planner finds the command in its list, and no typed
   call fills its inputs.
2. The planner fills `Input` in one typed call.
3. The arguments are checked: `check_arguments` checks each field against its type and its options
   (`in_request` here), and the file's own rules run.
4. A refused call is filled once more with the refusal in view. A second refusal is asked of the
   user as a question when the turn may ask; otherwise the text planner takes over.
5. The call's phases run, and the steps they return become the agent's plan. Code runs the steps it
   can (see the step grammar below).

There is no separate planning step in front of a matched call and no fast path around it.

The one-page guide is `docs/planning_command_file_layout.md` in the fastWorkflow repository (not in
the wheel), with the preview; `docs/planning_command_phases.md` covers phases, facts, code-run steps
and answers. The module documentation of `fastworkflow.planning` (the form),
`fastworkflow.planning_commands` and `fastworkflow.planning_phases` (the runtime) ships with the
package and is the source of truth.

Code references below name a module of the `fastworkflow` package and a symbol in it.

## Planning command or command?

| Write a command (`_commands/<name>.py`) when | Write a planning command (`_commands/<name>.py`) when |
|---|---|
| it is one action on the application (wraps code, returns an `Output`) | a family of requests is answered by the same several commands in a known order (find, open, read, answer) |
| the agent should choose it by intent | the agent keeps getting the order, a value or the answer wrong for that family |
| it needs to read state or call a backend | the job only needs text: `PlanGenerator` methods build text and must not call a backend or read state (`planning_commands` module doc) |

A planning command adds no capability: every step it writes runs an existing command (checked at
load). A one-command planning command is a command with a worse name. Fixed steps with no parameters
are allowed (a `PlanGenerator` with one method taking no `input`): the text planner adapts them. A
planning command with no parameters may also have phases: every call has the one plan, and the
planner's call has no arguments (`planning_commands.py` `planning_command_from_parts`).

## The file

- Name: the file stem is the planning command's name, a Python identifier (`show_item.py`;
  `workflow_agent.py` `_read_planning_command_file`). A `.py` file or folder whose name starts with
  `_` or `.` is not a planning command: `_<name>.py` is a helper module that planning command files
  import (shared lookup tables and phrases, never code that chooses steps); `_split.py` holds the
  split settings. A `<Context>/` folder holds the planning commands of that context
  (`workflow_agent.py` `_planning_command_files`).
- Order (`docs/planning_command_file_layout.md`): module docstring, imports, `Signature`, shared facts
  models, `PlanGenerator` (`__phases__` first, then the phases in run order, then `_helpers`),
  module-level tables and shared functions.
- A file that fails to import or is malformed is logged and skipped at run time (the other
  planning commands still load); `fastworkflow command preview` fails naming it
  (`workflow_agent.py` `_load_planning_command_file`, `command_preview.py` `render_previews`).

A complete phased planning command, for a toy catalog whose commands are `find_item query=`
(Output `item_uids`, `labels`), `open_item_by_uid item_uid=` (Output `label`), `list_parts`
(paged by `start`; Output `labels`) and `count_checks` (Output `total`):

```python
"""show_item: one named item, its parts and, when asked, its open checks.

Finds the item by name, opens the match and reads every part. When the request asks
about checks (`input.checks`), a phase counts them. The last phase writes the answer
from the facts the earlier phases filled.
"""
from pydantic import BaseModel, Field

from fastworkflow.planning import count, pick

from .count_checks import Signature as CountChecks
from .find_item import Signature as FindItem
from .list_parts import Signature as ListParts
from .open_item_by_uid import Signature as OpenItemByUid


class Signature:
    """Show one named item and its parts, and its open checks when asked. One job per item."""

    plain_utterances = ["what parts does the Alpha item have", "show item Beta and its checks"]

    class Input(BaseModel):
        name: str = Field(description="the item's name",
                          json_schema_extra={"in_request": "subject"})
        checks: bool = Field(default=False, description="the request asks about its open checks",
                             json_schema_extra={"verify": True})


class PlanGenerator:
    """find -> item -> checks (when asked) -> report."""

    def __phases__(self, input):
        checks = [self.checks] if input.checks else []
        return [self.find, self.item, *checks, self.report]

    def find(self, input) -> FindItem.Output:
        return [f"find_item query={input.name}"]

    class ItemFacts(BaseModel):
        opened: OpenItemByUid.Output
        parts: ListParts.Output

        @property
        def uid(self) -> str:
            return self.opened.item_uid

        @property
        def label(self) -> str:
            return self.opened.label

        @property
        def part_labels(self) -> list[str]:
            return self.parts.labels

    def item(self, input, find) -> ItemFacts:
        return [pick("open_item_by_uid", item_uid=find.item_uids[0], named=input.name),
                "list_parts (full list)"]

    def checks(self) -> CountChecks.Output:
        return ["count_checks"]

    def report(self, item, checks=None):
        lines = [f"say that {item.label} has the parts: {self._parts(item)}"]
        if checks is not None:
            lines.append(f"say that {item.label} has {checks.total} open checks")
        return lines

    def _parts(self, item):
        return ", ".join(item.part_labels) or "none"
```

The same job as a flat planning command (one executor run; the pick wording lets code open the
match):

```python
class PlanGenerator:
    def item(self, input):
        return [f"find_item query={input.name}",
                f"open_item_by_uid with the uid of the {input.name} match",
                "list_parts (full list)",
                "say which parts the item has"]
```

## `Input` rules

| Rule | Where |
|---|---|
| A field is `str`, `bool`, `int`, `Literal[...]` of text (an enum), or `Optional[str] = None`; any other type or `Optional` fails the load | `planning_commands.py` `_spec_from_field` |
| A `bool` defaults to false; `default=True` fails the load (the model sets it true only when the request asks) | `planning_commands.py` `_parse_param` |
| No default: required. A `str` the request may leave out defaults to `""` or is `Optional[str] = None` (the code then gets `None`); a default like `"NOT_FOUND"` is written into the steps | `planning_commands` module doc; `input_instance` |
| Names `user_query`, `available_commands`, `steps_before`, `steps_after`, `reasoning` are reserved | `planning_commands.py` `RESERVED_NAMES` |
| `in_request` (in `json_schema_extra`) is allowed on a `str`, `int` or `list[str]` field; `"subject"` on a `str` only; see below | `planning_commands.py` `_spec_from_field`, `check_arguments` |
| The retired options `subject`, `means` and `strip` fail the load, naming the new spelling | `planning_commands.py` `RETIRED_OPTIONS` |
| A rule across fields is a pydantic validator on `Input` raising `ValueError` with what to fix; the request is `(info.context or {}).get("request", "")`, `""` when the template is rendered | `planning_commands.py` `check_input` |
| `validate_extracted_parameters(workflow, command, input, command_text=None)` returns `None` to accept, and raises `fastworkflow.Refused(message, field=...)` to refuse a call; a refusal gets the planner one more typed call, a second falls back to the text planner | `planning_commands.py` `_validate_arguments`, `typed_planning_call` |

### Field options (`json_schema_extra`)

The options below are set in a planning command's `Input` field's `json_schema_extra`. They apply
to planning commands' `Input` fields today (`planning_commands.PLANNING_COMMAND_FIELD_KEYS`, read by
`_spec_from_field`). Execution commands get the same options in a later release. Any other key fails
the load.

- `in_request: True` (`str`, `int` or `list[str]`): the value must be in the request. A `str` must
  appear in it exactly as written (a verbatim substring); a `list[str]` keeps each item the request
  writes. An `int` must be a number the request writes (digits, or the words zero to twenty outside the
  other arguments' text); otherwise it takes the only number written, or its default when that one is
  written, and with several candidates the call fails.
- `in_request: {"ignore": ["group", "unit"]}` (`str` only): the named words are removed from either end of
  the value before the check, for example a kind word the model appends to a name ("Payroll group" becomes
  "Payroll"). It implies `in_request`. `in_request: {"subject": True, "ignore": [...]}` is the subject with
  the same words removed.
- `in_request: "subject"` (`str` only): the value names a record of the request (a person, a group, a
  product, a code). It must hold one of the request's subjects whole, or a name-like word, and must not
  cut one. A value cut short at the start or end of exactly one subject becomes that subject ("Ann" of
  "Ann Lee"). An optional argument takes its default when refused; a required one fails the call.
- `verify: True` (`bool`, an enum, or a `str` with `in_request: "subject"`): a classifier decides the
  value again from the request alone, and its answer replaces the planner's when it is sure enough. It
  reads the field's `description`. It needs `JEV_API_KEY`: without the key no classifier is available and
  the check does not run (`planning_commands.check_verdicts` returns nothing when there is no model).
- `verify: {"group": "a group of people", "unit": "an organisation unit"}` (an enum): the same check,
  with each value's meaning, which the classifier reads. Its keys must be values of the enum.

A field with no check is cleaned in code: `strip_words(text, words)` from `fastworkflow.planning_commands`,
called in the field's validator or in `Signature.validate_extracted_parameters`. An `Input` validator's
returned value is written back to the arguments (a refusing validator still refuses). The retired
`subject`, `means` and `strip` options fail the load and name the new spelling.

The code appends the rule sentence of `in_request` (and of `in_request: "subject"`) to the field's description for the
planner (`extraction_sentences`), so the description does not repeat it. Full rules: the
`planning_commands` module doc.

## Phases and facts (typed by commands)

- Each public method of `PlanGenerator` is a phase (`planning.py` `planned`, building the
  runtime's `planning_phases.Phase`), in the order written; `_name` methods are helpers. Each phase
  is a fresh executor run with its own step budget (`PHASE_MAX_STEPS`, default 15), except an
  answer-only phase (one model call, or none). The last phase writes the answer.
- Its return annotation is its facts model. Two forms:
  - a facts model whose fields are each typed with a command's `Signature.Output` (or `Input`),
    `show: ShowBook.Output`, with the other values as properties or methods of the model; a copy
    (`fact_from`, or a `"from"` on a facts field) is not read and fails the load (design 3.12);
  - a command's `Signature.Output` class, a phase "typed by its command": its fields are filled from
    the output of that command in the phase's runs (`typed_phase_facts.py`
    `fill_typed_phase_output`).
  Either way the facts come from the runs of that phase only. A required fact no run holds fails the
  phase, and the job stops with the facts found so far. A model only one phase fills is nested right
  above the method; one that depends on the input is `@phase(output=lambda input: ...)`.
- `from_("<command>.<field>")` on a command's `Input` field names the command that supplies the
  value, so the planner can run it first (`planning_commands.py` `resolve_field_sources`). It
  replaces the hand-written `available_from` hint: a hand-written one is an error that
  `command preview --check` reports (the command stays matchable). See the
  `declare-parameter-producers` skill.
- Its parameters bind by name: `input`, and earlier phases' facts (`planning_phases.py`
  `bound_arguments`). A phase that may have been skipped is read through `=None`. A parameter must
  name an earlier phase with facts in at least one call the inputs give, or the load fails
  (`planning_phases.py` `check_bindings`), so a misspelling is caught.
- `return []` skips the phase. It is always decided when the phase is reached: the runtime calls the
  method once with the real facts (`planning.py` `_phase_functions`: it is the phase's `when`, and
  its steps are kept for the run). A skipped phase passes nothing on.
- The shape of a call: `__phases__(self, input)` returns the phase methods it runs, in order
  (default: all). A call of one phase that reads and fills no facts runs flat, as one plan
  (`planning.py` `_runs_flat`). A phase repeated for several cases is one method each, calling one
  shared module function (a `self._helper` called with a fact is not run in the template, it reads as
  its name).
- `@phase(over="<phase>.<list field or property>")` (or a function of `input` giving the items) fans a phase out,
  one sub-plan per item (at most `FANOUT_MAX_ITEMS`, 10, unless `max_items` says otherwise); its one
  parameter that is neither `input` nor a phase gets the item; `max_items`, `noun`, `bind`
  tune it (`docs/planning_command_phases.md`, "Fan-out phases"). `@phase(optional=True)`: the job goes
  on when it fails.
- The template (what the planner sees; `planning_commands.py` `placeholder_template`) calls each
  method once per input with `planning_phases.Placeholder` facts: `<find.total>`, `<find.uids[0]>`,
  `<find.labels[*]>`; a `self._helper` called on one reads `<helper in words>`; every test on one
  (`bool`, `len`, `==`, `<`, `in`) reads as true and is noted, and the phase's lines end
  `(only when find.uids)`. `x is None` is not seen. Write wording that tests facts (`a or b`) in a
  helper.
- At load (`planning_commands.py` `_checked_plan`, `placeholder_template`, `check_step_commands`;
  `workflow_agent.py` `step_command_problems`): a method without `self`, a bad annotation, a parameter
  that cannot bind, a facts field typed with anything but a command's `Output` or `Input` (a copy
  included), a method that fails on placeholder facts or writes `None` into a step, and a step naming
  a command the workflow lacks or a parameter its `Signature.Input` lacks fail the load.
- A planning command imports from `fastworkflow.planning` what it writes: `from_`,
  `pick`, `run`, `full_list`, `when`, `do_not_run`, `phase`, `num`, `count`, `listed`.

## Step helpers

`fastworkflow.planning` writes steps as strings the framework reads (the grammar below). Plain
strings work too; the helpers only make the wording exact.

| Helper | Writes | What it does |
|---|---|---|
| `run("cmd", k=v)` | `cmd k=v` | A command step code runs once every required value is given. `run` on another planning command is a nested call (see below). |
| `full_list("cmd", k=v)` | `cmd k=v (full list)` | A paged command step code reads through every page, up to `CODE_MAX_PAGES` (20). |
| `when("condition", step)` | `only when condition, step` | A conditional step the agent decides whether to run. |
| `do_not_run("cmd")` | `do not run cmd` | A negated guard: code passes it over and never holds it to a run. |
| `pick("cmd", item_uid=uid, named=name)` | `cmd item_uid=<uid>` | Opens one row. With a concrete uid, the uid is written. |
| `pick("cmd", named=name)` | `cmd with the uid of the <name> match` | Without a uid: code runs `cmd` with the uid of the one row of the listing the step before it read that holds `name` whole (or the one row, when the listing has one). `named=` is required; at most one `*_uid` parameter. |
| `num`, `count`, `listed` | text | How a number, a count or a list reads in a step (`planning.py`). |

## Nesting, the cycle guard and the step budget

- A step naming another planning command runs it as a nested call, with its own phases and its own
  agent, and the nested call's output is the step's run: `run("book_by_title", title=input.title)`.
  Its steps number after the step that called it.
- A planning command cannot call itself, even through others. A call cycle is reported as an error
  that names the cycle ("a planning command cannot call itself, even through others: A -> B -> A",
  `planning_commands.py` `_cycle_problems`), and the commands on it are skipped. At run time, a call
  of a planning command already running in the chain is refused with "is already running; call its
  steps directly."
- The turn-wide step budget is `FW_TURN_MAX_STEPS` (default 400; `plan_frame.py` `turn_max_steps`).
  It counts the model steps the nested planning-command calls of one turn take together, and it
  resets each turn. Once it is spent, a further nested call is refused with "The step budget for
  nested commands is spent; continue with the remaining steps directly." The setting is read from the
  workflow's env file first, then the process environment; a value that is not a positive integer
  logs a warning and the default applies.
- `PHASE_MAX_STEPS` (15) is a different budget: the steps of one phase's executor run.

## Descriptions written for people

- The `Signature` docstring says the job in one or two sentences, as a person would describe it.
  The planner's matcher reads it (its whitespace is collapsed to one line). Make the first
  sentence the plain purpose and put conditions after it: when the intent classifier is unsure,
  the tie-break offers each candidate by its first sentence only (see the `evaluate-intent-routing`
  skill).
- A field's `description` says what the field holds, in words a person would use. The planner reads
  it; so does a `verify` check. Do not repeat the rule sentences the options add (`in_request`).
- `plain_utterances` are example requests in the words people use for that job. Keep them generic,
  and do not copy names from a test workflow.
- Step wording is not written for people: the framework reads it, so use the grammar below.

## `__phases__`, `__result__` and `Signature.Output`

- `__phases__(self, input)` returns the phase methods a call runs, in order. It returns the class's
  own phase methods (`self.find`, ...); a base's method the class overrides is returned as
  `self.<name>`, and the base's is called from it with `super()`.
- A `__call__` method on a `PlanGenerator` is a load error: "PlanGenerator defines __call__; rename
  it to __phases__" (`planning.py` `planned`).
- `__result__(self, input, <phase names>)` builds the call's answer once the phases have run. It
  returns a `str`, a `fastworkflow.CommandResponse`, or the `Signature.Output` model
  (`planning_phases.py` `command_response_from_call_result`). Its parameters name `input` or a phase
  of the call; any other name is a load error.
- A planning command that declares `Signature.Output` must have a `__result__` that builds it: "nothing
  builds it" otherwise. An `Output` field cannot declare `from_`; the value is built in `__result__`.
- If `__result__` raises or returns something else, one model call writes the answer instead
  (`planning_phases.py` `run_call_result`, logged as a warning).

## Reuse: inherit from `_` modules only

- A `PlanGenerator` may subclass a `PlanGenerator` defined in a `_` helper module
  (`from ._lookup import PlanGenerator as Base`). `self` reaches the base's phases, helpers and fact
  models; a method of the same name overrides a base phase and may extend its steps with
  `super().<phase>(...)`. A subclass that only overrides keeps the base's plan; one that adds phases
  writes `__phases__` (`planning.py` `_order_problem`).
- Subclassing another planning command's `PlanGenerator` is a load error (`planning_commands.py`
  `_plan_generator_inherits_foreign`). To reuse its whole job, call it in a step
  (`run("<name>", ...)`); to share phases, move them to a base class in a `_` module.

## The preview loop

```sh
fastworkflow command preview <workflow_folder>            # writes <workflow_folder>/../docs/planning_command_previews/<name>.md
fastworkflow command preview <workflow_folder> --check    # writes nothing; exit 1 when not current
```

1. Write or change the planning command file.
2. Run `command preview`. A load error names the file and its reason; fix it first.
3. Read the preview: the inputs, the phases table ("runs" says when each phase runs), the facts with
   their sources, and the steps of each different plan. A step rendered with `<phase.fact>`
   placeholders shows where a value comes from at run time. Under the steps, "Pages: step 2 reads up
   to 20 pages; step 3 reads page 1 only." says how code pages each step on a paged command
   (`planning_phases.step_paging`): only a note with a full-list phrase (`FULL_LIST_PHRASES`: "full
   list", "every page", ...) reads later pages, up to `CODE_MAX_PAGES` (20); "(all of them)" or no
   note reads page 1. A step that writes `start=` reads that page and is not listed.
4. Check each step against the grammar table below: is it run by code, held by the finish check,
   written into the answer as intended?
5. Commit the previews and run `--check` in CI (`command_preview.py` `check_previews`). It fails on a
   stale, missing or extra preview; a planning command file that does not load (a step naming an
   unknown command or parameter included); a phase fact source the workflow does not have; and a
   `from_` error reported by `planning_commands.resolve_field_sources`.

## Step grammar

Each step is one string, one line of the plan. The framework reads it in code: the finish check
(`observation_offloading/finish_check.py`), code-run steps and picks, and the answer-only phase
(`planning_phases.py`). A step is classified by the FIRST declared command it names; the text before
that command decides conditional, negated and answer steps (a condition word in its chain also makes
it conditional).

| Wording | What it triggers | Implemented at |
|---|---|---|
| `cmd k=v k2=v2`, the command the first word, every required field given, values that bind | Mechanical: code runs `cmd k=v k2=v2` before the model takes a step | `planning_phases.py` `_code_step`, `plan_code_steps` |
| `cmd` alone, every field defaulted (or no parameters) | Mechanical: code sends the bare command | `planning_phases.py` `_code_step` |
| `cmd k=v (note)`, `cmd k=v; note` | The chain ends at the first `(` or `;`; the note is never sent and never changes what runs | `planning_phases.py` `_code_step`; `finish_check.py` `step_arguments` |
| a concrete value `k=Alpha` | In a phase, the finish check and the fact fill accept only a run with that value (spacing, outer quotes, case ignored) | `finish_check.py` `step_arguments`; `planning_command_phases.md` "Facts" |
| `k=<the uid>`, `k={x}`, `k=[..]`, `k=...` | A placeholder, not a value: the step is open (the agent fills it); any run of the command with arguments satisfies it | `finish_check.py` `_PLACEHOLDER_RE` |
| `k="Acme (EU)"`, `k='Sales; West'` | A value in quotes right after `=` is kept whole: its `(`, `;` and `, ` do not end it (unquoted, `k=Acme (EU)` sends `k=Acme`) | `finish_check.py` `step_chain`, `_value_text` |
| `k=Ann Lee, no filter` | A value ends at `", "`; prose swallowed into it leaves the step open | `finish_check.py` `_value_text`; `planning_phases.py` `_code_step` ("prose in value") |
| a value whose first token is an id (16+ letters, digits, `_`, `-`; 3+ digits) | The value is that token alone, cut at the first space or comma | `finish_check.py` `_UID_TOKEN_RE` |
| `cmd` then prose not in parentheses (`go_up back to the list`), or the command not the first word (`then cmd k=v`) | Open: the agent runs it | `planning_phases.py` `_code_step` |
| a required field left out (`find_item` with `query` required) | Open | `planning_phases.py` `_code_step` |
| `open_x with the uid of the Alpha match`, right after a step code ran (`find_x query=Alpha`) | Pick: code sends `open_x x_uid=<uid>` when exactly one row of that run's Output holds `Alpha` whole, or it has one row; the one unwritten required field must be a uid (`uid`, `id`, `*_uid`, `*_id`). A uid containing `(` or `;` is quoted | `planning_phases.py` `pick_step`, `_is_uid_field`; `planning.py` `pick` |
| `cmd x_uid=<the Alpha uid> other=v` after the lookup | Pick with the other values sent as written; the uid's must be the only placeholder | `planning_phases.py` `pick_step` |
| `cmd (full list)`; also `every page`, `all pages`, `until complete`, `page through` (any case) in the note | Code reads page 1, then each later page the output names, as more code-run steps, up to `CODE_MAX_PAGES` (20) pages; then code goes on | `planning_phases.py` `FULL_LIST_PHRASES`, `_paging`, `_next_page`, `CODE_MAX_PAGES` |
| `cmd start=20` (`start`) | That page only; code never computes a next page | `facts.py` `PAGING_PARAMETERS` |
| `cmd start=<n>`, or a step `next page with start=<n> until the list is read` | Open: the agent pages | `planning_phases.py` `_code_step` |
| any other note, or none, on a paged command | Page 1 is what the step wants; code goes on | `planning_phases.py` `_paging` |
| a condition word before the command (or in its chain, next row): `if`, `only`, `when`, `whenever`, `unless`, `otherwise`, `optional(ly)`, `in case`, `as needed`, `where needed`, `where applicable`, `else` | Conditional: the agent decides; the finish check does not hold it; it stops code and keeps the phase from being answer-only | `finish_check.py` `_CONDITION_RE`, `_classify` |
| ANYWHERE in the step: `if needed`, `if necessary`, `if required`, `as needed`, `optional(ly)`, `skip when/if/unless`, `skip it when/if/unless` | Conditional, as above | `finish_check.py` `_ANYWHERE_CONDITION_RE` |
| `cmd k=v only when ...` (a condition word in the command's chain: after it, up to the first `(` or `;`) | Conditional, as if the condition came before the command, so the prose never joins the last value. A condition in a note (`cmd k=v (only when overdue)`) is never read: the step stays a plain command step, held to a run | `finish_check.py` `_classify` |
| a negation word before the command: `do not`, `don't`, `dont`, `never`, `none`, `not`, `no`, `nor`, `without`, `avoid`, `skip` | Negated (checked before conditions: `only when there is no match, cmd ...` is negated): code passes it over, never held to a run | `finish_check.py` `_NEGATION_RE`, `_classify` |
| first word an answer verb: `answer report present return say give summarize summarise state tell reply respond compare note identify end stop finish confirm verify conclude offer mention describe include combine format output provide write` (and `list_nothing`, `explain_answer`) | An answer step, even if it names a command later (`say what find_item found`): code passes it over, never held to a run | `finish_check.py` `_ANSWER_VERBS` |
| a step naming no command, no answer verb, no condition word (`read the notes`, `list the parts`) | "Other": not held to a run, but it stops code (an open step) | `finish_check.py` `_classify`; `planning_phases.py` `plan_code_steps` |
| `go_up` or `reset_context` alone | Code runs it; with prose after it, open | `planning_phases.py` `_CODE_NAVIGATION`, `plan_code_steps` |
| a phase whose steps hold no command, conditional or navigation step | Answer-only: no executor run, one model call with no tools (or none, below) | `planning_phases.py` `is_answer_only` |
| (answer-only) `say that P` | Verbatim, the only form code copies: code writes `P.` | `planning_phases.py` `answer_step_kind`, `_answer_line` |
| (answer-only) a head and its facts (`say which is riskier: A 3, B 5`, `say what X holds: a, b`, `compare ...: facts`) | Judgement, like any step but `say that`: only the author knows whether the facts already answer the head. To have code copy facts, write `say that what X holds: a, b` (`What X holds: a, b.`) | `planning_phases.py` `answer_step_kind` |
| (answer-only) no `": "` facts, starting `apply`, `do not`, `don't`, `never`, `make no`, `present them as options only`, `present it as options only`, AND a negation word (`no`, `not`, `none`, `never`, ...) | Guard: adds nothing; dropped from a code-written answer. Without a negation word (`apply a 10% discount to the total`) it is judgement | `planning_phases.py` `ANSWER_STEP_WORDS`; `finish_check.py` `_NEGATION_RE` |
| (answer-only) anything else (`say which is riskier`, `list the parts: a, b`) | Judgement: ONE such step and the model writes the whole answer | `planning_phases.py` `answer_by_code` |
| a step with `name`, `names`, `naming`, `list`, `lists`, `listing`, `every` or `each` outside the items, holding every item of a list fact (2+ items) | That list is checked: rows the answer leaves out are appended after "Also in the facts, not named above:"; the list is never cut | `planning_phases.py` `_NAMING_RE`, `_names_list_in_full`, `missing_list_items` |
| a list fact of more than `FACTS_MAX_LIST_ITEMS` (100) items joined by `", "`, `"; "`, `" \| "`, `","` or `";"` in a step with no naming word | Cut in the prompt to its first items and "(and K more; M in all)" | `planning_phases.py` `_LIST_SEPARATORS`, `capped_steps` |
| 4 or more items in a `", "` run of a code-written answer, each exactly an output value | Laid out as sub-bullets | `planning_phases.py` `MIN_BULLETED_ITEMS`, `bulleted_answer` |
| an empty string | Dropped | `planning_phases.py` `_checked_steps`; `planning_commands.py` `_call_steps` |
| a step whose first word is snake_case (or a known command): its command and the `key=` of its chain (before `(`, `;` or `,`) | Checked at load against the workflow's commands and their `Signature.Input`; an unknown one fails the load | `planning_commands.py` `step_commands`, `check_step_commands` |

Consequences worth designing for:

- Code runs only the LEADING mechanical steps and stops at the first open step. Put the steps that
  need judgement last, or in a phase of their own.
- Write a filter as an argument (`list_parts kind=gear`), never as a note: the note is not sent.
- Write an answer step's facts as the answer should read them: in a verbatim step every word after
  `say that` is copied, instructions included.
- Make the last phase answer-only when its steps can carry every fact, so code (or one call without
  tools) writes the answer; keep a judgement step only where a judgement is needed.
- `repeat()` and `@<skill>` include lines are gone: loop over `range()` bounded by an int field's
  `le`, and share steps through a helper module.

## Approval annotation (`approval`)

A planning command (and an execution command) may declare `approval = "mandatory"` or
`approval = "delegatable"` on its `PlanGenerator` (or `ResponseGenerator`). Use `mandatory` for an
action that cannot be undone, such as sending a critical email; `delegatable` when the user may
hand the decision over. Leave it out for everything else: the agent decides, with no gate. Any
other value, or a non-string value, fails the load and names the command and its file.

The annotation is enforced. `mandatory` asks before every run, including a run the user typed;
`delegatable` asks until a standing approval exists. The reply is `yes`, `no` or `always`, bound to
the exact parameters, single-use, and it expires. The listings mark the command, and
`planning_steps_needing_approval` lists the annotated commands a planning command's template calls.
It is not read/write gating: an easy-to-undo write needs no annotation. Full rules are in
`docs/planning_command_file_layout.md`.

A planning command runs in agent mode. In assistant mode an exactly named planning command runs only
when it is mechanical (code runs every step). One that needs the agent is refused, with the reasons
that name each step.

## Checklist

```
- [ ] The job needs several existing commands in a known order (else: a command)
- [ ] File stem is the planning command's name; the Signature docstring says the job; plain_utterances are generic
- [ ] Input types are str / bool / int / Literal / Optional[str] = None; bools default false
- [ ] Signature + top-level PlanGenerator; each phase a method with self, in run order; helpers _named
- [ ] Every fact field is typed with a command's Output or Input (or the phase returns a command's Output)
- [ ] Every Input handle that another command supplies carries from_ (not available_from)
- [ ] Input-dependent shape in __phases__; a skip is `if not <what it needs>: return []`;
      a phase that may be skipped is read through =None
- [ ] No __call__; a Signature.Output has a __result__ that builds it
- [ ] Inherits only from a PlanGenerator in a _ module; no nested call cycle
- [ ] The template (preview) reads well: wording that tests facts lives in helpers
- [ ] Each step checked against the grammar table (code-run, held, answered as intended)
- [ ] command preview written and committed; --check passes in CI
```

## Related

- Reading a phased planning command call in a trace (`fw.skill.phase` spans, `code_stop`,
  `answer_by_code`, `facts_error`): the `debug-workflow-conversations` skill.
- The commands a planning command's steps name, and their `Output` fields its facts copy: the
  `integrate-chat-agent` skill's reference for command files.
- The `from_` producer hints on `Input` fields, and their rules: the `declare-parameter-producers`
  skill.
- Whether a planning command change helped, measured on a fixed task set: the
  `optimize-workflow-with-feedback` skill.
