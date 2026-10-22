---
name: author-workflow-skills
description: >-
  Write fastWorkflow skill files (`_skills/<name>.py`): parameterized multi-command jobs whose
  steps are built in code and expanded into the agent's numbered plan, flat or as phases that
  pass facts to each other. Covers when to write a skill instead of a command, the file shape,
  the `Input` rules, `plan()`/`Phase`/`fact_from`, `plan()` returning None, the `skill preview`
  / `--check` loop, and the step wording the framework reads in code (what code runs, what the
  finish check holds, what writes the answer). Use when adding, reviewing or debugging a skill
  file, or when a skill's steps are run, checked or answered differently than intended.
---

# Writing workflow skills

A skill is one Python file, `_skills/<name>.py`, next to `_commands/`. Its `Signature`
describes a job; its `Input` declares the job's typed arguments; `steps(input)` (flat) or
`plan(input)` (phases) builds the job's steps as text. At run time the skill matcher picks
the skill for a request (`workflow_agent.match_skill`, from the description and
`plain_utterances`), the planner fills `Input` in one typed call
(`skill_commands.plan_skill_call`), the arguments are checked, and the steps become the
agent's numbered plan.

The full reference is three documents in the fastworkflow repository (not in the wheel):
`docs/skill_file_layout.md` (layout and the preview), `docs/skill_phases.md` (phases, facts,
code-run steps, answers) and `docs/skill_phases_code_run_steps.md` (the design of code-run
steps). The module documentation of `fastworkflow.skill_commands` and
`fastworkflow.skill_phases` ships with the package and is the format's source of truth.

Code references below name a module of the `fastworkflow` package and a symbol in it.

## Skill or command?

| Write a command (`_commands/<name>.py`) when | Write a skill (`_skills/<name>.py`) when |
|---|---|
| it is one action on the application (wraps code, returns an `Output`) | a family of requests is answered by the same several commands in a known order (find, open, read, answer) |
| the agent should choose it by intent | the agent keeps getting the order, a value or the answer wrong for that family |
| it needs to read state or call a backend | the job only needs text: `steps()` and `plan()` build text and must not call a backend or read state (`skill_commands` module doc) |

A skill adds no capability: every step it writes runs an existing command. A one-command
skill is a command with a worse name. Fixed steps with no parameters are allowed
(`steps()` with no argument): the text planner adapts them as a template.

## The file

- Name: the file stem is the skill's name, a Python identifier (`show_item.py`;
  `workflow_agent.py` `_read_skill_file`). `_<name>.py` is a helper module skill files
  import (shared lookup tables and phrases, never code that chooses steps); `_split.py` is
  the split settings; subfolders are skipped (`workflow_agent.py` `_skill_files`).
- Order (`docs/skill_file_layout.md`): module docstring, imports, `Signature`, fact models,
  `plan()` or `steps()`, then the step functions in the order they are called.
- A phased file has `plan()` and no `steps()`: the loader gives it
  `concatenated_steps(plan(input))` (`skill_commands.py` `command_from_module`,
  `_concatenated_steps_fn`). Write `steps()` only for a flat skill, or when `plan()` returns
  `None` for some inputs.
- A file that fails to import or is malformed is logged and skipped at run time (the other
  skills still load); `fastworkflow skill preview` fails naming it
  (`workflow_agent.py` `_load_skill_file`, `skill_preview.py` `render_previews`).

A complete phased skill, for a toy catalog whose commands are `find_item query=`
(Output `item_uids`, `labels`), `open_item_by_uid item_uid=` (Output `label`), `list_parts`
(paged by `start`; Output `labels`) and `count_checks` (Output `total`):

```python
"""show_item: one named item, its parts and, when asked, its open checks.

Finds the item by name, opens the match and reads every part. When the request asks
about checks (`input.checks`), a phase counts them. The last phase writes the answer
from the facts the earlier phases filled.
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
    label: str = fact_from("open_item_by_uid.label")
    part_labels: list[str] = fact_from("list_parts.labels")


class CheckFacts(BaseModel):
    total: int = fact_from("count_checks.total")


def plan(input: Signature.Input) -> list[Phase]:
    phases = [
        Phase("find", lambda: find_steps(input.name), output=ItemMatch),
        Phase("item", lambda find: item_steps(input.name, find), output=ItemFacts),
    ]
    if input.checks:
        phases += [Phase("checks", lambda: ["count_checks"], output=CheckFacts),
                   Phase("report", lambda item, checks: report_steps(item, checks))]
    else:
        phases.append(Phase("report", lambda item: report_steps(item)))
    return phases


def find_steps(name):
    return [f"find_item query={name}"]


def item_steps(name, find: Optional[ItemMatch] = None):
    if find is None:
        return [f"open_item_by_uid with the uid of the {name} match", "list_parts (full list)"]
    return [f"open_item_by_uid item_uid={find.item_uids[0]}", "list_parts (full list)"]


def report_steps(item: Optional[ItemFacts] = None, checks: Optional[CheckFacts] = None):
    if item is None:
        return ["say which parts the item has, and how many open checks when they were counted"]
    lines = [f"say that {item.label} has the parts: {', '.join(item.part_labels) or 'none'}"]
    if checks is not None:
        lines.append(f"say that {item.label} has {checks.total} open checks")
    return lines
```

The same job as a flat skill (one executor run; the pick wording lets code open the match):

```python
def steps(input: Signature.Input) -> list[str]:
    return [f"find_item query={input.name}",
            f"open_item_by_uid with the uid of the {input.name} match",
            "list_parts (full list)",
            "say which parts the item has"]
```

## `Input` rules

| Rule | Where |
|---|---|
| A field is `str`, `bool`, `int`, `Literal[...]` of text (an enum), or `Optional[str] = None`; any other `Optional` fails the load | `skill_commands.py` `_spec_from_field` |
| A `bool` defaults to false; `default=True` fails the load (the model sets it true only when the request asks) | `skill_commands.py` `_parse_param` |
| No default: required. A `str` the request may leave out defaults to `""` or is `Optional[str] = None` (the code then gets `None`); a default like `"NOT_FOUND"` is written into the steps | `skill_commands` module doc; `input_instance` |
| Names `user_query`, `available_commands`, `steps_before`, `steps_after`, `reasoning` are reserved | `skill_commands.py` `RESERVED_NAMES` |
| `json_schema_extra` keys: `strip` (str), `subject` (str), `verify` (bool, enum or subject str), `means` (enum), `in_request` (int); any other key fails the load | `skill_commands.py` `SKILL_FIELD_KEYS`, `_parse_param` |
| `description` is what the planner and a `verify` check read: write it as the rule for filling the field | `skill_commands` module doc |
| A rule across fields is a pydantic validator on `Input` raising `ValueError` with what to fix; the request is `(info.context or {}).get("request", "")`, `""` when the template is rendered | `skill_commands.py` `check_input` |
| `validate_extracted_parameters` normalises an argument that only means something with others; `(False, message)` refuses. A refusal gets the planner one more typed call; a second falls back to the text planner | `skill_commands.py` `validate_call`, `plan_skill_call` |

Read the `skill_commands` module doc for what `subject`, `verify`, `strip` and `in_request`
check on a live call.

## Phases, facts and `fact_from`

- `Phase(name, steps_fn, output=None, when=True, over=None, item="item", max_items=None,
  bind=True, until=None, noun=None, optional=False)` (`skill_phases.py` `Phase`). Each phase
  is a fresh executor run with its own step budget (`FW_SKILL_PHASE_MAX_STEPS`, default
  15), except an answer-only phase (one model call, or none). The last phase writes the
  answer.
- `output` is a facts model; each field is `fact_from("<command>.<field>", **Field_kwargs)`
  (`skill_phases.py` `fact_from`): a field of that command's `Output`, else its `Input`. It is
  filled in code from the runs of that phase only. A required fact no run holds fails the
  phase, and the job stops with the facts found so far.
- `steps_fn`'s parameters bind by name to earlier phases' outputs
  (`skill_phases.py` `bound_arguments`). Unbound: the default; no default: the phase
  fails.
  - A phase the input leaves out (`when=False`, or not listed) is known in `plan()`: build
    the later phase with or without that parameter, under the same `if` (the example's
    `report`).
  - A phase skipped at run time (a callable `when` that is false, a failed
    `optional=True` phase) is read through a parameter defaulting to `None`
    (`lambda item, extra=None: ...`).
  - A `None`-defaulted parameter must name an earlier phase with an output in at least one
    plan the inputs give, or the load fails (`skill_phases.py` `check_bindings`,
    `skill_commands.py` `_checked_plan`), so a misspelling is caught.
- Phase functions take each fact as `Optional[...] = None` and write the generic wording
  first when it is `None`: the flat steps and the template are built with `None` for
  every parameter without a default (`skill_phases.py` `concatenated_steps`).
- `over="<phase>.<list field>"` (or a list) fans a phase out, one sub-plan per item;
  `until`, `max_items`, `noun`, `bind` tune it (`docs/skill_phases.md`, "Fan-out phases").
- `plan()` returning `None` or `[]` runs that call flat, from `steps()`
  (`skill_phases.py` `is_flat_plan`); such a file must define `steps()`, since
  `concatenated_steps` raises on a flat result. A plan whose phases are all `when=False` is
  an error, not a flat run.
- At load, `plan()` runs on placeholder inputs (`skill_commands.py` `_checked_plan`):
  not a list of `Phase`, a repeated or invalid name, a parameter that cannot bind, or a
  malformed `"from"` fails the load. A `"from"` naming a command or field the workflow
  lacks is only a warning at load (`skill_commands.py` `check_phase_sources`) and fails
  `skill preview --check`.
- A skill imports from `fastworkflow.skill_phases` only what it writes: `Phase`,
  `fact_from`, `facts_text`, and rarely `concatenated_steps` or `PhaseError`
  (`skill_phases.py` `__all__`, which also lists helpers workflow tests use).
  `facts_text(model)` renders facts as `field: value; ...` text for an answer step.

## The preview loop

```sh
fastworkflow skill preview <workflow_folder>            # writes <workflow_folder>/../docs/skill_previews/<skill>.md
fastworkflow skill preview <workflow_folder> --check    # writes nothing; exit 1 when not current
```

1. Write or change the skill file.
2. Run `skill preview`. A load error names the file and its reason; fix it first.
3. Read the preview: the inputs, the phases table ("runs" says when each phase runs),
   the facts with their sources, and the steps of each different plan. A step rendered
   with `<phase.fact>` placeholders shows where a value comes from at run time.
4. Check each step against the grammar table below: is it run by code, held by the finish
   check, written into the answer as intended?
5. Commit the previews and run `--check` in CI (`skill_preview.py` `check_previews`):
   it fails on a stale, missing or extra preview, a skill file that does not load, and a
   fact source the workflow does not have.

## Step grammar

Each step is one string, one line of the plan. The framework reads it in code: the finish
check (`observation_offloading/finish_check.py`), code-run steps and picks, and the
answer-only phase (`skill_phases.py`). A step is classified by the FIRST declared command
it names; the text before that command decides conditional, negated and answer steps.

| Wording | What it triggers | Implemented at |
|---|---|---|
| `cmd k=v k2=v2`, the command the first word, every required field given, values that bind | Mechanical: code runs `cmd k=v k2=v2` before the model takes a step | `skill_phases.py` `_code_step`, `plan_code_steps` |
| `cmd` alone, every field defaulted (or no parameters) | Mechanical: code sends the bare command | `skill_phases.py` `_code_step` |
| `cmd k=v (note)`, `cmd k=v; note` | The chain ends at the first `(` or `;`; the note is never sent and never changes what runs | `skill_phases.py` `_code_step`; `finish_check.py` `step_arguments` |
| a concrete value `k=Alpha` | In a phase, the finish check and the fact fill accept only a run with that value (spacing, outer quotes, case ignored) | `finish_check.py` `step_arguments`; `skill_phases.md` "Facts" |
| `k=<the uid>`, `k={x}`, `k=[..]`, `k=...` | A placeholder, not a value: the step is open (the agent fills it); any run of the command with arguments satisfies it | `finish_check.py` `_PLACEHOLDER_RE` |
| `k=Ann Lee, no filter` | A value ends at `", "`; prose swallowed into it leaves the step open | `finish_check.py` `_value_text`; `skill_phases.py` `_code_step` ("prose in value") |
| a value whose first token is an id (16+ letters, digits, `_`, `-`; 3+ digits) | The value is that token alone, cut at the first space or comma | `finish_check.py` `_UID_TOKEN_RE` |
| `cmd` then prose not in parentheses (`go_up back to the list`), or the command not the first word (`then cmd k=v`) | Open: the agent runs it | `skill_phases.py` `_code_step` |
| a required field left out (`find_item` with `query` required) | Open | `skill_phases.py` `_code_step` |
| `open_x with the uid of the Alpha match`, right after a step code ran (`find_x query=Alpha`) | Pick: code sends `open_x x_uid=<uid>` when exactly one row of that run's Output holds `Alpha` whole, or it has one row; the one unwritten required field must be a uid (`uid`, `id`, `*_uid`, `*_id`) | `skill_phases.py` `pick_step`, `_is_uid_field` |
| `cmd x_uid=<the Alpha uid> other=v` after the lookup | Pick with the other values sent as written; the uid's must be the only placeholder | `skill_phases.py` `pick_step` |
| `cmd (full list)`; also `every page`, `all pages`, `until complete`, `page through` (any case) in the note | Code reads page 1; when it names a later page, code stops and the agent pages | `skill_phases.py` `FULL_LIST_PHRASES`, `_paging`, `_more_pages` |
| `cmd start=20` (`start`, `offset`, `skip`) | That page only; code never computes a next page | `facts.py` `PAGING_PARAMETERS` |
| `cmd start=<n>`, or a step `next page with start=<n> until the list is read` | Open: the agent pages | `skill_phases.py` `_code_step` |
| any other note, or none, on a paged command | Page 1 is what the step wants; code goes on | `skill_phases.py` `_paging` |
| a condition word BEFORE the command: `if`, `only`, `when`, `whenever`, `unless`, `otherwise`, `optional(ly)`, `in case`, `as needed`, `where needed`, `where applicable`, `else` | Conditional: the agent decides; the finish check does not hold it; it stops code and keeps the phase from being answer-only | `finish_check.py` `_CONDITION_RE`, `_classify` |
| ANYWHERE in the step: `if needed`, `if necessary`, `if required`, `as needed`, `optional(ly)`, `skip when/if/unless`, `skip it when/if/unless` | Conditional, as above | `finish_check.py` `_ANYWHERE_CONDITION_RE` |
| `cmd k=v only when ...` (condition after the command, not one of the anywhere phrases) | A plain command step, held to a run. The prose joins the last value: code sends `cmd k=v only when ...` as written. To make it conditional, put the condition before the command or use an anywhere phrase | `finish_check.py` `_classify`, `_value_text` |
| a negation word before the command: `do not`, `don't`, `dont`, `never`, `none`, `not`, `no`, `nor`, `without`, `avoid`, `skip` | Negated (checked before conditions: `only when there is no match, cmd ...` is negated): code passes it over, never held to a run | `finish_check.py` `_NEGATION_RE`, `_classify` |
| first word an answer verb: `answer report present return say give summarize summarise state tell reply respond compare note identify end stop finish confirm verify conclude offer mention describe include combine format output provide write` (and `list_nothing`, `explain_answer`) | An answer step, even if it names a command later (`say what find_item found`): code passes it over, never held to a run | `finish_check.py` `_ANSWER_VERBS` |
| a step naming no command, no answer verb, no condition word (`read the notes`, `list the parts`) | "Other": not held to a run, but it stops code (an open step) | `finish_check.py` `_classify`; `skill_phases.py` `plan_code_steps` |
| `go_up` or `reset_context` alone | Code runs it; with prose after it, open | `skill_phases.py` `_CODE_NAVIGATION`, `plan_code_steps` |
| a phase whose steps hold no command, conditional or navigation step | Answer-only: no executor run, one model call with no tools (or none, below) | `skill_phases.py` `is_answer_only` |
| (answer-only) `say that P` | Verbatim: code writes `P.` | `skill_phases.py` `answer_step_kind`, `_answer_line` |
| (answer-only) `say`/`compare`/`present` head + `": "` + facts (`say what X holds: a, b`) | Verbatim: `What X holds: a, b.`; `compare ...: f` gives `F.`; `present ..., and say that P: f` gives `P. F.` | `skill_phases.py` `ANSWER_STEP_WORDS`, `_answer_line` |
| (answer-only) no `": "` facts, starting `apply`, `do not`, `don't`, `never`, `make no`, `change nothing`, `present them as options only`, `present it as options only` | Guard: adds nothing; dropped from a code-written answer | `skill_phases.py` `ANSWER_STEP_WORDS` |
| (answer-only) anything else (`say which is riskier`, `list the parts: a, b`) | Judgement: ONE such step and the model writes the whole answer | `skill_phases.py` `answer_by_code` |
| a step with `name`, `names`, `naming`, `list`, `lists`, `listing`, `every` or `each` outside the items, holding every item of a list fact (2+ items) | That list is checked: rows the answer leaves out are appended after "Also in the facts, not named above:"; the list is never cut | `skill_phases.py` `_NAMING_RE`, `_names_list_in_full`, `missing_list_items` |
| a list fact of more than `FW_SKILL_FACTS_MAX_LIST_ITEMS` (100) items joined by `", "`, `"; "`, `" \| "`, `","` or `";"` in a step with no naming word | Cut in the prompt to its first items and "(and K more; M in all)" | `skill_phases.py` `_LIST_SEPARATORS`, `capped_steps` |
| 4 or more items in a `", "` run of a code-written answer, each exactly an output value | Laid out as sub-bullets | `skill_phases.py` `MIN_BULLETED_ITEMS`, `bulleted_answer` |
| an empty string | Dropped | `skill_phases.py` `_checked_steps`; `skill_commands.py` `_call_steps` |

Consequences worth designing for:

- Code runs only the LEADING mechanical steps and stops at the first open step. Put the
  steps that need judgement last, or in a phase of their own.
- Write a filter as an argument (`list_parts kind=gear`), never as a note: the note is not
  sent.
- Write an answer step's facts as the answer should read them: in a verbatim step every
  word after `say that` is copied, instructions included.
- Make the last phase answer-only when its steps can carry every fact, so code (or one
  call without tools) writes the answer; keep a judgement step only where a judgement is
  needed.
- `repeat()` and `@<skill>` include lines are gone: loop over `range()` bounded by an int
  field's `le`, and share steps through a helper module.

## Checklist

```
- [ ] The job needs several existing commands in a known order (else: a command)
- [ ] File stem is the skill name; Signature docstring says the job; plain_utterances are generic
- [ ] Input types are str / bool / int / Literal / Optional[str] = None; bools default false
- [ ] A phased file has plan() and no steps() unless plan() can return None
- [ ] Every fact field uses fact_from with a real "<command>.<field>"
- [ ] Phases left out by the input: both Phase(...) shapes in plan(); skipped at run time: =None
- [ ] Every phase function handles None facts with the generic wording first
- [ ] Each step checked against the grammar table (code-run, held, answered as intended)
- [ ] skill preview written and committed; --check passes in CI
```

## Related

- Reading a phased skill call in a trace (`fw.skill.phase` spans, `code_stop`,
  `answer_by_code`, `facts_error`): the `debug-workflow-conversations` skill.
- The commands a skill's steps name, and their `Output` fields its facts copy: the
  `integrate-chat-agent` skill's reference for command files.
- Whether a skill change helped, measured on a fixed task set: the
  `optimize-workflow-with-feedback` skill.
