---
name: author-workflow-skills
description: >-
  Write fastWorkflow skill files (`_skills/<name>.py`): parameterized multi-command jobs whose
  steps are built in code and expanded into the agent's numbered plan, flat or as phases that
  pass facts to each other. Covers when to write a skill instead of a command, the file shape
  (`Signature` + `PlanGenerator`), the `Input` rules, phases, facts and `fact_from`, `@phase`,
  `__call__`, the `skill preview` / `--check` loop, and the step wording the framework reads
  in code (what code runs, what the finish check holds, what writes the answer). Use when
  adding, reviewing or debugging a skill file, or when a skill's steps are run, checked or
  answered differently than intended.
---

# Writing workflow skills

A skill is one Python file, `_skills/<name>.py`, next to `_commands/`, written like a command
file. Its `Signature` describes a job and its `Input` declares the job's typed arguments,
exactly as a command's; where a command has a `ResponseGenerator` (it generates a response), a
skill has a `PlanGenerator` (it generates a plan): each public method is a phase that returns
its steps as text. At run time the skill matcher picks the skill for a request
(`workflow_agent.match_skill`, from the description and `plain_utterances`), the planner fills
`Input` in one typed call (`skill_commands.plan_skill_call`), the arguments are checked, and
the steps become the agent's numbered plan.

The one-page guide is `docs/skill_file_layout.md` in the fastworkflow repository (not in the
wheel), with the preview; `docs/skill_phases.md` covers phases, facts, code-run steps and
answers. The module documentation of `fastworkflow.skill` (the form),
`fastworkflow.skill_commands` and `fastworkflow.skill_phases` (the runtime) ships with the
package and is the source of truth.

Code references below name a module of the `fastworkflow` package and a symbol in it.

## Skill or command?

| Write a command (`_commands/<name>.py`) when | Write a skill (`_skills/<name>.py`) when |
|---|---|
| it is one action on the application (wraps code, returns an `Output`) | a family of requests is answered by the same several commands in a known order (find, open, read, answer) |
| the agent should choose it by intent | the agent keeps getting the order, a value or the answer wrong for that family |
| it needs to read state or call a backend | the job only needs text: `PlanGenerator` methods build text and must not call a backend or read state (`skill_commands` module doc) |

A skill adds no capability: every step it writes runs an existing command (checked at load).
A one-command skill is a command with a worse name. Fixed steps with no parameters are
allowed (a `PlanGenerator` with one method taking no `input`): the text planner adapts them.

## The file

- Name: the file stem is the skill's name, a Python identifier (`show_item.py`;
  `workflow_agent.py` `_read_skill_file`). `_<name>.py` is a helper module skill files
  import (shared lookup tables and phrases, never code that chooses steps); `_split.py` is
  the split settings; subfolders are skipped (`workflow_agent.py` `_skill_files`).
- Order (`docs/skill_file_layout.md`): module docstring, imports, `Signature`, shared facts
  models, `PlanGenerator` (`__call__` first, then the phases in run order, then `_helpers`),
  module-level tables and shared functions.
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
from pydantic import BaseModel, Field

from fastworkflow.skill import fact_from, pick


class Signature:
    """Show one named item and its parts, and its open checks when asked. One job per item."""

    plain_utterances = ["what parts does the Alpha item have", "show item Beta and its checks"]

    class Input(BaseModel):
        name: str = Field(description="the item's name exactly as the request writes it",
                          json_schema_extra={"subject": True})
        checks: bool = Field(default=False, description="the request asks about its open checks",
                             json_schema_extra={"verify": True})


class PlanGenerator:
    """find -> item -> checks (when asked) -> report."""

    def __call__(self, input):
        checks = [self.checks] if input.checks else []
        return [self.find, self.item, *checks, self.report]

    class ItemMatch(BaseModel):
        uids: list[str] = fact_from("find_item.item_uids", min_length=1)
        labels: list[str] = fact_from("find_item.labels")

    def find(self, input) -> ItemMatch:
        return [f"find_item query={input.name}"]

    class ItemFacts(BaseModel):
        uid: str = fact_from("open_item_by_uid.item_uid")
        label: str = fact_from("open_item_by_uid.label")
        part_labels: list[str] = fact_from("list_parts.labels")

    def item(self, input, find) -> ItemFacts:
        return [pick("open_item_by_uid", item_uid=find.uids[0], named=input.name),
                "list_parts (full list)"]

    class CheckFacts(BaseModel):
        total: int = fact_from("count_checks.total")

    def checks(self) -> CheckFacts:
        return ["count_checks"]

    def report(self, item, checks=None):
        lines = [f"say that {item.label} has the parts: {self._parts(item)}"]
        if checks is not None:
            lines.append(f"say that {item.label} has {checks.total} open checks")
        return lines

    def _parts(self, item):
        return ", ".join(item.part_labels) or "none"
```

The same job as a flat skill (one executor run; the pick wording lets code open the match):

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

- Each public method of `PlanGenerator` is a phase (`skill.py` `planned`, building the
  runtime's `skill_phases.Phase`), in the order written; `_name` methods are helpers. Each
  phase is a fresh executor run with its own step budget (`FW_SKILL_PHASE_MAX_STEPS`,
  default 15), except an answer-only phase (one model call, or none). The last phase writes
  the answer.
- Its return annotation is its facts model; each field is `fact_from("<command>.<field>",
  **Field_kwargs)` (`skill_phases.py` `fact_from`): a field of that command's `Output`, else
  its `Input`. It is filled in code from the runs of that phase only. A required fact no run
  holds fails the phase, and the job stops with the facts found so far. A model only one
  phase fills is nested right above the method; one that depends on the input is
  `@phase(output=lambda input: ...)`.
- Its parameters bind by name: `input`, and earlier phases' facts (`skill_phases.py`
  `bound_arguments`). A phase that may have been skipped is read through `=None`. A
  parameter must name an earlier phase with facts in at least one call the inputs give, or
  the load fails (`skill_phases.py` `check_bindings`), so a misspelling is caught.
- `return []` skips the phase. It is always decided when the phase is reached: the runtime
  calls the method once with the real facts (`skill.py` `_phase_functions`: it is the
  phase's `when`, and its steps are kept for the run). A skipped phase passes nothing on.
- The shape of a call: `__call__(self, input)` returns the phase methods it runs, in order
  (default: all). A call of one phase that reads and fills no facts runs flat, as one plan
  (`skill.py` `_runs_flat`). A phase repeated for several cases is one method each, calling
  one shared module function (a `self._helper` called with a fact is not run in the
  template, it reads as its name).
- `@phase(over="<phase>.<list field>")` (or a function of `input` giving the items) fans a
  phase out, one sub-plan per item; its one parameter that is neither `input` nor a phase
  gets the item; `until`, `max_items`, `noun`, `bind` tune it (`docs/skill_phases.md`,
  "Fan-out phases"). `@phase(optional=True)`: the job goes on when it fails.
- The template (what the planner sees; `skill_commands.py` `placeholder_template`) calls
  each method once per input with `skill_phases.Placeholder` facts: `<find.total>`,
  `<find.uids[0]>`, `<find.labels[*]>`; a `self._helper` called on one reads `<helper in
  words>`; every test on one (`bool`, `len`, `==`, `<`, `in`) reads as true and is noted,
  and the phase's lines end `(only when find.uids)`. `x is None` is not seen. Write wording
  that tests facts (`a or b`) in a helper.
- At load (`skill_commands.py` `_checked_plan`, `placeholder_template`,
  `check_step_commands`; `workflow_agent.py` `step_command_problems`): a method without
  `self`, a bad annotation, a parameter that cannot bind, a malformed `"from"`, a method
  that fails on placeholder facts or writes `None` into a step, and a step naming a command
  the workflow lacks or a parameter its `Signature.Input` lacks fail the load. A `"from"`
  naming a command or field the workflow lacks is only a warning at load
  (`check_phase_sources`) and fails `skill preview --check`.
- A skill imports from `fastworkflow.skill` what it writes: `fact_from`, `pick`, `phase`,
  `num`, `count`, `listed`.

## The preview loop

```sh
fastworkflow skill preview <workflow_folder>            # writes <workflow_folder>/../docs/skill_previews/<skill>.md
fastworkflow skill preview <workflow_folder> --check    # writes nothing; exit 1 when not current
```

1. Write or change the skill file.
2. Run `skill preview`. A load error names the file and its reason; fix it first.
3. Read the preview: the inputs, the phases table ("runs" says when each phase runs),
   the facts with their sources, and the steps of each different plan. A step rendered
   with `<phase.fact>` placeholders shows where a value comes from at run time. Under the
   steps, "Pages: step 2 reads up to 20 pages; step 3 reads page 1 only." says how code
   pages each step on a paged command (`skill_phases.step_paging`): only a note with a
   full-list phrase (`FULL_LIST_PHRASES`: "full list", "every page", ...) reads later
   pages, up to `CODE_MAX_PAGES` (20); "(all of them)" or no note reads page 1. A step that writes `start=` reads that
   page and is not listed.
4. Check each step against the grammar table below: is it run by code, held by the finish
   check, written into the answer as intended?
5. Commit the previews and run `--check` in CI (`skill_preview.py` `check_previews`):
   it fails on a stale, missing or extra preview, a skill file that does not load (a step
   naming an unknown command or parameter included), a fact source the workflow does not
   have.

## Step grammar

Each step is one string, one line of the plan. The framework reads it in code: the finish
check (`observation_offloading/finish_check.py`), code-run steps and picks, and the
answer-only phase (`skill_phases.py`). A step is classified by the FIRST declared command
it names; the text before that command decides conditional, negated and answer steps (a
condition word in its chain also makes it conditional).

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
| `cmd (full list)`; also `every page`, `all pages`, `until complete`, `page through` (any case) in the note | Code reads page 1, then each later page the output names, as more code-run steps, up to `CODE_MAX_PAGES` (20) pages; then code goes on | `skill_phases.py` `FULL_LIST_PHRASES`, `_paging`, `_next_page`, `CODE_MAX_PAGES` |
| `cmd start=20` (`start`, `offset`, `skip`) | That page only; code never computes a next page | `facts.py` `PAGING_PARAMETERS` |
| `cmd start=<n>`, or a step `next page with start=<n> until the list is read` | Open: the agent pages | `skill_phases.py` `_code_step` |
| any other note, or none, on a paged command | Page 1 is what the step wants; code goes on | `skill_phases.py` `_paging` |
| a condition word before the command (or in its chain, next row): `if`, `only`, `when`, `whenever`, `unless`, `otherwise`, `optional(ly)`, `in case`, `as needed`, `where needed`, `where applicable`, `else` | Conditional: the agent decides; the finish check does not hold it; it stops code and keeps the phase from being answer-only | `finish_check.py` `_CONDITION_RE`, `_classify` |
| ANYWHERE in the step: `if needed`, `if necessary`, `if required`, `as needed`, `optional(ly)`, `skip when/if/unless`, `skip it when/if/unless` | Conditional, as above | `finish_check.py` `_ANYWHERE_CONDITION_RE` |
| `cmd k=v only when ...` (a condition word in the command's chain: after it, up to the first `(` or `;`) | Conditional, as if the condition came before the command, so the prose never joins the last value. A condition in a note (`cmd k=v (only when overdue)`) is never read: the step stays a plain command step, held to a run | `finish_check.py` `_classify` |
| a negation word before the command: `do not`, `don't`, `dont`, `never`, `none`, `not`, `no`, `nor`, `without`, `avoid`, `skip` | Negated (checked before conditions: `only when there is no match, cmd ...` is negated): code passes it over, never held to a run | `finish_check.py` `_NEGATION_RE`, `_classify` |
| first word an answer verb: `answer report present return say give summarize summarise state tell reply respond compare note identify end stop finish confirm verify conclude offer mention describe include combine format output provide write` (and `list_nothing`, `explain_answer`) | An answer step, even if it names a command later (`say what find_item found`): code passes it over, never held to a run | `finish_check.py` `_ANSWER_VERBS` |
| a step naming no command, no answer verb, no condition word (`read the notes`, `list the parts`) | "Other": not held to a run, but it stops code (an open step) | `finish_check.py` `_classify`; `skill_phases.py` `plan_code_steps` |
| `go_up` or `reset_context` alone | Code runs it; with prose after it, open | `skill_phases.py` `_CODE_NAVIGATION`, `plan_code_steps` |
| a phase whose steps hold no command, conditional or navigation step | Answer-only: no executor run, one model call with no tools (or none, below) | `skill_phases.py` `is_answer_only` |
| (answer-only) `say that P` | Verbatim, the only form code copies: code writes `P.` | `skill_phases.py` `answer_step_kind`, `_answer_line` |
| (answer-only) a head and its facts (`say which is riskier: A 3, B 5`, `say what X holds: a, b`, `compare ...: facts`) | Judgement, like any step but `say that`: only the author knows whether the facts already answer the head. To have code copy facts, write `say that what X holds: a, b` (`What X holds: a, b.`) | `skill_phases.py` `answer_step_kind` |
| (answer-only) no `": "` facts, starting `apply`, `do not`, `don't`, `never`, `make no`, `present them as options only`, `present it as options only`, AND a negation word (`no`, `not`, `none`, `never`, ...) | Guard: adds nothing; dropped from a code-written answer. Without a negation word (`apply a 10% discount to the total`) it is judgement | `skill_phases.py` `ANSWER_STEP_WORDS`; `finish_check.py` `_NEGATION_RE` |
| (answer-only) anything else (`say which is riskier`, `list the parts: a, b`) | Judgement: ONE such step and the model writes the whole answer | `skill_phases.py` `answer_by_code` |
| a step with `name`, `names`, `naming`, `list`, `lists`, `listing`, `every` or `each` outside the items, holding every item of a list fact (2+ items) | That list is checked: rows the answer leaves out are appended after "Also in the facts, not named above:"; the list is never cut | `skill_phases.py` `_NAMING_RE`, `_names_list_in_full`, `missing_list_items` |
| a list fact of more than `FW_SKILL_FACTS_MAX_LIST_ITEMS` (100) items joined by `", "`, `"; "`, `" \| "`, `","` or `";"` in a step with no naming word | Cut in the prompt to its first items and "(and K more; M in all)" | `skill_phases.py` `_LIST_SEPARATORS`, `capped_steps` |
| 4 or more items in a `", "` run of a code-written answer, each exactly an output value | Laid out as sub-bullets | `skill_phases.py` `MIN_BULLETED_ITEMS`, `bulleted_answer` |
| an empty string | Dropped | `skill_phases.py` `_checked_steps`; `skill_commands.py` `_call_steps` |
| a step whose first word is snake_case (or a known command): its command and the `key=` of its chain (before `(`, `;` or `,`) | Checked at load against the workflow's commands and their `Signature.Input`; an unknown one fails the load | `skill_commands.py` `step_commands`, `check_step_commands` |

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
- [ ] Signature + PlanGenerator; each phase a method with self, in run order; helpers _named
- [ ] Every fact field uses fact_from with a real "<command>.<field>"
- [ ] Input-dependent shape in __call__; a skip is `if not <what it needs>: return []`;
  a phase that may be skipped is read through =None
- [ ] The template (preview) reads well: wording that tests facts lives in helpers
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
