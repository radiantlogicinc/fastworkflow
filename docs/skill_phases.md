# Skill phases

A skill with parameters (`_skills/<name>.py`, see the `fastworkflow.skill_commands`
module documentation) normally expands into one flat list of steps that one executor run
works through. A long skill can instead be written as **phases**: bounded sub-plans run
one after another, each by a fresh executor run with its own step budget, with the facts
one phase found passed in code to the next.

## The skill file

```python
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow.skill_phases import Phase, fact_from


class Signature:
    """Show one named book and who has it on loan."""

    class Input(BaseModel):
        name: str = Field(description="the book's title")
        notes: bool = Field(default=False, description="the request asks about its notes")


class Found(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids", min_length=1)


class BookFacts(BaseModel):
    uid: str = fact_from("open_book_by_uid.book_uid")
    loan_uids: list[str] = fact_from("list_loans.loan_uids")


def book_steps(find: Optional[Found] = None):
    uid = find.book_uids[0] if find else "<the uid of the match>"
    return [f"open_book_by_uid book_uid={uid}", "list_loans (full list)"]


def report_steps(book: Optional[BookFacts] = None):
    if book is None:
        return ["say who has the book on loan"]
    return [f"say that {book.uid} has loans {', '.join(book.loan_uids) or 'none'}"]


def plan(input: Signature.Input) -> list[Phase]:
    return [
        Phase("find", lambda: ["open_library", f"find_book query={input.name}"], output=Found),
        Phase("book", lambda find: book_steps(find), output=BookFacts),
        Phase("notes", lambda book: [f"count_notes on {book.uid if book else 'the book'}"],
              when=input.notes),
        Phase("report", lambda book: report_steps(book)),
    ]
```

- `plan(input) -> list[Phase]` is optional. A skill without it runs as ONE phase, named
  after the skill, whose steps are `steps(input)` (`skill_commands.call_phases`; see
  "Steps run by code" below). `plan()` needs a skill with parameters. It may return
  `None` (or `[]`) for some inputs: that call then runs the same way, as one phase of
  `steps(input)` (see "Phased for some inputs only" below).
- `Phase(name, steps_fn, output=None, when=True, over=None, item="item", max_items=None,
  bind=True, until=None, noun=None, optional=False)`:
  - `name`: a Python identifier, unique in the plan.
  - `steps_fn`: returns the phase's steps (a list of str). Its parameters are bound
    **by name** to the outputs of earlier phases, as typed objects: `lambda find: ...`
    receives the `find` phase's filled `Found`. A zero-argument `steps_fn` receives
    nothing. A parameter is bound only when an earlier phase of that name ran and declares
    an `output`; otherwise it keeps its default, and one without a default fails the
    phase (`skill_phases.bound_arguments`). Which default to rely on depends on when the
    earlier phase is left out:
    - left out by the input (`when=False`, or not listed): `plan()` knows it, so build
      the later phase with or without that parameter (both `Phase(...)` calls in
      `plan()`, under the same `if`), not with a default;
    - skipped at run time (a callable `when` that is false, an `optional` phase that
      failed): `plan()` cannot know it, so the parameter defaults to `None`
      (`lambda find=None: ...`) and the function handles `None`.

    The phase functions a `steps_fn` calls take each fact as `Optional[...] = None`
    anyway: building the flat `steps()` and the template passes `None` for every
    parameter without a default (`concatenated_steps`). A parameter defaulting to `None`
    must still name an earlier phase with an output in at least one plan the inputs give,
    so a misspelled name fails the load (`check_bindings`); a parameter with another
    default (`lambda find, kind=kind: ...`) keeps that value and is not bound.
  - `output`: a facts model. Each field names the command field it is copied from,
    `json_schema_extra={"from": "<command>.<field>"}`, written `fact_from("<command>.<field>")`
    (`Field` keyword arguments pass through: `fact_from("find_book.book_uids",
    min_length=1)`): a field of that command's `Signature.Output` (returned as
    `CommandResponse.output`), else of its `Signature.Input`. See "Command outputs and facts" in `observation_search.md`.
    `None`: the phase passes nothing on.
  - `when=False` leaves the phase out, the same as not listing it. `when` may also be a
    callable of earlier phases' outputs, decided just before the phase runs (see
    "Conditions on earlier facts" below).
  - `over`, `item`, `max_items`, `bind`, `until`, `noun`: a fan-out phase, run once per item (see
    "Fan-out phases" below).
  - `optional`: `True` for a phase the request does not need (it reads something the
    answer can do without); its failure does not stop the job (see "Optional phases"
    below). A phase is required by default. A bool: `plan()` decides it from the input
    (`optional=not input.group`).
- `steps(input)` is optional in a file with `plan()`: the loader gives such a file
  `concatenated_steps(plan(input))` as its `steps()` (`skill_commands.command_from_module`).
  That is the phases' steps joined in order, each built with no earlier outputs, every
  parameter without a default given `None`. Skill matching, the planner's view of the
  skill call and the template fallback read it. A phase with a callable `when` is part of
  the concatenation (it may run). Write `steps()` only when `plan()` returns `None` for
  some inputs (below).

### Phased for some inputs only

```python
def plan(input):
    if not input.person:
        return None          # run this call flat, from steps()
    return [Phase("find", ...), Phase("person", ...), Phase("report", ...)]


def steps(input):
    return [...]             # the flat steps, written out
```

`plan()` returning `None` or `[]` (`skill_phases.is_flat_plan`) means "run this call
flat": `phases_for` returns `None`, and the skill call (`plan_skill_call`) gets one phase
named after the skill whose steps are `steps(input)` (`call_phases`), exactly as a skill
without `plan()` does. Such a file must define its own `steps()`: `concatenated_steps(None)` raises, and a file with
no `steps()` whose `plan()` runs flat for some placeholder input fails the load. A plan
whose phases are all left out by `when=False` is still an error, not a flat run.

### Conditions on earlier facts

```python
Phase("details", lambda find: detail_steps(find), output=DetailFacts,
      when=lambda find: bool(find.item_uids))
```

A callable `when` has its parameters bound by name to earlier phases' outputs, exactly
like `steps_fn` (a fan-out phase's name gives the list of its items' outputs; a parameter
with nothing to bind keeps its default). It is called when the runner reaches the phase
(`skill_phases.runs_now`), and never again when the phase resumes after ask_user. A
false result skips the phase: it runs nothing, fills no output and writes no answer, so
it counts as not run and a later parameter named after it gets its default (so give that
parameter one, `=None`, as above). Its
`fw.skill.phase` span has `skipped: true` and no child. A `when` that raises, or a
parameter with nothing to bind, fails the phase like a `steps_fn` that fails. Phase
numbering ("part N of M") still counts it. When the last phase is skipped the answer is
that of the last phase that ran, so keep the answer phase unconditional. At load a
`when` parameter without a default must name an earlier phase with an output.

### Load-time checks

`plan()` runs on the same placeholder inputs the template is rendered from. A result that
is not a list of `Phase`, a repeated or invalid name, a parameter no earlier phase with an
output provides (and no default), a parameter defaulting to `None` that no earlier phase
with an output provides in any of the results, or a malformed `"from"` fails the load
(`SkillFormatError`): the skill is logged as an error and skipped, and `fastworkflow skill
preview` fails naming it. Each output model is then checked against the workflow's command
signatures (`check_fact_sources`): a `"from"` naming a command or field the workflow does
not have is logged as a warning and the skill still loads, so a workflow can add a
command's `Output` after the skill that reads it; `fastworkflow skill preview --check`
fails on it, so a misspelled source is caught before a run. A plain command name that several
contexts define passes when some context resolves every field of the model; at run time
it resolves against the context of the phase's record.

## How a phased skill call runs

When the planner's skill call (`plan_skill_call`) is for a skill with `plan()`, the
sub-task runs its phases in order instead of one executor run on the whole list.

1. **Steps.** The phase's `steps_fn` is called with the earlier outputs bound. Its steps
   are the phase's plan; the planner does not rewrite them. Extra steps the skill call
   kept from the request before the skill's own (`steps_before`) run first, in a phase
   of their own, `extra_steps_before` (`skill_phases.steps_before_phase`): the agent runs
   them (code never runs planner steps), and the first phase then runs exactly as it
   does with no extra steps before it, its code-run steps included. The workflow's
   current context and the facts carry over from it as between any two phases; it fills
   no output. Its answer is put before the job's answer, and it is rebuilt with the plan
   after a restore. Extra steps after the skill's own (`steps_after`) go after the last
   phase's steps. One exception (`skill_phases.extra_steps_phase`):
   when the last phase is meant to be answer-only (its steps, built with no earlier
   outputs, name no command), an extra COMMAND step after the skill's own would turn it
   into an executor run and take away the answer-only handling (step 7). Such steps run
   instead in a phase of their own, `extra_steps`, inserted just before the last phase
   (its executor gets only those steps, framed as one part); the extra steps that name
   no command (`say ...`) stay with the last phase, which stays answer-only. That phase's
   answer is added after the job's answer. It is rebuilt with the plan after a restore.
2. **A sub-plan.** The phase runs as a fresh executor run with a step budget of
   `FW_SKILL_PHASE_MAX_STEPS` (default 15; read from the workflow's env file, then the
   process environment). Its step numbers continue the turn's, so observation aliases
   stay unique. The executor of a phase other than the last gets only that phase's
   steps, framed as "part N of M of a larger job: do only these steps, then finish;
   later parts will do the rest". The request itself is left out of its input, not
   even as background, so the run cannot take on the rest of the job; the steps carry
   every value it needs. Replans after an error or an ask_user answer plan from that
   same input. The last phase's input is the request followed by its steps: it does
   them and answers the request. Code runs the phase's leading mechanical steps before
   the executor's model takes a step (see "Steps run by code" below).
3. **Finish check per phase.** The finish check reads the phase's own steps against the
   outputs archived from the phase's first step on. In a phase it also compares a step's
   concrete argument values (`values=True`, the rule facts use below): a run of
   `find_book query=Emma` does not meet the step `find_book query=Dune`. The finish
   check of a run that is not a phase does not compare values.
4. **No hand-off.** A phase that reaches its budget just ends; its output is still filled
   from what it ran. A phased skill never uses the step-cap hand-off (progress summary
   plus a fresh plan), nor does a skill without phases (one phase with the executor's
   25-step cap, see "Steps run by code"). Sub-tasks that are not skill calls keep the
   hand-off and the 25-step cap unchanged.
5. **Facts.** After the run, the phase's `output` is filled in code
   (`facts_from_runs`), with no model call, from the command runs archived during **this
   phase** (`runs_for_facts`). Runs of earlier phases never count.
   - The phase's record is the one the workflow is on when the phase's run ends (its
     `"<Context> <label>"` clause, `""` at the root). One exception (`planned_record`):
     a phase that does not fan out, whose successful runs of the commands its steps name
     (with any values) all ran on, or entered, one record, and that then ended elsewhere
     (it wandered off with other commands), is filled on that record. A run that moved
     from one record to another touches both, and a step's command run on another record
     counts too, so a phase whose steps span two records is never re-bound.
   - A command that ran on a record, or entered one, counts only on the phase's record:
     the phase's runs that ran on it, plus the run that entered it (the command that
     opened the record ran on the context before it, and is counted when the next run of
     the phase, or for the last run the end of the phase, is on the record). A phase
     that opens a record and reads it therefore fills from that record and never from
     another record it also visited.
   - A command that ran only on bare contexts during the phase (the root, or a context
     with no label, such as a hub a search runs from) counts wherever the phase ended.
     A phase that searches from a hub keeps its search when it then opened a record.
   - Only runs that do one of the phase's own steps count (`runs_on_plan`). The phase's
     steps are read as the finish check reads them (`finish_check.parse_plan`), and a run
     of a command a step names counts only when it satisfies one of that command's steps
     (`finish_check.satisfies`). A step that gives an argument a concrete value right
     after its command (`find_book query=Dune`, `open_book_by_uid book_uid=b1`) is
     satisfied only by a run that ran with that value: the parameter of that name when the
     run has it, else any of its parameters, compared with spacing, outer quotes and case
     ignored. Parameters the step does not name do not matter, so each page of the planned
     query (`start=20`) still joins. A placeholder (`book_uid=<the uid>`) or prose
     (`open_book_by_uid with the uid of the Dune match`) names no value: any run of the
     command with arguments satisfies it, as before. A command no step names keeps the
     rules above. So when the executor also looks up something the phase did not plan (a
     second title, another person), that lookup never fills the fact, however broad or
     late it is; when only such a run exists, the fill fails (`MissingFactError`) with a
     message saying the planned run was not found and naming the step. It never falls
     back to an off-plan run.
   - One exception (`runs_finding_planned_values`): the executor often runs a planned
     lookup in another form than planned, the value misspelt (and corrected by the
     command), cut short by parameter extraction, or not given at all (the whole
     catalogue listed). When no run of the phase does a step with concrete values, a run
     of the step's command counts for it when it:
     - succeeded and has an output;
     - did not move the workflow (a lookup or a listing, never an opener: a run that
       opened another record never stands in for the one the step opens); and
     - holds every concrete value of the step, whole, as one scalar of its output
       (`output_names_values`: `Dune Messiah` among the match labels; `Dune Messiah 2`
       does not name it), compared with spacing, outer quotes and case ignored.

     - holds them in ONE row of its output (`narrowed_output`): list fields of one
       length are the columns of one table (uids and their labels), and exactly one row
       must hold every value found in the rows (a value in no list may be a scalar the
       output echoes, such as a scope). No single row (the value in two rows, or in none):
       the run is refused as before.

     The run's output is then cut to that row (the table's columns become one-item
     lists; other fields stay), and its facts are filled from it under the rules above.
     So a lookup that listed the whole catalogue gives the planned record's uid, never
     the first row's. A listing of another scope
     (`scope=shelf` for a step `scope=branch scope_uid=x1`) whose output does not name
     `branch` never counts, so a fact never comes from an unrelated record. When a run
     does the step itself, a run in another form is not read at all.
   - A planned step that opens the record the workflow is already on is done by the run
     that opened it (`runs_that_entered_record`). Such a step cannot run: the agent is
     told "You are already in context ..." and no command runs, so no run is archived
     for it. This happens when an earlier phase already opened the record. When no run
     of the phase does the step, the turn's latest earlier run counts as the phase's
     opening run, before its own runs, if it:
     - succeeded and satisfies the step (the step's concrete values, such as the uid it
       opens, are the values it ran with);
     - entered the phase's record (it ran on another clause, and the next run of the
       turn ran on the record, or the phase ended on the record); and
     - names the record: each parameter it was given is a word of the record's
       `"<Context> <label>"` clause.

     Both the opening command's Input facts (the uid) and its Output facts (the label)
     then come from that real run. A run that only read the record (a listing) never
     counts this way, and neither does a run that entered a different record, so a phase
     never takes an earlier phase's readings or claims a record it is not on. If no run
     meets all three conditions, the fill fails as above. The phase's finish check
     already counts the "already in context" reply as a run of the step, since that
     reply is not a failure marker and carries the step's value.
   - A planned opening step with concrete values (`list_books book_uid=b1`, a listing
     that opens one row) is also done by a run of ANOTHER command that entered the
     same record with the same value under the same parameter name
     (`open_book_by_uid book_uid=b1`; `runs_standing_in_for_openers`), from the phase or
     before it, when no run of the step's own command does it. The stand-in must have
     succeeded, entered the phase's record and name it, as above. Facts sourced from the
     planned command's Input fields then take the stand-in's value; facts from its
     Output fields are not filled by it.
   - A plain command name is resolved against the record's context.
   - A list fact joins the pages of the record's unfiltered listing only: the same page
     read again under another spelling of "not given" (`start=None`/`start=0`, `None`/
     `NOT_FOUND`) counts once, and a filtered re-read is left out ("Command outputs and
     facts" in `observation_search.md`).
   - A scalar fact of a listing read over several pages (the same command and the same
     non-paging arguments) comes from the page that carries it, not from the last page
     read (`facts._scalar_of_pages`, after the broadest-query rule): a later page that
     listed no row (read past the end) carries nothing; `None` is not carried (a
     whole-scope count only the first page computes); a bool is true when any page says
     so; any other value is the latest page's that carries one. So a count, total or
     `complete` flag from page one is never overwritten by a later or an empty page.
6. **Failure.** A required fact no run holds (`MissingFactError`), values that do not
   fit the model, a `steps_fn` that raises, or a parameter with nothing to bind fails the
   phase. Nothing is guessed. The skill call stops there: no later phase runs (unless the
   phase is `optional`, see "Optional phases"). The sub-task's answer is the facts the
   phases that ran filled, rendered in code for a person to read
   (`skill_phases.found_facts`, after the line `FOUND_FACTS_LEAD`): one block per phase
   with an output (per item of a fan-out phase), headed `<phase>:`, one `- <label>:
   <value>` line per field (`readable_facts`). Field names are written as words
   (`permission_labels` is "permissions"); a list shows its first
   `STOPPED_FACTS_MAX_ITEMS` (10) items and "(and K more; M in all)"; a uid list with a
   list of labels of the same length beside it is left out (the labels name its rows),
   while a record's own uid stays, on one line with its label (`- Dune (uid b1)`). Then a
   line naming the failed phase, the reason (the field and its source) and the phases not
   run. No text a phase's run wrote is shown: the failed phase's own answer
   may report what it never read. The turn goes on with its other sub-tasks and
   completes; the failure is visible in the answer and on the phase's span.
7. **The answer.** The sub-task has one answer: the **last** phase's. The last phase
   writes it from the facts its steps carry, built in code by its `steps_fn` (for
   example a final phase that runs no command and whose steps state the values to
   report). Earlier phases' answers are not shown, except when the call fails. A
   fan-out last phase's answer is its items' answers, in item order. When a fan-out was
   capped, the answer (and a failure's answer) ends with a note saying so.
   - **Answer-only phases.** A phase whose steps name no command is answer-only: its
     steps, read with `finish_check.parse_plan` against the workflow's commands (and its
     output's source commands), hold no command step, no conditional step and no
     navigation step (`skill_phases.is_answer_only`). Such a phase is not an executor
     run and has no tools: one model call (`skill_phases.write_answer`, under the
     executor's LM) writes its answer from its input, the request followed by the
     framing and the phase's steps. The framing says "Do not run any command; write the
     answer from these facts only." (`ANSWER_ONLY_RULE`) and to copy every name, id and
     number exactly. It takes no executor steps (`steps_used` 0), and no finish check
     applies. So the steps must carry the earlier facts as finished text: build them in
     `steps_fn`, for example with `facts_text(model)` (`fastworkflow.skill_phases`, from
     `observation_offloading.facts`), which renders a facts model as
     `uid: b1; loan_uids: l1, l2; total: 3` (lists joined with `, `, `none` for an empty
     list or `None`, nested models in parentheses; `fields=` picks fields, `sep=` the
     separator). If no workflow command names can be read, no phase is answer-only.
   - **Code writes the answer when the steps already are the answer**
     (`skill_phases.answer_by_code`). Each step is read in code (`answer_step_kind`):
     - *verbatim*: it starts with `say that`, or it is an instruction head (`say`,
       `compare`, `present`) followed by `": "` and the facts (`say what b1 holds: l1, l2`);
     - *guard*: no facts, only what not to do (`apply no change`, `present them as options
       only; apply none of them`);
     - *judgement*: anything else (`say which is riskier`, `list the loans: l1, l2`).

     The words are one constant, `ANSWER_STEP_WORDS`. When every step is verbatim or a
     guard (and one at least is verbatim), there is no model call: `answer_from_steps`
     writes one line per step, in step order, unnumbered, guards dropped. `say that P` is
     `P.`; `say what X: facts` keeps its question as a label (`What X: facts.`; also `say
     which`, `say who`, `say whether`, `say for ...`); `present ..., and say that P: facts`
     is `P. Facts.`; `compare ...: facts` is `Facts.`. One judgement step and the model
     writes the whole answer, as below. This is always on (its temporary switch was
     removed after stages 12 and 13 confirmed the gain). The phase's span says which
     (`answer_by_code`).
     So write a step's facts as the answer should read them: an instruction left in the
     facts (`..., and name every one of them:`) is copied into the answer.
   - **The answer-only answer is checked in code** (`skill_phases.answer_only_answer`).
     The call is also told to name every item of every list the steps give, never to
     shorten one, count it instead or write a placeholder. Then:
     - DSPy field markers (`[[ ## completed ## ]]`) are stripped (`clean_answer`; every
       phase's answer, and every turn's final answer, too), and so is what a reply copied
       that is never part of an answer: a line that is only a marker's name (`completed`),
       a short line echoing a template placeholder (`Input was: {final_answer}`), and the
       phase framing (`This is part 6 of 6 (answer).`, `ANSWER_ONLY_RULE` and the other
       sentences of `phase_query`);
     - an empty reply (nothing, only a code fence or markers: fewer than
       `MIN_ANSWER_CHARS` letters and digits, `is_empty_answer`) is retried once; still
       empty, the answer is written in code from the steps (`answer_from_steps`, as
       above: every step but a guard);
     - every row of the lists the steps give in full that the answer does not name is
       appended after "Also in the facts, not named above:" (`missing_list_items`). A
       list is a list field of text (two items or more) of an earlier phase's output whose
       every item is whole words of ONE step that tells the answer to name or list
       (`name`, `list`, `every`, `each`); a list given only as background is not checked.
       List fields of one length are columns of one table (uids and
       their labels), so a row is missing when one of its values is, and it is written as
       the steps write it (`Emma (b2)`). So a list is never dropped, cut short or garbled
       out of the answer; steps that give only a count check nothing. A naming word must
       stand outside the list's own items: an item "Accounts listed twice" does not make
       its step a naming step.
     - **Long background lists are cut short in the prompt** (`capped_steps`): a list
       field of an earlier output with more than `FW_SKILL_FACTS_MAX_LIST_ITEMS` items
       (default 100; read like the other settings), that a step gives in order joined by
       one separator (`", "`, `"; "`, ...), is shown as its first items and "(and K more;
       M in all)". A list the steps tell the answer to name (the append rule's lists
       above) is never cut, and the append reads the uncut lists. So a record's 857
       findings given as background no longer flood the call. `facts_text(...,
       max_items=N)` caps a skill's own rendering the same way (no cap by default).
8. **ask_user.** A phase may ask the user. The turn suspends as usual; the phased call's
   state (skill, arguments, current phase, filled outputs, answers) rides the session's
   suspension state, and on resume the phase continues and the remaining phases run.

## Optional phases

```python
Phase("group", lambda find: group_steps(find), output=GroupFacts,
      when=lambda find: bool(find.group_uids), optional=not input.group)
```

A phase is required by default: its failure stops the job (step 6 above). An
`optional=True` phase is one the request does not need. When it fails (its fill, its
`steps_fn`, a callable `when` that raises, a fan-out item, its `over`), the job goes on:

- the phase counts as not run: no output (a later parameter named after it keeps its
  default), and the answers its runs wrote and the notes it added are dropped;
- its `fw.skill.phase` span still ends with status `error` and `facts_error`;
- the next phase runs, and the answer is written as for a phase whose `when` was false.

So a later `steps_fn` that tells "not run" from "skipped" reads the earlier facts: the
answer phase of a skill whose optional group phase failed sees `group=None` while the
match list still has the group, and says the group was found but not read. Keep the answer
phase required. The skill preview shows "optional (its failure does not stop the job)" in
the phase's "runs" cell.

## Fan-out phases

A phase may repeat once per item: per value of a list the skill gives (from its input),
or per value of a list field of an earlier phase's output (the ids a search found). Each
item is its own sub-plan: its own executor run (or answer-only call), budget, finish
check and fill.

```python
from fastworkflow.skill_phases import Phase, fact_from, facts_text


class Found(BaseModel):
    item_uids: list[str] = fact_from("find_item.item_uids", min_length=1)


class ItemFacts(BaseModel):
    uid: str = fact_from("open_item_by_uid.item_uid")
    part_labels: list[str] = fact_from("list_parts.labels")


def plan(input):
    return [
        Phase("find", lambda: ["open_catalog", f"find_item query={input.query}"], output=Found),
        # One sub-plan per uid the search found; `uid` is the item.
        Phase("items", lambda uid=None: [f"open_item_by_uid item_uid={uid or '<the uid>'}",
                                         "list_parts (full list)"],
              output=ItemFacts, over="find.item_uids", item="uid"),
        # `items` is the list of the per-item ItemFacts, in item order. No command: answer-only.
        Phase("report", lambda items=None: [
            "say which parts each item has: "
            + " | ".join(facts_text(i) for i in items or []) if items else
            "say which parts each item has"]),
    ]
```

- `over`: a list of items (`over=[n.strip() for n in input.names.split(",")]`), or
  `"<earlier phase>.<list field>"`, a list field of an earlier phase's output (of a
  fan-out phase's outputs: that field of each, joined). Any other string fails the load,
  as does a reference to a phase that is not earlier or has no output, or to a field its
  output does not have. Empty values and repeats are dropped, order kept.
- `item`: the name of the `steps_fn` parameter the item is bound to (default `"item"`);
  it must not be an earlier phase's name. Other parameters bind to earlier outputs as
  usual. In the flat `steps()` (`concatenated_steps`), a fan-out over a list gives its
  steps once per item; one over an earlier field gives them once, with the item `None`
  (or the parameter's default).
- `max_items`: at most this many items run (default `FW_SKILL_FANOUT_MAX_ITEMS`, else 10).
  The rest are left out, and the answer ends with "Note: only the first N of M items were
  covered (part '<name>' runs at most N items)."
- `until`: a callable that stops the fan-out early (`skill_phases.stops_after`). It is
  called after each item's output is filled. Its parameter named after the phase gets the
  list of the items' outputs so far (typed, in item order); its other parameters bind to
  earlier phases' outputs. True stops it: the remaining items do not run, and the
  answer ends with "Note: part '<name>' stopped after K of M items because its stop
  condition was met." (none when no item was left). `max_items` still caps the items
  run; a fan-out capped before its condition held ends with the cap note instead
  (`fan_out_note`). `until` needs `over`, and at load its parameters must bind (the
  phase's own name only when it declares an `output`). For "the first two items that pass":
  `until=lambda checks: sum(c.passed for c in checks) >= 2`.
- `noun`: the plural noun the answer's notes call the items (`"people"`, `"holders"`).
  With it the notes leave out the internal phase name: "Note: only the first N of M
  people were covered." and "Note: the work stopped after K of M people because its stop
  condition was met." Without it they read as above. It must be a non-empty str and
  needs `over`; the skill preview shows it in the "fans out" column. A note already in the
  answer is not repeated, so two fan-outs over the same items (one job split in two to fit
  the step budget) with the same noun say it once.
- A later phase's parameter named after the fan-out phase receives the **list** of the
  per-item outputs (`list[ItemFacts]`), in item order.
- Each item's executor input is framed "This is part 2 of 3 (items) for item 1 of 2 (u1)
  of a larger job. ..."; an item of a last phase answers the request for that item only.
- An item suspended on ask_user resumes on that item; the items, the item in progress and
  the outputs so far ride the suspension state.

**Record binding** (`bind=True`, the default). Each item is bound to one record: a
`"<Context> <label>"` clause that names the item (the item as whole words of the clause),
among the clauses the item's runs ran on or moved the workflow to, and the one its run
ended on (the latter when it names the item, else the last such clause;
`skill_phases.item_record`). Labels that carry the item (a uid, a name) make this work;
use `bind=False` for items that are not records (a per-name search from a hub), and the
item's output is then filled where its run ended, as for any phase.

- The item's output is filled on its bound record, wherever its run ended
  (`fill_output(..., record=<bound>, ended_on=<where it ended>)`), so it never takes a
  reading of another item's record.
- The item's finish check, besides the plan steps, flags (`item_record_issues`):
  - a step that ran on another item's record: a run of one of the phase's command steps
    whose clause names another item of the phase and not this one. Leaving the previous
    item's record (`go_up`, a navigation step) and the run that opened this item's record
    from there are not flagged;
  - an item never opened: no run ran on, or moved to, a record that names the item.
  The note ("Record check: this part is about item u2 only. ...") goes back to the agent
  like the finish check's, once, with iterations left. The flags are also recorded on the
  item's span (`record_issues`; `[]` on a correct run).
- An item with an output that was never opened fails the phase ("item u2 was never
  opened: ..."), and with it the skill call, as any failed fill does.

## Steps run by code

Most steps a `steps_fn` builds are exact commands with every value filled in
(`open_library`, `open_book_by_uid book_uid=b1`). Code runs those itself, before the
executor's model takes a step (`skill_phases.CodeSteps`), and the agent gets only what
is left.

**Which step code runs** (`skill_phases.code_step`, `plan_code_steps`). A step is
mechanical when `finish_check.parse_plan` reads it as a command step (so never a
conditional, negated or answer step), its first word is its command, and its **chain**
(the text after the command, cut at the first `(` or `;`):

- is empty and every field of the command's parameters class has a default (or the
  command has no parameters class); or
- binds with the key=value parser (`_extract_parameters_from_key_values`, which declines
  placeholders `<...>`, `{...}`, `...`, unknown names and values that do not validate),
  every bound value equals the step's own value (`finish_check.step_arguments`, so prose
  swallowed into a value, `shelf_uid=s1, no filter`, is caught), and every required
  field is given.

A core `go_up` or `reset_context` written alone also runs. Code sends `command chain`:
the rest of the step is a note and is never sent. A command several contexts define is
mechanical only when it reads the same under each one's parameters class. Any other
step is open.

**Picks** (`skill_phases.pick_step`). An open step right after a step code ran also runs
when it opens one row of that step's Output: `open_book_by_uid with the uid of the Dune
match` after `find_book query=Dune`. Code reads the step's run from the archive and sends
`open_book_by_uid book_uid=<uid>` only when:

- the step is a command step, its command has the same required fields under every
  context, and the step leaves exactly one of them unwritten: a uid (named `uid` or `id`,
  or ending `_uid` or `_id`, plural too: the test readable facts use for uids). The step
  writes no value of its own (prose or a placeholder), or its ONLY placeholder is that
  uid's and its other values are concrete
  (`list_access resource_type=app resource_uid=<the Dune uid> filter=Ann`): code then
  sends those values as written, with the uid filled in
  (`list_access resource_uid=<uid> resource_type=app filter=Ann`);
- the step names a concrete value of the step before it (`Dune`), paging values aside;
- that run is the step's one successful run, its observation names no later page, and
  its Output has exactly one uid list; rows are read as `narrowed_output` reads them
  (list fields of one length are the columns of one table); and
- exactly one row holds the named value whole (`Dune`, never `Dune 1`; spacing, quotes
  and case ignored), or the Output has exactly one row at all.

Otherwise (two exact matches, several rows and none exact, no rows) the step is open as
before and the agent picks. The thought says which: "Step 3 of the plan, run by code
(the only exact match of step 2)." or "(the only row of step 2)."

**What runs.** Code walks the phase's own steps from the first. Each runs through the
agent's own `execute_workflow_query` tool inside the executor run
(`fastWorkflowReAct._run_loop` with `code=`; the runner sets the agent's `code_steps`
attribute, which `forward()` takes and clears), so it is an ordinary trajectory step,
thought "Step 2 of the plan, run by code.": outputs, the archive, facts, spans, the
repeat guard and the finish check see a normal run. Answer steps and negated steps are
passed over.

- Code steps do not count toward `FW_SKILL_PHASE_MAX_STEPS`; the budget is the agent's.
- A retry never re-runs a code step, and a phase resumed after ask_user runs no code.
- Planner steps (`steps_before` / `steps_after`) are never run by code; the
  `steps_before` run in a phase of their own that runs no code (`code_stop` "off"), and
  a `steps_after` step that runs
  anything (not an answer or negated step) is an open step: the agent runs it.
- A command with side effects runs like any other: its values come from the checked
  skill input or from facts filled in code.
- The command text is `name k=v` or a bare name, so parameter extraction binds it in
  code (`deterministic_form` `key_value`, or `bare`: nothing after the name and every
  field defaulted).

**Navigation.** When a step's command is not available where the workflow is, code goes
there first only along one shortest path whose moves need no value (`go_up`,
`reset_context`, an opener with no parameters): `context_navigation.code_path`, the
planner behind the agent's refusal text. Each move is a step of its own ("Run by code to
reach the context of step 3."). Otherwise the step runs anyway, the tool's refusal names
the way, and code stops there. A step that opens the record the workflow is already on
gets the "You are already in context ..." reply, which counts as done.

**Hand-off.** Code stops at the first open step, after a step that failed (a
failure marker in its observation, a command run of it that did not succeed, or no run
at all other than the "already in context" reply), or by
the paging rule below. The agent's loop then goes on with the code steps in its
trajectory, and the phase's framing has one more sentence: "Steps marked 'run by code'
in the trajectory are done; do the remaining steps." (stripped from answers by
`clean_answer`). A failed step is the agent's last observation; code never fails a phase
by itself. Code does not resume after an open step.

When every step ran by code, a phase whose answer is not shown ends with an empty
answer and no model call. The last phase and the extra-steps phase still go on into the
agent's loop from code's last step (fix-hia1.34): the model takes at least one step,
weighing the outputs code got, and finishes (the finish check reads that finish) or runs
a command the request needs that the plan lacked; then the answer call every executor
run ends with. The budget is the agent's, as above; no code step runs again. The last
phase's framing then says "Every step below has run (by code); ..." in place of "Do these
steps", with the copy rule answer-only phases get (`phase_query(code_ran_all=True)`, given
to the loop as `CodeSteps.done_inputs`). Fan-out items whose answer is not shown run the
same way as a phase, so such an item of mechanical steps makes no model call; the item
record check still runs.

**Paging is literal.** Code sends only a paging value (`facts.PAGING_PARAMETERS`) the
step writes: `list_loans start=20` runs that page, a step with none runs page 1. Code
never computes a next page. A step that writes the paging value as a placeholder
(`start=<n>`), in its chain or as a step of its own with no command (`next page with
start=<n> until the list is read`), is open: the agent pages. One exception: when a step
on a paged command writes no paging value, its note holds one of
`skill_phases.FULL_LIST_PHRASES` ("full list", "every page", "all pages", "until
complete", "page through"; any case), and page 1's observation names a later page (a
paging parameter with a positive value, `start=10`), code stops after that step and the
agent reads the rest. Its thought then says "(its first page)". Any other note, or none:
page 1 is what the step wants, and code goes on.

**Always on.** Code-run steps are always on (the temporary switch,
`FW_SKILL_CODE_STEPS`, was removed after stages 12 and 13 confirmed the gain).

**A skill without phases runs as one phase** (fix-6llp.11). A skill call with no
`plan()`, or whose `plan()` returns no phases for its input, runs through the same
runner as one phase named after the skill, whose steps are `steps(input)`
(`skill_commands.call_phases`). Code runs its leading mechanical steps and picks as
above; the agent runs the rest and answers the request. Its step budget is the
executor's own cap (`max_iters`, 25), not `FW_SKILL_PHASE_MAX_STEPS`, and, as for any
phase, there is no step-cap hand-off: a run that reaches the cap just ends. Planner
steps before and after the skill's own are handled as in a phased call (a phase of their
own before; after, left to the agent). Suspended on ask_user and restored, the phase is
rebuilt from the skill and resumes. Nothing changes for the app developer.

**Writing steps for code.**

- A note goes in parentheses after the chain (`list_loans (for the due dates)`) and is
  never sent. Prose after the command that is not in parentheses (`go_up back to the
  library`) leaves the step to the agent.
- A note never changes what runs: write a filter as an argument
  (`list_loans status=overdue`), not as a note (`list_loans (only overdue ones)`).
- For every page, say so in the note with one of the `FULL_LIST_PHRASES`
  (`list_loans (full list)`), or write the paging as a step of its own with `start=`
  (`next page with start=<n> until the list is read`), which the agent pages. A note
  that asks for more pages in other words gets page 1.
- A step the agent must decide is written as conditional: a condition word before the
  command ("only when it is overdue, renew_loan ..."; `finish_check._CONDITION_RE`), or
  one of the phrases that count anywhere in the step ("if needed", "as needed",
  "optional", "skip when ..."; `_ANYWHERE_CONDITION_RE`). "renew_loan ... only when
  overdue" is a plain command step: the finish check holds it to a run, and the prose
  joins its last value (`loan_uid=l1 only when overdue`). A negation word
  before the command (`_NEGATION_RE`: "skip when paid, renew_loan ...", "only when
  there is no match, ...") makes the step negated instead: code passes it over and it is never
  held to a run. A concrete mutating step (`renew_loan loan_uid=l1`) is run by code.

## Tracing

Each phase run (each item of a fan-out phase) is one `fw.skill.phase` span (span
contract v22, `fw.skill.phase` v5), the parent of that run's `fw.agent.execute`:

| attribute | meaning |
|---|---|
| `skill`, `phase`, `phase_index`, `phase_count` | which phase of which skill (index 0-based) |
| `step_budget`, `first_step` | the run's budget and its first step number |
| `answer_only` | the phase named no command: one model call with no tools wrote its answer (its `fw.agent.execute` is the child, with no tool calls) |
| `answer_by_code` | an answer-only phase only: `true` when code wrote its answer from its steps, with no model call (every step verbatim or a guard, `answer_by_code`) |
| `item`, `item_index`, `item_count`, `items_total` | one item of a fan-out phase: the item, its 0-based index of `item_count` items run, of `items_total` before the cap |
| `record_issues` | a bound item's record-check flags (a list, `[]` when none); `record` is then the item's bound record |
| `skipped` | the phase's callable `when` was false: nothing ran, no child span (`false` with `facts_error` when deciding it failed) |
| `stop_condition_met` | an item of a fan-out phase with `until`: the condition held after it, so the fan-out stops |
| `resumed` | a run resumed after ask_user (it opens a new span) |
| `steps` | the phase's numbered plan |
| `steps_used`, `exhausted`, `suspended` | executor steps the run took (code steps included); whether it reached its budget or suspended |
| `steps_by_code` | steps code ran before the agent, moves included; model steps = `steps_used - steps_by_code` |
| `code_stop` | why code stopped: `done` (every step), `open step N`, `step N failed`, `no path for step N` (refused, no no-value path), `more pages at step N`, `stopped` (the executor's step hook ended the run), `off` (no code: the phase of the planner's `steps_before`, or its steps could not be read) |
| `record`, `facts` | the record its output was filled on, and the filled output (`null` for no output) |
| `facts_error` | why the phase failed (status `error`) |
| `error_type` | the exception that ended the run, when one did |

`steps_by_code` and `code_stop` are absent on an answer-only phase and on a resumed run.
A step code ran is an ordinary `fw.agent.step` (v7) with `by_code: true` and a thought
that says "run by code"; its `fw.agent.tool_call` child is unchanged. A phase run wholly
by code still has its `fw.agent.execute`, with no model child spans unless it writes the
answer. A command code sent bare is bound with `fw.nlu.param_extraction`
`deterministic_form` `bare`.
