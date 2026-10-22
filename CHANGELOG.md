# Changelog

## Unreleased

### Added

- **Phase facts: stand-in openers, re-bind to the planned record, plainer
  step values; `name=value` parameters parsed in code** (fix-hia1.10).
  In agent mode a command written as `cmd key=value key2=value2`, every key a
  field of its Input, is parsed before the LLM fallback: a value runs to the
  next ` key=` or the end, quotes and a trailing "(...)" note go, types come
  from the Input model; any doubt falls back as before (the span records it as
  `xml_regex`, so no contract changes). A plan step's value also ends at ", " prose, and
  an identifier value at its first token (`step_arguments`). A planned opener
  with concrete values (`list_x x_uid=U`) is done by a successful run of
  another command that entered the phase's record with the same value under
  the same name (`runs_standing_in_for_openers`); facts from the planned
  command's Input take it. A non-fan-out phase whose step commands all ran on
  one record, and that ended elsewhere, is filled on that record
  (`planned_record`).
- **A plural noun for a fan-out's items** (fix-hia1, `docs/skill_phases.md`).
  `Phase(..., over=..., noun="people")` makes the answer's notes count in that
  noun and leave out the internal phase name: "Note: only the first 3 of 5
  people were covered." and "Note: the work stopped after 3 of 5 people
  because its stop condition was met." (`cap_note` / `fan_out_note` take
  `noun`). Without a noun the wording is unchanged. A noun on a phase that
  does not fan out, or one that is not a non-empty str, fails the load. Skill
  previews show it in the "fans out" column. A note already in the answer is
  not repeated (two fan-outs over the same items with one noun say it once).
- **Skill phases: flat for some inputs, conditions on earlier facts, fan-out
  that stops on a condition** (fix-hia1.9, `docs/skill_phases.md`). `plan()`
  may return `None` or `[]` to run that call flat, from the file's own
  `steps()` (`is_flat_plan`; `phases_for` returns `None`). `Phase.when` may
  be a callable bound by name to earlier outputs, decided when the phase is
  reached (`runs_now`); a skipped phase counts as not run.
  `Phase(..., over=..., until=callable)` stops a fan-out once the outputs so
  far (its parameter named after the phase) meet a condition (`stops_after`);
  `max_items` still caps, and `fan_out_note` says "stopped" or "capped".
  `fw.skill.phase` v3 (span contract v18) adds `skipped` and
  `stop_condition_met`. Skill previews show a callable `when` / `until` by its
  source (`condition_text`), add a "fans out" column, mark phases whose steps
  read earlier facts as rendered with placeholder facts, render a flat
  variant's `steps()`, and pass a fan-out phase's output on as a list.
- **Answer-only skill phases** (fix-hia1.7, `docs/skill_phases.md`). A phase
  whose steps name no command (`skill_phases.is_answer_only`: no command,
  conditional or navigation step as `finish_check.parse_plan` reads them) is
  not an executor run: one model call with no tools (`write_answer`) writes its
  answer from the request and its steps, told "Do not run any command; write
  the answer from these facts only." It cannot call a workflow command. New
  `facts_text(model)` (`observation_offloading.facts`, re-exported from
  `skill_phases`) renders a facts model as `field: value` text for steps.
  `fw.skill.phase` v2 (span contract v17) adds `answer_only`.
- **Fan-out skill phases with record binding** (fix-hia1.3).
  `Phase(..., over=<list> | "<phase>.<field>", item="item", max_items=None,
  bind=True)` runs the phase once per item, each its own sub-plan; a later
  phase gets the list of per-item outputs. At most `max_items` /
  `FW_SKILL_FANOUT_MAX_ITEMS` (default 10) items run, and the answer says when
  it was capped. Each item is bound to the record that names it: its output
  is filled there, and its finish check flags a step on another item's record
  and an item never opened (`item_record_issues`; span `record_issues`).
  `fw.skill.phase` v2 adds `item`, `item_index`, `item_count`, `items_total`,
  `record_issues`.
- **Skill previews and a skill-file layout convention** (fix-553q,
  `fastworkflow.skill_preview`, `docs/skill_file_layout.md`).
  `fastworkflow skill preview <workflow_folder> [--out <dir>] [--check]` loads
  every skill with the skill loader and writes `<dir>/<skill>.md` (default
  `<workflow_folder>/../docs/skill_previews`): description, inputs (type,
  default, sample value, `verify` / `subject` flags, description, other
  options), `plain_utterances`, and the steps for sample inputs, rendered once
  per combination of the inputs that change them; a phased skill also lists
  its phases, when each runs, and the facts each output holds with their
  `"from"` sources, with earlier facts rendered as placeholders
  (`<find.item_uids[0]>`). Byte-stable; `--check` writes nothing and exits 1
  when a preview is stale, missing or extra. The layout doc fixes one order
  for a skill file (docstring, `Signature`, fact models, `plan()` / `steps()`
  as the table of contents, then the step functions), naming, plain-string
  steps, no step logic in helpers, and conditions in `plan()`.
- **Skill previews that read alone** (fix-553q follow-up). Each different plan
  is rendered once, titled by the fewest inputs that decide it ("Plan when
  compare = true"); the "runs" column names the deciding input, and one line
  says when `plan()` returns phases at all. Output models, `when` and fan-outs
  that differ by input carry the inputs that give them. Past 16 plans, a
  pairwise cover of the inputs plus one plan per phase layout is rendered.
  Int inputs are varied (the sample and 0 or 3), and samples respect `ge` /
  `le`. A fan-out renders each listed item (never `None`), or the first and a
  later item of an earlier phase's list; an empty list says it runs nothing.
  Placeholder facts are consistent (a uid fact is the entry of the earlier
  uid list, a fan-out item's output carries its item), str placeholders
  survive `split`, and a float fact is 999999 in arithmetic so `int()` /
  `float()` never print a plausible 0. A callable `when` shows its closure
  values and parameter defaults, joined without "( " gaps.
- **Skill phases: a skill run as bounded sub-plans** (fix-hia1.2,
  `fastworkflow.skill_phases`, `docs/skill_phases.md`). A skill file with params
  may define `plan(input) -> list[Phase]`, with `Phase(name, steps_fn,
  output=None, when=True)`; a skill without it is unchanged. `steps_fn`'s
  parameters are bound by name to earlier phases' filled outputs (typed facts
  models); `when=False` leaves a phase out; `steps(input)` stays the flat
  concatenation (`concatenated_steps`) that matching and the template read, and
  a file with only `plan()` gets it. The loader runs `plan()` on the placeholder
  inputs: bad phases, unbindable parameters or a malformed `"from"` fail the
  load; a `"from"` naming a command field the workflow lacks is a logged warning
  (`check_phase_sources`). A skill call of a phased skill runs each phase as a
  fresh executor run with a step budget of `FW_SKILL_PHASE_MAX_STEPS` (default
  15), its own steps (not rewritten by the planner) and its own finish check;
  after it, the phase's output is filled in code from the phase's own command
  runs on the record it ended on (plus the run that entered that record). Only
  runs that do one of the phase's own steps count (fix-hia1.5,
  `runs_on_plan`, `finish_check.satisfies`): a step giving a concrete value
  (`find_x query=Ann`) takes only runs carrying that value, after
  normalisation, and each page of that query; a placeholder or prose argument
  keeps the looser match. An off-plan lookup of another value never fills the
  fact; with only that run, the fill fails saying the planned run was not
  found. A planned opening step the workflow was already on (refused unrun
  with "You are already in context ...", which archives no run; e.g. an
  earlier phase opened the record) is done by the turn's earlier run that
  satisfies the step, entered the phase's record, and whose parameters are
  words of the record's clause (fix-hia1.6, `runs_that_entered_record`); the
  opening command's Input and Output facts come from that run, and a run
  that only read the record, or entered another one, never counts.
  A phase's finish check applies the same value rule to its steps
  (`values=True`); the finish check of any other run is unchanged. A
  missing fact stops the call: no later phase runs, and the answer is what the
  phases that ran found plus a line naming the failed phase and field. The
  sub-task's answer is the last phase's. A phased skill is never handed off
  (`_hand_off`); other sub-tasks keep the hand-off and the 25-step cap. A phase
  suspended on ask_user resumes and the rest run (the call's state rides the
  session state, `skill_phase_run`). New span `fw.skill.phase` (contract v1;
  aggregate span contract 16) per phase run. `SkillCallPlan` gains `phases`,
  `steps_before`, `steps_after`.

- **Command outputs archived as JSON; facts filled from them** (fix-hia1.1,
  `CommandResponse.output`, `observation_offloading.facts`). A command may
  return its `Signature.Output` as `CommandResponse(response=..., output=...)`;
  the field is optional and appended (default `None`), and the agent's
  observation and prompt are unchanged. Each command an archived execute step
  ran is stored beside the step's text in the new additive
  `offload_command_runs` table (name, success, record context, parameters and
  Output as JSON, redacted like the text), read with
  `RuntimeHandleArchive.command_runs`; older stores gain the table on open and
  their rows read as before. `facts_from_runs` / `facts_from_archive` fill a
  pydantic model whose fields declare `json_schema_extra={"from":
  "<command>.<field>"}` from the Output or Input of successful runs, with no
  model call: latest run for a scalar, runs joined in order for a list, a
  missing required field is a `MissingFactError`, an unknown or ambiguous
  source a `FactSourceError` (`check_fact_sources`, `workflow_signatures`). A
  plain command name resolves against the record's context; `Context/command`
  is accepted. A scalar comes from the latest run of the broadest query (fewest
  given parameters, page position ignored), as list pages do, so a later
  filtered read no longer replaces the whole list's `total`.

- **Finish check: every command step of the plan ran** (fix-3xts,
  `observation_offloading.finish_check`). When the agent picks `finish`, the
  run's initial plan is read against the outputs it archived, in code (no model
  call): a command step whose command never ran successfully is unmet. Unmet
  steps become one note (at most 1 KB, "Step N (cmd) was not run: run it now.")
  that replaces the finish observation, once per agent run and only with two or
  more iterations left (`fastWorkflowReAct.finish_reviewer`). Every checked
  finish records a `finish_check` offload event. It fails open: no plan, no
  archive or an error means no note.

- **Split settings as one command-form file, `_skills/_split.py`**
  (`workflow_agent.split_settings`, `SplitSettings`): the split stage's
  per-workflow settings are one Python file written like a command file, every
  attribute optional (absent file: the built-in defaults). `class Signature`'s
  docstring (`inspect.cleandoc`) replaces `SplitIntoSubtasks`'s instructions;
  optional pydantic `Signature.Input` (`job_types`, `user_query`) and
  `Signature.Output` (`subtasks`) give a splitter field a new description (a
  field the splitter does not have fails the load); `identifier_patterns` is a
  list of compiled patterns (`[re.compile(r"\bacct_[0-9]+")]`) for the split
  merge and skill-call arguments; `subject_hints` is the referent step's text.
  The file is imported as a skill file is (package-relative imports work); the
  leading `_` keeps it from being loaded as a skill. A file that does not
  import (an invalid pattern raises in `re.compile`) or declares a malformed
  attribute fails the load with a `SkillFormatError` naming it, logged; an
  invalid pattern is no longer skipped with a warning.

- **The skill matcher reads a skill's `plain_utterances` as examples**
  (`workflow_agent._matcher_job_types`, `MATCHER_EXAMPLES = 9`): in the
  `MatchSkill` job-type listing a skill's line is followed by
  `e.g. "...", "..."` with its first nine `Signature.plain_utterances`
  (whitespace collapsed); a skill that declares none keeps its plain
  `- name: description` line, and the `job_types` field description mentions
  examples only when some skill has them. Skills without params now carry
  their utterances too (`SkillFile.plain_utterances`;
  `workflow_agent.skill_utterances(folder)` gives them by name). The
  splitter's job types (`_job_types_for`) stay description-only. No span
  fields change. Nine, not four: measured offline on an 18-skill workflow
  with nine utterances each, nine shown matched 0.863 of a tuning set against
  0.834 with four (skill sub-tasks right 562 vs 552 of 567), for about 1.4k
  more prompt tokens.
- **Skills as Python skill files** (`fastworkflow/skill_commands.py`,
  `workflow_agent.load_skills`): a skill is now a file `_skills/<name>.py`,
  laid out and written like a command file at the root of `_commands`: the
  file stem is the skill's name, so it is a Python identifier (snake_case; a
  stem that is not one fails the load with a `SkillFormatError`), a `.py` file
  whose name starts with `_` is a helper the skill files import
  (`from ._kinds import ...`), not a skill, and a subfolder of `_skills` is
  reserved for a future skill context (like `_commands/<Context>/`): it is
  logged and skipped, its files not loaded. `class Signature` carries the description (its
  docstring, what the skill matcher and the text planner read), a pydantic
  `Input` (`str`, `bool`, `int`, `Literal[...]` for an enum; `Field(default=...,
  description=...)`; the skill options `strip`, `subject`, `verify`, `means`,
  `in_request` in `json_schema_extra`), and optionally
  `plain_utterances`, a `template` (the text the planner falls back to) and
  `validate_extracted_parameters(workflow, command, cmd_parameters,
  command_text=None)`. A module-level `steps(input) -> list[str]` returns the
  steps, written with Python f-strings, `if` and `for n in repeat(input.count)`
  (`repeat` refuses more than `REPEAT_LIMIT`); lookup tables are plain dicts; a
  step `@<skill>` includes a skill without params. A skill
  without params is a file whose `steps()` returns its fixed lines. Without a
  declared `template`, the template is `steps()` rendered on placeholder inputs
  (`skill_commands.placeholder_template`): every combination of `<name>` for a
  str (and "" when that is its default), true and false for a bool, an int's
  default and each enum value, merged into one list where a step not every
  combination writes ends with `(only when ...)` (`person is given`,
  `kind is group`, `not flag`) and the steps of an enum's values are joined
  (`<find_group or find_unit, by kind> query=<name>`); a repeated part shows
  its steps for the int's default, so a skill declares `template` when the
  planner should see how it repeats. The file is imported like a command file
  (package-relative imports work); one that fails to import is logged and
  skipped. The one file replaces SKILL.md (see Removed).
- **Skill arguments go through the command parameter screen**
  (`utils.signatures.screen_parameter_values`, factored out of
  `InputForParamExtraction.validate_parameters`, which calls it unchanged): a
  skill call's `str` and `int` arguments lose quotes and angle brackets around
  the whole value, a placeholder or an out-of-range number (a negative count,
  a declared `ge`/`le` bound) is refused (an optional argument takes its
  default, a required one fails the call), and the extractor's "not given"
  number or a made-up extreme counts as not given. A str argument is found in
  the request with Unicode dashes (a non-breaking hyphen) folded to `-`. A skill
  file's `validate_extracted_parameters` sees the checked arguments
  (`skill_commands.validate_call`; `plan_skill_call` takes `workflow=`) and can
  refuse the call (the planner falls back to the template) or correct a field.
  An argument that only applies when other arguments take some value is reset
  there (`if not cmd_parameters.include_risk: cmd_parameters.owner = ""`); skill
  files have no condition option of their own, and a field declaring `requires`
  in `json_schema_extra` fails the load with a `SkillFormatError` pointing to
  the hook. The rendered template runs the hook on each placeholder combination
  and leaves out the ones it changes, so its `(only when ...)` conditions follow
  the hook; a combination the hook refuses is left out too.
- **Skill-call argument checks** (`fastworkflow/skill_commands.py`), all optional
  per parameter (in a skill file, an `Input` field's `json_schema_extra`), so a
  skill without them plans as before:
  - `verify: true` on a `bool` or `enum` parameter: when Jev is configured, the
    planner path (`plan_skill_call(..., check_lm=...)`, passed by
    `build_query_with_next_steps`) has Jev decide that argument again from the
    request alone (one `Choice` call per verified parameter, in parallel with the
    planner's typed call); a verdict at or above `VERIFY_THRESHOLD` (0.65)
    replaces the planner's argument and stands in for a missing or invalid enum.
    An enum's `means` (a dict) gives Jev each value's meaning (`"group": "a
    group, which people are members of"`). A failed check leaves the planner's argument.
    This targets a wrong record kind for a name ("Payroll" planned as a feed) and
    flags set or left unset against what the request asks.
  - `in_request: true` on an `int` parameter: when the request writes numbers
    (digits or the words zero to twenty) outside the other arguments' text, the
    argument must be one of them, else it takes the only one written (or its
    default when written) and with several others the call falls back; "the
    first two" no longer runs as the default 3.
  - The model's `steps_before` / `steps_after` lose `<...>` placeholders (with
    the `key=` holding one): `list_logins login_uid=<Ann Lee's uid>` becomes
    `list_logins`, which the agent runs on the record in context instead of
    passing the placeholder on as a value. A step holding a uid-shaped token the
    request does not write is dropped; their descriptions now say so.
  - `subject: true` on a `str` parameter: the argument must name a record of
    the request: hold one of its subjects (`split_merge.request_subjects`, with
    the workflow's identifier patterns) whole, or a name-like word, and cut no
    subject. "I" of "Before I approve" or "Payroll" of "Payroll_Approver" is
    refused (an optional argument takes its default, a required one falls back);
    a name cut short mid-word or at a space at a subject's start or end
    ("Payroll_Admin", "Ann" of "Ann Lee") becomes the whole subject. `verify: true` on a subject has Jev choose among the
    request's subjects or "none" (a name/person swap, a right's name planned as
    a group's).
  - Any `str` argument inside a match of the workflow's identifier patterns
    becomes the whole match ("4421" of "acct_4421"); commands loaded by
    `load_skill_commands` carry the patterns (`SkillCommand.identifiers`).
  - An extra step naming a command outside the expansion is kept only when a
    part of the request the expansion leaves open asks for it: a word of the
    command's name (not `open`, `list`, `find`, `uid`, ...) shares a stem with a
    request word no word of the expansion's text shares one with. Navigation and
    lookup steps the model adds around the expansion are dropped.

- **Skills as parameterized commands** (`fastworkflow/skill_commands.py`): a
  skill may declare typed parameters (its skill file's `Input`: `str`, `bool`,
  `int` or an enum, with a default, a description and, for a string, `strip`
  words) that `steps(input)` expands into its steps. The
  description is still what the skill matcher sees; a skill without parameters
  is read as before. An `@<skill>` include is plain text and cannot name a skill
  with params. A malformed skill fails the load with a `SkillFormatError`
  naming its file. On a turn's initial plan, when the sub-task's matched skill
  has params, `build_query_with_next_steps(..., skill_command=...)` asks the
  planner LM for the skill's typed arguments (one `ChainOfThought` over a
  signature built from the params, plus `steps_before` / `steps_after` for parts
  the skill does not cover; same adapter and catalog as the text planner),
  applies the argument guards and expands the skill in code into the numbered
  plan. Guards: a string argument loses quotes and its strip words (a kind word
  appended to a name: "NORTHFEED feed" -> "NORTHFEED", "Ann Lee's login" ->
  "Ann Lee") and must then be written in the request (case-insensitive; the
  request's spelling is used) or the call falls back (an optional one takes its
  default); an optional string repeating another argument takes its default; a
  bool always defaults to false and the signature tells the model to set it only
  when the request asks; an enum must be one of its values; extra steps naming no
  available command, naming the skill, or naming only commands the expansion
  already runs are dropped; a repeat is capped at `REPEAT_LIMIT`. Any failure
  (bad arguments, a guard, an exception) falls back to the text planner with
  the skill's template; replans and hand-off continuations are unchanged.
  `_call_agent_for_query` gets the template and the command from one skill
  match (`workflow_agent.planner_skill_for`; `plan_template_for` is its
  template half). Whether a chosen record kind fits what the find command
  returns is not checked at execution time.
  Span contract: `fw.planner.plan` / `fw.planner.replan` v4 (aggregate 14) add
  `skill`, `skill_args` and `skill_fallback`; `plan_source` may be `skill_call`.

- **Deterministic split merge** (`fastworkflow/split_merge.py`): the
  splitter's sub-tasks now go through `merge_subtasks(user_query, subtasks)`,
  code only, after the model (and its one retry). It drops meta lines ("(No
  tag action is requested.)") and sub-tasks contained in another, joins a
  sub-task that names no subject, or the same subjects as the one before it,
  to that one ("How risky is Ann Lee?" + "Which findings are driving that?"),
  and joins per-record sub-tasks that the request asks to compare ("the
  riskier of Billing and Shipping") into the request's own sentence. When one
  job is left, the request itself runs, verbatim, so qualifiers the model
  dropped ("Do not apply anything") are back. Subjects are found in the text
  alone (names, quoted and back-ticked spans, uids, and the identifiers the
  workflow declares in an optional `_skills/identifier_patterns.txt`, one
  regular expression per line, since identifier formats are the workflow's
  own; "X and Y Ltd" is one unit). A failing merge keeps the model's split.
  Limits on the request that the model dropped ("Apply none of them.", "I do
  not need the full population.") are appended to the sub-tasks they apply to
  (all of them, or the ones carrying the sentence a pointing-back limit
  qualifies), and a name extending another name of the request
  ("Payroll_Approver" of "Payroll") joins that name's sub-task when it points
  back ("among them", in the sub-task or in the request sentence naming it,
  since the model may drop it). A sub-task naming part of the subjects before
  it joins them when it checks a change ("Show me the tags on Ann Lee before
  and after" after "Put `on-leave` on Ann Lee").

- **Split merge, more clean-ups** (`split_merge.py`, code only): a sub-task
  quoting a value the request never gives ("Apply the tag 'offboarding'") is
  dropped; a request sentence naming a record no sub-task names (a job the
  model dropped) is put back, verbatim, in place; a name of a sub-task that is
  a run of words of exactly one request name ("Sons" of "Hart and Sons and
  Sons") stands for that name.

- **Referents written into sub-tasks** (`workflow_agent.resolve_referents`):
  after the merge, a sub-task that leaves some of the request's records
  unnamed gets one Jev call choosing which of them, if any, its "it", "its",
  "that team" or "the customer" stands for, told what the workflow counts as a
  record by its optional `_skills/subject_hints.md` (natural language). The
  choice is written in (`split_merge.with_referent`): the one pointing phrase
  is replaced ("its holders" -> "Payroll_Approver's holders"), or the record is
  appended ("(for Ann Lee)") when there are none or several. Without a Jev key
  nothing changes.

- **Referents next to their description, and cut subjects put back**
  (`split_merge.py`, `workflow_agent.resolve_referents`): when Jev has chosen
  the record a sub-task points back to and the sub-task holds a description
  of it ("the customer `on-leave`"), a second Jev call picks that phrase and
  the record is written right after it ("Tag the customer Ann Lee
  `on-leave`"), instead of only into "its tags". A sub-task taken from a
  request sentence that lost one of that sentence's subjects is replaced by
  the sentence itself, even when another sub-task names the subject.

- **Placeholder arguments refused** (`utils/signatures.py`,
  `InputForParamExtraction.validate_parameters`): a parameter value that is
  one bracketed token is unwrapped when it holds a real value
  (`<9a1b2c3d...>`) and refused, with an error telling the agent to pass the
  real value from an earlier output, when it is only a placeholder
  (`<item_uid>`, `<uid>`). Every way a command gets its parameters passes
  through this check.
  - A value wholly inside one quote pair (straight, typographic, guillemet or
    backtick) loses the quotes: a filter of `"Ann Lee"` runs as `Ann Lee`.
    Quotes the request itself writes inside quotes are kept.
    `unwrap_quoted_value` is public for other callers.
  - A placeholder-shaped value without brackets (`desired-tag`, `your_uid`,
    `uid` for an `item_uid`) that the request never writes is refused the same
    way. In agent mode it is checked against the user's message, not the
    agent's command.
  - The extractor's "not given" marker for an int (`-sys.maxsize`) no longer
    reaches the command on a field with a default or an Optional one: the
    field gets its default (`None`). The same goes for a number the model
    made up (int32/int64 limits, beyond 2**53, inf/nan); on a field with no
    default it is a parameter error. Declared `ge`/`gt`/`le`/`lt` bounds are
    enforced, and a field named as a position, count or size (`start`,
    `offset`, `limit`, `page_size`...) with no declared lower bound may not
    be negative.
  - `fw.nlu.param_extraction` v4 records `unwrapped_quoted_values`,
    `defaulted_sentinel_values` and `out_of_range_values` (aggregate span
    contract 15).

- **The repeat block covers the same call only**: a command that returned
  the same result twice in a row is refused in the next step only when it is
  sent with the same text again; it stays in the available commands, and with
  other arguments (`start=42`, the next page) it runs. Before, the whole
  command was blocked for that step, so its next page could not be read.

- **Composite skills**: a step that is exactly `@<other-skill>` is replaced,
  when the skills load, by that skill's steps (recursively, in order; its
  description is not included). The
  splitter's job types, the skill matcher and the planner template see only
  the expanded steps. An unknown skill or a cycle raises `ValueError` naming
  the include chain. A skill without `@` lines loads as before.

### Removed

- **Skill `repeat()` / `REPEAT_LIMIT` and `@<skill>` include lines**
  (fix-ajir.18): a skill file's `steps()` loops with `range()` (bound the int
  field with `le`) and shares steps through a helper module
  (`_skills/_<name>.py`). A step written `@<name>` is now plain text.

- **`run_fastapi_mcp --keep_dspy_history`** (fix-50pd): the bounded-history
  "inspect mode" of the server's DSPy memory policy is gone, with
  `install_policy(keep_history=..., history_entries=...)`,
  `INSPECT_POLICY_SETTINGS`, `BoundedHistory` and `SERVER_DSPY_HISTORY_ENTRIES`.
  `fw.llm.call` usage, cost and cache status come from each call's own result
  (fix-sgsp), so DSPy history is now off in every server, the chatbot's
  included; the chatbot no longer passes the flag, and passing it is an error.

- **`FW_SPLIT_REQUESTS`**: request splitting is always on. The switch already
  defaulted to `1` and was on in every evaluation run; `"0"` no longer turns
  it off (the variable is not read), and the commented line is gone from
  `examples/fastworkflow.env`.

- **The finish check's model pass and its switch** (fix-3xts follow-up,
  fix-uyab): the one model call that judged whether each executed step's
  output held what the step asked for is gone, with its filters, prompt and
  "Step N asks for ..." note lines. In a live run it helped alone 3 times, was
  ignored as a false alarm 19 times and led to 1 made-up answer. Gone with it:
  `FW_FINISH_CHECK` (the command check is always on when a run has a plan; no
  `commands`/`full` modes and no off switch), the `LLM_FINISH_CHECK` model
  role and `LITELLM_API_KEY_FINISH_CHECK`, and the `context_budget.FINISH_CHECK`
  budget (`finish_check_max_bytes`, `FW_FINISH_CHECK_MAX_BYTES`), which sized
  only that model's prompt. `finish_check` events no longer carry
  `unmet_by_model`, `vetoes`, `model_called`, `prompt_bytes`, `error` or
  `usage`.

- **The split stage's text files**: `_skills/split_instructions.md`,
  `_skills/identifier_patterns.txt` and `_skills/subject_hints.md` are no
  longer read (nor `workflow_agent._skill_file_text`); their contents go in
  `_skills/_split.py` (`Signature` docstring, `identifier_patterns`,
  `subject_hints`), word for word.

- **SKILL.md skills**: `_skills/<name>/SKILL.md` is no longer read; a skill is
  the file `_skills/<name>.py` only. This covers the plain form of 3.6.0 (the
  description on the first line, one step per line), which becomes a file with
  `class Signature` (the description as its docstring) and
  `def steps(): return [...]` (the step lines), and the front-matter form with
  its `{param}` / `[if]` / `[repeat]` step language, which was never released.
  A folder under `_skills` is not a skill (see the skill-context note above).

### Changed

- **The finish check runs on every run that has a plan** (fix-3xts follow-up):
  it no longer waits for `FW_FINISH_CHECK`, and it reads plan steps against
  their commands' runs only. A run with no durable archive gets no note
  (`reason: "no archive"`) instead of every step reported as not run.

- **Finish check: fewer false "not run" notes** (fix-uyab, part):
  an archived run counts for the command its text resolved to, read from the
  session's action log (`resolved_command_names`; `check`, `attribute_rows`
  and `review_finish` take `resolved=`), so an alias satisfies its command's
  step; a step with "(skip when ...)", "(skip if ...)" or "(skip unless ...)"
  after its command is conditional; and a step that writes an argument right
  after its command (`command key=<...>`) is not satisfied by a run of the
  command with no arguments at all.

- **A skill call's extra step is kept for a synonym its command uses**
  (`fix-9x93`; `skill_commands.command_texts`, `command_descriptions`,
  `command_utterances`): `filter_extra_steps` (`texts=`) matches the request's
  open words against a command's name and the words that recur in at least two
  of its texts (the first sentence of its catalog description and each of its
  `plain_utterances`; a command with one text gives that text's words), not
  the name alone, so "every invoice" now keeps `list_bills` whose utterances
  say "invoices". `plan_skill_call` builds the texts from the catalog and the
  workflow's routing definition (no model call). "please", "want", "need" and
  "has" joined the filter's stop words.

- **Every experiment pins a benchmark version** (`fix-ujat`): the store and
  `ExperimentController.create_experiment` refuse an experiment without
  `benchmark_id`, `benchmark_version` and `benchmark_digest_sha256`
  (`MissingBenchmarkPin`). An unpinned experiment could carry any task ids
  and sat in the Benchmarks tab beside the real benchmarks, with a contest
  and a winner of its own. Ad-hoc work is plain conversations (`/initialize`
  with no experiment bootstrap), listed under **ad-hoc conversations**.
  Removed: the "Experiments without a benchmark" rail branch and the
  ad-hoc comparison group (`selection.GROUP_ADHOC`);
  `comparison_group_identity` refuses an unpinned row. An older store's
  unpinned experiments are left off the rail and their turns list with the
  ad-hoc conversations.

- **Experiments driven by a user simulator** (`fix-va9v`): an experiment can
  be created with `user_simulator: true` (UI registration
  `POST /api/benchmarks/<id>/experiments`, duplicate, or
  `ExperimentController.create_experiment(user_simulator=True)`). Its
  simulator turns each task prompt into its own utterances, so its runs'
  first messages are not checked against the prompts: validity reads
  `simulated_user` instead of valid | invalid. Completeness is still checked.
  The flag is set once, at creation; a registered experiment takes it from
  its registration. It shows read-only on the experiment and registration
  pages and in their API responses. **Schema v10** adds
  `experiments.user_simulator` and `experiment_registrations.user_simulator`.
  A v9 store is refused, as with every older schema, and is not migrated.

- **Benchmark experiments are checked against their version** (`fix-mgoi`):
  every experiment that names a benchmark version must declare exactly that
  version's task IDs, whether or not it was registered in the UI (before, only
  UI-registered ones were checked). Each recorded experiment now reports two
  verdicts, derived from its attempts and recorded turns: **complete |
  incomplete** (every benchmark task got its declared runs) and **valid |
  invalid** (every run's first message matched its task's saved prompt; tasks
  without a saved prompt are not checked). Both show on the benchmark page's
  experiment cards with "N of M tasks", and on the experiment page with the
  runs that are short or that sent another prompt. A store verdict of
  `invalid` for missing runs (`attempt_shortfall`, `never_completed`) now
  reads "incomplete" in the UI.

- **Experiments and winners are partitioned by benchmark version**
  (`fix-873a`): each version of a benchmark is its own contest, so the Studio
  lists one version's experiments at a time (rail: benchmark → version →
  experiments) and each version has its own winner. The winner is now plain:
  `POST /api/experiments/<id>/winner` sets it, `DELETE` removes it, and having
  no winner is normal. Removed: the automatic first-experiment winner, the
  `keep`/`undecided` decisions and `expected_selection_id` on the winner,
  `POST .../winner/decisions`, `GET /api/benchmarks/<id>/experiments` (now
  `.../versions/<v>/experiments`), and the refusal to delete the current
  winner (deleting it leaves the version with no winner). Existing live DBs
  are not migrated: experiments already in a cross-version group stay there.
  `create_experiment`'s
  `initialize_winner` is renamed `join_comparison_group`. Per-task best runs
  are unchanged.

- **The observability record bounds itself in long-lived servers**
  (`fix-nczr`): the writer re-runs the retention prune hourly instead of only
  at recorder startup, and once only turn records are left over
  `FW_OBS_DB_MAX_BYTES` the oldest finished non-experiment turns are evicted
  whole, so the cap is a real bound. No new settings. A malformed `FW_OBS_*`
  integer now logs a warning, and so does an unsealed experiment holding the
  file over the cap.

### Fixed

- **One malformed skill file no longer fails every turn** (fix-ajir.3). A skill
  file whose declaration is malformed (or whose name is not an identifier) is
  logged as an error and skipped, as one that does not import already was; the
  workflow's other skills load, and the result is kept for the process instead
  of every skill being imported again on each turn. `fastworkflow skill preview`
  (with or without `--check`) fails naming each skipped file and its error.

- **Command modules that import other command modules load safely from
  several threads** (fix-nhhq, `python_utils.get_module`). Loading a command
  file by path always executed a fresh copy and swapped it into `sys.modules`
  without Python's import lock, so a concurrent `from ..Ctx.mod import X` in
  another command module could get the half-executed copy ("cannot import name
  X"); and a load that failed left its broken module in `sys.modules`, so every
  later import of it failed the same way for the rest of the process. The
  loader now takes the module's import lock, reuses a module already imported
  from the same file (re-executing it in place only if the file changed on
  disk), marks a fresh module initialising while it runs, and removes it from
  `sys.modules` if it fails.

- **The size-cap prune no longer deletes a running turn's offload evidence**
  (fix-q2fy, `ObservabilityStore.prune`). A turn's record is written when it
  ends, so the cap now evicts offload evidence only of turns with a finished
  turn record; evidence of a turn that may still be running (or suspended)
  goes only by the age horizon, which also clears a crashed turn's orphans.
  The bytes it holds while the DB stays over the cap are reported as
  `in_flight_bytes` in the `prune_over_cap_bound_bytes` diagnostic. No schema
  change.

## 3.6.0 — DSPy 3.4, request splitting, workflow facts on classes

### Added

- **Request splitting** (`b647f25`): the agent splits a request into
  independent sub-tasks (workflow skills from `_skills/` as job types,
  optional `_skills/split_instructions.md`) and runs them one by one; the
  25-step cap hands a sub-task to a fresh executor (at most 2 hand-offs).
- **Execution guards**: multi-command calls are refused, a command that
  returns the same result again is blocked for a step, and unparsable
  replies and LLM timeouts are recovered as bad steps.
- **`LLM_TIMEOUT_SECONDS`** bounds each LLM call (default 120); parameter
  validation hooks may declare `command_text` to receive the command as written.

### Changed

- **dspy is required at `>=3.4.0,<3.5`** with the `typesafe` extra. Every
  module imports litellm before dspy, and `import fastworkflow` repairs
  DSPy's lazy `openai` stand-in when dspy was imported first, so a bare
  `import fastworkflow` no longer loads litellm. Provider cache misses are
  recorded as misses again (DSPy 3.4 flags only hits).
- **`workflow_runtime.json` is retired** (`c4c901b`): a context's callback
  class declares `occupiable = False`, a command's ResponseGenerator declares
  `descends_to` / `descend_parameter`; workflow identity is the
  framework-computed content hash.
- **Observation offloading is simplified** (`4257d2a`): observations are
  saved once per step and offloaded when the whole executor prompt exceeds
  `FW_TRAJECTORY_MAX_BYTES`. **Observability schema v9**: offload rows lose
  `scope_id` / `offload_order`, `offload_subjects` is dropped.
- **The initial planner gets every prior turn summary** instead of the last
  trajectory; the planner and agent tools get prescriptive context navigation.
- `POST /post_feedback` and `GET /feedback` accept a retired execution key
  ([R9]); `/post_feedback` returns the turn's comments.
- **Observation handles use the ReAct step index** (`O0`, `O1`, …): execute ordinals,
  `ordinal_offset`, forced-replan continuation segments, and `alias_conflict` are
  removed; max-iters exhaustion follows the base ReAct loop.

### Security

- Floors: `transformers>=5.17.0` (CVE-2026-75104, CVE-2026-80047; pulls
  tokenizers 0.23.3), `anyio>=4.14.2` (CVE-2026-63374, CVE-2026-64847),
  `fsspec>=2026.6.0` (CVE-2026-104851), `multidict>=6.9.1` (CVE-2026-104874).
- OpenVEX v7 (`security/vex/fastworkflow.openvex.json`) marks not_affected:
  litellm CVE-2026-12772/12795/12796 (proxy-only, fixed after 1.82.2), the
  transformers ZDI checkpoint batch CVE-2025-14921/14924/14926–14930, and
  torch CVE-2026-4538, CVE-2025-3000/3121/3136/4287/55554.

Releases before 3.4.0 were announced in their merge-commit subjects
(`feat: v3.2.0 — observability store, chatbot debug UI, …`) and are recoverable
with `git tag` and `git log --first-parent main`. This file starts at 3.4.0; it
does not backfill them.

## 3.5.1 — agent observation and tool-selection fixes, exact prompts in the debug UI

### Added

- **The exact prompt of an over-cap LLM call** (`fix-ra63`): an `fw.llm.call`
  whose `messages` exceed the 16 KB attribute cap is split at its DSPy field
  markers and each distinct piece is stored once per turn in a new
  `prompt_slots` table (additive; no schema version bump). The span carries a
  `prompt_slots_ref`, `ObservabilityStore.prompt_as_sent` rebuilds the messages
  and checks them against the recorded digest, and the debug UI shows them
  under "LLM input as sent" (`/api/prompt/…`).
  Pieces are credential-scrubbed and removed by prune,
  forget-channel, clear-conversations and restart-attempt.

### Changed

- **Capture profiles and context handles are gone**: `FW_OBS_CAPTURE_PROFILE`
  and the manifest `capture` field are removed (a manifest that still declares
  `capture` or `handle_projector` fails validation); recorded values get the
  credential scrub only. Span contract 10 (`fw.command.execute` v4,
  `fw.agent.tool_call` v2) records `context_before` / `context_after` as the
  plain context type name instead of a handle mapping. **Observability schema v8**: `experiments` and
  `offload_evidence` lose `capture_profile` / `capture_policy_version`, and the
  `sealed_turn_comments` control table is dropped (see Removed); a v7
  store is refused on open like every older one (a one-off
  `migrate_obs_v7_to_v8.py` at the repo root converts existing stores).
- **The observation line says where a command ran and whether it moved the
  context** (#86): `Observation O3 (execute_workflow_query ran in Directory)`,
  with `; and resulted in a context change` when it did. The root context
  prints as `global`, and a command rejected before dispatch still gets its
  clause.
- **dspy is required at `>=3.3.0,<3.4`** (`fix-5sm7`, `fix-cbjx`): 3.3.0 added
  the JSON-retry hook that keeps the available commands in the agent's retry
  prompt, and 3.4.0's lazy `openai` import breaks `import dspy; import litellm`.

### Removed

- **The workspace viewer and the blinded review sidecar** (`fix-0gh0`):
  `run_chatbot --workspace-manifest`, the picker's workspace-manifest list,
  `/api/workspace/*`, `/api/select_workspace`, `/api/review/*`, `/trace`,
  `fastworkflow/review/` and `observability/workspace.py` are gone, as are the
  navigation nodes' `source` field and the feedback response's `read_only` /
  `annotated` flags. Comments always go to the workflow's live database, so
  the `sealed_turn_comments` table that only workspace mode wrote is dropped.
  Experiment sealing is unchanged: `seal_workspace_evidence` still writes
  `sealed_archives`, and `selection.store_for` still reads a sealed member's
  archive.
- **`fw.agent.step` contract v4**: adds `repaired_tool_name`.

### Fixed

- **A workflow command named as the agent's tool runs** (`fix-8q7a`): when the
  model writes `next_tool_name: open_directory` and that command is listed for
  the current context, the step runs as the `execute_workflow_query` call it
  meant instead of becoming an invalid-tool step (three of which end a turn).
- **A moved or reinstalled fastworkflow no longer breaks every turn**
  (`fix-u2zw`): a persisted `command_directory.json` whose recorded module
  paths no longer exist is rebuilt instead of failing with "Could not import
  module from path".
- **`finish` called with arguments finishes** (`fix-njhf`): arguments the model
  passes to `finish` (`analysis=...`, `command=...`) are ignored instead of
  raising "Arg X is not in the tool's args", so the finish check runs.

## 3.5.0 — observability debug UI hardening and winner-selection fixes

The adversarial review of the `run_chatbot` observability UI (`fix-hzux`) is
resolved, and experiment winner selection no longer has a dead end on deletion
or recovery.

### Added

- **Process logs in the UI** (`fix-hzux.3`): the spawned FastAPI server's and
  training's logs are reachable from the debug UI instead of "the terminal".
- **Parent watch for the FastAPI child** (`fix-hzux.4`):
  `run_fastapi_mcp --parent_pid` exits when `run_chatbot` dies uncleanly
  (SIGHUP, SIGKILL, crash), so the next launch no longer finds an orphan on its
  port.
- **Derived-turn cache** (`fastworkflow.observability.derived_cache`,
  `fix-hzux.1`): `/api/turns` reuses per-turn derivations keyed on span count
  and last row, so a warm scan recomputes only turns that gained spans.
- **`fastworkflow.observability.turn_derivations`** (`fix-cnoc`): the pure turn
  derivations (`execution_ledger`, `turn_decision_signals`,
  `is_low_confidence`, LLM-call cost helpers) live in core observability, with
  no dependency on the debug UI. `cost_rollup` lives in
  `fastworkflow.observability.comparison`. The old `run_chatbot` import paths
  re-export the same objects.

### Changed

- **`run_chatbot` split into per-domain modules** (`fix-hzux.11`): `server.py`
  and `selection_api.py` become `handler_*` modules behind one route table and
  one auth chokepoint (`fix-hzux.10`).
- **The SPA ships as ordered source parts** (`fix-hzux.12`):
  `run_chatbot/static/index.html` is replaced by `static/src/*`, concatenated
  at serve time; the wheel includes `static/src/*`.
- **The "stdlib-only" chatbot is stdlib-only** (`fix-hzux.17`): `fastworkflow`
  and `fastworkflow.experiment` import their heavy members lazily (PEP 562), so
  starting the chatbot no longer loads dspy, litellm or fastapi.
- **Navigation and the workflow picker are lighter** (`fix-hzux.6`, `.7`):
  `/api/navigation` sends summaries with an ETag, and the picker stops polling
  when it is not shown.
- **Workflow activation no longer blocks a request for seconds** or mutates
  session state other threads read without a lock (`fix-hzux.5`).
- **Initial winner election picks the group's oldest member** (`fix-kkod`), not
  whichever registration happens to trigger it.
- **A group's sole experiment can be deleted even when it is the automatic
  winner** (`fix-65ik`, part): the pointer is withdrawn with an ordinary
  `retire` decision, and the next experiment of the group is elected
  automatically. A winner whose group has other members is still refused (409).
- **Clearer selection errors**: `ExperimentSelected` and `NoCurrentSelection`
  say why and what to do.

### Fixed

- **Deleted unused experiments stay out of winner selection** (`fix-jfy5`).
- **FastAPI feedback enforces channel ownership** on reads and writes; an
  unknown turn and another channel's turn get the same 404 (`fix-bnym`).
- **FastAPI feedback resolves recorded pass selectors** and refuses spans
  outside the named pass (`fix-jxkk`).
- **Paired task feedback filters on the queried side's attempt** (`fix-ptu1`).
- **Leaked SQLite connections**: per-request observability store connections
  are closed deterministically instead of by cyclic GC (`fix-hzux.8`).
- **Request bodies are size-capped**, and a negative `Content-Length` no longer
  reads to EOF (`fix-hzux.9`).
- **`run_chatbot` reports its own failures**: access log restored,
  `logger.exception` on 500s (`fix-hzux.2`); the SPA no longer swallows errors
  in empty catches (`fix-hzux.15`).
- **Trace view and transcript stay bounded**: multi-MB span payloads render
  lazily and superseded fetches are cancelled (`fix-hzux.16`).
- **The debug rail is keyboard-operable** (`fix-hzux.14`).

### Known limits

- A permanently lost evidence store still leaves new comparison groups in its
  workflow without a winner, and a winner-less group cannot be promoted into
  (`fix-65ik`, open).

## 3.4.0 — observation offloading and search

**Observation offloading and answer-time rehydration become the framework's
behaviour for every workflow.** They are no longer modes a deployment opts into.
A fastWorkflow tool agent compacts its trajectory, keeps every observation
reachable, and answers over the evidence behind its labels rather than over the
labels themselves.

### Added

- **Observation offloading** (`fastworkflow.observation_offloading`): canonical
  `O{n}` aliases on every execute observation, an eager SQLite archive, a packed
  trajectory target with offload labels, `search_memory` over one stored
  observation, and segmented continuation with forced replans.
- **Context-instance line** (`fastworkflow.context_identity`): every execute
  observation names the context instance the command ran in, so a listing
  produced inside a context is still attributable to that instance by a reader
  that cannot use the order of the commands.
- **Answer-time rehydration** (`fastworkflow.answer_rehydration`): the extract
  call is given its own copy of the trajectory with the evidence behind offload
  labels put back, under a byte budget, so the answer is written over evidence
  rather than over pointers.
- **Finish reminder**: a `finish` action that never opened a named item of the
  request goes back to the loop once, and only while iterations remain.
  (Replaced in this release by the finish-time execution check below; the
  roster nudge that implemented it, `fastworkflow.answer_coverage`, is removed.)
- **Finish-time execution check** (`observation_offloading/finish_check.py`,
  fix-4dsr): when the agent chooses `finish`, a decision model judges, for
  every step of the turn's initial plan and every subject the request names,
  whether the turn's record shows the step carried out; unexecuted steps are
  named in one note and the agent returns to the loop -- once per turn, only
  with iterations left, and it may finish anyway. It checks execution, not
  whether the request was answered. Off unless `FW_FINISH_CHECK=jev` and
  `JEV_API_KEY` are both set (it sends the request, the plan and a redacted
  summary of every step to TypeSafe); fails open to an unchecked finish; one
  `finish_check` offload event per finish and an `fw.finish_check` span.
  Measured offline on 82 recorded downstream-workflow attempts it was not tuned on (910
  labelled step x subject pairs, pre-registered): precision 0.77, recall 0.89.
  Install with the `jev` extra.
- **Structured plan** (`fastworkflow.turn_plan`): with the finish check on, the
  planner returns its steps (commands, sub-steps, optional and user-gated
  flags) and the request's subjects as data, and the agent still reads a
  numbered list. A plan that does not parse falls back to the plain-text
  planner. Without the check the planner is unchanged, because the structured
  call is slower (median 12.1 s against 3.7 s on three todo-list requests with
  `cerebras/gpt-oss-120b`).
- **Context budgets** (`fastworkflow.context_budget`): one input — the model's
  context window — and every byte budget derived from it as a fixed fraction.
  `budget_provenance()` returns the input, its source and every budget.
- **Searches answered without the search model.** An archived observation of at
  most 256 bytes is returned verbatim with up to three better-matching handles
  of the turn (status `short_verbatim`). A request for every row of a listing is
  answered by copying the rows (`observation_offloading/listing.py`: aligned,
  markdown and tab-separated listings, served only when the parse is provably
  complete), cut at the answer bound with a line saying how many were shown
  (status `rows_served`).
- **Optional search router** (`observation_offloading/search_router.py`, the
  `jev` extra; `router` is kept as a deprecated alias of it): a decision-model call (TypeSafe Jev) that decides whether a
  search over a listing wants every row. Off unless `FW_SEARCH_ROUTER=jev` and
  `JEV_API_KEY` are set; what it sends is redacted by the capture policy; one
  attempt, a 2-second timeout, and it fails open to the search model. Every
  search event records its verdict. A workflow may add examples in
  `search_router_examples.json`.
- **`fw.search.route` span** (contract v1) for each routing call under a traced
  turn: the verdict, latency and tokens, never the question or the evidence.

### Changed

- **Structured planning is disabled** (2026-09-28, owner decision; supersedes
  the "Structured plan" entry under Added and the later structured-planner
  notes in this release). The planner always makes a plain-text plan, with the
  finish check on or off; with it on, the check's plan is `parse_text_plan`'s
  (`source="text"`, no subjects). The structured signatures, the
  `STRUCTURED_PLAN_GUIDE` prompt, `turn_plan.render`, the structured adapter,
  the zero-steps / parse-error fallback and the planner span's subject
  redaction are commented out, not deleted, and not behind a flag: nothing
  re-enables them. `fw.planner.plan` / `.replan` keep contract v2 (and
  `SPAN_CONTRACT_VERSION` 8): `plan_source` is only `text` or `none`, and
  `subjects` is always `[]`. The `TurnPlan` / `PlanStep` / `PlanSubject`
  models, the plan's persistence across a resume (fix-ju1v) and the
  adapter's `use_json_adapter_fallback` parameter (fix-6jzi) are unchanged.
  The tests of the structured path are skipped, not removed.
- **Known-name guard**: a known command name is never answered by a context
  that does not own it; the declining prediction carries a hint naming where the
  command lives. When no context on the chain owns it, that hint is the whole
  response and the next message goes through ordinary intent detection, rather
  than `you_misunderstood` and its clarification stage, which only matches the
  current context's commands and so could not route the command the hint names.
  The guard also covers a reply to `you_misunderstood`, which is matched against
  the same full command set; the ambiguity stage, which matches a short
  suggestion list, is still excluded. With several owners, the hint names each
  entering command beside the context it enters.
- **Threshold separation**: `write_ambiguity_thresholds` is the single writer
  for both ambiguity files and establishes a non-empty ambiguity band where the
  artifacts are produced, with `TIER_AMBIGUITY_MIN_SEPARATION` and
  `SINGLE_LABEL_RESOLUTION_FLOOR`. A tier threshold of exactly 1.0, which a
  float32-saturated sweep can pick, publishes the flat 0.99 cap instead of
  aborting the workflow's training run.
- **Workflow fingerprint scope rule v2**: a root `benchmarks/` tree and the
  runtime observability store leave `workflow_content_entries`.
  `WORKFLOW_SCOPE_RULE_VERSION` 1 → 2, so a v1 declaration reads as
  `incomparable`, not stale.
- **Synthetic utterance generation** sends `temperature` only; `top_p` is gone,
  because current Bedrock Claude models reject both together and every
  generation call was a hard `BadRequestError`.
- `fw.nlu.intent` span contract v2 → v3.
- `fw.command.execute` span contract v2 → v3: the four auto-navigation
  attributes are gone with the two-step dispatch that wrote them.
- The aggregate span-contract version is 5. Version 4's note named
  auto-navigation keys that `SPAN_CONTRACTS` does not contain; 5 is the number
  that matches those contracts. It became 6 when `fw.search.route` was added
  (no existing span changed), 7 with `fw.finish_check` and `fw.agent.step` v3,
  and is now 8: `fw.planner.plan` / `fw.planner.replan` v2 (fix-3x76).
- **Offload labels say the evidence comes back.** A label reads "Offloaded
  observation O… returned by … It is restored in full when the final answer is
  written, so search it with search_memory only for a value you need for your
  next step", and the agent's instructions say the same. The earlier wording,
  which opened with an instruction to search, is still recognised. In recorded
  runs this cut requests to search whole tables by 70-95%. (Since fix-2hxv a
  label ends with the short "Restored in full for the final answer."; the
  promise and the search guidance are stated once, in the agent's instructions
  and the `search_memory` description. Both earlier wordings still parse.)
- **Answers about an incomplete listing name a real narrowing input.** The
  search signature gains a `narrowing` input listing the producing command's
  optional, non-selecting inputs; an answer that needs rows not shown ends with
  `To reach them: <command> <input>=<value>`, or says a narrower re-run is
  needed when the command declares none.
- **Continuation allows 3 forced replans** (4 segments, a 100-step ceiling),
  up from 2 (3 segments, 75 steps). `FW_MAX_FORCED_REPLANS` moves it (see
  below).
- **Intent `signal_version`** no longer carries a threshold-semantics segment.
  It now reads `intent-classifier/<artifact version>/...`, so a version string
  identifies the artifact behind a signal and nothing else.
- **Offloading evidence lives in the observability store.** Archived
  observations, their subjects and the offloading runtime's diagnostic events
  are rows of `offload_evidence`, `offload_subjects` and `offload_events` in the
  workflow's `observability.sqlite3` (feature markers `offload_evidence_v1`,
  `offload_events_v1`; no schema-version bump). They are keyed by turn, and
  `forget_channel`, Clear conversations and retention pruning delete them in
  the same transactions that delete the turn record. Events are read back with
  `ObservabilityStore.offload_events(...)`.
- **Evidence is redacted when it is written.** With
  `FW_OFFLOAD_EVIDENCE_REDACTION=on` (the default) responses and event text are
  stored as the trace sink's credential scrub and capture policy leave them;
  `off` stores them verbatim for development. The process running a turn keeps
  the raw text of its redacted observations in memory until the turn is over,
  so the agent's own reads stay exact; a turn resumed in another process reads
  the redacted text.
- **Observability recording is always on**, for fastWorkflow's entry points and
  for programs that embed the library alike. The database is owner-only (0600
  file, 0700 directory) and pruned by `FW_OBS_RETENTION_DAYS` and
  `FW_OBS_DB_MAX_BYTES`. `get_observability_sink()` no longer takes
  `entry_point`, and returns `None` only when the store cannot be opened.
- **An execution context opens its own sink.** A `WorkflowExecutionContext`
  given no sink opens the bound app workflow's sink in `bind_app_workflow()`,
  and follows a rebind to another workflow's database. A sink passed to the
  constructor or to `set_trace_sink()` is never replaced, and an explicit
  `tracing.NoOpTraceSink()` records no spans or turn records;
  `set_trace_sink(None)` returns the context to its automatic sink. Offloading
  evidence is written to the workflow's database whatever the sink, and the
  offloading archive prunes that database once per process when it opens it.
- The `search_memory` input bound has no tuning override; it is derived from the
  search model's context window only. It is a quarter of that window (131,072
  bytes at the reference window), up from 3/128 (12,288 bytes). (Superseded
  in part by fix-deus below: the derived value is now capped at 131,072 bytes
  and `FW_SEARCH_OBSERVATION_MAX_BYTES` overrides it.)
- **An observability database from an older schema version is replaced, not
  refused.** The store has not shipped in a release before, so such a file can
  only be a local development database: opening it with the writer deletes it
  (and its `-wal`/`-shm`) and creates a fresh store in place, with a warning
  naming the path and both versions. A database from a newer build is still
  refused and never touched, and the read-only viewer refuses an older one
  without deleting it.
- **The Jev features share one client module** (`observation_offloading/jev_client.py`,
  fix-xeg1): the optional SDK import, `JEV_API_KEY`, the default model,
  redaction of what is sent, one-attempt clients and a per-process cache, for
  both the search router and the finish check.
- **A Jev feature that is asked for but cannot start says so** (fix-jdcm).
  `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` set to anything other than `jev`, set
  without `typesafe-sdk` installed, or set without `JEV_API_KEY`, logs one
  warning per cause per process and the feature stays off; an unset flag is
  still silent. Clients are cached by a fingerprint of the key, never the key,
  so a changed key builds a new client instead of reusing the old one. (More
  causes since fix-2so9 and fix-7tz5, and a flag may name a registered
  provider since fix-5uva; see below.)
- **The forced-replan bound is a setting** (fix-vd5c): `FW_MAX_FORCED_REPLANS`
  overrides the default of 3, clamped to `[0, 10]`; a non-integer is ignored,
  with a warning either way. `forced_replan` and `forced_replan_wall` events
  carry `reached_limit`, and `agent_installed` records the bound in force. The
  default rests on one downstream-workflow task, not on a measured tail across workflows.
- **Planner spans say which planner produced the plan** (fix-3x76):
  `fw.planner.plan` / `fw.planner.replan` v2 add `plan_source` (`structured`,
  `text`, `text_fallback`, `none`) and the structured plan's `subjects`,
  redacted by the capture policy. A `finish_check` event also carries a score
  table, one entry per checked step x subject (x part), capped at 200.
- **`finish_check` events are recorded for every finish an attached check
  sees** (fix-7pba), with a reason even when nothing was asked: `disabled`
  (`FW_EVAL_FINISH_REMINDERS=0`), `cap reached`, `nothing to check` (every step
  optional or user-gated), `no plan`, `no room to act`, `error`. The step count
  the note states includes the room later forced-replan segments still hold
  (`shown_iterations_left`); the two-iteration gate is still the current
  segment's.
- **Stored `finish_check` events name steps and subjects by index**
  (fix-7scj): `flagged_steps` holds the step index, the subject's index into the
  plan's subjects, its kind and `p_unmet`, never plan-step text or subject
  names.
- **Search events say which path a search took** (fix-wheb): `router_enabled`,
  `listing_parsed`, `listing_shape`, `listing_skip_reason` and `for_report` on
  every search event, so a `router` of `None` is never ambiguous;
  `related_scores` and `subject_recorded` on `short_verbatim`;
  `served_over_bound` and `trailing_lines_dropped` on `rows_served`. The
  search-observation budget moves into `context_budget` as
  `SEARCH_OBSERVATION`, and `budget_provenance()` reports it as
  `search_observation_max_bytes` beside `search_window_tokens` and
  `search_window_source`, the window it is cut from.
- **The `search_memory` description matches the path the agent has**
  (fix-8uj0): it is built with the agent, states the read bound as the byte
  count the search will apply, and, when a router is attached, says an all-rows
  search may be answered by copying rows instead of forbidding such searches.
  The router's `for_report` verdict is recorded and does not change the path.
- **Offload labels and served listings are shorter** (fix-2hxv): both end with
  "Restored in full for the final answer." instead of the full restore promise,
  which cost ~100 bytes per label and made fewer observations worth offloading.
  (Since fix-94m9 the mark is "Normally restored for the final answer."; see
  below.)
- **Narrowing inputs include defaulted ones** (fix-swqi): an input with a
  non-None default (`limit: int = 50`) is offered as a way to narrow a listing,
  not only one typed `Optional`. A required field's sentinel default
  (`NOT_FOUND`, `INVALID_INT_VALUE`, `INVALID_FLOAT_VALUE`) does not count.
- **Fewer archive reads** (fix-314g): `list_summaries` measures stored rows
  with `length()` on the BLOB, without reading their content, and an alias the
  archive has no subject for is remembered (bounded) instead of re-read on every
  related-handle lookup.
- The settings above, `FW_SEARCH_ROUTER_MODEL`, `FW_FINISH_CHECK_MODEL` and
  `JEV_API_KEY` are in the example env templates, commented out, with a
  data-egress warning (fix-9l9e); stale docs, comments and this changelog's
  span-contract number are corrected (fix-42ig).
- **Vendor calls have hard wall-clock bounds** (fix-i94q): the router's 2 s,
  the finish check's 4 s per call and 8 s per check are enforced during the
  call, not checked before it, and the check's 8 s now include building the
  ledger. Each call runs on its own daemon thread, at most 4 process-wide; a
  call with every worker busy fails at once as `VendorBusy`, one cut off at its
  cap is `CallTimedOut`, one cut off by a deadline or budget is `OutOfTime`. An
  abandoned request is never retried and never delays interpreter exit, but
  may still be billed vendor-side. The SDK's own timeout is the cap plus 0.5 s.
- **One vendor budget per turn** (fix-sotm): 10 s of time actually spent on
  vendor calls, router and finish check together, and at most 3 routing calls.
  Past either, `route` returns `router_budget` with no call and no span, the
  search model answers, and the search records `listing_skip_reason`
  `router_budget`. `vendor_ms` (the turn's running total) is on every router
  record and on a checked `finish_check` event.
- **The Jev endpoint is `FW_JEV_BASE_URL`** (fix-2so9): default
  `https://api.typesafe.ai`; https only, or plain http to a loopback address;
  no credentials, query or fragment. A rejected value warns once, without the
  URL, and turns both features off; a non-default host is logged once. The
  SDK's `TYPESAFE_BASE_URL` is ignored with one warning, so it can no longer
  redirect the key and payloads.
- **`typesafe-sdk` is pinned exactly to 0.7.2** (fix-em0l), with an SDK
  contract test on a loopback stand-in as the review gate for a bump.
- **Jev failures are diagnosable** (fix-conv): an `error` `finish_check` event
  and a router error record carry `error_status`, `error_request_id`,
  `error_code` (one machine-readable token from the body, never its text) and
  `error_stage` (`ledger`, `request`, `budget`, `check`, `note` on the check;
  `redaction`, `request`, `budget` on the router). A failure is warned about at
  most once per 5 minutes per (feature, error type, HTTP status), counting the
  ones not logged, instead of once per process; a ledger-build failure is
  logged with its traceback. No span-contract change.
- **The Jev features stay off when the capture policy would withhold what they
  send** (fix-7tz5): under a capture profile that withholds command output
  (e.g. `evidence`), an unknown profile, or
  `FW_OFFLOAD_EVIDENCE_REDACTION=off`, one warning per cause. A declared field
  policy that keeps observations whole is still allowed. A value the policy
  withholds at call time sends nothing: the finish check records the new
  reason `policy_withheld` (unwarned, `requests` 0) and the router returns the
  error `policy_withheld` (`error_stage` `redaction`).
- **One decision says whether the finish check is on for a plan** (fix-rm98):
  `workflow_agent.finish_check_active(agent)`, read by the planner. With
  `FW_EVAL_FINISH_REMINDERS=0` nothing the check adds runs -- no note and no
  structured plan -- while the checker stays attached and records `disabled`.
  The switch is read from the workflow's `fastworkflow.env` first, then the
  process; any other set value still fails the agent build, and an empty value
  now counts as unset.
- **The structured planner runs only for a turn's initial plan** (fix-5vtw);
  replans after a parameter-extraction error or an `ask_user` reply use the
  plain-text planner, and the check keeps checking the initial plan. **Its
  call no longer takes DSPy's hidden JSON-adapter retry** (fix-7eu6), which
  dropped the available-commands prelude: an unparseable reply goes straight
  to the text planner, at most 3 calls instead of 6.
  `CommandsSystemPreludeAdapter` gains `use_json_adapter_fallback` (default
  `True`, unchanged for the agent loop and intent clarification). (Since
  fix-6jzi that retry, where it stays on, keeps the command list; see Fixed.)
- **Large finish checks are bounded** (fix-az9q): a plan naming more than 12
  subjects is checked step by step only (reason `subjects capped`, no subject
  name sent); the ledger is kept under 96 KiB by dropping rows' `head`, then
  `refers_to`, oldest first; an over-long ledger is halved at most once, and a
  half still too large is an `error`. Checked events gain `subjects_capped`,
  `ledger_bytes` and `ledger_rows_trimmed`. No recorded turn is trimmed (max
  52 KB).
- **The plan survives a cross-process `ask_user` resume** (fix-ju1v): with a
  checker attached, `turn_plan` and `turn_plan_status` are session state and
  the resumed agent's step record is seeded from the suspension. A "no plan"
  event carries `no_plan_cause` (`planner_empty`, `plan_unreadable`,
  `lost_on_resume`, `not_planned`); a resume from a suspension the
  context-window fallback had cut is skipped as `ledger incomplete`.
- **The finish-check note says to skip what the user declined, and not to
  repeat or bypass** (fix-hfbr, partial; fix-03lt). Its tail now reads "Run any
  that are still needed, or say in your answer why not. Do not repeat a change
  this turn's record shows was already made, and do not make a change the user
  has not confirmed: ask them instead. Skip any step the user has since
  declined or changed. You have N steps left." Every `finish_check` event
  carries `user_replies`, the turn's `ask_user` step count. The checked plan is
  still the initial one. fix-hfbr stays open on a framework-owned approval
  gate (fix-47a9). (Since 2026-09-28 that gate is deferred and fix-hfbr no
  longer waits on it; see the next entry.)
- **The finish check only asks about provably read-only steps** (fix-hfbr).
  A step is checked only if every command it and its parts name is declared
  `read_only` in the workflow's runtime manifest (the metadata registered at
  startup, else `workflow_runtime.json`; a qualified name counts by its last
  `/` segment, at the most severe kind any key of that name declares). A step
  naming a `write` command or one the manifest does not declare is never
  asked about, its text is not sent, and it is never named in the note, so a
  false flag cannot drive a repeated or unconfirmed change. A step naming no
  command is still checked. Undeclared is treated as not read-only here (the
  safe direction for a note that asks the agent to act); the owner's
  "treat undeclared as read-only" decision was for fix-47a9's approval gate.
  **Behaviour change:** a workflow without a manifest, or with one that
  cannot be read, now has only its command-less steps checked. Every
  `finish_check` event recorded after the plan is read carries
  `unchecked_for_effect`, the steps skipped this way; when that is every
  step the check would hold the agent to, the reason is the new
  `no read-only steps` and nothing is sent. Replayed on the 82 recorded
  downstream attempts with that workflow's manifest (cached answers): write-step flags 7 to 0,
  precision 0.80, recall 0.88 to 0.87. No span contract changes; with
  `FW_FINISH_CHECK` unset nothing changes.
- **The listing parser is fail-closed** (fix-zoup, fix-36gv, fix-k56c; see
  Fixed): it returns a listing only when it is sure the rows are the whole
  listing. Its recognition is widened conservatively (fix-1593): Unicode
  column names, the compact `|-|-|` markdown separator and pipe-less
  `a | b` / `--- | ---` tables. On 4,000 random listings it refuses 702 the old
  parser accepted and accepts none the old parser refused; with the router
  off only `listing_parsed` telemetry changes.
- **Served rows fit the answer bound exactly, or go to the model** (fix-3zxk):
  see Fixed. The closing reads "rows 1-N of the M rows listed in O5".
- **Short-observation hints compare against the observation itself**
  (fix-lzdz): the header names the observation's own subject, and another
  handle is offered only when it mentions the question's words strictly more;
  all hint wordings changed. The event gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`.
  Relatedness reads Unicode letters and digits (fix-1593).
- **Broad scopes are not enumerated** (fix-y570): under the process-default
  scope and the between-turns fallback, `list_summaries` returns nothing and the
  hint says other observations are not listed in this scope; it also filters
  by `channel_id` now. The archive's `get()`/`list()` cross-channel read under
  those scopes is fix-tyzj, still open. (Fixed since by fix-tyzj; see Fixed.)
- **Backend lines shaped like framework markers are quoted** (fix-znxq,
  verbatim-quoting part): in a short observation and in served rows (preamble,
  column line, rows), a line containing `[search_memory`, `Observation O<n> (`
  or an offload-label prefix is printed after a visible `> `. The finish-check
  part of fix-znxq is folded into fix-hfbr. (Since 2026-09-28 an adversarial
  test covers it: with a Jev stand-in answering "not executed, and applies" to
  every question and injected tool output claiming a write step did not run,
  the note names only the read-only and command-less steps, never the write
  steps, their text or the injected text;
  `test_a_ledger_that_claims_writes_went_unexecuted_never_gets_a_write_step_named`.)
- **The search evidence bound has a ceiling and an override** (fix-deus): the
  derived quarter of the search window is capped at 131,072 bytes
  (`SEARCH_OBSERVATION_CEILING_BYTES`), `FW_SEARCH_OBSERVATION_MAX_BYTES`
  overrides it (and may exceed the cap), and the search window is the smaller
  of `FW_MODEL_CONTEXT_TOKENS` and the search model's metadata instead of the
  setting outright. `budget_provenance()["overrides"]` reports the override.
- **The finish check's identifier resolution is generic, and its calibration
  is stated** (fix-ft18): `refers_to` also resolves from aligned, markdown and
  tabbed listings (first 64 KiB of each response, block by block), parameter
  values and clause tokens; recorded downstream ledgers are unchanged. Every event
  records `calibration` (the calibration tag), and a non-default
  `FW_FINISH_CHECK_MODEL` warns once per process. The structured planner's
  `kind` description no longer uses workflow-specific examples.
- **Decision-model features go through a provider interface** (fix-5uva,
  stages 1-2): `fastworkflow.observation_offloading.decision` (`YesNo`,
  `OneOf`, `DecisionProvider`, `ProviderFailure`, `StateTooLarge`,
  `OutOfTime`, `register_decision_provider`, `unregister_decision_provider`),
  with Jev as `jev_client.JevProvider`. `FW_FINISH_CHECK` and
  `FW_SEARCH_ROUTER` accept a provider name registered from code, never a
  module path; selecting one warns once that the thresholds were calibrated
  with Jev. `finish_check` events and router records gain `provider`. What Jev
  receives is the same request bodies as before, key order included, pinned by
  a golden test.
  A LiteLLM provider (stage 3) is deferred.
- **Integration tests reach real HTTP** (fix-dln2, partial): a threaded
  loopback stand-in for TypeSafe (`tests/jev_stub.py`, fixture `jev_stub`)
  drives the real SDK; the structured planner is tested with DSPy's `DummyLM`
  and the plan's cross-process resume through real `process_turn` turns.
- **The restore promise says "normally"** (fix-94m9). Default path,
  agent-visible, and **not re-measured**: the offload-label and served-rows
  mark (`labels.LABEL_RESTORE_MARK`) is now "Normally restored for the final
  answer." (was "Restored in full for the final answer."; +1 byte per label);
  the `search_memory` tool description and the `WorkflowAgentSignature`
  docstring (agent and extractor instructions) say every observation "is
  normally restored in full" and add "If the answer's evidence limit is
  reached, the oldest observations are not restored and the answer names
  them."; and the rehydration note's prefix (`NOT_REHYDRATED_PREFIX`) is now
  "Not rehydrated for the answer (evidence exists under these observations,
  but the answer's evidence limit was reached; say in the final answer that
  the rows of these observations are not included in it): ", so the extractor
  states the omission. The note appears only when the evidence budget binds.
  Labels in every earlier wording still parse.
- **`search_memory`'s narrowing inputs come from the command dispatched**
  (fix-v9jy): `_execute_workflow_query` files the qualified command each
  execute alias ran (`remember_dispatched_command`, on
  `agent.dispatched_commands`, current turn's scope only; CME commands such as
  `abort` and `go_up` are not filed), and `describe_command_inputs` uses it
  when its bare name matches, so two contexts' same-named commands are told
  apart. Otherwise, and after a cross-process resume, it falls back to the
  previous rule. No prompt, schema or event field changes.
- **`search.py` is split** (fix-vpe3): the relatedness score, related-handle
  suggestions and short-observation answer move to
  `observation_offloading/related.py`; `search.py` re-exports every name. The
  search event is built once from a shared base, so its keys come out in
  another order with the same values (stored with `sort_keys`). No behaviour
  change.
- **The decision-model wording is pinned to its measurement** (fix-8ee5):
  `tests/fixtures/decision_measurements.json` records the offline measurement
  behind the finish check's and router's published figures (calibration
  tag, model, thresholds; finish check precision 0.767
  [0.722, 0.810], recall 0.888 [0.832, 0.936] on 910 pairs over 82 attempts;
  router all-rows precision 1.0 [0.970, 1.0] on 126, recall 0.962
  [0.914, 0.984] on 131, at threshold 0.5; each with n and 95% interval) and
  the sha256 of every evidence file. `tests/test_decision_measurements.py`
  re-renders every finish-check and router question template with fixed
  arguments and fails when a digest changes, even if the golden request bodies
  were regenerated to match. The evidence itself is not in the repo; see
  Migration.

### Fixed

- A malformed model reply that arrives after a tool has already run now fails
  the turn instead of silently re-running the whole trajectory. The re-run could
  repeat the side effects of the commands already executed and leave the
  archived evidence out of step with the answer.
- A known command name followed by a newline or a tab is now recognised by the
  known-name guard and by the owning context's exact match, not only when the
  name is followed by a space.
- A command whose name has capital letters is matched by its own context's
  exact match whatever case it is typed in, and is never refused by the
  known-name guard as belonging to another context. The guard compared a
  lowercased name against the context's command names as spelled, so it named
  the current context as the foreign owner.
- A failure while releasing the previous turn's process-local evidence no
  longer aborts the turn that is starting.
- An observability database that no trace sink ever opens -- a context built
  with `tracing.NoOpTraceSink()` -- is now pruned too: the offloading archive
  prunes each database once per process when it first opens it. Its offload
  tables used to grow without bound.
- The finish-time roster nudge no longer fires in a workflow whose contexts
  declare no instance identity, where every named item looked untouched, and
  it no longer tells the agent that nothing about those items was retrieved;
  it says only that they were never the subject of a command. (The roster
  nudge was then removed in favour of the finish-time execution check.)
- The foreign-context hint no longer ends with "Then run it there" when no
  entry command is declared; it says to move into the owning context first.
- Replanning and answer-time rehydration apply the agent's own execute
  numbering to a handle line or offload label, as compaction already did, so a
  command response shaped like one on a step that was never annotated is kept
  as response and never resolved to another observation's evidence. Each such
  line is recorded as a `foreign_line_ignored` event.
- The replan skeleton's policy label is `greedy_trajectory_budget`; it named
  a fixed 28 KB bound it does not use outside the reference window.
- A reply to `you_misunderstood` that matches none of the current context's
  commands now lists what can be done there. It used to raise
  `KeyError: 'what can i do?'`, because the fallback it substitutes was
  registered only for the ambiguity clarification stage.
- The finish check asks about the whole step when a step concerns a subject
  but none of its sub-steps does; such a step was never asked about, so it
  could not be flagged (fix-ba7k).
- Text plans parse a plan written on one numbered line ("1. a. 2. b. 3. c")
  into its steps, and read "optional" and "needs the user" from the whole step
  rather than its first 40 characters, without taking "with optional filters"
  for an optional step (fix-nzmy).
- A structured planner call that returns no steps falls back to the plain-text
  planner instead of leaving the turn unchecked, and a text plan that cannot be
  parsed leaves the check without a plan instead of failing the turn. With the
  check off no text plan is parsed at all (fix-j8yn).
- The async agent loop runs the finish check off the event loop, so its HTTP
  calls no longer block other coroutines (fix-13lt).
- `FinishChecker.note()` never raises and always closes its `fw.finish_check`
  span, also when the check fails after the span opens (fix-re11).
- A step output whose first sentence opens by saying it is empty ("No accounts
  found for Alan.", "Identity not found") still counts as empty, but one that
  goes on to a counted clause or picks from a counted set is a result: "0
  errors, 12 updated" and "None of the 5 are overdue". Subject
  names are found in an output as whole words, so "Al" is not found in "Alan"
  (fix-tu2g).
- Finish-check tests assert structure and key facts; one golden test pins the
  measured note wording (fix-mbcb).
- An over-long finish-check ledger is recognised from the error body's
  `max_tokens_exceeded` code, not from `str(error)`, so a refusal worded for
  humans still halves the ledger instead of failing the check (fix-oivq).
- The TypeSafe SDK's DEBUG logging no longer writes whole, unredacted request
  and response bodies: a filter on the `typesafe_sdk` logger drops those
  records whatever the log level and keeps its INFO status line (fix-qgeu).
- Finish-check subject names and kinds, `refers_to` keys and
  `names_in_output` now pass the same outbound filter as every other value
  sent; a credential used as a subject name or listing id never reaches the
  wire. The note shown to the agent keeps the raw names (fix-hu4f).
- A step the NLU stage stopped before any command ran (parameter extraction,
  an ambiguous or misunderstood command) is recorded at dispatch and shows in
  the ledger as outcome `error`, instead of reading as executed (fix-lnzw).
- A short-observation search no longer blocks for 30 s and raises when the
  related-handle lookup cannot read the archive: `list_summaries` waits at most
  0.5 s, any failure is caught, and the hint says other observations could not
  be listed (fix-kvq0).
- A refused listing parse is final: `parse_table` no longer retries from every
  later line, so a 104 KB listing refused at its end takes ~2 ms instead of
  2.5-4.6 s, and a short observation is not parsed at all (fix-zoup,
  fix-n4z9).
- An aligned listing no longer ends on an ordinary row -- a wrapped label, a
  row with separators collapsed to one space, or empty trailing cells -- while
  claiming to be complete (fix-36gv).
- The first block of a listing is no longer presented as the whole
  observation: a second group or table after it, a `total`/`sum`/`count` last
  row, or text saying there is more (`remaining=`, `has_more=true`, `N of M`,
  `page=2`, `next cursor`, `truncated`, …) refuses the parse (fix-k56c).
- Served rows never exceed the answer bound: `served_rows` picks the largest
  row count whose closing line fits and returns `None` when none fits, and the
  search goes to the model with `listing_skip_reason` `rows_do_not_fit`. A
  large preamble used to give a 5,410 B answer against a 3,000 B bound
  (fix-3zxk).
- **Cold-resume archive location.** An agent built while a context restores a
  suspended turn has no active workflow yet, so it opened the offload archive
  from an empty workflow path: a database named after the working directory's
  basename (unless `FASTWORKFLOW_WORKFLOW_ID` was set). This hit the default
  path. Evidence archived before the suspension was unreachable by
  `search_memory` and answer rehydration, and everything archived after the
  resume went to the stray database for the context's life, where channel
  erasure never reached it. `build_tool_agent` now reads the session's bound
  app workflow first, then the active one; the router's examples and the
  finish check's effect lookup use the same path. Warm turns open the same
  database as before.
- The `ask_user` replan after a cross-process resume plans from the request
  and trajectory again (fix-ksdu): it ran before `resume()` restored them and
  got `{}` for both. `ContinuationReAct.planner_view()` falls back to a copy of
  the suspended stash's `input_args` and `trajectory` (offload labels, no
  `action_N` keys); a same-process replan prompt is byte-identical. Session
  state is unchanged.
- DSPy's JSON-adapter retry after an unparseable chat-format reply keeps the
  available-commands list (fix-6jzi), in the agent loop, intent clarification
  and the plain-text planner; call counts are unchanged. It relies on DSPy's
  **private** `ChatAdapter._make_json_adapter_fallback()` hook (checked on
  DSPy 3.3.0; the pin still allows `^3.0.1`), overridden to return the new
  `CommandsSystemPreludeJSONAdapter`; a guard in
  `tests/test_chat_adapter_commands.py` fails if DSPy renames it. On an
  installed DSPy without the hook, importing `fastworkflow.utils.chat_adapter`
  logs one warning that the JSON retry will drop the available commands list;
  behaviour is otherwise unchanged.
- The offload archive no longer reads across channels (fix-tyzj): `get`,
  `list`, `get_subject`, `capture_record`, `forget_subject` and the
  stored-digest check filter on `channel_id` as well as the turn key, and
  `put_subject` updates only a row of the same channel. A scope pairing another
  channel with this turn's key reads nothing, cannot replace or forget this
  channel's subjects, and still collides on persist. The collision's
  `PersistenceError` now reads "runtime handle alias is already stored for this
  turn (different text or another channel)"; it said "collides with different
  text", which was wrong for identical text from another channel.
- A labelled last row -- first cell ending with `:`, or one word then `:` and a
  space (`Note:  2 items`) -- is a footer, not a row (fix-4riz): the strict
  parse refuses the listing, as for a `total`/`sum`/`count` row, and the lax
  read drops it. Over 3,832 recorded downstream-workflow texts, no accept/refuse decision
  changed in either mode.
- Test isolation (fix-4fw3): the SDK timeout contract asserts the private
  `_config.timeout` the pinned 0.7.2 SDK actually has, and tests that register a
  litellm model restore litellm's model tables and caches afterwards
  (`restore_litellm_model_cost` fixture).

### Migration

- **Structured planning is disabled** (2026-09-28). Its code is commented
  out, not deleted, and there is no setting or flag to turn it back on. With
  `FW_FINISH_CHECK` on, the finish check now gets text plans only: those have
  no subjects, so the per-subject questions are never asked of a live plan,
  only per-step ones. Planning with the check on is no slower than without it
  (a single plain-text planner call). A reader of `fw.planner.plan` spans sees
  `plan_source` `text` / `none` and `subjects` `[]`; records written before
  this change may still hold `structured` or `text_fallback`.
- `LLM_OBSERVATION_SEARCH` is the recommended setting for the model that answers
  `search_memory`. When it is unset, search runs on `LLM_AGENT`, which also
  fixes the budget the evidence is cut to, since that budget is sized from the
  search model's own context window.
- `LITELLM_API_KEY_OBSERVATION_SEARCH` is the recommended credential for that
  role. When it is unset, search uses the credential configured for `LLM_AGENT`.
- Offloading evidence now lives in `observability.sqlite3`. An evidence file
  left by an earlier build beside it (`observability.sqlite3.offload-handles.sqlite3`,
  its write-ahead-log files and any `.preserve` marker) is deleted the first time
  the store opens; its contents are not migrated.
- Experiment evidence is no longer preserved: Clear conversations, and
  forgetting an experiment run's channel, erase its offloading evidence like any
  other turn's.
- Recording can no longer be turned off. A program that embeds the library gets
  the same owner-only, pruned record as the entry points; to keep it elsewhere,
  set `FASTWORKFLOW_STATE_ROOT`. A `WorkflowExecutionContext` built without a
  sink now records into its workflow's database on its own; pass
  `trace_sink=tracing.NoOpTraceSink()` where a context must record no spans or
  turn records. That does not stop observation offloading, which still writes
  its evidence, subjects and events to the same database.
- The offloading runtime's events are not written to a separate file; read them
  with `ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.
- To correct the `search_memory` input bound, set `FW_MODEL_CONTEXT_TOKENS` or
  point `LLM_OBSERVATION_SEARCH` at the intended model. (Since fix-deus, set
  `FW_SEARCH_OBSERVATION_MAX_BYTES` to move the bound itself; a
  `FW_MODEL_CONTEXT_TOKENS` larger than the search model's own window no longer
  raises it.)
- The roster nudge is gone: without `FW_FINISH_CHECK=jev` and `JEV_API_KEY` a
  `finish` is never interrupted. `FW_EVAL_FINISH_REMINDERS=0` still turns the
  note off when the check is on (and, since fix-rm98, the structured planner
  too). `fastworkflow.answer_coverage` no longer
  exists, and a step span marks a note with `finish_check_note` where it used to
  say `roster_nudge` (`fw.agent.step` v3; span contract 7).
- The aggregate span contract is 8 (was 7): `fw.planner.plan` and
  `fw.planner.replan` are v2 and gain `plan_source` and `subjects`; with the
  finish check on, `plan` is the structured plan rendered as a numbered list
  with "(optional)" / "(needs the user)" flags. Spans recorded before read with
  both keys absent. A reader that pins the aggregate number must accept 8.
- `finish_check` events recorded before this change carry step text and
  subject names in `flagged_steps`; new ones carry indices. Events are now also
  recorded for skipped finishes, so count checks by `reason`, not by event.
- Search events gain the path fields above; `listing_shape` is no longer only
  on `rows_served` events.
- Install the Jev features with `fastworkflow[jev]`. `fastworkflow[router]`
  still works as a deprecated alias.
- `JEV_API_KEY` keeps its name; it does not follow `LITELLM_API_KEY_<ROLE>`
  because it is the TypeSafe SDK's own key, not a litellm role's.
- Turning on `FW_SEARCH_ROUTER` or `FW_FINISH_CHECK` sends turn content to
  TypeSafe with only credential patterns scrubbed by default (since fix-7tz5,
  neither turns on at all under a withholding capture profile or with
  redaction off), and the tokens
  those calls use are not in `fw.llm.call` cost totals. The finish check's
  8-second budget is best-effort: it is checked before each call, so one call
  can run up to its 4-second timeout past it. (No longer true since fix-i94q:
  every bound is a hard wall-clock cutoff; see the migration notes below.)
- **Session state gains two additive keys**, `turn_plan` (the plan as JSON, or
  `null`) and `turn_plan_status` (`planned`, `planner_empty`,
  `plan_unreadable`, `lost_on_resume`, `not_planned`); there is no
  `SCHEMA_VERSION` bump. A state written before this change resumes with the
  plan lost (`no_plan_cause` `lost_on_resume`) rather than failing. With the
  finish check off every blob carries `turn_plan: null` and
  `turn_plan_status: "not_planned"`. A react suspension may carry an optional
  `dispatch_outcomes` key.
- **`typesafe-sdk` is pinned to exactly 0.7.2.** `pip install
  'fastworkflow[jev]'` no longer resolves a later 0.7.x; an environment that
  already has another version installed is downgraded or conflicts. A bump
  must rerun `tests/test_jev_client.py::test_the_sdk_surface_fastworkflow_uses`
  and the golden-body tests in `tests/test_decision_provider.py`.
- **`TYPESAFE_BASE_URL` is ignored.** A deployment that pointed the SDK at
  another endpoint with it must set `FW_JEV_BASE_URL` instead (https, or http
  only to loopback); until then the features call the default endpoint and log
  one warning naming the variable.
- **Event shapes gain fields; none is removed or renamed.** `finish_check`
  events: `provider`, `calibration` and `user_replies` on every event;
  `no_plan_cause` on "no plan"; `vendor_ms`, `subjects_capped`, `ledger_bytes`
  and `ledger_rows_trimmed` on a checked finish; `error_status`,
  `error_request_id`, `error_code` and `error_stage` on "error"; new reasons
  `policy_withheld`, `subjects capped` and `ledger incomplete` (a capped check
  reports `subjects capped` where it used to report `every step executed` or
  `unexecuted steps`; read `fired` for the outcome). `search_memory` events:
  the `router` record gains `vendor_ms`, `provider` and the four `error_*`
  fields, with new error values `router_budget`, `policy_withheld`,
  `CallTimedOut`, `OutOfTime` and `VendorBusy`; `listing_skip_reason` gains
  `router_budget` and `rows_do_not_fit`; `short_verbatim` gains `own_score`,
  `related_lookup_failed`, `related_scope_refused` and `related_lookup_error`,
  and now always has `listing_parsed: false`. Span attributes and the span
  contract are unchanged.
- **Later on 2026-09-28**: `finish_check` events and router records gain
  `vendor_calls_in_flight` (vendor worker slots held, only when above 0);
  `listing_skip_reason` gains `policy_withheld` (until then counted as
  `router_error`), and a withheld route no longer uses one of the turn's 3
  routing calls. Every vendor call now passes the SDK a timeout of its cutoff
  plus 0.5 s (the client's own if shorter), and an abandoned call gives its
  worker slot back at most 0.5 s after its cutoff even if a trickled body keeps
  it running (`jev_client.calls_orphaned`); while 8 such requests still run
  (`jev_client.ORPHANED_CALLS_MAX`) every new call fails at once as
  `VendorBusy` and sends nothing, so a trickling endpoint cannot pile up
  threads and sockets, and `finish_check` events and router records carry
  `vendor_calls_orphaned` (only when above 0). The finish check's ledger reads
  labels from a page or one group of a longer listing
  (`parse_table(text, require_complete=False)`; rows are still served only from
  a complete one), its archive reads wait at most 0.5 s for a locked database
  and a failed read now ends the check as `error` at stage `ledger` (until
  then the inline text was used). Its context-clause reads share the process
  caches every other clause read uses, the "no subject" answer included, so a
  second check on a turn reads no alias from the archive again. With `FW_EVAL_FINISH_REMINDERS=0` a check
  that is attached no longer seeds a resumed trajectory, records dispatch
  outcomes or restores the suspended plan. The async entry (`aforward`) now
  resets the vendor budget and dispatch outcomes per turn, as `forward` does.
  `calibration` stays on events only: it is not a `fw.finish_check` span
  attribute, and the span contract is unchanged.
- **`FW_EVAL_FINISH_REMINDERS` is read from the workflow's `fastworkflow.env`
  first**, then the process, and `0` now also turns off the structured planner:
  a control arm run with it gets the plain-text planner and no plan markers. A
  value in the env file other than `0` now fails the agent build, as one in the
  process did; an empty value is unset.
- **The search evidence bound halves on the example model.** The example
  configuration's `mistral/mistral-small-latest` reports a 262,144-token
  window, which gave a 262,144-byte bound; the 131,072-byte ceiling now applies,
  so observations of 128-256 KB are searched as a disclosed prefix. Set
  `FW_SEARCH_OBSERVATION_MAX_BYTES=262144` to keep the old bound. No recorded
  downstream observation is over 5.4 KB.
- **`listing.served_rows` returns `Optional[tuple]`**: `None` when not one row
  fits the bound. A caller outside the framework must handle it.
  `parse_table` keeps its return shape and returns `None` more often.
- **Vendor-call limits are module constants**, not settings
  (`jev_client.VENDOR_WORKERS`, `TURN_VENDOR_SECONDS`,
  `ROUTER_CALLS_PER_TURN`; `finish_check.SUBJECTS_MAX`, `LEDGER_MAX_BYTES`,
  `LISTING_PARSE_MAX_BYTES`). A turn resumed in another process starts a fresh
  vendor budget.
- **Agent-visible wording changed on the default path without re-measurement**
  (fix-94m9): the offload-label mark, the `search_memory` description, the
  `WorkflowAgentSignature` docstring and the rehydration note prefix (texts
  under Changed). A deployment that compared agent runs against a baseline
  taken before this change is comparing different prompts. A reader that
  matched the exact old mark "Restored in full for the final answer." must
  accept "Normally restored for the final answer."; `labels.LABEL_RE` parses
  both.
- **fix-6jzi depends on a private DSPy hook**
  (`ChatAdapter._make_json_adapter_fallback`). A DSPy upgrade must rerun
  `tests/test_chat_adapter_commands.py`; if the hook is gone the retry silently
  loses the command list again.
- **Changing a finish-check or router question now fails
  `tests/test_decision_measurements.py`** (fix-8ee5). Update the digests in
  `tests/fixtures/decision_measurements.json` only together with a new
  measurement recorded there. **The measurement evidence lives only in local
  directories outside this repository** (with the manifest-gate replay
  scripts and the manifest generator) and must be archived somewhere durable;
  the manifest holds only their sha256 and paths.
- `fastworkflow.observation_offloading.related` is a new module (fix-vpe3);
  every moved name is still importable from `observation_offloading.search`.

### Known limits

Offloading and search ship with documented limits rather than silent ones: what
a turn resumed in another process reads, which part of the evidence is stored
unredacted, when pruning runs, the worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.
