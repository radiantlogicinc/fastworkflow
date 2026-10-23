# Skill phases: code runs the mechanical steps

Status: built (fix-6llp.1 to .4, .6, .7); owner decisions recorded (see Decisions).
Where the build differs from this design, see "As built"; the user-facing description
is "Steps run by code" in `skill_phases.md`.

## Why

A phase's steps are built in code by its `steps_fn`. Most of them are already exact
commands with every value filled in: `open_library`, `open_book_by_uid book_uid=b1`,
`list_loans (full list)`. Today an LLM executor still runs each one. It can drop an
argument, send a placeholder, skip a step, run the wrong command, loop on a record it
is already on, or wander off the plan.

In a benchmark workflow with 15 phased skills (stage 11: 3 runs, 407 phase runs, 1190
planned phase steps):

- 999 of 1190 planned steps (84%) were mechanical by a hand-checked rule.
- 259 of 407 phases had only mechanical steps. In them the executor took 1502 steps
  where 833 were planned.
- 12 of the 20 failed phases first went wrong on a mechanical step. 11 of those were in
  phases with only mechanical steps.
- 131 steps were literal name lookups (`find_book query=Dune`). 111 of them returned
  exactly one row. In 5 failed jobs the executor mangled, dropped or over-ran such a
  lookup.

So: let code run the steps that need no judgment, through the same path the agent's
tool uses, and give the agent only what is left.

## Summary

1. A step is **mechanical** when the existing parsers read it as one command with
   every required parameter given a concrete value. No annotation and no new format.
2. Code runs the phase's **leading** mechanical steps, one by one, through the agent's
   own `execute_workflow_query` tool, inside the executor run. Each is a normal
   trajectory step, so outputs, the archive, facts, spans, the finish check and the
   repeat guard all see a normal run.
3. Code stops at the first step that is not mechanical, or at the first step that fails.
   The agent then gets the run, with the code-run steps already in its trajectory.
4. A phase whose steps were all run by code needs no agent loop: no LLM call at all,
   except for a phase that writes an answer (the last phase, the extra-steps phase),
   which gets the one answer call every executor run ends with.
5. When a step's command is not available where the workflow is, code goes there first
   only when the existing navigation planner finds a path that needs no values
   (`go_up`, `reset_context`, a hub opener). Otherwise the agent takes over.
6. Paging stays literal: code sends only the paging value a step writes itself. A step
   that writes it as a placeholder (`start=<n>`) goes to the agent. A step that asks for
   more pages only in a note runs page 1; when more pages remain, code hands over.
   (Superseded by fix-ajir.36: a full-list note is paged by code, up to
   `CODE_MAX_PAGES`; see "Paging is literal" in `docs/skill_phases.md`.)
7. A failed code step hands over to the agent with the failure in its trajectory. Code
   never fails a phase by itself.
8. One temporary switch, on by default, for the stage-12 A/B run; removed after stage 12
   if the gain is confirmed. Two new span attributes.
9. Later, only if stage-12 data justifies it: binding "the uid of the match" in code, and
   the same treatment for flat skill calls.

## 1. Which step is mechanical

The rule runs at the start of each code step, on the phase's numbered plan (the text
`steps_fn` built), with the workflow's command schemas. It uses only what exists:

- `finish_check.parse_plan` classifies the step and finds its command;
- `finish_check.step_arguments` gives its concrete values (the chain right after the
  command, cut at `(` or `;`);
- `ParameterExtraction._extract_parameters_from_key_values` binds the same chain to the
  command's parameters class, in code, as it does for the agent today;
- the parameters class says which fields are required.

A step is mechanical when all of these hold:

1. `parse_plan` gives `KIND_COMMAND`. So conditional steps ("only when", "skip when",
   "if needed"), negated steps ("do not run X") and answer steps are never run by code.
   One addition: a core navigation command (`go_up`, `reset_context`) with nothing after
   it also counts (`parse_plan` marks it `KIND_SKIPPED`, since the finish check does
   not hold the agent to it).
2. The command is the step's first word.
3. The **chain** is the text after the command, cut at the first `(` or `;` that is not
   inside a value quoted right after its `=` (`query="Acme (EU)"`, `finish_check.step_chain`). Text in
   parentheses after the chain is a note for the reader. It is never sent. A builder that
   writes the value (`skill.pick`, `pick_step`, `workflow_agent._key_value_text`) quotes it
   when it contains `(` or `;` (`finish_check.quote_step_value`).
4. Either the chain is empty and every field of the parameters class has a default
   (a bare command; a command with no parameters class also counts), or:
   - the key=value extractor binds the chain (it already declines placeholders `<...>`,
     `{...}`, `...`, unknown names, example values and values that do not validate);
   - every value it bound equals the step's own value (`plain_value` of the bound value
     equals `step_arguments`). This catches prose swallowed into the last value, such
     as `scope_uid=x1, no filter`;
   - every required field of the class is given.
5. Code runs the chain, not the step text: `command k1=v1 k2=v2`.

What this does with the open shapes found by hand in the benchmark (offline replay of
the 1190 stage-11 steps with the real parsers and command schemas):

| shape (hand class) | example (neutral) | rule | count |
|---|---|---|---|
| no-arg read / hub opener / go_up | `list_loans (full list)`, `open_library`, `go_up` | mechanical | 723 |
| concrete args | `open_book_by_uid book_uid=b1 (this opens it)` | mechanical | 254 |
| paging step | `next page with start=<n> ... until the whole list is read` | open (no command, placeholder; see 4) | 7 |
| NAME_SEARCH | `find_book query=Dune` | **mechanical**: the query is a literal | 131 |
| NAME_SEARCH + pick note | `find_book query=Dune (take a match whose name starts with Dune)` | mechanical lookup; the note is for the next step | 7 |
| FILTER_FROM_WORDS | `list_shelves filter=rare first editions (for its shelf_code)` | **mechanical**: the filter is a literal | 4 |
| PICK_MATCH | `open_book_by_uid with the uid of the Dune match`, `list_loans shelf_code=<that shelf_code>` | open (prose, placeholder) | 31 |
| CONDITION | `show_items kind=<the family its record names ...>` | open (placeholder) | 1 |
| CORRECTION | `suggest_fixes on Ann (the current record) and present them as options` | open (prose after the command) | 17 |
| prose after a no-arg command | `go_up back to the library`, `list_actions for the full list` | open (prose) | 15 |

Agreement with the hand classes: 1026 of 1190. Every difference is one of three kinds:

- 142 literal lookups (NAME_SEARCH, FILTER_FROM_WORDS) are mechanical by the rule and
  open by hand. That is intended. Their values come from the skill's input, which the
  skill call has already checked against the request (`skill_commands`, the argument
  checks). Code sends exactly that value. Picking among the matches stays open (the next
  step).
- 15 steps are open by the rule and mechanical by hand: a no-arg command followed by
  prose (`go_up back to the library`). The rule cannot tell prose from an argument, so
  it leaves them to the agent. Putting the prose in parentheses makes them mechanical
  (a step-text change in the workflow, not in the framework).
- 7 paging steps are open by the rule and mechanical by hand: they name no command and
  write the paging value as a placeholder. Paging is literal (section 4), so the agent
  pages.

With this rule, 340 of 407 phases (84%) are mechanical end to end and 1062 of 1190
steps (89%) are in a leading mechanical run.

The rule is one function in `skill_phases`: `code_step(step, parameters_class)` returns
the command text to run, or `None`.

## 2. How code runs a step

Code calls the agent's own tool: `agent.tools["execute_workflow_query"]`, the closure
built in `workflow_agent.initialize_workflow_tool_agent`. That is the path every agent
step takes:

- `_explicit_agent_command` resolves the name and refuses one not available here;
- `reenter_from_parent` handles "open another record of the context I am in";
- `_execute_workflow_query` runs `CommandExecutor.invoke_command`, opens
  `fw.agent.tool_call`, appends the turn output and the action record, holds the command
  run for the archive (`_hold_command_run`), applies the repeat guard and handles
  parameter-extraction and ambiguity states;
- `annotated_observation` prints the alias and the context, as the agent sees it.

The archive row and the alias exist only inside a ReAct step
(`current_execute_alias` reads `tool_name_<i>` without `observation_<i>`). So the code
step is written into the executor's trajectory exactly as `_run_loop` writes a step:

```
trajectory[f"thought_{i}"] = "Step 2 of the plan, run by code."
trajectory[f"tool_name_{i}"] = "execute_workflow_query"
trajectory[f"tool_args_{i}"] = {"command": "find_book query=Dune"}
observation = self.tools["execute_workflow_query"](command=...)
trajectory[f"observation_{i}"] = observation
on_step_end(); on_step_complete(i, trajectory)   # repeat guard; archive + compaction
```

This lives in `fastWorkflowReAct` (`utils/react.py`) as `_run_code_steps(trajectory,
idx, source)`, called by `forward()` after it resets the trajectory and `run_state`,
before `_run_loop`. Each code step opens and closes its own `fw.agent.step` span, as a
model step does. `source` is a small object from `skill_phases` that, given the
trajectory so far, returns the next command to run or `None` (stop). It holds the rule
and navigation. ReAct only runs and records.

Points that follow from reuse:

- **Budget.** Code steps do not add to `iteration_counter`. The phase's step budget
  (`FW_SKILL_PHASE_MAX_STEPS`) bounds the agent's own steps only.
- **Retry.** `_call_agent_with_retry` already refuses to retry once any observation
  exists, so a code step never runs twice.
- **ask_user.** Code never asks. A later agent step may suspend; resume uses the stashed
  trajectory, which holds the code steps. Code does not run again on resume.
- **Parameter extraction.** The command text is `name k=v`, so extraction binds it in
  code (`deterministic_form` `key_value`). A bare command currently falls through to the
  LLM extractor (in the benchmark, 713 of 1769 extractions were LLM calls; bare
  commands are one source). Add a `bare` form: no text after the command name and every field has a
  default binds the defaults in code. This also helps the agent's own bare commands.

**Did the step succeed?** After the step is archived, code reads the step's archived
command runs (`_archived_command_runs`, filtered to the step index). The step succeeded
when the observation has no `finish_check.FAILURE_MARKERS`, and every run it made
succeeded. The "You are already in context ..." reply has no marker and no run: it counts
as done, as the finish check and `runs_that_entered_record` already treat it.

## 3. Navigation

Before a code step, code checks the command is available where the workflow is: it
calls `_explicit_agent_command` (no side effect; it raises when the command is not
here). When it raises `CommandNotFoundError`:

1. Code asks the existing planner for the way: `live_context_chain`,
   `collect_descend_edges` and `plan_navigation(current, homes, chain, edges)`, the same
   calls `unavailable_command_message` makes for the agent's refusal text.
2. Code goes there itself only when there is one shortest path and none of its
   `NavStep`s `needs` a value or has `required_to_enter`. That covers `go_up`,
   `reset_context` and hub openers with no parameters. Each move is its own code step,
   visible in the trajectory ("run by code to reach <Context>").
3. Otherwise (no path, a path that needs a uid, two equal homes) code runs the step
   anyway. The tool returns the usual refusal, which names the way. Code stops there and
   the agent continues from that refusal, as it does today.

`AlreadyInContextError` needs nothing new: `reenter_from_parent` goes up and re-opens
another record of the same context; the same record gets the "already in context" reply,
which counts as done (section 2). This removes the loop seen in one failed phase, where
the executor sent the same opener 14 times.

No navigation map is added. Commands, contexts and descend edges are read from the
workflow as the agent's refusal text already reads them.

Of the 11 failed all-mechanical benchmark phases, 3 started in the wrong context. In
two, the phase's first step is a hub command or a record opener whose home is reached
with no value (`reset_context`, then a hub opener): code gets there. In the third, the
step's command lives in a record that only a uid opens: it goes to the agent, as before.

Navigation depends on where the earlier phases left the workflow. When the earlier phases
were also run by code, a phase starts on the record they opened, and a no-value path can
exist where one phase alone (starting wherever the agent had wandered) found none. The
offline simulation found this: replayed one phase at a time, some failed phases stayed
with the agent; replayed as a chain of code-run phases, they reached the answer phase.

## 4. Hand-off to the agent

The runner builds the `source` for the phase and passes it to the executor's
`forward()`. Code runs the leading mechanical steps. Then:

- **An open step is next.** The ReAct loop starts at the next index, with the code steps
  in its trajectory. The phase framing (`phase_query`) gets one sentence: "Steps marked
  'run by code' in the trajectory are done; do the remaining steps." The agent sees the
  code-run outputs the way it sees its own earlier steps. The finish check reads the
  phase's plan against everything archived from the phase's first step, so code-run
  steps count. Code does not resume after an open step: later steps may depend on what
  the agent picked.
- **A code step failed** (a refusal, a uid that matches nothing, an exception, a
  blocked repeat, a parameter-extraction error). Code stops. The agent gets the phase
  with the failure visible as the last observation in its trajectory, and decides as it
  does today (search again, go elsewhere, or finish). Code never fails the phase by
  itself, not even for a failed record opener: the fill decides, as for any run. Reason:
  the agent can often recover (a moved record, a stale uid), and failing in code would
  make code stricter than the agent it replaces.
- **Every step ran by code.** No agent loop.
  - A phase whose answer is not shown (every phase but the last; the extra-steps phase is
    shown) returns at once with an empty answer. No LLM call. Its output is filled from
    the archived runs as for any phase (`fill_output`).
  - The last phase (and the extra-steps phase) needs its answer. `forward()` skips the
    loop and calls `_extract_prediction` on the trajectory, the one answer call every
    executor run ends with today. One LLM call instead of a ReAct loop plus that call.
  - Steps that are not commands and do not stop code: answer steps (`say ...`,
    `KIND_SKIPPED` with no command) and negated steps (`do not run X`). In the last
    phase the answer call reads them from its input. In other phases they have no
    effect, since those answers are never shown.
- **Fan-out items.** Each item is its own phase run. Its steps carry the item's value,
  so they are usually mechanical and the item runs with no LLM call. The item record
  check (`item_record_issues`) still runs on the item's span.
- **Paging is literal.** Code runs only the paging value (`start=`, `offset=`, `skip=`;
  `facts.PAGING_PARAMETERS`) that the step itself writes: `list_loans start=20` runs
  that page, a step with no paging value runs page 1. Code never computes a next page.
  A step that writes the paging value as a placeholder (`start=<n>`), in its chain or
  as a step of its own that names no command (`next page with start=<n> ...`), is open:
  code stops there and the agent pages, as today. A paging request inside a note
  (`list_loans (full list: next page with start=<n> ... until complete)`) is a note like
  any other: never sent, so code runs page 1. Owner decision: when the step's command
  takes a paging parameter (`facts.PAGING_PARAMETERS`), the step writes no paging value
  itself, and the observation shows more pages remain, code stops after that step and
  hands the rest of the phase to the agent, with the page-1 run in its trajectory. When
  page 1 is the whole list, code goes on. No note parsing and no step-text edits. `facts`
  already joins pages the agent reads (`_scalar_of_pages`). Reason: reading pages until
  a stop condition would be behaviour no step wrote.
- **Planner steps.** `steps_before` / `steps_after` are written by the planner LLM, not
  by `steps_fn`, and their values were not checked. Code never runs them: the
  `steps_before` run first in a phase of their own (`extra_steps_before`, run by the
  agent), so the skill's first phase keeps its code run (fix-6llp.12; until then they
  were put before the first phase's steps, and that phase ran no code at all).

## 5. Configuration

Always on for phased skills. There was one temporary switch, `FW_SKILL_CODE_STEPS`
(default on; `0` turned it off), kept for the stage-12 A/B run. It was removed after
stages 12 and 13 confirmed the gain (fix-6llp.10), together with
`FW_SKILL_ANSWER_BY_CODE` (code-written answers, fix-hia1.30, are always on too).

Skills without `plan()` (flat skill calls) run as one phase and get the same treatment
(As built 11). Sub-tasks that are not skill calls are unchanged.

## 6. Observability

Existing spans and rows carry everything; two attributes make code steps easy to find.

- `fw.agent.step` (v7): `by_code: true` on a step code ran. Its `thought` says "run by
  code"; `tool_name`, `tool_args`, `observation` are as for any step. Its child
  `fw.agent.tool_call` is unchanged.
- `fw.skill.phase` (v4): `steps_by_code` (how many trajectory steps code ran, navigation
  included) and `code_stop` (why code stopped: `done`, `open step N`,
  `step N failed`, `no path for step N`, `off`). `steps_used` keeps counting all
  steps; model steps = `steps_used - steps_by_code`.
- `fw.agent.execute`: unchanged. A phase run with no LLM call still has it, with no LM
  child spans.
- `fw.nlu.param_extraction`: `deterministic_form` gains the value `bare` (no new
  attribute).
- Store: code steps are ordinary `offload_evidence` rows and `offload_command_runs`
  rows. No schema change.

Span contract: `SPAN_CONTRACT_VERSION` 20 to 21, `fw.agent.step` v6 to v7,
`fw.skill.phase` v3 to v4, with the usual comment in `tracing.py`.

## 7. What disappears or simplifies

- In a phase run entirely by code: no finish-check note, no repeat guard trips, no item
  record-check note, no budget exhaustion, no "already in context" loops, no wandering.
  The planned steps are the steps run (plus navigation).
- The per-phase budget is spent only on open steps. In the benchmark, the 340 phases
  that are mechanical end to end took 1997 executor steps (each an LLM call) for 975
  planned steps. They would take none, plus one answer call for answering phases.
- The fill's repair rules for agent deviations (`planned_record` re-binding,
  `runs_finding_planned_values`, `runs_standing_in_for_openers`) are still needed for
  agent-run steps. Keep them. Stage 12 measures how often they still fire (span
  attributes and fill logs); remove what never fires in a later bead.
- fix-hia1's deferred idea "a phase opens its record by code before the executor
  starts" is covered by this design.

## 8. Risks and tests

What could break:

- **Step text that is not exactly a command.** The rule only runs a chain that both
  parsers agree on and that fills every required field. Anything else goes to the
  agent, as today. A note in parentheses is never sent. Risk: a note that changes what
  the command should do ("list_loans (only overdue ones)") is dropped by code where the
  agent might have added a filter. Skill authors write such a value as an argument.
  The replay check lists every step with a note so the owner can scan them.
- **Commands with side effects.** The rule cannot tell a read from a write, and does
  not need to (Decisions, 1): code runs any mechanical step, including a mutating one.
  The skill author wrote it with concrete values; the values came from the checked skill
  input or from earlier facts; the agent would send the same command; a retry never
  re-runs it. An author who wants the agent to decide writes the step as conditional
  ("only when ..."), which `parse_plan` already leaves to the agent.
- **Values that did not come from the request.** Step values come from the skill's
  input (checked against the request before any phase runs) or from facts filled in
  code from command outputs. Planner-written steps are never run by code.
- **Planner mistakes in skill arguments.** Code runs exactly the steps the skill builds
  from its arguments. When the planner set an argument wrong (a flag off that the
  request needed, a name under the wrong kind), the agent often covered it with an
  unplanned step (a read the plan did not ask for, a lookup under another kind). Code
  does not, so such a mistake now shows as a missing fact or a weaker answer. The
  offline simulation found a few such jobs. The fix belongs in the skill arguments
  (their checks or the planner's guidance), not in code.
- **Wrong context.** Covered by section 3; any doubt goes to the agent.
- **Paging requests in notes.** A step that asks for every page only in its note runs
  page 1 in code; the note is not sent. When the observation shows more pages remain,
  code stops after that step and the agent reads the rest (section 4), so no list is
  silently cut. The cost is that such a phase is no longer run end to end by code when
  its list is long. The replay check lists every such note (in the benchmark, 190 steps
  carried a paging request in a note).
- **Answers.** The last phase now writes its answer from a trajectory the agent did not
  build. The answer call is the same one used today after `finish`.

Tests:

- **Offline replay.** A dev script outside the wheel classifies stored
  `fw.skill.phase` plans with the rule and the workflow's command schemas, with no run.
  Acceptance on the benchmark: every hand-mechanical step is mechanical by the rule,
  except the prose-after-command steps listed for a text fix; every hand-open step the
  rule calls mechanical is a literal lookup; every pick, condition and correction step
  is open. The prototype already gives these numbers (section 1).
- **Unit tests** with the `todo_list_workflow` test workflow and a phased toy skill, no
  model (a scripted executor fails the test if it is called):
  - the rule: bare, key=value, placeholder, prose after the command, prose in the last
    value, missing required field, conditional, negated, core `go_up`, a literal paging
    value (mechanical), a paging placeholder (open);
  - a phase of only mechanical steps runs with no LLM call; its output is filled; its
    archive rows, `fw.agent.step` spans (`by_code`) and `steps_by_code` are written;
  - a mixed phase: code runs the prefix, the scripted agent sees those steps and does
    the rest; the finish check counts the code steps;
  - a failing code step (a uid that matches nothing) hands over with the failure as
    the last observation;
  - navigation: a step whose command lives in a hub is reached by `go_up` /
    `reset_context` + hub opener in code; a path that needs a uid goes to the agent;
  - "already in context" counts as done;
  - a paging request in a note runs page 1 only; code computes no next page; when more
    pages remain, code hands over after that step;
  - last phase: one answer call, no loop;
  - `FW_SKILL_CODE_STEPS=0` gives today's behaviour (switch since removed, fix-6llp.10);
  - suspend/resume after a code prefix does not re-run code steps.
- **Stage-12 benchmark.** Same models, tasks and settings as stage 11, three runs,
  graded the same way. Compare stopped jobs, checklist lines, executor steps, LLM calls
  and wall time per phase.

## 9. Code size

| file | change | lines (approx.) |
|---|---|---|
| `skill_phases.py` | `code_step` rule; the step source (prefix walk, success check); `phase_query` sentence | 110 |
| `context_navigation.py` | `code_path(workflow, command)`: a no-value path or `None` | 30 |
| `utils/react.py` | `_run_code_steps`; `forward()` hook; answer-only end | 50 |
| `workflow_execution_context.py` | build the source in `_run_skill_phase`; span attributes; switch | 40 |
| `parameter_extraction.py` | `bare` form | 15 |
| `tracing.py` | contract bump | 10 |
| tests | rule, runner, navigation, hand-off, spans | 290 |
| `docs/skill_phases.md` | "Steps run by code" section, tracing table rows | 50 |

About 255 lines of framework code. Removing the switch after stage 12 takes out about
10 of them. No change to skill files, command files or text formats.

## Beads

| bead | what | priority | depends on |
|---|---|---|---|
| fix-6llp.1 | the mechanical-step rule; `bare` extraction form | P1 | |
| fix-6llp.2 | run the leading mechanical steps in the executor run; the switch | P1 | .1 |
| fix-6llp.3 | no-value navigation | P1 | .2 |
| fix-6llp.4 | hand-off to the agent, literal paging, fan-out items | P1 | .2 |
| fix-6llp.6 | span attributes and contract bump | P1 | .2 |
| fix-6llp.7 | offline replay of stored plans | P1 | .1 |
| fix-6llp.8 | docs | P2 | .3, .4, .6, .7, .9 |
| fix-6llp.9 | owner decisions (closed; see Decisions) | P1 | |
| fix-6llp.10 | remove `FW_SKILL_CODE_STEPS` after stage 12 confirms the gain (done: removed after stages 12 and 13) | P2 | .2; the stage-12 benchmark |
| fix-6llp.5 | find-one (deferred until after stage 12) | P3 | .4 |
| fix-6llp.11 | consider code-run steps for flat skill plans | P3 | the stage-12 benchmark |

The stage-12 benchmark is tracked in the benchmark workflow's own tracker. It waits
for .1 to .4 and .6.

## Expected gain

From the stage-11 measurement:

- About 11 stopped jobs fixed (all-mechanical phases that failed on a mechanical step).
  Up to 4 more from literal lookups run exactly (5 jobs failed on a mangled or dropped
  lookup). 2 failed on a wrong plan and 1 on a path that needs a uid: not fixed.
- About 10 checklist lines per run of 177.
- About 2000 fewer executor LLM calls per three-run stage, and faster phases.

An offline simulation over the stored stage-11 runs (no model; the phases' code runs
chained, each failed job scored as its other runs scored) is more cautious: about 8
checklist lines per run from code alone, about 11 to 12 when the agent also completes
the open steps it is handed. The benchmark workflow's own note lists the jobs.

The 7 paging steps that are now open were all in phases that succeeded, so literal
paging does not change the fixed-job count.

## Decisions

The owner's answers to the design's open questions (fix-6llp.9):

1. **Mutating commands.** Code may run a mutating command when its step is concrete.
   There is no per-command flag: a flag would be a registry, and the values already pass
   the skill-argument checks.
2. **The switch.** `FW_SKILL_CODE_STEPS` was temporary: kept for the stage-12 A/B run,
   and removed after stages 12 and 13 confirmed the gain (section 5).
3. **Find-one.** Deferred (see "Later").
4. **A failed code step** hands over to the agent with the failure visible in its
   trajectory. It never fails the phase by itself.
5. **Flat skill calls.** Built after stage 13 (fix-6llp.11): a skill without phases
   runs as one phase (As built 11).
6. **Paging** stays literal (section 4): code runs only the paging value the step itself
   writes. A step on a paged command that writes none runs page 1. Code hands the rest
   of the phase to the agent after page 1 only when the step's note holds one of
   `FULL_LIST_PHRASES` and the observation names a later page; any other note, or none,
   means page 1 is what the step wants (fix-6llp.4, refined after measuring: 3
   hand-overs on the stage-11 plans instead of 66).

## Later, only if stage 12 data justifies it

- **Exactly-one-match lookups ("find-one").** A step like `open_book_by_uid with the
  uid of the Dune match` is open: one required parameter has no value. The idea was to
  bind it in code after a code-run lookup (`find_book query=Dune`) whose Output narrows
  to one row holding the lookup's value (`skill_phases.narrowed_output`), taking the
  value from the row's list field named `P` or `P + "s"` (`book_uid` from `book_uids`).
  Deferred because that name match binds a step parameter by a hidden naming
  convention, which the design avoids. Built after stage 12 (fix-6llp.5) without it:
  `skill_phases.pick_step` sends the opener's one required uid field the value of the
  Output's one uid list, at the one row that holds the value the step names exactly
  (or the only row); see "Picks" in `docs/skill_phases.md`. On the stage-12 plans
  code runs 30 of the 30 pick hand-overs; 29 open the uid the agent opened, and the
  30th opens the exact-name record where the agent opened the `... 1` one.
- **Flat skill calls** (skills without `plan()`): the same code-run treatment for their
  steps. Done (fix-6llp.11): see As built 11.
- **The switch.** `FW_SKILL_CODE_STEPS` was removed after stages 12 and 13 confirmed
  the gain (fix-6llp.10).
- **The fill's repair rules** for agent deviations (section 7): removed in a later bead
  only where stage 12 shows they never fire.

## As built

The build follows the design except for these points (the code is the reference):

1. **No `_run_code_steps`.** Code steps run in `fastWorkflowReAct._run_loop` itself,
   called with `code=<CodeSteps>`: each `(thought, command)` from `code.next()` becomes
   the step's prediction in place of the model's, and is written, traced and hooked
   (`on_step_end`, `on_step_complete`) like a model step, then reported with
   `code.ran(index, observation)`. `forward()` calls it once with `code=` before the
   normal loop, and skips that loop when `code.finished`.
2. **The source is an attribute, not an argument.** The runner
   (`WorkflowExecutionContext._phase_code_steps`) builds `skill_phases.CodeSteps` and sets
   it as the agent's `code_steps` attribute just before the run; `forward()` takes it
   and clears it, so it serves that one run only. A source that cannot be built is
   logged and the agent runs every step.
3. **Own steps only.** `CodeSteps` reads the plan with `plan_code_steps` and keeps the
   first `own_steps` entries (the phase's own steps); `steps_after` are never run by
   code (the first one that runs anything is an open step, so the agent runs it; answer
   and negated ones are passed over); `steps_before` run in a phase of their own, which
   has no own steps (fix-6llp.12). The rule is `code_step` (`_code_step` also
   gives the reason, for the replay); a command several contexts define must read the
   same under each one's parameters class ("differs by context" otherwise).
4. **Paging rule.** The design's "hand over whenever more pages remain" (section 4) was
   narrowed: code hands over after page 1 only when the step's note (the text after the
   chain, from the first `(` or `;`) holds one of `skill_phases.FULL_LIST_PHRASES`
   ("full list", "every page", "all pages", "until complete", "page through"; any case)
   and the observation names a later page. "More pages" (`_more_pages`) is the
   observation naming one of the command's own paging parameters with a positive value
   (`start=10`), as a paged command tells its reader; there is no other more-pages
   signal. Such a step's thought says "(its first page)". Superseded by fix-ajir.36
   (owner-approved): code now reads each later page the observation names
   (`_next_page`), up to `CODE_MAX_PAGES` (20), and no longer stops with `more pages at
   step N`; the agent spent a model step per page and long lists ran out the phase
   budget.
5. **`code_stop` values.** Besides `done`, `open step N`, `step N failed`, `no path for
   step N` and `off`: `more pages at step N` (the paging rule) and `stopped` (the
   executor's `on_step_complete` hook ended the run; no agent loop follows). `no path
   for step N` is a step refused as not available here; a failed navigation move is
   `step N failed`. `off` means no source (the `steps_before` phase, which has no
   own steps, or a source that could not be built; the switch, `FW_SKILL_CODE_STEPS`,
   is gone). Both attributes are absent on an answer-only phase
   and on a resumed run.
6. **Navigation.** `context_navigation.code_path(workflow, command)` uses
   `home_contexts` (the most specific contexts defining the command), now shared with
   the agent's refusal in `workflow_agent._explicit_agent_command`. Code computes the
   moves once per step, before running it.
7. **Success check.** `finish_check.run_failed` on the observation, then every archived
   command run at that step index succeeded; a step with no run counts as done only
   for the "You are already in context" reply.
8. **Framing.** The `_CODE_STEPS_DONE` sentence is added only when code runs the plan's
   first step that runs anything (`CodeSteps.runs_any`), and is one of the framing
   sentences `clean_answer` strips.
9. **`bare` extraction form.** `_extract_parameters_from_bare` is tried after
   `key_value` and `positional`, before the LLM: text empty or only the command name,
   every field defaulted.
10. **Replay.** The dev script is
    `.claude/skills/fastworkflow-diagnostics-and-tooling/scripts/replay_code_steps.py`
    (outside the wheel). On stage 11: 1119 of 1190 steps mechanical, 340 of 407 phases
    (83%) mechanical end to end, 1062 steps in a leading run, 1026 of 1190 agreeing with
    the hand classes.
11. **Flat skill calls run as one phase** (fix-6llp.11). A skill call with no `plan()`,
    or whose `plan()` returns no phases for its input, runs through the phased runner as
    one phase named after the skill, whose steps are `steps(input)`
    (`skill_commands.call_phases`; `phases_for` still returns `None` for such a call).
    So `CodeSteps` runs its leading mechanical steps and picks, as in any phase. Its
    budget is the executor's own `max_iters` (25; `step_budget` on its `fw.skill.phase`
    span says so), not `FW_SKILL_PHASE_MAX_STEPS`, and there is no step-cap hand-off.
    Planner `steps_before` run in their own phase first; `steps_after` are left to the
    agent. A flat call suspended on ask_user and restored is rebuilt with `call_phases`
    and resumes. No new concept for app developers: a skill without phases runs as one
    phase. Offline replay of the stored flat skill calls (1236 calls, 9510 steps):
    2552 steps (26%) in a leading mechanical run, and at the first open step after a
    code-run step 1035 pick-shaped steps (790 prose or uid-placeholder only, 245 with
    other concrete values) that code runs when the lookup has one matching row.
12. **Picks with other values** (fix-6llp.11). A pick step may write other values
    besides the uid when the uid's placeholder is its ONLY placeholder and the step
    leaves no other required field unwritten
    (`list_access resource_type=app resource_uid=<the Dune uid> filter=Ann`): code sends
    the uid and the step's other values as written. Any other placeholder, or a value
    that does not bind as written, leaves the step to the agent.
