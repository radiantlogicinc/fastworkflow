"""Whether a planning command needs the agent, read statically (fix-3rsq.10.8, design 3.4).

Assistant mode has no intent detection: it matches a command by its exact name and nothing else. An
exactly named planning command runs there only when it is MECHANICAL: its phases are code steps and
mechanical fan-outs, with its arguments from ordinary parameter extraction. A planning command NEEDS
THE AGENT, and assistant mode refuses it with the 3.4 message extended by the reasons, when any of
these holds:

- a phase has a PROSE step: a plain line the agent reads and carries out, such as ``say that ...``;
- a phase has a CONDITIONAL step (``when(...)``): the agent decides whether to run it;
- a phase is DECIDED AT RUN TIME: whether it runs, or what it steps, depends on facts an earlier
  phase filled (a test on a fact, or a callable ``when`` or ``until``);
- a parameter carries ``verify``: a classifier model decides it from the request;
- the planning command has no ``__result__``: the agent writes the answer from the last phase's facts.

Code steps (``run``, ``run(..., named=...)``, ``run(..., all_pages=True)``, ``do_not_run``), fan-outs over an input or an earlier
phase's list, input-only conditions and ``__result__`` are mechanical.

Nothing here calls a model or runs a step. The phases are built as the planner sees them (placeholder
facts, every combination of the input the template grid covers), so a reason names the phase and the
step or fact that causes it, and an author can see what to change.
"""
from __future__ import annotations

from fastworkflow import planning as planning_api
from fastworkflow import planning_commands
from fastworkflow.planning_phases import (PhaseError, _template_kwargs, building_template,
                                          tests_noted)

__all__ = [
    "PROSE_HINT",
    "REASON_RESULT",
    "planning_command_needs_agent",
    "refusal_message",
]

#: The author's fix for each rule, appended to its reason so the reason says what to change.
PROSE_HINT = "a code step (run, run with named=, or run with all_pages=True) would make it mechanical"
WHEN_HINT = "a fixed step, or a condition on the input rather than a fact"
RUN_TIME_HINT = "a step that depends on the input alone, not on what an earlier phase found"
VERIFY_HINT = "a parameter the request states outright, without verify"
REASON_RESULT = ("it has no __result__, so the agent writes the answer from the last phase's facts "
                 "(a __result__ that returns the text makes it mechanical)")


def _loaded(workflow_folderpath: str, name: str) -> planning_commands.PlanningCommandCommand:
    command = planning_commands.planning_command(workflow_folderpath, name)
    if command is None:
        raise ValueError(f"{name!r} is not a planning command of {workflow_folderpath}")
    return command


def _shape(seen: list[tuple[str, object]]) -> tuple[tuple[str, object], ...]:
    """The steps' kinds in order: ``(phase, helper)``, a prose step's helper None."""
    return tuple((name, getattr(step, "helper", None)) for name, step in seen)


def _step_lines(command, env: dict):
    """One template-grid combination's steps as the planner sees them.

    Returns ``(steps, run_time)``: *steps* is ``[(phase name, step)]``, each step as the phase
    wrote it (a ``Step`` carries its helper, a plain ``str`` is prose); *run_time* is
    ``[(phase name, detail)]`` for each phase decided at run time, *detail* naming the facts it
    decides on. A phase is decided at run time when a test on one of its facts is made while
    the template is built (``if find.uids``), or when its steps differ with those facts absent
    (``if book is None: return []``, which a placeholder never reads as absent), or when it
    fans out until a callable ``until`` says stop. Raises PhaseError or PlanningCommandCallError
    for a combination the planner would not see.
    """
    earlier: set[str] = set()
    run_time: list[tuple[str, str]] = []
    phases = planning_commands.phases_for(command, env) if command.plan_fn is not None else None
    if phases is None:
        with planning_api.steps_by_phase_noted() as by_phase:
            lines = planning_commands._call_steps(
                command.steps_fn, planning_commands.input_instance(command, env), command.path)
        if by_phase:
            return list(by_phase), run_time
        return [(command.name, line) for line in lines], run_time
    steps: list[tuple[str, object]] = []
    for phase in phases:
        kwargs = _template_kwargs(phase, earlier)
        fact_names = sorted(name for name in kwargs if name in earlier)
        if phase.output is not None:
            earlier.add(phase.name)
        with building_template(), tests_noted() as tests, \
                planning_api.steps_by_phase_noted() as present:
            try:
                phase.steps_fn(**kwargs)
            except Exception as error:  # noqa: BLE001 - a phase the template cannot build
                raise PhaseError(f"phase {phase.name!r} fails while its template is built: "
                                 f"{type(error).__name__}: {error}") from None
        steps.extend(present)
        absent_shape = None
        if fact_names:
            with building_template(), planning_api.steps_by_phase_noted() as absent:
                try:
                    phase.steps_fn(**{**kwargs, **{n: None for n in fact_names}})
                    absent_shape = _shape(absent)
                except Exception:  # noqa: BLE001 - a phase that needs its facts (fails without them)
                    absent_shape = None
        if tests:
            run_time.append((phase.name, f"it tests the earlier facts {', '.join(tests)}"))
        elif absent_shape is not None and absent_shape != _shape(present):
            run_time.append((phase.name, f"it is skipped or changes when the earlier facts "
                                          f"{', '.join(fact_names)} are absent"))
        elif phase.until is not None:
            run_time.append((phase.name, "its fan-out stops on a callable until over the earlier facts"))
    return steps, run_time


def planning_command_needs_agent(workflow_folderpath: str,
                                 name: str) -> tuple[bool, list[str]]:
    """``(needs_agent, reasons)`` for the planning command *name* (qualified, e.g. ``Library/loan_report``)
    of the workflow at *workflow_folderpath*.

    *reasons* is the list of rules that make it need the agent, in phase order, each saying the
    cause and, where there is one, the change that would make it mechanical. An empty list means
    the planning command is mechanical and runs in assistant mode by its exact name. Raises
    ValueError when *name* is not a loaded planning command of the workflow (also when its file
    does not load), and when no combination of its arguments builds its steps.
    """
    command = _loaded(workflow_folderpath, name)
    reasons: dict[str, None] = {}

    for p in command.params:
        if p.verify:
            reasons[f"field {p.name!r} carries verify: a classifier model decides it from the "
                    f"request ({VERIFY_HINT})"] = None

    grid = planning_commands._template_grid(command)
    built = 0
    for env in grid:
        try:
            steps, run_time = _step_lines(command, env)
        except (PhaseError, planning_commands.PlanningCommandCallError):
            continue
        built += 1
        for phase_name, step in steps:
            text = planning_commands.without_condition_note(step)
            helper = getattr(step, "helper", None)
            if helper is None:
                reasons[f"phase {phase_name!r} has a prose step the agent carries out: "
                        f"{text!r} ({PROSE_HINT})"] = None
            elif helper == "when":
                reasons[f"phase {phase_name!r} has a conditional step the agent decides whether "
                        f"to run: {text!r} ({WHEN_HINT})"] = None
        for phase_name, detail in run_time:
            reasons[f"phase {phase_name!r} is decided at run time: {detail} ({RUN_TIME_HINT})"] = None
    if built == 0:
        raise ValueError(f"{name!r}: no combination of its arguments builds its steps")

    if not command.uses_result:
        reasons[REASON_RESULT] = None
    return bool(reasons), list(reasons)


def refusal_message(qualified_name: str, reasons: list[str]) -> str:
    """The assistant-mode refusal (design 3.4) for a planning command that needs the agent, with
    the reasons from ``planning_command_needs_agent``."""
    return (f"{qualified_name} is a planning command; it runs in agent mode. "
            f"It needs agent mode because {'; '.join(reasons)}.")
