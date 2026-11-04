"""``fastworkflow command run-code`` and ``fastworkflow.planning.run_code``: one call of a command, with no model.

A planning command's call runs phase by phase, as the runtime runs it (``planning_phases``): each
phase's steps as the runtime writes them, each step code can run run against the workflow's real
execution commands (``CommandExecutor.perform_action`` on a real ``Workflow``), each move, the facts
filled from those runs, and what ``__result__`` returns. The walk stops at the first step code cannot
run (an open step: the agent would take over there), exit 0. A step that names a planning command
runs that call the same way, indented. An execution command runs and returns its ``CommandOutput``.

Nothing here calls a model: the planning steps are read by code (``plan_code_steps``, ``pick_step``),
and the commands a step runs are the workflow's own. Their side effects are real, so run it on a
development instance of the app (its help says so).

``run_code(workflow_folder, command, input=None, start_with=())`` returns a ``CodeRun``; a run that
cannot go on raises ``RunCodeError`` (its ``text`` is what it printed before it stopped).
"""
from __future__ import annotations

import dataclasses
import json
import os
import uuid
from typing import Any, Callable, Iterable, Optional

CORE = ["go_up", "reset_context"]


class RunCodeError(Exception):
    """A run that cannot go on: a command that does not load, a step that fails, a hook that raises.

    ``text`` is what the run printed before it stopped."""

    def __init__(self, message: str, text: str = ""):
        super().__init__(message)
        self.text = text


@dataclasses.dataclass
class CodeRun:
    """The result of a run: what it prints (``text``), a planning command's answer and ``Output``
    (``answer``, ``output``), the facts of its top-level phases (``facts``), and the open step the run
    stopped at (``open_step``, None when the run completed)."""

    text: str
    answer: str = ""
    output: Any = None
    facts: dict = dataclasses.field(default_factory=dict)
    open_step: Optional[str] = None


def _plain(name: str) -> str:
    from fastworkflow.observation_offloading.finish_check import plain_command_name

    return plain_command_name(name)


def _open_session(folder: str):
    """A real ``Workflow`` on *folder*, started as ``fastworkflow run`` starts one with its startup
    action: the workflow's ``startup_action.json`` when it has one. No env file, no model."""
    import fastworkflow
    from fastworkflow import Action
    from fastworkflow.command_executor import CommandExecutor
    from fastworkflow.command_routing import RoutingDefinition

    if fastworkflow.RoutingRegistry is None:   # init sets the registries; {} is enough: no model is read
        fastworkflow.init({})
    # The command-class cache is keyed by command name alone (fix-3rsq.phase10-routing-class-cache), so a
    # run after another workflow in the same process would get that workflow's classes (its ``startup``
    # opens the other workflow's root object). Each run loads its own classes.
    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()
    workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"run-code-{uuid.uuid4().hex}")
    startup = os.path.join(folder, "startup_action.json")
    if os.path.isfile(startup):
        with open(startup, encoding="utf-8") as fh:
            action = Action(**json.load(fh))
        try:
            CommandExecutor.perform_action(workflow, action)
        except Exception as error:  # noqa: BLE001 - reported with the command that failed
            raise RunCodeError(f"startup action {action.command_name!r} failed: "
                               f"{type(error).__name__}: {error}") from error
    return workflow


class _Walk:
    """One run: the session, the workflow's planning commands and parameter classes, and the lines printed."""

    def __init__(self, folder: str, workflow):
        from fastworkflow import planning_commands, workflow_agent
        from fastworkflow import planning_phases as sp
        from fastworkflow.command_routing import RoutingRegistry

        self.folder = folder
        self.wf = workflow
        self.sp = sp
        self.planning_commands = planning_commands
        self.routing = RoutingRegistry.get_definition(folder)
        self.planning = planning_commands.resolve_field_sources(folder).planning_commands
        self.planning_map = workflow_agent.load_planning_command_map(folder)
        self.parameters = sp.parameter_classes(self.routing, self.planning)
        self.planning_names = {_plain(name).lower() for name in self.planning}
        # The names a step may name, as the run reads them (WorkflowExecutionContext._workflow_command_names):
        # the routed commands (qualified in a context) and the planning commands.
        self.commands = sorted({n for n in self.routing.routing_definition_map if n != "wildcard"}
                               | set(self.planning))
        self.lines: list[str] = []
        self.call_runs: list[dict] = []
        self.open_step: Optional[str] = None
        self.top_facts: dict = {}

    # ------------------------------------------------------------------ output and names

    def emit(self, indent: int, text: str) -> None:
        self.lines.extend(" " * indent + line for line in text.splitlines() or [""])

    def text(self) -> str:
        return "\n".join(self.lines)

    def context_name(self) -> str:
        """The name of the workflow's current context (its root object's too: ``Library``), ``""`` for global."""
        try:
            return str(getattr(self.wf, "current_command_context_name", "") or "")
        except Exception:  # noqa: BLE001 - a context that cannot be named is the global one here
            return ""

    def clause(self) -> str:
        from fastworkflow.context_identity import context_clause_for

        try:
            return context_clause_for(self.wf)
        except Exception:  # noqa: BLE001
            return ""

    def qualify(self, name: str) -> str:
        """The routing name of execution command *name* in the workflow's current context."""
        names = self.routing.routing_definition_map
        context = self.context_name()
        if context and f"{context}/{name}" in names:
            return f"{context}/{name}"
        if name in names:
            return name
        found = [n for n in names if n != "wildcard" and _plain(n) == name]
        return found[0] if len(found) == 1 else name

    def planning_key(self, name: str) -> Optional[str]:
        """The planning command *name* (plain or ``Context/name``) names, preferring the current context."""
        if name in self.planning_map:
            return name
        keys = sorted(k for k in self.planning_map if _plain(k).lower() == name.lower())
        context = self.context_name()
        for key in keys:
            if context and key.lower() == f"{context}/{name}".lower():
                return key
        return keys[0] if keys else None

    def hook(self, where: str, fn: Callable[[], Any]) -> Any:
        """Call *fn* (a planning command's hook, or a command's code); a raise is a RunCodeError naming *where*."""
        try:
            return fn()
        except RunCodeError:
            raise
        except Exception as error:  # noqa: BLE001 - the run stops, naming the hook
            raise RunCodeError(f"{where} raised {type(error).__name__}: {error}") from error

    # ------------------------------------------------------------------ code steps

    def execute(self, text: str, name: str):
        """Run one step *text* (``name k=v ...``) as the workflow's own command: ``(CommandOutput, parameters,
        routing name)``. The routing name is read before the run, which may move the workflow."""
        from fastworkflow import Action
        from fastworkflow.command_executor import CommandExecutor
        from fastworkflow import ModuleType
        from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
            ParameterExtraction)

        qualified = self.qualify(name)
        cls = self.routing.get_command_class(qualified, ModuleType.COMMAND_PARAMETERS_CLASS)
        try:
            params: dict = {}
            if cls is not None:
                bound = (cls() if text.strip() == name else
                         ParameterExtraction._extract_parameters_from_key_values(text, cls, qualified))
                params = bound.model_dump() if bound is not None else {}
            action = Action(command_name=qualified, command=text, parameters=params)
            out = CommandExecutor.perform_action(self.wf, action)
        except RunCodeError:
            raise
        except Exception as error:  # noqa: BLE001 - the step fails, naming the command
            raise RunCodeError(f"step {text!r} failed: {qualified} raised "
                               f"{type(error).__name__}: {error}") from error
        if not out.command_response.success:
            raise RunCodeError(f"step {text!r} failed: {out.command_response.response}")
        return out, params, qualified

    def run_code_step(self, run: str, indent: int, runs: list[dict]) -> None:
        name = run.split()[0]
        before = self.clause()
        out, params, qualified = self.execute(run, name)
        after = self.clause()
        self.emit(indent, f"code runs: {run}")
        if after and after != before:
            self.emit(indent, f"now in: {after}")
        output = out.command_response.output
        entry = {"command": qualified, "input": params, "output": output.model_dump() if output is not None else {},
                 "success": True, "context_clause": before}
        runs.append(entry)
        self.call_runs.append(entry)

    def nested_arguments(self, run: str) -> tuple[str, dict]:
        """``(planning command key, arguments)`` of a step that names a planning command."""
        name = run.split()[0]
        key = self.planning_key(name)
        if key is None:
            raise RunCodeError(f"step {run!r} names {name}, which is not a planning command of this workflow")
        classes = [c for c in self.parameters(name) if c is not None]
        cls = classes[0] if classes else None
        bound = None
        if cls is not None:
            bound = cls() if run.strip() == name else self._extract(run, cls, name)
        return key, (bound.model_dump() if bound is not None else {})

    @staticmethod
    def _extract(run: str, cls, name: str):
        from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
            ParameterExtraction)

        return ParameterExtraction._extract_parameters_from_key_values(run, cls, name)

    # ------------------------------------------------------------------ planning commands

    def run_planning(self, key: str, args: dict, indent: int, stack: tuple[str, ...]) -> Optional[tuple[str, Any]]:
        """Run the planning command *key* with *args*. Returns None when a step is open (the run stops),
        else ``(answer, output)``, the answer and Output of ``__result__`` (``("", None)`` without one)."""
        sp = self.sp
        if key in stack:
            raise RunCodeError(f"{key} calls itself through its steps (a cycle)")
        command = self.planning_map[key]
        phases = self.hook(f"the call of {key}", lambda: self._phases(command, args))
        if len(phases) == 1 and phases[0].name == _plain(key):
            self.emit(indent, "Runs flat: one plan, one executor run.")
        outputs: dict = {}
        facts: dict = {}
        for index, phase in enumerate(phases):
            last = index == len(phases) - 1
            where = f"the hook {phase.name}() of {key}"
            if phase.fans_out:
                items, _ = self.hook(where, lambda: sp.fan_out_items(phase, outputs))
                filled = []
                if not items:
                    self.emit(indent, f"Phase {phase.name}: no items, so it runs nothing")
                for item in items:
                    self.emit(indent, f"Phase {phase.name} (item {item})")
                    text, runs, stop, nested = self.run_steps(key, command, phase, outputs, last, indent,
                                                              stack, item)
                    if stop:
                        return None
                    if phase.output is not None:
                        filled.append(self.fill(phase, runs, text, nested))
                        self.emit(indent + 2, f"facts: {filled[-1]!r}")
                outputs[phase.name] = filled
                continue
            if not self.hook(where, lambda: sp.runs_now(phase, outputs)):
                self.emit(indent, f"Phase {phase.name}: skipped (its method returned [])")
                continue
            self.emit(indent, f"Phase {phase.name}")
            text, runs, stop, nested = self.run_steps(key, command, phase, outputs, last, indent, stack)
            if stop:
                return None
            if phase.output is not None:
                outputs[phase.name] = self.fill(phase, runs, text, nested)
                facts[phase.name] = outputs[phase.name]
                self.emit(indent + 2, f"facts: {outputs[phase.name]!r}")
        if not stack:
            self.top_facts = facts
        if not command.uses_result:
            return "", None
        return self.result(key, command, args, outputs, phases, indent)

    def _phases(self, command, args):
        return self.planning_commands.call_phases(command, args)

    def run_steps(self, key, command, phase, outputs, last, indent, stack, item=None):
        """One phase's steps: each code runs, each planning command step runs (indented), an open step stops."""
        sp = self.sp
        if item is None:
            steps = self.hook(f"the hook {phase.name}() of {key}", lambda: sp.phase_steps(phase, outputs))
        else:
            steps = self.hook(f"the hook {phase.name}() of {key}",
                              lambda: sp.phase_steps(phase, outputs, item))
        text = self.planning_commands.number(steps)
        for line in text.splitlines():
            self.emit(indent + 2, line)
        runs: list[dict] = []
        nested: list = []   # the Outputs of the planning command calls this phase's steps made
        open_step = False
        for step, run, _ in sp.plan_code_steps(text, self.commands, CORE, self.parameters):
            if run == "":
                continue
            if run is None:
                found = sp.pick_listing(step, self.call_runs, self.parameters(step.command))
                if found is not None:
                    run, _ = sp.pick_step(step, *found, self.parameters(step.command))
            if run is None:
                self.emit(indent + 2, f"step {step.number} is open: the agent would run it and the rest")
                self.open_step = f"step {step.number} of {phase.name}: {step.text}"
                open_step = True
                break
            if run.split()[0].lower() in self.planning_names:
                nested_key, nested_args = self.nested_arguments(run)
                self.emit(indent + 2, f"step {step.number} runs planning command {run.split()[0]}:")
                result = self.run_nested(nested_key, nested_args, indent + 4, stack + (key,))
                if result is None:
                    open_step = True
                    break
                if result[1] is not None:
                    nested.append(result[1])
                continue
            self.run_code_step(run, indent + 2, runs)
        if not open_step:
            if sp.is_answer_only(text, self.commands, CORE):
                if sp.answer_by_code(text):
                    self.emit(indent + 2, "answer written by code:")
                    self.emit(indent + 4, sp.answer_from_steps(text))
                else:
                    self.emit(indent + 2, "answer written by the model (one call, no tools)")
            elif last and not command.uses_result:
                self.emit(indent + 2, "the agent writes the answer from these runs")
        return text, runs, open_step, nested

    def run_nested(self, key: str, args: dict, indent: int, stack: tuple[str, ...]):
        command = self.planning_map[key]
        full = {p.name: p.default for p in command.params if not p.required} | args
        self.validate(command, full)
        return self.run_planning(key, full, indent, stack)

    def fill(self, phase, runs, text, nested):
        """The phase's facts: the Output of the planning command call its steps made, when it has one that fits
        the phase's facts model; else ``fill_output``, on the record the phase ended on (or the one its steps ran on)."""
        sp = self.sp
        if nested:
            from pydantic import ValidationError

            value = nested[-1]
            if isinstance(value, phase.output):
                return value
            try:
                return phase.output.model_validate(value.model_dump())
            except ValidationError:
                pass
        ended_on = self.clause()
        record = ended_on or None
        rebound = sp.planned_record(runs, text, ended_on=ended_on, commands=self.commands,
                                    core_commands=CORE)
        if rebound is not None:
            record = rebound
        return self.hook(f"the facts of phase {phase.name}",
                         lambda: sp.fill_output(phase.output, runs, record=record, plan_text=text,
                                                commands=self.commands, core_commands=CORE,
                                                ended_on=ended_on or None))

    def validate(self, command, args: dict) -> None:
        try:
            self.planning_commands._validate_arguments(command, args, "")
        except self.planning_commands.PlanningCommandInputError as error:
            raise RunCodeError(str(error)) from error

    def result(self, key, command, args, outputs, phases, indent):
        """What ``__result__`` returns for these phase outputs: ``(answer, Output)``."""
        from fastworkflow import planning_commands as pc

        sp = self.sp

        def call():
            result_fn = sp._result_fn_on_class(command.plan_cls)
            kwargs = sp.bound_result_arguments(
                result_fn, pc.input_instance(command, args), sp.typed_phase_outputs(outputs, phases),
                phase_names=frozenset(pc._phase_names_for_command(command)))
            return result_fn(command.plan_cls(), **kwargs)

        raw = self.hook(f"the hook __result__() of {key}", call)
        if isinstance(raw, str):
            answer, output = raw, None
        else:
            answer, output = raw.response, getattr(raw, "output", None)
        self.emit(indent, "answer written by __result__:")
        for line in str(answer).splitlines():
            self.emit(indent + 2, line)
        if output is not None:
            self.emit(indent, f"output: {output!r}")
        return answer, output


def run_code(workflow_folder: str, command: str, input: Optional[dict] = None,
             start_with: Iterable[str] = ()) -> CodeRun:
    """Run one call of *command* on the workflow at *workflow_folder*, with no model (see the module doc).

    *input* gives the command's arguments (validated as a call's are). *start_with* are steps run by code
    in order before the command, to put the workflow where the command starts (``open_book book_uid=b1``).
    Raises ``RunCodeError`` when the command does not load, a step fails, or a hook raises.
    """
    folder = os.path.abspath(workflow_folder)
    try:
        from fastworkflow.command_directory import check_no_planning_commands_folder

        check_no_planning_commands_folder(folder)
    except ValueError as error:
        raise RunCodeError(str(error)) from error
    workflow = _open_session(folder)
    walk = _Walk(folder, workflow)
    try:
        for step in start_with:
            walk.execute(step, step.split()[0])
            walk.emit(0, f"started with: {step}")
        key = walk.planning_key(command)
        if key is not None:
            return _planning_run(walk, key, input or {})
        return _execution_run(walk, command, input)
    except RunCodeError as error:
        error.text = walk.text()
        raise


def _planning_run(walk: _Walk, key: str, given: dict) -> CodeRun:
    command = walk.planning_map[key]
    if "/" in key:
        context = key.split("/")[0]
        if walk.context_name() != context:
            raise RunCodeError(f"{key} is a {context} command: the workflow must be on a {context} record "
                               f"first; give --start-with (e.g. open_book book_uid=b1)")
    args = {p.name: p.default for p in command.params if not p.required} | given
    walk.validate(command, args)
    result = walk.run_planning(key, args, 0, ())
    if result is None:
        return CodeRun(text=walk.text(), facts=walk.top_facts, open_step=walk.open_step)
    answer, output = result
    return CodeRun(text=walk.text(), answer=answer, output=output, facts=walk.top_facts)


def _execution_run(walk: _Walk, command: str, input: Optional[dict]) -> CodeRun:
    from fastworkflow import Action
    from fastworkflow.command_executor import CommandExecutor

    name = _plain(command)
    qualified = walk.qualify(name)
    if qualified not in walk.routing.routing_definition_map:
        raise RunCodeError(f"no command named {command!r} in this workflow")
    text = json.dumps(input) if input else qualified
    action = Action(command_name=qualified, command=text, parameters=dict(input or {}))
    try:
        out = CommandExecutor.perform_action(walk.wf, action)
    except Exception as error:  # noqa: BLE001 - the command failed, naming it
        raise RunCodeError(f"{qualified} raised {type(error).__name__}: {error}") from error
    response = out.command_response
    walk.emit(0, qualified)
    walk.emit(2, f"context: {out.context or '(global)'}")
    walk.emit(2, f"response: {response.response}")
    if response.output is not None:
        walk.emit(2, f"output: {response.output!r}")
    if not response.success:
        raise RunCodeError(f"{qualified} did not succeed: {response.response}", walk.text())
    return CodeRun(text=walk.text(), answer=response.response, output=response.output)


def run_code_main(args) -> int:
    """``fastworkflow command run-code``: print the run; exit 0 when it completes or stops at an open step,
    1 when the command does not load, a step fails or a hook raises (the message goes to stderr)."""
    import sys

    try:
        input_value = json.loads(args.input) if args.input else None
    except json.JSONDecodeError as error:
        print(f"run-code: --input is not JSON: {error}", file=sys.stderr)
        return 1
    if input_value is not None and not isinstance(input_value, dict):
        print("run-code: --input must be a JSON object", file=sys.stderr)
        return 1
    try:
        result = run_code(args.workflow_folder, args.command, input_value, args.start_with or ())
    except RunCodeError as error:
        if error.text:
            print(error.text)
        print(f"run-code: {error}", file=sys.stderr)
        return 1
    print(result.text)
    return 0
