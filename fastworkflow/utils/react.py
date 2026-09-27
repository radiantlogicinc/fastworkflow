import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError
from litellm import exceptions as litellm_exceptions

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

from fastworkflow import tracing
from fastworkflow.utils.dspy_logger import DSPyForward

logger = logging.getLogger(__name__)

# Temporary disable-only controls retained for controlled runs and provenance.
# Historical four-arm reproduction uses its pinned revisions.
EVAL_FINISH_REMINDERS_ENV = "FW_EVAL_FINISH_REMINDERS"


def _evaluation_control(
    name: str, *, default_enabled: bool = True
) -> tuple[bool, str | None]:
    raw = os.environ.get(name)
    if raw is None:
        return default_enabled, None
    if raw != "0":
        raise ValueError(f"{name} must be exactly 0 when set")
    return False, raw

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class AskUserSuspend(BaseException):
    """
    Raised by ask_user when no user_message_queue is configured (Topology B).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; the loop catches this explicitly and returns a suspended sentinel.
    """

    def __init__(self, clarification_request: str):
        self.clarification_request = clarification_request
        super().__init__(clarification_request)


class NoSuspendedAgentStateError(RuntimeError):
    """Resume requested but no suspended ReAct trajectory exists.

    Happens when ``_awaiting_user`` is set after the trajectory was already
    consumed (e.g. a deferred resume still in flight that later failed, or a
    restored blob that lost ``react``) so a second message cannot honestly
    continue the turn. Embedders map this to HTTP 409 Conflict — not 500.
    """


class fastWorkflowReAct(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10,
                 on_step_complete: Callable[[int, dict], bool] | None = None):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.iteration_counter = 0
        self.finish_reminders_enabled, reminder_override = _evaluation_control(
            EVAL_FINISH_REMINDERS_ENV
        )
        self.evaluation_control_overrides = {
            name: value
            for name, value in ((EVAL_FINISH_REMINDERS_ENV, reminder_override),)
            if value is not None
        }

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        instr.extend(f"({idx + 1}) {tool}" for idx, tool in enumerate(tools.values()))
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        # Build the ReAct signature with trajectory input.
        # available_commands is injected into system message by CommandsSystemPreludeAdapter
        # (see fastworkflow/utils/chat_adapter.py) and is NOT included in the trajectory
        # formatting to avoid token bloat across iterations.
        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

        self.inputs = {}
        self.current_trajectory = {}
        self._on_step_complete = on_step_complete
        self._suspended: dict[str, Any] | None = None
        # True when the most recent _run_loop ended because max_iters was
        # reached without the agent selecting the `finish` tool.
        self._exhausted_last_run = False
        # How many finish-check notes this TURN has injected (ido-8ps.27's roster
        # nudge until fix-4dsr). The cap is one, so a turn can be reminded and
        # can then still decide it is done.
        self._finish_notes_fired = 0
        # Attached by the builder (observation_offloading/agent.py): the
        # finish-time execution check and where to read the turn's plan. None
        # means no check.
        self.finish_checker = None
        self.plan_source = None
        # How many times the context-window fallback has truncated a trajectory
        # in this process. Only read as a delta around one call (ido-8ps.18, to
        # tell an extract that overflowed from one that did not); it changes
        # nothing about what the fallback does.
        self._truncation_count = 0

    def _note_step(self, idx: int, tool_name: str) -> None:
        """A step has been chosen and mirrored, and has not been dispatched yet.

        A no-op here. ``StructuredContinuationReAct`` overrides it to assign the
        step's ``O`` ordinal, which has to be decided before the tool runs
        because the command inside the tool declares a result handle under it.
        """

    def clear_suspension(self) -> None:
        """Drop any in-memory suspended ReAct state (used on abort/finalize)."""
        self._suspended = None

    def export_suspended(self) -> dict[str, Any] | None:
        """Return a JSON-serializable copy of suspended ReAct state, or None."""
        if self._suspended is None:
            return None
        return {
            "trajectory": dict(self._suspended["trajectory"]),
            "idx": self._suspended["idx"],
            "input_args": dict(self._suspended["input_args"]),
            "max_iters": self._suspended["max_iters"],
            "clarification": self._suspended.get("clarification"),
            "iteration_counter": self.iteration_counter,
            "finish_notes_fired": getattr(self, "_finish_notes_fired", 0),
        }

    def import_suspended(self, data: dict[str, Any]) -> None:
        """Restore suspended ReAct state from export_suspended() output."""
        self._suspended = {
            "trajectory": dict(data["trajectory"]),
            "idx": data["idx"],
            "input_args": dict(data["input_args"]),
            "max_iters": data["max_iters"],
            "clarification": data.get("clarification"),
        }
        self.iteration_counter = data.get("iteration_counter", 0)
        # A suspension exported before the roster nudge was replaced carries its
        # count under the old key; the cap it enforces is the same.
        self._finish_notes_fired = data.get(
            "finish_notes_fired", data.get("roster_nudges_fired", 0)
        )

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    @DSPyForward.intercept
    def forward(self, **input_args):
        self.inputs = input_args
        self.clear_suspension()

        # Reset the full-trajectory mirror at the start of each logical turn.
        # resume() must NOT reset it, so a suspended->resumed turn accumulates one
        # coherent trajectory. current_trajectory is a SEPARATE object from the
        # working `trajectory` below (which is what gets stashed in _suspended),
        # so mirroring into it never corrupts suspend/resume bookkeeping.
        self.current_trajectory = {}
        self._finish_notes_fired = 0

        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        idx = 0
        exception_count = 0

        suspended = self._run_loop(
            trajectory, idx, input_args, max_iters, exception_count
        )
        if suspended is not None:
            return suspended

        extract = self._extract_prediction(trajectory, **input_args)
        return dspy.Prediction(
            trajectory=trajectory, exhausted=self._exhausted_last_run, **extract
        )

    def resume(self, observation: str):
        """Resume a suspended run after the user answered an ask_user clarification."""
        if self._suspended is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )

        stash = self._suspended
        trajectory = stash["trajectory"]
        idx = stash["idx"]
        input_args = stash["input_args"]
        max_iters = stash["max_iters"]

        # Keep self.inputs pointing at the active run's arg dict so any mid-run refresh
        # (e.g. available_commands re-scoping after a context switch) mutates the same
        # dict this loop unpacks on each step.
        self.inputs = input_args

        trajectory[f"observation_{idx}"] = observation
        # Mirror the resumed observation (the user's ask_user answer) into
        # current_trajectory. Without this the highest-value context — what the
        # user said in response to the clarification — would be missing from the
        # trajectory the planner and distillation see.
        self.current_trajectory[f"observation_{idx}"] = observation
        idx += 1
        self.iteration_counter += 1
        self._suspended = None

        suspended = self._run_loop(trajectory, idx, input_args, max_iters, 0)
        if suspended is not None:
            return suspended

        extract = self._extract_prediction(trajectory, **input_args)
        return dspy.Prediction(
            trajectory=trajectory, exhausted=self._exhausted_last_run, **extract
        )

    def _run_loop(
        self,
        trajectory: dict[str, Any],
        idx: int,
        input_args: dict[str, Any],
        max_iters: int,
        exception_count: int,
    ):
        """
        Run the ReAct tool loop until finish, max_iters, or AskUserSuspend.

        Returns a suspended Prediction, or None when the loop completed normally.
        Sets ``self._exhausted_last_run`` when the loop ends because max_iters
        was reached without the agent selecting the `finish` tool.
        """
        self._exhausted_last_run = False
        # Host for the fw.agent.step spans, bound by the caller around the whole
        # agent run. None outside an observed turn, where every helper no-ops.
        host = tracing.current_host()
        while True:
            # Opened before the reasoning call so a step that fails to pick a
            # tool is still a recorded step rather than a gap in the trace.
            step_span = tracing.start_span(
                host,
                tracing.SPAN_AGENT_STEP,
                attributes={"step_index": idx},
            )
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
                if pred is None:
                    raise ValueError("Tool returned is None")
            except ValueError as err:
                invalid_tool_obs = (
                    f"Agent failed to select a valid tool: {_fmt_exc(err)}"
                )
                trajectory[f"observation_{idx}"] = invalid_tool_obs
                self.current_trajectory[f"observation_{idx}"] = invalid_tool_obs
                idx += 1
                recovery_thought = (
                    "To execute a command, I should use one of the available tools"
                )
                recovery_obs = (
                    "Use the appropriate tool with proper arguments (correctly formatted)"
                )
                trajectory[f"thought_{idx}"] = recovery_thought
                trajectory[f"observation_{idx}"] = recovery_obs
                self.current_trajectory[f"thought_{idx}"] = recovery_thought
                self.current_trajectory[f"observation_{idx}"] = recovery_obs
                idx += 1
                exception_count += 1
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_ERROR,
                    attributes={
                        "observation": invalid_tool_obs,
                        "recovered": exception_count <= 2,
                    },
                )
                if exception_count > 2:
                    break
                continue
            except BaseException as err:
                # Anything else from the reasoning call — AdapterParseError,
                # provider errors, control signals. The caller's retry loop
                # re-enters this method, and a step span left on the stack
                # would parent the ENTIRE retried attempt under a phantom
                # span that is never emitted. Close it, then propagate.
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_ERROR,
                    attributes={"error_type": type(err).__name__},
                )
                raise

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            step_status = tracing.STATUS_OK
            step_attributes = {
                "step_index": idx,
                "thought": pred.next_thought,
                "tool_name": pred.next_tool_name,
                "tool_args": pred.next_tool_args,
            }

            # Mirror the full step into current_trajectory (consumed by the planner
            # for replanning and by distillation as the agent trajectory). Keep the
            # legacy action_{idx} entry too for any consumer that still reads it.
            self.current_trajectory[f"thought_{idx}"] = pred.next_thought
            self.current_trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            self.current_trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            self.current_trajectory[f"action_{idx}"] = (
                f"{pred.next_tool_name}: {pred.next_tool_args}"
            )
            # The step exists and its tool is known, and nothing has dispatched
            # yet: the one moment a subclass can number it before any code the
            # tool calls asks what its number is (ido-7qd).
            self._note_step(idx, pred.next_tool_name)

            try:
                observation = self.tools[pred.next_tool_name](**pred.next_tool_args)
                trajectory[f"observation_{idx}"] = observation
                self.current_trajectory[f"observation_{idx}"] = observation
                step_attributes["observation"] = _as_text(observation)
            except AskUserSuspend as err:
                self._suspended = {
                    "trajectory": trajectory,
                    "idx": idx,
                    "input_args": input_args,
                    "max_iters": max_iters,
                    "clarification": err.clarification_request,
                }
                # The step really did end here — the human wait that follows is
                # fw.ask_user's to record, and this span must not stay open
                # across a suspension that may resume in another process.
                step_attributes["clarification"] = err.clarification_request
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.STATUS_AWAITING_USER,
                    attributes=step_attributes,
                )
                return dspy.Prediction(
                    suspended=True,
                    clarification=err.clarification_request,
                    exhausted=False,
                )
            except Exception as err:
                error_observation = (
                    f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
                )
                trajectory[f"observation_{idx}"] = error_observation
                self.current_trajectory[f"observation_{idx}"] = error_observation
                step_attributes["observation"] = error_observation
                step_attributes["tool_error"] = type(err).__name__
                step_status = tracing.STATUS_ERROR
            except BaseException as err:
                # Control signals from a tool (e.g. CommandCancelledError) end
                # the run — close the step span so a cancelled turn keeps its
                # last step record instead of leaking an open span.
                tracing.end_span(
                    host,
                    step_span,
                    status=tracing.status_for_dispatch_exception(err),
                    attributes={**step_attributes, "error_type": type(err).__name__},
                )
                raise

            # ido-8ps.27 / fix-4dsr: the one interception point. The finish action has been
            # recognised and its "Completed." observation written, the answer
            # has NOT been extracted yet, and the loop is by definition not
            # exhausted. If the finish check finds plan steps the turn's record
            # shows no command carrying out, and there is budget to reach them,
            # the observation of this step becomes a bounded note naming them
            # and the loop continues. At most one per turn; never on
            # exhaustion; never an ask_user round.
            nudge = ""
            if pred.next_tool_name == "finish":
                nudge = self._intercept_finish(trajectory, idx, input_args, max_iters)
                if nudge:
                    step_attributes["observation"] = nudge
                    step_attributes["finish_check_note"] = True

            tracing.end_span(
                host, step_span, status=step_status, attributes=step_attributes
            )

            # Step-completion callback for distillation: lets external code inspect
            # each completed step and stop execution early (e.g. on trajectory
            # divergence). Placed AFTER the AskUserSuspend catch so it can never
            # swallow a suspension, and it does not touch _suspended state.
            # getattr guard: resume() may run on an instance built via __new__
            # (test helpers) that never set this attribute.
            on_step_complete = getattr(self, "_on_step_complete", None)
            if on_step_complete and not on_step_complete(idx, trajectory):
                break

            if pred.next_tool_name == "finish" and not nudge:
                break

            idx += 1
            self.iteration_counter += 1
            if self.iteration_counter >= max_iters:
                logger.warning("Max iterations reached")
                self._exhausted_last_run = True
                break

        return None

    def _intercept_finish(self, trajectory, idx, input_args, max_iters) -> str:
        """The finish action's one interception point, for BOTH loops.

        The note, and the rule that a fired note REPLACES this step's
        observation and returns control to the loop, used to live inline in
        ``_run_loop`` -- so ``aforward`` recognised finish and broke with no
        note and no fired-note bookkeeping: two loops, two different
        behaviours for the same rule.
        The note and the trajectory writes are here; what stays with each loop
        is what only that loop has -- the sync loop's step span attributes.

        Returns the note, or ``""`` when there is none, which is what each loop
        tests to decide whether to break on the finish action.
        """
        nudge = self._finish_check_note(input_args, max_iters)
        if nudge:
            trajectory[f"observation_{idx}"] = nudge
            self.current_trajectory[f"observation_{idx}"] = nudge
        return nudge

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        # The per-TURN state the note's "at most one" is counted in, reset here
        # for the same reason `forward` resets it (ido-dpx/F15): this call is a
        # logical turn, and a turn inherits neither the previous turn's mirror
        # nor its note count.
        self.current_trajectory = {}
        self._finish_notes_fired = 0
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                # A fired nudge replaces this step's observation and returns
                # control to the loop; no nudge ends the turn, as before.
                if not self._intercept_finish(
                    trajectory, idx, input_args, max_iters
                ):
                    break
            # What `_finish_check_note` reads to know how much room is left. The sync
            # loop has always counted its steps here; without the same count the
            # async loop would offer a note on a turn with nothing left to do.
            self.iteration_counter += 1

        extract = await self._async_extract_prediction(trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _finish_check_note(self, input_args, max_iters) -> str:
        """The finish-time execution check's note, or ``""``.

        Called from ``_intercept_finish``, the one place a finish action is
        recognised, before answer extraction. The check itself
        (``fastworkflow.observation_offloading.finish_check``) is attached by whoever builds the agent
        as ``finish_checker``; without one, or with it switched off, there is no
        note and the turn finishes as it always did.

        ``iterations_left`` is what the agent would still have AFTER spending
        this step on the note: the loop increments the counter once more and
        stops at ``max_iters``. The check declines below its minimum, which is
        how "never on exhaustion" is kept -- a turn with no room is a turn the
        note cannot help.
        """
        checker = getattr(self, "finish_checker", None)
        if checker is None or not getattr(self, "finish_reminders_enabled", True):
            return ""
        if getattr(self, "_finish_notes_fired", 0) >= 1:
            return ""
        left = int(max_iters) - int(getattr(self, "iteration_counter", 0)) - 1
        try:
            text = checker.note(self, input_args, iterations_left=left)
        except Exception as error:  # noqa: BLE001 - a note must never fail a turn
            logger.warning("finish check skipped: %s: %s", type(error).__name__, error)
            return ""
        if text:
            self._finish_notes_fired = getattr(self, "_finish_notes_fired", 0) + 1
        return text

    def _rehydrate_for_extract(self, trajectory):
        """``(trajectory_for_the_extractor, report, budget, scope_id)``.

        It returns a COPY in which offload labels carry the stored evidence
        behind them (see ``fastworkflow.answer_rehydration``). The loop's own
        trajectory is then never the object passed on, so neither rehydration nor
        a truncation of the rehydrated copy can change what the turn recorded. A
        failure anywhere here falls back to the plain call: an answer over
        pointers is worse than one over evidence and far better than no answer.
        """
        from fastworkflow import answer_rehydration
        from fastworkflow.observation_offloading.state import record_event

        budget = answer_rehydration.max_bytes_from_env()
        scope = getattr(self, "continuation_scope", None)
        scope_id = getattr(scope, "scope_id", None)
        record_event(
            {
                "kind": "rehydration_started",
                "scope_id": scope_id,
                "budget_bytes": budget,
                "bytes_before": answer_rehydration.trajectory_bytes(trajectory),
            }
        )
        try:
            pairs = getattr(self, "execute_ordinal_pairs", None)
            rehydrated, report = answer_rehydration.rehydrate(
                trajectory,
                scope=scope,
                archive=getattr(self, "observation_archive", None),
                budget=budget,
                executes=pairs(trajectory) if callable(pairs) else None,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "answer rehydration skipped: %s: %s", type(error).__name__, error
            )
            record_event(
                {
                    "kind": "rehydration_failed",
                    "scope_id": scope_id,
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                }
            )
            return trajectory, None, budget, scope_id
        return rehydrated, report, budget, scope_id



    def _record_extract_finished(
        self, report, *, budget, scope_id, started, truncations_before
    ):
        """Close the rehydration record for one extract call."""
        from fastworkflow.observation_offloading.state import record_event

        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        overflowed = getattr(self, "_truncation_count", 0) > truncations_before
        if overflowed:
            record_event(
                {
                    "kind": "rehydration_overflow",
                    "scope_id": scope_id,
                    "truncations": (
                        getattr(self, "_truncation_count", 0) - truncations_before
                    ),
                    "budget_bytes": budget,
                    "bytes_after": report.bytes_after,
                }
            )
        record_event(
            {
                "kind": "rehydration_finished",
                "scope_id": scope_id,
                "extract_duration_ms": duration_ms,
                "extract_prompt_tokens": _extract_prompt_tokens(),
                "rehydration_overflow": overflowed,
                **report.as_event(),
            }
        )

    def _extract_prediction(self, trajectory, **input_args):
        """The extract call, with rehydration."""
        selected, report, budget, scope_id = self._rehydrate_for_extract(trajectory)
        if report is None:
            return self._call_with_potential_trajectory_truncation(
                self.extract, selected, **input_args
            )
        truncations_before = getattr(self, "_truncation_count", 0)
        started = time.monotonic()
        try:
            return self._call_with_potential_trajectory_truncation(
                self.extract, selected, **input_args
            )
        finally:
            self._record_extract_finished(
                report, budget=budget, scope_id=scope_id, started=started,
                truncations_before=truncations_before,
            )

    async def _async_extract_prediction(self, trajectory, **input_args):
        """``_extract_prediction`` for the async loop, same rules."""
        selected, report, budget, scope_id = self._rehydrate_for_extract(trajectory)
        if report is None:
            return await self._async_call_with_potential_trajectory_truncation(
                self.extract, selected, **input_args
            )
        truncations_before = getattr(self, "_truncation_count", 0)
        started = time.monotonic()
        try:
            return await self._async_call_with_potential_trajectory_truncation(
                self.extract, selected, **input_args
            )
        finally:
            self._record_extract_finished(
                report, budget=budget, scope_id=scope_id, started=started,
                truncations_before=truncations_before,
            )

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except litellm_exceptions.BadRequestError: 
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                self._count_truncation()
                trajectory = self.truncate_trajectory(trajectory)
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                self._count_truncation()
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                self._count_truncation()
                trajectory = self.truncate_trajectory(trajectory)

    def _count_truncation(self) -> None:
        """Bookkeeping only: the fallback's behaviour is untouched."""
        self._truncation_count = getattr(self, "_truncation_count", 0) + 1

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        # The coverage statement is a rule ABOUT the trajectory, not a step of
        # it, and it is the one key whose whole job is to be read. Dropping it as
        # "the oldest tool call information" would be a bug. It exists only on
        # the extractor's copy and only with the flag on, so with the flag off
        # this line selects exactly the keys it always did.
        # (The coverage statement and its key were removed with
        # answer_coverage in fix-4dsr; nothing writes such a key any more, so
        # every key is a step key.)
        keys = list(trajectory)
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


def _extract_prompt_tokens() -> int | None:
    """Prompt tokens of the most recent LM call, when the history holds them.

    The extract call is measured because it is the single large one at answer
    time. History can be disabled, empty, or carry no usage block, and
    none of those is an error: the measure is then simply absent.
    """
    try:
        lm = dspy.settings.lm
        history = getattr(lm, "history", None) or []
        if not history:
            return None
        usage = history[-1].get("usage") or {}
        tokens = usage.get("prompt_tokens")
        return int(tokens) if tokens else None
    except Exception:  # noqa: BLE001 - a measurement must never fail a turn
        return None


def _as_text(value: Any) -> str:
    """A span-safe rendering of a tool observation.

    Tools return whatever their author chose; span attributes are serialized
    to JSON by the store, so an exotic object would poison the write. The
    trajectory keeps the real value — only the trace gets the text.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return repr(type(value))


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adapter format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
