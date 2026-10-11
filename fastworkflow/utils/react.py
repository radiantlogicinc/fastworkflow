import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError
from litellm import exceptions as litellm_exceptions

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from dspy.utils.exceptions import AdapterParseError
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

from fastworkflow import tracing
from fastworkflow.utils.dspy_logger import DSPyForward

logger = logging.getLogger(__name__)

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
    # Called after each completed tool step (not on suspension); class default for agents built via __new__.
    on_step_end: Callable[[], None] | None = None

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

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self._instructions_head = instr
        self._disabled_tools: frozenset[str] = frozenset()
        self.react = dspy.Predict(self._react_signature())
        self.extract = dspy.ChainOfThought(fallback_signature)

        self.inputs = {}
        self.trajectory: dict[str, Any] = {}
        # Caller-owned state for one executor run; reset in forward().
        self.run_state: dict[str, Any] = {}
        self._on_step_complete = on_step_complete
        self._suspended: dict[str, Any] | None = None
        # True when the most recent _run_loop ended because max_iters was
        # reached without the agent selecting the `finish` tool.
        self._exhausted_last_run = False
        # How many times the context-window fallback has fired in this process.
        # Only read as a delta around one call (to tell an extract
        # that overflowed from one that did not).
        self._truncation_count = 0
        # Steps the fallback has dropped from the model's view of self.trajectory
        # in the current turn. The canonical trajectory is never trimmed.
        self._dropped_steps = 0

    _disabled_tools: frozenset[str] = frozenset()  # class default: agents built via __new__ (tests) have none

    @property
    def disabled_tools(self) -> frozenset[str]:
        """Tools the agent may not use now: not listed to the model, refused if called.

        Assign a collection to change it; the ReAct signature is rebuilt to match.
        """
        return self._disabled_tools

    @disabled_tools.setter
    def disabled_tools(self, names) -> None:
        self._disabled_tools = frozenset(names)
        self.react.signature = self._react_signature()

    def _react_signature(self):
        """The ReAct signature, listing the tools that are not disabled."""
        enabled = {name: tool for name, tool in self.tools.items() if name not in self._disabled_tools}
        instr = [
            *self._instructions_head,
            *(f"({idx + 1}) {tool}" for idx, tool in enumerate(enabled.values())),
            "When providing `next_tool_args`, the value inside the field must be in JSON format",
        ]
        # Build the ReAct signature with trajectory input.
        # available_commands is injected into system message by CommandsSystemPreludeAdapter
        # (see fastworkflow/utils/chat_adapter.py) and is NOT included in the trajectory
        # formatting to avoid token bloat across iterations.
        return (
            dspy.Signature({**self.signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(enabled.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

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
            "dropped_steps": self._dropped_steps,
            "disabled_tools": sorted(self._disabled_tools),
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
        self._dropped_steps = data.get("dropped_steps", 0)
        self.disabled_tools = data.get("disabled_tools", [])

    def planner_view(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The request and trajectory a mid-turn replan plans from.

        Each comes from this agent when set, else from the suspended stash: a
        process that only imported a suspension has neither until ``resume()``.
        """
        inputs = self.inputs
        trajectory = self.trajectory
        stash = self._suspended
        if stash is not None:
            if not inputs:
                inputs = dict(stash["input_args"])
            if not trajectory:
                trajectory = dict(stash["trajectory"])
        return inputs, trajectory

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def prompt_overhead_bytes(self, input_args: dict[str, Any]) -> int:
        """UTF-8 bytes of the executor prompt that is not the trajectory.

        The messages ``self.react`` sends for ``input_args`` with an empty
        trajectory: instructions, inputs, command list and demos.
        """
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        messages = adapter.format(
            self.react.signature,
            demos=getattr(self.react, "demos", []),
            inputs={**input_args, "trajectory": ""},
        )
        return sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages)

    @DSPyForward.intercept
    def forward(self, **input_args):
        self.inputs = input_args
        self.clear_suspension()

        self.trajectory = {}
        self.run_state = {}
        self.iteration_counter = 0
        self._dropped_steps = 0

        max_iters = input_args.pop("max_iters", self.max_iters)
        idx = input_args.pop("first_step", 0)
        exception_count = 0

        suspended = self._run_loop(
            self.trajectory, idx, input_args, max_iters, exception_count
        )
        if suspended is not None:
            return suspended

        extract = self._extract_prediction(self.trajectory, **input_args)
        return dspy.Prediction(
            trajectory=self.trajectory, exhausted=self._exhausted_last_run, **extract
        )

    def resume(self, observation: str):
        """Resume a suspended run after the user answered an ask_user clarification."""
        if self._suspended is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )

        stash = self._suspended
        self.trajectory = stash["trajectory"]
        idx = stash["idx"]
        input_args = stash["input_args"]
        max_iters = stash["max_iters"]

        # Keep self.inputs pointing at the active run's arg dict so any mid-run refresh
        # (e.g. available_commands re-scoping after a context switch) mutates the same
        # dict this loop unpacks on each step.
        self.inputs = input_args

        self.trajectory[f"observation_{idx}"] = observation
        idx += 1
        self.iteration_counter += 1
        self._suspended = None

        suspended = self._run_loop(self.trajectory, idx, input_args, max_iters, 0)
        if suspended is not None:
            return suspended

        extract = self._extract_prediction(self.trajectory, **input_args)
        return dspy.Prediction(
            trajectory=self.trajectory, exhausted=self._exhausted_last_run, **extract
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
            repaired_tool_name = None
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
                if pred is None:
                    raise ValueError("Tool returned is None")
            # A reply no adapter could parse, or a call that timed out, is a bad
            # step, not a failed turn: it is recovered from like an invalid tool
            # choice, so one bad completion cannot throw away every step taken.
            except (ValueError, AdapterParseError, litellm_exceptions.Timeout) as err:
                repaired = _command_named_as_tool(
                    err, self.tools, input_args.get("available_commands")
                )
                if repaired is not None:
                    repaired_tool_name = repaired.pop("repaired_tool_name")
                    pred = dspy.Prediction(**repaired)
                else:
                    invalid_tool_obs = (
                        f"Agent failed to select a valid tool: {_fmt_exc(err)}"
                    )
                    trajectory[f"observation_{idx}"] = invalid_tool_obs
                    idx += 1
                    recovery_thought = (
                        "To execute a command, I should use one of the available tools"
                    )
                    recovery_obs = (
                        "Use the appropriate tool with proper arguments (correctly formatted)"
                    )
                    trajectory[f"thought_{idx}"] = recovery_thought
                    trajectory[f"observation_{idx}"] = recovery_obs
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
                # Anything else from the reasoning call — provider errors,
                # control signals. The caller's retry loop
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
            if repaired_tool_name is not None:
                step_attributes["repaired_tool_name"] = repaired_tool_name

            try:
                if pred.next_tool_name in self.disabled_tools:
                    observation = f"{pred.next_tool_name} is not available now."
                else:
                    observation = self.tools[pred.next_tool_name](**pred.next_tool_args)
                trajectory[f"observation_{idx}"] = observation
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

            tracing.end_span(
                host, step_span, status=step_status, attributes=step_attributes
            )
            if self.on_step_end is not None:
                self.on_step_end()

            # Step-completion callback for distillation: lets external code inspect
            # each completed step and stop execution early (e.g. on trajectory
            # divergence). Placed AFTER the AskUserSuspend catch so it can never
            # swallow a suspension, and it does not touch _suspended state.
            # getattr guard: resume() may run on an instance built via __new__
            # (test helpers) that never set this attribute.
            on_step_complete = getattr(self, "_on_step_complete", None)
            if on_step_complete and not on_step_complete(idx, trajectory):
                break

            if pred.next_tool_name == "finish":
                break

            idx += 1
            # Still counted: execute_workflow_query reads iteration_counter <= 0
            # as "this command came from the user".
            self.iteration_counter += 1
            # The cap is a hand-off signal: the caller summarises progress and
            # runs a fresh executor to continue the same sub-task.
            if self.iteration_counter >= max_iters:
                logger.warning("Max iterations reached")
                self._exhausted_last_run = True
                break

        return None

    def _rehydrate_for_extract(self, trajectory):
        """``(trajectory_for_the_extractor, report, budget, scope)``.

        It returns a COPY in which offload labels carry the stored evidence
        behind them (see ``fastworkflow.answer_rehydration``). The loop's own
        trajectory is then never the object passed on, so neither rehydration nor
        a truncation of the rehydrated copy can change what the turn recorded. A
        failure anywhere here falls back to the plain call: an answer over
        pointers is worse than one over evidence and far better than no answer.
        """
        from fastworkflow import answer_rehydration
        from fastworkflow.observation_offloading.state import (
            current_scope,
            durable_archive,
            record_event,
        )

        budget = answer_rehydration.max_bytes_from_env()
        scope = current_scope()
        archive = durable_archive()
        record_event(
            {
                "kind": "rehydration_started",
                "budget_bytes": budget,
                "bytes_before": answer_rehydration.trajectory_bytes(trajectory),
            },
            scope=scope, store=archive,
        )
        try:
            rehydrated, report = answer_rehydration.rehydrate(
                trajectory,
                scope=scope,
                archive=archive,
                budget=budget,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "answer rehydration skipped: %s: %s", type(error).__name__, error
            )
            record_event(
                {
                    "kind": "rehydration_failed",
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                },
                scope=scope, store=archive,
            )
            return trajectory, None, budget, scope
        return rehydrated, report, budget, scope



    def _record_extract_finished(
        self, report, *, budget, scope, started, truncations_before
    ):
        """Close the rehydration record for one extract call."""
        from fastworkflow.observation_offloading.state import durable_archive, record_event

        duration_ms = round((time.monotonic() - started) * 1000.0, 3)
        overflowed = getattr(self, "_truncation_count", 0) > truncations_before
        if overflowed:
            record_event(
                {
                    "kind": "rehydration_overflow",
                    "truncations": (
                        getattr(self, "_truncation_count", 0) - truncations_before
                    ),
                    "budget_bytes": budget,
                    "bytes_after": report.bytes_after,
                },
                scope=scope, store=durable_archive(),
            )
        record_event(
            {
                "kind": "rehydration_finished",
                "extract_duration_ms": duration_ms,
                "extract_prompt_tokens": _extract_prompt_tokens(),
                "rehydration_overflow": overflowed,
                **report.as_event(),
            },
            scope=scope, store=durable_archive(),
        )

    def _extract_prediction(self, trajectory, **input_args):
        """The extract call, with rehydration."""
        selected, report, budget, scope = self._rehydrate_for_extract(trajectory)
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
                report, budget=budget, scope=scope, started=started,
                truncations_before=truncations_before,
            )

    def _model_view(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        """A copy of ``trajectory`` without the oldest steps the fallback dropped.

        Steps are dropped by step index, not by key count: a parse-error recovery
        writes an ``observation_<n>`` with no thought, then a full thought/observation
        pair, so steps do not all have 4 keys.
        """
        dropped = set(_step_indexes(trajectory)[: self._dropped_steps])
        return {k: v for k, v in trajectory.items() if _step_index(k) not in dropped}

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(self._model_view(trajectory)),
                )
            except (litellm_exceptions.BadRequestError, ContextWindowExceededError):
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                self._count_truncation()
                self.truncate_trajectory(trajectory)

    def _count_truncation(self) -> None:
        """Bookkeeping only: the fallback's behaviour is untouched."""
        self._truncation_count = getattr(self, "_truncation_count", 0) + 1

    def truncate_trajectory(self, trajectory):
        """Drop the oldest step from what the model is shown. ``trajectory`` is never changed.

        Each call adds one step to ``_dropped_steps``; ``_model_view`` applies
        that to a copy when the trajectory is formatted.
        """
        if len(_step_indexes(trajectory)) - self._dropped_steps < 1:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )
        self._dropped_steps += 1
        return trajectory


_STEP_KEY = re.compile(r"^(?:thought|tool_name|tool_args|observation)_(\d+)$")


def _step_index(key: str) -> int | None:
    """The step index a trajectory key belongs to, or None for an unrecognised key."""
    match = _STEP_KEY.match(key)
    return int(match.group(1)) if match else None


def _step_indexes(trajectory: dict[str, Any]) -> list[int]:
    """Distinct step indexes in trajectory order, oldest first."""
    return list(dict.fromkeys(
        index for index in (_step_index(k) for k in trajectory) if index is not None
    ))


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


_FIELD_MARKER = re.compile(r"\[\[ ## (\w+) ## \]\]")
_COMMAND_TOKEN = re.compile(r"[A-Za-z_][\w/]*")
_COMMAND_TOOL = "execute_workflow_query"
_ARG_KEY = re.compile(r"[A-Za-z_]\w*")
_TAG_LIKE = re.compile(r"</?[^<>]+>")


def _lm_responses(err: BaseException):
    """Raw LM replies carried by the adapter parse errors chained under *err*."""
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "lm_response", None)
        if isinstance(response, str):
            yield response
        current = current.__cause__ or current.__context__


def _reply_fields(response: str) -> dict[str, Any]:
    """The output fields of a chat-format or JSON-format reply, unvalidated."""
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        parts = _FIELD_MARKER.split(text)
        return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return parsed if isinstance(parsed, dict) else {}


def _command_named_as_tool(
    err: BaseException, tools: dict[str, Any], available_commands: Any
) -> dict[str, Any] | None:
    """The step the agent meant when it named a workflow command as its tool.

    Small models write ``next_tool_name: open_item_explorer`` instead of calling
    ``execute_workflow_query`` with that command. When the rejected name is a
    command listed for the current context and its args are a JSON object
    whose values hold no tag-like text, return the equivalent
    ``execute_workflow_query`` step; otherwise None.
    """
    if _COMMAND_TOOL not in tools or not isinstance(available_commands, str):
        return None
    for response in _lm_responses(err):
        fields = _reply_fields(response)
        name = fields.get("next_tool_name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if name in tools or not _COMMAND_TOKEN.fullmatch(name):
            continue
        if not re.search(rf"^- {re.escape(name)}\s*$", available_commands, re.MULTILINE):
            continue
        args = fields.get("next_tool_args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                continue
        if not isinstance(args, dict):
            continue
        values = {
            key: value if isinstance(value, str) else json.dumps(value)
            for key, value in args.items()
        }
        # The workflow reads parameters back with tag regexes and no unescaping,
        # so a value holding tag-like text would arrive cut or altered.
        if not all(
            isinstance(key, str) and _ARG_KEY.fullmatch(key) and not _TAG_LIKE.search(value)
            for key, value in values.items()
        ):
            continue
        command = " ".join(
            [name] + [f"<{key}>{value}</{key}>" for key, value in values.items()]
        )
        return {
            "next_thought": str(fields.get("next_thought", "")),
            "next_tool_name": _COMMAND_TOOL,
            "next_tool_args": {"command": command},
            "repaired_tool_name": name,
        }
    return None


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
