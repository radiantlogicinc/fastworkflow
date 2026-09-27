"""Segmented ReAct with a greedy planner skeleton bounded by the trajectory budget (Arm D)."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional

import dspy

from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading.archive import RuntimeHandleScope, RuntimeHandleArchive
from fastworkflow.observation_offloading.compact import (
    EXECUTE_TOOL_NAME,
    execute_ordinals,
    min_offload_saving_bytes_from_env,
    record_foreign_line,
    step_indexes,
)
from fastworkflow.observation_offloading.labels import (
    command_response,
    is_offload_label,
    label_alias,
    offload_label,
    offload_saving_bytes,
    owns_line,
    printed_alias,
    replacement_saves_space,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import (
    archive,
    default_scope,
    record_event,
)
from fastworkflow.utils.dspy_logger import DSPyForward
from fastworkflow.utils.react import NoSuspendedAgentStateError, fastWorkflowReAct

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERS = 25
DEFAULT_CONTINUATION_PLAN = "Continue unfinished requested work."
#: How many times the harness may force a replan before the turn stops making
#: segments. 3 forced replans is 4 segments, which at ``DEFAULT_MAX_ITERS`` is a
#: 100-step ceiling. A constant since ``ido-pyw.1``: it is a property of the
#: continuation design, not a deployment setting.
#: (2 forced replans / 3 segments / 75 steps until 2026-09-27. Raised because
#: the ido review-then-audit task needs 2-3 segments: two attempts failed at the
#: 3-segment wall, while three attempts with no practical limit finished in 2, 3
#: and 3 segments -- fix-uykd.)
MAX_FORCED_REPLANS = 3
MAX_REPLAN_CHARS = 2_000
#: The replan skeleton is the continuation planner's view of the trajectory,
#: and it gets the trajectory's budget. It is not the next segment's
#: trajectory: ``_force_replan`` hands the skeleton only to the planner, and the
#: next segment continues on the trajectory it had, with the replan artifact
#: appended. The constant is the value at the reference context
#: window; ``replan_trajectory_skeleton`` resolves the effective one per call.
REPLAN_OBSERVATION_MAX_BYTES = context_budget.REFERENCE_TRAJECTORY_MAX_BYTES


class ContinuationPlanSignature(dspy.Signature):
    """Produce a short continuation plan after a harness-enforced segment limit.

    Use the complete step skeleton to identify unfinished work and avoid
    repeating failed commands. Some observations may be inline and others may
    be metadata labels; never reconstruct missing raw observation text. Return
    at most eight short numbered steps.
    """

    user_query: str = dspy.InputField()
    trajectory_skeleton: str = dspy.InputField()
    next_steps: str = dspy.OutputField(
        desc="At most eight short numbered continuation steps; no raw data dump"
    )


def replan_trajectory_skeleton(
    trajectory: Mapping[str, Any],
    *,
    greedy_max_bytes: Optional[int] = None,
    min_offload_saving_bytes: Optional[int] = None,
    ordinal_offset: int = 0,
    executes: Optional[list[tuple[int, int]]] = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Label every eligible observation, then inline newest execute slots until the bound.

    Eligibility is the same rule compaction uses: the label must
    free at least ``min_offload_saving_bytes``. A skeleton is the one place the
    agent cannot ask for anything back before it plans, so replacing a 300 B
    fact with a 400 B pointer to it was always a bad trade; a label merely
    shorter than its observation is no longer enough.
    """

    if min_offload_saving_bytes is None:
        min_offload_saving_bytes = min_offload_saving_bytes_from_env()
    if greedy_max_bytes is None:
        greedy_max_bytes = context_budget.trajectory_max_bytes()
    if executes is None:
        executes = execute_ordinals(trajectory, ordinal_offset=ordinal_offset)
    execute_aliases = {
        f"observation_{step_index}": f"O{ordinal}"
        for step_index, ordinal in executes
    }
    skeleton: dict[str, Any] = {}
    observation_keys: list[str] = []
    for key, value in trajectory.items():
        if not key.startswith("observation_"):
            skeleton[key] = value
            continue
        observation_keys.append(key)
        suffix = key.removeprefix("observation_")
        try:
            index = int(suffix)
        except ValueError:
            index = -1
        text = str(value)
        alias = execute_aliases.get(key, f"S{index}" if index >= 0 else f"S-{suffix}")
        # On an execute step only a line of ours naming the step's ledger alias
        # is ours; a look-alike on a step that was never annotated is response.
        is_execute = key in execute_aliases
        foreign = is_execute and (
            (is_offload_label(text) or printed_alias(text) is not None)
            and not owns_line(text, alias)
        )
        if foreign:
            record_foreign_line("replan", index, text, alias, scope)
        if is_offload_label(text) and not foreign:
            alias = label_alias(text) or alias
            skeleton[key] = text
        else:
            args = trajectory.get(f"tool_args_{suffix}") or {}
            command = str(args.get("command") or "execute_workflow_query")
            # The printed handle line is presentation; label text, its authored
            # description lookup and the savings rule all use the exact response.
            original = command_response(text, alias) if is_execute else strip_alias_line(text)
            label = offload_label(alias=alias, command_name=command, response=original,
                                 description=describe_output(command, original) if describe_output else "")
            worth_labelling = (
                replacement_saves_space(original, label)
                and offload_saving_bytes(original, label) >= min_offload_saving_bytes
            )
            skeleton[key] = (label if key in execute_aliases and worth_labelling
                             else value)

    execute_keys = [key for key in observation_keys if key in execute_aliases]
    inlined_keys: list[str] = []
    measured = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    for key in reversed(execute_keys):
        current_bytes = len(str(skeleton[key]).encode("utf-8"))
        candidate_bytes = len(str(trajectory[key]).encode("utf-8"))
        if measured - current_bytes + candidate_bytes <= greedy_max_bytes:
            skeleton[key] = trajectory[key]
            measured = measured - current_bytes + candidate_bytes
            inlined_keys.append(key)
    measured_bytes = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    # A byte target cannot override the no-expansion rule or invent searchable
    # O handles for non-command observations. Keep that irreducible evidence
    # and report the overage instead of aborting a successful tool trajectory.
    # Only label text that is durably resolvable by search_memory in this turn.
    store = selected_archive
    selected_scope = scope or default_scope()
    persistence_failures: list[str] = []
    for key in execute_keys:
        shown = str(trajectory[key])
        if skeleton[key] == shown or (
            is_offload_label(shown) and owns_line(shown, execute_aliases[key])
        ):
            continue
        suffix = key.removeprefix("observation_")
        command = str((trajectory.get(f"tool_args_{suffix}") or {}).get("command") or "execute_workflow_query")
        text = command_response(shown, execute_aliases[key])
        try:
            store = store or archive()
            store.persist(selected_scope, alias=execute_aliases[key],
                          offload_order=int(execute_aliases[key][1:]), command_name=command,
                          step_index=int(suffix), text=text,
                          text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        except Exception as error:
            skeleton[key] = trajectory[key]
            persistence_failures.append(execute_aliases[key])
            record_event({"kind": "replan_offload_refused", "alias": execute_aliases[key],
                          "scope_id": selected_scope.scope_id, "error": type(error).__name__})
    measured_bytes = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    inlined_keys = [
        key for key in execute_keys
        if not (is_offload_label(str(skeleton[key]))
                and owns_line(str(skeleton[key]), execute_aliases[key]))
    ]
    inlined_aliases = [execute_aliases[key] for key in execute_keys if key in inlined_keys]
    labeled_aliases = [execute_aliases[key] for key in execute_keys if key not in inlined_keys]
    metadata = {
        "policy": "greedy_trajectory_budget",
        "inlined_aliases": inlined_aliases,
        "labeled_aliases": labeled_aliases,
        "measured_bytes": measured_bytes,
        "greedy_max_bytes": greedy_max_bytes,
        "over_target": measured_bytes > greedy_max_bytes,
        "persistence_failures": persistence_failures,
    }
    return skeleton, metadata


def _next_step_index(trajectory: Mapping[str, Any]) -> int:
    indexes = [
        int(key.removeprefix("tool_name_"))
        for key in trajectory
        if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()
    ]
    return max(indexes, default=-1) + 1


class StructuredContinuationReAct(fastWorkflowReAct):
    """Three segments of max_iters with at most two greedy, trajectory-budget replans.

    ``scope_factory`` is called once per ``forward`` so the handle scope (and
    with it the ``O{n}`` alias namespace) belongs to the turn being run, not to
    the turn the agent happened to be constructed in. The bound scope travels
    with the suspended state so an ask_user resume, in this process or another,
    keeps writing and reading the same handles.
    """

    def __init__(
        self,
        *args: Any,
        scope_factory: Optional[Callable[[], RuntimeHandleScope]] = None,
        turn_runtime: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forced_replans = 0
        self.truncated_execute_steps = 0
        #: ido-7qd. The turn's authoritative execute numbering: step index ->
        #: ``O`` ordinal. Assigned once, before the tool runs, and never
        #: reassigned or removed, so the alias a command declares and stamps
        #: under is the alias printed on its observation and archived with it.
        #: Counting a trajectory cannot be that authority: the dispatch-side
        #: mirror starts empty in a process that only imported a suspension,
        #: while the working trajectory it resumes already holds the earlier
        #: steps. Entries for truncated steps stay, because the next ordinal is
        #: one past the highest ever issued.
        self.execute_ordinal_by_step: dict[int, int] = {}
        self.continuation_scope: RuntimeHandleScope | None = None
        self.continuation_scope_id: str | None = None
        self._scope_factory = scope_factory
        self.turn_runtime = turn_runtime
        self.max_forced_replans = MAX_FORCED_REPLANS

    @property
    def total_segments(self) -> int:
        """Segments a turn may run: the first plus one per allowed forced replan."""
        return getattr(self, "max_forced_replans", MAX_FORCED_REPLANS) + 1

    def bind_scope(self) -> RuntimeHandleScope | None:
        """Resolve the scope for the turn that is starting; reclaim the one it replaces.

        Binding a different scope is this agent saying the previous turn is
        over, which makes it the earliest honest moment to release that turn's
        process-local state: its hot payloads, its archive registry and its
        context clauses. Everything released is still on disk; only residency
        goes.

        A suspension is the one thing that is not over, so a still-suspended
        agent reclaims nothing. ``forward`` clears the suspension before it gets
        here, so the guard is for a caller that binds a scope by hand.

        That release includes the raw in-flight copies of the previous turn's
        redacted evidence, because "the agent has bound the next turn" is the
        strongest statement this process can make that the previous one is
        finished -- its summary is recorded, its answer is delivered, and no
        read of it can still be part of it. A suspension keeps them, by the one
        condition below.
        """
        factory = getattr(self, "_scope_factory", None)
        if factory is None:
            return getattr(self, "continuation_scope", None)
        previous = getattr(self, "continuation_scope", None)
        scope = factory()
        runtime = getattr(self, "turn_runtime", None)
        if runtime is None:
            from fastworkflow.agent_runtime import build_turn_runtime

            runtime = build_turn_runtime(
                previous or scope,
                archive=getattr(self, "observation_archive", None),
            )
            self.turn_runtime = runtime
        if (
            previous is not None
            and previous != scope
            and getattr(self, "_suspended", None) is None
        ):
            try:
                runtime.finish_scope(previous)
            except Exception as exc:  # noqa: BLE001
                # Same shape as the sibling in
                # WorkflowExecutionContext._reclaim_offloading_scope: reclaiming
                # the PREVIOUS turn's scope must not abort the turn that is
                # starting.
                logger.debug(
                    "bind_scope: could not reclaim the previous "
                    f"offloading scope ({type(exc).__name__}: {exc})"
                )
        self.continuation_scope = scope
        self.continuation_scope_id = scope.scope_id
        runtime.bind_scope(scope)
        return scope

    def export_suspended(self) -> dict[str, Any] | None:
        data = super().export_suspended()
        if data is not None:
            data["forced_replans"] = self.forced_replans
            data["truncated_execute_steps"] = getattr(self, "truncated_execute_steps", 0)
            scope = getattr(self, "continuation_scope", None)
            if scope is not None:
                data["continuation_scope"] = asdict(scope)
        return data

    def import_suspended(self, data: dict[str, Any]) -> None:
        super().import_suspended(data)
        self.forced_replans = int(data.get("forced_replans", 0))
        self.truncated_execute_steps = int(data.get("truncated_execute_steps", 0))
        raw_scope = data.get("continuation_scope")
        if isinstance(raw_scope, Mapping):
            scope = RuntimeHandleScope(**raw_scope)
            self.continuation_scope = scope
            self.continuation_scope_id = scope.scope_id
            runtime = getattr(self, "turn_runtime", None)
            if runtime is None:
                from fastworkflow.agent_runtime import build_turn_runtime

                runtime = build_turn_runtime(
                    scope, archive=getattr(self, "observation_archive", None)
                )
                self.turn_runtime = runtime
            else:
                runtime.bind_scope(scope)
        # ido-7qd. The turn continues here, so its numbering must too. Nothing
        # new is persisted for this: the suspended trajectory carries every
        # execute step that survives, and ``truncated_execute_steps`` carries
        # the ones that do not, which together are exactly what issued the
        # aliases already printed on those observations. A fresh agent that
        # skipped this restarted at O1 and collided with its own O1.
        self.execute_ordinal_by_step = dict(
            execute_ordinals(
                self._suspended["trajectory"],
                ordinal_offset=self.truncated_execute_steps,
            )
        )

    def _note_step(self, idx: int, tool_name: str) -> None:
        """Number an execute step before it is dispatched, once and for good."""
        if str(tool_name or "") != EXECUTE_TOOL_NAME:
            return
        # getattr: a loop can be driven on an instance built via __new__ (test
        # helpers), the same reason _on_step_complete is read that way.
        ledger = getattr(self, "execute_ordinal_by_step", None)
        if ledger is None:
            ledger = self.execute_ordinal_by_step = {}
        if idx not in ledger:
            ledger[idx] = self.next_execute_ordinal()

    def next_execute_ordinal(self) -> int:
        """One past the highest ordinal this turn has ever issued."""
        ledger = getattr(self, "execute_ordinal_by_step", None) or {}
        return max(ledger.values(), default=0) + 1

    def execute_ordinal_pairs(
        self, trajectory: Mapping[str, Any]
    ) -> list[tuple[int, int]]:
        """The ledger's ``(step_index, ordinal)`` pairs for the steps *trajectory* still has.

        Every consumer of execute numbering -- the printed alias line, the
        archive key, the offload label, the replan skeleton -- reads the turn's
        numbering from here, so none of them can disagree with the alias a
        command already declared under.
        """
        ledger = getattr(self, "execute_ordinal_by_step", None) or {}
        present = [
            index
            for index in step_indexes(trajectory)
            if str(trajectory.get(f"tool_name_{index}") or "") == EXECUTE_TOOL_NAME
        ]
        if any(index not in ledger for index in present):
            # This agent never numbered these steps -- a trajectory handed in
            # from outside the loop, a rehydrated skeleton. It has no authority
            # over them, so the rule stands in for the whole trajectory rather
            # than half of it.
            return execute_ordinals(
                trajectory,
                ordinal_offset=int(getattr(self, "truncated_execute_steps", 0) or 0),
            )
        return [(index, ledger[index]) for index in present]

    def truncate_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        """Drop the oldest surviving step, remembering how many executes are gone.

        The base class pops the first four keys in insertion order. Here steps
        are removed by index so a ``replan_N`` artifact is never mistaken for a
        step key, and execute steps are counted so ``execute_ordinals`` keeps
        assigning the aliases the surviving observations were persisted under.
        """
        indexes = step_indexes(trajectory)
        if not indexes:
            return super().truncate_trajectory(trajectory)
        oldest = indexes[0]
        if str(trajectory.get(f"tool_name_{oldest}") or "") == "execute_workflow_query":
            self.truncated_execute_steps = getattr(self, "truncated_execute_steps", 0) + 1
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            trajectory.pop(f"{prefix}_{oldest}", None)
        return trajectory

    def _finish_prediction(
        self,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
    ) -> dspy.Prediction:
        # The one extract call of a segmented turn: agent-selected finish and
        # the replan wall both end here. It goes through _extract_prediction so
        # answer-time rehydration (ido-8ps.18) reaches the configuration that
        # actually runs; with the flag off it is the identical call it was.
        extract = self._extract_prediction(trajectory, **input_args)
        return dspy.Prediction(
            trajectory=trajectory,
            exhausted=self._exhausted_last_run,
            **extract,
        )

    def _force_replan(
        self,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
    ) -> None:
        completed_segment = self.forced_replans + 1
        next_segment = completed_segment + 1
        skeleton, observation_metadata = replan_trajectory_skeleton(
            trajectory,
            executes=self.execute_ordinal_pairs(trajectory),
            ordinal_offset=getattr(self, "truncated_execute_steps", 0),
            scope=getattr(self, "continuation_scope", None),
            selected_archive=getattr(self, "observation_archive", None),
            describe_output=getattr(self, "describe_output", None),
        )
        trigger = (
            f"segment {completed_segment} reached the {self.max_iters}-iteration "
            "limit without agent-selected finish"
        )
        host = tracing.current_host()
        # Same key set as build_query_with_next_steps: fw.planner.replan has one
        # SpanContract, so every producer writes {model, replan_trigger, plan}.
        # Segment bookkeeping and the injected artifact go to record_event below.
        span = tracing.start_span(
            host,
            tracing.SPAN_PLANNER_REPLAN,
            kind=tracing.KIND_LLM,
            attributes={
                "model": getattr(dspy.settings.lm, "model", None),
                "replan_trigger": "structured_continuation_segment_limit",
            },
        )
        # The planner is advisory. Stock ReAct falls back to extract() at the
        # iteration limit, so a planner hiccup here (rate limit, timeout, parse
        # failure) degrades to the default continuation plan rather than
        # aborting a turn that has already done the work. Non-Exception
        # BaseExceptions (KeyboardInterrupt, suspension) still propagate.
        planner_error: str | None = None
        try:
            prediction = dspy.Predict(ContinuationPlanSignature)(
                user_query=str(input_args.get("user_query") or ""),
                trajectory_skeleton=json.dumps(
                    skeleton, ensure_ascii=False, default=str
                ),
            )
            plan = str(prediction.next_steps or "").strip()[:MAX_REPLAN_CHARS]
        except Exception as error:  # noqa: BLE001
            planner_error = f"{type(error).__name__}: {error}"[:300]
            logger.warning(
                "continuation planner failed at segment %d; using the default plan: %s",
                completed_segment, planner_error,
            )
            plan = ""
            tracing.end_span(
                host, span, status=tracing.STATUS_ERROR, attributes={"plan": plan}
            )
        except BaseException:
            tracing.end_span(host, span, status=tracing.STATUS_ERROR)
            raise
        else:
            tracing.end_span(host, span, attributes={"plan": plan})
        artifact = (
            f"HARNESS REPLAN — segment {next_segment} of {self.total_segments}. "
            f"Reason: {trigger}.\n{plan or DEFAULT_CONTINUATION_PLAN}"
        )
        artifact_key = f"replan_{completed_segment}"
        trajectory[artifact_key] = artifact
        self.current_trajectory[artifact_key] = artifact
        self.forced_replans += 1
        self.iteration_counter = 0
        record_event(
            {
                "kind": "forced_replan",
                "scope_id": getattr(self, "continuation_scope_id", None),
                "completed_segment": completed_segment,
                "next_segment": next_segment,
                "max_segments": self.total_segments,
                "reason": trigger,
                "plan": plan,
                "planner_error": planner_error,
                "artifact": artifact,
                "skeleton_steps": len(
                    [key for key in skeleton if key.startswith("tool_name_")]
                ),
                **observation_metadata,
            }
        )

    def _run_segments(
        self,
        trajectory: dict[str, Any],
        idx: int,
        input_args: dict[str, Any],
        max_iters: int,
    ) -> dspy.Prediction:
        while True:
            suspended = self._run_loop(trajectory, idx, input_args, max_iters, 0)
            if suspended is not None:
                return suspended
            if not self._exhausted_last_run:
                return self._finish_prediction(trajectory, input_args)
            if self.forced_replans >= getattr(self, "max_forced_replans", MAX_FORCED_REPLANS):
                total_segments = self.total_segments
                record_event(
                    {
                        "kind": "forced_replan_wall",
                        "scope_id": getattr(self, "continuation_scope_id", None),
                        "completed_segment": total_segments,
                        "max_segments": total_segments,
                        "reason": (
                            f"segment {total_segments} reached the "
                            f"{max_iters}-iteration limit"
                        ),
                    }
                )
                return self._finish_prediction(trajectory, input_args)
            self._force_replan(trajectory, input_args)
            idx = _next_step_index(trajectory)

    @DSPyForward.intercept
    def forward(self, **input_args: Any) -> dspy.Prediction:
        self.inputs = input_args
        self.clear_suspension()
        self.current_trajectory = {}
        self.iteration_counter = 0
        self.forced_replans = 0
        self.truncated_execute_steps = 0
        self.execute_ordinal_by_step = {}
        # ido-8ps.27: one roster nudge per TURN, and a turn here is a forward()
        # across all of its segments, not a segment. (fix-4dsr: the same cap now
        # applies to the finish-check note that replaced the roster nudge.)
        self._finish_notes_fired = 0
        self.bind_scope()
        trajectory: dict[str, Any] = {}
        max_iters = int(input_args.pop("max_iters", self.max_iters))
        return self._run_segments(trajectory, 0, input_args, max_iters)

    def resume(self, observation: str) -> dspy.Prediction:
        if self._suspended is None:
            raise NoSuspendedAgentStateError("No suspended ReAct state to resume")
        stash = self._suspended
        trajectory = stash["trajectory"]
        idx = int(stash["idx"])
        input_args = stash["input_args"]
        max_iters = int(stash["max_iters"])
        self.inputs = input_args
        trajectory[f"observation_{idx}"] = observation
        self.current_trajectory[f"observation_{idx}"] = observation
        self._suspended = None
        self.iteration_counter += 1
        return self._run_segments(trajectory, idx + 1, input_args, max_iters)
