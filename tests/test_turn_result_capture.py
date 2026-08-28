"""TurnResult capture on WorkflowExecutionContext (v2.21, bead fix-yy1.5).

Mirrors the fixtures/patterns of tests/test_execution_context_agent.py (real
todo_list_workflow, MagicMock only at the agent/LLM boundary) and
tests/test_execution_context_concurrency.py (classmethod fake for
CommandExecutor.invoke_command on the deterministic path).
"""

from __future__ import annotations

import re
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow import (
    FW_ARTIFACT_REF_KEY,
    CommandOutput,
    CommandResponse,
    TurnOutput,
    TurnResult,
    TurnStatus,
    mint_turn_key,
)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.turn import (
    collect_artifact_responses,
    merge_artifact_responses_into,
    validate_artifacts_serializable,
    warn_on_unserializable_artifacts,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

TURN_KEY_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{12}$")


# ----------------------------------------------------------------------
# Fixtures and helpers (mirroring test_execution_context_agent.py)
# ----------------------------------------------------------------------


@pytest.fixture
def todo_workflow_path() -> str:
    return str(
        Path(__file__).parent.joinpath("todo_list_workflow").resolve()
    )


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def _set_agents(ctx, agent, clarification_agent=None):
    """Set the workflow tool agent plus a non-None clarification agent (parity
    with WorkflowExecutionContext._initialize_agent_functionality)."""
    ctx._workflow_tool_agent = agent
    ctx._intent_clarification_agent = (
        clarification_agent if clarification_agent is not None else MagicMock()
    )


def _make_agent_ctx(todo_workflow_path, monkeypatch):
    """Agent-mode context against the real todo workflow; only the LLM/agent
    boundary is faked (same seams as test_execution_context_agent.py)."""
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"turncap-agent-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, with_agent_inputs_and_trajectory=False,
        planning_insights=None, planner_lm=None, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do",
        lambda session: "commands",
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx, wf


def _make_assistant_ctx(todo_workflow_path, monkeypatch, response_text="ok"):
    """Deterministic/assistant-mode context; CommandExecutor.invoke_command is
    faked at the NLU boundary (same pattern as test_execution_context_concurrency.py)
    because the test workflow ships no trained intent models."""
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"turncap-assist-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False)
    ctx.bind_app_workflow(wf)

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=
                fastworkflow.CommandResponse(response=f"{response_text}:{command}"),
        )

    monkeypatch.setattr(
        CommandExecutor, "invoke_command", classmethod(fake_invoke)
    )
    return ctx, wf


def _patch_invoke_with(monkeypatch, fn):
    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fn))


# ----------------------------------------------------------------------
# 1-2: Exports and turn-key grammar
# ----------------------------------------------------------------------


class TestExportsAndTurnKey:
    def test_turn_symbols_exported_from_package_root(self):
        assert TurnStatus.COMPLETED.value == "completed"
        assert TurnStatus.AWAITING_USER.value == "awaiting_user"
        assert FW_ARTIFACT_REF_KEY == "__fw_artifact_ref__"
        assert callable(mint_turn_key)
        assert issubclass(TurnResult, object)
        # the same objects are reachable via the turn module
        import fastworkflow.turn as turn_mod

        assert turn_mod.TurnResult is TurnResult
        assert turn_mod.mint_turn_key is mint_turn_key
        assert callable(turn_mod.validate_artifacts_serializable)
        assert callable(turn_mod.warn_on_unserializable_artifacts)

    def test_mint_turn_key_grammar(self):
        key = mint_turn_key()
        assert TURN_KEY_RE.match(key), key
        # keys are unique across calls
        assert mint_turn_key() != key

    def test_mint_turn_key_deterministic_with_injected_seams(self):
        now = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
        key1 = mint_turn_key(now=now, uuid_hex="0123456789ab")
        key2 = mint_turn_key(now=now, uuid_hex="0123456789ab")
        assert key1 == "20260102T030405.123456Z-0123456789ab"
        assert key1 == key2
        assert TURN_KEY_RE.match(key1)


# ----------------------------------------------------------------------
# 3-4: CommandOutput forward-compat constructor and ask_user helpers
# ----------------------------------------------------------------------


class TestCommandOutputShape:
    def test_singular_command_response_is_canonical(self):
        r = CommandResponse(response="hello")
        out = CommandOutput(command_response=r)
        assert out.command_response is r
        assert out.success is True
        dumped = out.model_dump()
        assert "command_response" in dumped
        assert "command_responses" not in dumped

    def test_legacy_list_keyword_is_rejected(self):
        r = CommandResponse(response="legacy")
        with pytest.raises(ValueError, match="command_response=CommandResponse"):
            CommandOutput(command_responses=[r])

    def test_ask_user_helpers_round_trip(self):
        entry = CommandOutput(
            command_name="ask_user",
            command_parameters="Which task did you mean?",
            command_response=CommandResponse(response="", success=False),
        )
        assert entry.is_ask_user is True
        assert entry.question == "Which task did you mean?"
        assert entry.user_reply == ""  # unanswered
        assert entry.success is False

        entry.command_response.response = "the urgent one"
        entry.command_response.success = True
        assert entry.user_reply == "the urgent one"
        assert entry.success is True

    def test_helpers_false_or_none_on_normal_outputs(self):
        out = CommandOutput(
            command_name="list_todo_lists",
            command_response=CommandResponse(response="3 lists"),
        )
        assert out.is_ask_user is False
        assert out.question is None
        assert out.user_reply is None


# ----------------------------------------------------------------------
# 5-6, 9: Deterministic path (process_turn / process_message / A30 reset)
# ----------------------------------------------------------------------


class TestDeterministicTurn:
    def test_process_turn_completed_with_answer_aliasing(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        result = ctx.process_turn("list my todo lists")

        assert isinstance(result, TurnOutput)
        assert result.status == TurnStatus.COMPLETED
        assert result.success is True
        assert TURN_KEY_RE.match(result.turn_key), result.turn_key
        assert result.command_outputs

        captured = result.command_outputs[-1]
        # answer is plain text: the last captured output's first response text
        assert isinstance(result.answer, str)
        assert result.answer == captured.command_response.response
        # per-command timing is retained on command_outputs (nested provenance)
        assert captured.started_at is not None
        assert captured.duration_ms is not None
        assert captured.duration_ms >= 0

    def test_process_message_is_removed(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        # The deprecated bridge is gone at 3.0.0; callers use process_turn() (or
        # the internal _execute_message() when a raw CommandOutput is needed).
        assert not hasattr(ctx, "process_message")

        output = ctx._execute_message("list my todo lists")
        assert isinstance(output, fastworkflow.CommandOutput)
        assert not isinstance(output, TurnResult)
        assert not isinstance(output, TurnOutput)
        assert output.success
        assert "list my todo lists" in output.command_response.response

    def test_failed_deterministic_command_marks_turn_failed(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        def failing_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command.split()[0] if command else "",
                command_response=
                    fastworkflow.CommandResponse(
                        response="could not do that", success=False
                    ),
            )

        _patch_invoke_with(monkeypatch, failing_invoke)

        result = ctx.process_turn("delete a missing list")

        # the turn ran to completion; success is derived from the command's
        # success code (no turn-level failure_reason for a command failure)
        assert result.status == TurnStatus.COMPLETED
        assert result.failure_reason is None
        assert result.success is False
        # the per-command failure is visible on command_outputs
        assert result.command_outputs[-1].success is False

    def test_new_turn_resets_accumulator_a30(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        first = ctx.process_turn("first message")
        second = ctx.process_turn("second message")

        assert first.turn_key != second.turn_key
        # command_outputs do not accumulate across completed turns
        assert len(first.command_outputs) == 1
        assert len(second.command_outputs) == 1
        assert second.command_outputs[0] is not first.command_outputs[0]


# ----------------------------------------------------------------------
# 7: Agent path (completed / exhausted) with a fake agent
# ----------------------------------------------------------------------


class TestAgentTurn:
    def test_agent_turn_completed(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        mock_agent = MagicMock()
        mock_agent.return_value = SimpleNamespace(
            final_answer="Agent finished successfully"
        )
        _set_agents(ctx, mock_agent)

        result = ctx.process_turn("list my tasks")

        assert isinstance(result, TurnOutput)
        assert result.status == TurnStatus.COMPLETED
        assert result.success is True
        assert result.failure_reason is None
        assert "Agent finished successfully" in result.answer
        assert TURN_KEY_RE.match(result.turn_key)
        mock_agent.assert_called_once()

    def test_agent_turn_success_false_when_a_command_failed(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        # The agent always phrases its final answer as success; the turn must
        # still report success=False because an underlying command failed.
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        from fastworkflow.workflow_agent import _execute_workflow_query

        def fake_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command,
                command_response=
                    fastworkflow.CommandResponse(
                        response=f"ran {command}",
                        success=command != "mark_completed",  # this one fails
                    ),
            )

        _patch_invoke_with(monkeypatch, fake_invoke)

        def fake_forward(**kwargs):
            _execute_workflow_query("get_todo_list", ctx)  # succeeds
            _execute_workflow_query("mark_completed", ctx)  # fails
            return SimpleNamespace(final_answer="All set! I marked it complete.")

        mock_agent = MagicMock(side_effect=fake_forward)
        _set_agents(ctx, mock_agent)

        result = ctx.process_turn("mark my task done")

        assert result.status == TurnStatus.COMPLETED
        assert result.failure_reason is None  # not a turn-level failure reason
        # ...but the agent masked a command failure in its prose; success catches it
        assert "All set" in result.answer
        assert result.success is False
        assert [o.command_name for o in result.command_outputs] == [
            "get_todo_list",
            "mark_completed",
        ]
        assert result.command_outputs[0].success is True
        assert result.command_outputs[1].success is False

    def test_agent_turn_exhausted_marks_failure(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        mock_agent = MagicMock()
        mock_agent.return_value = SimpleNamespace(
            final_answer="Ran out of iterations", exhausted=True
        )
        _set_agents(ctx, mock_agent)

        result = ctx.process_turn("do an impossible thing")

        # the turn failed to complete: status carries the failure, failure_reason
        # elaborates it (orthogonal to success)
        assert result.status == TurnStatus.FAILED
        assert isinstance(result.answer, str)
        assert result.failure_reason == "max_iters_exhausted"
        # success is purely command-based: this mock ran no commands, so no
        # command failed -> success is True even though the turn FAILED to complete
        assert result.success is True
        dumped = result.model_dump()
        assert "success" in dumped  # computed field is serialized
        assert dumped["success"] is True


# ----------------------------------------------------------------------
# 8 + 9 (resume half): Suspension/resume turn capture
# ----------------------------------------------------------------------


class TestSuspendResumeTurnCapture:
    def test_suspend_resume_same_turn_key_and_ordering(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        from fastworkflow.workflow_agent import _execute_workflow_query

        def fake_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command,
                command_response=
                    fastworkflow.CommandResponse(response=f"ran {command}"),
            )

        _patch_invoke_with(monkeypatch, fake_invoke)

        # The fake agent does one real tool call on each side of the suspension.
        def fake_forward(**kwargs):
            _execute_workflow_query("get_todo_list", ctx)
            return SimpleNamespace(suspended=True, clarification="Which task?")

        def fake_resume(observation):
            _execute_workflow_query("mark_completed", ctx)
            return SimpleNamespace(final_answer="All done")

        mock_agent = MagicMock(side_effect=fake_forward)
        mock_agent.resume = MagicMock(side_effect=fake_resume)
        _set_agents(ctx, mock_agent)

        first = ctx.process_turn("list my tasks")

        assert first.status == TurnStatus.AWAITING_USER
        assert first.success is False  # awaiting_user is not a success
        assert ctx.awaiting_user
        # the awaiting-user signal is the status; answer is the clarification text
        assert isinstance(first.answer, str)
        assert len(first.command_outputs) == 2
        ask_entry = first.command_outputs[1]
        assert ask_entry.is_ask_user
        assert ask_entry.success is False  # unanswered
        assert ask_entry.question == "Which task?"
        assert first.command_outputs[0].command_name == "get_todo_list"

        second = ctx.process_turn("the urgent one")

        # Resume continues the SAME logical turn — no reset [A30.2]
        assert second.turn_key == first.turn_key
        assert second.status == TurnStatus.COMPLETED
        assert not ctx.awaiting_user
        assert "All done" in second.answer

        # ordering preserved: pre-suspension tool call, ask_user, post-suspension
        assert [o.command_name for o in second.command_outputs] == [
            "get_todo_list",
            "ask_user",
            "mark_completed",
        ]

        filled = second.command_outputs[1]
        assert filled.is_ask_user
        assert filled.user_reply == "the urgent one"
        assert filled.success is True
        # per-command timing (incl. ask_user think-time) is retained on the
        # command output, visible via command_outputs.
        assert filled.duration_ms is not None
        assert filled.duration_ms >= 0
        mock_agent.assert_called_once()
        mock_agent.resume.assert_called_once()


# ----------------------------------------------------------------------
# 10: Failure capture in _execute_workflow_query
# ----------------------------------------------------------------------


class TestFailureCapture:
    def test_failed_tool_call_captured_and_exception_propagates(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        from fastworkflow.workflow_agent import _execute_workflow_query

        def exploding_invoke(cls, session, command: str):
            raise RuntimeError("boom: storage offline")

        _patch_invoke_with(monkeypatch, exploding_invoke)

        ctx._begin_turn("trigger failure")
        with pytest.raises(RuntimeError, match="storage offline"):
            _execute_workflow_query("delete_todo_list", ctx)

        assert len(ctx._turn_outputs) == 1
        entry = ctx._turn_outputs[0]
        assert entry.success is False
        artifacts = entry.command_response.artifacts
        assert artifacts["error_type"] == "RuntimeError"
        assert "storage offline" in artifacts["error_message"]
        assert "RuntimeError" in artifacts["traceback"]
        assert entry.started_at is not None
        assert entry.duration_ms is not None


# ----------------------------------------------------------------------
# 11: command_outputs_with_artifacts property
# ----------------------------------------------------------------------


class TestCommandOutputsWithArtifacts:
    def test_only_artifact_bearing_outputs_are_selected(self):
        # arbitrary client-chosen artifact keys; the framework never inspects them
        chart_output = CommandOutput(
            command_name="export_csv",
            command_response=
                CommandResponse(
                    response="exported", artifacts={"some_client_key": "a,b\n1,2"}
                ),
        )
        text_output = CommandOutput(
            command_name="list_todo_lists",
            command_response=CommandResponse(response="3 lists"),
        )

        turn_output = TurnOutput(
            turn_key=mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="exported your data",
            command_outputs=[text_output, chart_output],
        )

        assert turn_output.command_outputs_with_artifacts == [chart_output]


# ----------------------------------------------------------------------
# 11b: Artifact projection (Topic 5 — merge per-command artifacts into the
#      single user-facing CommandResponse from the agent finalize path)
# ----------------------------------------------------------------------


class TestArtifactProjection:
    def test_collect_artifact_responses_flattens_artifact_bearing_only(self):
        # the framework is key-agnostic: any non-empty artifacts dict qualifies
        artifact_out = CommandOutput(
            command_name="export_csv",
            command_response=
                CommandResponse(
                    response="exported",
                    artifacts={"anything": "a,b\n1,2", "mime": "text/csv"},
                ),
        )
        text_out = CommandOutput(
            command_name="list_todo_lists",
            command_response=CommandResponse(response="3 lists"),
        )

        projected = collect_artifact_responses([text_out, artifact_out])

        # original objects returned (no copy), artifact-bearing only, in order
        assert projected == [artifact_out.command_response]
        assert projected[0] is artifact_out.command_response

    def test_merge_artifact_responses_into_merges_and_suffixes_collisions(self):
        target = CommandResponse(
            response="answer",
            artifacts={"conversation_summary": "summary", "shared": "existing"},
        )
        merge_artifact_responses_into(
            target,
            [
                CommandResponse(
                    response="tool",
                    artifacts={"client_blob": "data", "shared": "incoming"},
                )
            ],
        )

        assert target.artifacts["conversation_summary"] == "summary"
        assert target.artifacts["client_blob"] == "data"
        assert target.artifacts["shared"] == "existing"
        assert target.artifacts["shared_1"] == "incoming"

    def test_agent_finalize_surfaces_command_artifacts(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        from fastworkflow.workflow_agent import _execute_workflow_query

        def fake_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command,
                command_response=
                    fastworkflow.CommandResponse(
                        response=f"ran {command}",
                        # arbitrary client keys — the framework preserves them verbatim
                        artifacts={
                            "client_blob": "a,b\n1,2",
                            "client_kind": "text/csv",
                        },
                    ),
            )

        _patch_invoke_with(monkeypatch, fake_invoke)

        def fake_forward(**kwargs):
            _execute_workflow_query("export_csv", ctx)
            return SimpleNamespace(final_answer="Exported your data")

        mock_agent = MagicMock(side_effect=fake_forward)
        _set_agents(ctx, mock_agent)

        output = ctx._execute_message("export my todos")

        # single user-facing response: agent text plus merged tool artifacts
        answer = output.command_response
        assert "Exported your data" in answer.response
        assert answer.artifacts["client_blob"] == "a,b\n1,2"
        assert answer.artifacts["client_kind"] == "text/csv"

    def test_process_turn_answer_unaffected_by_projection(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        from fastworkflow.workflow_agent import _execute_workflow_query

        def fake_invoke(cls, session, command: str):
            return fastworkflow.CommandOutput(
                command_name=command,
                command_response=
                    fastworkflow.CommandResponse(
                        response=f"ran {command}",
                        artifacts={"client_blob": "x,y\n3,4"},
                    ),
            )

        _patch_invoke_with(monkeypatch, fake_invoke)

        def fake_forward(**kwargs):
            _execute_workflow_query("export_csv", ctx)
            return SimpleNamespace(final_answer="Done")

        mock_agent = MagicMock(side_effect=fake_forward)
        _set_agents(ctx, mock_agent)

        result = ctx.process_turn("export my todos")

        # answer is the synthesized agent text; structured artifacts are NOT on the
        # answer (it is plain text) — they live on the per-command outputs.
        assert isinstance(result.answer, str)
        assert "Done" in result.answer
        # command_outputs_with_artifacts is derived from the per-command outputs
        assert len(result.command_outputs_with_artifacts) == 1
        artifact_output = result.command_outputs_with_artifacts[0]
        assert artifact_output.command_name == "export_csv"
        assert artifact_output.command_response.artifacts["client_blob"] == "x,y\n3,4"


# ----------------------------------------------------------------------
# 12-13: Eager artifact validation
# ----------------------------------------------------------------------


class TestArtifactValidation:
    def _bad_output(self) -> CommandOutput:
        return CommandOutput(
            command_name="bad_cmd",
            command_response=
                CommandResponse(response="x", artifacts={"bad": object()}),
        )

    def test_append_turn_output_warns_on_unserializable_artifacts(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        monkeypatch.delenv("FW_EAGER_ARTIFACT_VALIDATION", raising=False)

        with pytest.warns(UserWarning, match="Unserializable command artifacts"):
            ctx.append_turn_output(self._bad_output())

    def test_warning_suppressed_when_validation_disabled(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        monkeypatch.setenv("FW_EAGER_ARTIFACT_VALIDATION", "0")

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            ctx.append_turn_output(self._bad_output())

        assert len(ctx._turn_outputs) == 1  # still appended

    def test_validate_artifacts_serializable_flags_and_accepts(self):
        bad = CommandOutput(
            command_name="bad_cmd",
            command_response=
                CommandResponse(response="x", artifacts={"obj": object()}),
        )
        problems = validate_artifacts_serializable(bad)
        assert len(problems) == 1
        assert "'obj'" in problems[0]
        assert "bad_cmd" in problems[0]

        good = CommandOutput(
            command_name="good_cmd",
            command_response=
                CommandResponse(
                    response="x",
                    artifacts={
                        "nested": {
                            "list": [1, "two", 3.0, None, True],
                            "tuple": (1, 2),
                            "deep": {"a": {"b": ["c", {"d": None}]}},
                        },
                        "scalar": 42,
                        "none": None,
                    },
                ),
        )
        assert validate_artifacts_serializable(good) == []


# ----------------------------------------------------------------------
# 14: Internal TurnResult (full capture, retained behind the public projection)
# ----------------------------------------------------------------------


class TestInternalTurnResult:
    def test_build_turn_result_composes_turn_output_and_internal_fields(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        # Reproduce process_turn()'s internals to inspect the full TurnResult
        # (which process_turn no longer returns — it returns turn_result.turn_output).
        command_output = ctx._execute_message("list my todo lists")
        turn_result = ctx._build_turn_result(command_output)

        assert isinstance(turn_result, TurnResult)
        # internal-only observability/persistence fields live on TurnResult
        assert turn_result.user_message == "list my todo lists"
        assert turn_result.started_at is not None
        assert turn_result.completed_at is not None
        assert turn_result.suspended_ms == 0
        # the public slice is composed as turn_output
        assert isinstance(turn_result.turn_output, TurnOutput)
        assert turn_result.turn_output.status == TurnStatus.COMPLETED
        assert isinstance(turn_result.turn_output.answer, str)
        # command outputs (with per-command timing) live on the turn_output
        captured = turn_result.turn_output.command_outputs[-1]
        assert captured.started_at is not None
        assert captured.duration_ms is not None
        assert captured.duration_ms >= 0


# ----------------------------------------------------------------------
# 14a: Conversation-memory columns (fix-24f.1; Phase-7 design §2.1, ruling I5)
#
# The turns table is only usable as conversation memory if a turn record
# carries the history entry that turn produced. These are the cases that decide
# whether a row is memory or merely a trace.
# ----------------------------------------------------------------------


class TestConversationMemoryStamping:
    def test_deterministic_turn_stamps_the_entry_it_appended(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)

        command_output = ctx._execute_message("list my todo lists")
        turn_result = ctx._build_turn_result(command_output)

        newest = ctx.conversation_history.messages[-1]
        assert turn_result.conversation_summary == newest["conversation summary"]
        assert turn_result.conversation_traces == newest["conversation_traces"]
        assert turn_result.conversation_summary

    def test_agent_turn_stamps_the_summarized_entry(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)

        def run(*args, **kwargs):
            # _run_agent clears the action log at turn start, so the actions
            # that select the LLM-summary path have to arrive during the run.
            ctx.append_action_log({"command_name": "x", "response": "y"})
            return SimpleNamespace(
                final_answer="done", suspended=False, exhausted=False
            )

        agent = MagicMock(side_effect=run)
        _set_agents(ctx, agent)

        command_output = ctx._execute_message("do the thing")
        turn_result = ctx._build_turn_result(command_output)

        assert turn_result.conversation_summary == "summary"
        assert turn_result.conversation_traces == "{}"

    def test_turn_that_appends_nothing_leaves_the_columns_empty(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """A pre-existing entry must not be attributed to this turn.

        The row is write-once, so stamping the newest entry unconditionally
        would permanently record the previous turn's summary against this one.
        """
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        ctx.append_conversation_turn("an earlier turn", "{}")

        ctx._begin_turn("a turn that appends nothing")
        turn_result = ctx._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )

        assert turn_result.conversation_summary is None
        assert turn_result.conversation_traces is None

    def test_restored_turn_baselines_on_the_restored_history(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """Ruling I5: a cross-process resume rebuilds the baseline from the
        restored history, so the entry that arrived with the history is not
        re-attributed to the resumed turn."""
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        ctx._begin_turn("original message")
        state = ctx.serialize_state(channel_id="chan")
        state["conversation_history_turns"] = [
            {
                "conversation summary": "history that came back with the restore",
                "conversation_traces": "{}",
                "feedback": None,
            }
        ]

        resumed, _wf2 = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        resumed.apply_serialized_state(state)
        assert len(resumed.conversation_history.messages) == 1

        turn_result = resumed._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        assert turn_result.conversation_summary is None

        resumed.append_conversation_turn("appended after resume", "{}")
        turn_result = resumed._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        assert turn_result.conversation_summary == "appended after resume"

    def test_last_completed_turn_key_names_the_turn_feedback_belongs_to(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """Ruling I3: feedback keys off a real turn, never off max(ordinal).

        Only a turn that both completed and contributed a memory entry can carry
        feedback — the memory window filters on exactly that, so any other row
        would file the feedback where no reader joins it.
        """
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        assert ctx.last_completed_turn_key is None

        command_output = ctx._execute_message("list my todo lists")
        turn_result = ctx._build_turn_result(command_output)
        assert ctx.last_completed_turn_key == turn_result.turn_output.turn_key
        assert ctx.last_turn_added_memory is True

        # A turn that appends nothing must not steal the key from the turn that
        # did, or feedback moves to a turn the user never saw.
        first_key = ctx.last_completed_turn_key
        ctx._begin_turn("a turn that appends nothing")
        ctx._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        assert ctx.last_completed_turn_key == first_key
        assert ctx.last_turn_added_memory is False

    def test_last_completed_turn_key_survives_a_cross_process_resume(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        command_output = ctx._execute_message("list my todo lists")
        expected = ctx._build_turn_result(command_output).turn_output.turn_key

        state = ctx.serialize_state(channel_id="chan")
        assert state["last_completed_turn_key"] == expected

        resumed, _wf2 = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        resumed.apply_serialized_state(state)
        assert resumed.last_completed_turn_key == expected

    def test_history_read_through_the_property_survives_a_swap(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch
    ):
        """Distillation replaces the history object and truncates it back to a
        pre-pass length; the guard must not stamp off a stale list."""
        import dspy

        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch)
        ctx.append_conversation_turn("before", "{}")
        ctx._begin_turn("a turn whose history gets rolled back")
        ctx.append_conversation_turn("during", "{}")
        ctx._conversation_history = dspy.History(
            messages=list(ctx.conversation_history.messages[:1])
        )

        turn_result = ctx._build_turn_result(
            fastworkflow.CommandOutput(
                command_name="x",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        )
        assert turn_result.conversation_summary is None


# ----------------------------------------------------------------------
# 15: TurnOutput (public) — shape and calculated success
# ----------------------------------------------------------------------


class TestTurnOutput:
    def _turn_output(self, **overrides) -> TurnOutput:
        export_output = CommandOutput(
            command_name="export_csv",
            command_response=
                CommandResponse(
                    response="exported", artifacts={"payload": "a,b\n1,2"}
                ),
            started_at=datetime.now(timezone.utc),
            duration_ms=12,
        )
        kwargs = dict(
            turn_key=mint_turn_key(),
            status=TurnStatus.COMPLETED,
            failure_reason=None,
            answer="exported your data",
            command_outputs=[export_output],
        )
        kwargs.update(overrides)
        return TurnOutput(**kwargs)

    def test_answer_is_text_and_contract_fields_present(self):
        out = self._turn_output()
        assert isinstance(out.answer, str)
        assert out.success is True
        assert len(out.command_outputs_with_artifacts) == 1
        assert out.command_outputs_with_artifacts[0].command_name == "export_csv"
        # per-command structured results live on the command outputs, not on answer
        assert out.command_outputs[0].command_response.artifacts["payload"]

    def test_success_is_only_all_command_outputs_succeeded(self):
        # success is purely all(command_outputs succeeded) — orthogonal to
        # status and failure_reason.
        assert self._turn_output(status=TurnStatus.COMPLETED).success is True
        # orthogonal: a FAILED turn whose commands all succeeded still has
        # success=True (consumer combines status + failure_reason + success)
        assert (
            self._turn_output(
                status=TurnStatus.FAILED, failure_reason="max_iters_exhausted"
            ).success
            is True
        )
        # a command failure → success False regardless of status
        failed_command = CommandOutput(
            command_name="mark_completed",
            command_response=CommandResponse(response="nope", success=False),
        )
        assert (
            self._turn_output(
                status=TurnStatus.COMPLETED, command_outputs=[failed_command]
            ).success
            is False
        )
        # empty command outputs → vacuously True (nothing failed)
        assert (
            self._turn_output(
                status=TurnStatus.COMPLETED, command_outputs=[]
            ).success
            is True
        )

    def test_public_shape_has_only_consumer_fields(self):
        dumped = self._turn_output().model_dump()
        assert set(dumped) == {
            "turn_key",
            "status",
            "failure_reason",
            "answer",
            "command_outputs",
            "success",  # computed field
        }
        # internal-only fields live on TurnResult, never on the public TurnOutput
        assert "user_message" not in dumped
        assert "started_at" not in dumped
        assert "suspended_ms" not in dumped
