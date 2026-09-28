"""The finish check across a cross-process ask_user resume, and steps the NLU stage stopped.

Real turns through ``WorkflowExecutionContext.process_turn`` on the real test
workflows, with a real tool agent and a real finish checker built from settings.
Nothing leaves the machine: every LLM role (agent, planner, parameter extraction,
summary) is ``litellm_proxy/<role>`` pointed at a loopback OpenAI-compatible
stand-in defined here, which answers each call by the output fields its prompt
asks for and refuses parameter extraction with a 400; the decision model is the
loopback Jev stand-in (``jev_stub`` fixture). A suspended turn is serialized,
saved and loaded through a real ``DiskSessionStateStore`` and applied to a NEW
context, which is what a second worker process does.

``tests/todo_list_workflow`` is untrained, so no command can be dispatched on it
(intent detection needs its ``threshold.json``); the tests that need a command
to run, or to be stopped by the NLU stage, use the trained
``tests/hello_world_workflow``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import state_paths
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.observation_offloading import finish_check, jev_client
from fastworkflow.observation_offloading.finish_check import CHECK_ENV, KEY_ENV, MODEL_ENV, build_ledger
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events
from fastworkflow.runtime_manifest import (
    RuntimeManifest,
    clear_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.session_state_store import DiskSessionStateStore
from fastworkflow.turn_plan import TurnPlan
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

needs_sdk = pytest.mark.skipif(jev_client.TypeSafeClient is None, reason="typesafe-sdk not installed")

TESTS = Path(__file__).parent
TODO_WORKFLOW = str(TESTS.joinpath("todo_list_workflow").resolve())
HELLO_WORKFLOW = str(TESTS.joinpath("hello_world_workflow").resolve())
EXAMPLE_ENV = TESTS.parent / "fastworkflow" / "examples" / "fastworkflow.env"
LLM_ROLES = ("LLM_AGENT", "LLM_PLANNER", "LLM_PARAM_EXTRACTION", "LLM_RESPONSE_GEN",
             "LLM_CONVERSATION_STORE", "LLM_OBSERVATION_SEARCH", "LLM_SYNDATA_GEN")
PARAMS_MODEL = "params"

ADD_BOTH = "add_two_numbers <first_num>5</first_num><second_num>3</second_num>"
ADD_ONE = "add_two_numbers <first_num>5</first_num>"


# ---------------------------------------------------------------------------
# The loopback LLM stand-in
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS_RE = re.compile(r"Your output fields are:\n(.*?)(?:\n\n|All interactions)", re.S)
_FIELD_RE = re.compile(r"^\d+\. `(\w+)`", re.M)


class LlmStub:
    """An OpenAI-compatible ``/chat/completions`` on 127.0.0.1, answering in DSPy's chat format.

    A prompt asking for ``next_tool_name`` gets the next of ``agent_steps``
    (``(tool, args)``); any other gets ``answers`` for the fields it asks for.
    The ``params`` model (parameter extraction) is refused with a 400, so a
    command missing a parameter stops at the parameter-extraction stage.
    """

    def __init__(self) -> None:
        self.agent_steps: list[tuple[str, dict[str, Any]]] = []
        self.answers: dict[str, Any] = {
            "reasoning": "r",
            # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
            # "subjects": [],
            # "steps": [],
            "next_steps": "1. Carry on",
            "final_answer": "Done.",
            "conversation_summary": "summary",
        }
        self.prompts: list[str] = []
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                status, payload = stub._reply(body)
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="llm-stub", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _reply(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if str(body.get("model") or "").endswith(PARAMS_MODEL):
            return 400, {"error": {"message": "parameter extraction is refused here",
                                   "type": "invalid_request_error", "code": "refused"}}
        prompt = "\n".join(str(m.get("content") or "") for m in body.get("messages") or [])
        section = _OUTPUT_FIELDS_RE.search(prompt)
        fields = _FIELD_RE.findall(section.group(1)) if section else []
        with self._lock:
            self.prompts.append(prompt)
            if "next_tool_name" in fields:
                tool, args = self.agent_steps.pop(0)
                values = {"next_thought": f"use {tool}", "next_tool_name": tool, "next_tool_args": args}
            else:
                values = {name: self.answers.get(name, "x") for name in fields}
        content = "".join(
            f"[[ ## {name} ## ]]\n{value if isinstance(value, str) else json.dumps(value)}\n\n"
            for name, value in values.items()) + "[[ ## completed ## ]]"
        return 200, {"id": f"stub-{uuid.uuid4().hex}", "object": "chat.completion", "created": 0,
                     "model": body.get("model"),
                     "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": content}}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _init(llm: LlmStub, tmp_path: Path) -> None:
    env = dict(dotenv_values(EXAMPLE_ENV))
    env.update({role: f"litellm_proxy/{role.lower()}" for role in LLM_ROLES})
    env["LLM_PARAM_EXTRACTION"] = f"litellm_proxy/{PARAMS_MODEL}"
    env.update({"LITELLM_PROXY_API_BASE": llm.base_url, "LITELLM_PROXY_API_KEY": "stub-key",
                "FW_LM_CACHE": "0",
                "FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.init(env_vars=env)


@pytest.fixture
def llm(tmp_path, monkeypatch):
    stub = LlmStub()
    for name in (CHECK_ENV, KEY_ENV, MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    _init(stub, tmp_path)
    RoutingRegistry.clear_registry()
    reset_runtime_state()
    finish_check._CHECKERS.clear()
    yield stub
    RoutingRegistry.clear_registry()
    finish_check._CHECKERS.clear()
    stub.close()


@pytest.fixture
def checked(llm, jev_stub, monkeypatch):
    """``FW_FINISH_CHECK=jev`` with a key, against the Jev stand-in; returns (llm, jev).

    Both workflows' effects are registered as a startup would: the check only
    asks about steps whose commands are declared read-only. The todo list
    declares nothing of its own, so ``what_can_i_do`` is the core manifest's.
    """
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "stub-key")
    jev_client._WARNED.clear()
    clear_runtime_metadata()
    register_runtime_metadata(TODO_WORKFLOW, merge_and_gate(
        RuntimeManifest(schema_version=1, manifest_version="1.0.0"), deployment_features={}))
    register_runtime_metadata(HELLO_WORKFLOW, merge_and_gate(
        RuntimeManifest(schema_version=1, manifest_version="1.0.0",
                        commands={"add_two_numbers": {"effect": {"kind": "read_only"}}}),
        deployment_features={}))
    yield llm, jev_stub
    clear_runtime_metadata()
    jev_client._WARNED.clear()


def _context(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    ctx.bind_app_workflow(fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id))
    return ctx


def _move(ctx: WorkflowExecutionContext, workflow_path: str, channel_id: str, store_dir: Path,
          *, edit=None) -> WorkflowExecutionContext:
    """Serialize *ctx*'s suspended turn, save and load it, and apply it to a new context."""
    store = DiskSessionStateStore(str(store_dir))
    blob = ctx.serialize_state(channel_id=channel_id)
    if edit is not None:
        edit(blob)
    store.save(channel_id, blob)
    ctx.close()
    moved = _context(workflow_path, channel_id)
    moved.apply_serialized_state(store.load(channel_id))
    return moved


def _finish_events() -> list[dict[str, Any]]:
    return [e for e in snapshot_events() if e["kind"] == "finish_check"]


def _sent_ledgers(jev) -> list[list[dict[str, Any]]]:
    return [body["state"]["ledger"] for body in jev.bodies if "ledger" in (body.get("state") or {})]


def _plan(*steps: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"text": "", "commands": [], "optional": False, "needs_user": False, **step} for step in steps]


def _steps(plan: TurnPlan | None) -> list[tuple[str, list[str], bool, bool]]:
    """What the finish check reads from each step of *plan*."""
    assert plan is not None
    return [(step.text, step.commands, step.optional, step.needs_user) for step in plan.steps]


# The planner is plain text only, so the finish check's plan is what
# parse_text_plan recovers from these: backticked command names and
# user-gated phrasing carry what the structured fields used to.
ADD_PLAN_TEXT = "1. Add 5 and 3 with `add_two_numbers`"
ADD_STEPS = [("Add 5 and 3 with `add_two_numbers`", ["add_two_numbers"], False, False)]


@pytest.fixture
def channel_id():
    return f"plan-resume-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# The plan and the ledger across a cross-process resume
# ---------------------------------------------------------------------------

TODO_TURN = {
    "path": TODO_WORKFLOW,
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # "plan": _plan({"text": "Show the commands available", "commands": ["what_can_i_do"]},
    #               {"text": "Ask the user which list to open", "needs_user": True}),
    "plan_text": ("1. Show the commands available with `what_can_i_do`\n"
                  "2. Ask the user which list to open and wait for the user's confirmation"),
    "plan_steps": [("Show the commands available with `what_can_i_do`", ["what_can_i_do"], False, False),
                   ("Ask the user which list to open and wait for the user's confirmation", [], False, True)],
    "steps": [("what_can_i_do", {}), ("ask_user", {"clarification_request": "Which list?"})],
    "rows": ["{}", '{"clarification_request": "Which list?"}'],
}
HELLO_TURN = {
    "path": HELLO_WORKFLOW,
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # "plan": _plan({"text": "Add 5 and 3", "commands": ["add_two_numbers"]},
    #               {"text": "Ask the user whether to add more", "needs_user": True}),
    "plan_text": (f"{ADD_PLAN_TEXT}\n"
                  "2. Ask the user whether to add more and wait for the user's confirmation"),
    "plan_steps": [*ADD_STEPS,
                   ("Ask the user whether to add more and wait for the user's confirmation", [], False, True)],
    "steps": [("execute_workflow_query", {"command": ADD_BOTH}),
              ("ask_user", {"clarification_request": "Add more?"})],
    "rows": [ADD_BOTH, '{"clarification_request": "Add more?"}'],
}


@needs_sdk
@pytest.mark.parametrize("turn", [TODO_TURN, HELLO_TURN], ids=["todo_list", "hello_world"])
def test_a_turn_resumed_in_another_process_is_checked_against_its_plan_and_whole_ledger(
        checked, tmp_path, channel_id, turn):
    llm, jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers["steps"] = turn["plan"]
    llm.answers["next_steps"] = turn["plan_text"]
    llm.agent_steps = list(turn["steps"])
    first = _context(turn["path"], channel_id)
    first.process_turn("do the thing")
    assert first.awaiting_user
    agent = first.workflow_tool_agent
    assert agent.finish_checker is not None
    plan = first._turn_plan
    assert plan is not None and first._turn_plan_status == "planned"
    assert _steps(plan) == turn["plan_steps"]
    in_process = build_ledger(agent, [])
    assert [row["command"] for row in in_process] == turn["rows"]

    resumed = _move(first, turn["path"], channel_id, tmp_path / "state")

    assert resumed._turn_plan == plan and resumed._turn_plan_status == "planned"
    assert build_ledger(resumed.workflow_tool_agent, []) == in_process
    llm.agent_steps = [("finish", {})]
    resumed.process_turn("no")
    assert not resumed.awaiting_user
    event = _finish_events()[-1]
    assert event["reason"] == "every step executed" and event["requests"] == 2
    sent, = _sent_ledgers(jev)
    assert [row["command"] for row in sent] == turn["rows"]
    with suppress(Exception):
        resumed.close()


@needs_sdk
def test_a_state_written_without_the_plan_resumes_with_the_plan_lost(checked, tmp_path, channel_id):
    llm, jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers["steps"] = TODO_TURN["plan"]
    llm.answers["next_steps"] = TODO_TURN["plan_text"]
    llm.agent_steps = list(TODO_TURN["steps"])
    first = _context(TODO_WORKFLOW, channel_id)
    first.process_turn("do the thing")
    assert first.awaiting_user
    assert _steps(first._turn_plan) == TODO_TURN["plan_steps"]

    def written_before_the_key(blob):
        del blob["turn_plan"], blob["turn_plan_status"]

    resumed = _move(first, TODO_WORKFLOW, channel_id, tmp_path / "state", edit=written_before_the_key)

    assert resumed._turn_plan is None and resumed._turn_plan_status == "lost_on_resume"
    llm.agent_steps = [("finish", {})]
    resumed.process_turn("no")
    event = _finish_events()[-1]
    assert (event["reason"], event["no_plan_cause"]) == ("no plan", "lost_on_resume")
    assert jev.requests == []
    with suppress(Exception):
        resumed.close()


@needs_sdk
def test_a_suspension_the_context_window_fallback_cut_is_not_checked(checked, tmp_path, channel_id):
    llm, jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers["steps"] = HELLO_TURN["plan"]
    llm.answers["next_steps"] = HELLO_TURN["plan_text"]
    llm.agent_steps = list(HELLO_TURN["steps"])
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("do the thing")
    assert first.awaiting_user
    assert _steps(first._turn_plan) == HELLO_TURN["plan_steps"]
    agent = first.workflow_tool_agent
    # What the context-window fallback does to the working trajectory; the mirror keeps the step.
    agent.truncate_trajectory(agent._suspended["trajectory"])
    assert agent.truncated_execute_steps == 1 and "tool_name_0" in agent.current_trajectory

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state")

    assert resumed.workflow_tool_agent.ledger_incomplete is True
    llm.agent_steps = [("finish", {})]
    resumed.process_turn("no")
    assert _finish_events()[-1]["reason"] == "ledger incomplete"
    assert jev.requests == []
    with suppress(Exception):
        resumed.close()


@needs_sdk
def test_a_planner_that_returns_nothing_is_the_no_plan_cause(checked, channel_id):
    llm, jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers.update({"steps": [], "next_steps": ""})
    llm.answers.update({"next_steps": ""})
    llm.agent_steps = [("finish", {})]
    ctx = _context(TODO_WORKFLOW, channel_id)
    ctx.process_turn("do the thing")
    assert ctx._turn_plan_status == "planner_empty"
    event = _finish_events()[-1]
    assert (event["reason"], event["no_plan_cause"]) == ("no plan", "planner_empty")
    assert jev.requests == []
    ctx.close()


# ---------------------------------------------------------------------------
# Steps the NLU stage stopped before any command ran
# ---------------------------------------------------------------------------

@needs_sdk
def test_a_step_stopped_at_parameter_extraction_is_an_error_and_go_up_is_a_result(checked, channel_id):
    llm, jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers["steps"] = _plan({"text": "Add 5 and 3", "commands": ["add_two_numbers"]})
    llm.answers["next_steps"] = ADD_PLAN_TEXT
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_ONE}),
                       ("execute_workflow_query", {"command": "go_up"}),
                       ("execute_workflow_query", {"command": ADD_BOTH}),
                       ("finish", {})]
    ctx = _context(HELLO_WORKFLOW, channel_id)
    ctx.process_turn("add 5 and 3")
    assert _steps(ctx._turn_plan) == ADD_STEPS

    agent = ctx.workflow_tool_agent
    assert agent.dispatch_outcomes == {"0": "not_run", "1": "ran", "2": "ran"}
    assert "PARAMETER EXTRACTION ERROR" in agent.current_trajectory["observation_0"]
    sent, = _sent_ledgers(jev)
    assert [(row["command"], row["outcome"]) for row in sent] == [
        (ADD_ONE, "error"), ("go_up", "result"), (ADD_BOTH, "result")]
    ctx.close()


@needs_sdk
def test_dispatch_outcomes_ride_the_suspension_with_string_keys(checked, tmp_path, channel_id):
    llm, _jev = checked
    # Structured planning disabled 2026-09-28 (owner decision); kept for reference.
    # llm.answers["steps"] = _plan({"text": "Add 5 and 3", "commands": ["add_two_numbers"]})
    llm.answers["next_steps"] = ADD_PLAN_TEXT
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_ONE}),
                       ("ask_user", {"clarification_request": "What is the second number?"})]
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("add 5")
    assert first.awaiting_user
    assert _steps(first._turn_plan) == ADD_STEPS
    blob = first.serialize_state(channel_id=channel_id)
    assert blob["react"]["dispatch_outcomes"] == {"0": "not_run"}

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state")

    agent = resumed.workflow_tool_agent
    assert agent.dispatch_outcomes == {"0": "not_run"}
    # The question is still unanswered, so its row is empty.
    assert [row["outcome"] for row in build_ledger(agent, [])] == ["error", "empty"]
    with suppress(Exception):
        resumed.close()


@needs_sdk
def test_a_check_switched_off_records_restores_and_seeds_nothing(checked, tmp_path, channel_id,
                                                                  monkeypatch):
    """The control arm: a check attached but FW_EVAL_FINISH_REMINDERS=0 is inactive everywhere."""
    llm, jev = checked
    monkeypatch.setenv("FW_EVAL_FINISH_REMINDERS", "0")
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_ONE}),
                       ("ask_user", {"clarification_request": "What is the second number?"})]
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("add 5")
    assert first.awaiting_user
    agent = first.workflow_tool_agent
    assert agent.finish_checker is not None and agent.finish_reminders_enabled is False
    assert agent.dispatch_outcomes == {}

    def with_a_plan(blob):
        # What an active check would have stored: a switched-off one must not restore it.
        blob["turn_plan"] = {"steps": _plan({"text": "Add 5 and 3", "commands": ["add_two_numbers"]}),
                             "subjects": []}
        blob["turn_plan_status"] = "planned"
        assert TurnPlan.model_validate(blob["turn_plan"]).steps

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state", edit=with_a_plan)

    agent = resumed.workflow_tool_agent
    assert agent.current_trajectory == {} and agent.mirror_restored is False
    assert resumed._turn_plan is None
    assert jev.requests == []
    with suppress(Exception):
        resumed.close()


def test_without_the_check_nothing_is_recorded_persisted_or_seeded(llm, tmp_path, channel_id):
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_ONE}),
                       ("ask_user", {"clarification_request": "What is the second number?"})]
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("add 5")
    assert first.awaiting_user
    assert first.workflow_tool_agent.finish_checker is None
    assert first.workflow_tool_agent.dispatch_outcomes == {}
    blob = first.serialize_state(channel_id=channel_id)
    assert "dispatch_outcomes" not in blob["react"]
    assert (blob["turn_plan"], blob["turn_plan_status"]) == (None, "not_planned")

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state")

    agent = resumed.workflow_tool_agent
    assert agent.current_trajectory == {} and agent.mirror_restored is False
    assert resumed._turn_plan is None and resumed._turn_plan_status == "not_planned"
    with suppress(Exception):
        resumed.close()


# ---------------------------------------------------------------------------
# What the ask_user replan plans from, without the check
# ---------------------------------------------------------------------------

def _replan_request(llm: LlmStub) -> str:
    """The filled-in part of the last replan prompt, from its ``agent_inputs`` onward."""
    prompt = [p for p in llm.prompts if "user_response" in p and "agent_trajectory" in p][-1]
    return prompt[prompt.rfind("[[ ## agent_inputs ## ]]"):]


@pytest.mark.parametrize("cold", [False, True], ids=["same_process", "other_process"])
def test_the_ask_user_replan_sees_the_request_and_trajectory_after_a_resume(llm, tmp_path, channel_id, cold):
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}),
                       ("ask_user", {"clarification_request": "Add more?"})]
    ctx = _context(HELLO_WORKFLOW, channel_id)
    ctx.process_turn("add 5 and 3 then ask me")
    assert ctx.awaiting_user
    if cold:
        ctx = _move(ctx, HELLO_WORKFLOW, channel_id, tmp_path / "state")
    agent = ctx.workflow_tool_agent
    assert agent.finish_checker is None
    inputs, trajectory = agent.planner_view()
    if cold:
        assert agent.inputs == {} and agent.current_trajectory == {}
        assert inputs == agent._suspended["input_args"] and inputs is not agent._suspended["input_args"]
        assert trajectory == agent._suspended["trajectory"]
    else:
        # A warm replan is given the very objects it always was.
        assert inputs is agent.inputs and trajectory is agent.current_trajectory
        assert inputs and "action_0" in trajectory

    llm.agent_steps = [("finish", {})]
    ctx.process_turn("no")

    request = _replan_request(llm)
    assert "add 5 and 3 then ask me" in request
    assert '"observation_0"' in request and "sum_of_two_numbers" in request
    assert "add_two_numbers" in request
    with suppress(Exception):
        ctx.close()


# ---------------------------------------------------------------------------
# Where the evidence lives after a resume, without the check
# ---------------------------------------------------------------------------

def test_a_context_resuming_a_suspended_turn_archives_in_the_workflows_own_database(
        llm, tmp_path, channel_id):
    """The resumed agent is built while applying the state, before any turn has
    made the workflow active. It must still open the workflow's archive: the
    one the first process wrote the turn's evidence to."""
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}),
                       ("ask_user", {"clarification_request": "Add more?"})]
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("add 5 and 3 then ask me")
    assert first.awaiting_user
    expected = state_paths.observability_db(HELLO_WORKFLOW)
    agent = first.workflow_tool_agent
    scope = agent.continuation_scope
    assert agent.observation_archive.db_path == expected
    assert agent.observation_archive.get(scope, "O1") is not None
    # The dispatch filed the command the step ran under its alias.
    assert agent.dispatched_commands[scope.scope_id]["O1"].rsplit("/", 1)[-1] == "add_two_numbers"

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state")

    moved = resumed.workflow_tool_agent
    assert moved.observation_archive.db_path == expected
    assert moved.continuation_scope == scope
    before = moved.observation_archive.get(scope, "O1")
    assert before is not None and "sum_of_two_numbers" in before["text"]
    # A process that imported the suspension has filed nothing for earlier steps.
    assert moved.dispatched_commands == {}

    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}), ("finish", {})]
    resumed.process_turn("yes, once more")
    assert not resumed.awaiting_user
    after = moved.observation_archive.get(moved.continuation_scope, "O2")
    assert after is not None and "sum_of_two_numbers" in after["text"]
    # Nothing was opened under the working directory's name.
    stray = os.path.join(state_paths.state_root(), "workflows",
                         state_paths.workflow_id(os.getcwd()), "observability.sqlite3")
    assert stray != expected and not os.path.exists(stray)
    with suppress(Exception):
        resumed.close()
