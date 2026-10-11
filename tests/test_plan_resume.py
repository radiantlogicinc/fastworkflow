"""Cross-process ask_user resume: planner view and evidence archive."""

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
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    observability_db_path,
    reset_observation_state,
    scope_for_host,
)
from fastworkflow.session_state_store import DiskSessionStateStore
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

TESTS = Path(__file__).parent
HELLO_WORKFLOW = str(TESTS.joinpath("hello_world_workflow").resolve())
EXAMPLE_ENV = TESTS.parent / "fastworkflow" / "examples" / "fastworkflow.env"
LLM_ROLES = ("LLM_AGENT", "LLM_PLANNER", "LLM_PARAM_EXTRACTION", "LLM_RESPONSE_GEN",
             "LLM_CONVERSATION_STORE", "LLM_OBSERVATION_SEARCH", "LLM_SYNDATA_GEN")
PARAMS_MODEL = "params"

ADD_BOTH = "add_two_numbers <first_num>5</first_num><second_num>3</second_num>"
ADD_ONE = "add_two_numbers <first_num>5</first_num>"
ADD_PLAN_TEXT = "1. Add 5 and 3 with `add_two_numbers`"


class LlmStub:
    """OpenAI-compatible loopback stub for DSPy chat-format replies."""

    def __init__(self) -> None:
        self.agent_steps: list[tuple[str, dict[str, Any]]] = []
        self.answers: dict[str, Any] = {
            "reasoning": "r",
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

            def do_POST(self) -> None:  # noqa: N802
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
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
            name="llm-stub", daemon=True,
        )
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
        section = re.search(r"Your output fields are:\n(.*?)(?:\n\n|All interactions)", prompt, re.S)
        fields = re.findall(r"^\d+\. `(\w+)`", section.group(1), re.M) if section else []
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
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    _init(stub, tmp_path)
    RoutingRegistry.clear_registry()
    reset_observation_state()
    yield stub
    RoutingRegistry.clear_registry()
    stub.close()


def _context(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    ctx.bind_app_workflow(fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id))
    return ctx


def _move(ctx: WorkflowExecutionContext, workflow_path: str, channel_id: str, store_dir: Path,
          *, edit=None) -> WorkflowExecutionContext:
    store = DiskSessionStateStore(str(store_dir))
    blob = ctx.serialize_state(channel_id=channel_id)
    if edit is not None:
        edit(blob)
    store.save(channel_id, blob)
    ctx.close()
    moved = _context(workflow_path, channel_id)
    moved.apply_serialized_state(store.load(channel_id))
    return moved


@pytest.fixture
def channel_id():
    return f"plan-resume-{uuid.uuid4().hex}"


def test_a_step_stopped_at_parameter_extraction_replans(llm, channel_id):
    llm.answers["next_steps"] = ADD_PLAN_TEXT
    llm.agent_steps = [
        ("execute_workflow_query", {"command": ADD_ONE}),
        ("execute_workflow_query", {"command": "go_up"}),
        ("execute_workflow_query", {"command": ADD_BOTH}),
        ("finish", {}),
    ]
    ctx = _context(HELLO_WORKFLOW, channel_id)
    ctx.process_turn("add 5 and 3")
    agent = ctx.workflow_tool_agent
    assert "PARAMETER EXTRACTION ERROR" in agent.trajectory["observation_0"]
    ctx.close()


def _replan_request(llm: LlmStub) -> str:
    prompt = [p for p in llm.prompts if "user_response" in p and "agent_trajectory" in p][-1]
    return prompt[prompt.rfind("[[ ## agent_inputs ## ]]"):]


@pytest.mark.parametrize("cold", [False, True], ids=["same_process", "other_process"])
def test_the_ask_user_replan_sees_the_request_and_trajectory_after_a_resume(
        llm, tmp_path, channel_id, cold):
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}),
                       ("ask_user", {"clarification_request": "Add more?"})]
    ctx = _context(HELLO_WORKFLOW, channel_id)
    ctx.process_turn("add 5 and 3 then ask me")
    assert ctx.awaiting_user
    if cold:
        ctx = _move(ctx, HELLO_WORKFLOW, channel_id, tmp_path / "state")
    agent = ctx.workflow_tool_agent
    inputs, trajectory = agent.planner_view()
    if cold:
        assert agent.inputs == {} and agent.trajectory == {}
        assert inputs == agent._suspended["input_args"] and inputs is not agent._suspended["input_args"]
        assert trajectory == agent._suspended["trajectory"]
    else:
        assert inputs is agent.inputs and trajectory is agent.trajectory
        assert inputs and "tool_name_0" in trajectory

    llm.agent_steps = [("finish", {})]
    ctx.process_turn("no")

    request = _replan_request(llm)
    assert "add 5 and 3 then ask me" in request
    assert '"observation_0"' in request and "sum_of_two_numbers" in request
    assert "add_two_numbers" in request
    with suppress(Exception):
        ctx.close()


def test_a_context_resuming_a_suspended_turn_archives_in_the_workflows_own_database(
        llm, tmp_path, channel_id):
    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}),
                       ("ask_user", {"clarification_request": "Add more?"})]
    first = _context(HELLO_WORKFLOW, channel_id)
    first.process_turn("add 5 and 3 then ask me")
    assert first.awaiting_user
    expected = state_paths.observability_db(HELLO_WORKFLOW)
    scope = scope_for_host(first)
    assert observability_db_path(first) == expected
    assert archive_for_path(expected).get(scope, "O0") is not None

    resumed = _move(first, HELLO_WORKFLOW, channel_id, tmp_path / "state")

    assert observability_db_path(resumed) == expected
    assert scope_for_host(resumed) == scope
    before = archive_for_path(expected).get(scope, "O0")
    assert before is not None and "sum_of_two_numbers" in before["text"]

    llm.agent_steps = [("execute_workflow_query", {"command": ADD_BOTH}), ("finish", {})]
    resumed.process_turn("yes, once more")
    assert not resumed.awaiting_user
    after = archive_for_path(expected).get(scope_for_host(resumed), "O0")
    assert after is not None and "sum_of_two_numbers" in after["text"]
    stray = os.path.join(state_paths.state_root(), "workflows",
                         state_paths.workflow_id(os.getcwd()), "observability.sqlite3")
    assert stray != expected and not os.path.exists(stray)
    with suppress(Exception):
        resumed.close()
