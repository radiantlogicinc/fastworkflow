"""Live chat streaming transport and chat-side artifact rendering.

Covers fix-9eg.20.1 (stream live interactions through the existing turn
transport) and the contracts the chat UI (fix-9eg.20.2 / fix-9eg.20.3) codes
to. No mocks (testing_rules.mdc): the FastAPI tests drive the REAL app over
tests/hello_world_workflow, with a real TurnRegistry, real ChannelRuntimes and
real HTTP through TestClient.

How the real route is exercised without paying for a model: the session's
runtime is put in deterministic mode, so a POST to ``/invoke_agent_stream``
runs the real CME pipeline over tests/hello_world_workflow — real intent
detection, real trace queue, real frames on a real HTTP body, in both NDJSON
and SSE framing — with no planner call. The AGENT path itself (the planner
choosing commands) is the one thing these tests cannot reach for free; that
limit is stated in the report rather than papered over.

A few transport invariants are driven through ``run_owned_turn`` with a
synthetic unit of work instead, because they are about what the transport does
when the WORK misbehaves (a consumer that leaves, a failure recorded after the
answer was delivered) and a real workflow will not misbehave on demand.

The environment is hermetic in both directions: env files are written into
tmp_path (so the suite does not skip on a developer box with no
``passwords/.env``, and no real key is ever read), and the state root is
redirected per test (fastapi_hermetic pattern).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests import test_run_chatbot_server as chatbot_fixtures
from fastworkflow.run_fastapi_mcp.turns import (
    STREAM_FORMAT_HEADER,
    STREAM_TURN_KEY_HEADER,
    TurnStreamChannel,
    compute_idempotency_key,
    encode_stream_frame,
)

# The DOM test needs a real chatbot server over a real seeded store; these are
# that module's fixtures, reused rather than reseeded. `workflow_path` is one
# of them (a tmp_path workflow with a redirected state root), so the FastAPI
# fixtures below deliberately use a different name — a collision here would
# point the seeded store at the repo's own hello_world workflow.
#
# Bound by assignment rather than imported by name: pytest registers a fixture
# under the name it is bound to either way, but a test taking `chatbot_server`
# as a parameter would be flagged as redefining an import (F811).
seeded_db = chatbot_fixtures.seeded_db
chatbot_server = chatbot_fixtures.server
workflow_path = chatbot_fixtures.workflow_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hello_world_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "tests", "hello_world_workflow")
    if not os.path.isdir(path):
        pytest.skip(f"hello_world_workflow not found at {path}")
    return path


@pytest.fixture
def env_files(tmp_path):
    """Workflow env files written from the shipped template.

    Placeholder keys only: nothing in this module reaches a model, and a test
    that silently started spending would be a bug, not a feature.
    """
    package_path = fastworkflow.get_fastworkflow_package_path()
    env_file = tmp_path / "fastworkflow.env"
    passwords_file = tmp_path / "fastworkflow.passwords.env"
    shutil.copy(
        os.path.join(package_path, "examples", "fastworkflow.env"), env_file
    )
    passwords_file.write_text(
        "\n".join(
            f"LITELLM_API_KEY_{role}=placeholder-never-used"
            for role in (
                "SYNDATA_GEN",
                "PARAM_EXTRACTION",
                "PLANNER",
                "AGENT",
                "CONVERSATION_STORE",
            )
        )
        + "\n"
    )
    return str(env_file), str(passwords_file)


@pytest.fixture
def app_module(hello_world_path, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path", hello_world_path,
        "--env_file_path", env_file,
        "--passwords_file_path", passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(
        env_file, passwords_file, tmp_path / "workflow_contexts"
    )
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _completed_output() -> fastworkflow.TurnOutput:
    return fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=fastworkflow.TurnStatus.COMPLETED,
    )


async def _runtime_for(app_module, channel_id: str):
    await app_module.ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=app_module.session_manager,
        workflow_path=app_module.ARGS.workflow_path,
        run_startup=False,
    )
    return await app_module.session_manager.get_session(channel_id)


async def _drain(stream: TurnStreamChannel) -> list[dict]:
    return [frame async for frame in stream.frames()]


def _stream_a_real_deterministic_turn(
    app_module, channel_id, query="what can i do", stream_format="ndjson",
    timeout_seconds=30,
):
    """POST a real turn to the real route and return (response, headers).

    The one concession to cost: the session's runtime answers deterministically
    instead of through the planner, so the whole route runs — CME pipeline,
    trace queue, registry, streaming body — without a model call. It is the
    real WorkflowExecutionContext either way; only the routing decision inside
    it differs, and that decision is the paid one.
    """
    client = TestClient(app_module.app)
    init = client.post(
        "/initialize",
        json={"channel_id": channel_id, "stream_format": stream_format},
    )
    assert init.status_code == 200, init.text
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def make_deterministic():
        runtime = await app_module.session_manager.get_session(channel_id)
        runtime.execution_context._run_as_agent = False
        return runtime

    asyncio.run(make_deterministic())

    response = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": query, "timeout_seconds": timeout_seconds},
    )
    return client, headers, response


def _parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _parse_sse(body: str) -> list[dict]:
    """The SSE framing, read the way an SSE client reads it."""
    frames = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        fields: dict[str, list[str]] = {}
        for line in block.split("\n"):
            name, _, value = line.partition(":")
            fields.setdefault(name, []).append(value.lstrip(" "))
        frames.append({
            "type": fields["event"][0],
            "seq": int(fields["id"][0]),
            "data": json.loads("\n".join(fields["data"])),
        })
    return frames


async def _own_a_streaming_turn(
    app_module, channel_id, work, stream, logical_resolver=None
):
    """Run one streaming turn exactly as the endpoint does, and drain it."""
    runtime = await _runtime_for(app_module, channel_id)
    registry = app_module.turn_registry

    async def owned(execn):
        stream.bind(
            execn.turn_key,
            logical_resolver
            or (lambda: app_module.resolve_logical_turn_key(execn, runtime, registry)),
        )
        try:
            await app_module.run_owned_turn(
                runtime,
                registry,
                execn,
                lambda: work(stream),
                app_module.session_manager,
                on_done=lambda: _finish(stream),
            )
        finally:
            stream.close()

    async def _finish(channel):
        channel.close()

    execn = await registry.start_or_get_active(
        channel_id,
        kind="invoke_agent_stream",
        idempotency_key=f"idem-{channel_id}",
        run_turn=lambda e: asyncio.create_task(owned(e)),
    )
    frames = await _drain(stream)
    await execn.done_event.wait()
    return execn, frames


# ---------------------------------------------------------------------------
# fix-9eg.20.1 — frame order, identity and deduplication
# ---------------------------------------------------------------------------


def test_frames_are_gapless_and_name_the_execution_that_produced_them(app_module):
    """A client can order, deduplicate and attribute every frame it reads."""
    channel_id = _channel("frames")
    minted = {}

    async def work(stream):
        await stream.emit("trace", {"direction": "agent_to_workflow",
                                    "raw_command": "add_two_numbers"})
        # The workflow mints its logical turn key partway through the turn,
        # which is exactly why identity is resolved per frame.
        minted["key"] = "logical-turn-key"
        await stream.emit("trace", {"direction": "workflow_to_agent",
                                    "command_name": "add_two_numbers",
                                    "success": True})
        output = _completed_output()
        await stream.emit("output", output.model_dump(mode="json"))
        return output

    stream = TurnStreamChannel()
    execn, frames = asyncio.run(
        _own_a_streaming_turn(
            app_module, channel_id, work, stream,
            logical_resolver=lambda: minted.get("key"),
        )
    )

    assert [frame["seq"] for frame in frames] == [0, 1, 2]
    assert [frame["type"] for frame in frames] == ["trace", "trace", "output"]
    assert {frame["turn_key"] for frame in frames} == {execn.turn_key}
    # Identity appears as soon as it exists and is never back-dated onto a
    # frame that went out before the key was minted.
    assert frames[0]["logical_turn_key"] is None
    assert frames[1]["logical_turn_key"] == "logical-turn-key"
    assert sum(frame["type"] in ("output", "error") for frame in frames) == 1
    assert execn.error is None
    assert not app_module.turn_registry.has_active(channel_id)


def test_nothing_is_emitted_after_the_terminal_frame():
    """The drain loop stops at the sentinel, so a late emit is dropped, not lost."""
    stream = TurnStreamChannel()
    stream.bind("exec-1")
    assert stream.emit_nowait("trace", {"a": 1})["seq"] == 0
    stream.close()
    assert stream.emit_nowait("output", {"late": True}) is None
    stream.close()  # idempotent: the guarantee path calls it blind

    frames = asyncio.run(_drain(stream))
    assert [frame["type"] for frame in frames] == ["trace"]


def test_a_stopped_consumer_neither_cancels_nor_reruns_the_turn(app_module):
    """Ownership is not delivery: nobody is reading, and the turn still finishes.

    This is the disconnect case. The work runs exactly once, its result is the
    authoritative one, and the frames the absent consumer never read are simply
    left in the queue — they are never re-sent to anybody.
    """
    channel_id = _channel("nobodyreads")
    runs = []

    async def work(stream):
        runs.append(1)
        await stream.emit("trace", {"direction": "agent_to_workflow",
                                    "raw_command": "x"})
        output = _completed_output()
        await stream.emit("output", output.model_dump(mode="json"))
        return output

    async def body():
        runtime = await _runtime_for(app_module, channel_id)
        registry = app_module.turn_registry
        stream = TurnStreamChannel()

        async def owned(execn):
            stream.bind(execn.turn_key)
            try:
                await app_module.run_owned_turn(
                    runtime, registry, execn, lambda: work(stream),
                    app_module.session_manager,
                )
            finally:
                stream.close()

        execn = await registry.start_or_get_active(
            channel_id,
            kind="invoke_agent_stream",
            idempotency_key="nobody-reads",
            run_turn=lambda e: asyncio.create_task(owned(e)),
        )
        await execn.done_event.wait()
        return execn, await _drain(stream)

    execn, frames = asyncio.run(body())

    assert runs == [1], "the work ran a second time for a consumer that left"
    assert execn.result is not None and execn.error is None
    assert [frame["seq"] for frame in frames] == [0, 1]
    assert not app_module.turn_registry.has_active(channel_id)


def test_two_channels_streams_stay_isolated(app_module):
    """Concurrent channels each own their frames, their seq and their key."""
    first, second = _channel("iso_a"), _channel("iso_b")

    def make_work(marker):
        async def work(stream):
            await stream.emit("trace", {"raw_command": marker})
            output = _completed_output()
            await stream.emit("output", output.model_dump(mode="json"))
            return output
        return work

    async def body():
        stream_a, stream_b = TurnStreamChannel(), TurnStreamChannel()
        a, b = await asyncio.gather(
            _own_a_streaming_turn(app_module, first, make_work("A"), stream_a),
            _own_a_streaming_turn(app_module, second, make_work("B"), stream_b),
        )
        return a, b

    (execn_a, frames_a), (execn_b, frames_b) = asyncio.run(body())

    assert execn_a.turn_key != execn_b.turn_key
    assert [f["seq"] for f in frames_a] == [0, 1] == [f["seq"] for f in frames_b]
    assert {f["turn_key"] for f in frames_a} == {execn_a.turn_key}
    assert {f["turn_key"] for f in frames_b} == {execn_b.turn_key}
    assert frames_a[0]["data"]["raw_command"] == "A"
    assert frames_b[0]["data"]["raw_command"] == "B"


def test_a_real_workflow_turn_streams_its_interactions_before_it_finishes(
    app_module,
):
    """End to end over a real workflow: hello_world, its real CME pipeline, the
    real trace queue, and the same streaming_work the endpoint runs.

    Deterministic ("/") so no model is called — the trace queue is written by
    ``_process_message``, which is the same public agent/workflow exchange the
    agent path reports. What is proven here is that real interactions reach the
    stream BEFORE the turn's output, and that the output the stream carries is
    the turn's own authoritative result.
    """
    from fastworkflow.run_fastapi_mcp.utils import (
        run_process_message_with_trace_stream,
    )

    channel_id = _channel("realturn")
    stream = TurnStreamChannel()

    async def work(channel):
        async def on_trace(trace_json: dict) -> None:
            await channel.emit("trace", trace_json)

        turn_output = await run_process_message_with_trace_stream(
            runtime_holder["runtime"], "/what_can_i_do", 60,
            app_module.session_manager, on_trace,
        )
        await channel.emit("output", turn_output.model_dump(mode="json"))
        return turn_output

    runtime_holder = {}

    async def body():
        runtime_holder["runtime"] = await _runtime_for(app_module, channel_id)
        return await _own_a_streaming_turn(app_module, channel_id, work, stream)

    execn, frames = asyncio.run(body())

    kinds = [frame["type"] for frame in frames]
    assert kinds[-1] == "output", kinds
    traces = [frame for frame in frames if frame["type"] == "trace"]
    assert len(traces) >= 2, "the workflow reported no interaction at all"
    # An interaction is delivered before the answer: that is the whole point.
    assert kinds.index("trace") < kinds.index("output")
    assert traces[0]["data"]["direction"] == "agent_to_workflow"
    assert traces[0]["data"]["raw_command"] == "/what_can_i_do"
    assert traces[1]["data"]["direction"] == "workflow_to_agent"
    assert traces[1]["data"]["success"] is True

    # Only the public exchange crosses the wire. A field outside this set would
    # be a new disclosure, and the chat panel renders whatever arrives.
    assert set(traces[0]["data"]) == {
        "direction", "raw_command", "command_name", "parameters",
        "response_text", "success", "timestamp_ms",
    }

    # The streamed output IS the execution's result, and the logical key the
    # frames carry is the one a client would poll or replay.
    output = frames[-1]["data"]
    assert output["status"] == "completed"
    assert output["turn_key"] == execn.result.turn_key
    assert frames[-1]["logical_turn_key"] == execn.result.turn_key
    assert output["answer"] and "hello_world_workflow" in output["answer"]


# ---------------------------------------------------------------------------
# The real route, end to end
# ---------------------------------------------------------------------------


def test_the_real_endpoint_streams_interactions_then_one_output(app_module):
    """A real POST to /invoke_agent_stream over a real workflow.

    Everything in the path is the shipped code: session auth, the registry, the
    stream channel, the encoder, the HTTP body. What comes back is what a
    client actually gets.
    """
    channel_id = _channel("route")
    client, headers, resp = _stream_a_real_deterministic_turn(app_module, channel_id)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    execution_key = resp.headers[STREAM_TURN_KEY_HEADER]
    assert resp.headers[STREAM_FORMAT_HEADER] == "ndjson"

    frames = _parse_ndjson(resp.text)
    kinds = [f["type"] for f in frames]
    assert kinds[-1] == "output", kinds
    assert kinds.count("output") + kinds.count("error") == 1
    assert kinds.index("trace") < kinds.index("output")
    assert [f["seq"] for f in frames] == list(range(len(frames)))
    assert {f["turn_key"] for f in frames} == {execution_key}

    traces = [f for f in frames if f["type"] == "trace"]
    assert traces[0]["data"]["direction"] == "agent_to_workflow"
    assert traces[0]["data"]["raw_command"] == "what can i do"
    assert traces[-1]["data"]["direction"] == "workflow_to_agent"

    output = frames[-1]["data"]
    assert output["status"] == "completed" and output["success"] is True
    assert "hello_world_workflow" in output["answer"]

    # Both handles reach the finished turn. The LOGICAL key is on every frame
    # and is what the store is keyed by; the EXECUTION key from the header
    # keeps working after the execution retires because the registry keeps
    # the mapping for a bounded window, which is what a client that lost its
    # body before the first frame has to recover with.
    assert frames[0]["logical_turn_key"] == output["turn_key"]
    by_execution_key = client.get(f"/turns/{execution_key}", headers=headers)
    assert by_execution_key.status_code == 200
    assert by_execution_key.json()["answer"] == output["answer"]

    stored = client.get(f"/turns/{output['turn_key']}", headers=headers)
    assert stored.status_code == 200
    assert stored.json()["answer"] == output["answer"]

    replay = client.get(f"/turns/{output['turn_key']}/trace", headers=headers)
    assert replay.status_code == 200
    assert replay.json()["spans"], "the missed interactions must be replayable"


def test_the_real_endpoint_honours_an_sse_session(app_module):
    """A session initialized as SSE gets SSE framing, not a silent downgrade.

    The payloads are identical to NDJSON's `data`; only the framing differs,
    with seq as the event id and the turn identity on the response head.
    """
    channel_id = _channel("routesse")
    _client, _headers, resp = _stream_a_real_deterministic_turn(
        app_module, channel_id, stream_format="sse"
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers[STREAM_FORMAT_HEADER] == "sse"
    assert resp.headers[STREAM_TURN_KEY_HEADER]

    frames = _parse_sse(resp.text)
    assert [f["seq"] for f in frames] == list(range(len(frames)))
    assert [f["type"] for f in frames][-1] == "output"
    assert frames[0]["data"]["direction"] == "agent_to_workflow"
    assert "hello_world_workflow" in frames[-1]["data"]["answer"]
    # Payload-only data: identity is not duplicated into the SSE payload.
    assert "turn_key" not in frames[0]["data"]


def test_a_missed_delivery_deadline_does_not_end_the_stream(app_module):
    """The deadline governs delivery, not ownership.

    run_process_message_with_trace_stream reports the deadline and keeps
    owning the executor, so the turn's remaining interactions and its output
    still arrive. Reporting that as `error` (which it did) both ended the
    stream by contract and told the client the turn had failed.
    """
    channel_id = _channel("deadline")
    _client, _headers, resp = _stream_a_real_deterministic_turn(
        app_module, channel_id, timeout_seconds=0
    )

    assert resp.status_code == 200
    frames = _parse_ndjson(resp.text)
    kinds = [f["type"] for f in frames]
    assert "timeout" in kinds, kinds
    timeout_frame = frames[kinds.index("timeout")]
    assert timeout_frame["data"]["still_running"] is True
    assert timeout_frame["data"]["timeout_seconds"] == 0
    assert "timed out" in timeout_frame["data"]["detail"]

    # Non-terminal in both senses: frames follow it, and the turn still ends
    # with its real answer rather than an error.
    assert kinds.index("timeout") < len(kinds) - 1
    assert kinds[-1] == "output"
    assert kinds.count("error") == 0
    assert frames[-1]["data"]["success"] is True
    assert "hello_world_workflow" in frames[-1]["data"]["answer"]


def test_a_body_dropped_before_the_first_frame_recovers_by_its_header_key(
    app_module,
):
    """The disconnect the header exists for, and used to lose.

    A client reads X-FW-Turn-Key before any frame, then loses the body. The
    execution key it holds is not something the store knows — the store is
    keyed by the logical key — and a chat execution is dropped from the
    registry the moment it retires. So the answer the turn went on to produce
    used to be unreachable by the only handle that client had, which is an
    invitation to submit the query a second time.

    Here the body is abandoned before the first frame is read, the turn is
    allowed to finish on its own, and the header key is polled afterwards.
    """
    channel_id = _channel("dropped")

    with TestClient(app_module.app) as client:
        init = client.post("/initialize", json={"channel_id": channel_id})
        assert init.status_code == 200
        headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

        async def make_deterministic():
            runtime = await app_module.session_manager.get_session(channel_id)
            runtime.execution_context._run_as_agent = False

        asyncio.run(make_deterministic())

        with client.stream(
            "POST",
            "/invoke_agent_stream",
            headers=headers,
            json={"user_query": "what can i do", "timeout_seconds": 30},
        ) as response:
            assert response.status_code == 200
            execution_key = response.headers[STREAM_TURN_KEY_HEADER]
            # Leave without reading a single frame.

        # The turn owns itself: it finishes whether or not anyone is reading.
        for _ in range(200):
            if not app_module.turn_registry.has_active(channel_id):
                break
            client.get("/probes/healthz")   # let the server's loop run
        assert not app_module.turn_registry.has_active(channel_id)

        recovered = client.get(f"/turns/{execution_key}", headers=headers)
        assert recovered.status_code == 200, recovered.text
        body = recovered.json()
        assert body["status"] == "completed"
        assert "hello_world_workflow" in body["answer"]
        logical_key = body.get("logical_turn_key") or body["turn_key"]

        # And the interactions the dead body never delivered are replayable
        # through the same handle.
        replay = client.get(f"/turns/{execution_key}/trace", headers=headers)
        assert replay.status_code == 200
        assert replay.json()["spans"]

    # Recovery read the one turn that ran; it did not cause another.
    store = app_module._observability_store()
    rows = store.list_turns(channel_id=channel_id)
    assert len(rows) == 1, rows
    assert rows[0]["turn_key"] == logical_key


def test_another_channel_cannot_recover_someone_elses_retired_turn(app_module):
    """The recovery window is not a hole in [A39].

    A retired execution key resolves for the channel that owns it and for
    nobody else — the same answer an unknown key gets, over real HTTP with a
    real second session's token.
    """
    owner = _channel("owner")
    client, headers, resp = _stream_a_real_deterministic_turn(app_module, owner)
    assert resp.status_code == 200
    execution_key = resp.headers[STREAM_TURN_KEY_HEADER]
    logical_key = _parse_ndjson(resp.text)[-1]["data"]["turn_key"]
    assert client.get(f"/turns/{execution_key}", headers=headers).status_code == 200

    intruder = _channel("intruder")
    other = TestClient(app_module.app)
    init = other.post("/initialize", json={"channel_id": intruder})
    assert init.status_code == 200
    other_headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    for key in (execution_key, logical_key):
        stolen = other.get(f"/turns/{key}", headers=other_headers)
        assert stolen.status_code == 404, (key, stolen.text)
        stolen_trace = other.get(f"/turns/{key}/trace", headers=other_headers)
        assert stolen_trace.status_code == 404, (key, stolen_trace.text)


def test_a_retired_key_alias_is_bounded_and_scoped_to_its_channel(app_module):
    """The alias is a recovery window, not a second index of turns.

    It answers only the channel that owns the turn (same rule as a live
    lookup), and it expires — a client that has not recovered within the
    retention window is not owed an in-memory record forever.
    """
    from fastworkflow.run_fastapi_mcp import turns as turns_mod

    registry = turns_mod.TurnRegistry(retention_seconds=0.0)
    execn = turns_mod.TurnExecution(
        turn_key="exec-1",
        channel_id="chan-a",
        kind="invoke_agent_stream",
        idempotency_key="k",
    )
    execn.logical_turn_key = "logical-1"
    execn.exec_state = turns_mod.ExecState.DONE
    registry._by_key[execn.turn_key] = execn
    registry._active_by_channel["chan-a"] = execn.turn_key

    asyncio.run(registry.clear_active("chan-a", "exec-1"))

    assert registry.get_by_key_or_logical("exec-1") is None
    assert registry.resolve_retired_logical_key("exec-1", "chan-b") is None
    # retention_seconds=0 makes the window already over, which is what a late
    # recovery attempt meets.
    assert registry.resolve_retired_logical_key("exec-1", "chan-a") is None
    assert "exec-1" not in registry._retired_aliases

    live = turns_mod.TurnRegistry(retention_seconds=300.0)
    execn.ttl_expires_at = None
    live._by_key["exec-2"] = execn
    execn.turn_key = "exec-2"
    live._active_by_channel["chan-a"] = "exec-2"
    asyncio.run(live.clear_active("chan-a", "exec-2"))
    assert live.resolve_retired_logical_key("exec-2", "chan-a") == "logical-1"
    assert live.resolve_retired_logical_key("exec-2", "chan-b") is None


def test_a_failure_recorded_after_the_output_is_not_a_second_ending(app_module):
    """Exactly one terminal frame, through the real owned-turn lifecycle.

    run_owned_turn's post-work bookkeeping (the turn count, the conversation
    window) shares the try that sets ``execn.error``, so a failure there lands
    AFTER the work has already emitted its output frame. The endpoint's
    completion callback would then have delivered a second, contradicting
    ending. Here the work emits its answer and the turn then fails for real:
    the client keeps the answer, and the failure stays in the execution's
    record, which is where a recovering client reads the authoritative
    account anyway.
    """
    channel_id = _channel("lateboom")
    stream = TurnStreamChannel()

    async def work(channel):
        await channel.emit("trace", {"direction": "agent_to_workflow",
                                     "raw_command": "add 2 and 3"})
        await channel.emit("output", _completed_output().model_dump(mode="json"))
        raise RuntimeError("bookkeeping after the answer blew up")

    async def drive():
        runtime = await _runtime_for(app_module, channel_id)
        registry = app_module.turn_registry

        async def owned(execn):
            stream.bind(execn.turn_key)

            async def finish_stream() -> None:
                # The endpoint's completion callback, verbatim in shape.
                if execn.error is not None and not stream.terminated:
                    await stream.emit("error", {"detail": "Internal error"})
                stream.close()

            try:
                await app_module.run_owned_turn(
                    runtime, registry, execn, lambda: work(stream),
                    app_module.session_manager, on_done=finish_stream,
                )
            finally:
                stream.close()

        execn = await registry.start_or_get_active(
            channel_id,
            kind="invoke_agent_stream",
            idempotency_key=f"idem-{channel_id}",
            run_turn=lambda e: asyncio.create_task(owned(e)),
        )
        frames = await _drain(stream)
        await execn.done_event.wait()
        return execn, frames

    execn, frames = asyncio.run(drive())

    assert execn.error == "bookkeeping after the answer blew up"
    assert [f["type"] for f in frames] == ["trace", "output"]
    assert stream.terminated
    # And the channel refuses the late ending rather than relying on the
    # caller to remember not to send one.
    assert stream.emit_nowait("error", {"detail": "too late"}) is None


# ---------------------------------------------------------------------------
# Wire encodings
# ---------------------------------------------------------------------------


def test_ndjson_frames_carry_the_whole_envelope():
    frame = {"type": "trace", "seq": 3, "turn_key": "exec-1",
             "logical_turn_key": "log-1", "data": {"raw_command": "hi"}}
    line = encode_stream_frame(frame, "ndjson")
    assert line.endswith("\n")
    assert json.loads(line) == frame


def test_sse_frames_keep_their_documented_shape_and_carry_seq_as_the_id():
    frame = {"type": "output", "seq": 7, "turn_key": "exec-1",
             "logical_turn_key": "log-1", "data": {"answer": "5"}}
    text = encode_stream_frame(frame, "sse")
    lines = text.split("\n")
    assert lines[0] == "id: 7"
    assert lines[1] == "event: output"
    # `data` stays the payload alone: an SSE reader gets its turn identity from
    # the X-FW-Turn-Key response header, not from a changed payload shape.
    assert json.loads(lines[2][len("data: "):]) == {"answer": "5"}
    assert text.endswith("\n\n")


# ---------------------------------------------------------------------------
# Admission answers: a retry must never become a second turn
# ---------------------------------------------------------------------------


def _initialize(client: TestClient, channel_id: str):
    resp = client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _occupy(app_module, channel_id: str, idempotency_key: str, kind: str):
    """Leave one live execution on the channel (nothing will finish it)."""
    async def body():
        await _runtime_for(app_module, channel_id)
        return await app_module.turn_registry.start_or_get_active(
            channel_id,
            kind=kind,
            idempotency_key=idempotency_key,
            run_turn=lambda e: asyncio.create_task(asyncio.sleep(30)),
        )

    return asyncio.run(body())


def test_a_duplicate_submission_rejoins_the_live_turn_instead_of_running_twice(
    app_module,
):
    """The hang this replaces: a retried identical query used to be handed a
    body nothing would ever write to, because the registry deduped it onto the
    running execution and the factory that owns the frames never ran.

    It is now answered the deferred shape every other turn endpoint answers,
    carrying the key of the turn that IS running — so the caller polls it
    instead of submitting the command a second time.
    """
    channel_id = _channel("dupe")
    query = "add 2 and 3"
    client = TestClient(app_module.app)
    headers = _initialize(client, channel_id)
    running = _occupy(
        app_module,
        channel_id,
        compute_idempotency_key(channel_id, "invoke_agent_stream", query),
        "invoke_agent_stream",
    )

    resp = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": query, "timeout_seconds": 5},
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["turn_key"] == running.turn_key
    assert body["exec_state"] == "running"
    assert body["reason"] == "duplicate_submission"
    assert resp.headers[STREAM_TURN_KEY_HEADER] == running.turn_key
    # One execution on the channel, still the original one.
    assert app_module.turn_registry.active_turn_key(channel_id) == running.turn_key


def test_a_different_query_on_a_busy_channel_is_refused_with_the_active_key(
    app_module,
):
    """409 stays 409 — but it now says WHICH turn, so a client can watch it
    rather than guess or retry blindly."""
    channel_id = _channel("busy")
    client = TestClient(app_module.app)
    # Tokens first: /initialize reports a channel with a live execution as 202.
    headers = _initialize(client, channel_id)
    running = _occupy(app_module, channel_id, "occupier", "invoke_agent")

    resp = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": "something else entirely", "timeout_seconds": 5},
    )

    assert resp.status_code == 409
    body = resp.json()
    assert "already in progress" in body["detail"]
    assert body["reason"] == "channel_busy"
    assert body["turn_key"] == running.turn_key


def test_the_stream_identity_headers_are_readable_cross_origin(app_module):
    """The chatbot page is served from its own loopback origin, so a header the
    CORS layer does not expose is a header the browser hides — and the client
    would have nothing but the query to recover with."""
    from fastapi.middleware.cors import CORSMiddleware

    exposed = None
    for middleware in app_module.app.user_middleware:
        if middleware.cls is CORSMiddleware:
            exposed = middleware.kwargs.get("expose_headers")
    assert exposed is not None, "CORSMiddleware is not installed"
    assert STREAM_TURN_KEY_HEADER in exposed
    assert STREAM_FORMAT_HEADER in exposed


# ---------------------------------------------------------------------------
# fix-9eg.20.2 / fix-9eg.20.3 — what the chat page ships
# ---------------------------------------------------------------------------


def _spa_bytes() -> bytes:
    return run_chatbot_server.load_index_html()


class TestWhatThePageShips:
    """Properties of the shipped file itself.

    Deliberately only two, and only ones that ARE facts about the artifact
    rather than restatements of its source. Everything the page *does* —
    parsing both framings, deduplicating, reporting a missed deadline,
    recovering by reading rather than resubmitting, rendering artifacts beside
    the answer — is asserted by running that code in a DOM below, because a
    test that greps for the function it is testing passes whether or not the
    function works.
    """

    def test_the_page_never_assigns_untrusted_markup(self):
        # Artifacts and traces carry workflow output; the page builds nodes and
        # sets textContent, and HTML only ever reaches a sandboxed frame.
        assert b"innerHTML" not in _spa_bytes()

    def test_both_turn_endpoints_stay_wired(self):
        page = _spa_bytes()
        # Agent turns stream; deterministic "/" commands keep the
        # non-streaming endpoint, which has no stream to consume.
        assert b"/invoke_agent_stream" in page
        assert b"/invoke_assistant" in page


@pytest.fixture
def real_stream_bodies(app_module, tmp_path):
    """Bodies captured from the real endpoint, for the browser to parse.

    Three real turns over tests/hello_world_workflow: the same turn in each
    framing, and one whose delivery deadline expires mid-turn. Recording them
    here and replaying them in the DOM is what lets the page's reader be
    tested against the server's actual bytes without the browser test needing
    a FastAPI server of its own.
    """
    bodies = {}
    for name, kwargs in (
        ("ndjson", {}),
        ("sse", {"stream_format": "sse"}),
        ("ndjson_timeout", {"timeout_seconds": 0}),
    ):
        _client, _headers, resp = _stream_a_real_deterministic_turn(
            app_module, _channel(f"dom_{name}"), **kwargs
        )
        assert resp.status_code == 200, resp.text
        bodies[name] = resp.text

    path = tmp_path / "real_stream_bodies.json"
    path.write_text(json.dumps(bodies))
    return path


def _run_dom(script_name: str, *args, timeout: int = 60):
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name(script_name)
    result = subprocess.run(
        ["node", str(script), jsdom_root, *[str(a) for a in args]],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_reads_real_server_frames_in_a_real_dom(
    real_stream_bodies, tmp_path
):
    """The page's reader, on the endpoint's own bytes, in a browser.

    Covers what the source-string assertions used to claim: partial frames in
    BOTH framings (chunked at boundaries that fall inside frames), duplicate
    frames dropped by seq, a missed delivery deadline shown as "still
    working" with the answer that followed it still rendered.
    """
    page = tmp_path / "index.html"
    page.write_bytes(_spa_bytes())
    _run_dom("chatbot_stream_frames_dom.cjs", page, real_stream_bodies)


def test_chat_activity_and_artifacts_in_a_real_dom(chatbot_server):
    """The page's own code, in a browser DOM, against a real server.

    Every artifact body is fetched over HTTP from the seeded store, and the
    recovery path is driven against a server that genuinely does not have the
    turn; nothing is stubbed.
    """
    _run_dom(
        "chatbot_chat_stream_dom.cjs",
        f"http://127.0.0.1:{chatbot_server.port}/?token={chatbot_server.token}",
        chatbot_fixtures.TURN1,
        chatbot_fixtures.HTML_ARTIFACT_ID,
        chatbot_fixtures.TEXT_ARTIFACT_ID,
    )
