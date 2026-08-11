"""Integration tests for streaming as a registered turn (Release B, step 7).

`/invoke_agent_stream` used to run an entire turn outside the turn lifecycle: no
`TurnExecution`, a `runtime.lock.locked()` busy check of its own, and a 504 that
abandoned the executor future without awaiting it. Three consequences the design
enumerates — an unrelated query answered as somebody else's clarification, a
context snapshotted and closed while a detached thread still mutates it, and a
runtime evicted between response construction and first body iteration.

It is now admitted through the registry like every other turn, which is what
makes it visible to the 409 guard, to eviction, and to the shutdown drain.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

import fastworkflow


@pytest.fixture
def hello_world_workflow_path():
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    return workflow_path


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(hello_world_workflow_path, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        hello_world_workflow_path,
        "--env_file_path",
        env_file,
        "--passwords_file_path",
        passwords_file,
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


def _initialize(client: TestClient, channel_id: str, stream_format: str = "ndjson"):
    resp = client.post(
        "/initialize",
        json={"channel_id": channel_id, "stream_format": stream_format},
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_a_streaming_turn_completes_and_retires_itself(app_module):
    """The happy path still streams, and the execution ends up terminal."""
    channel_id = _channel("stream")

    with TestClient(app_module.app) as client:
        headers = _initialize(client, channel_id)
        resp = client.post(
            "/invoke_agent_stream",
            headers=headers,
            json={"user_query": "add 2 and 3", "timeout_seconds": 60},
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    assert events, "stream produced no events"
    assert {e["type"] for e in events} <= {"trace", "output", "error"}
    assert events[-1]["type"] in ("output", "error")

    # Registered, ran, and cleared its own active pointer.
    assert not app_module.turn_registry.has_active(channel_id)


def test_a_streaming_turn_is_visible_to_the_registry_while_it_runs(app_module):
    """Ownership, not delivery: the 409 guard and eviction both read this pointer."""
    channel_id = _channel("visible")
    seen = {}

    async def body():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        registry = app_module.turn_registry
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            started.set()
            await release.wait()
            return fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=fastworkflow.TurnStatus.COMPLETED,
            )

        runtime = await app_module.session_manager.get_session(channel_id)
        execn = await registry.start_or_get_active(
            channel_id,
            kind="invoke_agent_stream",
            idempotency_key="stream-visible",
            run_turn=lambda e: asyncio.create_task(
                app_module.run_owned_turn(
                    runtime, registry, e, work, app_module.session_manager
                )
            ),
        )

        await started.wait()
        seen["active_during"] = registry.has_active(channel_id)
        seen["busy_during"] = channel_id in app_module.session_manager.busy_channel_ids()

        release.set()
        await execn.done_event.wait()
        seen["active_after"] = registry.has_active(channel_id)
        seen["error"] = execn.error

    asyncio.run(body())

    assert seen["active_during"], "a running stream was invisible to the registry"
    assert seen["busy_during"], "a running stream looked idle to the shutdown drain"
    assert not seen["active_after"]
    assert seen["error"] is None


def test_a_second_stream_on_a_busy_channel_is_rejected_with_409(app_module):
    """Streaming now shares admission control with every other endpoint."""
    channel_id = _channel("busy")

    async def occupy():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        # A live execution nothing will finish, so the channel stays busy.
        await app_module.turn_registry.start_or_get_active(
            channel_id,
            kind="invoke_agent",
            idempotency_key="occupier",
            run_turn=lambda e: asyncio.create_task(asyncio.sleep(30)),
        )

    client = TestClient(app_module.app)
    headers = _initialize(client, channel_id)
    asyncio.run(occupy())

    resp = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": "add 2 and 3", "timeout_seconds": 5},
    )

    assert resp.status_code == 409
    assert "already in progress" in resp.json()["detail"]


def test_closed_admission_refuses_new_turns(app_module):
    """Shutdown closes admission atomically, so nothing registers behind the drain."""
    channel_id = _channel("closed")

    async def body():
        registry = app_module.turn_registry.__class__()
        await registry.close_admission()
        assert registry.admission_closed
        with pytest.raises(app_module.AdmissionClosedError):
            await registry.start_or_get_active(
                channel_id,
                kind="invoke_agent",
                idempotency_key="late",
                run_turn=lambda e: asyncio.create_task(asyncio.sleep(0)),
            )

    asyncio.run(body())


def test_the_delivery_deadline_does_not_abandon_the_executor(app_module):
    """A 504 used to release the lock while the executor thread kept mutating state.

    The deadline now governs delivery only: the client is told, and the turn keeps
    the lock and the registry pointer until the work actually exits.
    """
    channel_id = _channel("deadline")
    observed = {}

    async def body():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        runtime = await app_module.session_manager.get_session(channel_id)
        timeouts = []

        async def on_timeout(detail):
            timeouts.append(detail)
            observed["still_locked_at_timeout"] = runtime.lock.locked()

        async def slow_work():
            async with runtime.lock:
                return await app_module.run_process_message_with_trace_stream(
                    runtime,
                    "add 2 and 3",
                    0,  # deadline already passed on the first poll
                    app_module.session_manager,
                    lambda _t: None,
                    on_timeout=on_timeout,
                )

        output = await slow_work()
        observed["timeouts"] = timeouts
        observed["output_returned"] = output is not None
        observed["lock_free_after"] = not runtime.lock.locked()

    asyncio.run(body())

    assert observed["timeouts"], "the client was never told about the deadline"
    assert observed["still_locked_at_timeout"], "ownership was dropped at the deadline"
    assert observed["output_returned"], "the executor result was abandoned"
    assert observed["lock_free_after"]
