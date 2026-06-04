"""FastAPI Topology-B session store and cancel_pending integration tests."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def hello_world_workflow_path():
    import fastworkflow

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
def app_module(hello_world_workflow_path, env_files):
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
    import fastworkflow
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return main


def _initialize(client: TestClient, channel_id: str) -> dict:
    resp = client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200
    return resp.json()


def _authorize(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_cancel_pending_endpoint(app_module):
    channel_id = f"cancel_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    init = _initialize(client, channel_id)
    headers = _authorize(init["access_token"])

    async def setup_pending():
        runtime = await app_module.session_manager.get_session(channel_id)
        assert runtime is not None
        runtime.execution_context._awaiting_user = True
        runtime.execution_context._pending_clarification_request = "clarify?"
        app_module.session_manager.session_state_store.save(
            channel_id,
            runtime.execution_context.serialize_state(channel_id=channel_id),
        )

    asyncio.run(setup_pending())

    resp = client.post("/cancel_pending", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert not app_module.session_manager.session_state_store.exists(channel_id)


def test_cold_rehydrate_after_cache_evict(app_module):
    channel_id = f"rehydrate_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    _initialize(client, channel_id)

    async def setup_and_check():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        runtime = await app_module.session_manager.get_session(channel_id)
        ctx = runtime.execution_context
        ctx._awaiting_user = True
        ctx._suspended_user_message = "original"
        ctx._pending_clarification_request = "Pick one?"
        app_module.session_manager.session_state_store.save(
            channel_id, ctx.serialize_state(channel_id=channel_id)
        )
        await app_module.session_manager.evict_live_session(channel_id)

        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        runtime2 = await app_module.session_manager.get_session(channel_id)
        assert runtime2.execution_context.awaiting_user
        assert runtime2.execution_context._pending_clarification_request == "Pick one?"

    asyncio.run(setup_and_check())
