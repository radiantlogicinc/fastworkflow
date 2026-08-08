"""Integration tests for request-scoped credential handling (Release B, step 13).

The caller's JWT used to be written into shared workflow state by the dependency
that authenticated the request — before the turn was admitted, and on every
lookup. Two requests on one channel therefore interleaved: turn A running,
request B writes token B, B is rejected with 409, and A reads B's credential.
It was also durable-shaped state, so once checkpointing writes `workflow.context`
the credential would be written to disk in the clear.

It is now carried on the execution and installed only inside the accepted turn,
which preserves the documented read contract without either hazard.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest

import fastworkflow
from fastworkflow.run_fastapi_mcp.turns import (
    CREDENTIAL_CONTEXT_KEY,
    installed_credential,
)


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
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return main


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def _runtime(app_module, channel_id):
    await app_module.ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=app_module.session_manager,
        workflow_path=app_module.ARGS.workflow_path,
        run_startup=False,
    )
    return await app_module.session_manager.get_session(channel_id)


def test_a_credential_is_absent_outside_a_turn(app_module):
    """Session creation must not put a request-scoped secret into shared state."""
    channel_id = _channel("cred")

    async def body():
        runtime = await _runtime(app_module, channel_id)
        return runtime.execution_context.app_workflow.context

    context = asyncio.run(body())

    assert CREDENTIAL_CONTEXT_KEY not in context


def test_the_credential_is_readable_during_its_own_turn_and_gone_after(app_module):
    """The documented read contract survives; its lifetime is the turn."""
    channel_id = _channel("scope")
    seen = {}

    async def body():
        runtime = await _runtime(app_module, channel_id)
        context = runtime.execution_context.app_workflow.context

        with installed_credential(runtime, "token-A"):
            seen["during"] = context.get(CREDENTIAL_CONTEXT_KEY)
        seen["after"] = context.get(CREDENTIAL_CONTEXT_KEY)
        seen["key_present_after"] = CREDENTIAL_CONTEXT_KEY in context

    asyncio.run(body())

    assert seen["during"] == "token-A"
    assert seen["after"] is None
    assert not seen["key_present_after"]


def test_a_rejected_request_cannot_overwrite_a_running_turns_credential(app_module):
    """The barrier: B is rejected with 409 while A is running, and A still sees token A.

    This is the interleaving the old pre-admission write produced, and the reason
    the credential now travels on the execution rather than being written by the
    dependency that authenticated the request.
    """
    channel_id = _channel("barrier")
    observed = {}

    async def body():
        runtime = await _runtime(app_module, channel_id)
        registry = app_module.turn_registry
        context = runtime.execution_context.app_workflow.context

        a_started = asyncio.Event()
        a_may_finish = asyncio.Event()

        async def work_a():
            a_started.set()
            await a_may_finish.wait()
            observed["a_reads"] = context.get(CREDENTIAL_CONTEXT_KEY)
            return fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=fastworkflow.TurnStatus.COMPLETED,
            )

        execn_a = await registry.start_or_get_active(
            channel_id,
            kind="invoke_agent",
            idempotency_key="turn-a",
            http_bearer_token="token-A",
            run_turn=lambda e: asyncio.create_task(
                app_module.run_owned_turn(
                    runtime, registry, e, work_a, app_module.session_manager
                )
            ),
        )
        await a_started.wait()

        # Request B authenticates with its own token and is refused admission.
        with pytest.raises(app_module.ChannelBusyError):
            await registry.start_or_get_active(
                channel_id,
                kind="invoke_agent",
                idempotency_key="turn-b",
                http_bearer_token="token-B",
                run_turn=lambda e: asyncio.create_task(asyncio.sleep(0)),
            )

        observed["during_b_rejection"] = context.get(CREDENTIAL_CONTEXT_KEY)
        a_may_finish.set()
        await execn_a.done_event.wait()
        observed["after"] = context.get(CREDENTIAL_CONTEXT_KEY)
        observed["error"] = execn_a.error

    asyncio.run(body())

    assert observed["error"] is None
    assert observed["during_b_rejection"] == "token-A"
    assert observed["a_reads"] == "token-A", "a rejected request leaked its credential"
    assert observed["after"] is None


def test_a_previous_value_is_restored_rather_than_dropped(app_module):
    """Nested installation must not clear a value the outer scope owns."""
    channel_id = _channel("nested")
    seen = {}

    async def body():
        runtime = await _runtime(app_module, channel_id)
        context = runtime.execution_context.app_workflow.context

        with installed_credential(runtime, "outer"):
            with installed_credential(runtime, "inner"):
                seen["inner"] = context.get(CREDENTIAL_CONTEXT_KEY)
            seen["restored"] = context.get(CREDENTIAL_CONTEXT_KEY)

    asyncio.run(body())

    assert seen["inner"] == "inner"
    assert seen["restored"] == "outer"


def test_the_credential_survives_a_failing_turn_without_leaking(app_module):
    """An exception inside the turn must not leave the secret installed."""
    channel_id = _channel("boom")
    seen = {}

    async def body():
        runtime = await _runtime(app_module, channel_id)
        context = runtime.execution_context.app_workflow.context

        with pytest.raises(RuntimeError):
            with installed_credential(runtime, "token-A"):
                raise RuntimeError("command blew up")
        seen["after"] = CREDENTIAL_CONTEXT_KEY in context

    asyncio.run(body())

    assert not seen["after"]
