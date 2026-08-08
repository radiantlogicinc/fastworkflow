"""Integration tests for manager-owned eviction leases (Release B, step 6).

`get_session()` hands back a runtime *after* releasing the manager lock. Between
that return and the moment the caller registers a turn or takes `runtime.lock`,
both halves of the busy predicate read false while the runtime is very much in
use — so the LRU could evict and close a runtime a request was working with, and
a cold recreation would then find the old workflow still alive in the weak global
registry and overwrite its context, leaving two runtimes and two locks around one
mutable workflow.

A lease is what closes that window: taken under the same manager lock that hands
out the runtime, released only when the block that uses it ends.

These drive a real `ChannelSessionManager` at a cap of one with real runtimes, so
the over-capacity path actually executes rather than being simulated.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest

import fastworkflow
from fastworkflow.run_fastapi_mcp.utils import ChannelSessionManager


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
        env_file, passwords_file, tmp_path / "speedict"
    )
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def _create(app_module, manager, channel_id, *, startup_action=None):
    await app_module.ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=manager,
        workflow_path=app_module.ARGS.workflow_path,
        startup_action=startup_action,
        run_startup=startup_action is not None,
    )


def _cap_one_manager() -> ChannelSessionManager:
    return ChannelSessionManager(max_live_sessions=1)


def test_a_leased_session_is_not_evicted(app_module):
    """The window the lease exists for: held by a request, invisible to the predicate."""
    manager = _cap_one_manager()
    first, second = _channel("leased"), _channel("newcomer")

    async def body():
        await _create(app_module, manager, first)

        # Nothing is registered and no lock is held, so both halves of the union
        # predicate are false — exactly the state that used to make this a victim.
        async with manager.leased_session(first) as runtime:
            assert runtime is not None
            assert not manager._has_work_in_flight(first)
            await _create(app_module, manager, second)
            return set(manager._sessions)

    live = asyncio.run(body())

    assert first in live, "a leased runtime was evicted out from under its holder"
    assert second in live
    assert len(live) == 2, "cache correctly stayed over its target rather than evicting"


def test_the_lease_releases_and_the_channel_becomes_evictable(app_module):
    """A lease must not pin forever, or it would trade one leak for another."""
    manager = _cap_one_manager()
    first, second = _channel("released"), _channel("after")

    async def body():
        await _create(app_module, manager, first)
        async with manager.leased_session(first):
            pass
        assert first not in manager._leases

        await _create(app_module, manager, second)
        return set(manager._sessions)

    live = asyncio.run(body())

    assert live == {second}


def test_overlapping_leases_are_refcounted(app_module):
    """Two requests can hold the same runtime; the first to finish must not unpin it."""
    manager = _cap_one_manager()
    first, second = _channel("refcount"), _channel("newcomer")

    async def body():
        await _create(app_module, manager, first)

        async with manager.leased_session(first):
            async with manager.leased_session(first):
                assert manager._leases[first] == 2
            # One holder left; still pinned.
            assert manager._leases[first] == 1
            await _create(app_module, manager, second)
            return set(manager._sessions)

    live = asyncio.run(body())

    assert first in live and second in live


def test_a_held_runtime_lock_blocks_eviction_without_a_registry_execution(app_module):
    """The union predicate: /invoke_agent_stream runs a whole turn with no TurnExecution.

    Keyed off the registry pointer alone, that channel reads idle for the entire
    streaming turn and is a valid victim.
    """
    manager = _cap_one_manager()
    streaming, newcomer = _channel("streaming"), _channel("newcomer")

    async def body():
        await _create(app_module, manager, streaming)
        runtime = await manager.get_session(streaming)

        async with runtime.lock:
            assert manager.is_channel_busy is None or not manager.is_channel_busy(streaming)
            assert manager._has_work_in_flight(streaming)
            await _create(app_module, manager, newcomer)
            return set(manager._sessions)

    live = asyncio.run(body())

    assert streaming in live, "a channel mid-turn was evicted because no turn was registered"


@pytest.mark.parametrize("with_startup_action", [False, True])
def test_creation_never_evicts_the_runtime_it_is_creating(app_module, with_startup_action):
    """Design section 10.3: the new runtime is the only apparently safe victim.

    Before its startup runs, a new runtime has no registry pointer, a free lock,
    and no command-context object — so when every older session is pinned, the
    manager's own overflow sweep picks the channel it is in the middle of
    creating. Run with and without a startup action, because the two differ in
    when the pinning context appears.
    """
    manager = _cap_one_manager()
    pinned, newcomer = _channel("pinned"), _channel("newcomer")
    action = (
        fastworkflow.Action(
            command_name="add_two_numbers",
            parameters={"first_num": 1.0, "second_num": 2.0},
        )
        if with_startup_action
        else None
    )

    async def body():
        await _create(app_module, manager, pinned)
        runtime = await manager.get_session(pinned)

        # Hold the only older candidate busy, so the newcomer is the sole
        # apparently-safe victim during its own creation.
        async with runtime.lock:
            await _create(app_module, manager, newcomer, startup_action=action)
            return set(manager._sessions)

    live = asyncio.run(body())

    assert newcomer in live, "create_session evicted the runtime it was creating"
    assert pinned in live


def test_an_initialization_lease_covers_a_channel_with_no_session_yet(app_module):
    """The lease is keyed by channel_id, not by runtime, so it can precede creation."""
    manager = _cap_one_manager()
    channel_id = _channel("initlease")

    async def body():
        async with manager.initialization_lease(channel_id):
            assert manager._leases[channel_id] == 1
            assert channel_id not in manager._sessions
        return manager._leases.get(channel_id)

    assert asyncio.run(body()) is None
