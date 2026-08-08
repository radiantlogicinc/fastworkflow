"""Integration tests for the manager, streaming and shutdown half of design §16.4.

Release B's lifecycle rules only hold if three separate things agree about what
"busy" means: the eviction sweep, the streaming endpoint, and the shutdown drain.
`tests/test_fastapi_session_leases.py` covers the sweep and the lease; this file
covers what those two do not reach — the shutdown drain (§10.5), the streaming
turn's position in the lifecycle after admission and disconnect (§10.4), and the
manager's explicit lifecycle operations and pinned accounting (§10.2, §10.3,
§10.6). The DSPy half of §16.4 lives in `tests/test_server_dspy_memory.py`.

The one that matters most here is §10.5 point 3: a deadline is a bound on
waiting, not a license to write. With Release B persistence in place, snapshotting
a runtime whose queued turn has not run yet and then closing the context under it
makes a stale snapshot authoritative on the next creation — silent state loss that
no later read can detect.

Everything runs against real runtimes, a real `ChannelSessionManager`, a real
`TurnRegistry` and the real `lifespan` shutdown closures. Nothing is mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.run_fastapi_mcp.turns import ExecState, TurnRegistry
from fastworkflow.run_fastapi_mcp.utils import ChannelRuntime, ChannelSessionManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def isolated_env_file(env_files, tmp_path) -> str:
    """A copy of the real env file whose SPEEDDICT_FOLDERNAME is private.

    The override has to live in the *file*, not in `os.environ` and not in a
    later `fastworkflow.init()` call: the lifespan re-runs `init()` from
    `ARGS.env_file_path` on startup, and `get_env_var` reads that mapping before
    it ever looks at the process environment. Anything set another way is
    silently replaced the moment the lifespan starts, and these tests would then
    write channel checkpoints and conversations into the developer's real folder.
    """
    env_file, _ = env_files
    kept = [
        line
        for line in Path(env_file).read_text().splitlines()
        if not line.strip().startswith("SPEEDDICT_FOLDERNAME=")
    ]
    kept.append(f"SPEEDDICT_FOLDERNAME={tmp_path / 'speedict'}")
    isolated = tmp_path / "fastworkflow.env"
    isolated.write_text("\n".join(kept) + "\n")
    return str(isolated)


@pytest.fixture
def app_module(hello_world_workflow_path, env_files, isolated_env_file):
    _, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        hello_world_workflow_path,
        "--env_file_path",
        isolated_env_file,
        "--passwords_file_path",
        passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)

    previous_env = fastworkflow._env_vars
    fastworkflow.init(
        {**dotenv_values(isolated_env_file), **dotenv_values(passwords_file)}
    )
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    # Pin warnings are throttled per (workflow, reason) for the life of the
    # process, so a test that counts them has to start from a clean slate.
    checkpoint.reset_warnings()
    try:
        yield main
    finally:
        checkpoint.reset_warnings()
        if fastworkflow.RoutingRegistry:
            fastworkflow.RoutingRegistry.clear_registry()
        # An interpreter left pointing at a deleted temp directory is a trap for
        # whichever test file runs next.
        fastworkflow.init(previous_env or {})


@pytest.fixture
def fastworkflow_logs(caplog):
    """caplog, wired to the logger that actually emits these records.

    The fastWorkflow logger sets `propagate = False`, so its records never reach
    the root handler pytest installs and a plain `caplog` assertion would pass
    against a message that was never logged.
    """
    fw_logger = logging.getLogger("fastWorkflow")
    caplog.set_level(logging.DEBUG, logger="fastWorkflow")
    fw_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        fw_logger.removeHandler(caplog.handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _cap_one_manager() -> ChannelSessionManager:
    """A manager that is over target the moment a second session appears."""
    return ChannelSessionManager(max_live_sessions=1)


def _add_action() -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name="add_two_numbers",
        parameters={"first_num": 2.0, "second_num": 3.0},
    )


async def _create(
    app_module,
    channel_id: str,
    *,
    manager: Optional[ChannelSessionManager] = None,
) -> ChannelRuntime:
    """Build a runtime exactly as the server's /initialize does, and return it."""
    manager = manager or app_module.session_manager
    await app_module.ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=manager,
        workflow_path=app_module.ARGS.workflow_path,
        run_startup=False,
    )
    return await manager.get_session(channel_id)


async def _queued_turn(
    app_module,
    channel_id: str,
    gate: asyncio.Event,
    *,
    kind: str = "invoke_agent",
    manager: Optional[ChannelSessionManager] = None,
    registry=None,
):
    """Register an execution whose task has not yet reached ``runtime.lock``.

    This is the exact shape §10.5 names as the defect: QUEUED, so the registry
    pointer is live while ``runtime.lock.locked()`` is False. The task parks on
    ``gate`` *before* entering ``run_owned_turn``, so the window can be held open
    for as long as a test needs instead of being raced.
    """
    manager = manager or app_module.session_manager
    registry = registry or app_module.turn_registry
    runtime = await manager.get_session(channel_id)

    async def work():
        return fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=fastworkflow.TurnStatus.COMPLETED,
        )

    async def gated(execn) -> None:
        await gate.wait()
        await app_module.run_owned_turn(runtime, registry, execn, work, manager)

    return await registry.start_or_get_active(
        channel_id,
        kind=kind,
        idempotency_key=f"gated-{channel_id}",
        run_turn=lambda execn: asyncio.create_task(gated(execn)),
    )


def _suspend(runtime: ChannelRuntime) -> None:
    """Put a runtime into the state a restored ask_user session is in.

    A suspended session is pinned (§16.3): its snapshot carries neither the
    logical-turn accumulator nor the CME continuation keys, so retiring one would
    lose the pre-suspension outputs. Driven through the real serialize/apply
    round trip so the flag is set the way the server sets it.
    """
    ctx = runtime.execution_context
    state = ctx.serialize_state(channel_id=runtime.channel_id)
    ctx.apply_serialized_state({**state, "awaiting_user": True})
    assert ctx.awaiting_user


def _is_open(runtime: ChannelRuntime) -> bool:
    """Has this runtime's execution context escaped being closed?

    ``ctx.close()`` evicts the CME workflow from the process registry, so its
    continued presence there is the observable difference between a runtime
    shutdown skipped and one it closed.
    """
    cme_id = runtime.execution_context.cme_workflow.id
    return fastworkflow.Workflow.get_workflow(cme_id) is not None


def _messages(caplog, level: int) -> list[str]:
    """Captured messages logged at exactly ``level``."""
    return [r.getMessage() for r in caplog.records if r.levelno == level]


@dataclass
class _ShutdownSteps:
    """The real lifespan shutdown closures, callable with a test-sized deadline."""

    drain: Callable[..., Any]
    finalize: Callable[[list[str]], Any]
    stop: Callable[[list[str]], Any]


@contextlib.asynccontextmanager
async def running_lifespan(app_module):
    """Start the real app lifespan and expose its three shutdown steps.

    ``wait_for_active_turns_to_complete``, ``finalize_conversations_on_shutdown``
    and ``stop_all_chat_sessions`` are closures inside ``lifespan`` and the drain
    deadline is hardcoded at 30 s at the single call site, so there is no seam to
    reach them through. Reading them off the suspended generator's frame runs the
    production functions unmodified, with a deadline a test can afford; nothing
    is replaced, stubbed or reimplemented.
    """
    cm = app_module.lifespan(app_module.app)
    await cm.__aenter__()
    locals_at_yield = cm.gen.ag_frame.f_locals
    missing = [
        name
        for name in (
            "wait_for_active_turns_to_complete",
            "finalize_conversations_on_shutdown",
            "stop_all_chat_sessions",
        )
        if name not in locals_at_yield
    ]
    assert not missing, (
        f"lifespan no longer defines {missing}; the shutdown sequence was "
        "restructured and these tests need to be pointed at its new shape"
    )
    steps = _ShutdownSteps(
        drain=locals_at_yield["wait_for_active_turns_to_complete"],
        finalize=locals_at_yield["finalize_conversations_on_shutdown"],
        stop=locals_at_yield["stop_all_chat_sessions"],
    )
    try:
        yield steps
    finally:
        # The real shutdown runs on the way out of the lifespan. Leave it nothing
        # to summarize: generate_topic_and_summary is a live LLM call, and it
        # would also make an un-drained gate hold the process for the full 30 s.
        for channel_id in list(app_module.session_manager._sessions):
            await app_module.session_manager.remove_session(channel_id)
        await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Shutdown quiescence (design §10.5, invariant 30)
# ---------------------------------------------------------------------------

def test_shutdown_closes_admission_before_it_starts_draining(app_module):
    """Hazard: a turn registered behind an empty scan is shut down underneath it.

    Without an atomic closed state, the drain's last scan and a submission race:
    the scan sees nothing, returns, and the turn that registered a moment later
    has its context snapshotted and closed while it runs. Closing admission has
    to happen before the first scan and hold for the whole drain, not after the
    last one.
    """
    busy, latecomer = _channel("busy"), _channel("late")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as steps:
            await _create(app_module, busy)
            gate = asyncio.Event()
            execn = await _queued_turn(app_module, busy, gate)

            async def submit_mid_drain():
                # Admission closes as the drain's first act, so observing it
                # closed while a channel is still draining is the ordering claim.
                # Bounded, so an ordering that never closes it fails here instead
                # of hanging the suite.
                deadline = time.monotonic() + 5.0
                while (
                    not app_module.turn_registry.admission_closed
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
                seen["closed_mid_drain"] = app_module.turn_registry.admission_closed
                seen["drain_still_running"] = (
                    busy in app_module.session_manager.busy_channel_ids()
                )
                try:
                    await app_module.turn_registry.start_or_get_active(
                        latecomer,
                        kind="invoke_agent",
                        idempotency_key="latecomer",
                        run_turn=lambda e: asyncio.create_task(asyncio.sleep(0)),
                    )
                    seen["refused"] = False
                except app_module.AdmissionClosedError:
                    seen["refused"] = True
                finally:
                    # Always release, or a failed expectation would turn into the
                    # drain sitting out its whole deadline.
                    gate.set()

            watcher = asyncio.create_task(submit_mid_drain())
            seen["remaining"] = await steps.drain(max_wait_seconds=10)
            await watcher
            await execn.done_event.wait()
            seen["latecomer_registered"] = app_module.turn_registry.has_active(
                latecomer
            )

    asyncio.run(body())

    assert seen["closed_mid_drain"], (
        "admission was still open once the drain had already begun waiting"
    )
    assert seen["drain_still_running"], (
        "the drain had already finished, so nothing about ordering was observed"
    )
    assert seen["refused"], "a turn was admitted after shutdown began draining"
    assert not seen["latecomer_registered"]
    assert seen["remaining"] == [], "the drain did not wait for the queued turn"


def test_a_queued_turn_that_never_took_the_lock_still_counts_as_busy(app_module):
    """Hazard: the drain reads not-busy for work that is registered but unstarted.

    ``_active_turn_channel_ids`` used to test ``rt.lock.locked()`` alone, and a
    QUEUED execution whose task has not been scheduled yet holds no lock. That
    channel reads idle, and shutdown snapshots a context the turn is about to
    mutate. The drain must use the union of the registry pointer and the lock.
    """
    channel_id = _channel("queued")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as steps:
            runtime = await _create(app_module, channel_id)
            gate = asyncio.Event()
            execn = await _queued_turn(app_module, channel_id, gate)

            seen["exec_state"] = execn.exec_state
            # The half of the union that reports wrongly, recorded so a fixture
            # that stopped reproducing the shape fails loudly instead of quietly.
            seen["lock_held"] = runtime.lock.locked()
            seen["registry_active"] = app_module.turn_registry.has_active(channel_id)
            seen["at_deadline"] = await steps.drain(max_wait_seconds=0)

            gate.set()
            await execn.done_event.wait()
            seen["after_completion"] = await steps.drain(max_wait_seconds=0)

    asyncio.run(body())

    assert seen["exec_state"] is ExecState.QUEUED
    assert not seen["lock_held"], "the lock was held, so this is not the queued shape"
    assert seen["registry_active"]
    assert seen["at_deadline"] == [channel_id], (
        "the drain treated a queued turn's channel as quiescent"
    )
    assert seen["after_completion"] == []


def test_a_lease_with_no_execution_and_no_lock_still_counts_as_busy(app_module):
    """Hazard: the interval between handing out a runtime and admitting its turn.

    In that window the request owns the runtime, no execution exists and the lock
    is free — both halves of the union predicate read false. A drain built only
    on the union would finalize and close a runtime a request is holding, which
    is the same detached-runtime failure the lease was introduced for.
    """
    channel_id = _channel("leased")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as steps:
            await _create(app_module, channel_id)

            async with app_module.session_manager.leased_session(channel_id) as runtime:
                seen["lock_held"] = runtime.lock.locked()
                seen["registry_active"] = app_module.turn_registry.has_active(
                    channel_id
                )
                seen["union"] = app_module.session_manager._has_work_in_flight(
                    channel_id
                )
                seen["while_leased"] = await steps.drain(max_wait_seconds=0)

            seen["after_release"] = await steps.drain(max_wait_seconds=0)

    asyncio.run(body())

    assert not seen["lock_held"]
    assert not seen["registry_active"]
    assert not seen["union"], "the union predicate alone should not see this channel"
    assert seen["while_leased"] == [channel_id], (
        "the drain ignored a lease, so a held runtime could be closed at the deadline"
    )
    assert seen["after_release"] == []


def test_a_still_busy_runtime_is_neither_finalized_nor_closed_past_the_deadline(
    app_module, fastworkflow_logs
):
    """Hazard: the deadline is treated as permission to write and close anyway.

    This is the one that loses data. The drain's remaining channels have work
    that has not run: their contexts are about to be mutated. Writing a snapshot
    taken before that mutation and then closing the context underneath it makes
    the stale snapshot authoritative on the next creation, and nothing later can
    tell that the newer state ever existed. After its deadline shutdown may only
    complain — loudly, naming the channels — and leave them to the host.
    """
    busy, quiet = _channel("busy"), _channel("quiet")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as steps:
            busy_runtime = await _create(app_module, busy)
            await _create(app_module, quiet)
            gate = asyncio.Event()
            execn = await _queued_turn(app_module, busy, gate)

            remaining = await steps.drain(max_wait_seconds=0)
            await steps.finalize(remaining)
            await steps.stop(remaining)

            seen["remaining"] = remaining
            seen["busy_open"] = _is_open(busy_runtime)
            output = busy_runtime.execution_context.process_action(_add_action())
            seen["busy_usable"] = bool(output.success)
            # That check appended a conversation turn; drop it so the lifespan's
            # own shutdown has nothing to summarize (an LLM call).
            busy_runtime.execution_context.clear_conversation_history()

            gate.set()
            await execn.done_event.wait()
            seen["turn_error"] = execn.error

    asyncio.run(body())

    assert seen["remaining"] == [busy]
    assert seen["busy_open"], (
        "shutdown closed a context whose queued turn had not run yet"
    )
    assert seen["busy_usable"], "the skipped runtime came out of shutdown unusable"
    assert seen["turn_error"] is None

    errors = [m for m in _messages(fastworkflow_logs, logging.ERROR) if busy in m]
    assert errors, (
        "the deadline expired with work in flight and nothing was logged at ERROR"
    )
    assert any("NOT be finalized or closed" in m for m in errors)


def test_a_quiescent_runtime_is_still_finalized_and_closed_in_that_shutdown(
    app_module, fastworkflow_logs
):
    """Hazard: the skip for busy channels quietly becoming a skip for everything.

    Skipping is per channel. A runtime with nothing in flight must go through the
    normal path in the same shutdown that spares its busy neighbour, or the fix
    for the deadline hazard turns into "graceful shutdown stopped shutting down".
    """
    busy, quiet = _channel("busy"), _channel("quiet")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as steps:
            await _create(app_module, busy)
            quiet_runtime = await _create(app_module, quiet)
            gate = asyncio.Event()
            execn = await _queued_turn(app_module, busy, gate)

            remaining = await steps.drain(max_wait_seconds=0)
            seen["quiet_skipped"] = quiet in remaining
            seen["quiet_open_before"] = _is_open(quiet_runtime)

            await steps.finalize(remaining)
            await steps.stop(remaining)
            seen["quiet_open_after"] = _is_open(quiet_runtime)

            gate.set()
            await execn.done_event.wait()

    asyncio.run(body())

    assert not seen["quiet_skipped"], "a quiescent channel was treated as busy"
    assert seen["quiet_open_before"]
    assert not seen["quiet_open_after"], (
        "shutdown left a quiescent runtime open, so nothing was shut down"
    )
    # Finalization is per channel too: the deadline complaint names only the
    # channel that was actually working.
    deadline_errors = _messages(fastworkflow_logs, logging.ERROR)
    assert all(quiet not in m for m in deadline_errors)


# ---------------------------------------------------------------------------
# Streaming lifecycle (design §10.4, invariant 21)
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def asgi_client(app_module):
    """Drive the real app over ASGI from the caller's own event loop.

    ``TestClient`` runs the app in a portal on a different loop, so a test cannot
    hold a turn open on one side and issue a request on the other. These
    orderings are exactly about two requests overlapping, so both have to live in
    one loop.
    """
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://manager-shutdown-matrix"
    ) as client:
        yield client


async def _auth_headers(client, channel_id: str) -> dict[str, str]:
    resp = await client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _session_data(app_module, channel_id: str):
    """The SessionData a validated JWT would have produced for this channel."""
    return app_module.SessionData(
        channel_id=channel_id,
        token_type="access",
        issued_at=0,
        expires_at=0,
        jti=uuid.uuid4().hex,
    )


def test_a_normal_turn_is_refused_while_a_streaming_turn_owns_the_channel(app_module):
    """Hazard: an unrelated query answered as somebody else's clarification.

    A stream used to own the channel through ``runtime.lock`` and nothing else,
    so the registry saw no active execution and a normal turn could queue behind
    it. When the stream suspended on ask_user and released the lock, the queued
    query was read as the answer to the clarification. Streaming shares one
    admission gate with every other turn, so the second request is refused.

    The endpoint is called directly because it returns while its turn is still
    QUEUED — the ownership window this asserts on — which no HTTP client can
    observe from outside.
    """
    channel_id = _channel("streamowns")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            response = await app_module.invoke_agent_stream(
                app_module.InvokeRequest(user_query="add 2 and 3", timeout_seconds=60),
                _session_data(app_module, channel_id),
            )
            seen["stream_owns"] = app_module.turn_registry.has_active(channel_id)
            seen["stream_kind"] = app_module.turn_registry.get(
                app_module.turn_registry.active_turn_key(channel_id)
            ).kind

            blocked = await client.post(
                "/invoke_agent",
                headers=headers,
                json={"user_query": "unrelated question", "timeout_seconds": 5},
            )
            seen["status"] = blocked.status_code
            seen["detail"] = blocked.json().get("detail", "")

            async for _chunk in response.body_iterator:
                pass

    asyncio.run(body())

    assert seen["stream_owns"], "the streaming turn never took the channel"
    assert seen["stream_kind"] == "invoke_agent_stream"
    assert seen["status"] == 409, (
        "a normal turn was admitted while a streaming turn owned the channel"
    )
    assert "already in progress" in seen["detail"]


def test_a_streaming_turn_is_refused_while_a_normal_turn_owns_the_channel(app_module):
    """Hazard: the reverse ordering, where the stream is the one that jumps in.

    The occupying turn here is a real registered execution that goes on to
    complete normally, so this also shows the refusal is not a permanently stuck
    pointer: the channel becomes available again once the turn finishes.
    """
    channel_id = _channel("normalowns")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            gate = asyncio.Event()
            execn = await _queued_turn(app_module, channel_id, gate)

            blocked = await client.post(
                "/invoke_agent_stream",
                headers=headers,
                json={"user_query": "add 2 and 3", "timeout_seconds": 5},
            )
            seen["status"] = blocked.status_code
            seen["detail"] = blocked.json().get("detail", "")

            gate.set()
            await execn.done_event.wait()
            seen["free_afterwards"] = not app_module.turn_registry.has_active(
                channel_id
            )

    asyncio.run(body())

    assert seen["status"] == 409, (
        "a stream was admitted while a normal turn owned the channel"
    )
    assert "already in progress" in seen["detail"]
    assert seen["free_afterwards"]


def test_eviction_spares_a_channel_whose_streaming_turn_is_registered(app_module):
    """Hazard: a runtime evicted between admission and first body iteration.

    A streaming turn is registered before Starlette begins consuming the body, and
    during that interval its lock is free. Keyed off the lock alone the channel
    reads idle and the sweep pops and closes it, after which a cold recreation
    finds the old workflow still alive in the weak global registry and overwrites
    its context — two runtimes and two locks around one mutable workflow.

    Wired the way the server wires it, so the predicate under test is the one
    production uses.
    """
    manager = _cap_one_manager()
    registry = TurnRegistry()
    manager.is_channel_busy = registry.has_active
    streaming, newcomer = _channel("streaming"), _channel("newcomer")
    seen: dict[str, Any] = {}

    async def body():
        streaming_runtime = await _create(app_module, streaming, manager=manager)
        gate = asyncio.Event()
        execn = await _queued_turn(
            app_module,
            streaming,
            gate,
            kind="invoke_agent_stream",
            manager=manager,
            registry=registry,
        )
        seen["lock_held"] = streaming_runtime.lock.locked()

        # Creation's own overflow sweep runs here, with the newcomer holding only
        # its initialization lease.
        await _create(app_module, newcomer, manager=manager)
        seen["after_creation"] = set(manager._sessions)

        # And again with no lease anywhere, which is the sweep a later request
        # triggers.
        seen["retired_one"] = await manager._retire_one_candidate()
        seen["after_sweep"] = set(manager._sessions)

        gate.set()
        await execn.done_event.wait()

    asyncio.run(body())

    assert not seen["lock_held"], "the lock was held, so this is not the window at issue"
    assert seen["after_creation"] == {streaming, newcomer}
    assert seen["retired_one"], (
        "the sweep retired nothing at all, so sparing the stream proves nothing"
    )
    assert seen["after_sweep"] == {streaming}, (
        "the sweep evicted the channel a registered streaming turn owned"
    )


def test_an_abandoned_response_body_does_not_end_the_streaming_turn(app_module):
    """Hazard: client disconnect deciding when a workflow stops mutating state.

    Delivery and ownership are separate. If the turn's completion depended on
    somebody draining the response body, a client that hung up would leave the
    execution non-terminal and its active pointer set forever — the channel would
    be permanently 409, and eviction and shutdown would both keep skipping it.
    The body here is partially consumed and then closed, which is what Starlette
    does when the client goes away.
    """
    channel_id = _channel("disconnect")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            await _auth_headers(client, channel_id)
            response = await app_module.invoke_agent_stream(
                app_module.InvokeRequest(user_query="add 2 and 3", timeout_seconds=60),
                _session_data(app_module, channel_id),
            )
            turn_key = app_module.turn_registry.active_turn_key(channel_id)
            assert turn_key, "the streaming turn was never registered"
            execn = app_module.turn_registry.get(turn_key)

            seen["consumed"] = await response.body_iterator.__anext__()
            await response.body_iterator.aclose()

            await asyncio.wait_for(execn.done_event.wait(), timeout=90)
            seen["exec_state"] = execn.exec_state
            seen["still_active"] = app_module.turn_registry.has_active(channel_id)
            seen["lock_released"] = not (
                await app_module.session_manager.get_session(channel_id)
            ).lock.locked()

    asyncio.run(body())

    assert seen["consumed"], "nothing was delivered, so nothing was abandoned"
    assert seen["exec_state"] is ExecState.DONE, (
        "the execution never reached a terminal state after the body was dropped"
    )
    assert not seen["still_active"], "the abandoned turn never cleared its own pointer"
    assert seen["lock_released"]


# ---------------------------------------------------------------------------
# Manager lifecycle operations (design §10.2, §10.3, §10.6)
# ---------------------------------------------------------------------------

def _stored_record(manager: ChannelSessionManager, channel_id: str, workflow_path: str):
    """This channel's committed checkpoint, read the way a cold worker reads it."""
    return manager.checkpoint_store.load_for_adoption(
        deployment_id=checkpoint.deployment_id(),
        workflow_fingerprint=checkpoint.workflow_fingerprint(workflow_path),
        channel_id=channel_id,
    )


@pytest.mark.parametrize("operation", ["remove_session", "evict_live_session"])
def test_explicit_removal_stays_distinct_from_retirement(app_module, operation):
    """Hazard: folding the explicit operations into the retirement path.

    §10.6 keeps evict/remove/terminate as three different things, and these two
    are preserved deliberately rather than because anything in production calls
    them (the only caller in the tree is a test). Routing them through retirement
    would change their contract in two ways at once: they would start refusing
    channels retirement considers pinned, and they would start writing a durable
    checkpoint for a caller that asked to drop a session, not to persist it.
    """
    manager = _cap_one_manager()
    channel_id = _channel("explicit")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id, manager=manager)
        _suspend(runtime)

        seen["evictable"] = checkpoint.assess(runtime).evictable
        seen["retirement_refused"] = not await manager._retire_one_candidate()
        seen["live_after_retirement"] = channel_id in manager._sessions

        await getattr(manager, operation)(channel_id)
        seen["live_after_explicit"] = channel_id in manager._sessions
        seen["open_after_explicit"] = _is_open(runtime)
        seen["record"] = _stored_record(
            manager, channel_id, app_module.ARGS.workflow_path
        )

    asyncio.run(body())

    assert not seen["evictable"], "the channel was not pinned, so nothing is proven"
    assert seen["retirement_refused"]
    assert seen["live_after_retirement"], "retirement dropped a pinned session"

    assert not seen["live_after_explicit"], (
        f"{operation} consulted eligibility instead of removing the session"
    )
    assert not seen["open_after_explicit"], f"{operation} left the context open"
    assert seen["record"] is None, (
        f"{operation} wrote a checkpoint; explicit removal is not retirement"
    )


def test_pinned_channels_hold_the_cache_over_target_and_are_counted(
    app_module, fastworkflow_logs
):
    """Hazard: an over-target cache with no way to tell why it is over target.

    Pinning is a steady state, not a failure, so the sweep must keep returning
    without evicting for as long as the pin lasts — and it has to say how many
    candidates it refused, because "over capacity" with no count is unactionable:
    an operator cannot distinguish workflows that never implemented the
    serialization hooks from a sweep that is simply broken.
    """
    manager = _cap_one_manager()
    first, second, evictable = (
        _channel("pinned_a"),
        _channel("pinned_b"),
        _channel("evictable"),
    )
    seen: dict[str, Any] = {}

    async def body():
        _suspend(await _create(app_module, first, manager=manager))
        _suspend(await _create(app_module, second, manager=manager))
        await _create(app_module, evictable, manager=manager)

        # Creation's own sweeps ran while the pins were still being established
        # and reported the count as it stood then; the assertions below are about
        # the steady state, so start counting from here.
        fastworkflow_logs.clear()

        # First sweep: the one evictable candidate goes, the pinned pair cannot.
        await manager._evict_oldest_if_needed()
        seen["after_first"] = set(manager._sessions)

        # And it stays that way however often the sweep runs.
        for _ in range(3):
            await manager._evict_oldest_if_needed()
        seen["after_repeats"] = set(manager._sessions)
        seen["records"] = [
            _stored_record(manager, channel_id, app_module.ARGS.workflow_path)
            for channel_id in (first, second)
        ]

    asyncio.run(body())

    assert seen["after_first"] == {first, second}, (
        "the sweep either kept the evictable session or dropped a pinned one"
    )
    assert seen["after_repeats"] == {first, second}, (
        "a pinned session was evicted by a later sweep"
    )
    assert seen["records"] == [None, None]
    assert len(seen["after_repeats"]) > manager.max_live_sessions

    warnings = [
        m
        for m in _messages(fastworkflow_logs, logging.WARNING)
        if "no candidate could be retired" in m
    ]
    assert warnings, "the cache sat over target without saying so"
    assert all("(2 pinned)" in m for m in warnings), (
        f"the warning did not name how many candidates were pinned: {warnings}"
    )
