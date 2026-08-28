"""Integration tests for the run_fastapi_mcp retention bounds (Release A).

Two families:

* **Turn retention** — a finished execution is kept only if something can still
  look it up, and then only within a bounded count and age. Live executions are
  never swept, a launch that never started is rolled back, and the registry lock
  is never held across store I/O (it gates turn submission on *every* channel).
* **Conversation bounds** — the durable conversation record is append-only and
  the in-memory history is a window over its newest turns. The ordering is the
  whole point: a turn is durably recorded before it can be dropped from memory,
  so windowing memory never shortens the durable record.

Everything runs against real runtimes, a real SQLite-backed conversation store and
the real turn engine. Turn bodies are plain callables rather than trained
commands, so no model or LLM call is required.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import os
import sys
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.conversation_history_io import extract_turns_from_history
from fastworkflow.run_fastapi_mcp import server_memory
from fastworkflow.run_fastapi_mcp.turns import TurnRegistry, submit_turn
from fastworkflow.run_fastapi_mcp.utils import (
    MAX_CONVERSATION_TURNS_IN_MEMORY,
    save_last_turn_feedback,
    trim_conversation_window,
)
from fastworkflow.utils.logging import logger


def _record_turn(runtime, summary: str, traces: str | None = None) -> None:
    """One recorded conversation turn, the way a real turn records itself.

    Since the Phase-7 consolidation there is no incremental save to call: the
    turn record IS the durable conversation, and it is written by the finalize
    chokepoint inside the turn. So this drives a real logical turn — begin,
    append the history entry, finalize — and then windows memory exactly as
    ``turns._run_turn`` does after the work returns.
    """
    ctx = runtime.execution_context
    ctx._begin_turn(summary)
    ctx.append_conversation_turn(summary, traces)
    ctx._build_turn_result(
        fastworkflow.CommandOutput(
            command_name="",
            command_response=fastworkflow.CommandResponse(response="ok"),
        )
    )
    trim_conversation_window(runtime, logger)


def _durable_summaries(runtime, conv_id: int | None = None) -> list[str]:
    """Every usable turn of a conversation, oldest first, as summaries."""
    store = runtime.observability_store
    assert store is not None, "no observability store, so nothing is durable"
    runtime.execution_context.trace_sink.flush()
    return [
        entry["conversation summary"]
        for entry in store.get_memory_window(
            runtime.channel_id,
            conv_id if conv_id is not None else runtime.active_conversation_id,
            1_000_000,
        )
    ]


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


async def _build_runtime(app_module, channel_id: str):
    await app_module.ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=app_module.session_manager,
        workflow_path=app_module.ARGS.workflow_path,
        run_startup=False,
    )
    return await app_module.session_manager.get_session(channel_id)


def _turn_output(answer: str = "ok") -> fastworkflow.TurnOutput:
    return fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=fastworkflow.TurnStatus.COMPLETED,
        answer=answer,
    )


async def _run_one_turn(
    app_module,
    runtime,
    registry: TurnRegistry,
    *,
    kind: str,
    idempotency_key: str,
    work=None,
    wait_seconds: float = 30.0,
):
    """Submit a turn through the real engine and wait for it to finish."""

    def work_fn() -> fastworkflow.TurnOutput:
        if work is not None:
            work()
        return _turn_output()

    return await submit_turn(
        runtime,
        registry,
        work_fn,
        app_module.session_manager,
        wait_seconds=wait_seconds,
        kind=kind,
        idempotency_key=idempotency_key,
    )


def _payload_turn(index: int, size_bytes: int = 4096) -> dict:
    """A request-sized conversation turn, unique per index."""
    return {
        "conversation summary": f"turn-{index}",
        "conversation_traces": f"{index}:" + ("x" * size_bytes),
        "feedback": None,
    }


# ---------------------------------------------------------------------------
# Turn retention (design section 16.1)
# ---------------------------------------------------------------------------

def test_retained_startup_turns_are_capped_at_the_newest_ones(app_module):
    """More completed startups than the cap retains exactly the newest ones."""
    registry = TurnRegistry(max_retained_terminal=5)
    channel_id = _channel("cap")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        completed = []
        for i in range(12):
            execn = await _run_one_turn(
                app_module,
                runtime,
                registry,
                kind="initialize_startup",
                idempotency_key=f"startup-{i}",
            )
            completed.append(execn.turn_key)
        return completed

    completed = asyncio.run(body())

    assert set(registry._by_key) == set(completed[-5:])


def test_expired_retained_startups_are_removed(app_module):
    """The age window drops a retained startup once a later completion sweeps."""
    registry = TurnRegistry(retention_seconds=0.2)
    channel_id = _channel("ttl")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        first = await _run_one_turn(
            app_module, runtime, registry, kind="initialize_startup",
            idempotency_key="startup-first",
        )
        assert first.turn_key in registry._by_key
        assert first.ttl_expires_at is not None

        await asyncio.sleep(0.4)
        second = await _run_one_turn(
            app_module, runtime, registry, kind="initialize_startup",
            idempotency_key="startup-second",
        )
        return first, second

    first, second = asyncio.run(body())

    assert first.turn_key not in registry._by_key
    assert second.turn_key in registry._by_key


def test_non_startup_terminal_executions_are_not_retained(app_module):
    """Nothing looks up a finished agent/action turn, so nothing keeps it."""
    registry = TurnRegistry()
    channel_id = _channel("nonstartup")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        agent = await _run_one_turn(
            app_module, runtime, registry, kind="invoke_agent",
            idempotency_key="agent-1",
        )
        startup = await _run_one_turn(
            app_module, runtime, registry, kind="initialize_startup",
            idempotency_key="startup-1",
        )
        return agent, startup

    agent, startup = asyncio.run(body())

    assert agent.turn_key not in registry._by_key
    assert startup.turn_key in registry._by_key
    # The completing request still renders its own result from its local
    # reference, whether or not the registry kept the record.
    assert agent.result is not None
    assert agent.error is None


def test_live_executions_survive_age_and_count_sweeps(app_module):
    """A running execution is never swept, and is stamped with a TTL only once it finishes."""
    registry = TurnRegistry(max_retained_terminal=1, retention_seconds=0.01)
    slow_channel = _channel("slow")
    other_channel = _channel("other")

    async def body():
        slow_runtime = await _build_runtime(app_module, slow_channel)
        other_runtime = await _build_runtime(app_module, other_channel)

        # Defer: the wait window expires long before the work does.
        slow = await _run_one_turn(
            app_module, slow_runtime, registry, kind="initialize_startup",
            idempotency_key="slow", work=lambda: time.sleep(2.0),
            wait_seconds=0.2,
        )
        assert not slow.is_terminal
        assert slow.ttl_expires_at is None

        # Overflow and age sweeps run on every completion. They must not be able
        # to reach a live execution, even at a cap of one and a 10 ms window.
        for i in range(3):
            await _run_one_turn(
                app_module, other_runtime, registry, kind="initialize_startup",
                idempotency_key=f"filler-{i}",
            )
        assert registry.evict_terminal() >= 0

        assert slow.turn_key in registry._by_key
        assert registry.has_active(slow_channel)
        assert slow.ttl_expires_at is None

        await slow.done_event.wait()
        return slow

    slow = asyncio.run(body())

    assert slow.is_terminal
    assert slow.ttl_expires_at is not None


def test_task_launch_failure_rolls_back_both_registry_pointers():
    """A launch that raises leaves no record: it could never reach a terminal state."""
    registry = TurnRegistry()
    channel_id = _channel("launchfail")

    def failing_launch(_execn):
        raise RuntimeError("event loop refused the task")

    async def body():
        with pytest.raises(RuntimeError, match="event loop refused"):
            await registry.start_or_get_active(
                channel_id,
                kind="initialize_startup",
                idempotency_key="never-launched",
                run_turn=failing_launch,
            )

    asyncio.run(body())

    assert registry._by_key == {}
    assert registry.active_turn_key(channel_id) is None
    assert not registry.has_active(channel_id)


def test_identical_completed_turns_are_not_deduplicated(app_module):
    """Two identical requests are two turns: repeating a query can be intentional."""
    registry = TurnRegistry()
    channel_id = _channel("dedupe")
    calls = []

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        first = await _run_one_turn(
            app_module, runtime, registry, kind="invoke_agent",
            idempotency_key="same-key", work=lambda: calls.append(1),
        )
        second = await _run_one_turn(
            app_module, runtime, registry, kind="invoke_agent",
            idempotency_key="same-key", work=lambda: calls.append(1),
        )
        return first, second

    first, second = asyncio.run(body())

    assert first.turn_key != second.turn_key
    assert len(calls) == 2


def test_no_store_io_while_the_registry_lock_is_held(app_module):
    """Store I/O under the registry lock would block turn submission on every channel.

    The turn record is written synchronously on the caller thread now (Phase 7
    §2.4), which makes this sharper than it was against the legacy store: the
    write is squarely on the turn path, so holding the registry lock across it
    would serialize every channel's submissions behind one channel's disk.
    """
    registry = TurnRegistry()
    channel_id = _channel("lockio")
    io = {"operations": 0, "under_registry_lock": 0}

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        store = runtime.observability_store
        assert store is not None
        original_connect = store._connect

        def watched_connect(*args, **kwargs):
            io["operations"] += 1
            if registry._lock.locked():
                io["under_registry_lock"] += 1
            return original_connect(*args, **kwargs)

        store._connect = watched_connect
        try:
            await _run_one_turn(
                app_module, runtime, registry, kind="initialize_startup",
                idempotency_key="lockio-1",
                # The work has to reach the finalize chokepoint, because that is
                # what does the store I/O this test is watching for.
                work=lambda: _record_turn(runtime, "a turn worth saving"),
            )
        finally:
            store._connect = original_connect

    asyncio.run(body())

    assert io["operations"] > 0, "the turn did no store I/O at all"
    assert io["under_registry_lock"] == 0


def test_creation_locks_do_not_accumulate(app_module):
    """A channel seen once must not keep a lock forever; the mapping holds weak values."""
    manager = app_module.session_manager
    channels = [_channel("lock") for _ in range(50)]

    for channel_id in channels:
        lock = manager.get_creation_lock(channel_id)
        assert lock is manager.get_creation_lock(channel_id)

    del lock
    gc.collect()

    assert all(channel_id not in manager._creation_locks for channel_id in channels)


# ---------------------------------------------------------------------------
# Conversation bounds (design section 16.2)
# ---------------------------------------------------------------------------

def test_durable_conversation_keeps_turns_the_memory_window_dropped(app_module):
    """Windowing memory must never shorten the durable record.

    This is what fails against the naive ordering (trim memory, then save the
    whole in-memory history), which turns a memory leak into data loss.
    """
    channel_id = _channel("window")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 15

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(total_turns):
            _record_turn(runtime, f"turn-{i}")
        return runtime

    runtime = asyncio.run(body())

    messages = runtime.execution_context.conversation_history.messages
    assert len(messages) == MAX_CONVERSATION_TURNS_IN_MEMORY
    assert messages[-1]["conversation summary"] == f"turn-{total_turns - 1}"

    assert _durable_summaries(runtime) == [
        f"turn-{i}" for i in range(total_turns)
    ]


def test_recording_a_turn_writes_only_that_turn(app_module):
    """Write volume over N turns is linear, not quadratic.

    The legacy store's full-replace save made turn n rewrite turns 1..n-1 — a
    latency and durable-growth defect on its own, before any memory argument. It
    was fixed there by an incremental append, and is now structural: a turn is
    one row keyed by its own turn_key, and turn rows are write-once, so nothing
    can rewrite a turn that has already landed.

    Pinned by row identity rather than by counting bytes, because that is what
    the property actually is now.
    """
    channel_id = _channel("linear")
    turn_count = 12

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(turn_count):
            _record_turn(runtime, f"turn-{i}", traces="x" * 4096)
        runtime.execution_context.trace_sink.flush()
        return runtime

    runtime = asyncio.run(body())
    store = runtime.observability_store
    rows = store.list_turns(channel_id=channel_id, limit=1000)

    assert len(rows) == turn_count, "a turn wrote more than its own row"
    assert len({row["turn_key"] for row in rows}) == turn_count
    # Ordinals are dense from 1, so no turn was written twice under two keys.
    assert sorted(row["ordinal"] for row in rows) == list(range(1, turn_count + 1))


def test_conversation_summary_reads_the_durable_record_not_the_window(app_module):
    """Summarizing the window would silently mean "topic of the last N turns"."""
    channel_id = _channel("summary")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 10

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(total_turns):
            _record_turn(runtime, f"turn-{i}")
        runtime.execution_context.trace_sink.flush()
        return runtime

    runtime = asyncio.run(body())

    in_memory = extract_turns_from_history(
        runtime.execution_context.conversation_history
    )
    for_summary = app_module._conversation_turns_for_summary(runtime)

    assert len(in_memory) == MAX_CONVERSATION_TURNS_IN_MEMORY
    assert len(for_summary) == total_turns
    assert for_summary[0]["conversation summary"] == "turn-0"


def test_activate_conversation_restores_the_window_without_duplicating_it(app_module):
    """A restored conversation is already durable; re-recording it would duplicate it.

    The high-water mark this used to assert on is gone (ruling C5): a turn is
    recorded under its own turn_key at finalize, so there is no index into the
    live message list that a restore has to keep aligned. What still has to hold
    is the outcome that mark existed for — activating a conversation must not
    append a second copy of it — and it now holds structurally, because nothing
    re-records a turn that already has a key.
    """
    channel_id = _channel("activate")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 12

    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def seed():
        runtime = await app_module.session_manager.get_session(channel_id)
        for i in range(total_turns):
            _record_turn(runtime, f"turn-{i}", traces="x" * 64)
        runtime.execution_context.trace_sink.flush()
        return runtime.active_conversation_id

    conv_id = asyncio.run(seed())

    resp = client.post(
        "/activate_conversation", headers=headers, json={"conversation_id": conv_id}
    )
    assert resp.status_code == 200

    async def check():
        runtime = await app_module.session_manager.get_session(channel_id)
        assert len(runtime.execution_context.conversation_history.messages) == (
            MAX_CONVERSATION_TURNS_IN_MEMORY
        )
        # Windowing straight after activation must not change the record.
        trim_conversation_window(runtime, logger)
        assert len(_durable_summaries(runtime, conv_id)) == total_turns

        # A genuinely new turn still lands, exactly once.
        _record_turn(runtime, "brand new")
        summaries = _durable_summaries(runtime, conv_id)
        assert len(summaries) == total_turns + 1
        assert summaries[-1] == "brand new"

    asyncio.run(check())


def test_in_memory_conversation_bytes_plateau(app_module):
    """A hot channel's in-memory history must stop growing, not just grow slower."""
    channel_id = _channel("plateau")
    samples = {}

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(240):
            _record_turn(runtime, f"turn-{i}", traces="x" * 4096)
            if i in (79, 159, 239):
                samples[i] = server_memory.conversation_memory_metrics([runtime])
        return runtime

    runtime = asyncio.run(body())

    assert samples[79]["turns"] == MAX_CONVERSATION_TURNS_IN_MEMORY
    assert samples[239]["turns"] == MAX_CONVERSATION_TURNS_IN_MEMORY
    # Bytes track the window, so tripling the request count must not move them.
    assert samples[239]["approx_bytes"] == pytest.approx(
        samples[79]["approx_bytes"], rel=0.05
    )
    store = runtime.observability_store
    runtime.execution_context.trace_sink.flush()
    assert store.count_usable_turns(channel_id, runtime.active_conversation_id) == 240


def test_feedback_on_an_already_durable_turn_is_persisted(app_module):
    """Feedback edits a recorded turn, which a write-once turn row cannot express.

    Hence the separate ``feedback`` table, joined into the memory window: turn
    rows stay write-once while feedback stays mutable [R3].
    """
    channel_id = _channel("feedback")

    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def seed():
        runtime = await app_module.session_manager.get_session(channel_id)
        for i in range(3):
            _record_turn(runtime, f"turn-{i}")
        runtime.execution_context.trace_sink.flush()
        return runtime.active_conversation_id

    conv_id = asyncio.run(seed())

    resp = client.post(
        "/post_feedback",
        headers=headers,
        json={"binary_or_numeric_score": 1, "nl_feedback": "useful"},
    )
    assert resp.status_code == 200

    async def check():
        runtime = await app_module.session_manager.get_session(channel_id)
        runtime.execution_context.trace_sink.flush()
        window = runtime.observability_store.get_memory_window(
            channel_id, conv_id, 1_000_000
        )
        assert len(window) == 3, "feedback must not duplicate the turn"
        assert window[-1]["feedback"]["nl_feedback"] == "useful"
        assert window[0]["feedback"] is None

    asyncio.run(check())


def test_an_undurable_turn_is_not_trimmed_out_of_memory(app_module):
    """The successor to the stale-high-water-mark guard (rulings I1/I2).

    The old hazard: a mark that outran the history certified unrecorded turns as
    durable, and the trim then dropped them. There is no mark now — the trim
    asks the sink whether the last record actually landed. So the hazard becomes
    "the write degraded and we trimmed anyway", and the answer is to defer: the
    turn stays in memory, and the pending-retry ring still owes it to the store.
    """
    channel_id = _channel("stalemark")
    seen: dict = {}

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(MAX_CONVERSATION_TURNS_IN_MEMORY + 3):
            _record_turn(runtime, f"turn-{i}")
        seen["trimmed_while_healthy"] = len(
            runtime.execution_context.conversation_history.messages
        )

        # A wedged DB, as the sink sees it.
        sink = runtime.execution_context.trace_sink
        sink._sync_breaker_until = time.monotonic() + 300
        _record_turn(runtime, "must-not-be-lost")
        seen["ack"] = runtime.execution_context.last_turn_record_stored
        seen["in_memory_after"] = [
            m["conversation summary"]
            for m in runtime.execution_context.conversation_history.messages
        ]
        seen["ring_depth"] = sink.pending_retry_depth()
        return runtime

    runtime = asyncio.run(body())

    assert seen["trimmed_while_healthy"] == MAX_CONVERSATION_TURNS_IN_MEMORY, (
        "the window was not being enforced at all, so deferring proves nothing"
    )
    assert seen["ack"] is False, "the degraded path was never taken"
    assert "must-not-be-lost" in seen["in_memory_after"]
    assert len(seen["in_memory_after"]) == MAX_CONVERSATION_TURNS_IN_MEMORY + 1, (
        "the trim ran on a turn that was not durable yet, which is the loss the "
        "ack gate exists to prevent"
    )
    assert seen["ring_depth"] >= 1, "the record never entered the retry ring"
    # And the ring settles it: the turn reaches the store regardless.
    assert "must-not-be-lost" in _durable_summaries(runtime)


def test_a_cold_restore_rebuilds_memory_from_the_turns_table(app_module):
    """Gate 1 ([R3], §2.3): memory comes from the observability DB now.

    The window is bounded by the read, not by slicing a full hydration, so a
    long conversation is never resident in full just to produce twenty turns.
    """
    channel_id = _channel("coldrestore")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 8

    async def seed():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(total_turns):
            _record_turn(runtime, f"turn-{i}")
        runtime.execution_context.trace_sink.flush()
        return runtime.active_conversation_id

    conv_id = asyncio.run(seed())

    async def restore():
        # Evict without clearing durable state, then create cold.
        await app_module.session_manager.evict_live_session(channel_id)
        runtime = await _build_runtime(app_module, channel_id)
        return runtime

    runtime = asyncio.run(restore())

    messages = runtime.execution_context.conversation_history.messages
    assert runtime.active_conversation_id == conv_id, (
        "the cold restore started a new conversation instead of continuing one"
    )
    assert len(messages) == MAX_CONVERSATION_TURNS_IN_MEMORY, (
        "the restore read the whole conversation rather than the window"
    )
    assert [m["conversation summary"] for m in messages] == [
        f"turn-{i}" for i in range(total_turns - MAX_CONVERSATION_TURNS_IN_MEMORY, total_turns)
    ]
    # And feedback given right after a restore has a turn to attach to.
    assert runtime.execution_context.last_completed_turn_key is not None


def test_a_cold_restore_reuses_an_empty_conversation_instead_of_minting(app_module):
    """An idle channel must not accumulate a conversation row per cold start.

    /initialize mints a conversation id eagerly so the first turn's records are
    attributed. A channel that never sends a message and is then evicted has
    exactly one conversation, with no turns. Minting again on the next cold
    start strands that one and does it again on every restore after that.

    This is the case where NOTHING has turns; when something does, the
    step-back rules (ruling I7) apply instead and the conversation with turns
    wins — the test below covers that.
    """
    channel_id = _channel("emptyreuse")

    async def seed():
        runtime = await _build_runtime(app_module, channel_id)
        assert runtime.active_conversation_id > 0, "no id was minted eagerly"
        return runtime.active_conversation_id

    reserved = asyncio.run(seed())

    async def restore():
        await app_module.session_manager.evict_live_session(channel_id)
        return await _build_runtime(app_module, channel_id)

    runtime = asyncio.run(restore())

    assert runtime.active_conversation_id == reserved, (
        f"the restore minted {runtime.active_conversation_id} instead of reusing "
        f"the reserved-but-empty {reserved}"
    )
    assert runtime.execution_context.conversation_history.messages == [], (
        "an empty conversation restored history from somewhere"
    )


def test_a_cold_restore_steps_back_to_the_last_conversation_with_turns(app_module):
    """The step-back binds the stepped-back conversation as the active one.

    Legacy parity, ratified as ruling I7: a conversation with turns outranks a
    newer empty one, and whichever is restored is also the one bound, so the
    next turn continues it rather than being recorded against a third.
    """
    channel_id = _channel("stepback")

    async def seed():
        runtime = await _build_runtime(app_module, channel_id)
        store = runtime.observability_store
        _record_turn(runtime, "conversation with content")
        with_turns = runtime.active_conversation_id
        # Two empties on top, so the newest has nothing and the step-back has to
        # look past it.
        store.mint_conversation_id(channel_id)
        runtime.execution_context.trace_sink.flush()
        return with_turns

    with_turns = asyncio.run(seed())

    async def restore():
        await app_module.session_manager.evict_live_session(channel_id)
        return await _build_runtime(app_module, channel_id)

    runtime = asyncio.run(restore())

    summaries = [
        m["conversation summary"]
        for m in runtime.execution_context.conversation_history.messages
    ]
    assert summaries == ["conversation with content"], (
        "the step-back did not restore the conversation that had turns"
    )
    assert runtime.active_conversation_id == with_turns, (
        "memory was restored from one conversation while another was bound, so "
        "the next turn would be recorded somewhere else"
    )


def test_feedback_after_an_activation_lands_on_the_activated_conversation(app_module):
    """Activating a conversation moves the feedback target with it (ruling I3).

    Feedback is keyed by ``last_completed_turn_key``, which names the turn this
    process ran last. An activation replaces the in-memory history with another
    conversation's, so leaving that key alone would file the user's feedback
    against a turn of the conversation they just navigated away from.
    """
    channel_id = _channel("activatefeedback")

    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def seed():
        runtime = await app_module.session_manager.get_session(channel_id)
        store = runtime.observability_store
        # Conversation A, then rotate to B and record a turn there, so the
        # process's last completed turn belongs to B while A is activated.
        _record_turn(runtime, "conversation A turn")
        conv_a = runtime.active_conversation_id
        runtime.active_conversation_id = store.mint_conversation_id(channel_id)
        runtime.execution_context.bind_observability_identity(
            conversation_id=runtime.active_conversation_id
        )
        runtime.execution_context.clear_conversation_history()
        _record_turn(runtime, "conversation B turn")
        conv_b = runtime.active_conversation_id
        runtime.execution_context.trace_sink.flush()
        return conv_a, conv_b

    conv_a, conv_b = asyncio.run(seed())

    resp = client.post(
        "/activate_conversation", headers=headers, json={"conversation_id": conv_a}
    )
    assert resp.status_code == 200

    resp = client.post(
        "/post_feedback",
        headers=headers,
        json={"binary_or_numeric_score": 1, "nl_feedback": "about A"},
    )
    assert resp.status_code == 200

    async def check():
        runtime = await app_module.session_manager.get_session(channel_id)
        runtime.execution_context.trace_sink.flush()
        store = runtime.observability_store
        window_a = store.get_memory_window(channel_id, conv_a, 100)
        window_b = store.get_memory_window(channel_id, conv_b, 100)
        return window_a, window_b

    window_a, window_b = asyncio.run(check())

    assert [entry["feedback"] for entry in window_b] == [None], (
        "the feedback landed on the conversation the user navigated away from"
    )
    assert len(window_a) == 1
    assert window_a[0]["feedback"]["nl_feedback"] == "about A"


def test_feedback_is_not_written_into_a_mismatched_conversation(app_module):
    """Feedback follows the turn it was given on, not the active conversation.

    Keying by turn_key (ruling I3/C4) is what makes this structural rather than
    guarded: repointing the runtime at another conversation cannot move the
    feedback, because the key names a row, not a position.
    """
    channel_id = _channel("mismatch")
    seen: dict = {}

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(2):
            _record_turn(runtime, f"turn-{i}")
        seen["original_conv"] = runtime.active_conversation_id
        seen["fed_turn_key"] = runtime.execution_context.last_completed_turn_key

        # Point the runtime at a different, shorter conversation after the turn
        # the feedback belongs to has already been recorded.
        store = runtime.observability_store
        other_id = store.mint_conversation_id(channel_id)
        runtime.active_conversation_id = other_id
        runtime.execution_context.conversation_history.messages[-1]["feedback"] = {
            "nl_feedback": "belongs to the other conversation"
        }

        save_last_turn_feedback(runtime, logger)
        runtime.execution_context.trace_sink.flush()
        seen["other_window"] = store.get_memory_window(channel_id, other_id, 100)
        seen["original_window"] = store.get_memory_window(
            channel_id, seen["original_conv"], 100
        )
        return runtime

    asyncio.run(body())

    assert seen["other_window"] == [], (
        "the other conversation gained a turn it never had"
    )
    fed = [entry for entry in seen["original_window"] if entry["feedback"]]
    assert len(fed) == 1, "the feedback did not land on exactly one turn"
    assert fed[0]["conversation summary"] == "turn-1", (
        "the feedback landed on the wrong turn of the original conversation"
    )


def test_readiness_probe_reports_memory_metrics_on_demand(app_module):
    """The soak harness reads retention metrics over HTTP; probes stay cheap by default."""
    # The context manager runs the lifespan, which is what marks the app ready.
    with TestClient(app_module.app) as client:
        plain = client.get("/probes/readyz")
        detailed = client.get("/probes/readyz?memory=true")

    assert plain.status_code == 200
    assert "memory" not in plain.json()

    assert detailed.status_code == 200
    memory = detailed.json()["memory"]
    assert set(memory) == {
        "live_sessions",
        "retained_turns",
        "dspy_cache",
        "conversations",
    }
    assert set(memory["dspy_cache"]) == {
        "entries",
        "max_entries",
        "approx_bytes",
        "disk_cache_enabled",
    }
    assert set(memory["conversations"]) == {"turns", "approx_bytes"}
    assert json.dumps(memory)
