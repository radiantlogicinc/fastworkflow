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
from fastworkflow.run_fastapi_mcp.conversation_store import ConversationStore
from fastworkflow.run_fastapi_mcp.turns import TurnRegistry, submit_turn
from fastworkflow.run_fastapi_mcp.utils import (
    MAX_CONVERSATION_TURNS_IN_MEMORY,
    save_conversation_incremental,
    save_last_turn_feedback,
)
from fastworkflow.utils.logging import logger


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


class _LockAwareConversationStore(ConversationStore):
    """A real store that records whether the registry lock was held during its I/O."""

    def __init__(self, channel_id: str, base_folder: str, registry: TurnRegistry):
        super().__init__(channel_id, base_folder)
        self._registry = registry
        self.io_operations = 0
        self.io_under_registry_lock = 0

    def _get_db(self):
        self.io_operations += 1
        if self._registry._lock.locked():
            self.io_under_registry_lock += 1
        return super()._get_db()


def test_no_store_io_while_the_registry_lock_is_held(app_module):
    """Store I/O under the registry lock would block turn submission on every channel."""
    registry = TurnRegistry()
    channel_id = _channel("lockio")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        watched = _LockAwareConversationStore(
            channel_id, runtime.conversation_store.db_path.rsplit(os.sep, 1)[0], registry
        )
        runtime.conversation_store = watched
        runtime.execution_context.append_conversation_turn("a turn worth saving")

        await _run_one_turn(
            app_module, runtime, registry, kind="initialize_startup",
            idempotency_key="lockio-1",
        )
        return watched

    watched = asyncio.run(body())

    assert watched.io_operations > 0, "the turn did no store I/O at all"
    assert watched.io_under_registry_lock == 0


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
            runtime.execution_context.append_conversation_turn(f"turn-{i}")
            save_conversation_incremental(runtime, extract_turns_from_history, logger)
        return runtime

    runtime = asyncio.run(body())

    messages = runtime.execution_context.conversation_history.messages
    assert len(messages) == MAX_CONVERSATION_TURNS_IN_MEMORY
    assert messages[-1]["conversation summary"] == f"turn-{total_turns - 1}"

    durable = runtime.conversation_store.get_conversation(
        runtime.active_conversation_id
    )
    assert [t["conversation summary"] for t in durable["turns"]] == [
        f"turn-{i}" for i in range(total_turns)
    ]


class _CountingKVStore:
    """Forwards to a real KVStore and tallies the JSON bytes handed to it."""

    def __init__(self, db, tally: dict):
        self._db = db
        self._tally = tally

    def __setitem__(self, key, value):
        self._tally["bytes_written"] += len(json.dumps(value).encode("utf-8"))
        self._tally["writes"] += 1
        self._db[key] = value

    def __getitem__(self, key):
        return self._db[key]

    def __delitem__(self, key):
        del self._db[key]

    def __contains__(self, key):
        return key in self._db

    def get(self, key, default=None):
        return self._db.get(key, default)

    def close(self):
        self._db.close()


class _CountingConversationStore(ConversationStore):
    """A real store whose write volume can be measured."""

    def __init__(self, channel_id: str, base_folder: str):
        super().__init__(channel_id, base_folder)
        self.tally = {"bytes_written": 0, "writes": 0}

    def _get_db(self):
        return _CountingKVStore(super()._get_db(), self.tally)


def test_incremental_save_writes_only_the_new_turns(app_module, tmp_path):
    """Write volume over N turns is linear, not quadratic.

    Replacing the whole turn list per save made turn n rewrite turns 1..n-1 — a
    latency and durable-growth defect on its own, before any memory argument.
    The old full-replace path is measured alongside so the bound is a comparison
    rather than a magic constant.
    """
    turn_count = 40
    turns = [_payload_turn(i) for i in range(turn_count)]
    payload_bytes = sum(len(json.dumps(t).encode("utf-8")) for t in turns)

    appending = _CountingConversationStore("appending", str(tmp_path))
    for turn in turns:
        appending.append_conversation_turns(1, [turn])

    replacing = _CountingConversationStore("replacing", str(tmp_path))
    for i, turn in enumerate(turns):
        replacing.save_conversation_turns(1, turns[: i + 1])

    assert [t["conversation summary"] for t in appending.get_conversation(1)["turns"]] == [
        t["conversation summary"] for t in turns
    ]
    # Linear: each turn's bytes, plus a small metadata record per save.
    assert appending.tally["bytes_written"] < 2 * payload_bytes
    # Quadratic: ~N/2 times the payload. The point is the growth rate, not the ratio.
    assert replacing.tally["bytes_written"] > 8 * appending.tally["bytes_written"]


def test_conversation_summary_reads_the_durable_record_not_the_window(app_module):
    """Summarizing the window would silently mean "topic of the last N turns"."""
    channel_id = _channel("summary")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 10

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(total_turns):
            runtime.execution_context.append_conversation_turn(f"turn-{i}")
            save_conversation_incremental(runtime, extract_turns_from_history, logger)
        return runtime

    runtime = asyncio.run(body())

    in_memory = extract_turns_from_history(
        runtime.execution_context.conversation_history
    )
    for_summary = app_module._conversation_turns_for_summary(runtime)

    assert len(in_memory) == MAX_CONVERSATION_TURNS_IN_MEMORY
    assert len(for_summary) == total_turns
    assert for_summary[0]["conversation summary"] == "turn-0"


def test_activate_conversation_restores_the_window_and_the_high_water_mark(app_module):
    """A restored conversation is already durable; re-appending it would duplicate it."""
    channel_id = _channel("activate")
    total_turns = MAX_CONVERSATION_TURNS_IN_MEMORY + 12

    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def seed():
        runtime = await app_module.session_manager.get_session(channel_id)
        conv_id = runtime.conversation_store.reserve_next_conversation_id()
        runtime.conversation_store.append_conversation_turns(
            conv_id, [_payload_turn(i, size_bytes=64) for i in range(total_turns)]
        )
        return conv_id

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
        assert runtime.durable_turn_count == MAX_CONVERSATION_TURNS_IN_MEMORY

        # A save straight after activation must be a no-op, not a second copy.
        appended = save_conversation_incremental(
            runtime, extract_turns_from_history, logger
        )
        assert appended == 0
        assert runtime.conversation_store.count_conversation_turns(conv_id) == total_turns

        # A genuinely new turn still appends, exactly once.
        runtime.execution_context.append_conversation_turn("brand new")
        assert save_conversation_incremental(
            runtime, extract_turns_from_history, logger
        ) == 1
        durable = runtime.conversation_store.get_conversation(conv_id)
        assert len(durable["turns"]) == total_turns + 1
        assert durable["turns"][-1]["conversation summary"] == "brand new"

    asyncio.run(check())


def test_in_memory_conversation_bytes_plateau(app_module):
    """A hot channel's in-memory history must stop growing, not just grow slower."""
    channel_id = _channel("plateau")
    samples = {}

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(240):
            runtime.execution_context.append_conversation_turn(
                f"turn-{i}", conversation_traces="x" * 4096
            )
            save_conversation_incremental(runtime, extract_turns_from_history, logger)
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
    assert runtime.conversation_store.count_conversation_turns(
        runtime.active_conversation_id
    ) == 240


def test_feedback_on_an_already_durable_turn_is_persisted(app_module):
    """Feedback edits a recorded turn, which the append path cannot express."""
    channel_id = _channel("feedback")

    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

    async def seed():
        runtime = await app_module.session_manager.get_session(channel_id)
        for i in range(3):
            runtime.execution_context.append_conversation_turn(f"turn-{i}")
            save_conversation_incremental(runtime, extract_turns_from_history, logger)
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
        durable = runtime.conversation_store.get_conversation(conv_id)
        assert len(durable["turns"]) == 3, "feedback must not duplicate the turn"
        assert durable["turns"][-1]["feedback"]["nl_feedback"] == "useful"
        assert durable["turns"][0]["feedback"] is None

    asyncio.run(check())


def test_a_stale_inline_turns_field_is_ignored(app_module, tmp_path):
    """After the sqlite migration, only per-turn keys are authoritative.

    Pre-migration RocksDB stores could keep an inline ``turns`` list. New
    ``.sqlite3`` stores ignore that field so a poisoned inline list cannot
    duplicate or reorder the durable per-turn entries.
    """
    from fastworkflow.kvstore import KVStore

    store = ConversationStore("rollback", str(tmp_path))
    conv_id = store.reserve_next_conversation_id()
    for i in range(3):
        store.append_conversation_turns(conv_id, [_payload_turn(i, size_bytes=32)])

    db = KVStore(store.db_path)
    conv = db[f"conv:{conv_id}"]
    conv["turns"] = [_payload_turn(i, size_bytes=32) for i in range(4)]
    db[f"conv:{conv_id}"] = conv
    db.close()

    summaries = [t["conversation summary"] for t in store.get_conversation(conv_id)["turns"]]
    assert summaries == [f"turn-{i}" for i in range(3)]
    assert store.count_conversation_turns(conv_id) == 3

    store.append_conversation_turns(conv_id, [_payload_turn(3, size_bytes=32)])
    summaries = [t["conversation summary"] for t in store.get_conversation(conv_id)["turns"]]
    assert summaries == [f"turn-{i}" for i in range(4)]


def test_summary_read_never_materializes_turn_payloads(app_module, tmp_path):
    """Loading a whole conversation to compute a topic string reintroduces the growth."""
    store = ConversationStore("summaries", str(tmp_path))
    conv_id = store.reserve_next_conversation_id()
    store.append_conversation_turns(
        conv_id, [_payload_turn(i, size_bytes=8192) for i in range(5)]
    )

    summaries = store.get_conversation_summaries(conv_id)

    assert [s["conversation summary"] for s in summaries] == [
        f"turn-{i}" for i in range(5)
    ]
    assert all(set(s) == {"conversation summary"} for s in summaries)
    assert len(json.dumps(summaries)) < 1024


def test_a_stale_high_water_mark_re_appends_rather_than_dropping_turns(app_module):
    """A mark that outran the history must not certify unrecorded turns as durable."""
    channel_id = _channel("stalemark")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(3):
            runtime.execution_context.append_conversation_turn(f"turn-{i}")
            save_conversation_incremental(runtime, extract_turns_from_history, logger)

        # A mark established against some other history: the guard must fall back
        # to re-appending, because a duplicate is recoverable and a loss is not.
        runtime.durable_turn_count = 99
        runtime.execution_context.append_conversation_turn("must-not-be-lost")
        appended = save_conversation_incremental(
            runtime, extract_turns_from_history, logger
        )
        return runtime, appended

    runtime, appended = asyncio.run(body())

    assert appended == 4
    durable = runtime.conversation_store.get_conversation(
        runtime.active_conversation_id
    )
    assert "must-not-be-lost" in [
        t["conversation summary"] for t in durable["turns"]
    ]


def test_feedback_is_not_written_into_a_mismatched_conversation(app_module):
    """Rewriting by position needs the mark and the conversation to be the same one."""
    channel_id = _channel("mismatch")

    async def body():
        runtime = await _build_runtime(app_module, channel_id)
        for i in range(2):
            runtime.execution_context.append_conversation_turn(f"turn-{i}")
            save_conversation_incremental(runtime, extract_turns_from_history, logger)

        # Point the runtime at a different, shorter conversation while the mark
        # still describes the one it was reading.
        other_id = runtime.conversation_store.reserve_next_conversation_id()
        runtime.conversation_store.append_conversation_turns(
            other_id, [_payload_turn(0, size_bytes=32)]
        )
        runtime.active_conversation_id = other_id
        runtime.execution_context.conversation_history.messages[-1]["feedback"] = {
            "nl_feedback": "belongs to the other conversation"
        }

        save_last_turn_feedback(runtime, extract_turns_from_history, logger)
        return runtime, other_id

    runtime, other_id = asyncio.run(body())

    other = runtime.conversation_store.get_conversation(other_id)
    assert len(other["turns"]) == 1
    assert other["turns"][0]["feedback"] is None


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
