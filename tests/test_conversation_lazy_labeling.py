"""When a conversation gets its topic and summary, and what that costs.

A conversation used to be labeled at process shutdown, which meant one
synchronous LLM call per live channel inside a 30 s termination grace period —
and, because the live-session cache evicts long before SIGTERM, the ~92% of
conversations that had already been retired were never labeled at all. Labeling
now happens when a conversation is actually *used* as a chat: after a completed
chat turn, and when an unlabeled conversation is activated. These tests are about
the three questions that arrangement raises.

**WHERE the call runs.** After the turn reaches ``DONE``, its registry pointer is
cleared and its ``done_event`` is set — never beside the incremental save inside
``runtime.lock``, which is where the upstream issue asked for it. From there the
channel is still busy: it is in ``busy_channel_ids()``, so the shutdown drain
waits on the LLM call, the user's own 200/202 waits on it, and a concurrent
request on that channel is answered 409 for the length of a round trip. Three
tests below observe the channel *while a generation is in flight* and assert the
opposite of each.

**WHICH turns count.** ``_run_turn`` persists unconditionally once ``work_fn``
returns, including when the turn suspended at ``ask_user``, so "it was persisted"
is not "the exchange finished". Gating is on the turn's own ``TurnStatus`` and on
its kind: a clarifying question and a programmatic ``/perform_action`` do not
name a conversation.

**HOW OFTEN.** Labeling once and never refreshing gives a 40-turn thread the
title of its first exchange, permanently — the same defect as summarizing the
20-turn memory window, reached from the other end. The schedule is geometric, so
the assertion that matters is a count: 17 chat turns must cost three LLM calls,
at turns 1, 4 and 16.

No LLM is called. The one substitution is a recording stand-in bound over the
topic-generation entry point each call site resolves at call time — the same seam
``tests/test_manager_shutdown_matrix.py`` and
``tests/test_conversation_topic_generation_bounds.py`` use. Everything else is
real: the app, the session manager, the turn registry, the real ``_run_turn`` and
``run_owned_turn`` lifecycles, the real lifespan shutdown closures and a real
SQLite conversation store.

What these CANNOT prove without a live provider: that a real
``LLM_CONVERSATION_STORE`` round trip returns a *good* title, and that it honours
the deadline it is sent (that is litellm's contract with the transport, and
``test_conversation_topic_generation_bounds.py`` covers the fact that a deadline
is sent at all). Slowness and failure are reproduced here by a stand-in that
sleeps or raises, which is what makes the ordering assertions deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.run_fastapi_mcp import turns as turns_module
from fastworkflow.run_fastapi_mcp.turns import ExecState
from fastworkflow.run_fastapi_mcp.utils import (
    ChannelRuntime,
    ChannelSessionManager,
    _label_milestones_reached,
)


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
    """A copy of the real env file whose FASTWORKFLOW_STATE_ROOT is private.

    The override has to live in the *file*: the lifespan re-runs
    ``fastworkflow.init()`` from ``ARGS.env_file_path`` on startup, and
    ``get_env_var`` reads that mapping before it ever looks at the process
    environment. These tests write conversation records, so anything set another
    way would land in the developer's real state folder.
    """
    env_file, _ = env_files
    kept = [
        line
        for line in Path(env_file).read_text().splitlines()
        if not line.strip().startswith("FASTWORKFLOW_STATE_ROOT=")
    ]
    kept.append(f"FASTWORKFLOW_STATE_ROOT={tmp_path / 'workflow_contexts'}")
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


# ---------------------------------------------------------------------------
# The recording stand-in for topic generation
# ---------------------------------------------------------------------------

class _Labeler:
    """Stands in for ``generate_topic_and_summary`` and records every call.

    Records what it was handed, when it finished, and signals when it starts, so
    a test can observe the channel *during* a generation rather than only after
    it. ``turn_counts`` is the length of the turn list of each call, which is the
    conversation's durable turn count at that moment — the signal the refresh
    schedule is defined in terms of.

    Blocking with ``time.sleep`` on purpose: this runs in an executor thread, and
    a sleep there is exactly the shape of a slow provider. If it were ever called
    on the event loop instead, every ordering test in this file would fail.
    """

    def __init__(self, *, seconds: float = 0.0, fail: bool = False):
        self.seconds = seconds
        self.fail = fail
        self.turn_counts: list[int] = []
        self.finished_at: list[float] = []
        self.started = threading.Event()

    def __call__(self, turns: list[dict[str, Any]]) -> tuple[str, str]:
        self.turn_counts.append(len(turns))
        self.started.set()
        if self.seconds:
            time.sleep(self.seconds)
        self.finished_at.append(time.monotonic())
        if self.fail:
            raise TimeoutError("LM request timed out")
        # Distinct per call so a second label cannot be mistaken for the first,
        # and so the store's uniqueness check never has a real collision to
        # resolve (which would be a different test's subject).
        index = len(self.turn_counts)
        return f"Topic {index}", f"Summary {index}"

    @property
    def count(self) -> int:
        return len(self.turn_counts)


def _install(monkeypatch, app_module, labeler: _Labeler) -> _Labeler:
    """Bind one labeler over BOTH module attributes that resolve generation.

    ``__main__`` owns the rotate and activate call sites; ``turns`` owns the
    post-turn one. Sharing a single counter across both is what makes "at most
    one generate" an assertion about the conversation rather than about one code
    path — a duplicate that arrived through the other module would otherwise be
    invisible.
    """
    monkeypatch.setattr(app_module, "generate_topic_and_summary", labeler)
    monkeypatch.setattr(turns_module, "generate_topic_and_summary", labeler)
    return labeler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _add_action() -> dict:
    return {
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 2.0, "second_num": 3.0},
    }


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


async def _chat_turn(
    app_module,
    runtime: ChannelRuntime,
    *,
    message: str,
    kind: str = "invoke_agent",
    status=fastworkflow.TurnStatus.COMPLETED,
    manager: Optional[ChannelSessionManager] = None,
):
    """Run one real turn through ``submit_turn`` and return its execution.

    The work function records a conversation turn through the execution
    context's own public API and reports the outcome as a real ``TurnOutput``,
    which is what lets a test choose the status (a suspension, a failure) without
    needing a trained workflow or a live agent. Everything downstream — the
    incremental save, the persistence of suspended state, the registry
    retirement, the labeling hook — is the production path.
    """
    manager = manager or app_module.session_manager

    def work_fn() -> fastworkflow.TurnOutput:
        runtime.execution_context.append_conversation_turn(f"{message} -> ok")
        return fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(), status=status
        )

    return await app_module.submit_turn(
        runtime,
        app_module.turn_registry,
        work_fn,
        manager,
        wait_seconds=30,
        kind=kind,
        idempotency_key=app_module.compute_idempotency_key(
            runtime.channel_id, kind, message
        ),
    )


def _seed_conversation(runtime: ChannelRuntime, turn_count: int) -> int:
    """A durable, unlabeled conversation of ``turn_count`` turns.

    Written through the real store, because every lazy trigger reads the durable
    record rather than the in-memory window; a conversation that exists only in
    memory would take a different branch entirely.
    """
    conv_id = runtime.conversation_store.reserve_next_conversation_id()
    runtime.conversation_store.append_conversation_turns(
        conv_id,
        [
            {
                "conversation summary": f"seeded exchange {i} -> ok",
                "conversation_traces": None,
                "feedback": None,
            }
            for i in range(turn_count)
        ],
    )
    return conv_id


def _label_state(runtime: ChannelRuntime, conv_id: Optional[int] = None):
    return runtime.conversation_store.get_conversation_label_state(
        conv_id if conv_id is not None else runtime.active_conversation_id
    )


async def _await_generation_start(labeler: _Labeler, timeout: float = 10.0) -> None:
    """Yield the loop until the executor thread is inside the generation."""
    deadline = time.monotonic() + timeout
    while not labeler.started.is_set():
        assert time.monotonic() < deadline, "the generation never started"
        await asyncio.sleep(0.005)


@contextlib.asynccontextmanager
async def asgi_client(app_module):
    """Drive the real app over ASGI from the caller's own event loop.

    ``TestClient`` runs the app in a portal on a different loop, so a test could
    not hold a generation open on one side and issue a request on the other.
    """
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://conversation-lazy-labeling"
    ) as client:
        yield client


async def _auth_headers(client, channel_id: str) -> dict[str, str]:
    resp = await client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@contextlib.asynccontextmanager
async def running_lifespan(app_module):
    """Start the real app lifespan and expose its three shutdown steps.

    Same approach as ``tests/test_manager_shutdown_matrix.py``: the shutdown
    steps are closures inside ``lifespan`` and the drain deadline is hardcoded at
    30 s at its single call site, so reading them off the suspended generator's
    frame is the only way to run the production functions with a deadline a test
    can afford. Nothing is replaced or reimplemented.
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
    try:
        yield (
            locals_at_yield["wait_for_active_turns_to_complete"],
            locals_at_yield["finalize_conversations_on_shutdown"],
            locals_at_yield["stop_all_chat_sessions"],
        )
    finally:
        # The real shutdown runs on the way out, with its hardcoded 30 s drain.
        # Dropping the sessions first keeps a channel a failing test left busy
        # from holding the process for that whole window.
        for channel_id in list(app_module.session_manager._sessions):
            await app_module.session_manager.remove_session(channel_id)
        await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Nothing is labeled that was not chatted with
# ---------------------------------------------------------------------------

def test_initialize_only_channels_are_never_labeled(app_module, monkeypatch):
    """The workload this whole change exists for: sessions that only start up.

    A client that initializes with a startup action and then goes away has had no
    conversation with anybody. Under the old arrangement each of those channels
    still cost one LLM call at SIGTERM, in front of the checkpoint writes. Asserted
    by count rather than by the absence of a log line, so re-adding a call under
    the name either call site resolves fails here.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channels = [_channel(f"initonly{i}") for i in range(3)]
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as (drain, finalize, stop):
            async with asgi_client(app_module) as client:
                for channel_id in channels:
                    resp = await client.post(
                        "/initialize",
                        json={
                            "channel_id": channel_id,
                            "user_id": "u_initonly",
                            "startup_action": _add_action(),
                            "timeout_seconds": 30,
                        },
                    )
                    assert resp.status_code == 200, resp.text

                seen["states"] = {}
                for channel_id in channels:
                    runtime = await app_module.session_manager.get_session(channel_id)
                    seen["states"][channel_id] = _label_state(runtime)

            seen["remaining"] = await drain(max_wait_seconds=5)
            await finalize(seen["remaining"])
            await stop(seen["remaining"])

    asyncio.run(body())

    assert labeler.count == 0, (
        f"{labeler.count} topic-generation call(s) were made for channels that "
        "only ever ran their startup turn"
    )
    assert seen["remaining"] == []
    for channel_id, (topic, turn_count) in seen["states"].items():
        assert turn_count >= 1, (
            f"{channel_id} recorded no turn, so the no-label claim is vacuous"
        )
        assert topic == "", f"{channel_id} was labeled: {topic!r}"


def test_perform_action_does_not_label_the_conversation(app_module, monkeypatch):
    """A programmatic dispatch is not a chat.

    ``/perform_action`` names a command and its parameters directly, so a title
    generated from it describes the client's wiring rather than anything a user
    said. It still records a conversation turn, which is what makes the kind gate
    — and not the "was anything persisted" question — the thing that has to
    decide this.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("action")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            resp = await client.post(
                "/perform_action",
                headers=headers,
                json={"action": _add_action(), "timeout_seconds": 30},
            )
            seen["status"] = resp.status_code
            seen["body"] = resp.text
            runtime = await app_module.session_manager.get_session(channel_id)
            seen["state"] = _label_state(runtime)

    asyncio.run(body())

    assert seen["status"] == 200, seen["body"]
    topic, turn_count = seen["state"]
    assert turn_count >= 1, "the action recorded no turn, so nothing was gated"
    assert labeler.count == 0
    assert topic == ""


def test_a_turn_that_suspends_at_ask_user_is_not_labeled_until_it_completes(
    app_module, monkeypatch
):
    """The reason the gate is on ``TurnStatus`` and not on "was it persisted".

    ``_run_turn`` persists unconditionally once ``work_fn`` returns, and a turn
    that stopped to ask a clarifying question has already recorded half an
    exchange. Labeling there titles a conversation after a question, and under a
    schedule that refreshes geometrically that title is what a picker shows for a
    long time. The second half of this test is the point: the label is deferred,
    not lost.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("askuser")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)

        execn = await _chat_turn(
            app_module,
            runtime,
            message="add some numbers",
            status=fastworkflow.TurnStatus.AWAITING_USER,
        )
        await execn.task
        seen["suspended_status"] = execn.result.status
        seen["after_suspension"] = (labeler.count, _label_state(runtime))

        execn = await _chat_turn(app_module, runtime, message="2 and 3")
        await execn.task
        seen["after_completion"] = (labeler.count, _label_state(runtime))

    asyncio.run(body())

    assert seen["suspended_status"] is fastworkflow.TurnStatus.AWAITING_USER
    count, (topic, turn_count) = seen["after_suspension"]
    assert turn_count == 1, (
        "the suspended turn was not persisted, so nothing was gated"
    )
    assert count == 0, "a conversation was titled from a clarifying question"
    assert topic == ""

    count, (topic, turn_count) = seen["after_completion"]
    assert count == 1, "the label was lost rather than deferred"
    assert topic == "Topic 1"
    assert turn_count == 2


# ---------------------------------------------------------------------------
# Where the call runs: off the turn's critical path, and out of the drain
# ---------------------------------------------------------------------------

def test_a_completed_chat_turn_is_labeled_after_it_reaches_done(
    app_module, monkeypatch
):
    """The R7 placement, asserted from three directions at once.

    While the generation is in flight the channel must have no live execution, no
    held runtime lock and no lease — which is to say it must be absent from
    ``busy_channel_ids()``, the union predicate the shutdown drain and the 409
    guard are both built on. And the request that submitted the turn must already
    have its answer: it is released by ``done_event``, and the first thing the
    labeling path can block on is an executor await that hands the loop back.

    The generation is slow on purpose. With an instant one every ordering below
    would pass by accident.
    """
    labeler = _install(monkeypatch, app_module, _Labeler(seconds=0.4))
    channel_id = _channel("afterdone")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)

        execn = await _chat_turn(app_module, runtime, message="where is my order")
        seen["returned_at"] = time.monotonic()
        seen["exec_state_at_return"] = execn.exec_state

        await _await_generation_start(labeler)
        # Sampled from the event loop, not from the executor thread, so this is
        # the same read the drain itself would make at this instant.
        seen["busy_channels"] = app_module.session_manager.busy_channel_ids()
        seen["registry_active"] = app_module.turn_registry.has_active(channel_id)
        seen["lock_held"] = runtime.lock.locked()
        seen["generation_in_flight"] = not labeler.finished_at

        await execn.task
        seen["state"] = _label_state(runtime)

    asyncio.run(body())

    assert labeler.count == 1
    assert seen["exec_state_at_return"] is ExecState.DONE, (
        "the request was released before the turn was DONE"
    )
    assert seen["generation_in_flight"], (
        "the generation had already finished, so nothing about placement was "
        "observed; raise the stand-in's duration"
    )
    assert channel_id not in seen["busy_channels"], (
        "the channel is busy while its topic is being generated, so the shutdown "
        "drain waits on an LLM call and a concurrent request gets a 409"
    )
    assert not seen["registry_active"]
    assert not seen["lock_held"]
    assert seen["returned_at"] < labeler.finished_at[0], (
        "the turn's own response waited for the topic generation"
    )

    topic, turn_count = seen["state"]
    assert (topic, turn_count) == ("Topic 1", 1)
    assert labeler.turn_counts == [1]


def test_a_channel_labeling_a_conversation_is_not_held_open_at_sigterm(
    app_module, monkeypatch
):
    """The acceptance criterion this epic exists for, at the drain itself.

    Not "the channel looks idle" but "the real ``wait_for_active_turns_to_complete``
    reports nothing left to wait for", asked with a zero deadline while an LLM
    call is genuinely running. Under the placement the upstream issue asked for
    this returns the channel, and the process then sits in its termination grace
    period waiting on a provider.
    """
    labeler = _install(monkeypatch, app_module, _Labeler(seconds=0.5))
    channel_id = _channel("sigterm")
    seen: dict[str, Any] = {}

    async def body():
        async with running_lifespan(app_module) as (drain, _finalize, _stop):
            runtime = await _create(app_module, channel_id)
            execn = await _chat_turn(app_module, runtime, message="where is my order")

            await _await_generation_start(labeler)
            seen["remaining"] = await drain(max_wait_seconds=0)
            seen["generation_in_flight"] = not labeler.finished_at

            await execn.task
            seen["state"] = _label_state(runtime)

    asyncio.run(body())

    assert seen["generation_in_flight"], (
        "the generation finished before the drain ran, so nothing was observed"
    )
    assert seen["remaining"] == [], (
        f"the drain would wait for {seen['remaining']}, which is a topic "
        "generation inside the termination grace period"
    )
    # And the label still lands: keeping it out of the drain must not mean
    # abandoning it.
    assert labeler.count == 1
    assert seen["state"][0] == "Topic 1"


# ---------------------------------------------------------------------------
# A conversation this process never labeled
# ---------------------------------------------------------------------------

def test_a_bounce_labels_the_conversation_on_its_first_chat_turn(
    app_module, monkeypatch
):
    """A conversation whose turns outlived the process that recorded them.

    This is the majority case in the motivating workload: the session was evicted
    or the process restarted, so nothing ever titled the conversation. The seeded
    turn count is deliberately away from a refresh milestone, so the only reason
    this can label is the blank-topic sentinel — a policy that fired on
    milestones alone would leave the conversation untitled here, which is exactly
    the state that makes it unreachable through ``/activate_conversation``'s topic
    lookup.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("bounce")
    seeded_turns = 5
    seen: dict[str, Any] = {}

    # Recorded rather than commented: if the schedule ever changes so that this
    # count IS a milestone, the test stops covering the sentinel and says so.
    assert _label_milestones_reached(seeded_turns + 1) == _label_milestones_reached(
        seeded_turns
    ), "the seeded turn count sits on a refresh milestone; pick another"

    async def body():
        runtime = await _create(app_module, channel_id)
        conv_id = _seed_conversation(runtime, seeded_turns)
        seen["conv_id"] = conv_id

        # The bounce: this process forgets the channel entirely, and a fresh
        # manager rebuilds it from the durable record the way a restarted pod
        # would.
        await app_module.session_manager.remove_session(channel_id)
        fresh = ChannelSessionManager()
        fresh.workflow_path = app_module.ARGS.workflow_path
        runtime = await _create(app_module, channel_id, manager=fresh)
        seen["restored_conversation_id"] = runtime.active_conversation_id
        seen["state_before"] = _label_state(runtime)

        execn = await _chat_turn(
            app_module, runtime, message="and one more thing", manager=fresh
        )
        await execn.task
        seen["state_after"] = _label_state(runtime)

    asyncio.run(body())

    assert seen["restored_conversation_id"] == seen["conv_id"], (
        "the bounce did not restore the conversation, so this labeled a new one"
    )
    assert seen["state_before"] == ("", seeded_turns)
    assert labeler.count == 1
    assert seen["state_after"] == ("Topic 1", seeded_turns + 1)
    assert labeler.turn_counts == [seeded_turns + 1], (
        "the label was generated from something other than the whole durable "
        "conversation"
    )


def test_activating_an_unlabeled_conversation_labels_it_from_stored_turns(
    app_module, monkeypatch
):
    """Opening a conversation is the other moment a title becomes useful.

    A conversation nobody has chatted in since the bounce gets no chat turn to
    ride on, and it is the one a user is most likely to be staring at in a
    picker. The second activation is the half that keeps this from becoming a
    per-click LLM call: a conversation that already has a title is not
    re-described.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("activate")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)
            conv_id = _seed_conversation(runtime, 3)

            resp = await client.post(
                "/activate_conversation",
                headers=headers,
                json={"conversation_id": conv_id},
            )
            seen["first"] = (resp.status_code, resp.text, labeler.count)
            seen["state_after_first"] = _label_state(runtime, conv_id)

            resp = await client.post(
                "/activate_conversation",
                headers=headers,
                json={"conversation_id": conv_id},
            )
            seen["second"] = (resp.status_code, resp.text, labeler.count)
            seen["state_after_second"] = _label_state(runtime, conv_id)

    asyncio.run(body())

    status, body_text, count = seen["first"]
    assert status == 200, body_text
    assert count == 1
    assert seen["state_after_first"] == ("Topic 1", 3)
    assert labeler.turn_counts == [3], (
        "activation labeled from something other than the stored turns"
    )

    status, body_text, count = seen["second"]
    assert status == 200, body_text
    assert count == 1, (
        "activating an already-titled conversation spent another LLM call"
    )
    assert seen["state_after_second"] == ("Topic 1", 3)


# ---------------------------------------------------------------------------
# Concurrency: at most one generate per writer
# ---------------------------------------------------------------------------

def test_a_concurrent_activate_and_first_chat_generate_at_most_once(
    app_module, monkeypatch
):
    """Two triggers racing for the same conversation.

    The guard on ``/activate_conversation`` is ``_reject_if_busy``, which reads
    the registry pointer — NOT ``runtime.lock``, which is released while a request
    defers and across ``AWAITING_USER``. So the ordinary outcome of this race is a
    409 for one of the two, and this asserts the invariant rather than an
    interleaving: whichever way it lands, the conversation is titled once.

    The seeded count is chosen so that neither ordering is entitled to two calls:
    with 5 turns the chat turn takes the conversation to 6, which is not a refresh
    milestone, so an activation that wins the race does not leave a milestone for
    the chat turn to cross afterwards.
    """
    labeler = _install(monkeypatch, app_module, _Labeler(seconds=0.1))
    channel_id = _channel("race")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)
            conv_id = _seed_conversation(runtime, 5)
            runtime.active_conversation_id = conv_id
            runtime.durable_turn_count = 0

            async def activate():
                return await client.post(
                    "/activate_conversation",
                    headers=headers,
                    json={"conversation_id": conv_id},
                )

            execn, resp = await asyncio.gather(
                _chat_turn(app_module, runtime, message="one more question"),
                activate(),
            )
            await execn.task
            seen["activate_status"] = resp.status_code
            seen["activate_body"] = resp.text
            seen["turn_error"] = execn.error
            seen["state"] = _label_state(runtime, conv_id)

    asyncio.run(body())

    assert seen["turn_error"] is None
    assert seen["activate_status"] in (200, 409), seen["activate_body"]
    assert labeler.count == 1, (
        f"{labeler.count} topic-generation calls for one conversation; the "
        f"triggers did not exclude each other (turn counts {labeler.turn_counts})"
    )
    assert seen["state"][0] == "Topic 1"


def test_an_activate_during_a_labeling_generation_does_not_start_a_second_one(
    app_module, monkeypatch
):
    """The window the 409 guard does not cover, closed deliberately.

    The labeling call runs after the registry pointer is cleared, which is what
    keeps it out of the drain — and therefore also means ``_reject_if_busy`` no
    longer refuses anything. An activation arriving in that window sees a topic
    that is still blank, because the generation in flight has not written yet, and
    without a second primitive it would start its own. The per-channel labeling
    lock is that primitive, and declining rather than queueing is what keeps a
    user's activation from waiting out somebody else's LLM round trip.
    """
    labeler = _install(monkeypatch, app_module, _Labeler(seconds=0.6))
    channel_id = _channel("inflight")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)

            execn = await _chat_turn(app_module, runtime, message="where is my order")
            await _await_generation_start(labeler)
            conv_id = runtime.active_conversation_id

            started = time.monotonic()
            resp = await client.post(
                "/activate_conversation",
                headers=headers,
                json={"conversation_id": conv_id},
            )
            seen["activate_seconds"] = time.monotonic() - started
            seen["activate_status"] = resp.status_code
            seen["activate_body"] = resp.text
            seen["count_during"] = labeler.count
            seen["generation_in_flight"] = not labeler.finished_at

            await execn.task
            seen["state"] = _label_state(runtime, conv_id)

    asyncio.run(body())

    assert seen["generation_in_flight"], (
        "the generation finished before the activation arrived, so the window "
        "this covers was never open"
    )
    assert seen["activate_status"] == 200, seen["activate_body"]
    assert seen["count_during"] == 1, (
        "the activation started a second generation for a conversation that was "
        "already being labeled"
    )
    assert seen["activate_seconds"] < 0.5, (
        f"the activation waited {seen['activate_seconds']:.2f}s, so it queued "
        "behind the in-flight generation instead of declining"
    )
    assert labeler.count == 1
    assert seen["state"] == ("Topic 1", 1)


# ---------------------------------------------------------------------------
# Failure: never fatal to the trigger, always retryable
# ---------------------------------------------------------------------------

def test_a_failed_generation_leaves_the_trigger_successful_and_the_topic_retryable(
    app_module, monkeypatch
):
    """A label nobody asked for must not fail the thing that triggered it.

    The chat turn succeeded and the activated conversation is activated; what is
    missing is a title, and the blank topic is precisely the sentinel that makes
    the next eligible trigger try again. The third act proves that: with a
    working provider the conversation is titled, without any special recovery
    path.

    Contrast ``/new_conversation``, which stays strict and answers 500 — see
    ``tests/test_conversation_topic_generation_bounds.py``. Rotate is an explicit
    user action that archives the thread, so a silent unlabeled archive is worse
    than a refusal; these triggers are opportunistic, so it is the other way
    round.
    """
    failing = _install(monkeypatch, app_module, _Labeler(fail=True))
    channel_id = _channel("failing")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)

            execn = await _chat_turn(app_module, runtime, message="where is my order")
            await execn.task
            seen["turn_error"] = execn.error
            seen["turn_status"] = execn.result.status
            seen["after_turn"] = (failing.count, _label_state(runtime))

            conv_id = runtime.active_conversation_id
            resp = await client.post(
                "/activate_conversation",
                headers=headers,
                json={"conversation_id": conv_id},
            )
            seen["activate"] = (resp.status_code, resp.text)
            seen["after_activate"] = (failing.count, _label_state(runtime, conv_id))

            working = _install(monkeypatch, app_module, _Labeler())
            execn = await _chat_turn(app_module, runtime, message="anything else")
            await execn.task
            seen["after_recovery"] = (working.count, _label_state(runtime, conv_id))

    asyncio.run(body())

    assert seen["turn_error"] is None, "a failed label failed the turn"
    assert seen["turn_status"] is fastworkflow.TurnStatus.COMPLETED
    assert seen["after_turn"] == (1, ("", 1))

    status, body_text = seen["activate"]
    assert status == 200, body_text
    assert seen["after_activate"] == (2, ("", 1)), (
        "the activation either did not retry the blank topic or wrote one anyway"
    )

    count, (topic, turn_count) = seen["after_recovery"]
    assert count == 1
    assert topic == "Topic 1", "the conversation never recovered a title"
    assert turn_count == 2


# ---------------------------------------------------------------------------
# The refresh schedule
# ---------------------------------------------------------------------------

def test_the_label_is_refreshed_geometrically_and_never_on_every_turn(
    app_module, monkeypatch
):
    """The test that stops a future change from making this per-turn.

    Both halves of the R6 argument in one count. It must fire more than once,
    because a title generated from the first exchange and never refreshed
    describes a 40-turn thread by its opening line — the same error as
    summarizing the 20-turn memory window, from the other end. And it must not
    fire per turn, because that is one LLM call per user message forever.

    The turn counts are asserted, not just the total: they are the conversation's
    durable turn count at each generation, so this pins *when* the refreshes
    happen and shows each one saw more of the thread than the last.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("schedule")
    total_turns = 17
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)
        for index in range(total_turns):
            execn = await _chat_turn(app_module, runtime, message=f"question {index}")
            await execn.task
        seen["state"] = _label_state(runtime)

    asyncio.run(body())

    assert seen["state"][1] == total_turns, "not every turn was recorded"
    assert labeler.turn_counts == [1, 4, 16], (
        f"{total_turns} chat turns produced generations at turn counts "
        f"{labeler.turn_counts}"
    )
    assert 1 < labeler.count < total_turns
    assert seen["state"][0] == "Topic 3", (
        "the newest label is not the one stored, so a refresh did not take"
    )


def test_a_rotate_labels_even_when_the_refresh_schedule_says_it_is_not_due(
    app_module, monkeypatch
):
    """``/new_conversation`` shares the helper without inheriting its schedule.

    Rotate is the trigger that always produces a label: it archives the thread,
    and a thread the user has finished with is the one most likely to be picked
    out of a list later, so it is the worst one to leave untitled. (Not because
    activation needs the topic — ``ActivateConversationRequest`` carries a
    ``conversation_id``, and ``get_conversation_by_topic`` has no production
    caller. A topic is a label for whoever is choosing, not a lookup key.) So
    rotate asks with ``force``, which skips the due check while still going
    through the same chokepoint — the same blank-topic policy, the same executor
    offload and the same mutual exclusion with a lazy fill.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("rotate")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)

            for index in range(2):
                execn = await _chat_turn(
                    app_module, runtime, message=f"question {index}"
                )
                await execn.task
            conv_id = runtime.active_conversation_id
            seen["after_turns"] = (labeler.count, _label_state(runtime, conv_id))

            resp = await client.post("/new_conversation", headers=headers)
            seen["rotate"] = (resp.status_code, resp.text)
            seen["after_rotate"] = (labeler.count, _label_state(runtime, conv_id))
            seen["rotated_to"] = runtime.active_conversation_id
            seen["conv_id"] = conv_id

    asyncio.run(body())

    count, (topic, turn_count) = seen["after_turns"]
    assert (count, topic, turn_count) == (1, "Topic 1", 2), (
        "the second chat turn refreshed the label, so the rotate below would "
        "not be proving anything about force"
    )

    status, body_text = seen["rotate"]
    assert status == 200, body_text
    count, (topic, turn_count) = seen["after_rotate"]
    assert count == 2, "the rotate did not label, or labeled twice"
    assert topic == "Topic 2"
    assert turn_count == 2
    assert seen["rotated_to"] > seen["conv_id"], "the rotate did not rotate"


def test_a_rotate_persists_unsaved_turns_before_it_labels_them(
    app_module, monkeypatch
):
    """Hazard: the one state a rotate could not escape, and the turn loss behind it.

    A turn persists from inside ``_run_turn``'s own try, so a store failure there
    leaves in-memory turns with nothing recorded. If a previous rotate had already
    reserved the next conversation id, the channel then holds turns, a nonzero
    ``active_conversation_id``, and no record under it. Labeling that state paid
    for a generation and then raised ``ValueError: Conversation N not found`` out
    of ``update_conversation_topic_summary``, surfacing as a 500 — so the rotate
    that would have cleaned the state up was the one operation unavailable, and
    every retry bought another provider call to fail the same way.

    Persisting before labeling closes both halves: the record exists, so the write
    is well-defined, and the turns are durable before the rotate clears the
    history that held them. This asserts the turns SURVIVE, not merely that the
    request returns 200 — a rotate that answered 200 by discarding them would be
    the worse bug.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("unsaved")
    seen: dict[str, Any] = {}

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await app_module.session_manager.get_session(channel_id)

            # Reproduce the state directly: an id reserved with no record written
            # (what a rotate leaves), plus a turn that never reached the store.
            runtime.active_conversation_id = (
                runtime.conversation_store.reserve_next_conversation_id()
            )
            runtime.durable_turn_count = 0
            runtime.execution_context.append_conversation_turn(
                "a turn whose persist never landed"
            )
            conv_id = runtime.active_conversation_id
            seen["record_before"] = runtime.conversation_store.get_conversation(conv_id)

            resp = await client.post("/new_conversation", headers=headers)
            seen["rotate"] = (resp.status_code, resp.text)
            seen["generates"] = labeler.count
            seen["label_after"] = _label_state(runtime, conv_id)
            stored = runtime.conversation_store.get_conversation(conv_id)
            seen["turns_after"] = [
                turn["conversation summary"] for turn in (stored or {}).get("turns", [])
            ]
            seen["rotated_to"] = runtime.active_conversation_id
            seen["conv_id"] = conv_id

    asyncio.run(body())

    assert seen["record_before"] is None, (
        "the reserved id already had a record, so this test is not exercising the "
        "reserved-but-unwritten state it exists for"
    )

    status, body_text = seen["rotate"]
    assert status == 200, (
        f"the rotate failed out of the reserved-but-unwritten state: {body_text}"
    )

    assert seen["turns_after"] == ["a turn whose persist never landed"], (
        "the rotate cleared the in-memory history without recording the turn it "
        "held, which is the loss persisting first exists to prevent"
    )

    topic, turn_count = seen["label_after"]
    assert seen["generates"] == 1, "the rotate did not label, or labeled twice"
    assert topic == "Topic 1"
    assert turn_count == 1
    assert seen["rotated_to"] > seen["conv_id"], "the rotate did not rotate"


# ---------------------------------------------------------------------------
# The streaming lifecycle: run_owned_turn and its on_done (fix-dzs.10)
# ---------------------------------------------------------------------------

async def _streaming_turn(
    app_module,
    runtime: ChannelRuntime,
    *,
    message: str,
    on_done=None,
    on_done_factory=None,
    fail: bool = False,
):
    """Drive the real ``run_owned_turn`` the way /invoke_agent_stream drives it.

    The work function is async and records its conversation turn through the
    execution context's own API, which is what the streaming endpoint's
    ``streaming_work`` does either side of emitting its events. Driving the
    lifecycle directly rather than the endpoint is deliberate: the endpoint's own
    query would run the NLU pipeline, and a turn that fails there sets
    ``execn.error`` and is not labeled at all, so the ordering under test would
    never be reached.
    """
    registry = app_module.turn_registry

    async def work():
        if fail:
            raise RuntimeError("the streaming work failed")
        runtime.execution_context.append_conversation_turn(message)
        return fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=fastworkflow.TurnStatus.COMPLETED,
            answer="streamed",
        )

    async def owned(execn) -> None:
        # on_done_factory closes the callback over ``execn``, which is how the
        # endpoint builds its own finish_stream: the callback has no parameters,
        # so reading the execution's outcome is only possible from the closure.
        await app_module.run_owned_turn(
            runtime,
            registry,
            execn,
            work,
            app_module.session_manager,
            on_done=on_done_factory(execn) if on_done_factory else on_done,
        )

    execn = await registry.start_or_get_active(
        runtime.channel_id,
        kind="invoke_agent_stream",
        idempotency_key=f"stream-{uuid.uuid4().hex[:8]}",
        run_turn=lambda e: asyncio.create_task(owned(e)),
    )
    await execn.task
    return execn


def test_a_streaming_turn_labels_its_conversation(app_module, monkeypatch):
    """Coverage the other tests do not reach: /invoke_agent_stream is a chat turn.

    ``run_owned_turn`` is a separate lifecycle from ``_run_turn`` — streaming has
    to emit while the work runs, so it cannot use the executor path — and it is
    the lifecycle behind ``invoke_agent``, the MCP-exposed tool. Hooking only
    ``_run_turn`` would leave every deployment whose clients stream permanently
    unlabeled, which no other test here would notice.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("streamlabel")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)
        await _streaming_turn(app_module, runtime, message="stream this")
        seen["generates"] = labeler.count
        seen["label"] = _label_state(runtime)

    asyncio.run(body())

    topic, turn_count = seen["label"]
    assert seen["generates"] == 1, "a streaming chat turn did not label its conversation"
    assert topic == "Topic 1"
    assert turn_count == 1


def test_the_stream_ends_before_the_label_generation_finishes(
    app_module, monkeypatch
):
    """THE test for fix-dzs.10. Without ``on_done`` this is what regresses.

    The label used to be the last thing in ``run_owned_turn``, and the endpoint
    emits its end-of-stream sentinel only once that returns — so EOF waited on an
    LLM round trip. The answer was already on the wire, so nothing was wrong with
    the content; what waited was the end of the body. An MCP client reads the
    whole stream as its tool result, so it paid the generation on every labeled
    turn, and a client with a shorter idle-read timeout could fail a turn whose
    answer it had already received.
    """
    labeler = _install(monkeypatch, app_module, _Labeler(seconds=0.6))
    channel_id = _channel("streameof")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)
        stream_ended_at: list[float] = []

        async def finish_stream() -> None:
            stream_ended_at.append(time.monotonic())

        await _streaming_turn(
            app_module, runtime, message="stream this", on_done=finish_stream
        )
        seen["stream_ended_at"] = stream_ended_at
        seen["generation_finished_at"] = list(labeler.finished_at)
        seen["generates"] = labeler.count

    asyncio.run(body())

    assert len(seen["stream_ended_at"]) == 1, "the stream was never closed out"
    assert seen["generates"] == 1, "no label was generated, so there was no wait to avoid"
    assert seen["stream_ended_at"][0] < seen["generation_finished_at"][0], (
        "the stream ended only after the label generation finished, so a client "
        "reading to EOF still pays for the label"
    )


def test_on_done_can_see_a_failed_turn_before_the_stream_ends(
    app_module, monkeypatch
):
    """``on_done`` runs after the error paths, which is what makes it usable.

    The streaming endpoint emits its terminal error event from inside ``on_done``
    and then the sentinel, because the body's drain loop stops at the sentinel and
    never reads anything after it. That is only correct if ``execn.error`` is
    already set by the time ``on_done`` is called.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("streamerr")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)
        observed: list[tuple[Optional[str], Any]] = []

        def make_finish(execn):
            async def finish_stream() -> None:
                # Exactly what the endpoint's own finish_stream reads to decide
                # whether a terminal error event goes out before the sentinel.
                observed.append((execn.error, execn.exec_state))

            return finish_stream

        await _streaming_turn(
            app_module,
            runtime,
            message="never recorded",
            on_done_factory=make_finish,
            fail=True,
        )
        seen["observed"] = observed
        seen["generates"] = labeler.count

    asyncio.run(body())

    assert len(seen["observed"]) == 1, "on_done was not called for a failed turn"
    error, state = seen["observed"][0]
    assert error is not None, (
        "on_done ran before execn.error was set, so the endpoint would emit its "
        "sentinel with no terminal error event and the client would see a "
        "silently truncated stream"
    )
    assert state is ExecState.DONE
    assert seen["generates"] == 0, (
        "a turn that raised was labeled; only COMPLETED and FAILED turns are"
    )


def test_a_failing_on_done_does_not_skip_the_label(app_module, monkeypatch):
    """A caller's delivery problem is not the lifecycle's problem.

    ``on_done`` is called from a ``finally``, so an exception escaping it would
    skip the trim and the label and surface as a lost task exception rather than
    anything actionable. It gets its own try for that reason.
    """
    labeler = _install(monkeypatch, app_module, _Labeler())
    channel_id = _channel("streamboom")
    seen: dict[str, Any] = {}

    async def body():
        runtime = await _create(app_module, channel_id)

        async def exploding_finish() -> None:
            raise RuntimeError("the client vanished mid-flush")

        await _streaming_turn(
            app_module, runtime, message="stream this", on_done=exploding_finish
        )
        seen["generates"] = labeler.count
        seen["label"] = _label_state(runtime)

    asyncio.run(body())

    topic, _ = seen["label"]
    assert seen["generates"] == 1, "a failing on_done skipped the conversation label"
    assert topic == "Topic 1"
