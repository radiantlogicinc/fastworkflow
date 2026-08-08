"""Integration tests for channel checkpointing through the real eviction path.

`tests/test_checkpoint_store.py` already pins down the store's own invariants in
isolation. What it cannot see is the seam above it: `ChannelSessionManager`
deciding *whether* a live runtime may be retired, `checkpoint.publish` projecting
one, and `_create_user_runtime` rebuilding a cold session from what landed. Every
hazard below lives in that seam, so every test here drives a real
`ChannelSessionManager` at a cap of one with real runtimes over real workflows —
the over-capacity sweep actually executes rather than being simulated, and the
restore that follows is the same code path a cold worker takes.

Two things are deliberately load-bearing in how these are written:

* **No mocks.** Application state is established by running the workflows' own
  commands as direct actions, which bypasses NLU and needs no trained model.
* **Nothing may keep a reference to a runtime being evicted.** `Workflow.create`
  keys a weak global registry by workflow id, so a surviving reference hands the
  cold session the *same* workflow object back — root context already set — and
  the restore raises instead of rebuilding. The server has no such reference; a
  test that keeps one is testing something the server never does.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.run_fastapi_mcp.turns import (
    CREDENTIAL_CONTEXT_KEY,
    TurnRegistry,
    installed_credential,
    submit_turn,
)
from fastworkflow.run_fastapi_mcp.utils import (
    ChannelSessionManager,
    ensure_user_runtime_exists,
    persist_pending_after_turn,
)
from fastworkflow.serialization_hooks import ProjectionOutcome
from fastworkflow.utils.logging import logger
from fastworkflow.workflow import _WORKFLOW_REGISTRY

TESTS_DIR = Path(__file__).parent

# hello_world has no command-context object at all, which makes it the right
# workflow for everything about workflow.context, launch reconciliation and
# startup: its projection is NO_CONTEXT, so nothing else pins the session and a
# failure to evict can only be the thing under test.
HELLO_WORLD = str(TESTS_DIR / "hello_world_workflow")

# todo_list_workflow's TodoListManager context class implements the full hook
# set (get_state/from_state/get_locator/find_by_locator), so it is the one that
# can exercise a real application-object round trip.
TODO_LIST = str(TESTS_DIR / "todo_list_workflow")

TODO_LIST_JSON = TESTS_DIR / "todo_list_workflow" / "application" / "todo_list.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def speedict_folder(tmp_path):
    """Point fastWorkflow's durable root at a private temp directory.

    Every checkpoint namespace these tests read is derived from
    SPEEDDICT_FOLDERNAME, so sharing the developer's real folder would let one
    test observe another's records — or the developer's — and would leave
    channel records behind after the run.

    The previous env is restored on the way out: an interpreter left pointing at
    a deleted temp directory is a trap for whatever test file runs next.
    """
    for path in (HELLO_WORLD, TODO_LIST):
        if not os.path.isdir(path):
            pytest.skip(f"golden workflow not found at {path}")

    previous_env = fastworkflow._env_vars
    speedict = tmp_path / "speedict"
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": str(speedict)})
    fastworkflow.RoutingRegistry.clear_registry()
    # Pin warnings are throttled per (workflow, reason) for the life of the
    # process, so a test that asserts on one has to start from a clean slate.
    checkpoint.reset_warnings()
    try:
        yield speedict
    finally:
        checkpoint.reset_warnings()
        fastworkflow.RoutingRegistry.clear_registry()
        fastworkflow.init(previous_env or {})


@pytest.fixture
def todo_list_json_preserved():
    """Undo the writes create_todo_list makes to the workflow's own JSON file."""
    original = TODO_LIST_JSON.read_text()
    yield
    TODO_LIST_JSON.write_text(original)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _cap_one_manager() -> ChannelSessionManager:
    """A manager that is over target the moment a second session appears."""
    return ChannelSessionManager(max_live_sessions=1)


def _action(command_name: str, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(command_name=command_name, parameters=parameters)


def _add_action() -> fastworkflow.Action:
    return _action("add_two_numbers", first_num=2.0, second_num=3.0)


async def _create(
    manager: ChannelSessionManager,
    channel_id: str,
    workflow_path: str,
    *,
    context: dict | None = None,
    startup_action: fastworkflow.Action | None = None,
) -> None:
    """Build a runtime exactly as the server's /initialize does."""
    await ensure_user_runtime_exists(
        channel_id=channel_id,
        session_manager=manager,
        workflow_path=workflow_path,
        context=context,
        startup_action=startup_action,
        run_startup=startup_action is not None,
    )


def _record(manager: ChannelSessionManager, channel_id: str, workflow_path: str):
    """This channel's committed record, read the way a cold worker reads it."""
    return manager.checkpoint_store.load_for_adoption(
        deployment_id=checkpoint.deployment_id(),
        workflow_fingerprint=checkpoint.workflow_fingerprint(workflow_path),
        channel_id=channel_id,
    )


def _checkpoint_files(speedict: Path) -> dict[str, bytes]:
    """Every byte under the checkpoint root, keyed by relative path.

    Raw bytes rather than parsed records: a value that leaked into a nested
    structure, a key name, or a manifest is still on disk regardless of which
    field a reader would have deserialized it into.
    """
    root = speedict / "channel_checkpoints"
    blobs: dict[str, bytes] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            blobs[str(path.relative_to(root))] = path.read_bytes()
    return blobs


@contextmanager
def _captured_warnings():
    """Collect WARNING records from the fastWorkflow logger.

    caplog cannot see them: that logger sets propagate=False, so its records
    never reach the root handler pytest installs.
    """
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _named_keys(message: str, label: str) -> list[str]:
    """The key names a reconciliation warning attributes to ``label``."""
    match = re.search(rf"\b{label}=\[(.*?)\]", message)
    assert match, f"warning did not report {label}=[...]: {message}"
    return [part.strip().strip("'\"") for part in match[1].split(",") if part.strip()]


def _count_calls(call_log: str) -> int:
    if not os.path.isfile(call_log):
        return 0
    with open(call_log) as handle:
        return sum(bool(line.strip()) for line in handle)


NO_HOOK_CONTEXT = '''
class Context:
    @classmethod
    def get_parent(cls, command_context_object) -> None:
        return None
'''

POKE_COMMAND = '''
import fastworkflow


class ResponseGenerator:
    def __call__(self, workflow, command) -> fastworkflow.CommandOutput:
        return fastworkflow.CommandOutput(
            command_responses=[fastworkflow.CommandResponse(response="poked")]
        )
'''


class Widget:
    """Anchor whose context class never implemented get_state."""

    def __init__(self) -> None:
        self.rows = ["a", "b"]


def _write_scratch_workflow(tmp_path: Path) -> str:
    """A workflow whose only context class has no serialization hooks.

    Written from source here rather than copied from a bundled workflow: none of
    the bundled ones both set a command-context object and lack a hook, and
    copying one would drag its trained ``___command_info`` along.
    """
    folderpath = tmp_path / "scratch_workflow"
    context_folder = folderpath / "_commands" / "Widget"
    context_folder.mkdir(parents=True)
    (context_folder / "_Widget.py").write_text(NO_HOOK_CONTEXT)
    (context_folder / "poke.py").write_text(POKE_COMMAND)
    (folderpath / "_commands" / "context_inheritance_model.json").write_text("{}")
    return str(folderpath)


# ---------------------------------------------------------------------------
# 1. No-eviction control
# ---------------------------------------------------------------------------

def test_below_the_cap_nothing_is_written_and_nothing_is_evicted(speedict_folder):
    """Guards against making every session pay for a bound only overflow needs.

    Snapshotting or retiring on any trigger other than being over target would
    put a multi-hundred-kilobyte write on a path that had no memory problem, and
    would close contexts clients are still using.
    """
    manager = ChannelSessionManager(max_live_sessions=2)
    first, second = _channel("under"), _channel("cap")

    async def body():
        await _create(manager, first, HELLO_WORLD)
        await _create(manager, second, HELLO_WORLD)
        return set(manager._sessions)

    live = asyncio.run(body())

    assert live == {first, second}
    assert _checkpoint_files(speedict_folder) == {}


# ---------------------------------------------------------------------------
# 2. Evict and rehydrate round trip
# ---------------------------------------------------------------------------

def test_an_evicted_channel_returns_with_its_application_state(
    speedict_folder, todo_list_json_preserved
):
    """The whole point of the mechanism: eviction must not be data loss.

    A command-context object is an ordinary Python instance no generic
    projection can reach, so without the author's hooks the LRU would either
    pin forever or drop the application's state on the floor. This drives the
    real sweep and then re-creates the channel cold, which is what a request
    arriving on a different worker does.
    """
    manager = _cap_one_manager()
    channel_id, newcomer = _channel("roundtrip"), _channel("newcomer")

    async def body():
        await _create(manager, channel_id, TODO_LIST, startup_action=_action("startup"))
        runtime = await manager.get_session(channel_id)
        ctx = runtime.execution_context
        ctx.process_action(
            _action("TodoListManager/create_todo_list", description="chores")
        )
        ctx.process_action(
            _action(
                "TodoList/add_child_todoitem",
                description="laundry",
                assign_to="Sam",
                is_complete=False,
            )
        )
        incarnation = runtime.session_incarnation
        live_root_id = id(ctx.app_workflow.root_command_context)

        # See the module docstring: a surviving reference makes the cold session
        # adopt this very workflow instead of rebuilding one.
        del ctx, runtime
        gc.collect()

        await _create(manager, newcomer, TODO_LIST)
        assert channel_id not in manager._sessions, "the sweep did not retire it"
        gc.collect()

        # The same /initialize shape a returning client sends, startup action
        # included. If startup re-ran it would raise, because restore has
        # already set the root context and a Workflow accepts one only once.
        await _create(manager, channel_id, TODO_LIST, startup_action=_action("startup"))
        return await manager.get_session(channel_id), incarnation, live_root_id

    restored, incarnation, live_root_id = asyncio.run(body())

    # Adoption, not a fresh lifetime: the cold session takes on the incarnation
    # the record names, so the record stays addressable rather than being
    # quarantined as a channel-id reuse.
    assert restored.session_incarnation == incarnation

    workflow = restored.execution_context.app_workflow
    manager_obj = workflow.root_command_context
    assert id(manager_obj) != live_root_id, "this is the pre-eviction object, not a restore"
    assert {id_: lst.description for id_, lst in manager_obj.lists.items()} == {
        1: "groceries",
        2: "chores",
    }

    chores = manager_obj.get_todo_list(2)
    laundry = chores.get_child_by_id(1)
    assert laundry.description == "laundry"
    assert laundry.assign_to == "Sam"

    # Identity, not equality. Two independent snapshots of the same node would
    # restore as two objects and current_command_context would no longer be a
    # node inside root_command_context — the application would then navigate one
    # tree while its commands mutated another.
    assert workflow.current_command_context is laundry
    assert workflow.command_context_for_response_generation is chores
    assert laundry.parent is chores
    assert chores.parent is manager_obj


# ---------------------------------------------------------------------------
# 3. Pinning: hook absent
# ---------------------------------------------------------------------------

def test_a_context_without_hooks_is_never_evicted(speedict_folder, tmp_path):
    """Guards against trading an allocation problem for silent state loss.

    A context class with no get_state cannot be projected, and a workflow whose
    author has not consented to a snapshot must keep its session alive. Sitting
    over target is visible and metered; a runtime dropped after a projection
    that quietly returned nothing is not.
    """
    scratch = _write_scratch_workflow(tmp_path)
    manager = _cap_one_manager()
    pinned, newcomer = _channel("nohook"), _channel("newcomer")

    async def body():
        await _create(manager, pinned, scratch)
        runtime = await manager.get_session(pinned)
        runtime.execution_context.app_workflow.root_command_context = Widget()

        eligibility = checkpoint.assess(runtime)
        await _create(manager, newcomer, HELLO_WORLD)
        return eligibility, set(manager._sessions)

    eligibility, live = asyncio.run(body())

    assert not eligibility.evictable
    assert eligibility.projection.outcome is ProjectionOutcome.HOOK_ABSENT
    assert "get_state" in eligibility.reason

    assert live == {pinned, newcomer}, "the cache must stay over target, not evict"
    assert _record(manager, pinned, scratch) is None


# ---------------------------------------------------------------------------
# 4. Pinning: awaiting_user
# ---------------------------------------------------------------------------

def test_a_suspended_channel_is_evicted_now_that_its_snapshot_is_complete(
    speedict_folder,
):
    """Decision 26 reversed (fix-g03.25): the snapshot is no longer incomplete.

    This test previously asserted the opposite. It pinned because the suspended
    snapshot carried neither the logical-turn accumulator nor the CME
    continuation keys, so restoring one lost everything the turn produced before
    it asked its question. Schema 2 carries both, and pinning every suspended
    session was itself unbounded: a user who walks away mid-question held a
    runtime for the life of the process, with no idle TTL to reclaim it.

    This workflow holds no context object, so nothing else influences the sweep
    — suspension is the only variable.
    """
    manager = _cap_one_manager()
    suspended, newcomer = _channel("awaiting"), _channel("newcomer")

    async def body():
        await _create(manager, suspended, HELLO_WORLD)
        runtime = await manager.get_session(suspended)
        ctx = runtime.execution_context

        # Suspending for real needs an LLM turn. These are the fields the ctx
        # itself treats as authoritative, set the way tests/test_fastapi_topology_b.py
        # does, and then persisted through the server's own post-turn writer.
        ctx._awaiting_user = True
        ctx._pending_clarification_request = "which one?"
        ctx._begin_turn("do the thing")
        ctx.append_ask_user_entry("which one?")
        persist_pending_after_turn(
            manager,
            runtime,
            fastworkflow.TurnOutput(
                turn_key=ctx._turn_key,
                status=fastworkflow.TurnStatus.AWAITING_USER,
                answer="which one?",
                command_outputs=list(ctx._turn_outputs),
            ),
        )

        eligibility = checkpoint.assess(runtime, manager.session_state_store)
        turn_key = ctx._turn_key
        await _create(manager, newcomer, HELLO_WORLD)
        return eligibility, set(manager._sessions), turn_key

    eligibility, live, turn_key = asyncio.run(body())

    assert eligibility.evictable, eligibility.reason
    assert live == {newcomer}, "a suspended channel should no longer defeat the cap"
    assert _record(manager, suspended, HELLO_WORLD) is not None

    # Evicting is only safe because the suspended state is still restorable.
    blob = manager.session_state_store.load(suspended)
    assert blob is not None
    assert blob["awaiting_user"] is True
    assert blob["turn"]["key"] == turn_key
    assert blob["turn"]["outputs"][0]["command_name"] == "ask_user"


# ---------------------------------------------------------------------------
# 5. Publish before pop
# ---------------------------------------------------------------------------

def test_a_failed_checkpoint_write_leaves_the_runtime_live_and_open(speedict_folder):
    """Guards against "best effort, then evict", which is how state disappears.

    If the pop happened before the write succeeded, a refused snapshot would
    take the only copy of the application's state with it. The failure is forced
    the way the design's strictness boundary produces it in the field: a value
    in workflow.context that the strict serializer refuses to encode rather than
    silently stringify.
    """
    manager = _cap_one_manager()
    channel_id, newcomer = _channel("failedwrite"), _channel("newcomer")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        runtime.execution_context.app_workflow.context["opaque"] = object()
        cme_id = runtime.execution_context._cme_workflow.id

        await _create(manager, newcomer, HELLO_WORLD)

        assert channel_id in manager._sessions, "a runtime was popped after a failed write"
        # close() unregisters the cme workflow. A closed context still executes
        # commands, so "it still works" is not evidence on its own.
        assert cme_id in _WORKFLOW_REGISTRY, "the runtime was closed after a failed write"

        output = runtime.execution_context.process_action(_add_action())
        return output, set(manager._sessions)

    output, live = asyncio.run(body())

    assert "5.0" in output.command_responses[0].response
    assert live == {channel_id, newcomer}
    assert _record(manager, channel_id, HELLO_WORLD) is None
    assert _checkpoint_files(speedict_folder) == {}, "a partial generation was left behind"


# ---------------------------------------------------------------------------
# 6. Empty context is a real snapshot
# ---------------------------------------------------------------------------

def test_a_context_the_application_emptied_does_not_refill_from_launch(speedict_folder):
    """Guards against treating "no keys" as "nothing was saved".

    Merging a snapshot over the launch configuration instead of replacing it
    resurrects every key the application deliberately deleted, and the
    application has no way to tell that its deletion was undone.
    """
    manager = _cap_one_manager()
    channel_id, newcomer = _channel("emptied"), _channel("newcomer")
    launch = {"tenant": "acme", "region": "us-east"}
    manager.set_launch_context(launch)

    async def body():
        await _create(manager, channel_id, HELLO_WORLD, context=dict(launch))
        runtime = await manager.get_session(channel_id)
        runtime.execution_context.app_workflow.context.clear()
        del runtime
        gc.collect()

        await _create(manager, newcomer, HELLO_WORLD)
        gc.collect()

        await _create(manager, channel_id, HELLO_WORLD, context=dict(launch))
        restored = await manager.get_session(channel_id)
        return dict(restored.execution_context.app_workflow.context or {})

    restored_context = asyncio.run(body())

    assert restored_context == {}


# ---------------------------------------------------------------------------
# 7. Three-way launch reconciliation
# ---------------------------------------------------------------------------

def test_restore_reconciles_launch_configuration_three_ways(speedict_folder):
    """Guards against a snapshot outranking a redeployment, and the reverse.

    Attribution needs all three inputs. With only the saved state and the
    current launch you cannot tell an operator's edit from an application's
    write, so a digest — which can say that launch configuration changed but not
    which key — can only guess, and a guess here silently reverts either the
    deployment or the application. All five outcomes are exercised in one
    restore, with distinct key names so the warning's attribution is unambiguous.
    """
    manager = _cap_one_manager()
    channel_id, newcomer = _channel("reconcile"), _channel("newcomer")

    launch_before = {
        "theme": "dark",           # untouched by everyone
        "edition": "first",        # the operator changes it
        "retired": "old-endpoint", # the operator removes it
        "owner": "ops",            # the application changes it; launch does not
        "region": "us-east",       # both change it
    }
    launch_after = {
        "theme": "dark",
        "edition": "second",
        "owner": "ops",
        "region": "eu-west",
        "feature_flag": "on",      # the operator adds it
    }

    async def body():
        manager.set_launch_context(launch_before)
        await _create(manager, channel_id, HELLO_WORLD, context=dict(launch_before))
        runtime = await manager.get_session(channel_id)
        context = runtime.execution_context.app_workflow.context
        context["owner"] = "team-a"
        context["region"] = "ap-south"
        context["app_note"] = "written by the application"
        del context, runtime
        gc.collect()

        await _create(manager, newcomer, HELLO_WORLD)
        gc.collect()

        # Redeploy: same channel, different launch configuration.
        manager.set_launch_context(launch_after)
        with _captured_warnings() as records:
            await _create(manager, channel_id, HELLO_WORLD, context=dict(launch_after))
        restored = await manager.get_session(channel_id)
        return dict(restored.execution_context.app_workflow.context or {}), records

    merged, records = asyncio.run(body())

    assert merged == {
        "theme": "dark",
        "edition": "second",                  # launch changed -> launch wins
        "owner": "team-a",                    # launch unchanged -> application keeps it
        "region": "eu-west",                  # both changed -> resolved to launch
        "feature_flag": "on",                 # added to launch -> present
        "app_note": "written by the application",
    }
    assert "retired" not in merged, "a key the operator removed came back"

    reconciliations = [
        record.getMessage()
        for record in records
        if "reconciliation" in record.getMessage()
    ]
    assert len(reconciliations) == 1, reconciliations
    message = reconciliations[0]

    assert channel_id in message
    assert _named_keys(message, "changed") == ["edition", "region"]
    assert _named_keys(message, "removed") == ["retired"]
    # The one line a digest-only implementation cannot produce: it knows a
    # conflict happened only if it can attribute both sides.
    assert _named_keys(message, "conflicts_resolved_to_launch") == ["region"]
    assert "theme" not in message, "an untouched key was reported as reconciled"
    assert "app_note" not in message


# ---------------------------------------------------------------------------
# 8. Credentials at rest
# ---------------------------------------------------------------------------

def test_the_callers_credential_is_never_written_to_disk(speedict_folder):
    """Guards against a request-scoped secret becoming durable.

    workflow.context is documented to carry the caller's JWT while a turn runs,
    and checkpointing writes workflow.context. Once those two facts meet, the
    credential lands on disk in the clear — stale, long-lived, and readable by
    anything with the volume. Asserting on the record's top-level keys would not
    catch a copy that travelled inside another value, so this reads the bytes.
    """
    manager = _cap_one_manager()
    channel_id, newcomer = _channel("credential"), _channel("newcomer")
    token = f"eyJhbGciOiJSUzI1NiJ9.not-a-real-jwt-{uuid.uuid4().hex}"
    launch = {"tenant": "acme"}
    manager.set_launch_context(launch)

    async def body():
        await _create(manager, channel_id, HELLO_WORLD, context=dict(launch))
        runtime = await manager.get_session(channel_id)

        with installed_credential(runtime, token):
            context = runtime.execution_context.app_workflow.context
            assert context[CREDENTIAL_CONTEXT_KEY] == token, (
                "the credential was not in shared state, so nothing was at risk "
                "and this test would prove nothing"
            )
            del context, runtime
            await _create(manager, newcomer, HELLO_WORLD)

        assert channel_id not in manager._sessions, "no snapshot was taken"

    asyncio.run(body())

    record = _record(manager, channel_id, HELLO_WORLD)
    assert record is not None
    # The rest of the launch context did persist, so the walk below is over a
    # record with real content rather than an empty tree.
    assert record.context["workflow_context"]["tenant"] == "acme"

    blobs = _checkpoint_files(speedict_folder)
    assert blobs, "nothing was written, so finding no credential means nothing"

    key_bytes = CREDENTIAL_CONTEXT_KEY.encode()
    token_bytes = token.encode()
    leaked_key = sorted(name for name, data in blobs.items() if key_bytes in data)
    leaked_token = sorted(name for name, data in blobs.items() if token_bytes in data)

    assert leaked_key == [], f"the credential's key name is on disk in {leaked_key}"
    assert leaked_token == [], f"the credential itself is on disk in {leaked_token}"
    assert record.context["workflow_context"] == {"tenant": "acme"}


# ---------------------------------------------------------------------------
# 9. Startup is not replayed
# ---------------------------------------------------------------------------

def test_a_startup_that_already_succeeded_is_not_run_again(
    speedict_folder, tmp_path, monkeypatch
):
    """Guards against eviction re-running a side effect the channel already had.

    Startup is the one turn a channel runs without asking, and workflows use it
    to create orders, send messages and initialize external state. If "did
    startup run" were read from in-process bookkeeping rather than from the
    durable record, every eviction would do it all again — and the count is the
    only assertion that notices, because the second run looks like a success.
    """
    call_log = str(tmp_path / "add_calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

    manager = _cap_one_manager()
    channel_id, newcomer = _channel("startup"), _channel("newcomer")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD, startup_action=_add_action())
        calls_after_first = _count_calls(call_log)
        gc.collect()

        await _create(manager, newcomer, HELLO_WORLD)
        assert channel_id not in manager._sessions, "the sweep did not retire it"
        gc.collect()

        await _create(manager, channel_id, HELLO_WORLD, startup_action=_add_action())
        restored = await manager.get_session(channel_id)
        return calls_after_first, _count_calls(call_log), restored.startup_state

    calls_after_first, calls_after_restore, startup_state = asyncio.run(body())

    assert calls_after_first == 1
    assert calls_after_restore == 1, "startup ran a second time after eviction"
    assert startup_state == checkpoint.STARTUP_SUCCEEDED


# ---------------------------------------------------------------------------
# 10. The startup outcome is durable even when the context never changes
# ---------------------------------------------------------------------------

def test_the_startup_outcome_is_committed_without_any_context_change(
    speedict_folder, tmp_path, monkeypatch
):
    """Invariant 25: a fact that changes restart behaviour cannot ride on a digest.

    A startup that mutates nothing leaves the context section byte-identical, so
    a store that skipped the write on an unchanged digest would leave the durable
    record saying "not attempted" and every restart would replay the startup.
    The first commit here establishes a record while startup has not run, so the
    second one has an identical context to be compared against and can only
    produce a new generation if the outcome is digested in its own right.

    Nothing is evicted: the record has to exist because the outcome was
    committed on its own schedule, not because a retirement happened to write one.
    """
    call_log = str(tmp_path / "add_calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

    manager = ChannelSessionManager(max_live_sessions=8)
    channel_id = _channel("invariant25")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        workflow = runtime.execution_context.app_workflow

        manager.commit_startup_state(runtime)
        before = _record(manager, channel_id, HELLO_WORLD)

        context_before = dict(workflow.context or {})
        runtime.execution_context.process_action(_add_action())
        context_after = dict(workflow.context or {})

        runtime.startup_state = checkpoint.STARTUP_SUCCEEDED
        runtime.startup_ran = True
        runtime.startup_idempotency_key = "startup-once"
        manager.commit_startup_state(runtime)
        after = _record(manager, channel_id, HELLO_WORLD)

        del workflow, runtime
        # Drop the live session WITHOUT publishing, so the only record on disk is
        # the one the startup commit wrote.
        await manager.evict_live_session(channel_id)
        gc.collect()

        await _create(manager, channel_id, HELLO_WORLD, startup_action=_add_action())
        revived = await manager.get_session(channel_id)
        return (
            before, after, context_before, context_after,
            _count_calls(call_log), revived.startup_state,
        )

    before, after, context_before, context_after, calls, startup_state = asyncio.run(body())

    assert context_before == context_after, (
        "the startup mutated workflow.context, so an unchanged-digest store would "
        "have written a record anyway and this test would prove nothing"
    )
    assert before.startup["state"] == checkpoint.STARTUP_NOT_ATTEMPTED
    assert after.context == before.context, "the context section is not identical"
    assert after.generation > before.generation, (
        "the startup outcome was folded into the context digest, so the "
        "no-write fast path swallowed it"
    )
    assert after.startup["state"] == checkpoint.STARTUP_SUCCEEDED
    assert after.startup["idempotency_key"] == "startup-once"

    assert calls == 1, "startup ran again despite a durable succeeded record"
    assert startup_state == checkpoint.STARTUP_SUCCEEDED


def test_a_startup_turn_commits_its_outcome_through_the_real_turn_engine(
    speedict_folder, tmp_path, monkeypatch
):
    """The wiring behind the previous test: /initialize runs startup as a turn.

    Committing the outcome at retirement instead is too late — the server sets
    run_startup=False and submits startup through the registry, so between the
    turn finishing and any eviction the durable record still says the startup was
    never attempted, and a restart in that window replays it.
    """
    call_log = str(tmp_path / "add_calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

    manager = ChannelSessionManager(max_live_sessions=8)
    registry = TurnRegistry()
    channel_id = _channel("startupturn")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        action = _add_action()

        execution = await submit_turn(
            runtime,
            registry,
            lambda: runtime.execution_context.process_action_turn(action),
            manager,
            wait_seconds=30.0,
            kind="initialize_startup",
            idempotency_key="startup-turn",
        )
        return execution, runtime.startup_state

    execution, startup_state = asyncio.run(body())

    assert execution.error is None
    assert _count_calls(call_log) == 1
    assert startup_state == checkpoint.STARTUP_SUCCEEDED

    record = _record(manager, channel_id, HELLO_WORLD)
    assert record is not None, "the startup turn left no durable record"
    assert record.startup["state"] == checkpoint.STARTUP_SUCCEEDED


# ---------------------------------------------------------------------------
# Suspension no longer pins by itself (fix-g03.25)
# ---------------------------------------------------------------------------


def test_suspended_session_is_evictable_once_its_state_is_durable():
    """An awaiting session used to pin unconditionally, which was unbounded.

    The pin existed because the pending blob lacked the logical-turn accumulator
    and the CME continuation keys. Schema 2 carries both, so what governs now is
    whether the blob actually reached the store -- not whether the session is
    suspended.
    """
    manager = _cap_one_manager()
    channel_id = _channel("suspended")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        ctx = runtime.execution_context
        ctx._awaiting_user = True
        ctx._pending_clarification_request = "Which one?"
        ctx._begin_turn("do the thing")
        ctx.append_ask_user_entry("Which one?")

        persist_pending_after_turn(manager, runtime, fastworkflow.TurnOutput(
            turn_key=ctx._turn_key,
            status=fastworkflow.TurnStatus.AWAITING_USER,
            answer="Which one?",
            command_outputs=list(ctx._turn_outputs),
        ))
        stored = manager.session_state_store.exists(channel_id)
        return checkpoint.assess(runtime, manager.session_state_store), stored

    eligibility, stored = asyncio.run(body())

    assert stored, "the suspending turn must write the blob eviction relies on"
    assert eligibility.evictable, eligibility.reason


def test_suspended_session_pins_when_its_state_never_reached_the_store():
    """The blob is the whole basis for evicting a suspended session.

    persist_pending_after_turn declines to write state it cannot encode
    losslessly. Evicting then would discard precisely what it refused to write,
    so absence of the blob has to pin.
    """
    manager = _cap_one_manager()
    channel_id = _channel("nostate")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        runtime.execution_context._awaiting_user = True
        # Deliberately no persist_pending_after_turn: this is the shape left
        # behind by a StateEncodingError.
        manager.session_state_store.clear(channel_id)
        return checkpoint.assess(runtime, manager.session_state_store)

    eligibility = asyncio.run(body())

    assert not eligibility.evictable
    assert "durably stored" in eligibility.reason


def test_assess_without_a_store_pins_a_suspended_session():
    """Omitting the store must read as 'cannot prove it was written'.

    A caller that has no store cannot check, and the safe answer for a caller
    that cannot check is to pin.
    """
    manager = _cap_one_manager()
    channel_id = _channel("nostore")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        runtime.execution_context._awaiting_user = True
        return checkpoint.assess(runtime)

    eligibility = asyncio.run(body())
    assert not eligibility.evictable


def test_mid_extraction_session_persists_instead_of_clearing():
    """A mid-extraction session is not awaiting_user but still holds state.

    persist_pending_after_turn used to clear the blob for any turn that was not
    a suspension, which dropped the partially extracted parameters the moment
    the session was evicted between turns.
    """
    manager = _cap_one_manager()
    channel_id = _channel("midext")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        ctx = runtime.execution_context
        cme = ctx._cme_workflow.context
        cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
        cme["command"] = "add two numbers"
        cme["command_name"] = "add_two_numbers"

        assert ctx.has_open_command()
        # A COMPLETED turn, deliberately: the turn ended (with an error asking
        # for the missing values), so nothing in the TurnOutput says "keep this".
        # Only has_open_command() can save the partial parameters here.
        persist_pending_after_turn(manager, runtime, fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=fastworkflow.TurnStatus.COMPLETED,
            answer="which numbers?",
            command_outputs=[
                fastworkflow.CommandOutput(
                    command_responses=[
                        fastworkflow.CommandResponse(
                            response="which numbers?", success=False
                        )
                    ]
                )
            ],
        ))
        return manager.session_state_store.load(channel_id)

    blob = asyncio.run(body())

    assert blob is not None, "mid-extraction state was cleared instead of saved"
    assert blob["cme"]["command_name"] == "add_two_numbers"
    assert blob["cme"]["command"] == "add two numbers"


def test_mid_extraction_persists_through_the_turns_engine_too():
    """turns._persist_after_turn is a second copy of the same decision.

    The turns engine has its own post-turn writer, and the mid-extraction rule
    has to hold in both. Testing only utils' copy let a mutation that reverted
    turns' copy pass unnoticed, which is how a duplicated policy drifts.
    """
    from fastworkflow.run_fastapi_mcp.turns import _persist_after_turn

    manager = _cap_one_manager()
    channel_id = _channel("midturns")

    async def body():
        await _create(manager, channel_id, HELLO_WORLD)
        runtime = await manager.get_session(channel_id)
        cme = runtime.execution_context._cme_workflow.context
        cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
        cme["command"] = "add two numbers"
        cme["command_name"] = "add_two_numbers"

        # A completed (not suspended) turn: the case that used to clear.
        _persist_after_turn(
            manager,
            runtime,
            fastworkflow.TurnOutput(
                turn_key="t-1",
                status=fastworkflow.TurnStatus.COMPLETED,
                answer="which numbers?",
            ),
        )
        return manager.session_state_store.load(channel_id)

    blob = asyncio.run(body())

    assert blob is not None, "the turns engine cleared mid-extraction state"
    assert blob["cme"]["command_name"] == "add_two_numbers"


# ---------------------------------------------------------------------------
# Pending-state retention (fix-6b4)
# ---------------------------------------------------------------------------


def test_reap_pending_state_protects_the_channels_the_process_holds():
    """The manager is what knows which channels are live; the store cannot.

    Driving the store directly would test the policy but not the wiring, and
    the wiring is the part that decides whether an in-use conversation is
    deleted.
    """
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    from fastworkflow.session_state_store import (
        SAVED_AT_KEY,
        PendingRetentionPolicy,
    )

    manager = ChannelSessionManager(max_live_sessions=8)
    live_channel = _channel("stillhere")
    gone_channel = _channel("walkedaway")

    async def body():
        await _create(manager, live_channel, HELLO_WORLD)
        store = manager.session_state_store

        # Both look identically abandoned on disk; only one is held live.
        for cid in (live_channel, gone_channel):
            store.save(cid, {"channel_id": cid, "awaiting_user": True})
            path = _Path(store._json_path(cid))
            blob = _json.loads(path.read_text())
            blob[SAVED_AT_KEY] = _time.time() - 400 * 86_400.0
            path.write_text(_json.dumps(blob))

        outcome = manager.reap_pending_state(
            PendingRetentionPolicy(max_age_seconds=86_400.0)
        )
        return outcome, store.exists(live_channel), store.exists(gone_channel)

    outcome, live_survived, gone_survived = asyncio.run(body())

    assert live_survived, "reaped a channel the process still holds live"
    assert not gone_survived, "the abandoned session was not reclaimed"

    # Counts are asserted as lower bounds, not equalities. This store lives in
    # the shared SPEEDDICT_FOLDERNAME rather than a tmp_path, so it also holds
    # pending blobs left by other tests and by previous runs -- an equality here
    # passes alone and fails in a full-suite run, which is exactly what it did.
    assert outcome.reclaimed >= 1
    assert outcome.protected >= 1
