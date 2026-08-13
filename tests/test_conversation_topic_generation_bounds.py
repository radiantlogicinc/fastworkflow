"""What bounds conversation topic/summary generation, and where that bound lives.

Topic generation is the one LLM call the conversation store makes. gh-65 asked
for it to be moved off the event loop with a "per-call timeout (e.g. 2-5s)"; the
obvious reading of that - ``asyncio.wait_for`` around ``loop.run_in_executor`` -
bounds the wrong thing. Cancelling the await does not cancel the thread. The
litellm request keeps running in the default executor, and CPython closes a loop
with ``shutdown_default_executor(asyncio.constants.THREAD_JOIN_TIMEOUT)``, which
is 300 s on 3.13. A "5 second timeout" built that way can hold process exit for
five minutes - strictly worse than the untimed in-loop call it replaced.

So the bound is at the LLM client: ``generate_topic_and_summary`` passes
``timeout=`` and a pinned ``num_retries`` into ``dspy.LM``, and dspy merges both
into the litellm request. The thread therefore ends by itself, whether or not
anyone is still waiting for the answer, and the caller can simply await it.

These tests assert that arrangement rather than its symptoms:

* the deadline reaches the *client*, not just the await (the assertion that
  fails if anyone reverts to the ``wait_for``-only design);
* ``/new_conversation`` no longer stalls the event loop for the length of a
  round trip, and does not return while generation is still running;
* the failure contract is unchanged - generation raises, and rotate stays
  strict.

No LLM is called. The single substitution in the LM tests is ``monkeypatch`` on
dspy's own ``litellm_completion`` - the real function dspy invokes at the
transport boundary - which records what it was handed and then raises instead of
going to the network. Everything above it is production code: the real
``get_lm``, a real ``dspy.LM``, the real signature and adapter. The FastAPI
tests substitute the topic-generation entry point the endpoint resolves at call
time, so that "slow" and "failing" are reproducible without a provider; the app,
the session manager, the runtime and the SQLite store are all real.

What is NOT provable here: that the provider actually honours the deadline it is
sent. That is litellm's contract with the transport and needs a live (or stalled)
endpoint to observe. What is provable, and what regressed before, is that a
deadline is sent at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.run_fastapi_mcp.conversation_store import (
    DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS,
    TOPIC_GENERATION_MAX_RETRIES,
    TOPIC_GENERATION_TIMEOUT_ENV_VAR,
    generate_topic_and_summary,
    resolve_topic_generation_timeout,
    topic_generation_timeout_seconds,
)

# Imported rather than mirrored: the worst case of the generation bound has to fit
# inside the drain `lifespan` gives in-flight work, and a local copy of 30 here
# would keep passing after someone changed the real one.
from fastworkflow.run_fastapi_mcp.utils import SHUTDOWN_DRAIN_SECONDS


class _StoppedAtTheTransport(Exception):
    """Raised in place of the HTTP request, once its parameters are recorded."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conversation_store_env():
    """A conversation-store LM configuration, restored afterwards.

    `fastworkflow._env_vars` is process-global and every other test file reads
    it, so it is snapshotted rather than rebuilt.
    """
    previous = dict(fastworkflow._env_vars or {})
    fastworkflow._env_vars = {
        "LLM_CONVERSATION_STORE": "mistral/mistral-small-latest",
        "LITELLM_API_KEY_CONVERSATION_STORE": "not-a-real-key",
    }
    try:
        yield fastworkflow._env_vars
    finally:
        fastworkflow._env_vars = previous


@pytest.fixture
def recorded_transport(monkeypatch):
    """What dspy hands to litellm, captured at dspy's own transport boundary.

    ``LM.forward`` resolves ``litellm_completion`` from its module globals on
    every call and passes it the fully merged request, so this sees exactly the
    parameters a real provider call would have been made with.
    """
    seen: dict[str, Any] = {}

    def record_and_stop(request, num_retries, cache=None):
        seen["request"] = request
        seen["num_retries"] = num_retries
        raise _StoppedAtTheTransport("no network in tests")

    monkeypatch.setattr("dspy.clients.lm.litellm_completion", record_and_stop)
    return seen


def _turns(marker: str) -> list[dict[str, Any]]:
    """Conversation turns in the shape the store hands to generation.

    The marker keeps each test's prompt distinct so a dspy cache entry from one
    test can never answer another's request.
    """
    return [{"conversation summary": f"where is my order ({marker})"}]


# ---------------------------------------------------------------------------
# The bound is on the work: the deadline reaches the LLM client
# ---------------------------------------------------------------------------

def test_the_deadline_is_sent_to_the_llm_client_not_wrapped_around_the_await(
    conversation_store_env, recorded_transport
):
    """The regression guard for the whole issue.

    A ``wait_for``-only design passes this file's other tests - the endpoint
    still answers, still fails, still frees the loop - and leaves a thread
    running that nothing can stop. The difference is visible in exactly one
    place: whether a deadline is present in the request handed to the transport.
    """
    with pytest.raises(Exception) as raised:
        generate_topic_and_summary(_turns("deadline"))
    assert "no network in tests" in str(raised.value)

    request = recorded_transport["request"]
    assert "timeout" in request, (
        "no deadline reached the litellm request, so the LLM call is unbounded "
        "and an executor thread running it cannot be made to stop"
    )
    assert request["timeout"] == DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS


def test_the_retry_count_is_pinned_because_it_multiplies_the_deadline(
    conversation_store_env, recorded_transport
):
    """A timeout is only a bound once the attempt count is known.

    litellm retries a timed-out request, so the wall-clock cost of a hung
    provider is the deadline times the attempts. dspy.LM defaults to
    ``num_retries=3``; leaving that alone would make the real worst case four
    times what the timeout advertises.
    """
    with pytest.raises(Exception):
        generate_topic_and_summary(_turns("retries"))

    assert recorded_transport["num_retries"] == TOPIC_GENERATION_MAX_RETRIES
    assert TOPIC_GENERATION_MAX_RETRIES < 3, (
        "the retry count is not pinned below dspy's default, so the effective "
        "bound is four attempts, not the advertised one"
    )


def test_the_worst_case_fits_inside_the_shutdown_drain():
    """The arithmetic that justifies the default, kept honest.

    This is the reasoning behind the number rather than a behaviour: attempts
    times deadline has to leave room inside the 30 s drain. Raising either
    constant without redoing that sum fails here.
    """
    attempts = TOPIC_GENERATION_MAX_RETRIES + 1
    worst_case = attempts * DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS
    assert worst_case < SHUTDOWN_DRAIN_SECONDS, (
        f"{attempts} attempts of {DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS}s is "
        f"{worst_case}s, which does not fit in the {SHUTDOWN_DRAIN_SECONDS}s drain"
    )
    assert DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS >= 10, (
        "a deadline this tight is inside the normal latency of a topic+summary "
        "generation, so ordinary provider slowness would fail a user's rotate"
    )


def test_the_deadline_is_configurable_per_deployment(
    conversation_store_env, recorded_transport
):
    """The value is a default, not a constant: an operator can retune it."""
    conversation_store_env[TOPIC_GENERATION_TIMEOUT_ENV_VAR] = "3.5"
    assert topic_generation_timeout_seconds() == 3.5

    with pytest.raises(Exception):
        generate_topic_and_summary(_turns("configurable"))

    assert recorded_transport["request"]["timeout"] == 3.5


def test_the_deadline_defaults_when_nothing_is_configured(
    conversation_store_env, monkeypatch
):
    """An unset variable is the common case and must not mean 'unbounded'.

    The process environment is cleared explicitly: it now outranks the workflow
    env file, so an ambient value on the developer's shell would otherwise decide
    this test's outcome.
    """
    monkeypatch.delenv(TOPIC_GENERATION_TIMEOUT_ENV_VAR, raising=False)
    assert TOPIC_GENERATION_TIMEOUT_ENV_VAR not in conversation_store_env
    assert (
        topic_generation_timeout_seconds()
        == DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS
    )
    assert resolve_topic_generation_timeout() == (
        DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS,
        "default",
    )


def test_the_process_environment_outranks_the_workflow_env_file(
    conversation_store_env, monkeypatch
):
    """An operator control must be reachable from where operators actually set it.

    On a server this knob is tuned on the deployment — against a slow provider,
    or under a tighter termination grace period — not by editing a workflow's
    env file inside the image. ``fastworkflow.get_env_var`` returns a supplied
    default BEFORE consulting ``os.environ``, so resolving through it with a
    default would have made the container variable silently unreachable. That is
    the same trap ``resolve_max_live_sessions`` was written to avoid, and this is
    the assertion that keeps this knob out of it.
    """
    conversation_store_env[TOPIC_GENERATION_TIMEOUT_ENV_VAR] = "3.5"
    monkeypatch.setenv(TOPIC_GENERATION_TIMEOUT_ENV_VAR, "7.5")

    assert resolve_topic_generation_timeout() == (7.5, "process environment")
    assert topic_generation_timeout_seconds() == 7.5


def test_the_workflow_env_file_is_used_when_the_process_has_no_override(
    conversation_store_env, monkeypatch
):
    """Precedence, not replacement: the env file still works on its own."""
    monkeypatch.delenv(TOPIC_GENERATION_TIMEOUT_ENV_VAR, raising=False)
    conversation_store_env[TOPIC_GENERATION_TIMEOUT_ENV_VAR] = "3.5"

    assert resolve_topic_generation_timeout() == (3.5, "workflow env file")


def test_an_empty_process_value_falls_through_instead_of_failing(
    conversation_store_env, monkeypatch
):
    """An exported-but-empty variable is how container templating leaves a blank.

    Treating "" as a configured value would turn a blank template into a hard
    failure, or worse a zero deadline.
    """
    monkeypatch.setenv(TOPIC_GENERATION_TIMEOUT_ENV_VAR, "")
    conversation_store_env[TOPIC_GENERATION_TIMEOUT_ENV_VAR] = "3.5"

    assert resolve_topic_generation_timeout() == (3.5, "workflow env file")


@pytest.mark.parametrize("bad", ["banana", "12s", "0", "-1"])
def test_a_malformed_deadline_is_rejected_and_names_its_source(
    conversation_store_env, monkeypatch, bad
):
    """Fail loudly, and say where the bad value came from.

    A deadline that cannot be parsed, or that is zero or negative, would
    otherwise surface as every conversation rotate failing at runtime with a
    cause the operator has to guess at. The source is in the message because the
    value can arrive from either of two places.
    """
    monkeypatch.setenv(TOPIC_GENERATION_TIMEOUT_ENV_VAR, bad)

    with pytest.raises(ValueError) as excinfo:
        resolve_topic_generation_timeout()

    message = str(excinfo.value)
    assert TOPIC_GENERATION_TIMEOUT_ENV_VAR in message
    assert "process environment" in message


def test_generation_raises_rather_than_returning_an_empty_topic(
    conversation_store_env, recorded_transport
):
    """The failure contract the store's blank-topic handling is built on.

    A blank topic means "not generated yet" and is what makes a conversation
    eligible for a later fill. If generation answered a transport failure with
    ``("", "")`` the two states would be indistinguishable, and a rotate that
    failed would look like one that simply has no title.
    """
    with pytest.raises(Exception):
        generate_topic_and_summary(_turns("raises"))


# ---------------------------------------------------------------------------
# /new_conversation: off the loop, and not abandoned
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

    Same reasoning as `tests/test_manager_shutdown_matrix.py`: the override has
    to live in the file, because `get_env_var` reads the loaded mapping before it
    looks at the process environment, and these tests write conversation records.
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
    try:
        yield main
    finally:
        if fastworkflow.RoutingRegistry:
            fastworkflow.RoutingRegistry.clear_registry()
        fastworkflow.init(previous_env or {})


@contextlib.asynccontextmanager
async def asgi_client(app_module):
    """Drive the real app from the caller's own event loop.

    ``TestClient`` runs the app in a portal on a different loop, which would
    hide the very thing these tests measure: whether *this* loop stays
    responsive while a request is generating a topic.
    """
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://topic-generation-bounds"
    ) as client:
        yield client


async def _auth_headers(client, channel_id: str) -> dict[str, str]:
    resp = await client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def _seed_conversation(app_module, channel_id: str):
    """A durable conversation with one turn, so the rotate has something to label.

    Written through the real store: `/new_conversation` summarizes the durable
    record rather than the in-memory window, so a conversation that exists only
    in memory would take the endpoint's empty-turns branch and never generate.
    """
    runtime = await app_module.session_manager.get_session(channel_id)
    assert runtime is not None
    runtime.active_conversation_id = (
        runtime.conversation_store.reserve_next_conversation_id()
    )
    runtime.conversation_store.append_conversation_turns(
        runtime.active_conversation_id,
        [
            {
                "conversation summary": "user asked where their order is",
                "conversation_traces": None,
                "feedback": None,
            }
        ],
    )
    return runtime


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_new_conversation_does_not_stall_the_event_loop_while_generating(app_module):
    """Hazard: one channel's LLM round trip freezing every other channel.

    DSPy is synchronous. Called from an ``async def`` endpoint it owns the loop
    thread for the whole round trip, so an unrelated request cannot even be
    parsed until the provider answers. The proof is a coroutine that keeps
    counting while the generation runs: on the loop it gets no turns at all,
    off the loop it gets thousands.
    """
    channel_id = _channel("offloop")
    generation_seconds = 0.4
    seen: dict[str, Any] = {}

    def slow_generate(turns):
        time.sleep(generation_seconds)
        return "Order Status Question", "The user asked where their order is."

    app_module.generate_topic_and_summary = slow_generate

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            await _seed_conversation(app_module, channel_id)

            finished = asyncio.Event()
            ticks = 0

            async def keep_counting():
                nonlocal ticks
                while not finished.is_set():
                    ticks += 1
                    await asyncio.sleep(0)

            counter = asyncio.create_task(keep_counting())
            resp = await client.post("/new_conversation", headers=headers)
            finished.set()
            await counter

            seen["status"] = resp.status_code
            seen["body"] = resp.text
            seen["ticks"] = ticks

    asyncio.run(body())

    assert seen["status"] == 200, seen["body"]
    assert seen["ticks"] > 100, (
        f"another coroutine got only {seen['ticks']} turn(s) during a "
        f"{generation_seconds}s generation, so the LLM call is still being made "
        "on the event loop"
    )


def test_new_conversation_does_not_return_while_generation_is_still_running(
    app_module,
):
    """The other half of the bound: the work is awaited, never abandoned.

    ``asyncio.wait_for`` around the executor would satisfy the test above and
    still be wrong - it returns while the thread runs on, and that orphan is
    what CPython then joins for up to 300 s at exit. Because the deadline is at
    the LLM client instead, the endpoint can simply wait, and the thread is
    always finished by the time it answers.
    """
    channel_id = _channel("noorphan")
    seen: dict[str, Any] = {}
    finished_at: list[float] = []

    def slow_generate(turns):
        time.sleep(0.3)
        finished_at.append(time.monotonic())
        return "Order Status Question", "The user asked where their order is."

    app_module.generate_topic_and_summary = slow_generate

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            await _seed_conversation(app_module, channel_id)
            resp = await client.post("/new_conversation", headers=headers)
            seen["returned_at"] = time.monotonic()
            seen["status"] = resp.status_code
            seen["body"] = resp.text

    asyncio.run(body())

    assert seen["status"] == 200, seen["body"]
    assert finished_at, (
        "the endpoint answered without the generation ever completing, so the "
        "work was abandoned to a thread nothing is waiting on"
    )
    assert seen["returned_at"] >= finished_at[0]


def test_a_failed_generation_fails_the_rotate_and_explains_itself(app_module):
    """Rotate stays strict, and the 500 has to be worth reading.

    Strict because rotate is an explicit user action and `/activate_conversation`
    finds conversations by topic: silently archiving an unlabelled thread hides
    it. Worth reading because the operator's next question is "why", and
    "Internal error in new_conversation()" does not answer it - the cause and the
    knob that bounds the call do.
    """
    channel_id = _channel("strict")
    seen: dict[str, Any] = {}

    def failing_generate(turns):
        raise TimeoutError("LM request timed out")

    app_module.generate_topic_and_summary = failing_generate

    async def body():
        async with asgi_client(app_module) as client:
            headers = await _auth_headers(client, channel_id)
            runtime = await _seed_conversation(app_module, channel_id)
            seen["conversation_id_before"] = runtime.active_conversation_id

            resp = await client.post("/new_conversation", headers=headers)
            seen["status"] = resp.status_code
            seen["detail"] = resp.json().get("detail", "")
            seen["conversation_id_after"] = runtime.active_conversation_id
            seen["record"] = runtime.conversation_store.get_conversation(
                runtime.active_conversation_id
            )

    asyncio.run(body())

    assert seen["status"] == 500
    assert seen["conversation_id_after"] == seen["conversation_id_before"], (
        "the rotate went ahead after generation failed, archiving a conversation "
        "with no topic"
    )
    assert seen["record"]["topic"] == "", "a failed generation still wrote a topic"

    detail = seen["detail"]
    assert "TimeoutError" in detail, f"the cause is not in the 500 detail: {detail}"
    assert TOPIC_GENERATION_TIMEOUT_ENV_VAR in detail, (
        f"the 500 does not name the knob that bounds the call: {detail}"
    )
    assert "NOT rotated" in detail, (
        f"the 500 does not say whether the conversation was rotated: {detail}"
    )
