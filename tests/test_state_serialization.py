"""Integration tests for the strict state serializer (Release B, step 8).

`WorkflowExecutionContext.serialize_state()` used to return
``json.loads(json.dumps(payload, default=str))``. That is the defect: an
unsupported object had already become an ordinary string before anything
downstream could object, so a strict check at the store validated the lossy
result and passed it.

The design is explicit that these must be exercised **through**
``ctx.serialize_state()`` rather than by calling the encoder directly — testing
the helper in isolation is exactly how the previous revision's strictness
requirement passed review while being unimplementable.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime

import pytest

import fastworkflow
from fastworkflow.state_serialization import (
    StateEncodingError,
    encode_state,
    state_digest,
    validate_state,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture(scope="module")
def initialized():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing")
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    return True


@pytest.fixture
def ctx(initialized):
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip("hello_world workflow not found")
    channel_id = f"strict_{uuid.uuid4().hex[:8]}"
    context = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    context.bind_app_workflow(
        fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
    )
    yield context
    context.close()


def _serialize(ctx):
    return ctx.serialize_state(channel_id="strict-channel")


# ---------------------------------------------------------------------------
# Through the real boundary
# ---------------------------------------------------------------------------

def test_clean_state_serializes_through_the_context(ctx):
    """The control: an ordinary suspended context still round-trips."""
    ctx._awaiting_user = True
    ctx._suspended_user_message = "which one?"
    ctx._pending_clarification_request = {"options": ["a", "b"], "count": 2}

    state = _serialize(ctx)

    assert state["awaiting_user"] is True
    assert state["pending_clarification_request"]["count"] == 2
    assert json.loads(encode_state(state)) == state


def test_a_non_json_value_is_rejected_rather_than_stringified(ctx):
    """This is the whole point: `default=str` used to make this silently 'work'.

    A datetime became "2026-08-06 00:00:00" and restore handed the application a
    string where it had stored an object — with nothing reporting the swap.
    """
    ctx._awaiting_user = True
    ctx._pending_clarification_request = {"deadline": datetime(2026, 8, 6)}

    with pytest.raises(StateEncodingError) as exc:
        _serialize(ctx)

    assert "datetime" in str(exc.value)
    assert "pending_clarification_request" in str(exc.value)


def test_a_non_string_dict_key_is_rejected(ctx):
    """JSON has no integer keys; coercing them makes 1 and "1" collide on restore."""
    ctx._awaiting_user = True
    ctx._pending_clarification_request = {1: "first"}

    with pytest.raises(StateEncodingError, match="not str"):
        _serialize(ctx)


def test_shared_mutable_containers_are_rejected(ctx):
    """Two paths to one list restore as two lists; anything relying on that is now wrong."""
    shared = ["x"]
    ctx._awaiting_user = True
    ctx._pending_clarification_request = {"a": shared, "b": shared}

    with pytest.raises(StateEncodingError, match="two independent copies"):
        _serialize(ctx)


def test_a_failed_encode_leaves_the_context_usable(ctx):
    """Invariant 8: a serialization failure keeps the runtime live."""
    ctx._awaiting_user = True
    ctx._pending_clarification_request = {"bad": {object()}}

    with pytest.raises(StateEncodingError):
        _serialize(ctx)

    # Still the authority on its own state, and still serializable once fixed.
    assert ctx.awaiting_user is True
    ctx._pending_clarification_request = {"good": True}
    assert _serialize(ctx)["pending_clarification_request"] == {"good": True}


# ---------------------------------------------------------------------------
# Encoder properties
# ---------------------------------------------------------------------------

def test_cycles_are_rejected():
    state: dict = {"self": None}
    state["self"] = state

    with pytest.raises(StateEncodingError):
        validate_state(state)


@pytest.mark.parametrize("value", [float("nan"), math.inf, -math.inf])
def test_non_finite_floats_are_rejected(value):
    with pytest.raises(StateEncodingError, match="no JSON representation"):
        validate_state({"n": value})


def test_a_non_dict_top_level_is_rejected():
    with pytest.raises(StateEncodingError, match="top level"):
        validate_state([1, 2, 3])


def test_encoding_is_canonical_so_digests_are_stable():
    """An unstable encoding would force a durable write on every retirement."""
    first = {"b": 1, "a": {"d": [1, 2], "c": "x"}}
    second = {"a": {"c": "x", "d": [1, 2]}, "b": 1}

    assert encode_state(first) == encode_state(second)
    assert state_digest(first) == state_digest(second)
    assert encode_state(first) == '{"a":{"c":"x","d":[1,2]},"b":1}'
    assert state_digest({"a": 1}) != state_digest({"a": 2})


def test_the_error_names_the_path_to_the_offending_value():
    """A rejection an author cannot locate is a rejection they cannot act on."""
    with pytest.raises(StateEncodingError, match=r"\$\.outer\.items\[1\]\.when"):
        validate_state({"outer": {"items": [{}, {"when": datetime(2026, 1, 1)}]}})
