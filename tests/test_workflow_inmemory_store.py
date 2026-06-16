"""Tests for the in-memory, weak-referenced Workflow session-state registry.

These cover the lifecycle guarantees that replaced the per-session speedict
(RocksDB) stores: live-object identity, context override on recreate,
parent/child topology + close() teardown, and — the point of fix-04r —
automatic eviction of abandoned workflows once they are garbage-collected
(no leak, no manual close() required).
"""

import gc

import fastworkflow
from fastworkflow.workflow import _WORKFLOW_REGISTRY


def _internal_path() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def test_get_workflow_returns_same_live_object():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    wf = fastworkflow.Workflow.create(_internal_path(), workflow_id_str="store-identity")
    try:
        wf.context["k"] = "v"
        wf.flush()
        got = fastworkflow.Workflow.get_workflow(wf.id)
        assert got is wf
        assert got.context == {"k": "v"}
    finally:
        wf.close()


def test_recreate_same_id_returns_existing_with_context_override():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    wf = fastworkflow.Workflow.create(_internal_path(), workflow_id_str="store-recreate")
    try:
        wf.context["k"] = "v"
        wf2 = fastworkflow.Workflow.create(
            _internal_path(),
            workflow_id_str="store-recreate",
            workflow_context={"x": 1},
        )
        assert wf2 is wf
        assert wf2.context == {"x": 1}
    finally:
        wf.close()


def test_child_topology_and_close_teardown():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    parent = fastworkflow.Workflow.create(_internal_path(), workflow_id_str="store-parent")
    child = fastworkflow.Workflow.create(
        _internal_path(), parent_workflow_id=parent.id
    )
    parent_id, child_id = parent.id, child.id

    assert child.parent_id == parent_id
    assert child_id in parent._children
    assert child_id in _WORKFLOW_REGISTRY

    assert parent.close() is True
    # root and descendant are both evicted by close()
    assert parent_id not in _WORKFLOW_REGISTRY
    assert child_id not in _WORKFLOW_REGISTRY
    assert fastworkflow.Workflow.get_workflow(parent_id) is None


def test_abandoned_workflow_is_auto_evicted_on_gc():
    """fix-04r: dropping all strong refs reclaims the registry entry (no leak)."""
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})
    wf = fastworkflow.Workflow.create(
        _internal_path(), workflow_id_str="store-ephemeral"
    )
    wid = wf.id
    assert wid in _WORKFLOW_REGISTRY

    # Drop the only strong reference and collect; the weak registry entry
    # must disappear without any explicit close().
    del wf
    gc.collect()

    assert wid not in _WORKFLOW_REGISTRY
    assert fastworkflow.Workflow.get_workflow(wid) is None
