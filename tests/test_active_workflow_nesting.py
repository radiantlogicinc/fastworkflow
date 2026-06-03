"""Tests for contextvar-backed active workflow stack nesting."""

import fastworkflow
from fastworkflow.active_workflow import (
    clear_workflow_stack,
    get_active_workflow,
    pop_active_workflow,
    push_active_workflow,
)


def test_push_pop_nesting():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    parent = fastworkflow.Workflow.create(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        workflow_id_str="parent-nest-test",
    )
    child = fastworkflow.Workflow.create(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        workflow_id_str="child-nest-test",
    )

    clear_workflow_stack()
    assert get_active_workflow() is None

    push_active_workflow(parent)
    assert get_active_workflow() is parent

    push_active_workflow(child)
    assert get_active_workflow() is child

    popped = pop_active_workflow()
    assert popped is child
    assert get_active_workflow() is parent

    popped = pop_active_workflow()
    assert popped is parent
    assert get_active_workflow() is None

    clear_workflow_stack()
