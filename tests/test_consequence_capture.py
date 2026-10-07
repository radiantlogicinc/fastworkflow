"""Context-before/after types and consequence on executed commands.

Architecture §12.0 deltas 2 and 4, §6.6.1, and FW-REQ-002's acceptance
criteria -- specifically the two that name this behavior directly:

* "An authorized navigation command records distinct context-before and
  context-after handles."
* "A non-navigation command records identical context-before and context-after
  handles."

The recorded value is the active context's TYPE name. fastWorkflow has no
framework-level identity for a context instance, so two equal values do not
prove the command stayed on the same object; the diagnosis reports that case as
unknown rather than unchanged.

**Unknown must not read as cheap.** An undeclared command's effect contract is
`unknown`, which §6.6.1 requires be treated as write-capable and floored at high
consequence. The failure mode is silent: `read_only` would produce a clean-looking
row that under-reports every command in every workflow without a manifest, which
is nearly all of them.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.runtime_manifest import (
    CommandDeclaration,
    EffectContract,
    RuntimeManifest,
    clear_runtime_metadata,
    get_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

# Moves the workflow's command context from TodoListManager down to the created
# TodoList (create_todo_list.py line 55).
NAVIGATING_COMMAND = "TodoListManager/create_todo_list"

# Reads and returns; never touches current_command_context.
NON_NAVIGATING_COMMAND = "TodoListManager/list_todo_lists"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"consequence-{uuid.uuid4().hex}",
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _last_tool_call(sink: RecordingTraceSink) -> tracing.Span:
    return sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1]


# ----------------------------------------------------------------------
# FW-REQ-002 acceptance criteria
# ----------------------------------------------------------------------


def test_a_navigation_command_records_distinct_context_types(ctx, sink):
    """create_todo_list descends TodoListManager -> TodoList."""
    ctx.process_action_turn(_action(NAVIGATING_COMMAND, description="groceries"))

    span = _last_tool_call(sink)
    assert span.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
    assert span.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoList"


def test_a_non_navigation_command_records_identical_context_types(ctx, sink):
    """list_todo_lists reads and returns; the workflow does not move."""
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    span = _last_tool_call(sink)
    assert span.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
    assert span.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoListManager"


def test_the_prose_path_records_context_types_too(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """FW-REQ-002 clause 5: capture semantics are shared across paths.

    The CME hop is stood in for because the test workflow ships no trained intent
    models; the dispatch it stands in for is the real `perform_action`, so the
    context reads under test run for real.
    """
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"context-prose-{uuid.uuid4().hex}"
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, wf, action):
        command_output = real_perform_action(
            workflow, _action(NAVIGATING_COMMAND, description="groceries")
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = NAVIGATING_COMMAND
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))

    try:
        context.process_turn("make me a grocery list")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert execute.attributes[tracing.ATTR_CONTEXT_BEFORE] == "TodoListManager"
        assert execute.attributes[tracing.ATTR_CONTEXT_AFTER] == "TodoList"
    finally:
        with suppress(Exception):
            context.close()


# ----------------------------------------------------------------------
# Consequence (§6.6.1)
# ----------------------------------------------------------------------


def test_an_undeclared_command_is_unknown_write_capable_and_high(ctx, sink):
    """The todo workflow ships no manifest, so nothing declares its effects.

    `unknown` rather than `read_only` is the whole rule: an absent contract is a
    reason for more caution, not less (§7.3, §6.6.1). `read_only` here would
    silently under-report every command of every workflow without a manifest.
    """
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
    assert consequence["effect_kind"] == "unknown"
    assert consequence["consequence_class"] == "high"
    assert consequence["assessor_version"] == "default/1"


def test_reversibility_and_blast_radius_stay_unknown(ctx, sink):
    """Nothing in the manifest schema declares either, so neither is guessed."""
    ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

    consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
    assert consequence["reversibility"] == "unknown"
    assert consequence["blast_radius"] == "unknown"


def test_a_declared_effect_contract_reaches_the_span(ctx, sink, todo_workflow_path):
    """The registry is what makes a declaration reachable at execution time.

    `check_startup_conformance` returned a `RuntimeMetadata` that both entry
    points discarded, so before this a workflow that declared `read_only` and one
    that declared nothing produced identical records.
    """
    manifest = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={
            NON_NAVIGATING_COMMAND: CommandDeclaration(
                effect=EffectContract(kind="read_only")
            )
        },
    )
    register_runtime_metadata(
        todo_workflow_path, merge_and_gate(manifest, deployment_features={}, env={})
    )
    try:
        ctx.process_action_turn(_action(NON_NAVIGATING_COMMAND))

        consequence = _last_tool_call(sink).attributes[tracing.ATTR_CONSEQUENCE]
        assert consequence["effect_kind"] == "read_only"
    finally:
        clear_runtime_metadata()


def test_an_unregistered_workflow_reads_as_unknown_not_read_only():
    """The fallback, at the helper rather than through a whole turn."""
    assert get_runtime_metadata("/no/such/workflow") is None
    consequence = tracing.consequence_assessment("/no/such/workflow", "anything")
    assert consequence["effect_kind"] == "unknown"


def test_the_registry_key_survives_an_unresolved_path(tmp_path):
    """A relative or symlinked path must find its own registration.

    `Workflow` resolves the folderpath it is given, so a table keyed on the raw
    CLI argument would miss — silently, reporting `unknown` for a workflow that
    declared its effects.
    """
    resolved = tmp_path / "wf"
    resolved.mkdir()
    unresolved = tmp_path / "." / "wf"

    metadata = merge_and_gate(None, deployment_features={}, env={})
    register_runtime_metadata(str(unresolved), metadata)
    try:
        assert get_runtime_metadata(str(resolved)) is metadata
    finally:
        clear_runtime_metadata()
