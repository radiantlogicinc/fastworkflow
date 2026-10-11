"""Public names of the fastworkflow package stay importable.

Captured from an unmodified ``import fastworkflow`` (dir of non-underscore
names) before heavy re-exports became lazy. Every name must still resolve,
with the same type and defining module, including names that now load through
PEP 562 ``__getattr__``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from pathlib import Path

import fastworkflow

# (name, type(obj).__name__, module qualifier). Qualifier is the module
# __name__ for modules, __module__ for everything else, and "" for None and
# for plain str constants (they have no __module__).
PUBLIC = (
    ("Action", "ModelMetaclass", "fastworkflow"),
    ("Any", "_AnyMeta", "typing"),
    ("BaseModel", "ModelMetaclass", "pydantic.main"),
    ("ChatSession", "type", "fastworkflow.chat_session"),
    ("ChatSessionDescriptor", "type", "fastworkflow"),
    ("CommandCancelledError", "type", "fastworkflow.workflow_execution_context"),
    ("CommandContextModel", "NoneType", ""),
    ("CommandOutput", "ModelMetaclass", "fastworkflow"),
    ("CommandResponse", "ModelMetaclass", "fastworkflow"),
    ("CommandTraceEvent", "type", "fastworkflow"),
    ("CommandTraceEventDirection", "EnumType", "fastworkflow"),
    ("EPHEMERAL", "_Ephemeral", "fastworkflow"),
    ("Enum", "EnumType", "enum"),
    ("FW_ARTIFACT_REF_KEY", "str", ""),
    ("LoggingMetricsSink", "type", "fastworkflow.metrics"),
    ("MCPContent", "ModelMetaclass", "fastworkflow"),
    ("MCPToolCall", "ModelMetaclass", "fastworkflow"),
    ("MCPToolResult", "ModelMetaclass", "fastworkflow"),
    ("MetricsSink", "_ProtocolMeta", "fastworkflow.metrics"),
    ("ModelPipelineRegistry", "NoneType", ""),
    ("ModuleType", "EnumType", "fastworkflow"),
    ("NLUPipelineStage", "EnumType", "fastworkflow"),
    ("NoOpMetricsSink", "type", "fastworkflow.metrics"),
    ("NoOpTraceSink", "type", "fastworkflow.tracing"),
    ("Optional", "_SpecialForm", "typing"),
    ("Recommendation", "ModelMetaclass", "fastworkflow"),
    ("RoutingDefinition", "NoneType", ""),
    ("RoutingRegistry", "NoneType", ""),
    ("Span", "type", "fastworkflow.tracing"),
    ("TraceSink", "_ProtocolMeta", "fastworkflow.tracing"),
    ("TurnOutput", "ModelMetaclass", "fastworkflow.turn"),
    ("TurnResult", "ModelMetaclass", "fastworkflow.turn"),
    ("TurnStatus", "EnumType", "fastworkflow.turn"),
    ("Union", "_SpecialForm", "typing"),
    ("UnsupportedStateVersion", "type", "fastworkflow"),
    ("Workflow", "type", "fastworkflow.workflow"),
    ("WorkflowExecutionContext", "type", "fastworkflow.workflow_execution_context"),
    ("active_workflow", "module", "fastworkflow.active_workflow"),
    ("chat_session", "module", "fastworkflow.chat_session"),
    ("clear_workflow_stack", "function", "fastworkflow.active_workflow"),
    ("context_budget", "module", "fastworkflow.context_budget"),
    ("contextlib", "module", "contextlib"),
    ("dataclass", "function", "dataclasses"),
    ("datetime", "type", "datetime"),
    ("get_active_workflow", "function", "fastworkflow.active_workflow"),
    ("get_env_var", "function", "fastworkflow"),
    ("get_fastworkflow_package_path", "function", "fastworkflow"),
    ("get_internal_workflow_path", "function", "fastworkflow"),
    ("get_workflow_id", "function", "fastworkflow"),
    ("init", "function", "fastworkflow"),
    ("kvstore", "module", "fastworkflow.kvstore"),
    ("metrics", "module", "fastworkflow.metrics"),
    ("mint_turn_key", "function", "fastworkflow.turn"),
    ("mmh3", "module", "mmh3"),
    ("model_validator", "function", "pydantic.functional_validators"),
    ("observability", "module", "fastworkflow.observability"),
    ("os", "module", "os"),
    ("pop_active_workflow", "function", "fastworkflow.active_workflow"),
    ("push_active_workflow", "function", "fastworkflow.active_workflow"),
    ("session_state_store", "module", "fastworkflow.session_state_store"),
    ("state_paths", "module", "fastworkflow.state_paths"),
    ("state_serialization", "module", "fastworkflow.state_serialization"),
    ("storage_keys", "module", "fastworkflow.storage_keys"),
    ("time", "module", "time"),
    ("tracing", "module", "fastworkflow.tracing"),
    ("turn", "module", "fastworkflow.turn"),
    ("utils", "module", "fastworkflow.utils"),
    ("workflow", "module", "fastworkflow.workflow"),
    ("workflow_execution_context", "module", "fastworkflow.workflow_execution_context"),
)


# Bound to None at import and rebound to these classes by `fastworkflow.init()`,
# so in a process where any earlier test called init() they are no longer None.
# The None captured above is asserted in a fresh interpreter instead.
INIT_ASSIGNED = {
    "CommandContextModel": "fastworkflow.command_context_model",
    "RoutingDefinition": "fastworkflow.command_routing",
    "RoutingRegistry": "fastworkflow.command_routing",
    "ModelPipelineRegistry": "fastworkflow.model_pipeline_training",
}


def _qualifier(obj: object) -> str:
    if obj is None or isinstance(obj, str):
        return ""
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    return getattr(obj, "__module__", "") or ""


def test_public_names_still_resolve():
    # Other imports in this process attach extra submodules onto the package.
    # The fresh-interpreter surface is checked in a subprocess; here every
    # captured name must still resolve to the same object it did before.
    for name, type_name, qualifier in PUBLIC:
        obj = getattr(fastworkflow, name)
        if name in INIT_ASSIGNED and obj is not None:
            assert isinstance(obj, type), name
            assert obj.__module__ == INIT_ASSIGNED[name], name
        else:
            assert type(obj).__name__ == type_name, name
            assert _qualifier(obj) == qualifier, name
        assert getattr(fastworkflow, name) is obj

    script = textwrap.dedent(
        f"""
        import fastworkflow
        names = [name for name in dir(fastworkflow) if not name.startswith("_")]
        print("\\n".join(names))
        print("--")
        for name in {sorted(INIT_ASSIGNED)!r}:
            print(name, type(getattr(fastworkflow, name)).__name__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    names_part, _, init_part = result.stdout.partition("--\n")
    found = [line for line in names_part.splitlines() if line]
    assert found == sorted(name for name, _, _ in PUBLIC)
    fresh_types = dict(line.split() for line in init_part.splitlines() if line)
    assert fresh_types == {name: "NoneType" for name in INIT_ASSIGNED}
