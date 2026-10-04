import contextlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os
import time
from typing import Any, Optional, Union

from pydantic import BaseModel, model_validator
import mmh3


class NLUPipelineStage(Enum):
    """Specifies the stages of the NLU Pipeline processing."""
    INTENT_DETECTION = 0
    INTENT_AMBIGUITY_CLARIFICATION = 1
    INTENT_MISUNDERSTANDING_CLARIFICATION = 2
    PARAMETER_EXTRACTION = 3

class Action(BaseModel):
    command_name: str
    command: str = ""   # only use is to display autocomplete item in the UI
    parameters: dict[str, Optional[Union[str, bool, int, float, BaseModel]]] = {}
    workflow_id: Optional[int] = None    # when creating a new action, this is set by the workflow

class Recommendation(BaseModel):
    summary: str
    suggested_actions: list[Action] = []

class CommandResponse(BaseModel):
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []
    recommendations: list[Recommendation] = []

class MCPToolCall(BaseModel):
    """MCP-compliant tool call request format"""
    name: str
    arguments: dict[str, Any] = {}

class MCPContent(BaseModel):
    """MCP content block"""
    type: str  # "text", "image", etc.
    text: Optional[str] = None
    # Add other content type fields as needed

class MCPToolResult(BaseModel):
    """MCP-compliant tool result format"""
    content: list[MCPContent]
    isError: bool = False

class CommandTraceEventDirection(str, Enum):
    AGENT_TO_WORKFLOW = "agent_to_workflow"
    WORKFLOW_TO_AGENT = "workflow_to_agent"

@dataclass
class CommandTraceEvent:
    direction: CommandTraceEventDirection
    raw_command: str | None               # for AGENT_TO_WORKFLOW
    command_name: str | None              # for WORKFLOW_TO_AGENT
    parameters: dict | str | None
    response_text: str | None
    success: bool | None
    timestamp_ms: int
    # Logical-turn correlation key (additive, observability design §3.1 [X7]).
    # None only for events emitted outside any logical turn.
    turn_key: str | None = None

class CommandOutput(BaseModel):
    """The result of one command execution.

    Note on ask_user entries (A7 role inversion): when ``command_name`` is
    ``"ask_user"``, the usual roles are inverted — ``command_parameters``
    holds the *agent's question* to the user, and the command response's
    ``response`` holds the *user's answer*. ``success=False`` on an ask_user
    entry means the question is still unanswered, not that anything failed.

    ``command_parameters`` honesty [A10]: in memory this is the typed Pydantic
    params instance (or a ``str`` question for ask_user). Record serialization
    emits ``model_dump()`` as a dict; restore therefore accepts ``dict`` as
    well. Declared as ``Any`` so dump→validate round-trips do not lie as ``str``.

    As of v3.0, each command carries exactly one ``command_response``. Turn-level
    multiplicity lives on ``TurnOutput.command_outputs`` / ``TurnResult``. Passing
    the legacy ``command_responses=[...]`` keyword raises ``ValueError``.
    """

    command_response: CommandResponse
    workflow_name: str = ""
    context: str = ""
    command_name: str = ""
    command_parameters: Any = None  # typed model in memory; dict in records [A10]
    started_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    # Joins this outcome to the span that produced it (arch §12.0 delta 1).
    # Optional with a default because §12.2 requires CommandOutput stay
    # compatible: every existing constructor call and every already-serialized
    # record must keep validating, and both do — an absent key reads as None.
    # None means "not dispatched through a call-id-stamping path", which is the
    # honest answer for a hand-built CommandOutput; it never means "no span".
    command_call_id: Optional[str] = None
    # [A7] structural marker for an ask_user exchange. Optional with a None
    # default for the same §12.2 reason as command_call_id: every existing
    # constructor call and every already-serialized record must keep
    # validating. None means "written before this field existed" and defers to
    # the name comparison in `is_ask_user`; True and False are authoritative.
    ask_user_entry: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_command_responses(cls, data: Any) -> Any:
        """Reject the pre-v3.0 ``command_responses`` list keyword with a clear error."""
        if isinstance(data, dict) and "command_responses" in data:
            raise ValueError(
                "CommandOutput no longer accepts command_responses=[...]; "
                "pass command_response=CommandResponse(...) (singular). "
                "Turn-level multiplicity lives on TurnOutput.command_outputs."
            )
        return data

    @property
    def success(self) -> bool:
        return self.command_response.success

    @property
    def is_ask_user(self) -> bool:
        """True if this entry is an ask_user clarification exchange. [A7]

        `ask_user_entry` is authoritative when set; the name comparison is the
        fallback for records serialized before the field existed. It cannot be
        the primary test any more: since fix-ajv.16 a FAILURE output carries the
        real routed command name with `success=False`, which is byte-identical
        to an unanswered question, and root-context command names are
        UNQUALIFIED — so a workflow defining a root command called `ask_user`
        would have its failures collected by `complete_ask_user_entry`, which
        would overwrite the error with the user's answer and mark it successful.
        The guarantee now rests on a field the framework sets, not on no
        workflow ever choosing a name. fix-ajv.17.
        """
        if self.ask_user_entry is not None:
            return self.ask_user_entry
        return self.command_name == "ask_user"

    @property
    def question(self) -> Optional[str]:
        """The agent's question to the user (ask_user entries only). [A7]"""
        return self.command_parameters if self.is_ask_user else None

    @property
    def user_reply(self) -> Optional[str]:
        """The user's answer to an ask_user question, if any. [A7]"""
        if self.is_ask_user:
            return self.command_response.response
        return None

    @property
    def command_aborted(self) -> bool:
        return self.command_response.artifacts.get("command_name", None) == "abort"

    @property
    def command_handled(self) -> bool:
        return self.command_response.artifacts.get("command_handled", False) is True

    @property
    def not_what_i_meant(self) -> bool:
        return (
            self.command_response.artifacts.get("command_name", None)
            == "misunderstood_intent"
        )

    def to_mcp_result(self) -> MCPToolResult:
        """Convert CommandOutput to MCP-compliant format"""
        return MCPToolResult(
            content=[
                MCPContent(type="text", text=self.command_response.response)
            ],
            isError=not self.success,
        )

class _Ephemeral:
    """Sentinel: this state is deliberately not persisted.

    Returned from a serialization hook to say "I know about this and I choose
    nothing", which pins the session WITHOUT the warning that an absent hook
    earns. It is a distinct object rather than None so that a hook which falls
    off the end of a function is a reportable bug instead of silent consent.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "fastworkflow.EPHEMERAL"

    def __bool__(self) -> bool:
        return False


EPHEMERAL = _Ephemeral()


class UnsupportedStateVersion(Exception):
    """Raised by a from_state hook that will not migrate the version it was given.

    The framework treats this as "this snapshot is not restorable": the record is
    quarantined and the request fails, rather than state being partly applied.
    """

    def __init__(self, state_version: int, message: Optional[str] = None):
        self.state_version = state_version
        super().__init__(
            message or f"cannot migrate state written at version {state_version}"
        )


class ModuleType(Enum):
    """Specifies which part of a command's implementation to load."""
    INPUT_FOR_PARAM_EXTRACTION_CLASS = 0
    COMMAND_PARAMETERS_CLASS = 1
    RESPONSE_GENERATION_INFERENCE = 2
    CONTEXT_CLASS = 3

_chat_session: Optional["ChatSession"] = None
class ChatSessionDescriptor:
    """Descriptor for accessing the global chat session.""" 
    def __get__(self, obj, objtype = None):
        global _chat_session
        return _chat_session
    
    def __set__(self, obj, value):
        global _chat_session
        if _chat_session:
            raise RuntimeError("Cannot set chat session. It is already set.")
        _chat_session = value
# Create the descriptor instance
chat_session = ChatSessionDescriptor()


_env_vars: dict = {}
CommandContextModel = None
RoutingDefinition = None
RoutingRegistry = None
ModelPipelineRegistry=None


def init(env_vars: dict):
    global _env_vars, CommandContextModel, RoutingDefinition, RoutingRegistry, ModelPipelineRegistry
    _env_vars = env_vars

    # Reconfigure log level from env_vars (dotenv files) if LOG_LEVEL is specified
    # This allows LOG_LEVEL to be set in fastworkflow.env files, not just OS environment
    if log_level := env_vars.get("LOG_LEVEL"):
        from .utils.logging import reconfigure_log_level
        reconfigure_log_level(log_level)

    # init before importing other modules so env vars are available
    from .command_context_model import CommandContextModel as CommandContextModelClass
    from .command_routing import RoutingDefinition as RoutingDefinitionClass
    from .command_routing import RoutingRegistry as RoutingRegistryClass
    from .model_pipeline_training import ModelPipeline

    # Assign to global variables
    CommandContextModel = CommandContextModelClass
    RoutingDefinition = RoutingDefinitionClass
    RoutingRegistry = RoutingRegistryClass
    ModelPipelineRegistry = ModelPipeline

    # Ensure DSPy logging is properly configured after all imports
    # This needs to happen after DSPy is imported by other modules
    import logging
    logging.getLogger("dspy").setLevel(logging.ERROR)
    logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)

    # ------------------------------------------------------------
    # Eager imports for heavy libraries that otherwise trigger lock
    # contention during the first wildcard command.  Importing them
    # once here (at server start-up) shifts the cost out of the
    # request path and takes advantage of Python's module cache.
    # ------------------------------------------------------------
    with contextlib.suppress(Exception):
        import datasets  # noqa: F401 – pre-load Hugging Face datasets

def get_env_var(var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """get the environment variable"""
    global _env_vars

    value = _env_vars.get(var_name)
    if value is None:
        if default is not None:
            return default
        value = os.getenv(var_name)

    if value is None:
        from fastworkflow.utils.logging import logger
        logger.warning(f"Environment variable '{var_name}' does not exist and no default value is provided.")

    try:
        if value is None:
            return None
        if var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        elif var_type is bool:
            if value.lower() in ('true', '1'):
                return True
            elif value.lower() in ('false', '0'):
                return False
            else:
                raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.")
        return str(value)  # Default case for str
    except ValueError as e:
        raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.") from e

def get_fastworkflow_package_path() -> str:
    """Get the fastworkflow package directory.
    
    This works both in development (when working in the fastworkflow repo)
    and when fastworkflow is pip installed.
    
    Returns:
        str: Path to the fastworkflow package directory
    """
    return os.path.dirname(os.path.abspath(__file__))

def get_internal_workflow_path(workflow_name: str) -> str:
    """Get the path to an internal fastworkflow workflow.
    
    Args:
        workflow_name: Name of the workflow in the _workflows directory
        
    Returns:
        str: Full path to the internal workflow
    """
    return os.path.join(get_fastworkflow_package_path(), "_workflows", workflow_name)

def get_workflow_id(workflow_id_str: str) -> int:
    return int(mmh3.hash(workflow_id_str))

from .active_workflow import (
    get_active_workflow,
    push_active_workflow,
    pop_active_workflow,
    clear_workflow_stack,
)

# Turn-level result types (fastworkflow.turn). Imported here — after
# CommandResponse and CommandOutput are defined — so the forward references
# inside TurnResult can be resolved against this module's types.
from fastworkflow.turn import (
    TurnResult,
    TurnOutput,
    TurnStatus,
    FW_ARTIFACT_REF_KEY,
    mint_turn_key,
)

_turn_types_namespace = {
    "CommandResponse": CommandResponse,
    "CommandOutput": CommandOutput,
    "TurnOutput": TurnOutput,
}
TurnOutput.model_rebuild(_types_namespace=_turn_types_namespace)
TurnResult.model_rebuild(_types_namespace=_turn_types_namespace)

# Observability sinks (fastworkflow.tracing / fastworkflow.metrics, both
# stdlib-only). Embedders implement TraceSink/MetricsSink and wire them via
# WorkflowExecutionContext (observability design §3.1).
from fastworkflow.tracing import (
    Span as Span,
    TraceSink as TraceSink,
    NoOpTraceSink as NoOpTraceSink,
)
from fastworkflow.metrics import (
    MetricsSink as MetricsSink,
    NoOpMetricsSink as NoOpMetricsSink,
    LoggingMetricsSink as LoggingMetricsSink,
)

# The ChatSessionDescriptor instance used to occupy this name until the eager
# `from .chat_session import ChatSession` replaced it with the submodule.
# Leave the name unbound so attribute access loads that submodule on demand
# instead of publishing the descriptor, which is not the public object.
del chat_session

# First-use re-exports. Workflow pulls numpy via kvstore; ChatSession and
# WorkflowExecutionContext pull dspy, litellm, fastapi, and starlette.
_LAZY_ATTRS = {
    "Workflow": ("fastworkflow.workflow", "Workflow"),
    "ChatSession": ("fastworkflow.chat_session", "ChatSession"),
    "WorkflowExecutionContext": (
        "fastworkflow.workflow_execution_context",
        "WorkflowExecutionContext",
    ),
    "CommandCancelledError": (
        "fastworkflow.workflow_execution_context",
        "CommandCancelledError",
    ),
}

__all__ = (
    "Action",
    "Any",
    "BaseModel",
    "ChatSession",
    "ChatSessionDescriptor",
    "CommandCancelledError",
    "CommandContextModel",
    "CommandOutput",
    "CommandResponse",
    "CommandTraceEvent",
    "CommandTraceEventDirection",
    "EPHEMERAL",
    "Enum",
    "FW_ARTIFACT_REF_KEY",
    "LoggingMetricsSink",
    "MCPContent",
    "MCPToolCall",
    "MCPToolResult",
    "MetricsSink",
    "ModelPipelineRegistry",
    "ModuleType",
    "NLUPipelineStage",
    "NoOpMetricsSink",
    "NoOpTraceSink",
    "Optional",
    "Recommendation",
    "RoutingDefinition",
    "RoutingRegistry",
    "Span",
    "TraceSink",
    "TurnOutput",
    "TurnResult",
    "TurnStatus",
    "Union",
    "UnsupportedStateVersion",
    "Workflow",
    "WorkflowExecutionContext",
    "active_workflow",
    "agent_runtime",
    "chat_session",
    "clear_workflow_stack",
    "context_budget",
    "contextlib",
    "dataclass",
    "datetime",
    "get_active_workflow",
    "get_env_var",
    "get_fastworkflow_package_path",
    "get_internal_workflow_path",
    "get_workflow_id",
    "init",
    "kvstore",
    "metrics",
    "mint_turn_key",
    "mmh3",
    "model_validator",
    "observability",
    "os",
    "pop_active_workflow",
    "push_active_workflow",
    "runtime_manifest",
    "session_state_store",
    "state_paths",
    "state_serialization",
    "storage_keys",
    "time",
    "tracing",
    "turn",
    "turn_plan",
    "utils",
    "workflow",
    "workflow_execution_context",
)


def __getattr__(name: str):
    # PEP 562: Workflow/ChatSession/WorkflowExecutionContext import numpy, dspy, litellm, and fastapi.
    import importlib

    spec = _LAZY_ATTRS.get(name)
    if spec is not None:
        module_name, attr_name = spec
        module = importlib.import_module(module_name)
        for public_name, (lazy_module, lazy_attr) in _LAZY_ATTRS.items():
            if lazy_module == module_name:
                globals()[public_name] = getattr(module, lazy_attr)
        return globals()[name]
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{name}":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
        raise
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(__all__) | {key for key in globals() if not key.startswith("_")})
