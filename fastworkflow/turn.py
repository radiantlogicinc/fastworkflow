"""Turn-level result types for fastWorkflow (v2.21 forward-compatible subset).

A *logical turn* is one user interaction with a workflow: every command
execution, clarification exchange, and failure that occurs between the user's
message and the final answer. ``TurnResult`` captures that turn.

This module ships the v2.21 (non-breaking) shapes. See
``docs/turn_result_design_final.md`` for the full v3.0 design; where the two
differ, the v2.21 shapes here are intentional (e.g. ``CommandOutput`` still
carries a ``command_responses`` list).

``CommandResponse`` and ``CommandOutput`` are forward references resolved at
the bottom of ``fastworkflow/__init__.py`` via ``TurnResult.model_rebuild``.
"""

from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, computed_field

from fastworkflow import CommandOutput, CommandResponse

# Key marking an artifacts-dict envelope whose value was offloaded to a store
# and replaced in place by a scoped reference. [A10][A47]
FW_ARTIFACT_REF_KEY = "__fw_artifact_ref__"

# Scalar types allowed inside CommandResponse.artifacts (plus dict/list/tuple
# containers of the same). Anything else is unserializable for turn records.
_ALLOWED_SCALAR_TYPES = (str, int, float, bool)
_ALLOWED_CONTAINER_TYPES = (dict, list, tuple)


def merge_artifact_responses_into(
    target: "CommandResponse",
    artifact_responses: list["CommandResponse"],
) -> None:
    """Merge artifact dicts from turn tool responses into one user-facing response. [Topic 5]

    Each key from every ``artifact_responses`` entry is copied into ``target.artifacts``.
    When a key already exists on ``target``, the incoming key is suffixed with
    ``_<increment>`` (1, 2, ...) until unused.
    """
    for artifact_response in artifact_responses:
        for key, value in artifact_response.artifacts.items():
            target_key = key
            if target_key in target.artifacts:
                increment = 1
                while f"{key}_{increment}" in target.artifacts:
                    increment += 1
                target_key = f"{key}_{increment}"
            target.artifacts[target_key] = value


def collect_artifact_responses(
    command_outputs: list["CommandOutput"],
) -> list["CommandResponse"]:
    """Flatten every command response that carries artifacts, in turn order. [A9][A20]

    Returns the subset of command responses (across every ``command_output``)
    whose ``artifacts`` dict is non-empty, projected to the flat
    ``CommandResponse`` shape so a single user-facing ``CommandOutput`` can
    surface every structured output without nested ``TurnResult`` serialization
    (no recursion). The returned responses are the original objects.

    The framework does not interpret artifact keys or values — it only preserves
    structured outputs that would otherwise be dropped. Which keys are meaningful
    (and how to render them) is entirely the consuming client's concern. [Topic 5]
    """
    return [
        response
        for command_output in command_outputs
        for response in command_output.command_responses
        if response.artifacts
    ]


class TurnStatus(str, Enum):
    """Terminal (or suspended) status of a logical turn. [A3]"""

    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


def mint_turn_key(now: Optional[datetime] = None, uuid_hex: Optional[str] = None) -> str:
    """Mint a new turn key: ``YYYYMMDDTHHMMSS.ffffffZ-<uuid4 hex, 12 chars>``.

    Colon-free and lexicographically sortable; the timestamp is the logical
    turn start in UTC. [A22][A24][A26]

    Args:
        now: Injectable timestamp (UTC) for deterministic tests.
            Defaults to ``datetime.now(timezone.utc)``.
        uuid_hex: Injectable uniqueness suffix for deterministic tests.
            Defaults to the first 12 hex chars of a fresh ``uuid4``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if uuid_hex is None:
        uuid_hex = uuid.uuid4().hex[:12]
    return f"{now.strftime('%Y%m%dT%H%M%S.%f')}Z-{uuid_hex}"


def _walk_artifact_value(value: Any) -> bool:
    """Return True if *value* is record-serializable (cheap recursive check)."""
    if value is None or isinstance(value, _ALLOWED_SCALAR_TYPES):
        return True
    if isinstance(value, dict):
        return all(
            _walk_artifact_value(k) and _walk_artifact_value(v)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_walk_artifact_value(item) for item in value)
    return False


def validate_artifacts_serializable(command_output: "CommandOutput") -> list[str]:
    """Cheap recursive type-walk over each command_response's artifacts dict.

    Allowed: ``None``/``str``/``int``/``float``/``bool``, and ``dict``/``list``/
    ``tuple`` compositions thereof. Returns a list of human-readable problem
    descriptions (empty list means everything is serializable). [X3a]
    """
    problems: list[str] = []
    command_name = getattr(command_output, "command_name", "") or ""
    for command_response in getattr(command_output, "command_responses", []) or []:
        artifacts = getattr(command_response, "artifacts", None)
        if not isinstance(artifacts, dict):
            continue
        problems.extend(
            f"artifacts[{key!r}] on command '{command_name}' is {type(value)}"
            for key, value in artifacts.items()
            if not _walk_artifact_value(value)
        )
    return problems


def warn_on_unserializable_artifacts(command_output: "CommandOutput") -> None:
    """Warn (never raise) if a command output carries unserializable artifacts.

    Controlled by the ``FW_EAGER_ARTIFACT_VALIDATION`` environment variable
    (on by default; set to ``"0"`` to disable). In v2.21 this only emits a
    ``warnings.warn``; from v3.0 the same problems are rejected when the turn
    record is filed. [X3a]
    """
    if os.environ.get("FW_EAGER_ARTIFACT_VALIDATION", "1") == "0":
        return
    if problems := validate_artifacts_serializable(command_output):
        warnings.warn(
            "Unserializable command artifacts detected: "
            + "; ".join(problems)
            + ". These artifact values will be rejected at turn-record filing "
            "from fastWorkflow v3.0 onward; store only None/str/int/float/bool "
            "and dict/list/tuple compositions thereof.",
            stacklevel=2,
        )


class TurnResult(BaseModel):
    """The complete capture of one logical turn. [A22]

    ``answer`` is the user-facing response; ``command_outputs`` is the
    chronological list of every command execution in the turn (including
    ask_user clarification exchanges). One logical turn = one key = one
    record, across any number of suspensions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_key: str
    conversation_id: Optional[int] = None
    ordinal: Optional[int] = None
    status: TurnStatus
    failure_reason: Optional[str] = None
    user_message: str
    refined_user_message: Optional[str] = None
    entry_workflow_name: str = ""
    entry_context: str = ""
    answer: "CommandResponse"
    command_outputs: list["CommandOutput"] = []
    continuation_of: Optional[str] = None
    trajectory_ref: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    suspended_ms: int = 0
    metadata: dict[str, Any] = {}

    @computed_field
    @property
    def success(self) -> bool:
        """The single wire predicate for the turn: ``answer.success``. [A6][A42]"""
        return self.answer.success

    @property
    def command_outputs_with_artifacts(self) -> list:
        """Command outputs carrying artifacts, in turn order. [A9][A20]

        The subset of ``command_outputs`` where any command response has a
        non-empty ``artifacts`` dict — i.e. the outputs that carry structured
        data beyond plain text, in the order they occurred within the turn.

        The framework does not interpret the artifact keys; a consuming client
        decides which of these are worth rendering richly (a "gallery", a chart,
        a download, etc.).
        """
        return [
            command_output
            for command_output in self.command_outputs
            if any(
                response.artifacts
                for response in command_output.command_responses
            )
        ]
