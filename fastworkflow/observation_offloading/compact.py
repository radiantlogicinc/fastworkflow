"""Oldest-first packed-trajectory compaction (Arm D)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping, Optional

from fastworkflow import context_budget
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.labels import (
    command_response,
    estimated_tokens,
    label_alias,
    offload_label,
    offload_saving_bytes,
    printed_subject,
    replacement_saves_space,
)
from fastworkflow.observation_offloading.state import (
    archive,
    default_scope,
    record_event,
)

#: An execute observation is worth offloading when replacing it with its own
#: label frees at least this many UTF-8 bytes of trajectory .
#: It replaces a 1,000-estimated-token floor (~4 KB) that asked how big the
#: observation was rather than how much residency the swap would buy: a 3 KB
#: listing page stayed resident for the whole turn while its label would have
#: cost ~400 B, and a 300 B fact could never be worth replacing at all because
#: its label is larger than it is. The label is the actual label for that step,
#: description and command text included, so the saving is the real one.
#: The value at the reference context window. The effective one is a fraction of
#: the model's window (``context_budget.OFFLOAD_MIN_SAVING``).
MIN_OFFLOAD_SAVING_BYTES = context_budget.REFERENCE_OFFLOAD_MIN_SAVING_BYTES
RECENT_OBSERVATIONS_PROTECTED = 5
#: The packed-trajectory target at the reference context window
#: (``context_budget.TRAJECTORY``).
PACKED_TARGET_BYTES = context_budget.REFERENCE_TRAJECTORY_MAX_BYTES
#: The tuning overrides. The budgets themselves come from the window.
TRAJECTORY_MAX_BYTES_ENV = context_budget.TRAJECTORY.override_env
MIN_OFFLOAD_SAVING_BYTES_ENV = context_budget.OFFLOAD_MIN_SAVING.override_env


#: The tool whose steps own the agent-visible ``O`` namespace. One name, so
#: the dispatch-side ledger and the printed alias agree on what counts.
EXECUTE_TOOL_NAME = "execute_workflow_query"

_STEP_KEY = re.compile(r"^(?:tool_name|observation)_(\d+)$")


def packed_target_bytes_from_env() -> int:
    """The packed-trajectory target for this run. See ``fastworkflow.context_budget``."""
    return context_budget.trajectory_max_bytes()


def min_offload_saving_bytes_from_env() -> int:
    """The minimum saving an offload must buy. ``0`` means "whenever the label is smaller"."""
    return context_budget.offload_min_saving_bytes()


def step_indexes(trajectory: Mapping[str, Any]) -> list[int]:
    """Every step index still present, ascending, gaps included.

    The base ReAct's context-window fallback pops the oldest step's keys, so the
    trajectory can start at step 3 or skip a step in the middle. Scanning the
    keys, rather than counting up from zero until the first miss, keeps
    compaction and replan skeletons working after such a truncation.
    """
    indexes: set[int] = set()
    for key in trajectory:
        match = _STEP_KEY.match(str(key))
        if match:
            indexes.add(int(match.group(1)))
    return sorted(indexes)


def execute_step_indexes(trajectory: Mapping[str, Any]) -> list[int]:
    """Step indexes of every ``execute_workflow_query`` step still present."""
    return [
        index
        for index in step_indexes(trajectory)
        if str(trajectory.get(f"tool_name_{index}") or "") == EXECUTE_TOOL_NAME
    ]


def archive_step(
    trajectory: Mapping[str, Any],
    step_index: int,
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
) -> bool:
    """Persist one execute observation under its ``O{step_index}`` alias. True if stored.

    Called once, when the step completes, so every execute observation is
    written exactly once and before any label can replace it. A failure to
    persist is recorded as ``archive_refused`` and the observation stays inline;
    compaction then finds no stored row and does not offload it.
    """
    if str(trajectory.get(f"tool_name_{step_index}") or "") != EXECUTE_TOOL_NAME:
        return False
    shown = trajectory.get(f"observation_{step_index}")
    if not isinstance(shown, str):
        return False
    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    alias = f"O{step_index}"
    original = command_response(shown, alias)
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    args = trajectory.get(f"tool_args_{step_index}")
    command = ""
    if isinstance(args, Mapping):
        command = str(args.get("command") or "")
    context_clause, context_changed = printed_subject(shown)
    try:
        store.persist(
            selected_scope,
            alias=alias,
            command_name=command,
            step_index=step_index,
            text=original,
            text_sha256=digest,
            context_clause=context_clause,
            context_changed=context_changed,
        )
    except Exception as error:  # noqa: BLE001
        record_event(
            {
                "kind": "archive_refused",
                "alias": alias,
                "step_index": step_index,
                "reason": "persistence_failed_original_retained",
                "error": type(error).__name__,
            }, scope=selected_scope, store=store
        )
        return False
    record_event({
        "kind": "observation_archived",
        "alias": alias,
        "step_index": step_index,
        "text_sha256": digest,
        "utf8_bytes": len(original.encode("utf-8")),
    }, scope=selected_scope, store=store)
    return True


def _over_packed_target(
    text: str,
    *,
    packed_target_bytes: int,
    packed_target_tokens: Optional[int],
) -> bool:
    if packed_target_tokens is not None:
        return estimated_tokens(text) > packed_target_tokens
    return len(text.encode("utf-8")) > packed_target_bytes


def compact_trajectory(
    trajectory: dict[str, Any],
    *,
    step_index: int,
    min_offload_saving_bytes: Optional[int] = None,
    recent_observations_protected: int = RECENT_OBSERVATIONS_PROTECTED,
    packed_target_tokens: Optional[int] = None,
    packed_target_bytes: Optional[int] = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> list[dict[str, Any]]:
    """Mutate trajectory observations in place. Return offload decisions.

    An execute observation is eligible when replacing it with its own label
    frees at least ``min_offload_saving_bytes`` UTF-8 bytes
    (``MIN_OFFLOAD_SAVING_BYTES``, or ``FW_OFFLOAD_MIN_SAVING_BYTES``). The
    label is built first for exactly that reason: eligibility is a property of
    the swap, not of the observation. Everything around it is unchanged --
    oldest-first order, the five most recent execute observations protected,
    the packed target, and ``replacement_saves_space``.

    ``step_index`` is the step that just completed. It is archived here, once,
    before any offload decision. An observation is offloadable only if the
    store holds it; one that could not be stored stays inline.

    Recency protection applies to the last ``recent_observations_protected``
    execute steps still present in the trajectory.
    """

    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    if min_offload_saving_bytes is None:
        min_offload_saving_bytes = min_offload_saving_bytes_from_env()
    if packed_target_tokens is None and packed_target_bytes is None:
        packed_target_bytes = packed_target_bytes_from_env()
    executes = execute_step_indexes(trajectory)
    if not executes:
        return []
    archive_step(trajectory, step_index, scope=selected_scope, selected_archive=store)
    if recent_observations_protected <= 0:
        protected_steps: set[int] = set()
    else:
        protected_steps = set(executes[-recent_observations_protected:])
    packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
    decisions: list[dict[str, Any]] = []
    for step_index in executes:
        key = f"observation_{step_index}"
        response = trajectory.get(key)
        if not isinstance(response, str):
            continue
        alias = f"O{step_index}"
        # Every offload decision is taken on the exact command response, so the
        # printed handle cannot shift eligibility, savings or the stored digest.
        original = command_response(response, alias)
        size = {
            "characters": len(original),
            "utf8_bytes": len(original.encode("utf-8")),
            "estimated_tokens": estimated_tokens(original),
        }
        recency_protected = step_index in protected_steps
        already_label = label_alias(response) == alias
        decision = {
            "alias": alias,
            "step_index": step_index,
            "action": "kept",
            "reason": "below_min_saving",
            "recency_protected": recency_protected,
            "response_size": size,
        }
        if already_label:
            decision["reason"] = "already_label"
            decisions.append(decision)
            continue
        if recency_protected:
            decision["reason"] = "recent_observation_protected"
            decisions.append(decision)
            continue
        # Eligibility is the saving, so the label has to exist before the
        # question can be asked. It is the label this step would really get --
        # same alias, same command text, same authored description -- never a
        # stand-in, or the measured saving would not be the one taken.
        args = trajectory.get(f"tool_args_{step_index}")
        command = ""
        if isinstance(args, Mapping):
            command = str(args.get("command") or "")
        label = offload_label(
            alias=alias,
            command_name=command or "execute_workflow_query",
            response=original,
            description=describe_output(command, original) if describe_output else "",
        )
        saving = offload_saving_bytes(original, label)
        decision["offload_saving_bytes"] = saving
        decision["label_size"] = {
            "characters": len(label),
            "utf8_bytes": len(label.encode("utf-8")),
        }
        if saving < min_offload_saving_bytes:
            decision["reason"] = "below_min_saving"
            decision["min_offload_saving_bytes"] = min_offload_saving_bytes
        elif not _over_packed_target(
            packed_text,
            packed_target_bytes=packed_target_bytes,
            packed_target_tokens=packed_target_tokens,
        ):
            decision["reason"] = "eligible_but_target_already_met"
        else:
            if not replacement_saves_space(original, label):
                decision["reason"] = "replacement_not_smaller"
                decisions.append(decision)
                continue
            if store.get(selected_scope, alias) is None:
                decision["reason"] = "not_archived"
                decisions.append(decision)
                continue
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
            packed_utf8_bytes_before = len(packed_text.encode("utf-8"))
            trajectory[key] = label
            decision["action"] = "offloaded"
            decision["reason"] = "oldest_eligible_until_target"
            decision["label"] = label
            decision["text_sha256"] = digest
            decision["packed_utf8_bytes_before"] = packed_utf8_bytes_before
            packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
            record_event(
                {
                    "kind": "offload",
                    "alias": alias,
                    "step_index": step_index,
                    "text_sha256": digest,
                    "estimated_tokens": size["estimated_tokens"],
                    "offload_saving_bytes": saving,
                    "packed_utf8_bytes_before": packed_utf8_bytes_before,
                    "packed_utf8_bytes_after": len(packed_text.encode("utf-8")),
                }, scope=selected_scope, store=store
            )
        decisions.append(decision)
    return decisions
