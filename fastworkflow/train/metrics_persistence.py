"""Persist train-time metrics into the observability store (Phase 6, fix-kw7.7).

The observability design (docs/fastworkflow_observability_studio_design.md §3.2,
§4 "Train metrics unpersisted") calls for one ``train_runs`` row per successful
training publication. This module is the seam between the train pipeline and
``ObservabilityStore.record_train_run``:

- ``collect_train_metrics`` assembles a metrics dict by READING the small JSON
  artifacts a successful train has just published into ``___command_info``
  (threshold files, heldout_evaluation.json, training_report.json, the version
  manifest). It never trains, never loads models, and degrades to a partial
  dict on any per-artifact read failure.
- ``persist_train_run_metrics`` writes the row, gated on ``FW_OBSERVABILITY``
  (train is a fastworkflow entry point, so the default is ON [R4]). Durability
  class [R14] applies: a persistence failure must never fail the training run,
  so every failure path is a ``logger.warning`` and a ``None`` return.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastworkflow.utils.logging import logger

# Small per-context JSON files published beside the models. Absent files are
# simply omitted from the metrics — carried-forward or older-layout contexts do
# not all have every variant.
_THRESHOLD_FILENAMES = (
    "threshold.json",
    "ambiguous_threshold.json",
    "tiny_ambiguous_threshold.json",
    "large_ambiguous_threshold.json",
)

_COMMAND_INFO_DIRNAME = "___command_info"


def _read_json(path: str) -> Optional[Any]:
    """Best-effort read of one small JSON artifact; None when absent/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _context_thresholds(context_dir: str) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for filename in _THRESHOLD_FILENAMES:
        data = _read_json(os.path.join(context_dir, filename))
        if isinstance(data, dict) and "confidence_threshold" in data:
            key = filename[: -len(".json")]
            thresholds[key] = data["confidence_threshold"]
    return thresholds


def collect_train_metrics(
    workflow_path: str,
    version_id: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the train-run metrics dict from published ``___command_info``.

    Read-only and cheap: only the small JSON artifacts of the CURRENT published
    layout are opened (no models, no retraining). Structure::

        {"version_id": ..., "seed": ..., "train_duration_seconds": ...,
         "contexts_retrained": [...], "contexts_carried_forward": [...],
         "models": {"tiny": ..., "large": ...},
         "contexts": {"<folder>": {"thresholds": {...}, "heldout": {...}}},
         "commands": {"<name>": {"seed_count": ..., "generated_count": ...,
                                 "row_count": ...}},
         "totals": {...heldout totals...}}

    Every section is best-effort; a missing artifact yields a missing section,
    never an exception.
    """
    metrics: dict[str, Any] = {"contexts": {}, "totals": {}}
    try:
        _collect_into(metrics, workflow_path, version_id)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a train run
        logger.warning(
            f"Train-metrics collection for {workflow_path} was cut short: {exc!r}"
        )
    return metrics


def _collect_into(
    metrics: dict[str, Any], workflow_path: str, version_id: Optional[str]
) -> None:
    info_dir = os.path.join(workflow_path, _COMMAND_INFO_DIRNAME)

    # -- version manifest: seed, duration, which contexts this run retrained --
    if version_id is not None:
        metrics["version_id"] = version_id
        try:
            from fastworkflow.train import artifact_versioning

            manifest = artifact_versioning.read_manifest(workflow_path, version_id)
        except Exception:
            manifest = {}
        for key in (
            "seed",
            "train_duration_seconds",
            "contexts_retrained",
            "contexts_carried_forward",
            "previous_version",
        ):
            if manifest.get(key) is not None:
                metrics[key] = manifest[key]

    # -- base model ids (env-configurable; defaults mirror the trainer's) -----
    try:
        import fastworkflow

        metrics["models"] = {
            "tiny": fastworkflow.get_env_var(
                "INTENT_DETECTION_TINY_MODEL",
                default="google/bert_uncased_L-4_H-128_A-2",
            ),
            "large": fastworkflow.get_env_var(
                "INTENT_DETECTION_LARGE_MODEL", default="distilbert-base-uncased"
            ),
        }
    except Exception:
        pass

    # -- per-context thresholds from the published compatibility layout -------
    # Contexts are discovered the same way _prune_stale_artifacts recognises
    # them: a directory holding threshold.json. RESERVED_TOPLEVEL_NAMES (the
    # versions/ store and the current pointer) are never context folders.
    try:
        from fastworkflow.train.artifact_versioning import RESERVED_TOPLEVEL_NAMES

        reserved = set(RESERVED_TOPLEVEL_NAMES)
    except Exception:
        reserved = {"versions", "current", "current.json"}
    try:
        entries = sorted(os.listdir(info_dir)) if os.path.isdir(info_dir) else []
    except OSError:
        entries = []
    for entry in entries:
        if entry in reserved:
            continue
        context_dir = os.path.join(info_dir, entry)
        if not os.path.isdir(context_dir):
            continue
        thresholds = _context_thresholds(context_dir)
        if thresholds:
            metrics["contexts"].setdefault(entry, {})["thresholds"] = thresholds

    # -- heldout evaluation summary (already merged across selective runs) ----
    heldout = _read_json(os.path.join(info_dir, "heldout_evaluation.json"))
    if isinstance(heldout, dict):
        if isinstance(heldout.get("totals"), dict):
            metrics["totals"] = heldout["totals"]
        for context_report in heldout.get("contexts") or []:
            if not isinstance(context_report, dict):
                continue
            context_name = context_report.get("context")
            if not isinstance(context_name, str):
                continue
            # "*" is not a legal folder name; it publishes under "global".
            folder = "global" if context_name == "*" else context_name
            summary = {
                key: context_report[key]
                for key in ("context", "in_distribution_f1", "routing", "escalation")
                if context_report.get(key) is not None
            }
            metrics["contexts"].setdefault(folder, {})["heldout"] = summary

    # -- per-command utterance counts from the training-data report -----------
    report = _read_json(os.path.join(info_dir, "training_report.json"))
    if isinstance(report, dict) and isinstance(report.get("rows"), list):
        commands: dict[str, Any] = {}
        for row in report["rows"]:
            if not isinstance(row, dict) or not row.get("command_name"):
                continue
            commands[row["command_name"]] = {
                key: row.get(key)
                for key in ("seed_count", "generated_count", "row_count", "status")
                if row.get(key) is not None
            }
        if commands:
            metrics["commands"] = commands


def persist_train_run_metrics(
    workflow_path: str,
    started_at: Optional[datetime],
    completed_at: Optional[datetime],
    metrics: dict[str, Any],
) -> Optional[str]:
    """Write one ``train_runs`` row for a just-published training run.

    Gated on ``FW_OBSERVABILITY`` (default ON — train is a fastworkflow entry
    point [R4]). Never raises: any failure is a warning and a ``None`` return,
    because metrics persistence must never fail the training run [R14].

    Returns the run_id written, or None when disabled or on failure.
    """
    try:
        from fastworkflow import observability_store, state_paths

        if not observability_store.observability_enabled(default_on=True):
            return None

        if completed_at is None:
            completed_at = datetime.now(timezone.utc)
        run_id = f"{completed_at:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"

        workflow_fingerprint: Optional[str] = None
        try:
            from fastworkflow.command_directory import (
                compute_commands_source_fingerprint,
            )

            workflow_fingerprint = compute_commands_source_fingerprint(workflow_path)
        except Exception as exc:
            logger.warning(
                f"Could not fingerprint {workflow_path} for the train_runs row: {exc!r}"
            )

        store = observability_store.ObservabilityStore(
            state_paths.observability_db(workflow_path)
        )
        store.record_train_run(
            run_id=run_id,
            workflow_fingerprint=workflow_fingerprint,
            started_at=started_at.isoformat() if started_at else None,
            completed_at=completed_at.isoformat(),
            metrics=metrics,
        )
        return run_id
    except Exception as exc:
        logger.warning(
            f"Could not persist train-run metrics for {workflow_path}: {exc!r}"
        )
        return None
