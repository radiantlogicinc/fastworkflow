"""Read-only projection of the recorded `train_runs` rows (`fix-9eg.2`).

`train.metrics_persistence` writes one row per published training run; nothing
until now read them back. This module is that reader, and it is a PROJECTION
only: it opens no artifact directory, runs no training, downloads nothing and
never writes. Everything it reports comes from rows already in the store, which
is what lets it run against a sealed post-mortem snapshot and against a
workspace's read-only stores on equal terms.

Three rules shape the whole file.

**Original labels are kept.** `in_distribution_f1`, `routing`, `escalation` and
the per-context `thresholds` are passed through under the names
`heldout_evaluation.json` gave them, against the dataset those names refer to.
They are held-out *intent-classification* measurements on synthetic utterances;
they are not a task-success rate, not a benchmark score, and nothing here
renames, averages or re-scales them into one. A viewer that saw "score: 0.93"
would reasonably read it as "93% of tasks passed", which the number does not
say about anything.

**Absence is reported, never filled in.** A run whose metrics were withheld by
the capture policy, or whose JSON will not parse, is a different state from a
run that recorded nothing and from a run with no training evidence at all;
`metrics_status` separates them. `record_train_run` classifies the metrics blob
as `opaque-payload`, so an evidence-profile deployment legitimately stores a
policy envelope there instead of the numbers, and a reader that rendered that
envelope as data would be reporting a badge as a measurement.

**A runtime link needs a recorded identity on both sides.** The only link this
module will draw is an exact match between the train run's recorded
`version_id` and an attempt's recorded `runtime_snapshot.workflow_model_version`
-- the same published-version id, written independently by the trainer and by
the binding server. Nothing else is accepted, and in particular
`workflow_fingerprint` is NOT a fallback: `___command_info/` is under no root
the source fingerprint hashes (see `experiment.readiness.workflow_model_version`),
so a retrain leaves the fingerprint byte-identical, and two runs agreeing on it
is not evidence that they served the same model set. Neither is "the newest
train run": a workflow on the pre-versioning layout records no version id at
all while being fully trained, so the newest row is a guess dressed as a fact.
Those cases resolve to an explicit `unavailable` status with the reason.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastworkflow.observability.capture_policy import is_capture_envelope

# How many experiments one runtime-link lookup will walk. A link is a
# convenience on a detail page, not a search: an unbounded scan of every
# experiment and every attempt of a large store would make opening one training
# run cost the whole experiment history.
RUNTIME_LINK_EXPERIMENT_SCAN = 100

# `metrics_status` vocabulary. Four states, because collapsing any two of them
# loses the difference between "we know there was nothing" and "we are not
# allowed to tell you" -- which is exactly what a reader of a redacted evidence
# bundle needs to see.
METRICS_RECORDED = "recorded"
METRICS_WITHHELD = "withheld"
METRICS_UNREADABLE = "unreadable"
METRICS_ABSENT = "absent"

# `runtime_link.status` vocabulary.
LINK_LINKED = "linked"
LINK_NO_MATCH = "no_match"
LINK_UNAVAILABLE = "unavailable"

# The manifest keys `collect_train_metrics` copies through, in the order a
# reader wants them. Names are the manifest's own.
_MANIFEST_KEYS = (
    "seed",
    "train_duration_seconds",
    "previous_version",
    "contexts_retrained",
    "contexts_carried_forward",
)

# The per-context heldout keys, under the names `heldout_evaluation.json` uses.
_HELDOUT_KEYS = ("context", "in_distribution_f1", "routing", "escalation")


def decode_metrics(raw: Any) -> tuple[str, dict[str, Any]]:
    """`(metrics_status, metrics)` for one row's `metrics_json` column.

    `metrics` is `{}` for every status but `recorded`, so callers may read it
    without branching first; the status is what they must render.
    """
    if raw is None or raw == "":
        return METRICS_ABSENT, {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return METRICS_UNREADABLE, {}
    if is_capture_envelope(decoded):
        return METRICS_WITHHELD, {}
    if not isinstance(decoded, dict):
        return METRICS_UNREADABLE, {}
    return METRICS_RECORDED, decoded


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _base_models(metrics: dict[str, Any]) -> dict[str, str]:
    """The base model ids the run was configured with, as recorded.

    These are the HuggingFace checkpoints the classifiers were fine-tuned from
    (`INTENT_DETECTION_TINY_MODEL` / `..._LARGE_MODEL`), not the trained model
    set's identity -- that is `version_id`. Kept apart because conflating them
    would let two different trained sets look like the same model.
    """
    models = metrics.get("models")
    if not isinstance(models, dict):
        return {}
    return {
        role: value
        for role, value in sorted(models.items())
        if isinstance(role, str) and isinstance(value, str) and value
    }


def _contexts(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per trained context folder, each keeping its own labels."""
    recorded = metrics.get("contexts")
    if not isinstance(recorded, dict):
        return []
    contexts: list[dict[str, Any]] = []
    for folder in sorted(recorded):
        body = recorded[folder]
        if not isinstance(body, dict):
            continue
        entry: dict[str, Any] = {"context_folder": folder}
        thresholds = body.get("thresholds")
        entry["thresholds"] = thresholds if isinstance(thresholds, dict) else {}
        heldout = body.get("heldout")
        if isinstance(heldout, dict):
            entry["heldout"] = {
                key: heldout[key] for key in _HELDOUT_KEYS if key in heldout
            }
        else:
            # Explicit: a context can be published with thresholds and no
            # held-out report (a carried-forward context of a selective run),
            # and an empty dict there would read as "evaluated, all zero".
            entry["heldout"] = None
        contexts.append(entry)
    return contexts


def training_run_summary(row: dict[str, Any]) -> dict[str, Any]:
    """The list-row projection of one `train_runs` row.

    Cheap by construction: the whole metrics blob is already in the row, so
    this costs a JSON parse and no further reads. The columns outside
    `metrics_json` (run_id, fingerprint, timestamps) are unpoliced by
    `record_train_run`, which is why a withheld run still lists.
    """
    status, metrics = decode_metrics(row.get("metrics_json"))
    version_id = _text(metrics.get("version_id"))
    summary: dict[str, Any] = {
        "run_id": row.get("run_id"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "workflow_fingerprint": row.get("workflow_fingerprint"),
        "metrics_status": status,
        "version_id": version_id,
        # Named for what it licenses rather than for the field: this is the
        # precondition for drawing a runtime link at all.
        "model_identity_recorded": version_id is not None,
        "base_models": _base_models(metrics),
        "contexts_retrained": _string_list(metrics.get("contexts_retrained")),
        "contexts_carried_forward": _string_list(
            metrics.get("contexts_carried_forward")
        ),
        "context_count": len(_contexts(metrics)),
    }
    for key in ("seed", "train_duration_seconds", "previous_version"):
        summary[key] = metrics.get(key) if key in metrics else None
    return summary


def list_training_runs(store: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    """Recorded training runs, newest first, as the list projection.

    Order is `list_train_runs`' own (`COALESCE(completed_at, started_at) DESC`)
    and is not re-sorted here: the store's order is the one every other reader
    of these rows sees.
    """
    return [training_run_summary(row) for row in store.list_train_runs(limit=limit)]


def _attempt_links(
    store: Any, version_id: str, experiments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attempts whose stamped snapshot names exactly this published version.

    Takes the already-read experiment window rather than re-reading it: the
    caller needs the same list to report `experiments_scanned`, and two reads
    of a live store could disagree about it.
    """
    links: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_id = experiment.get("experiment_id")
        if not experiment_id:
            continue
        for attempt in store.experiment_attempt_rows(experiment_id):
            snapshot = attempt.get("runtime_snapshot")
            if not isinstance(snapshot, dict):
                continue
            if _text(snapshot.get("workflow_model_version")) != version_id:
                continue
            links.append(
                {
                    "experiment_id": experiment_id,
                    "task_id": attempt.get("task_id"),
                    "attempt": attempt.get("attempt"),
                    "channel_id": attempt.get("channel_id"),
                    "conversation_id": attempt.get("conversation_id"),
                    "outcome": attempt.get("outcome"),
                    "started_at": attempt.get("started_at"),
                    # Carried so a reader can see the tree was unchanged across
                    # a retrain rather than inferring it; it is never what
                    # matched.
                    "workflow_fingerprint": snapshot.get("workflow_fingerprint"),
                    "workflow_model_legacy_layout": snapshot.get(
                        "workflow_model_legacy_layout"
                    ),
                }
            )
    return links


def runtime_link(
    store: Any,
    version_id: Optional[str],
    *,
    experiment_scan: int = RUNTIME_LINK_EXPERIMENT_SCAN,
) -> dict[str, Any]:
    """Which recorded runs served this training run's model set.

    Three answers, and the two negative ones are not the same thing:

    * `unavailable` -- this training run recorded no version id, so no link is
      derivable in either direction. Not "nothing ran on it".
    * `no_match` -- the identity is on record and no attempt in the scanned
      window stamped it. A real answer about the evidence present.
    * `linked` -- the attempts below stamped exactly this version.
    """
    if version_id is None:
        return {
            "status": LINK_UNAVAILABLE,
            "reason": (
                "this training run recorded no published version id, so no "
                "runtime run can be matched to it. A workflow trained on the "
                "pre-versioning artifact layout records none while being fully "
                "trained; the association is unavailable, not absent."
            ),
            "version_id": None,
            "attempts": [],
            "experiments_scanned": 0,
            "scan_bounded": False,
        }
    experiments = store.list_experiments(limit=experiment_scan)
    attempts = _attempt_links(store, version_id, experiments)
    return {
        "status": LINK_LINKED if attempts else LINK_NO_MATCH,
        "reason": None
        if attempts
        else (
            "no recorded attempt in the scanned window stamped this version id"
        ),
        "version_id": version_id,
        "attempts": attempts,
        "experiments_scanned": len(experiments),
        # Said out loud, because `no_match` over a truncated scan is a weaker
        # statement than `no_match` over all of them.
        "scan_bounded": len(experiments) >= experiment_scan,
    }


def training_run_detail(
    store: Any,
    run_id: str,
    *,
    experiment_scan: int = RUNTIME_LINK_EXPERIMENT_SCAN,
) -> Optional[dict[str, Any]]:
    """One run's full projection, or None when the store holds no such run.

    A primary-key read (`ObservabilityStore.get_train_run`), deliberately NOT
    a search of the list window: the list is bounded and newest-first, so
    resolving a detail out of it would make an old run stop being readable as
    soon as enough newer ones existed -- a 404 about the caller's page size
    rather than about the evidence. `None` here means the store holds no such
    row at all.
    """
    row = store.get_train_run(run_id)
    if row is None:
        return None
    status, metrics = decode_metrics(row.get("metrics_json"))
    detail = training_run_summary(row)
    detail["contexts"] = _contexts(metrics)
    totals = metrics.get("totals")
    detail["totals"] = totals if isinstance(totals, dict) else {}
    commands = metrics.get("commands")
    detail["commands"] = commands if isinstance(commands, dict) else {}
    detail["manifest"] = {
        key: metrics[key] for key in _MANIFEST_KEYS if key in metrics
    }
    detail["metrics_status"] = status
    detail["runtime_link"] = runtime_link(
        store, detail["version_id"], experiment_scan=experiment_scan
    )
    return detail


__all__ = [
    "LINK_LINKED",
    "LINK_NO_MATCH",
    "LINK_UNAVAILABLE",
    "METRICS_ABSENT",
    "METRICS_RECORDED",
    "METRICS_UNREADABLE",
    "METRICS_WITHHELD",
    "RUNTIME_LINK_EXPERIMENT_SCAN",
    "decode_metrics",
    "list_training_runs",
    "runtime_link",
    "training_run_detail",
    "training_run_summary",
]
