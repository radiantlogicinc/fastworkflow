"""Experiment provenance flattening.

Moved verbatim from ``run_chatbot.server``. No handler state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from fastworkflow.run_chatbot.turn_annotations import (
    _EXPERIMENT_PROVENANCE_COLUMNS,
    _SNAPSHOT_PROVENANCE_KEYS,
    _text_or_none,
)

# -- (c) provenance and comparability -------------------------------------------


def _flatten_provenance(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Nested provenance maps flatten to dotted keys; scalars and lists stay
    as they are, so a per-emitter contract version reads as
    `span_contract_versions.fw.turn: 1`."""
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten_provenance(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    else:
        out[prefix] = value


def _git_revision_of(record: Mapping[str, Any]) -> Optional[str]:
    """The engine's source revision, wherever a harness put it in the record.

    The evidence-run record (`EvidenceRun.as_record`) carries the
    ObservabilityProvenance only; the EngineProvenance with
    `source_revision` lives in the harness's RuntimeProvenance bundle, which
    this store never persists. These are the places a record might hold it;
    none of the trial's records did.
    """
    candidates = (
        record.get("engine"),
        (record.get("provenance") or {}).get("engine")
        if isinstance(record.get("provenance"), dict)
        else None,
        (record.get("runtime") or {}).get("engine")
        if isinstance(record.get("runtime"), dict)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("source_revision", "git_revision"):
                if _text_or_none(candidate.get(key)):
                    return candidate[key]
    for key in ("source_revision", "git_revision"):
        if _text_or_none(record.get(key)):
            return record[key]
    return None


def experiment_provenance(
    detail: Mapping[str, Any], attempts: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """An experiment's provenance, one field per row, keys verbatim.

    Three sources, each named on its field: the experiment row's own columns;
    the evidence-run records' `observability` block (ObservabilityProvenance:
    span-contract version, per-emitter versions, DB
    schema, the FW_OBS_* config in effect); and the attempts' runtime
    snapshots (workflow fingerprint and model version). A field two segments
    or two attempts disagree on is reported with every value and where each
    came from, never collapsed to one. `git_revision` is listed even when
    nothing recorded it, because its absence is the fact a reader needs.
    """
    fields: list[dict[str, Any]] = []

    def add(key: str, source: str, observations: list[tuple[str, Any]]) -> None:
        recorded = [(where, value) for where, value in observations if value is not None]
        if not recorded:
            fields.append(
                {"key": key, "source": source, "recorded": False, "value": None,
                 "consistent": True, "values": []}
            )
            return
        distinct: list[Any] = []
        for _, value in recorded:
            if value not in distinct:
                distinct.append(value)
        fields.append(
            {
                "key": key,
                "source": source,
                "recorded": True,
                "value": distinct[0] if len(distinct) == 1 else None,
                "consistent": len(distinct) == 1,
                "values": [{"where": where, "value": value} for where, value in recorded],
            }
        )

    for column in _EXPERIMENT_PROVENANCE_COLUMNS:
        add(column, "experiment", [("experiment", detail.get(column))])

    per_key: dict[str, list[tuple[str, Any]]] = {}
    revisions: list[tuple[str, Any]] = []
    for segment in detail.get("evidence_runs") or []:
        record = segment.get("record") if isinstance(segment, dict) else None
        if not isinstance(record, dict):
            continue
        where = f"evidence segment #{segment.get('seq')}"
        observability = record.get("observability")
        flat: dict[str, Any] = {}
        if isinstance(observability, dict):
            _flatten_provenance("", observability, flat)
        for key, value in flat.items():
            per_key.setdefault(key, []).append((where, value))
        revisions.append((where, _git_revision_of(record)))
    for key in sorted(per_key):
        add(key, "evidence_run", per_key[key])
    add("git_revision", "evidence_run", revisions)

    snapshot_keys: dict[str, list[tuple[str, Any]]] = {}
    for row in attempts:
        snapshot = row.get("runtime_snapshot")
        if not isinstance(snapshot, dict):
            continue
        where = f"attempt {row.get('task_id')}#{row.get('attempt')}"
        for key in _SNAPSHOT_PROVENANCE_KEYS:
            if key in snapshot:
                snapshot_keys.setdefault(key, []).append((where, snapshot[key]))
        features = snapshot.get("effective_features")
        if isinstance(features, dict):
            for name in sorted(features):
                snapshot_keys.setdefault(f"effective_features.{name}", []).append(
                    (where, features[name])
                )
    for key in sorted(snapshot_keys):
        add(key, "runtime_snapshot", snapshot_keys[key])

    return {
        "fields": fields,
        "recorded": sum(1 for field in fields if field["recorded"]),
        "unrecorded": sum(1 for field in fields if not field["recorded"]),
        "inconsistent": sum(1 for field in fields if not field["consistent"]),
    }


def provenance_differences(
    treatment: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every provenance field the two experiments do not agree on, verbatim.

    A field recorded on one side and not the other differs; a field recorded
    on neither does not (there is nothing to quote). A field a side's own
    segments disagree on is compared as the list of its observed values.
    """

    def by_key(provenance: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (field["source"], field["key"]): field
            for field in provenance.get("fields") or []
        }

    def value_of(field: Optional[Mapping[str, Any]]) -> Any:
        if field is None or not field.get("recorded"):
            return None
        if field.get("consistent"):
            return field.get("value")
        return [entry.get("value") for entry in field.get("values") or []]

    left, right = by_key(treatment), by_key(baseline)
    differences = []
    for source, key in sorted(set(left) | set(right)):
        t_value = value_of(left.get((source, key)))
        b_value = value_of(right.get((source, key)))
        if t_value is None and b_value is None:
            continue
        if t_value == b_value:
            continue
        differences.append(
            {"key": key, "source": source, "treatment": t_value, "baseline": b_value}
        )
    return differences
