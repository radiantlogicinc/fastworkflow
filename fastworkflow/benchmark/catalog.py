"""Workflow-local versioned benchmark catalogs, kept in the live observability DB.

A benchmark version is one row of ``benchmark_versions`` in the workflow's live
DB (``observability/control.py``, marker ``benchmark_catalog_v1``). The row is
the ONLY copy: there is no file mirror and no fallback. ``manifest_bytes`` is
the exact canonical JSON, and its sha256 is the digest a run pins. A row with
tasks is immutable (triggers abort UPDATE and DELETE); a row with no tasks is a
draft that ``replace_draft_version`` rewrites in place. Only tasks are frozen:
the benchmark's title and each version's description are kept in their own
tables (``set_details``) and laid over the manifest by ``load_version``. Reads
never create the DB; saving a version may.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from fastworkflow import state_paths
from fastworkflow.observability import control
from fastworkflow.observability.store import open_live_store, open_live_store_for_write

SCHEMA = "fastworkflow-benchmark/1"
_NATURAL_SORT_CHUNKS = re.compile(r"(\d+)")


class BenchmarkCatalogError(RuntimeError):
    """Base class for benchmark catalog failures."""


class BenchmarkManifestError(BenchmarkCatalogError, ValueError):
    """A benchmark version does not satisfy the v1 contract."""


class BenchmarkAlreadyExistsError(BenchmarkCatalogError):
    """Refused to overwrite an existing immutable benchmark version."""


class BenchmarkArchiveBlocked(BenchmarkCatalogError):
    """Archiving would have to archive an experiment that cannot be archived."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_segment(value: Any, field: str) -> str:
    raw = _required_text(value, field)
    normalized = PurePath(raw.replace("\\", "/"))
    if (
        Path(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ".." in normalized.parts
        or len(normalized.parts) != 1
    ):
        raise BenchmarkManifestError(
            f"{field} must be a single path segment and must not contain '/' or '..'"
        )
    return raw


def _natural_sort_key(text: str) -> list[object]:
    """Sort ``v1, v2, v10`` numerically rather than lexically."""
    return [
        (0, int(chunk)) if chunk.isdigit() else (1, chunk)
        for chunk in _NATURAL_SORT_CHUNKS.split(text)
        if chunk
    ]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: dict[str, Any]) -> bytes:
    text = json.dumps(data, sort_keys=True, indent=2) + "\n"
    return text.encode("utf-8")


def _require_json_native(value: Any, field: str) -> None:
    """Refuse Python values that JSON would coerce or cannot represent faithfully."""

    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise BenchmarkManifestError(f"{field} keys must be strings")
            _require_json_native(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_native(item, f"{field}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkManifestError(f"{field} must be a finite number")
        return
    raise BenchmarkManifestError(
        f"{field} must be a JSON-native value "
        "(object, array, string, number, boolean, or null)"
    )


def _validate_tasks(raw: Any, field: str = "tasks") -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise BenchmarkManifestError(f"{field} must be an array")
    seen: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkManifestError(f"{item_field} must be an object")
        task_id = _required_text(item.get("task_id"), f"{item_field}.task_id")
        if task_id in seen:
            raise BenchmarkManifestError(f"duplicate task_id {task_id!r}")
        seen.add(task_id)
        description = item.get("description", "")
        if not isinstance(description, str):
            raise BenchmarkManifestError(f"{item_field}.description must be text")
        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise BenchmarkManifestError(f"{item_field}.payload must be a JSON object")
        _require_json_native(payload, f"{item_field}.payload")
        task = {"task_id": task_id, "description": description, "payload": payload}
        if "prompt" in item:
            if not isinstance(item["prompt"], str):
                raise BenchmarkManifestError(f"{item_field}.prompt must be text")
            task["prompt"] = item["prompt"]
        tasks.append(task)
    return tasks


def _validate_manifest(
    data: Any,
    *,
    expected_benchmark_id: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BenchmarkManifestError("benchmark manifest must be a JSON object")
    schema = data.get("schema")
    if schema != SCHEMA:
        raise BenchmarkManifestError(
            f"schema must be {SCHEMA!r}, found {schema!r}"
        )
    benchmark_id = _required_text(data.get("benchmark_id"), "benchmark_id")
    version = _required_text(data.get("version"), "version")
    if expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
        raise BenchmarkManifestError(
            f"benchmark_id {benchmark_id!r} does not match expected "
            f"{expected_benchmark_id!r}"
        )
    if expected_version is not None and version != expected_version:
        raise BenchmarkManifestError(
            f"version {version!r} does not match expected {expected_version!r}"
        )
    description = data.get("description", "")
    if not isinstance(description, str):
        raise BenchmarkManifestError("description must be text")
    tasks = _validate_tasks(data.get("tasks"))
    result = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "version": version,
        "description": description,
        "tasks": tasks,
    }
    if "title" in data:
        result["title"] = _required_text(data["title"], "title")
    return result


def _db_path(workflow_folderpath: str | Path) -> str:
    return state_paths.observability_db(str(workflow_folderpath))


def _read_rows(
    workflow_folderpath: str | Path, sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """A read of the catalog; empty when the DB or the marker is absent."""
    return control.rows(
        open_live_store(_db_path(workflow_folderpath)),
        sql,
        params,
        feature=control.FEATURE_BENCHMARK_CATALOG_V1,
    )


def _parse_manifest(
    manifest_bytes: bytes,
    benchmark_id: str,
    version: str,
) -> dict[str, Any]:
    """Validate the stored canonical bytes; the digest is theirs, not a re-hash."""
    try:
        raw = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(
            f"benchmark version is not valid JSON: {exc}"
        ) from exc
    manifest = _validate_manifest(
        raw,
        expected_benchmark_id=benchmark_id,
        expected_version=version,
    )
    result = dict(manifest)
    result["digest_sha256"] = _sha256_bytes(manifest_bytes)
    return result


def archived_benchmark_ids(workflow_folderpath: str | Path) -> set[str]:
    """Benchmark ids hidden from the working lists. Empty when none are archived."""

    store = open_live_store(_db_path(workflow_folderpath))
    return {
        str(row["benchmark_id"])
        for row in control.rows(
            store,
            "SELECT benchmark_id FROM benchmark_archived",
            feature=control.FEATURE_BENCHMARK_ARCHIVE_V1,
        )
    }


def set_benchmark_archived(
    workflow_folderpath: str | Path, benchmark_id: str, archived: bool
) -> None:
    """Hide or show one benchmark. Does not seal or unseal its experiments."""

    benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    if benchmark_id not in list_benchmarks(workflow_folderpath):
        raise BenchmarkManifestError(f"benchmark not found: {benchmark_id}")
    store = open_live_store_for_write(_db_path(workflow_folderpath))
    now = datetime.now(timezone.utc).isoformat()
    with control.write(store) as conn:
        if archived:
            conn.execute(
                """INSERT INTO benchmark_archived (benchmark_id, archived_at)
                   VALUES (?, ?) ON CONFLICT(benchmark_id) DO UPDATE SET
                     archived_at=excluded.archived_at""",
                (benchmark_id, now),
            )
        else:
            conn.execute(
                "DELETE FROM benchmark_archived WHERE benchmark_id=?",
                (benchmark_id,),
            )


def list_benchmarks(workflow_folderpath: str | Path) -> list[str]:
    """Return the benchmark ids that have at least one stored version."""

    rows = _read_rows(
        workflow_folderpath,
        "SELECT DISTINCT benchmark_id FROM benchmark_versions ORDER BY benchmark_id",
    )
    return [row["benchmark_id"] for row in rows]


def list_versions(workflow_folderpath: str | Path, benchmark_id: str) -> list[str]:
    """Return the stored version ids of one benchmark, naturally sorted."""

    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    rows = _read_rows(
        workflow_folderpath,
        "SELECT version FROM benchmark_versions WHERE benchmark_id=?",
        (safe_benchmark_id,),
    )
    return sorted((row["version"] for row in rows), key=_natural_sort_key)


def load_version(
    workflow_folderpath: str | Path,
    benchmark_id: str,
    version: str,
) -> dict[str, Any]:
    """Load and validate one immutable benchmark version."""

    safe_benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    safe_version = _safe_segment(version, "version")
    rows = _read_rows(
        workflow_folderpath,
        "SELECT manifest_bytes FROM benchmark_versions "
        "WHERE benchmark_id=? AND version=?",
        (safe_benchmark_id, safe_version),
    )
    if not rows:
        raise BenchmarkManifestError(
            f"benchmark version not found: {safe_benchmark_id}/{safe_version}"
        )
    manifest = _parse_manifest(
        bytes(rows[0]["manifest_bytes"]), safe_benchmark_id, safe_version
    )
    return _with_details(workflow_folderpath, manifest)


def _with_details(workflow_folderpath: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest with its benchmark's current title and its version's description."""

    feature = control.FEATURE_BENCHMARK_DETAILS_V1
    title = control.rows(
        open_live_store(_db_path(workflow_folderpath)),
        "SELECT title FROM benchmark_titles WHERE benchmark_id=?",
        (manifest["benchmark_id"],), feature=feature,
    )
    description = control.rows(
        open_live_store(_db_path(workflow_folderpath)),
        "SELECT description FROM benchmark_version_descriptions "
        "WHERE benchmark_id=? AND version=?",
        (manifest["benchmark_id"], manifest["version"]), feature=feature,
    )
    if title:
        manifest["title"] = title[0]["title"]
    if description:
        manifest["description"] = description[0]["description"]
    return manifest


def set_details(
    workflow_folderpath: str | Path,
    benchmark_id: str,
    *,
    title: str | None = None,
    version: str | None = None,
    description: str | None = None,
) -> None:
    """Set the benchmark's title and/or one version's description (not frozen)."""

    benchmark_id = _safe_segment(benchmark_id, "benchmark_id")
    if title is not None:
        title = _required_text(title, "title")
    if description is not None:
        if not isinstance(description, str):
            raise BenchmarkManifestError("description must be text")
        version = _safe_segment(version, "version")
    now = datetime.now(timezone.utc).isoformat()
    store = open_live_store_for_write(_db_path(workflow_folderpath))
    with control.write(store) as conn:
        if title is not None:
            conn.execute(
                """INSERT INTO benchmark_titles (benchmark_id, title, updated_at)
                   VALUES (?, ?, ?) ON CONFLICT(benchmark_id) DO UPDATE SET
                     title=excluded.title, updated_at=excluded.updated_at""",
                (benchmark_id, title, now),
            )
        if description is not None:
            conn.execute(
                """INSERT INTO benchmark_version_descriptions
                     (benchmark_id, version, description, updated_at)
                   VALUES (?, ?, ?, ?) ON CONFLICT(benchmark_id, version) DO UPDATE SET
                     description=excluded.description, updated_at=excluded.updated_at""",
                (benchmark_id, version, description, now),
            )


def _spec_manifest(spec: Any) -> tuple[str, str, dict[str, Any]]:
    """Validate a version spec into its benchmark id, version and manifest."""

    if not isinstance(spec, dict):
        raise BenchmarkManifestError("spec must be a JSON object")
    benchmark_id = _safe_segment(spec.get("benchmark_id"), "benchmark_id")
    version = _safe_segment(spec.get("version"), "version")
    manifest = _validate_manifest(
        {
            "schema": SCHEMA,
            "benchmark_id": benchmark_id,
            "version": version,
            "description": spec.get("description"),
            "tasks": spec.get("tasks"),
            **({"title": spec["title"]} if "title" in spec else {}),
        },
        expected_benchmark_id=benchmark_id,
        expected_version=version,
    )
    return benchmark_id, version, manifest


def write_version(workflow_folderpath: str | Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Store a new immutable benchmark version in the live DB (created if absent)."""

    benchmark_id, version, manifest = _spec_manifest(spec)
    manifest_bytes = _canonical_json(manifest)
    digest = _sha256_bytes(manifest_bytes)
    store = open_live_store_for_write(_db_path(workflow_folderpath))
    try:
        with control.write(store) as conn:
            conn.execute(
                """INSERT INTO benchmark_versions
                   (benchmark_id, version, manifest_bytes, digest_sha256,
                    title, task_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    benchmark_id, version, manifest_bytes, digest,
                    manifest.get("title"), len(manifest["tasks"]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise BenchmarkAlreadyExistsError(
            f"benchmark version already exists and is immutable: {benchmark_id}/{version}"
        ) from exc
    result = dict(manifest)
    result["digest_sha256"] = digest
    return result


def replace_draft_version(
    workflow_folderpath: str | Path, spec: dict[str, Any], expected_digest: str
) -> dict[str, Any]:
    """Rewrite a draft (a stored version with no tasks) in place.

    The row is replaced only while it is still the draft the caller read
    (`expected_digest`); anything else raises `BenchmarkAlreadyExistsError`.
    """

    benchmark_id, version, manifest = _spec_manifest(spec)
    manifest_bytes = _canonical_json(manifest)
    digest = _sha256_bytes(manifest_bytes)
    store = open_live_store_for_write(_db_path(workflow_folderpath))
    try:
        with control.write(store) as conn:
            changed = conn.execute(
                """UPDATE benchmark_versions
                   SET manifest_bytes=?, digest_sha256=?, title=?, task_count=?
                   WHERE benchmark_id=? AND version=? AND task_count=0
                     AND digest_sha256=?""",
                (
                    manifest_bytes, digest, manifest.get("title"),
                    len(manifest["tasks"]), benchmark_id, version, expected_digest,
                ),
            ).rowcount
    except sqlite3.IntegrityError as exc:
        raise BenchmarkAlreadyExistsError(
            f"another benchmark version has these exact contents: {benchmark_id}/{version}"
        ) from exc
    if changed != 1:
        raise BenchmarkAlreadyExistsError(
            f"benchmark version is no longer the draft that was read: {benchmark_id}/{version}"
        )
    result = dict(manifest)
    result["digest_sha256"] = digest
    return result
