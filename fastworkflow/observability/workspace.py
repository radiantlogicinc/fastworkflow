"""Read-only, manifest-bound access to one or more observability stores.

Workspace manifests make experiment evidence portable without turning a
collection of SQLite files into one implicit database. Every read names its
``store_id``; turn keys and trace ids are never searched across stores.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Iterable, Iterator, Mapping, Optional

from fastworkflow.observability.store import (
    IncompatibleObservabilityDB,
    ReadOnlyObservabilityStore,
)

WORKSPACE_SCHEMA = "fastworkflow-observability-workspace/1"

# How many turn rows one page of an attempt's keys asks for. A page size, not
# a cap: the enumeration below keeps asking until a short page says the
# attempt is exhausted.
_TURN_PAGE = 1_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkspaceError(RuntimeError):
    """Base class for manifest and workspace read failures."""


class WorkspaceManifestError(WorkspaceError, ValueError):
    """The manifest does not satisfy the v1 workspace contract."""


class WorkspaceIntegrityError(WorkspaceError):
    """A sealed store no longer matches the evidence declared by its manifest."""


class UnknownWorkspaceStore(WorkspaceError, KeyError):
    """A read named no registered store."""


class UnknownLogicalExperiment(WorkspaceError, KeyError):
    """A read named no logical experiment in the manifest."""


class WorkspaceBusyError(WorkspaceError):
    """The bounded request-handle budget was exhausted."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_workflow_folderpath(value: Any) -> Optional[str]:
    """The workflow folder a sealed run named, or None.

    Optional and absolute. Optional because every manifest written before this
    field existed is still valid — the exp029 trial's is one — and a workspace
    that cannot say which workflow it came from must still open. Absolute
    because, unlike ``stores[].path``, this is NOT resolved against the
    manifest's own folder: an archive is copied around, and a relative folder
    reference would silently name a different tree in each copy.

    A named folder that is not there is not a manifest error. Evidence outlives
    the workflow checkout it was produced from; the field is kept as recorded
    and the catalogue is reported unavailable, which is a fact about this
    machine, not a defect in the archive.
    """
    if value is None:
        return None
    raw = _required_text(value, "workflow_folderpath")
    if not (Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()):
        raise WorkspaceManifestError(
            f"workflow_folderpath must be an absolute path, found {raw!r}"
        )
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_store_path(root: Path, value: Any, field: str) -> Path:
    raw = _required_text(value, field)
    candidate = Path(raw)
    if (
        candidate.is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ".." in PurePath(raw.replace("\\", "/")).parts
    ):
        raise WorkspaceManifestError(
            f"{field} must be relative and must not contain '..'"
        )
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except FileNotFoundError as exc:
        raise WorkspaceManifestError(f"{field} does not exist: {raw!r}") from exc
    except ValueError as exc:
        raise WorkspaceManifestError(
            f"{field} resolves outside the workspace root"
        ) from exc
    if not resolved.is_file():
        raise WorkspaceManifestError(f"{field} is not a file: {raw!r}")
    return resolved


def _store_mode(value: dict[str, Any], field: str) -> str:
    explicit = value.get(
        "mode", value.get("state", value.get("kind", value.get("status")))
    )
    if explicit is None and "sealed" in value:
        explicit = "sealed" if value["sealed"] is True else "live"
    if explicit == "live":
        raise WorkspaceManifestError(
            f"{field} is a live store; a workspace reads sealed archives only"
        )
    if explicit != "sealed":
        raise WorkspaceManifestError(f"{field}.mode must be 'sealed'")
    return str(explicit)


def _declared_digest(value: dict[str, Any], field: str) -> Optional[str]:
    declared: Any = value.get("sha256")
    if declared is None:
        declared = value.get("digest")
        if isinstance(declared, dict):
            if declared.get("algorithm", "sha256") != "sha256":
                raise WorkspaceManifestError(
                    f"{field}.digest.algorithm must be 'sha256'"
                )
            declared = declared.get("value", declared.get("sha256"))
    if declared is None:
        return None
    if not isinstance(declared, str) or not _SHA256_RE.fullmatch(declared.lower()):
        raise WorkspaceManifestError(f"{field}.sha256 must be 64 hexadecimal digits")
    return declared.lower()


@dataclass(frozen=True)
class WorkspaceStore:
    """One validated store declaration."""

    store_id: str
    label: str
    path: Path
    relative_path: str
    mode: str
    sha256: Optional[str]
    store_identity: Optional[str]


class _WorkspaceReadOnlyStore(ReadOnlyObservabilityStore):
    """Read-only store using an escaped URI; sealed archives are immutable."""

    # Overrides the raw opener, not `_connect`: the base class builds every
    # connection (construction probes and closing `with` blocks) from it.
    def _open_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"{Path(self.db_path).resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn


class ReadOnlyWorkspaceStoreRegistry:
    """Validated store ids plus a bounded request-scoped handle budget.

    A lease constructs a fresh store facade and releases it deterministically.
    The facade itself retains no connection: inherited read methods open and
    close one SQLite connection per call.
    """

    def __init__(self, stores: list[WorkspaceStore], max_open_handles: int = 8) -> None:
        if max_open_handles < 1:
            raise ValueError("max_open_handles must be positive")
        self._stores = {store.store_id: store for store in stores}
        self._handles = threading.BoundedSemaphore(max_open_handles)

    def descriptor(self, store_id: str) -> WorkspaceStore:
        try:
            return self._stores[store_id]
        except KeyError as exc:
            raise UnknownWorkspaceStore(f"unknown store_id {store_id!r}") from exc

    def store_id_for_identity(self, identity: str) -> str:
        """The manifest store whose DECLARED identity is `identity`.

        An `ExecutionRef` names its store by `store_identity()`; the manifest
        names the same store by `store_id`. A comparison comment whose two
        sides live in different archives needs the one translated into the
        other, and the manifest is the only place this build is willing to
        read it from: the declaration is already there, `_verify_store`
        already checks the archive still matches it, and answering from the
        declaration searches no directory and opens no file the manifest did
        not name.

        A store the manifest declares no identity for is NOT resolved by
        opening it to look: the declaration is the authorization, and an
        undeclared store raises rather than being adopted. Two declarations of
        the same identity are refused for the same reason -- the right answer
        is to fix the manifest, not to pick one.
        """
        matches = [
            store.store_id
            for store in self._stores.values()
            if store.store_identity is not None and store.store_identity == identity
        ]
        if not matches:
            raise UnknownWorkspaceStore(
                f"no workspace store declares evidence identity {identity!r}"
            )
        if len(matches) > 1:
            raise UnknownWorkspaceStore(
                f"evidence identity {identity!r} is declared by more than one "
                f"workspace store ({', '.join(sorted(matches))})"
            )
        return matches[0]

    @contextmanager
    def open(self, store_id: str) -> Iterator[ReadOnlyObservabilityStore]:
        descriptor = self.descriptor(store_id)
        if not self._handles.acquire(timeout=5.0):
            raise WorkspaceBusyError("read-only workspace handle limit reached")
        try:
            _verify_store(descriptor)
            yield _WorkspaceReadOnlyStore(str(descriptor.path))
        finally:
            self._handles.release()


def _verify_store(store: WorkspaceStore) -> dict[str, Any]:
    sidecars = [
        str(Path(f"{store.path}{suffix}"))
        for suffix in ("-wal", "-shm")
        if Path(f"{store.path}{suffix}").exists()
    ]
    actual_digest = _sha256(store.path)
    if sidecars:
        raise WorkspaceIntegrityError(
            f"sealed store {store.store_id!r} has SQLite sidecar(s): "
            + ", ".join(sidecars)
        )
    if store.sha256 is None:
        raise WorkspaceIntegrityError(
            f"sealed store {store.store_id!r} has no declared digest"
        )
    if actual_digest != store.sha256:
        raise WorkspaceIntegrityError(
            f"sealed store {store.store_id!r} sha256 mismatch: "
            f"expected {store.sha256}, found {actual_digest}"
        )
    try:
        actual_identity = _WorkspaceReadOnlyStore(str(store.path)).store_identity()
    except (sqlite3.Error, IncompatibleObservabilityDB) as exc:
        raise WorkspaceIntegrityError(
            f"store {store.store_id!r} cannot be opened read-only: {exc}"
        ) from exc
    if store.store_identity is not None and actual_identity != store.store_identity:
        raise WorkspaceIntegrityError(
            f"store {store.store_id!r} identity mismatch: expected "
            f"{store.store_identity!r}, found {actual_identity!r}"
        )
    return {
        "store_id": store.store_id,
        "label": store.label,
        "path": store.relative_path,
        "mode": store.mode,
        "sealed": True,
        "integrity": "verified",
        "sha256": store.sha256,
        "observed_sha256": actual_digest,
        "store_identity": actual_identity,
        "sidecars": sidecars,
    }


def _segment_ref(segment: dict[str, Any], field: str) -> tuple[str, str]:
    local = segment.get("local")
    if local is not None and not isinstance(local, dict):
        raise WorkspaceManifestError(f"{field}.local must be an object")
    local = local or {}
    store_id = segment.get("store_id", local.get("store_id"))
    experiment_id = segment.get(
        "local_experiment_id",
        local.get("experiment_id", segment.get("experiment_id")),
    )
    return (
        _required_text(store_id, f"{field}.store_id"),
        _required_text(experiment_id, f"{field}.local_experiment_id"),
    )


class ObservabilityWorkspace:
    """Parsed v1 workspace manifest and read-only query API."""

    def __init__(
        self,
        manifest_path: Path,
        manifest: dict[str, Any],
        stores: list[WorkspaceStore],
        max_open_handles: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.workspace_id = _required_text(manifest.get("workspace_id"), "workspace_id")
        self.label = _required_text(manifest.get("label"), "label")
        self.workflow_folderpath = _optional_workflow_folderpath(
            manifest.get("workflow_folderpath")
        )
        self._manifest = manifest
        self._stores = stores
        self.registry = ReadOnlyWorkspaceStoreRegistry(stores, max_open_handles)
        self._experiments = self._validate_experiments(manifest.get("experiments"))
        self._projected_attempts = self._validate_projected_attempts(
            manifest.get("projected_attempts", [])
        )

    @classmethod
    def load(
        cls, manifest_path: str | Path, *, max_open_handles: int = 8
    ) -> "ObservabilityWorkspace":
        path = Path(manifest_path).resolve(strict=True)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkspaceManifestError(
                f"workspace manifest is not valid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise WorkspaceManifestError("workspace manifest must be a JSON object")
        schema = value.get("schema", value.get("schema_version"))
        if schema != WORKSPACE_SCHEMA:
            raise WorkspaceManifestError(
                f"schema must be {WORKSPACE_SCHEMA!r}, found {schema!r}"
            )
        raw_stores = value.get("stores")
        if not isinstance(raw_stores, list) or not raw_stores:
            raise WorkspaceManifestError("stores must be a non-empty array")
        stores: list[WorkspaceStore] = []
        known: set[str] = set()
        root = path.parent.resolve()
        for index, raw in enumerate(raw_stores):
            field = f"stores[{index}]"
            if not isinstance(raw, dict):
                raise WorkspaceManifestError(f"{field} must be an object")
            store_id = _required_text(raw.get("store_id"), f"{field}.store_id")
            if store_id in known:
                raise WorkspaceManifestError(f"duplicate store_id {store_id!r}")
            known.add(store_id)
            mode = _store_mode(raw, field)
            digest = _declared_digest(raw, field)
            if digest is None:
                raise WorkspaceManifestError(
                    f"{field}.sha256 is required for a sealed store"
                )
            relative_path = _required_text(raw.get("path"), f"{field}.path")
            stores.append(
                WorkspaceStore(
                    store_id=store_id,
                    label=str(raw.get("label") or store_id),
                    path=_safe_store_path(root, relative_path, f"{field}.path"),
                    relative_path=relative_path,
                    mode=mode,
                    sha256=digest,
                    store_identity=(
                        _required_text(raw["store_identity"], f"{field}.store_identity")
                        if raw.get("store_identity") is not None
                        else None
                    ),
                )
            )
        workspace = cls(path, value, stores, max_open_handles)
        # A sealed declaration is not presented as sealed until its bytes,
        # sidecars, schema, and optional durable identity have been checked.
        for store in stores:
            _verify_store(store)
        return workspace

    def _validate_experiments(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise WorkspaceManifestError("experiments must be an array")
        known_stores = {store.store_id for store in self._stores}
        known_experiments: set[str] = set()
        result: list[dict[str, Any]] = []
        for index, value in enumerate(raw):
            field = f"experiments[{index}]"
            if not isinstance(value, dict):
                raise WorkspaceManifestError(f"{field} must be an object")
            logical_id = _required_text(
                value.get("experiment_id", value.get("logical_experiment_id")),
                f"{field}.experiment_id",
            )
            if logical_id in known_experiments:
                raise WorkspaceManifestError(
                    f"duplicate logical experiment_id {logical_id!r}"
                )
            known_experiments.add(logical_id)
            segments = value.get("segments")
            if not isinstance(segments, list) or not segments:
                raise WorkspaceManifestError(
                    f"{field}.segments must be a non-empty array"
                )
            normalized = dict(value)
            normalized["experiment_id"] = logical_id
            normalized_segments = []
            for segment_index, segment in enumerate(segments):
                segment_field = f"{field}.segments[{segment_index}]"
                if not isinstance(segment, dict):
                    raise WorkspaceManifestError(f"{segment_field} must be an object")
                store_id, local_id = _segment_ref(segment, segment_field)
                if store_id not in known_stores:
                    raise WorkspaceManifestError(
                        f"{segment_field} references unknown store_id {store_id!r}"
                    )
                normalized_segment = dict(segment)
                normalized_segment["segment_id"] = str(
                    segment.get("segment_id")
                    or segment.get("logical_segment_id")
                    or f"{logical_id}:{segment_index + 1}"
                )
                normalized_segment["store_id"] = store_id
                normalized_segment["local_experiment_id"] = local_id
                normalized_segments.append(normalized_segment)
            normalized["segments"] = normalized_segments
            result.append(normalized)
        return result

    def _validate_projected_attempts(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise WorkspaceManifestError("projected_attempts must be an array")
        known_stores = {store.store_id for store in self._stores}
        result = []
        for index, value in enumerate(raw):
            field = f"projected_attempts[{index}]"
            if not isinstance(value, dict):
                raise WorkspaceManifestError(f"{field} must be an object")
            copied = dict(value)
            refs = self._attempt_refs(copied)
            for ref in refs:
                store_id = _required_text(ref.get("store_id"), f"{field}.store_id")
                if store_id not in known_stores:
                    raise WorkspaceManifestError(
                        f"{field} references unknown store_id {store_id!r}"
                    )
                turn_ref = ref.get("turn_ref")
                if turn_ref is not None:
                    if not isinstance(turn_ref, dict):
                        raise WorkspaceManifestError(
                            f"{field}.turn_ref must be an object"
                        )
                    turn_store_id = _required_text(
                        turn_ref.get("store_id"), f"{field}.turn_ref.store_id"
                    )
                    _required_text(
                        turn_ref.get("logical_turn_key"),
                        f"{field}.turn_ref.logical_turn_key",
                    )
                    if turn_store_id not in known_stores:
                        raise WorkspaceManifestError(
                            f"{field} references unknown store_id " f"{turn_store_id!r}"
                        )
            for turn_ref in self._turn_refs(copied):
                store_id = _required_text(
                    turn_ref.get("store_id"), f"{field}.turn_refs.store_id"
                )
                _required_text(
                    turn_ref.get("logical_turn_key"),
                    f"{field}.turn_refs.logical_turn_key",
                )
                if store_id not in known_stores:
                    raise WorkspaceManifestError(
                        f"{field} references unknown store_id {store_id!r}"
                    )
            result.append(copied)
        return result

    @staticmethod
    def _attempt_refs(projected: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("attempt_refs", "sources"):
            value = projected.get(key)
            if value is not None:
                if not isinstance(value, list) or not all(
                    isinstance(item, dict) for item in value
                ):
                    raise WorkspaceManifestError(f"{key} must be an array of objects")
                return list(value)
        for key in ("attempt_ref", "source"):
            value = projected.get(key)
            if value is not None:
                if not isinstance(value, dict):
                    raise WorkspaceManifestError(f"{key} must be an object")
                return [value]
        if "store_id" in projected:
            return [projected]
        return []

    @staticmethod
    def _turn_refs(projected: dict[str, Any]) -> list[dict[str, Any]]:
        value = projected.get("turn_refs", projected.get("turns", []))
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise WorkspaceManifestError("turn_refs must be an array of objects")
        return list(value)

    @staticmethod
    def _logical_attempt(projected: dict[str, Any]) -> dict[str, Any]:
        nested = projected.get("logical_attempt")
        if isinstance(nested, dict):
            return nested
        return {
            "experiment_id": projected.get(
                "logical_experiment_id", projected.get("experiment_id")
            ),
            "task_id": projected.get("logical_task_id", projected.get("task_id")),
            "attempt": (
                nested
                if nested is not None
                else projected.get("logical_attempt_number", projected.get("attempt"))
            ),
        }

    @property
    def benchmark_catalogue_state(self) -> str:
        """``not_declared`` | ``available`` | ``unavailable``.

        Three states, not two: a manifest that never named a workflow folder
        and one whose folder is missing from this machine are different facts,
        and the second is the one worth showing a reader who expected to see
        the corpus a run was pinned to.
        """
        if self.workflow_folderpath is None:
            return "not_declared"
        return (
            "available" if Path(self.workflow_folderpath).is_dir() else "unavailable"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema": WORKSPACE_SCHEMA,
            "workspace_id": self.workspace_id,
            "label": self.label,
            "manifest_path": str(self.manifest_path),
            "workflow_folderpath": self.workflow_folderpath,
            "benchmark_catalogue": self.benchmark_catalogue_state,
            "store_count": len(self._stores),
            "experiment_count": len(self._experiments),
            "projected_attempt_count": len(self._projected_attempts),
            "read_only": True,
        }

    def stores(self) -> list[dict[str, Any]]:
        return [_verify_store(store) for store in self._stores]

    def experiments(self) -> list[dict[str, Any]]:
        return [dict(value) for value in self._experiments]

    def segments(self, experiment_id: str) -> list[dict[str, Any]]:
        return [dict(value) for value in self._experiment(experiment_id)["segments"]]

    def _experiment(self, experiment_id: str) -> dict[str, Any]:
        for experiment in self._experiments:
            if experiment["experiment_id"] == experiment_id:
                return experiment
        raise UnknownLogicalExperiment(
            f"unknown logical experiment_id {experiment_id!r}"
        )

    def tasks(self, experiment_id: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for segment in self.segments(experiment_id):
            with self.registry.open(segment["store_id"]) as store:
                for task in store.experiment_tasks(segment["local_experiment_id"]):
                    row = dict(task)
                    row.update(
                        {
                            "store_id": segment["store_id"],
                            "segment_id": segment["segment_id"],
                            "local_experiment_id": segment["local_experiment_id"],
                        }
                    )
                    tasks.append(row)
        return tasks

    def attempts(
        self,
        experiment_id: str,
        *,
        task_id: Optional[str] = None,
        turn_refs_for: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Attempt rows, optionally with the turn keys each one recorded.

        `turn_refs_for` bounds the TURN reading, which is the expensive half:
        pass a set of attempt numbers to read turns for those alone, or an
        empty set for attempt metadata and no turn read at all. The default
        reads them for every attempt, which is what the existing callers
        expect. An attempt whose turns were not read carries no `turn_refs`
        key rather than an empty list, so "this run recorded nothing" and
        "nobody asked about this run" stay different answers.
        """
        attempts: list[dict[str, Any]] = []
        wanted = (
            None
            if turn_refs_for is None
            else {int(attempt) for attempt in turn_refs_for}
        )
        for segment in self.segments(experiment_id):
            with self.registry.open(segment["store_id"]) as store:
                local_id = segment["local_experiment_id"]
                for attempt in store.experiment_attempt_rows(local_id, task_id=task_id):
                    row = dict(attempt)
                    row.update(
                        {
                            "store_id": segment["store_id"],
                            "segment_id": segment["segment_id"],
                            "local_experiment_id": local_id,
                        }
                    )
                    if wanted is None or int(row["attempt"]) in wanted:
                        row["turn_refs"] = [
                            {
                                "store_id": segment["store_id"],
                                "logical_turn_key": key,
                            }
                            for key in self._attempt_turn_keys(
                                store, local_id, row["task_id"], int(row["attempt"])
                            )
                        ]
                    attempts.append(row)
        return attempts

    def attempt_turn_keys(
        self,
        *,
        store_id: str,
        local_experiment_id: str,
        task_id: str,
        attempt: int,
    ) -> list[str]:
        """The turn keys of ONE attempt, named by full identity.

        An attempt number alone does not identify a run across a workspace --
        two archives, or two segments of one, can each record an attempt 1 --
        so the archive and the local experiment are named rather than searched
        for.
        """
        with self.registry.open(store_id) as store:
            return self._attempt_turn_keys(
                store, local_experiment_id, task_id, attempt
            )

    @staticmethod
    def _attempt_turn_keys(
        store: Any, local_experiment_id: str, task_id: str, attempt: int
    ) -> list[str]:
        """Every turn key of one attempt, in this route's own order.

        Paged by KEYSET, because one bounded read stops at its bound without
        saying so: a run of more turns than the bound would have produced a
        reference describing a shorter run than the archive holds, and a
        summary over it would have been over part of a run while reporting a
        whole one. Each page continues the previous one in the same
        `turn_key DESC` order the single read returned, so what a caller sees
        is unchanged except that it is now complete.
        """
        keys: list[str] = []
        cursor: Optional[str] = None
        while True:
            page = store.list_turns(
                experiment_id=local_experiment_id,
                task_id=task_id,
                attempt=attempt,
                limit=_TURN_PAGE,
                before_turn_key=cursor,
            )
            keys.extend(str(turn["turn_key"]) for turn in page)
            if len(page) < _TURN_PAGE:
                return keys
            cursor = str(page[-1]["turn_key"])

    def turn(self, store_id: str, logical_turn_key: str) -> Optional[dict[str, Any]]:
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; turn keys are never searched across stores"
            )
        with self.registry.open(store_id) as store:
            turn = store.get_turn(logical_turn_key)
        if turn is None:
            return None
        result = dict(turn)
        result["store_id"] = store_id
        result["logical_turn_key"] = logical_turn_key
        try:
            result["record"] = json.loads(result.pop("record_json"))
        except (KeyError, ValueError, TypeError):
            result["record"] = None
        return result

    def evidence_runs(
        self, store_id: str, local_experiment_id: str
    ) -> list[dict[str, Any]]:
        """The evidence segments one store persisted for one local experiment.

        Exactly what `ObservabilityStore.get_experiment` returns under
        `evidence_runs` (the `experiment_evidence_runs` rows, each with its
        decoded record), scoped to the named store like every other read here;
        an experiment the store does not know answers `[]`. Exposed so the
        read-only UI can badge a logical experiment's segments and attempts with
        the verdicts the archives already hold (fix-49m.6), rather than
        deriving a verdict from anything else.
        """
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; experiments are never searched across stores"
            )
        with self.registry.open(store_id) as store:
            experiment = store.get_experiment(local_experiment_id)
        if not experiment:
            return []
        return [dict(segment) for segment in experiment.get("evidence_runs") or []]

    def experiment(
        self, store_id: str, local_experiment_id: str
    ) -> Optional[dict[str, Any]]:
        """One store's experiment row with its evidence segments, or None.

        Exactly `ObservabilityStore.get_experiment`, scoped to the named store
        like every other read here. Exposed so the read-only UI can flatten a
        segment's provenance (fix-aou) from the columns the archive holds --
        capture regime, benchmark pin, workflow -- next to the evidence-run
        records `evidence_runs` already returns.
        """
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; experiments are never searched across stores"
            )
        with self.registry.open(store_id) as store:
            experiment = store.get_experiment(local_experiment_id)
        return dict(experiment) if experiment else None

    def attempts_in_store(
        self, store_id: str, local_experiment_id: str
    ) -> list[dict[str, Any]]:
        """One store's attempt rows for one local experiment, snapshots
        decoded, without the turn refs `attempts()` resolves per segment."""
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; experiments are never searched across stores"
            )
        with self.registry.open(store_id) as store:
            rows = store.experiment_attempt_rows(local_experiment_id)
        return [dict(row) for row in rows]

    def _scope_spans(
        self, store_id: str, logical_turn_key: str, spans: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """One turn's raw span rows, stamped with their workspace scope and
        with `attributes` decoded. Shared by `trace` and `traces` so the single
        and bulk readers cannot drift."""
        result = []
        for value in spans:
            span = dict(value)
            span["store_id"] = store_id
            span["logical_turn_key"] = logical_turn_key
            try:
                span["attributes"] = json.loads(span["attributes"])
            except (KeyError, ValueError, TypeError):
                pass
            result.append(span)
        return result

    def trace(self, store_id: str, logical_turn_key: str) -> list[dict[str, Any]]:
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; trace ids are never searched across stores"
            )
        with self.registry.open(store_id) as store:
            spans = store.get_spans(logical_turn_key)
        return self._scope_spans(store_id, logical_turn_key, spans)

    def traces(
        self, store_id: str, logical_turn_keys: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """`trace` for many turns of one store: ``{logical_turn_key: spans}``.

        The archived twin of `ObservabilityStore.spans_for_turns` (fix-tk5).
        Attempt and projected-attempt rows stamp every turn ref they carry;
        going through `trace` opened the store and issued one query per ref.
        This opens it once and reads the refs in chunks, answering rows in the
        same shape and order `trace` would. Every requested key is present in
        the answer, mapping to ``[]`` when the store holds no spans for it.
        """
        if not store_id:
            raise UnknownWorkspaceStore(
                "store_id is required; trace ids are never searched across stores"
            )
        keys = list(dict.fromkeys(key for key in logical_turn_keys if key))
        if not keys:
            return {}
        with self.registry.open(store_id) as store:
            spans_by_turn = store.spans_for_turns(keys)
        return {
            key: self._scope_spans(store_id, key, spans_by_turn.get(key) or [])
            for key in keys
        }

    def projected_attempts(
        self,
        *,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        resolved = []
        for projected in self._projected_attempts:
            logical = self._logical_attempt(projected)
            if (
                experiment_id is not None
                and logical.get("experiment_id") != experiment_id
            ):
                continue
            if task_id is not None and logical.get("task_id") != task_id:
                continue
            if attempt is not None and int(logical.get("attempt", -1)) != int(attempt):
                continue
            item = dict(projected)
            sources = []
            for ref in self._attempt_refs(projected):
                source = dict(ref)
                local_experiment_id = source.get(
                    "local_experiment_id", source.get("experiment_id")
                )
                local_task_id = source.get("local_task_id", source.get("task_id"))
                local_attempt = source.get("local_attempt", source.get("attempt"))
                if (
                    local_experiment_id is not None
                    and local_task_id is not None
                    and local_attempt is not None
                ):
                    with self.registry.open(str(source["store_id"])) as store:
                        matches = [
                            row
                            for row in store.experiment_attempt_rows(
                                str(local_experiment_id),
                                task_id=str(local_task_id),
                            )
                            if int(row["attempt"]) == int(local_attempt)
                        ]
                    source["resolved_attempt"] = matches[0] if matches else None
                turn_ref = source.get("turn_ref")
                if isinstance(turn_ref, dict):
                    source["resolved_turn"] = self.turn(
                        _required_text(turn_ref.get("store_id"), "turn_ref.store_id"),
                        _required_text(
                            turn_ref.get("logical_turn_key"),
                            "turn_ref.logical_turn_key",
                        ),
                    )
                sources.append(source)
            item["resolved_sources"] = sources
            item["resolved_turns"] = [
                self.turn(
                    _required_text(turn_ref.get("store_id"), "turn_ref.store_id"),
                    _required_text(
                        turn_ref.get("logical_turn_key"),
                        "turn_ref.logical_turn_key",
                    ),
                )
                for turn_ref in self._turn_refs(projected)
            ]
            resolved.append(item)
        return resolved


def load_observability_workspace(
    manifest_path: str | Path, *, max_open_handles: int = 8
) -> ObservabilityWorkspace:
    """Load and fully validate a read-only observability workspace."""

    return ObservabilityWorkspace.load(manifest_path, max_open_handles=max_open_handles)
