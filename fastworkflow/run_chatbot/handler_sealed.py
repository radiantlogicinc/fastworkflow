"""Package-private mixin: seal, browse and delete benchmark experiments (fix-ajir.24).

A development surface over functions that already exist. Nothing here runs by
itself: every seal, release and delete is one click by a person, and no route
prunes, seals or releases anything on a schedule.

- `GET /api/sealed-experiments` lists the sealed archives, the experiments
  ready to seal, the ones that cannot be sealed (still running, awaiting
  evaluation, or invalid), and the released ones.
- `POST /api/sealed-experiments` `{"experiment_id": ...}` seals one complete
  experiment with `ExperimentController.seal_workspace_evidence` into the
  workflow's `archives/` folder, then frees its turn-level evidence from the
  live DB (`drop_sealed_live_evidence`): the archive is the record from then
  on, and every read of that experiment's turns goes to it.
- `DELETE /api/sealed-experiments/<id>` removes a sealed experiment, archive
  file included (`delete_sealed_experiment`); refused while it is a contest's
  winner or a task's best run.
- `POST /api/evidence-releases` `{"experiment_id": ...}` records that an
  experiment will never be sealed: one still running is first closed as
  invalid (reason `operator`), then `release_experiment_evidence` records the
  release. Removing it afterwards is delete, below.
- `DELETE /api/released-experiments/<id>` removes a released experiment and
  every row naming it (`delete_released_experiment`). Refused while it is
  sealed, while it has not been released, or while it is a contest's winner
  or a task's best run.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import unquote

from fastworkflow import state_paths
from fastworkflow.observability import control
from fastworkflow.observability.archive_files import (
    archive_file_for,
    archives_directory,
    list_archived_experiments,
)
from fastworkflow.observability.store import (
    AttemptValueConflict,
    ExperimentNotFound,
    ExperimentNotSealed,
    IncompatibleObservabilityDB,
    SealedExperimentSelected,
    StoreIdentityMismatch,
    WriterStillOpen,
    open_live_store,
)
from fastworkflow.storage_keys import encode_path_component

SEALED_PREFIX = "/api/archived-experiments"
RELEASES_PATH = "/api/evidence-releases"
RELEASED_PREFIX = "/api/released-experiments"
_SEALABLE = ("complete", "capture_complete")
_UI_ACTOR = "observability UI"


def sealed_delete_path(path: str) -> bool:
    return path.startswith(SEALED_PREFIX + "/") and len(path) > len(SEALED_PREFIX) + 1


def released_delete_path(path: str) -> bool:
    return path.startswith(RELEASED_PREFIX + "/") and len(path) > len(RELEASED_PREFIX) + 1


def _all_experiments(store: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        batch = store.list_experiments(limit=200, offset=len(rows))
        rows.extend(batch)
        if len(batch) < 200:
            return rows


def seal_experiment_evidence(workflow_path: str, store: Any, experiment_id: str) -> dict[str, Any]:
    """Seal one complete experiment and free its live evidence.

    The same work as ``POST /api/sealed-experiments``. Returns the archive,
    what was freed, and a warning when the seal stands but the live rows
    could not be dropped.
    """
    from fastworkflow.experiment.runner import ExperimentController

    # `external=False`: this process is not driving the run. The seal still
    # refuses while THIS process holds a writer on the DB; the chatbot opens
    # none, and another process's writer is no obstacle. The copy is one read
    # transaction, and a closed experiment's late writes are refused by
    # claim-epoch fencing (single live DB design §3).
    controller = ExperimentController(
        workflow_path, store.store_identity(), migrate=False, external=False
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = os.path.join(
        state_paths.archives_dir(workflow_path),
        f"{encode_path_component(experiment_id)}-{stamp}.sqlite3",
    )
    archive = controller.seal_workspace_evidence(experiment_id, destination)
    freed, warning = None, None
    try:
        freed = controller.store.drop_sealed_live_evidence(experiment_id)
    except (ExperimentNotSealed, sqlite3.Error, OSError) as exc:
        # The seal stands; its live rows wait for prune instead.
        warning = f"sealed, but its live evidence was not freed: {exc}"
    return {"archive": archive, "freed": freed, "warning": warning}


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "experiment_id", "description", "status", "invalid_reason",
            "benchmark_id", "benchmark_version", "created_at", "completed_at",
        )
    }


class _SealedRoutes:
    def _sealed_archive_summary(self, store: Any, experiment_id: str) -> Optional[dict[str, Any]]:
        """The experiment's archive for its detail page, or None when unsealed."""
        archive = store.sealed_archive(experiment_id) if store is not None else None
        if archive is not None:
            shown = {key: archive[key] for key in (
                "path", "size_bytes", "archive_sha256", "readable")}
            shown["archived_at"] = archive["sealed_at"]
            return shown
        db_path = getattr(store, "db_path", None) if store is not None else None
        if not db_path:
            return None
        path = archive_file_for(
            os.path.join(os.path.dirname(db_path), "archives"), experiment_id
        )
        if path is None:
            return None
        stat = os.stat(path)
        return {
            "path": path,
            "size_bytes": stat.st_size,
            "readable": True,
            "archived_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
        }

    def _handle_sealed_list(self) -> None:
        """`GET /api/sealed-experiments`: what is sealed, ready, and not sealable."""
        try:
            store = self.chatbot.open_store()
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        if store is None:
            self._send_json({"archived": [], "ready": [], "unarchivable": [], "released": []})
            return
        experiments = {row["experiment_id"]: row for row in _all_experiments(store)}
        released = {
            row["experiment_id"]
            for row in control.rows(store, "SELECT experiment_id FROM evidence_releases")
        }
        selected: dict[str, str] = {}
        for row in control.rows(
            store, "SELECT experiment_id, scope_kind FROM selection_pointers"
        ):
            # A winner outranks a best run in the refusal the page warns of.
            if selected.get(row["experiment_id"]) != "experiment":
                selected[row["experiment_id"]] = row["scope_kind"]
        workflow_path = (self.chatbot.workflow_path or "").strip()
        started = time.perf_counter()
        scanned = (
            list_archived_experiments(archives_directory(workflow_path))
            if workflow_path else []
        )
        scan_seconds = time.perf_counter() - started
        sealed = []
        for archive in scanned:
            experiment_id = archive["experiment_id"]
            row = _summary(experiments.get(experiment_id) or archive)
            row.update(
                path=archive["path"],
                size_bytes=archive["size_bytes"],
                archived_at=archive["archived_at"],
                readable=archive["readable"],
                selected_as=selected.get(experiment_id),
            )
            sealed.append(row)
        sealed_ids = {row["experiment_id"] for row in sealed}
        ready, unsealable, released_rows = [], [], []
        for experiment_id, row in experiments.items():
            if experiment_id in sealed_ids:
                continue
            summary = _summary(row)
            if row.get("status") in _SEALABLE:
                ready.append(summary)
            elif experiment_id in released:
                summary["selected_as"] = selected.get(experiment_id)
                released_rows.append(summary)
            else:
                unsealable.append(summary)
        self._send_json({
            "archived": sealed,
            "ready": ready,
            "unarchivable": unsealable,
            "released": released_rows,
            "scan_seconds": scan_seconds,
        })

    def _post_seal(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        """`POST /api/sealed-experiments`: seal one experiment, then free its live rows."""
        from fastworkflow.experiment.runner import MissingExperimentLifecycleFeature

        workflow_path = (self.chatbot.workflow_path or "").strip()
        experiment_id = body.get("experiment_id")
        if not workflow_path:
            self._error(409, "select a workflow before archiving an experiment")
            return
        if not isinstance(experiment_id, str) or not experiment_id:
            self._error(400, "experiment_id is required")
            return
        try:
            store = self.chatbot.open_store()
            if store is None:
                self._error(404, "observability DB not found")
                return
            sealed = seal_experiment_evidence(workflow_path, store, experiment_id)
        except ExperimentNotFound:
            self._error(404, "experiment not found")
            return
        except (
            ValueError,  # not complete, already sealed, invalid, identity mismatch
            WriterStillOpen,
            MissingExperimentLifecycleFeature,
            IncompatibleObservabilityDB,
            FileExistsError,
        ) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json(
            {
                "archive": sealed["archive"],
                "freed": sealed["freed"],
                "warning": sealed["warning"],
            },
            status=201,
        )

    def _delete_sealed(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        """`DELETE /api/sealed-experiments/<id>`: the archive file and its rows."""
        experiment_id = unquote(path[len(SEALED_PREFIX) + 1:])
        try:
            store = open_live_store(self.chatbot.db_path or "", write=True)
            result = store.delete_sealed_experiment(experiment_id)
        except ExperimentNotSealed:
            self._error(404, "no archived experiment with that id")
            return
        except SealedExperimentSelected as exc:
            self._error(409, str(exc))
            return
        except (control.ControlUnavailable, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json(result)

    def _post_release(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        """`POST /api/evidence-releases`: let prune reach evidence never to be sealed."""
        experiment_id = body.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            self._error(400, "experiment_id is required")
            return
        try:
            store = open_live_store(self.chatbot.db_path or "", write=True)
            experiment = store.get_experiment(experiment_id)
            if experiment is None:
                raise ExperimentNotFound(experiment_id)
            status = experiment["status"]
            if status in _SEALABLE:
                self._error(409, f"experiment is {status}; seal it instead of releasing it")
                return
            if status != "invalid":
                status = store.complete_experiment(
                    experiment_id,
                    force_invalid="operator",
                    detail="closed from the chatbot so its evidence could be released",
                )
            store.release_experiment_evidence(
                experiment_id, _UI_ACTOR, "released from the chatbot"
            )
        except ExperimentNotFound:
            self._error(404, "experiment not found")
            return
        except (
            control.ControlUnavailable,
            IncompatibleObservabilityDB,
            AttemptValueConflict,
            StoreIdentityMismatch,
            ValueError,
        ) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json({"experiment_id": experiment_id, "status": status, "released": True})

    def _delete_released(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        """`DELETE /api/released-experiments/<id>`: every row naming a released experiment."""
        experiment_id = unquote(path[len(RELEASED_PREFIX) + 1:])
        try:
            store = open_live_store(self.chatbot.db_path or "", write=True)
            result = store.delete_released_experiment(experiment_id)
        except ExperimentNotFound:
            self._error(404, "experiment not found")
            return
        except SealedExperimentSelected as exc:
            self._error(409, str(exc))
            return
        except ValueError as exc:
            self._error(409, str(exc))
            return
        except (control.ControlUnavailable, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json(result)
