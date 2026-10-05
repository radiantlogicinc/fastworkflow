"""Package-private mixin of chatbot benchmark routes, moved verbatim from server.py."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Optional
from urllib.parse import unquote

from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.benchmark.catalog import (
    BenchmarkAlreadyExistsError,
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_analysis,
    load_version,
    write_analysis,
    write_version,
)
from fastworkflow.observability.store import IncompatibleObservabilityDB
from fastworkflow.run_chatbot.http_common import STORE_UNAVAILABLE


class _BenchmarkRoutes:
    def _post_benchmark_record(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        folder = self._benchmark_workflow_path(write=True)
        if folder is None:
            return
        benchmark_setup_path = path == "/api/benchmark-setup"
        try:
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            if benchmark_setup_path:
                self._send_json({"version": benchmark_setup.save_benchmark(folder, body)}, status=201)
            else:
                benchmark_id = unquote(path[len("/api/benchmarks/"):-len("/experiments")])
                # `runs_per_task` is the existing declared-attempt
                # count surfaced at setup; omitting it still means 1,
                # and n > 1 asks for no target, rubric or review gate.
                record = benchmark_setup.create_experiment(
                    folder, benchmark_id, body.get("version"),
                    body.get("description", ""),
                    runs_per_task=body.get("runs_per_task", 1),
                )
                self._send_json({"experiment": record}, status=201)
        except benchmark_setup.BenchmarkSetupConflict as exc:
            self._error(409, str(exc))
        except (BenchmarkManifestError, ValueError, TypeError) as exc:
            self._error(400, str(exc))

    def _post_benchmark_version(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_benchmark_post(path, body)

    def _put_benchmark_analysis(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_benchmark_analysis_put(path, body)

    def _delete_benchmark_experiment(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        folder = self._benchmark_workflow_path(write=True)
        if folder is None:
            return
        prefix = "/api/benchmark-experiments/"
        experiment_id = unquote(path[len(prefix):])
        try:
            record = benchmark_setup.delete_empty_experiment(folder, experiment_id)
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except benchmark_setup.BenchmarkSetupConflict as exc:
            self._error(409, str(exc))
            return
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"deleted": experiment_id, "benchmark_id": record["benchmark_id"]})

    def _patch_registration(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_registration_patch(path, body)

    def _benchmark_workflow_path(self, *, write: bool = False) -> Optional[str]:
        """Workflow folder for versioned benchmark corpus files.

        In workspace mode this is the folder the manifest named at seal time,
        and it is served for READS only: the corpus a sealed run was pinned to
        is part of reading that run's evidence, and refusing it left the pin as
        a digest with nothing behind it. Writes stay refused exactly as before
        — the folder is a live checkout that a read-only workspace must not
        touch, and `write=True` returns None before the manifest is consulted.

        A manifest with no folder keeps its 409, and so does one whose folder
        is gone: nothing was found to read, and the reason is quoted.
        """
        if self.chatbot.workspace is not None:
            if write:
                self._error(
                    403,
                    "workspace mode is read-only; benchmark corpus files cannot be changed",
                )
                return None
            declared = self.chatbot.workspace.workflow_folderpath
            if not declared:
                self._error(
                    409,
                    "benchmarks are available in live workflow mode only",
                )
                return None
            if not os.path.isdir(declared):
                self._error(
                    409,
                    "the workflow folder named by this workspace is not on "
                    f"this machine: {declared}",
                )
                return None
            return declared
        workflow_path = (self.chatbot.workflow_path or "").strip()
        if not workflow_path:
            self._error(
                409,
                "select a workflow before using benchmarks",
            )
            return None
        return workflow_path

    def _handle_benchmark_registration(self, experiment_id):
        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        try:
            record, manifest = benchmark_setup.experiment_manifest(folder, experiment_id)
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except (ValueError, BenchmarkManifestError) as exc:
            self._error(409, str(exc))
            return
        recorded, warning = False, None
        if record.get("store"):
            try:
                store = self.chatbot.open_store()
                detail = store.get_experiment(experiment_id) if store else None
                if detail is None:
                    warning = "experiment is not recorded in this workflow's live database"
                elif any(detail.get(key) != record[key] for key in
                        ("benchmark_id", "benchmark_version", "benchmark_digest_sha256")):
                    warning = "recorded experiment does not match its benchmark registration"
                else:
                    recorded = True
            except (OSError, sqlite3.Error, IncompatibleObservabilityDB) as exc:
                warning = str(exc)
        # The winner is read here, not fetched separately, because the detail
        # screen has to say whether THIS experiment is the one the contest
        # currently names -- and `workflow_winner` never creates the control,
        # so a read of a workflow nobody has decided in leaves it untouched.
        winner = None
        sole_member = False
        if self.chatbot.workspace is None:
            try:
                winner = benchmark_setup.workflow_winner(folder, experiment_id)
                sole_member = benchmark_setup.is_sole_group_member(folder, experiment_id)
            except (OSError, sqlite3.Error, ValueError) as exc:
                warning = warning or str(exc)
        is_winner = bool(winner and winner.get("experiment_id") == experiment_id)
        self._send_json({"experiment": record, "benchmark": manifest,
                         "recorded": recorded, "warning": warning,
                         "runs_per_task": record.get("runs_per_task"),
                         "winner": winner,
                         "is_winner": is_winner,
                         # The SAME `is_winner`, because the two must agree:
                         # deleting the current winner is refused while the
                         # group has other members (`fix-jfy5`,
                         # `ExperimentSelected` -> 409), so offering the action
                         # would put a button on the one experiment that cannot
                         # take it. A sole member may go (`fix-65ik`). This is
                         # the honest answer, not the guard -- the winner can
                         # move between this read and the DELETE, and that
                         # refusal stays where it is.
                         "can_delete": (record.get("store") is None
                                        and self.chatbot.workspace is None
                                        and (not is_winner or sole_member))})

    def _handle_registration_patch(self, path: str, body: dict[str, Any]) -> None:
        """`PATCH /api/benchmark-experiments/<id>` -- the author's description.

        The registration file is setup data, not evidence, and this route can
        only reach one whose runner has not claimed it:
        `update_experiment_description` refuses a bound registration under the
        same lock the binding takes.
        """
        experiment_id = unquote(path[len("/api/benchmark-experiments/"):]).rstrip("/")
        if not experiment_id:
            self._error(404, "not found")
            return
        if "description" not in body:
            self._error(400, 'nothing to patch: send {"description": "..."}')
            return
        folder = self._benchmark_workflow_path(write=True)
        if folder is None:
            return
        try:
            record = benchmark_setup.update_experiment_description(
                folder, experiment_id, body.get("description")
            )
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except benchmark_setup.BenchmarkSetupConflict as exc:
            self._error(409, str(exc))
            return
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"experiment": record})

    def _benchmark_experiments(self, benchmark_id):
        from .navigation import newest_experiments_first

        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        rows, warning = [], None
        if self.chatbot.workspace is not None:
            workspace = self.chatbot.workspace
            for logical in workspace.experiments():
                matches = [workspace.experiment(segment["store_id"], segment["local_experiment_id"])
                           for segment in workspace.segments(logical["experiment_id"])]
                benchmark_rows = [
                    row for row in matches
                    if row and row.get("benchmark_id") == benchmark_id
                ]
                versions = sorted(
                    {row["benchmark_version"] for row in benchmark_rows}
                )
                if versions:
                    rows.append(
                        dict(
                            logical,
                            benchmark_version=", ".join(versions),
                            workspace=True,
                            archived=all(
                                bool(row.get("archived")) for row in benchmark_rows
                            ),
                            created_at=max(
                                row.get("created_at") or "" for row in benchmark_rows
                            ),
                        )
                    )
        else:
            registrations = benchmark_setup.registered_experiments(folder, benchmark_id)
            rows = [dict(row, registered=True, status="registered", archived=False)
                    for row in registrations]
            registered = {row["experiment_id"]: row for row in rows}
            try:
                store = self.chatbot.open_store()
                if store:
                    offset = 0
                    while True:
                        batch = store.list_experiments(limit=200, offset=offset)
                        for row in batch:
                            if row.get("benchmark_id") == benchmark_id:
                                known = registered.get(row["experiment_id"])
                                if known is None:
                                    rows.append(row)
                                elif known.get("store"):
                                    known["archived"] = row["archived"]
                        if len(batch) < 200:
                            break
                        offset += len(batch)
            except IncompatibleObservabilityDB as exc:
                warning = STORE_UNAVAILABLE + str(exc)
        # One read for the whole list. The winner is a pointer on the contest,
        # not a property of any registration row, and the list is newest-first
        # so the experiment it names is often the oldest card on the page.
        winner_id = None
        winner_automatic = False
        if self.chatbot.workspace is None:
            try:
                winner = benchmark_setup.benchmark_winner(folder, benchmark_id)
            except (OSError, sqlite3.Error, ValueError) as exc:
                warning = warning or str(exc)
                winner = None
            if winner and winner.get("experiment_id"):
                winner_id = str(winner["experiment_id"])
                winner_automatic = bool(winner.get("automatic"))
        for row in rows:
            row["is_winner"] = bool(
                winner_id and row.get("experiment_id") == winner_id
            )
        self._send_json(
            {
                "experiments": newest_experiments_first(rows),
                "warning": warning,
                "winner_experiment_id": winner_id,
                "winner_automatic": winner_automatic,
            }
        )

    def _handle_benchmarks(self, path: str) -> None:
        """Read workflow-local benchmark catalogs from ``<workflow>/benchmarks/``."""
        workflow_path = self._benchmark_workflow_path(write=False)
        if workflow_path is None:
            return
        if path == "/api/benchmarks":
            payload = []
            for benchmark_id in list_benchmarks(workflow_path):
                payload.append(
                    {
                        "benchmark_id": benchmark_id,
                        "versions": list_versions(workflow_path, benchmark_id),
                    }
                )
            for row in payload:
                if row["versions"]:
                    try:
                        manifest = load_version(workflow_path, row["benchmark_id"], row["versions"][-1])
                        if "title" in manifest:
                            row["title"] = manifest["title"]
                    except BenchmarkManifestError:
                        pass
            self._send_json({"benchmarks": payload})
            return

        rest = path[len("/api/benchmarks/") :]
        benchmark_id, _, tail = rest.partition("/")
        benchmark_id = unquote(benchmark_id)
        if not benchmark_id:
            self._error(404, "not found")
            return
        if tail == "":
            self._send_json(
                {
                    "benchmark_id": benchmark_id,
                    "versions": list_versions(workflow_path, benchmark_id),
                }
            )
            return
        if tail == "experiments":
            self._benchmark_experiments(benchmark_id)
            return
        if tail == "analysis":
            try:
                analysis = load_analysis(workflow_path, benchmark_id)
            except BenchmarkManifestError as exc:
                self._error(400, str(exc))
                return
            self._send_json({"benchmark_id": benchmark_id, "analysis": analysis})
            return
        version_prefix, _, version = tail.partition("/")
        if version_prefix != "versions" or not version:
            self._error(404, "not found")
            return
        version = unquote(version)
        try:
            manifest = load_version(workflow_path, benchmark_id, version)
        except BenchmarkManifestError as exc:
            self._error(404, str(exc))
            return
        self._send_json({"version": manifest})

    def _handle_benchmark_post(self, path: str, body: Any) -> None:
        """Create one immutable benchmark version file under the workflow folder."""
        workflow_path = self._benchmark_workflow_path(write=True)
        if workflow_path is None:
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        spec = dict(body)
        if path.startswith("/api/benchmarks/") and path.endswith("/versions"):
            encoded_id = path[len("/api/benchmarks/") : -len("/versions")].rstrip("/")
            url_benchmark_id = unquote(encoded_id)
            if not url_benchmark_id:
                self._error(404, "not found")
                return
            body_benchmark_id = spec.get("benchmark_id")
            if body_benchmark_id is not None and body_benchmark_id != url_benchmark_id:
                self._error(
                    400,
                    "benchmark_id in body does not match the URL path",
                )
                return
            spec.setdefault("benchmark_id", url_benchmark_id)
        try:
            written = write_version(workflow_path, spec)
        except BenchmarkAlreadyExistsError as exc:
            self._error(409, str(exc))
            return
        except BenchmarkManifestError as exc:
            self._error(400, str(exc))
            return
        self._send_json({"version": written}, status=201)

    @staticmethod
    def _analysis_payload_from_body(body: dict[str, Any]) -> Any:
        """Accept a bare object or ``{"analysis": ...}`` wrapper."""
        if "analysis" in body:
            if set(body) - {"analysis"}:
                raise ValueError(
                    "unexpected fields: this route updates analysis only"
                )
            return body["analysis"]
        return body

    def _handle_benchmark_analysis_put(self, path: str, body: dict[str, Any]) -> None:
        """``PUT /api/benchmarks/<id>/analysis`` — mutable sibling analysis file."""
        workflow_path = self._benchmark_workflow_path(write=True)
        if workflow_path is None:
            return
        encoded_id = path[len("/api/benchmarks/") : -len("/analysis")].rstrip("/")
        benchmark_id = unquote(encoded_id)
        if not benchmark_id:
            self._error(404, "not found")
            return
        try:
            payload = self._analysis_payload_from_body(body)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        try:
            written = write_analysis(workflow_path, benchmark_id, payload)
        except BenchmarkManifestError as exc:
            self._error(400, str(exc))
            return
        self._send_json({"benchmark_id": benchmark_id, "analysis": written})
