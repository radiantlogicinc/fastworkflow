"""Package-private mixin of chatbot benchmark routes, moved verbatim from server.py."""

from __future__ import annotations

import sqlite3
from typing import Any, Optional
from urllib.parse import unquote

from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.benchmark.catalog import (
    BenchmarkAlreadyExistsError,
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_version,
    write_version,
)
from fastworkflow.benchmark.conformance import experiment_conformance
from fastworkflow.observability import control
from fastworkflow.observability.store import IncompatibleObservabilityDB
from fastworkflow.run_chatbot.http_common import STORE_UNAVAILABLE


class _BenchmarkRoutes:
    def _post_benchmark_record(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        folder = self._benchmark_workflow_path()
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
                    user_simulator=body.get("user_simulator", False),
                )
                self._send_json({"experiment": record}, status=201)
        except (benchmark_setup.BenchmarkSetupConflict, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
        except (BenchmarkManifestError, ValueError, TypeError) as exc:
            self._error(400, str(exc))

    def _write_refused(self, exc: sqlite3.OperationalError) -> None:
        """A busy live DB is 503 (retryable), a read-only one 409 (design §5)."""
        status = control.write_refusal_status(exc)
        if status is None:
            raise exc
        self._error(status, str(exc))

    def _post_benchmark_version(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_benchmark_post(path, body)

    def _delete_benchmark_experiment(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        prefix = "/api/benchmark-experiments/"
        experiment_id = unquote(path[len(prefix):])
        try:
            record = benchmark_setup.delete_empty_experiment(folder, experiment_id)
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except (benchmark_setup.BenchmarkSetupConflict, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"deleted": experiment_id, "benchmark_id": record["benchmark_id"]})

    def _patch_registration(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_registration_patch(path, body)

    def _benchmark_workflow_path(self) -> Optional[str]:
        """Workflow folder whose live DB holds the benchmark versions."""
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
        except (ValueError, BenchmarkManifestError, IncompatibleObservabilityDB) as exc:
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
        try:
            winner = benchmark_setup.workflow_winner(folder, experiment_id)
        except (OSError, sqlite3.Error, ValueError) as exc:
            warning = warning or str(exc)
        is_winner = bool(winner and winner.get("experiment_id") == experiment_id)
        self._send_json({"experiment": record, "benchmark": manifest,
                         "recorded": recorded, "warning": warning,
                         "runs_per_task": record.get("runs_per_task"),
                         "user_simulator": record.get("user_simulator", False),
                         "winner": winner,
                         "is_winner": is_winner,
                         # A registration nobody has run may always go; when it
                         # was the winner, its version simply has no winner.
                         "can_delete": record.get("store") is None})

    def _handle_registration_patch(self, path: str, body: dict[str, Any]) -> None:
        """`PATCH /api/benchmark-experiments/<id>` -- the author's description.

        The registration is setup data, not evidence, and this route can
        only reach one whose runner has not claimed it:
        `update_experiment_description` refuses a bound registration in the
        same kind of transaction the binding takes.
        """
        experiment_id = unquote(path[len("/api/benchmark-experiments/"):]).rstrip("/")
        if not experiment_id:
            self._error(404, "not found")
            return
        if "description" not in body:
            self._error(400, 'nothing to patch: send {"description": "..."}')
            return
        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        try:
            record = benchmark_setup.update_experiment_description(
                folder, experiment_id, body.get("description")
            )
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except (benchmark_setup.BenchmarkSetupConflict, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"experiment": record})

    def _benchmark_experiments(self, benchmark_id, version):
        """One benchmark VERSION's experiments and winner (`fix-873a`).

        There is deliberately no all-versions listing: two versions ran
        different task sets, so their experiments are never shown together.
        """
        from .navigation import newest_experiments_first

        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        rows, warning, store = [], None, None
        try:
            registrations = benchmark_setup.registered_experiments(folder, benchmark_id)
            rows = [dict(row, registered=True, status="registered", archived=False)
                    for row in registrations
                    if row.get("benchmark_version") == version]
            registered = {row["experiment_id"]: row for row in rows}
            store = self.chatbot.open_store()
            if store:
                offset = 0
                while True:
                    batch = store.list_experiments(limit=200, offset=offset)
                    for row in batch:
                        if (row.get("benchmark_id") == benchmark_id
                                and row.get("benchmark_version") == version):
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
        try:
            winner = benchmark_setup.benchmark_winner(folder, benchmark_id, version)
        except (OSError, sqlite3.Error, ValueError) as exc:
            warning = warning or str(exc)
            winner = None
        if winner and winner.get("experiment_id"):
            winner_id = str(winner["experiment_id"])
        for row in rows:
            row["is_winner"] = bool(
                winner_id and row.get("experiment_id") == winner_id
            )
        # Each recorded run, checked against this version (`fix-mgoi`).
        try:
            manifest = load_version(folder, benchmark_id, version)
        except BenchmarkManifestError:
            manifest = None
        if manifest is not None and store:
            for row in rows:
                recorded = store.get_experiment(row["experiment_id"])
                if recorded is not None:
                    row["conformance"] = experiment_conformance(store, manifest, recorded)
        self._send_json(
            {
                "experiments": newest_experiments_first(rows),
                "warning": warning,
                "benchmark_version": version,
                "winner_experiment_id": winner_id,
            }
        )

    def _handle_benchmarks(self, path: str) -> None:
        """Read the benchmark catalog from the workflow's live DB."""
        workflow_path = self._benchmark_workflow_path()
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
        version_prefix, _, version = tail.partition("/")
        if version_prefix != "versions" or not version:
            self._error(404, "not found")
            return
        version, _, sub = version.partition("/")
        version = unquote(version)
        if sub == "experiments":
            self._benchmark_experiments(benchmark_id, version)
            return
        if sub:
            self._error(404, "not found")
            return
        try:
            manifest = load_version(workflow_path, benchmark_id, version)
        except BenchmarkManifestError as exc:
            self._error(404, str(exc))
            return
        self._send_json({"version": manifest})

    def _handle_benchmark_post(self, path: str, body: Any) -> None:
        """Store one immutable benchmark version in the workflow's live DB."""
        workflow_path = self._benchmark_workflow_path()
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
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json({"version": written}, status=201)
