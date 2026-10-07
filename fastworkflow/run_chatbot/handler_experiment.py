"""Package-private mixin of experiment, setup, selection, and training-history routes, moved verbatim from server.py."""

from __future__ import annotations

import sqlite3
from typing import Any, Optional
from urllib.parse import unquote

from fastworkflow.experiment.setup import ExperimentSetups, SetupConflict
from fastworkflow.observability import training_history
from fastworkflow.observability.store import (
    FEATURE_EXPERIMENTS_V1,
    ExperimentNotFound,
    IncompatibleObservabilityDB,
    ObservabilityStore,
    ReadOnlyObservabilityStore,
)
from fastworkflow.run_chatbot.provenance import (
    experiment_provenance,
    provenance_differences,
)
from fastworkflow.run_chatbot.turn_annotations import (
    TRAINING_RUN_DEFAULT_LIMIT,
    TRAINING_RUN_MAX_LIMIT,
    annotate_attempt_rows,
    evidence_verdict,
)
from fastworkflow.run_chatbot import selection_api


class _ExperimentRoutes:
    def _post_selection(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_selection_api("POST", path, body=body)

    def _post_experiment_setup(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_setup(path, body=body, write=True)

    def _delete_selection(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        # Withdrawing a task's best run carries `expected_selection_id`,
        # so this DELETE has a body like the POST that installed it.
        self._handle_selection_api("DELETE", path, body=body)

    def _patch_experiment(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        self._handle_experiment_patch(path, body)

    @staticmethod
    def _training_limit(q: Any) -> int:
        """How many `train_runs` rows one read will look at.

        Bounded on both ends rather than passed through: `list_train_runs` has
        no keyset cursor, so a caller-chosen limit is the only bound this table
        has, and an unbounded one would let a single request pull every
        training run's metrics blob into memory at once.
        """
        try:
            requested = int(q("limit"))
        except (TypeError, ValueError):
            return TRAINING_RUN_DEFAULT_LIMIT
        return max(1, min(requested, TRAINING_RUN_MAX_LIMIT))

    def _handle_training_history(self, store: Any, path: str, q: Any) -> None:
        """`GET /api/training-runs` and `/api/training-run/<run_id>`.

        Read-only and store-scoped: the store the request already resolved is
        the one read. No training is started, no
        artifact directory is opened and nothing is downloaded -- every field
        comes from rows `train.metrics_persistence` already wrote.

        `limit` bounds the LIST only. The detail route resolves its run by
        primary key, so a run that has fallen out of the newest-first window
        is still readable -- a 404 here means the store holds no such run,
        never that the caller asked for too few rows.
        """
        if path == "/api/training-runs":
            self._send_json(
                {
                    "training_runs": training_history.list_training_runs(
                        store, limit=self._training_limit(q)
                    )
                }
            )
            return
        run_id = unquote(path[len("/api/training-run/"):]).rstrip("/")
        if not run_id:
            self._error(404, "not found")
            return
        detail = training_history.training_run_detail(store, run_id)
        if detail is None:
            self._error(404, "training run not found")
            return
        self._send_json({"training_run": detail})

    def _handle_setup(self, path, *, body=None, write=False):
        workflow_path = self.chatbot.workflow_path
        if not workflow_path:
            self._error(409, "Select a workflow before reviewing experiment setups")
            return
        setups = ExperimentSetups(workflow_path)
        rest = path[len("/api/experiment-setups") :].strip("/")
        parts = rest.split("/") if rest else []
        try:
            if write and not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if not parts:
                if write:
                    result = setups.save(body.get("spec"), body.get("expected_revision"))
                    self._send_json({"setup": result}, status=201)
                else:
                    self._send_json({"setups": setups.list()})
            elif len(parts) == 1 and not write:
                self._send_json({"setup": setups.get(unquote(parts[0]))})
            elif len(parts) == 2 and parts[1] == "decisions" and write:
                result = setups.decide(
                    unquote(parts[0]),
                    body.get("revision"),
                    body.get("digest"),
                    body.get("decision"),
                    body.get("reviewer"),
                    body.get("comment", ""),
                )
                self._send_json({"setup": result}, status=201)
            elif len(parts) == 2 and parts[1] == "export" and not write:
                current = setups.get(unquote(parts[0]))
                self._send_json(
                    setups.approved(
                        current["experiment_id"], current["revision"], current["digest"]
                    )
                )
            else:
                self._error(404, "not found")
        except SetupConflict as exc:
            self._error(409, str(exc))
        except KeyError:
            self._error(404, "setup not found")
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))

    def _handle_selection_api(
        self,
        method: str,
        path: str,
        *,
        query: Optional[dict[str, list[str]]] = None,
        body: Any = None,
    ) -> None:
        """Winner, best-run, comparison and pair-review routes (`selection_api`).

        Judgements are live-workflow only. The winner of a contest and the best
        run of a task live in the WORKFLOW's live DB control tables, not in the
        evidence.
        """
        workflow_path = (self.chatbot.workflow_path or "").strip()
        if not workflow_path:
            self._error(409, "select a workflow before using experiment selections")
            return
        if method == "GET":
            result = selection_api.handle_get(workflow_path, path, query or {})
        elif method == "POST":
            result = selection_api.handle_post(workflow_path, path, body)
        else:
            result = selection_api.handle_delete(workflow_path, path, body)
        if result is None:
            self._error(404, "not found")
            return
        status, payload = result
        self._send_json(payload, status=status)

    def _handle_experiments(
        self,
        store: ReadOnlyObservabilityStore,
        path: str,
        q: Any,
    ) -> None:
        """The `/api/experiment*` GET surface.

        The noun choice is deliberate: this surface never calls an experiment
        attempt a "run". An experiment has tasks, a task has attempts, and an
        attempt resolves to the channel/conversation/turn keys the existing
        trace views already render — so nothing here re-implements a viewer.

        A DB written before the experiment tables existed 404s with a reason a
        human can act on, rather than raising `no such table` behind a generic
        500.
        """
        if not store.has_feature(FEATURE_EXPERIMENTS_V1):
            self._error(404, "this database predates experiment recording")
            return
        if path == "/api/experiments":
            self._send_json(
                {
                    "experiments": store.list_experiments(
                        status=q("status"),
                        arm=q("arm"),
                        limit=self._int(q("limit"), 100),
                        offset=self._int(q("offset"), 0),
                    )
                }
            )
            return

        rest = path[len("/api/experiment/") :]
        # Split first, THEN decode: decoding first would let a %2F inside an id
        # invent a sub-path segment. The SPA sends encodeURIComponent(id) and
        # create_experiment accepts any caller-supplied id, so an id containing
        # a space or a slash would otherwise 404 forever.
        experiment_id, _, sub = rest.partition("/")
        experiment_id = unquote(experiment_id)
        if not experiment_id:
            self._error(404, "not found")
            return
        if sub == "":
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            detail["evidence"] = evidence_verdict(detail.get("evidence_runs"))
            detail["provenance"] = experiment_provenance(
                detail, store.experiment_attempt_rows(experiment_id)
            )
            self._send_json({"experiment": detail})
        elif sub == "tasks":
            if store.get_experiment(experiment_id) is None:
                self._error(404, "experiment not found")
                return
            self._send_json({"tasks": store.experiment_tasks(experiment_id)})
        elif sub == "attempts":
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            rows = store.experiment_attempt_rows(experiment_id, task_id=q("task"))
            annotate_attempt_rows(
                store, rows, evidence_verdict(detail.get("evidence_runs"))
            )
            self._send_json({"attempts": rows})
        elif sub == "score":
            try:
                self._send_json({"score": store.experiment_scores(experiment_id)})
            except ExperimentNotFound:
                self._error(404, "experiment not found")
        elif sub == "compare":
            # Resolve the experiment BEFORE branching on the baseline: folding
            # a missing experiment into `(detail or {})` reported it as 400
            # "no baseline" rather than 404 "experiment not found", which sends
            # the reader looking for the wrong thing.
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            baseline = q("baseline") or detail.get("baseline_experiment_id")
            if not baseline:
                self._error(
                    400,
                    "no baseline: pass ?baseline=<experiment_id> or set "
                    "baseline_experiment_id on the experiment",
                )
                return
            try:
                comparison = store.compare_experiments(experiment_id, baseline)
            except ExperimentNotFound as exc:
                self._error(404, f"experiment not found: {exc.experiment_id}")
                return
            # (c) the comparability check rides along on BOTH answers: a
            # provenance difference is quoted, never a refusal, so the 409 the
            # store already issues for a differing benchmark pin stays the
            # only thing that blocks the view.
            baseline_detail = store.get_experiment(baseline)
            comparison["provenance_differences"] = provenance_differences(
                experiment_provenance(
                    detail, store.experiment_attempt_rows(experiment_id)
                ),
                experiment_provenance(
                    baseline_detail or {}, store.experiment_attempt_rows(baseline)
                ),
            )
            # 409, not 200-with-a-flag: an incomparable pair is a refusal, and a
            # client that renders whatever it got would render a comparison of
            # two runs that share no task.
            status = 200 if comparison.get("comparable") else 409
            self._send_json(comparison, status=status)
        else:
            self._error(404, "not found")

    def _handle_experiment_patch(self, path: str, body: dict[str, Any]) -> None:
        """`PATCH /api/experiment/<id>` -- editable annotations.

        Admitted on the annotation argument in `[DR30]`: the invariant
        protected is "recorded observability data stays read-only over HTTP"
        (studio design §3.4, the access-control section), and `notes` plus
        `archived` are annotation columns that cannot alter any span, turn,
        artifact, attempt outcome or score.
        """
        rest = path[len("/api/experiment/") :]
        # partition, not split-and-discard: the GET side validates its sub-path
        # and 404s on an unknown one, and a write route that silently accepted
        # /api/experiment/<id>/anything would make every GET sub-path an
        # undocumented alias for the notes PATCH.
        experiment_id, _, sub = rest.partition("/")
        experiment_id = unquote(experiment_id)
        if not experiment_id or sub:
            self._error(404, "not found")
            return
        try:
            store = self.chatbot.open_store()
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        if store is None or not store.has_feature(FEATURE_EXPERIMENTS_V1):
            self._error(404, "this database predates experiment recording")
            return
        if "analysis" in body:
            self._error(400, "analysis is not a field of an experiment; use notes")
            return
        if set(body) - {"notes", "archived"}:
            self._error(
                400, "unexpected fields: this route updates notes and archived only"
            )
            return
        if not body:
            self._error(
                400, 'nothing to patch: send {"notes": "..."} or {"archived": true}'
            )
            return
        notes = body.get("notes")
        if "notes" in body and notes is not None and not isinstance(notes, str):
            self._error(400, "notes must be a string or null")
            return
        archived = body.get("archived")
        if "archived" in body and not isinstance(archived, bool):
            self._error(400, "archived must be true or false")
            return
        try:
            # `[DR53]`: the feature check above ran through the per-request
            # READ-ONLY handle, so a PATCH against a pre-experiments snapshot
            # cannot be what creates the tables in it.
            writable = ObservabilityStore.open_for_annotation(store.db_path)
            if "notes" in body:
                writable.update_experiment_notes(experiment_id, notes)
            if "archived" in body:
                writable.update_experiment_archived(experiment_id, archived)
        except ExperimentNotFound:
            self._error(404, "experiment not found")
            return
        except (OSError, sqlite3.Error) as exc:
            self._error(
                500, f"could not update experiment annotations: {type(exc).__name__}"
            )
            return
        self._send_json({"experiment": store.get_experiment(experiment_id)})
