"""Package-private mixin of chatbot workspace read routes, moved verbatim from server.py."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from fastworkflow.observability import training_history
from fastworkflow.observability.diagnosis import InvalidTurnQuery, search_turns
from fastworkflow.observability.workspace import (
    UnknownLogicalExperiment,
    UnknownWorkspaceStore,
    WorkspaceBusyError,
    WorkspaceIntegrityError,
)
from fastworkflow.run_chatbot.provenance import (
    benchmark_pin_check,
    experiment_provenance,
)
from fastworkflow.run_chatbot.turn_annotations import (
    _wire_bool,
    _wire_number,
    _workspace_segment_verdicts,
    annotate_projected_attempts,
    annotate_turn_detail,
    annotate_turn_diagnosis,
    annotate_turn_rows,
    annotate_workspace_attempts,
    diagnostic_store_id,
    evidence_verdict,
    turn_query_from_params,
    turn_span_stamps,
)


class _WorkspaceRoutes:
    def _search_turns_response(self, store: Any, q: Any, *, store_id: str = "") -> None:
        """Answer a turn search over the complete authorized dataset.

        Shared by the live rail and the store-scoped workspace route so both
        mean the same thing by the same code, which is the point of the
        predicate living in `diagnosis` rather than in either caller.

        The response keeps `turns` at its top level, so a client reading only
        that keeps working, and adds the counts, facets and continuation the
        list needs to say honestly how much of the dataset it has looked at.
        """
        try:
            query = turn_query_from_params(q)
        except InvalidTurnQuery as exc:
            self._error(400, str(exc))
            return
        try:
            with self.chatbot._turns_scan_lock:
                page = search_turns(
                    store,
                    query,
                    store_id=store_id or diagnostic_store_id(store),
                    with_facets=_wire_bool(q("facets"), "facets") is not False,
                    derived_cache=self.chatbot._derived_turn_cache,
                    page_stamps=turn_span_stamps,
                )
        except InvalidTurnQuery as exc:
            self._error(400, str(exc))
            return
        payload = page.as_dict()
        # The rail's existing chips read the cut-at-limit tally and the cost
        # roll-up, which are tier-1/2 stamps rather than diagnostic markers and
        # so are not part of the scan's projection. Stamping the PAGE keeps
        # that read bounded by the page: the scan may have walked the store,
        # but only these rows are rendered. When the scan already stamped the
        # page (spans were in hand), this is a no-op.
        annotate_turn_rows(store, payload["turns"])
        self._send_json(payload)

    def _handle_workspace(self, path: str, q: Any) -> None:
        """Read-only HTTP projection of a validated multi-store workspace."""
        workspace = self.chatbot.workspace
        if workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        try:
            if path in {"/api/workspace", "/api/workspace/summary"}:
                self._send_json({"workspace": workspace.summary()})
                return
            if path == "/api/workspace/stores":
                self._send_json({"stores": workspace.stores()})
                return
            if path == "/api/workspace/experiments":
                self._send_json({"experiments": workspace.experiments()})
                return
            if path == "/api/workspace/turns":
                # The scoped twin of `/api/turns`. `store_id` is required and is
                # resolved by the registry, which is what keeps a workspace read
                # inside the stores the manifest named: an unknown id raises
                # `UnknownWorkspaceStore` below rather than resolving a path.
                store_id = q("store_id") or ""
                if not store_id:
                    self._error(
                        400,
                        "workspace turn search requires store_id; turns are "
                        "never searched across stores",
                    )
                    return
                with workspace.registry.open(store_id) as scoped:
                    self._search_turns_response(scoped, q, store_id=store_id)
                return
            if path == "/api/workspace/training-runs":
                store_id = q("store_id") or ""
                if not store_id:
                    self._error(
                        400,
                        "workspace training-history reads require store_id; "
                        "training runs are never listed across stores",
                    )
                    return
                with workspace.registry.open(store_id) as scoped:
                    self._send_json(
                        {
                            "store_id": store_id,
                            "training_runs": training_history.list_training_runs(
                                scoped, limit=self._training_limit(q)
                            ),
                        }
                    )
                return
            training_prefix = "/api/workspace/training-run/"
            if path.startswith(training_prefix):
                rest = path[len(training_prefix):]
                encoded_store, separator, encoded_run = rest.partition("/")
                if not separator or not encoded_store or not encoded_run:
                    self._error(
                        400,
                        "training-run reads require both store_id and run_id",
                    )
                    return
                store_id = unquote(encoded_store)
                run_id = unquote(encoded_run)
                with workspace.registry.open(store_id) as scoped:
                    # A primary-key read: an old run stays readable no matter
                    # how many have been recorded since.
                    detail = training_history.training_run_detail(scoped, run_id)
                if detail is None:
                    self._error(404, "training run not found in the named store")
                    return
                detail["store_id"] = store_id
                self._send_json({"training_run": detail})
                return
            if path == "/api/workspace/task-feedback":
                self._handle_workspace_task_feedback(workspace, q)
                return
            if path == "/api/workspace/projected_attempts":
                attempt = None
                if q("attempt") is not None:
                    try:
                        attempt = int(q("attempt"))
                    except ValueError:
                        self._error(400, "attempt must be an integer")
                        return
                projected = workspace.projected_attempts(
                    experiment_id=q("experiment"),
                    task_id=q("task"),
                    attempt=attempt,
                )
                annotate_projected_attempts(workspace, projected)
                self._send_json({"projected_attempts": projected})
                return
            experiment_prefix = "/api/workspace/experiment/"
            if path.startswith(experiment_prefix):
                rest = path[len(experiment_prefix) :]
                encoded_id, separator, operation = rest.partition("/")
                experiment_id = unquote(encoded_id)
                if not separator or operation not in {
                    "segments",
                    "tasks",
                    "attempts",
                }:
                    self._error(404, "not found")
                    return
                if operation == "segments":
                    segments = workspace.segments(experiment_id)
                    for segment in segments:
                        segment["evidence"] = evidence_verdict(
                            workspace.evidence_runs(
                                segment["store_id"], segment["local_experiment_id"]
                            )
                        )
                        local = workspace.experiment(
                            segment["store_id"], segment["local_experiment_id"]
                        )
                        segment["provenance"] = experiment_provenance(
                            local or {},
                            workspace.attempts_in_store(
                                segment["store_id"], segment["local_experiment_id"]
                            ),
                        )
                        # Provenance lists the pin as recorded; this checks it
                        # against the catalogue file the manifest points at.
                        # Attached to the segment because the pin belongs to
                        # the local experiment row, and two segments of one
                        # logical experiment may have been pinned differently.
                        segment["benchmark_pin"] = benchmark_pin_check(
                            workspace.workflow_folderpath, local or {}
                        )
                    self._send_json({"segments": segments})
                elif operation == "tasks":
                    self._send_json({"tasks": workspace.tasks(experiment_id)})
                else:
                    rows = workspace.attempts(experiment_id, task_id=q("task"))
                    annotate_workspace_attempts(
                        workspace,
                        rows,
                        _workspace_segment_verdicts(workspace, experiment_id),
                    )
                    self._send_json({"attempts": rows})
                return
            for noun in ("turn", "trace", "spans"):
                prefix = f"/api/workspace/{noun}/"
                if not path.startswith(prefix):
                    continue
                rest = path[len(prefix) :]
                encoded_store, separator, encoded_key = rest.partition("/")
                if not separator or not encoded_store or not encoded_key:
                    self._error(
                        400,
                        f"{noun} reads require both store_id and logical_turn_key",
                    )
                    return
                store_id = unquote(encoded_store)
                logical_turn_key = unquote(encoded_key)
                if noun == "turn":
                    turn = workspace.turn(store_id, logical_turn_key)
                    if turn is None:
                        self._error(404, "turn not found in the named store")
                        return
                    spans = workspace.trace(store_id, logical_turn_key)
                    annotate_turn_detail(turn, spans)
                    try:
                        annotate_turn_diagnosis(
                            turn,
                            spans,
                            store_id=store_id,
                            low_confidence_below=_wire_number(
                                q("low_confidence_below"), "low_confidence_below"
                            ),
                        )
                    except InvalidTurnQuery as exc:
                        self._error(400, str(exc))
                        return
                    self._send_json({"turn": turn})
                else:
                    self._send_json(
                        {"spans": workspace.trace(store_id, logical_turn_key)}
                    )
                return
            self._error(404, "not found")
        except (UnknownWorkspaceStore, UnknownLogicalExperiment) as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
        except WorkspaceIntegrityError as exc:
            self._error(409, str(exc))
        except WorkspaceBusyError as exc:
            self._error(503, str(exc))
