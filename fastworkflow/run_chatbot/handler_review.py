"""Package-private mixin of chatbot review routes, moved verbatim from server.py."""

from __future__ import annotations

import sqlite3
from typing import Any
from urllib.parse import unquote

from fastworkflow.observability.workspace import (
    UnknownWorkspaceStore,
    WorkspaceBusyError,
    WorkspaceIntegrityError,
)
from fastworkflow.review.sidecar import (
    ReviewAuthorizationError,
    ReviewNotFoundError,
    ReviewValidationError,
    project_review_trace,
    project_review_turn,
)


class _ReviewRoutes:
    def _post_review_capture(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        review_answer_path = (
            path.startswith("/api/review/assignments/")
            and path.endswith("/answers")
        )
        review_adjudication_path = (
            path.startswith("/api/review/assignments/")
            and path.endswith("/adjudications")
        )
        if not review_answer_path and not review_adjudication_path:
            self._error(404, "not found")
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        if self.chatbot.workspace is None:
            self._error(
                409,
                "review answers require an active observability workspace",
            )
            return
        suffix = (
            "/answers" if review_answer_path else "/adjudications"
        )
        encoded_id = path[
            len("/api/review/assignments/") : -len(suffix)
        ].rstrip("/")
        if not encoded_id:
            self._error(404, "not found")
            return
        assignment_id = unquote(encoded_id)
        capability = self._review_capability()
        try:
            sidecar = self.chatbot.open_review_sidecar()
            # Authorize the path before appending an immutable revision.
            role = "rater" if review_answer_path else "adjudicator"
            sidecar.authorize_capability(assignment_id, capability, role)
            capture = (
                sidecar.capture_answer
                if review_answer_path
                else sidecar.capture_adjudication
            )
            captured = capture(
                capability,
                str(body.get("row_id") or ""),
                str(body.get("question_id") or ""),
                body.get("answer"),
            )
        except ReviewAuthorizationError as exc:
            self._error(403, str(exc))
            return
        except ReviewValidationError as exc:
            self._error(400, str(exc))
            return
        except ReviewNotFoundError as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
            return
        key = "answer" if review_answer_path else "adjudication"
        self._send_json({key: captured})

    def _post_review_assignment(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        if self.chatbot.workspace is None:
            self._error(
                409,
                "review assignments require an active observability workspace",
            )
            return
        try:
            created = self.chatbot.open_review_sidecar().create_assignment(body)
        except ReviewValidationError as exc:
            self._error(400, str(exc))
            return
        except sqlite3.IntegrityError:
            self._error(409, "an assignment with this id already exists")
            return
        self._send_json(created, status=201)

    def _handle_review_assignment(self, path: str) -> None:
        """Return a workspace-scoped assignment or its answer export."""
        if self.chatbot.workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        encoded_id = path[len("/api/review/assignments/") :]
        if "/rows/" in encoded_id:
            self._handle_review_evidence(encoded_id)
            return
        export = encoded_id.endswith("/export")
        progress = encoded_id.endswith("/progress")
        if export:
            encoded_id = encoded_id[: -len("/export")]
        elif progress:
            encoded_id = encoded_id[: -len("/progress")]
        if not encoded_id:
            self._error(404, "not found")
            return
        assignment_id = unquote(encoded_id)
        try:
            sidecar = self.chatbot.open_review_sidecar()
            if export:
                sidecar.authorize_capability(
                    assignment_id,
                    self._review_capability(),
                    "adjudicator",
                )
                assignment = sidecar.export_assignment(assignment_id)
            elif progress:
                assignment = sidecar.assignment_progress(
                    assignment_id, self._review_capability()
                )
            else:
                assignment = sidecar.get_assignment(assignment_id)
        except ReviewAuthorizationError as exc:
            self._error(403, str(exc))
            return
        except ReviewValidationError as exc:
            self._error(400, str(exc))
            return
        except ReviewNotFoundError:
            self._error(404, "review assignment not found")
            return
        if export:
            self._send_json({"export": assignment})
        elif progress:
            self._send_json({"progress": assignment})
        else:
            self._send_json({"assignment": assignment})

    def _handle_review_evidence(self, encoded_path: str) -> None:
        """Return the capability-gated evidence projection for one assigned row."""
        workspace = self.chatbot.workspace
        if workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        encoded_id, separator, rest = encoded_path.partition("/rows/")
        encoded_row_id, operation_separator, operation = rest.partition("/")
        if (
            not separator
            or not operation_separator
            or not encoded_id
            or not encoded_row_id
            or operation not in {"turn", "trace"}
        ):
            self._error(404, "not found")
            return
        assignment_id = unquote(encoded_id)
        row_id = unquote(encoded_row_id)
        try:
            progress = self.chatbot.open_review_sidecar().assignment_progress(
                assignment_id, self._review_capability()
            )
            row = next(
                (
                    candidate
                    for candidate in progress["assignment"]["rows"]
                    if candidate["id"] == row_id
                ),
                None,
            )
            if row is None:
                self._error(404, "review row not found")
                return
            turn_ref = row["turn_ref"]
            store_id = turn_ref.get("store_id")
            logical_turn_key = turn_ref.get("logical_turn_key")
            if not store_id or not logical_turn_key:
                self._error(400, "review row does not contain a scoped workspace turn")
                return
            blinded = bool(progress["assignment"]["blinded"])
            if operation == "turn":
                turn = workspace.turn(store_id, logical_turn_key)
                if turn is None:
                    self._error(404, "turn not found in the named store")
                    return
                self._send_json(
                    {"turn": project_review_turn(turn, blinded=blinded)}
                )
            else:
                self._send_json(
                    {
                        "spans": project_review_trace(
                            workspace.trace(store_id, logical_turn_key),
                            blinded=blinded,
                        )
                    }
                )
        except ReviewAuthorizationError as exc:
            self._error(403, str(exc))
        except ReviewValidationError as exc:
            self._error(400, str(exc))
        except ReviewNotFoundError:
            self._error(404, "review assignment not found")
        except UnknownWorkspaceStore as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
        except WorkspaceIntegrityError as exc:
            self._error(409, str(exc))
        except WorkspaceBusyError as exc:
            self._error(503, str(exc))
