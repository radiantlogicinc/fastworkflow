"""Package-private mixin of select, configure, train, and clear routes, moved verbatim from server.py."""

from __future__ import annotations

import os
from typing import Any

from fastworkflow.observability.workspace import WorkspaceError
from fastworkflow.run_chatbot.http_common import run_clear_conversations
from fastworkflow.run_chatbot.workflow_discovery import _looks_like_workflow


class _ControlPlaneRoutes:
    def _post_select_workspace(self, url_path: str, body: Any, query: dict[str, list[str]]) -> None:
        path = str(body.get("path") or "").strip()
        if not path or not os.path.isfile(path):
            self._error(400, f"not a file: {path!r}")
            return
        try:
            session = self.chatbot.activate_workspace(path)
        except (OSError, ValueError, WorkspaceError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"session": session})

    def _post_select_workflow(self, url_path: str, body: Any, query: dict[str, list[str]]) -> None:
        path = str(body.get("path") or "").strip()
        if not path or not os.path.isdir(path):
            self._error(400, f"not a directory: {path!r}")
            return
        if not _looks_like_workflow(path):
            self._error(
                400,
                f"{path} does not look like a fastWorkflow workflow "
                "(no _commands/ or ___command_info/ inside)",
            )
            return
        self._send_json({"session": self.chatbot.activate_workflow(path)})

    def _post_configure_env(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        try:
            session = self.chatbot.configure_env_files(
                env_content=body.get("env_content"),
                passwords_content=body.get("passwords_content"),
                create_from_templates=bool(body.get("create_from_templates")),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"session": session})

    def _post_train(self, url_path: str, body: Any, query: dict[str, list[str]]) -> None:
        path = str(body.get("path") or "").strip()
        if not path or not os.path.isdir(path):
            self._error(400, f"not a directory: {path!r}")
            return
        if not _looks_like_workflow(path):
            self._error(
                400,
                f"{path} does not look like a fastWorkflow workflow "
                "(no _commands/ or ___command_info/ inside)",
            )
            return
        try:
            result = self.chatbot.start_train(path)
        except OSError as exc:
            self._error(500, f"could not start training: {exc}")
            return
        except ValueError as exc:
            reason = str(exc)
            lowered = reason.lower()
            status = (
                409
                if ("already running" in lowered or "in progress" in lowered)
                else 400
            )
            self._error(status, reason)
            return
        self._send_json(result)

    def _post_clear_conversations(self, path: str, body: Any, query: dict[str, list[str]]) -> None:
        if body.get("confirm") != "clear all conversations":
            self._error(
                400,
                "confirmation required: confirm='clear all conversations'",
            )
            return
        if not self.chatbot.db_path or not os.path.exists(self.chatbot.db_path):
            self._send_json({"deleted": {}})
            return
        deleted = run_clear_conversations(
            self.chatbot.db_path, self.chatbot.workflow_path
        )
        self._send_json({"deleted": deleted})
