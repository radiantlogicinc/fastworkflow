"""Package-private mixin of chatbot navigation routes, moved verbatim from server.py."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any
from urllib.parse import quote

from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.benchmark.catalog import (
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_version,
)
from fastworkflow.observability.store import IncompatibleObservabilityDB
from fastworkflow.observability.workspace import UnknownWorkspaceStore
from fastworkflow.run_chatbot.http_common import STORE_UNAVAILABLE


class _NavigationRoutes:
    def _handle_trace_navigation(
        self, path: str, query: dict[str, list[str]]
    ) -> None:
        """Convert a durable scoped turn reference into SPA hash navigation."""
        if self.chatbot.workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        if path != "/trace":
            self._error(
                400,
                "unscoped /trace/<key> links are refused; provide store_id and "
                "logical_turn_key to /trace",
            )
            return
        store_id = (query.get("store_id") or [""])[0]
        logical_turn_key = (query.get("logical_turn_key") or [""])[0]
        if not store_id or not logical_turn_key:
            self._error(
                400,
                "trace navigation requires store_id and logical_turn_key",
            )
            return
        try:
            if self.chatbot.workspace.turn(store_id, logical_turn_key) is None:
                self._error(404, "turn not found in the named store")
                return
        except UnknownWorkspaceStore as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
            return
        fragment = (
            "store="
            + quote(store_id, safe="")
            + "&turn="
            + quote(logical_turn_key, safe="")
        )
        location = "/?token=" + quote(self.chatbot.token, safe="") + "#" + fragment
        self._send(
            303,
            b"",
            "text/plain; charset=utf-8",
            {"Location": location},
        )

    def _handle_navigation(self):
        from .navigation import build_navigation, read_source
        benchmarks, registrations, sources, warnings = [], [], [], []
        folder = self.chatbot.workflow_path
        workspace = self.chatbot.workspace
        if workspace is not None:
            folder = workspace.summary().get("workflow_folderpath")
        if folder and os.path.isdir(folder):
            for bid in list_benchmarks(folder):
                versions = list_versions(folder, bid)
                row = {"benchmark_id": bid, "versions": versions}
                if versions:
                    try:
                        row.update(load_version(folder, bid, versions[-1]))
                    except BenchmarkManifestError as exc:
                        warnings.append(str(exc))
                benchmarks.append(row)
                if workspace is None:
                    registrations.extend(benchmark_setup.registered_experiments(folder, bid))
        if workspace is not None:
            for descriptor in workspace.stores():
                sid = descriptor["store_id"]
                with workspace.registry.open(sid) as store:
                    sources.append(read_source(store, {"store_id": sid}))
            root = build_navigation(benchmarks, [], sources, warnings)
            self._send_navigation({"root": root})
            return
        try:
            store = self.chatbot.open_store()
            if store:
                sources.append({"store": store, "source": None})
        except (IncompatibleObservabilityDB, OSError, sqlite3.Error) as exc:
            warnings.append(STORE_UNAVAILABLE + str(exc))
        self._send_navigation(
            {"root": build_navigation(benchmarks, registrations, sources, warnings)}
        )

    def _send_navigation(self, payload: Any) -> None:
        """JSON navigation body with a content hash ETag (304 on If-None-Match)."""
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        etag = '"' + hashlib.sha256(body).hexdigest() + '"'
        if_none = self.headers.get("If-None-Match")
        if if_none and if_none == etag:
            self._send(
                304,
                b"",
                "application/json; charset=utf-8",
                {"ETag": etag},
            )
            return
        self._send(
            200,
            body,
            "application/json; charset=utf-8",
            {"ETag": etag},
        )
