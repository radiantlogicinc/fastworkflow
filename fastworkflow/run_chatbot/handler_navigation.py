"""Package-private mixin of chatbot navigation routes, moved verbatim from server.py."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any

from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.benchmark.catalog import (
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_version,
)
from fastworkflow.observability.store import IncompatibleObservabilityDB
from fastworkflow.run_chatbot.http_common import STORE_UNAVAILABLE


class _NavigationRoutes:
    def _handle_navigation(self):
        from .navigation import build_navigation
        benchmarks, registrations, warnings = [], [], []
        store = None
        folder = self.chatbot.workflow_path
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
        try:
            if benchmarks:
                known = {row["benchmark_id"] for row in benchmarks}
                registrations = [
                    row for row in benchmark_setup.registered_experiments(folder)
                    if row["benchmark_id"] in known
                ]
            store = self.chatbot.open_store()
        except (IncompatibleObservabilityDB, OSError, sqlite3.Error) as exc:
            warnings.append(STORE_UNAVAILABLE + str(exc))
        self._send_navigation(
            {"root": build_navigation(benchmarks, registrations, store, warnings)}
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
