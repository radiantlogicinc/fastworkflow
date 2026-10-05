"""fastWorkflow Chatbot debug mode: stdlib-only localhost read-only HTTP layer.

Design invariants (docs/fastworkflow_observability_studio_design.md §3.4):

- Access control: binds 127.0.0.1 only; a per-launch random bearer
  token (``secrets.token_urlsafe``) embedded in the printed URL (Jupyter
  pattern) is required on EVERY request (Authorization header or ``?token=``),
  compared in constant time (``hmac.compare_digest``); a strict Host/Origin
  allowlist rejects everything non-loopback with 403. Loopback hosts
  (``127.0.0.1`` / ``localhost`` / ``[::1]``) pass on ANY port — port
  forwarders (WSL relays, IDE port forwards) legitimately re-expose the
  server on a different local port — while the loopback-only rule is what
  defeats DNS rebinding, and the token stays the authentication.
- Rendering safety: the SPA page ships with a restrictive CSP
  (inline script allowed only via its own sha256 hashes — the page is one
  self-contained file); artifact responses carry
  ``default-src 'none'; sandbox`` so direct navigation is inert; the read
  layer only calls ObservabilityStore methods (parameterized queries).
- Read discipline: per-request store reads — every ObservabilityStore
  method opens its own short-lived connection, no held cursors — so WAL
  checkpointing by the writer never starves.
- Packaging: stdlib-only. This module must never import
  fastapi/uvicorn or any third-party HTTP dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.resources
import json
import logging
import os
import re
import secrets
import signal
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, NamedTuple, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from fastworkflow import state_paths
from fastworkflow.observability import feedback
from fastworkflow.observability.diagnosis import InvalidTurnQuery
from fastworkflow.observability.derived_cache import DerivedTurnCache
from fastworkflow.observability.store import (
    IncompatibleObservabilityDB,
    ObservabilityStore,
    ReadOnlyObservabilityStore,
    Redactor,
)
from fastworkflow.observability.workspace import (
    ObservabilityWorkspace,
    WorkspaceError,
    load_observability_workspace,
)
from fastworkflow.review.sidecar import (
    ReviewSidecar,
    ReviewValidationError,
)
from fastworkflow.run_chatbot import launcher
from fastworkflow.run_chatbot import selection_api

# Names moved verbatim into sibling modules. Re-exported so the handler below,
# tests, and other callers keep importing them from this module.
from fastworkflow.run_chatbot.turn_annotations import (
    LOW_CONFIDENCE_DEFAULT_MARGIN,
    TRAINING_RUN_DEFAULT_LIMIT,
    TRAINING_RUN_MAX_LIMIT,
    _wire_bool,
    _wire_number,
    _workspace_segment_verdicts,
    annotate_attempt_rows,
    annotate_projected_attempts,
    annotate_turn_detail,
    annotate_turn_diagnosis,
    annotate_turn_rows,
    annotate_workspace_attempts,
    cost_rollup,
    count_llm_calls_cut_at_limit,
    diagnostic_store_id,
    evidence_verdict,
    execution_ledger,
    is_low_confidence,
    llm_call_cost,
    llm_call_cut_at_limit,
    merge_cost_rollups,
    turn_decision_signals,
    turn_query_from_params,
    turn_span_stamps,
)
from fastworkflow.run_chatbot.provenance import (
    benchmark_pin_check,
    experiment_provenance,
    provenance_differences,
)
from fastworkflow.run_chatbot.workflow_discovery import (
    _looks_like_workflow,
    _rel_under,
    _workflow_is_trained,
    browse_directories,
    invalidate_workflow_candidate_walk_cache,
    list_workflow_candidates,
)

from fastworkflow.run_chatbot.handler_benchmark import _BenchmarkRoutes
from fastworkflow.run_chatbot.handler_control import _ControlPlaneRoutes
from fastworkflow.run_chatbot.handler_experiment import _ExperimentRoutes
from fastworkflow.run_chatbot.handler_feedback import _FeedbackRoutes
from fastworkflow.run_chatbot.handler_navigation import _NavigationRoutes
from fastworkflow.run_chatbot.handler_review import _ReviewRoutes
from fastworkflow.run_chatbot.handler_workspace import _WorkspaceRoutes
from fastworkflow.run_chatbot.http_common import (
    STORE_UNAVAILABLE,
    run_clear_conversations,
)

logger = logging.getLogger(__name__)

# Tolerates attributes (e.g. a future type="module") so adding one cannot
# silently produce an empty hash list — which would fail closed and brick the
# page. ChatbotServer.__init__ additionally asserts extraction succeeded.
_SCRIPT_RE = re.compile(rb"<script\b[^>]*>(.*?)</script>", re.DOTALL)

# Content types that may contain active content: only ever rendered inside a
# sandboxed iframe by the SPA; direct responses are additionally sandboxed via
# CSP (see _artifact_headers) [R22].
_HTMLISH_TYPES = ("text/html", "application/xhtml+xml", "image/svg+xml")


_INDEX_HTML_CACHE: bytes | None = None


def load_index_html() -> bytes:
    """The single self-contained SPA page, shipped as package data.

    Source is ordered parts under ``static/src``. Concatenating them in
    filename order is the served page, still one inline script.
    """
    global _INDEX_HTML_CACHE
    if _INDEX_HTML_CACHE is None:
        root = (
            importlib.resources.files("fastworkflow.run_chatbot")
            / "static"
            / "src"
        )
        names = sorted(entry.name for entry in root.iterdir() if entry.is_file())
        _INDEX_HTML_CACHE = b"".join((root / name).read_bytes() for name in names)
    return _INDEX_HTML_CACHE


def _inline_script_hashes(page: bytes) -> list[str]:
    """CSP sha256 sources for the page's own inline <script> blocks.

    The SPA is one self-contained file (no external requests), so
    ``script-src 'self'`` alone would block its inline script. Hash-sourcing
    keeps the policy restrictive: only the exact scripts shipped in the page
    execute; record-derived text can never inject a runnable script.
    """
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(m).digest()).decode() + "'"
        for m in _SCRIPT_RE.findall(page)
    ]


def _free_server_port(preferred: int) -> tuple[int, bool]:
    """(port to use, moved?) — the preferred port when it is free, otherwise a
    free ephemeral one. Anything may be squatting the default 8000 (an old
    server, another chatbot, an unrelated app); spawning onto a busy port is
    worse than moving: uvicorn takes seconds to fail its bind, and meanwhile
    the chat would connect to WHATEVER is already answering there — possibly a
    different workflow's server entirely."""
    import socket

    for candidate, moved in ((preferred, False), (0, True)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1], moved
        except OSError:
            continue
    return preferred, False


def _port_in_use(port: int) -> bool:
    """True when something on loopback accepts TCP connections on *port*."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return False


def _child_survived_startup(
    proc: Any,
    *,
    timeout_s: float = 1.0,
    interval_s: float = 0.05,
) -> bool:
    """True if *proc* is still alive after a bounded startup wait.

    Replaces a fixed ``sleep(1.0)`` so a child that dies immediately is
    noticed without always paying the full delay when it exits sooner.
    ``timeout_s=0`` is a single poll (used by tests that inject a FakeProc).
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        if proc.poll() is not None:
            return False
        now = time.monotonic()
        if now >= deadline:
            return True
        time.sleep(min(interval_s, deadline - now))


def _child_exited_message(exit_code: Any) -> str:
    return (
        "the FastAPI server exited immediately "
        f"(exit code {exit_code}) — its output is in "
        "the server log; check that the --server-port is free "
        "and the env files are valid"
    )


# How much of a process log one GET returns. The file itself is not trimmed
# (a live child holds it open); the bound is on the read.
_LOG_TAIL_DEFAULT = 200
_LOG_TAIL_MAX = 2000
_LOG_READ_MAX_BYTES = 1024 * 1024

# Credential env keys as they appear in log text (``KEY=value``). Same test as
# ``fastworkflow.utils.dspy_logger._is_secret_key``, inlined so this module
# does not import DSPy. ``*_TOKENS`` names usage counts and stays visible.
_LOG_ASSIGNMENT = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)"
)
_LOG_BEARER = re.compile(r"(?i)(Bearer\s+)\S+")


def _log_assignment_is_secret(key: str) -> bool:
    upper = str(key).upper()
    if any(marker in upper for marker in ("API_KEY", "SECRET", "PASSWORD")):
        return True
    return upper == "TOKEN" or (
        upper.endswith("_TOKEN") and not upper.endswith("_TOKENS")
    )


def redact_process_log(text: str, *extra_secrets: str) -> str:
    """Scrub a process-log excerpt before it leaves the chatbot.

    Reuses :class:`fastworkflow.observability.store.Redactor` for loaded
    secret values and well-known credential shapes, then also masks the
    chatbot token, any ``Bearer`` credential, and ``KEY=value`` assignments
    whose key names a secret.
    """

    def _redact_assignment(match: re.Match[str]) -> str:
        key = match.group("key")
        if not _log_assignment_is_secret(key):
            return match.group(0)
        return f"{key}=[REDACTED]"

    text = _LOG_ASSIGNMENT.sub(_redact_assignment, text)
    text = _LOG_BEARER.sub(r"\1[REDACTED]", text)
    for secret in extra_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return Redactor().redact(text)


def _parse_log_tail(query: dict[str, list[str]]) -> tuple[int, Optional[str]]:
    raw = (query.get("tail") or [None])[0]
    if raw is None or raw == "":
        return _LOG_TAIL_DEFAULT, None
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return 0, "tail must be an integer"
    if count < 1:
        return 0, "tail must be at least 1"
    return min(count, _LOG_TAIL_MAX), None


def read_log_tail(path: str, tail: int, *extra_secrets: str) -> dict[str, Any]:
    """Last ``tail`` lines of ``path``, from at most the last 1 MiB.

    Missing path or missing file is not an error: ``exists`` is false and
    ``lines`` is empty. ``truncated`` is true when the byte window or the
    line cap dropped earlier text.
    """
    if not path:
        return {"path": "", "exists": False, "lines": [], "truncated": False}
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return {"path": path, "exists": False, "lines": [], "truncated": False}
    truncated = False
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > _LOG_READ_MAX_BYTES:
            truncated = True
            handle.seek(size - _LOG_READ_MAX_BYTES)
        raw = handle.read(_LOG_READ_MAX_BYTES)
    if truncated:
        newline = raw.find(b"\n")
        raw = raw[newline + 1 :] if newline >= 0 else b""
    text = redact_process_log(
        raw.decode("utf-8", errors="replace"), *extra_secrets
    )
    lines = text.splitlines()
    if len(lines) > tail:
        lines = lines[-tail:]
        truncated = True
    return {
        "path": path,
        "exists": True,
        "lines": lines,
        "truncated": truncated,
    }


def _spawn_and_probe(plan: Any, probe_s: float) -> tuple[Any, Optional[str]]:
    """Spawn *plan*'s server and wait out the startup probe.

    Returns ``(proc, None)`` when the child is still up, else ``(None, reason)``.
    """
    try:
        proc = launcher.spawn_server(plan)
    except OSError as exc:
        return None, f"could not start the FastAPI server: {exc}"
    if not _child_survived_startup(proc, timeout_s=probe_s):
        return None, _child_exited_message(proc.returncode)
    return proc, None


_MAX_JSON_BODY_BYTES = 4 * 1024 * 1024


def _autodetect_env_files(workflow_path: str) -> tuple[str, str]:
    """Best-effort env-file discovery for the spawned server, in order:
    workflow-local files, then the bundled ``examples/`` shared files (when
    the workflow lives there). Missing files resolve to "" so the chatbot can
    offer a file picker or create workflow-local files from the templates."""
    wf = os.path.abspath(workflow_path)
    roots = [wf]
    parent = os.path.dirname(wf)
    if os.path.basename(parent) == "examples":
        roots.append(parent)

    def first_existing(filename: str) -> str:
        for root in roots:
            candidate = os.path.join(root, filename)
            if os.path.isfile(candidate):
                return candidate
        return ""

    return first_existing("fastworkflow.env"), first_existing(
        "fastworkflow.passwords.env"
    )


def _env_template_text(filename: str) -> str:
    resource = importlib.resources.files("fastworkflow") / "examples" / filename
    return resource.read_text(encoding="utf-8")


def _write_env_file(path: str, content: str) -> None:
    """Atomically write one workflow-local env file with owner-only access."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


class ChatbotServer:
    """The chatbot's local web layer.

    Ordinary observability reads use ``ReadOnlyObservabilityStore``. Explicit,
    token-gated control-plane actions select a workflow, configure missing env
    files, or clear recorded conversations.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        workflow_path: str = "",
        port: int = 0,
        token: Optional[str] = None,
        spawn_options: Optional[dict] = None,
        workspace_manifest_path: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or ""
        self.workflow_path = workflow_path
        self.workspace: Optional[ObservabilityWorkspace] = (
            load_observability_workspace(workspace_manifest_path)
            if workspace_manifest_path
            else None
        )
        self.workspace_manifest_path = (
            str(self.workspace.manifest_path) if self.workspace is not None else ""
        )
        self._review_sidecar: Optional[ReviewSidecar] = None
        self._review_sidecar_lock = threading.Lock()
        # Auto-spawn posture for the workflow's FastAPI server; see
        # run_chatbot_main. no_server=True keeps the chatbot debug-only.
        self.spawn_options = dict(spawn_options or {"no_server": True})
        self.server_proc = None  # the spawned FastAPI server (subprocess.Popen)
        self.server_url: Optional[str] = None
        self.spawn_error: Optional[str] = None
        self.server_note: Optional[str] = None  # e.g. "port 8000 busy; using 40123"
        self.env_file_path = ""
        self.passwords_file_path = ""
        self.env_setup_required = False
        # Single-user dev tool: the channel is an implementation detail the
        # developer never types, and it is FIXED rather than minted per launch.
        # A per-launch channel scattered every restart's conversations into its
        # own top-level group in the debug rail, so yesterday's turns were a
        # different "channel" from today's for no reason a developer could see.
        # Conversations still separate them; the channel no longer does.
        self.channel_id = "chatbot"
        self.user_id = "developer"
        self._activate_lock = threading.Lock()
        # Held only for session-field swap/read — never across terminate/spawn.
        self._session_state_lock = threading.Lock()
        self._train_lock = threading.Lock()
        # Most recent train log this process started, or "" until then.
        # Readers also fall back to the conventional path / a live pid file.
        self.train_log_path = ""
        # Serialize complete-dataset turn scans: concurrent ThreadingHTTPServer
        # handlers otherwise thrash the GIL decoding the same span JSON.
        self._turns_scan_lock = threading.Semaphore(1)
        self._derived_turn_cache = DerivedTurnCache()
        # Per-launch bearer token [R5]; overridable only for tests.
        self.token = token if token is not None else secrets.token_urlsafe(32)
        self.index_html = load_index_html()
        script_hashes = _inline_script_hashes(self.index_html)
        if b"<script" in self.index_html and not script_hashes:
            # Fail loudly at launch rather than serving a page whose own
            # script the CSP will block with no server-side signal.
            raise RuntimeError(
                "CSP hash extraction found no inline <script> blocks in the "
                "bundled SPA; the page would be blocked by its own policy"
            )
        script_srcs = " ".join(script_hashes)
        # connect-src: 'self' for the debug-mode read API, plus loopback-only
        # origins so TEST MODE can call the local FastAPI server
        # (/initialize, /invoke_agent, /invoke_assistant). Never a non-loopback
        # host — the SPA can only ever talk to servers on this machine [R19][R22].
        self.page_csp = (
            "default-src 'none'; "
            f"script-src 'self'{' ' + script_srcs if script_srcs else ''}; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' http://127.0.0.1:* http://localhost:*; "
            "img-src 'self' data:; "
            "frame-src 'self'"
        )

        server = self

        class _Handler(_ChatbotRequestHandler):
            chatbot = server

        # 127.0.0.1 only — never configurable to a wider bind [R5][R18].
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    def open_store(self) -> Optional[ReadOnlyObservabilityStore]:
        """Per-request READ-ONLY store handle, or None while the DB is absent
        or unopenable. The viewer never creates, migrates, or writes the DB
        it inspects — a missing DB (e.g. test-mode cold start before the
        first turn) serves empty views instead of an error.
        """
        if not self.db_path:
            return None  # no workflow selected yet
        try:
            return ReadOnlyObservabilityStore(self.db_path)
        except IncompatibleObservabilityDB:
            raise  # a newer-schema DB is a real error, surfaced per-request
        except Exception:
            return None

    def open_review_sidecar(self) -> ReviewSidecar:
        """Return the manifest-bound review store for the active workspace."""
        if self.workspace is None or not self.workspace_manifest_path:
            raise ReviewValidationError(
                "review assignments require an active observability workspace"
            )
        with self._review_sidecar_lock:
            if self._review_sidecar is None:
                self._review_sidecar = ReviewSidecar.from_workspace_manifest(
                    self.workspace_manifest_path
                )
            return self._review_sidecar

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    # -- workflow activation + server lifecycle -------------------------

    def _publish_session_state(self, **fields: Any) -> None:
        """Atomically replace the session fields readers may observe together."""
        with self._session_state_lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def resolved_server_log_path(self) -> str:
        """Absolute path of this workflow's server log, or ``""`` if none is selected."""
        with self._session_state_lock:
            workflow_path = self.workflow_path
        if not workflow_path:
            return ""
        return launcher.server_log_path(workflow_path)

    def resolved_train_log_path(
        self,
        workflow_path: Optional[str] = None,
        *,
        remembered: Optional[str] = None,
    ) -> str:
        """Most recent train log this server knows about, or ``""``.

        Preference: the path recorded by :meth:`start_train`, then the
        selected workflow's conventional log when a train is running or the
        file is already on disk.
        """
        if workflow_path is None or remembered is None:
            with self._session_state_lock:
                if workflow_path is None:
                    workflow_path = self.workflow_path or ""
                if remembered is None:
                    remembered = self.train_log_path or ""
        workflow_path = workflow_path or ""
        remembered = remembered or ""
        if remembered:
            return remembered
        if workflow_path:
            _pid_path, log_path = launcher.train_artifact_paths(
                workflow_path, create=False
            )
            if launcher.is_train_running(workflow_path) or os.path.isfile(log_path):
                return log_path
        return ""

    def session_payload(self) -> dict[str, Any]:
        """What the SPA needs to run without asking the developer anything.
        Reflects the LIVE child state — the SPA polls this, so a server that
        dies mid-session is reported honestly, not as 'running'."""
        with self._session_state_lock:
            workflow_path = self.workflow_path
            db_path = self.db_path
            server_proc = self.server_proc
            server_url = self.server_url
            server_note = self.server_note
            env_setup_required = self.env_setup_required
            env_file_path = self.env_file_path
            passwords_file_path = self.passwords_file_path
            spawn_error = self.spawn_error
            workspace = self.workspace
            channel_id = self.channel_id
            user_id = self.user_id
            train_log_path = self.train_log_path
            expect_encrypted_jwt = bool(self.spawn_options.get("expect_encrypted_jwt"))
            no_server = bool(self.spawn_options.get("no_server"))
        running = server_proc is not None and server_proc.poll() is None
        resolved_train_log = self.resolved_train_log_path(
            workflow_path, remembered=train_log_path
        )
        # An omitted spawn still reports server_url when --server-port named an
        # existing server, so the Advanced panel can be prefilled.
        expose_url = running or bool(no_server and server_url)
        payload = {
            "workflow_path": workflow_path,
            "workflow_name": (
                os.path.basename(os.path.abspath(workflow_path))
                if workflow_path
                else ""
            ),
            "db_path": db_path,
            "server_url": server_url if expose_url else None,
            "server_running": running,
            "server_exit_code": (
                server_proc.returncode
                if server_proc is not None and not running
                else None
            ),
            "server_note": server_note,
            "env_setup_required": env_setup_required,
            "env_file_path": env_file_path or None,
            "passwords_file_path": passwords_file_path or None,
            "channel_id": channel_id,
            "user_id": user_id,
            "jwt_mode": ("signed" if expect_encrypted_jwt else "unsigned"),
            "spawn_error": spawn_error,
            "server_log_path": (
                launcher.server_log_path(workflow_path) if workflow_path else None
            ),
            "train_log_path": resolved_train_log or None,
        }
        if workspace is not None:
            payload.update(
                {
                    "workspace_mode": True,
                    "workspace": workspace.summary(),
                    "workflow_path": "",
                    "workflow_name": "",
                    "db_path": "",
                    "server_url": None,
                    "server_running": False,
                    "env_setup_required": False,
                    "read_only": True,
                    "server_log_path": None,
                    "train_log_path": None,
                }
            )
        else:
            payload.update({"workspace_mode": False, "read_only": False})
        return payload

    def activate_workspace(self, manifest_path: str) -> dict[str, Any]:
        """Load a manifest selected through the token-gated browser picker."""
        with self._activate_lock:
            workspace = load_observability_workspace(manifest_path)
            with self._session_state_lock:
                old_proc = self.server_proc
            if old_proc is not None and old_proc.poll() is None:
                launcher.terminate_server(old_proc)
            self._publish_session_state(
                server_proc=None,
                server_url=None,
                workflow_path="",
                db_path="",
                workspace=workspace,
                workspace_manifest_path=str(workspace.manifest_path),
            )
            self._review_sidecar = None
            return self.session_payload()

    def activate_workflow(self, workflow_path: str) -> dict[str, Any]:
        """Point the chatbot at a workflow and (unless disabled) make sure its
        FastAPI server is running. Selecting a different workflow replaces the
        spawned server. Never raises: failures land in ``spawn_error`` and the
        chatbot stays usable as a trace viewer."""
        from fastworkflow import state_paths

        with self._activate_lock:
            workflow_path = os.path.abspath(workflow_path)
            db_path = state_paths.observability_db(workflow_path)
            with self._session_state_lock:
                same_workflow = (
                    os.path.abspath(self.workflow_path or "") == workflow_path
                )
                old_proc = self.server_proc

            if self.spawn_options.get("no_server"):
                external = self.spawn_options.get("server_port")
                self._publish_session_state(
                    workflow_path=workflow_path,
                    db_path=db_path,
                    spawn_error=None,
                    server_note=None,
                    server_url=(
                        f"http://127.0.0.1:{int(external)}" if external else None
                    ),
                )
                return self.session_payload()
            if (
                same_workflow
                and old_proc is not None
                and old_proc.poll() is None
            ):
                return self.session_payload()  # already serving this workflow

            if old_proc is not None and old_proc.poll() is None:
                launcher.terminate_server(old_proc)

            spawn_error: Optional[str] = None
            server_note: Optional[str] = None
            server_proc = None
            server_url: Optional[str] = None

            env_file = self.spawn_options.get("env_file_path") or ""
            passwords_file = self.spawn_options.get("passwords_file_path") or ""
            if not env_file or not passwords_file:
                auto_env, auto_passwords = _autodetect_env_files(workflow_path)
                env_file = env_file or auto_env
                passwords_file = passwords_file or auto_passwords
            env_setup_required = not (
                env_file
                and passwords_file
                and os.path.isfile(env_file)
                and os.path.isfile(passwords_file)
            )
            if env_setup_required:
                self._publish_session_state(
                    workflow_path=workflow_path,
                    db_path=db_path,
                    server_proc=None,
                    server_url=None,
                    env_file_path=env_file,
                    passwords_file_path=passwords_file,
                    env_setup_required=True,
                    spawn_error=None,
                    server_note=None,
                )
                return self.session_payload()

            preferred_port = int(
                self.spawn_options.get("server_port") or PREFERRED_SPAWN_PORT
            )
            server_port, moved = _free_server_port(preferred_port)
            server_note = (
                f"port {preferred_port} was busy; the server runs on {server_port} instead"
                if moved
                else None
            )
            if moved:
                logger.warning(
                    f"Chatbot server port {preferred_port} is busy; "
                    f"spawning the FastAPI server on {server_port} instead"
                )

            expect_encrypted = bool(self.spawn_options.get("expect_encrypted_jwt"))
            parent_pid = os.getpid()

            def _plan_for(port: int) -> Any:
                return launcher.plan_server_spawn(
                    workflow_path=workflow_path,
                    env_file_path=env_file,
                    passwords_file_path=passwords_file,
                    chatbot_origin=f"http://127.0.0.1:{self.port}",
                    server_port=port,
                    expect_encrypted_jwt=expect_encrypted,
                    # Loopback-only + loopback-pinned CORS + a chatbot that mints
                    # its own tokens via /initialize: unsigned dev JWTs are the
                    # default posture for the AUTO-spawned server (owner decision
                    # amending R19's opt-in flag; --expect-encrypted-jwt restores
                    # signed mode).
                    allow_unsigned_jwt=not expect_encrypted,
                    parent_pid=parent_pid,
                )

            plan = _plan_for(server_port)
            if not plan.ok:
                self._publish_session_state(
                    workflow_path=workflow_path,
                    db_path=db_path,
                    server_proc=None,
                    server_url=None,
                    env_file_path=env_file,
                    passwords_file_path=passwords_file,
                    env_setup_required=False,
                    spawn_error=plan.reason,
                    server_note=server_note,
                )
                return self.session_payload()

            try:
                server_proc = launcher.spawn_server(plan)
            except OSError as exc:
                self._publish_session_state(
                    workflow_path=workflow_path,
                    db_path=db_path,
                    server_proc=None,
                    server_url=None,
                    env_file_path=env_file,
                    passwords_file_path=passwords_file,
                    env_setup_required=False,
                    spawn_error=f"could not start the FastAPI server: {exc}",
                    server_note=server_note,
                )
                return self.session_payload()

            probe_s = float(self.spawn_options.get("startup_probe_seconds", 1.0))
            if not _child_survived_startup(server_proc, timeout_s=probe_s):
                died_msg = _child_exited_message(server_proc.returncode)
                # Child died at startup and something still holds the port:
                # the probe→close TOCTOU likely lost the race; retry once on a
                # fresh ephemeral port.
                if _port_in_use(server_port):
                    retry_port, _ = _free_server_port(0)
                    server_note = (
                        f"port {preferred_port} was busy; the server runs on "
                        f"{retry_port} instead"
                    )
                    logger.warning(
                        "Chatbot FastAPI spawn died with port %s still taken; "
                        "retrying once on %s",
                        server_port,
                        retry_port,
                    )
                    plan = _plan_for(retry_port)
                    if plan.ok:
                        server_proc, spawn_error = _spawn_and_probe(plan, probe_s)
                        if server_proc is not None:
                            server_url = plan.server_url
                    else:
                        server_proc = None
                        spawn_error = plan.reason
                else:
                    server_proc = None
                    spawn_error = died_msg
            else:
                server_url = plan.server_url

            self._publish_session_state(
                workflow_path=workflow_path,
                db_path=db_path,
                server_proc=server_proc,
                server_url=server_url,
                env_file_path=env_file,
                passwords_file_path=passwords_file,
                env_setup_required=False,
                spawn_error=spawn_error,
                server_note=server_note,
            )
            return self.session_payload()

    def configure_env_files(
        self,
        *,
        env_content: Optional[str] = None,
        passwords_content: Optional[str] = None,
        create_from_templates: bool = False,
    ) -> dict[str, Any]:
        """Install missing workflow-local env files, then activate the workflow."""
        if not self.workflow_path:
            raise ValueError("select a workflow before configuring env files")
        if launcher.is_bundled_example_path(self.workflow_path):
            # Writing into the packaged examples dir would land a passwords
            # file in site-packages — or, in a repo checkout, in a directory
            # git does not ignore. The bundled examples read the shared
            # examples/fastworkflow*.env templates instead.
            raise ValueError(
                "bundled examples cannot take workflow-local env files; copy "
                "the example to your own folder first, or edit the shared "
                "templates beside the examples directory"
            )
        max_bytes = 512 * 1024
        for label, content in (
            ("environment", env_content),
            ("passwords", passwords_content),
        ):
            if content is not None and not isinstance(content, str):
                raise TypeError(f"{label} file content must be text")
            if content is not None and len(content.encode("utf-8")) > max_bytes:
                raise ValueError(f"{label} file is larger than {max_bytes} bytes")

        env_target = os.path.join(self.workflow_path, "fastworkflow.env")
        passwords_target = os.path.join(
            self.workflow_path, "fastworkflow.passwords.env"
        )
        if create_from_templates:
            if not os.path.isfile(env_target):
                _write_env_file(env_target, _env_template_text("fastworkflow.env"))
            if not os.path.isfile(passwords_target):
                _write_env_file(
                    passwords_target,
                    _env_template_text("fastworkflow.passwords.env"),
                )
        if env_content is not None:
            _write_env_file(env_target, env_content)
        if passwords_content is not None:
            _write_env_file(passwords_target, passwords_content)

        self.spawn_options["env_file_path"] = ""
        self.spawn_options["passwords_file_path"] = ""
        return self.activate_workflow(self.workflow_path)

    def start_train(self, workflow_path: str) -> dict[str, Any]:
        """Spawn a detached ``fastworkflow train`` and return immediately.

        The child outlives this chatbot process. Shutdown does not signal it.
        Status is the pid file + ``_workflow_is_trained`` on later polls.
        """
        workflow_path = os.path.abspath(workflow_path)
        with self._train_lock:
            env_file = self.spawn_options.get("env_file_path") or ""
            passwords_file = self.spawn_options.get("passwords_file_path") or ""
            if not env_file or not passwords_file:
                auto_env, auto_passwords = _autodetect_env_files(workflow_path)
                env_file = env_file or auto_env
                passwords_file = passwords_file or auto_passwords
            plan = launcher.plan_train_spawn(
                workflow_path=workflow_path,
                env_file_path=env_file,
                passwords_file_path=passwords_file,
                already_trained=_workflow_is_trained(workflow_path),
            )
            if not plan.ok:
                raise ValueError(plan.reason)
            pid = launcher.spawn_detached_train(plan)
            self._publish_session_state(train_log_path=plan.log_path)
            return {
                "ok": True,
                "pid": pid,
                "training": True,
                "trained": False,
                "log_path": plan.log_path,
            }

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.server_proc is not None and self.server_proc.poll() is None:
            from fastworkflow.run_chatbot import launcher

            launcher.terminate_server(self.server_proc)
            self.server_proc = None


# Write dispatch. The 405 allowlist is this table: a path is admitted exactly
# when one row matches, and every row names a handler, so an admitted path
# cannot fall through into clear-conversations. `allowed_in_workspace=False`
# is the refusal that today sits after the earlier POST branches (and before
# both PATCH handlers); routes left True refuse inside their own handlers.
_LIVE_WORKSPACE_REFUSAL = (
    "workspace mode is read-only; live and destructive actions are disabled"
)
_PATCH_WORKSPACE_REFUSAL = (
    "workspace mode is read-only; experiment annotations cannot be changed"
)


class _WriteRoute(NamedTuple):
    method: str
    match: Callable[[str], bool]
    handler: str
    allowed_in_workspace: bool = True
    workspace_refusal: str = ""
    require_object_body: bool = False
    reads_body: bool = True


def _exact(expected: str) -> Callable[[str], bool]:
    def match(path: str, expected: str = expected) -> bool:
        return path == expected

    return match


def _benchmark_experiment_post(path: str) -> bool:
    return path.startswith("/api/benchmarks/") and path.endswith("/experiments")


def _benchmark_version_post(path: str) -> bool:
    return path == "/api/benchmarks" or (
        path.startswith("/api/benchmarks/") and path.endswith("/versions")
    )


def _setup_post(path: str) -> bool:
    return path == "/api/experiment-setups" or path.startswith(
        "/api/experiment-setups/"
    )


def _review_capture_post(path: str) -> bool:
    return path.startswith("/api/review/assignments/") and (
        path.endswith("/answers") or path.endswith("/adjudications")
    )


def _benchmark_analysis_put(path: str) -> bool:
    return path.startswith("/api/benchmarks/") and path.endswith("/analysis")


def _selection_delete(path: str) -> bool:
    return path.startswith(selection_api.EXPERIMENTS_PREFIX)


def _benchmark_registration_prefix(path: str) -> bool:
    return path.startswith("/api/benchmark-experiments/")


def _experiment_patch(path: str) -> bool:
    return path.startswith("/api/experiment/")


def _checked_write_routes(
    routes: tuple[_WriteRoute, ...],
) -> tuple[_WriteRoute, ...]:
    """A row with no handler cannot exist; the 405 set is these matchers."""
    for route in routes:
        if not route.handler:
            raise RuntimeError(f"{route.method} write route has no handler")
        if not route.allowed_in_workspace and not route.workspace_refusal:
            raise RuntimeError(
                f"{route.method} {route.handler} refuses workspace mode "
                "without a message"
            )
    return routes


_WRITE_ROUTES = _checked_write_routes((
    # The one write surface for recorded review notes (fix-9eg.16,
    # owner wording). Reads live on GET /api/feedback-notes and
    # GET /api/task-feedback, so a read is never spelled as a post.
    _WriteRoute("POST", _exact("/post_feedback"), "_post_feedback_note"),
    _WriteRoute("POST", _exact("/api/benchmark-setup"), "_post_benchmark_record"),
    _WriteRoute("POST", _benchmark_experiment_post, "_post_benchmark_record"),
    _WriteRoute("POST", selection_api.owns_write, "_post_selection"),
    _WriteRoute("POST", _setup_post, "_post_experiment_setup"),
    _WriteRoute("POST", _review_capture_post, "_post_review_capture"),
    _WriteRoute("POST", _exact("/api/review/assignments"), "_post_review_assignment"),
    _WriteRoute("POST", _benchmark_version_post, "_post_benchmark_version"),
    # Dispatched only after the workspace read-only refusal below. The rows
    # above are reached first, and each of those handlers does its own check.
    _WriteRoute(
        "POST",
        _exact("/api/select_workspace"),
        "_post_select_workspace",
        allowed_in_workspace=False,
        workspace_refusal=_LIVE_WORKSPACE_REFUSAL,
    ),
    _WriteRoute(
        "POST",
        _exact("/api/select_workflow"),
        "_post_select_workflow",
        allowed_in_workspace=False,
        workspace_refusal=_LIVE_WORKSPACE_REFUSAL,
    ),
    _WriteRoute(
        "POST",
        _exact("/api/configure_env"),
        "_post_configure_env",
        allowed_in_workspace=False,
        workspace_refusal=_LIVE_WORKSPACE_REFUSAL,
    ),
    _WriteRoute(
        "POST",
        _exact("/api/train"),
        "_post_train",
        allowed_in_workspace=False,
        workspace_refusal=_LIVE_WORKSPACE_REFUSAL,
    ),
    _WriteRoute(
        "POST",
        _exact("/api/clear_conversations"),
        "_post_clear_conversations",
        allowed_in_workspace=False,
        workspace_refusal=_LIVE_WORKSPACE_REFUSAL,
    ),
    _WriteRoute(
        "PUT",
        _benchmark_analysis_put,
        "_put_benchmark_analysis",
        require_object_body=True,
    ),
    _WriteRoute("DELETE", _selection_delete, "_delete_selection"),
    _WriteRoute(
        "DELETE",
        _benchmark_registration_prefix,
        "_delete_benchmark_experiment",
        reads_body=False,
    ),
    _WriteRoute(
        "PATCH",
        _benchmark_registration_prefix,
        "_patch_registration",
        allowed_in_workspace=False,
        workspace_refusal=_PATCH_WORKSPACE_REFUSAL,
        require_object_body=True,
    ),
    _WriteRoute(
        "PATCH",
        _experiment_patch,
        "_patch_experiment",
        allowed_in_workspace=False,
        workspace_refusal=_PATCH_WORKSPACE_REFUSAL,
        require_object_body=True,
    ),
))


def _match_write_route(method: str, path: str) -> Optional[_WriteRoute]:
    for route in _WRITE_ROUTES:
        if route.method == method and route.match(path):
            return route
    return None


class _ChatbotRequestHandler(
    _ReviewRoutes,
    _WorkspaceRoutes,
    _ExperimentRoutes,
    _BenchmarkRoutes,
    _FeedbackRoutes,
    _NavigationRoutes,
    _ControlPlaneRoutes,
    BaseHTTPRequestHandler,
):
    """Token-gated request handler. Observability queries are GET-only;
    explicit control-plane POSTs select a workflow, configure env, start
    train, or clear recorded conversations."""

    chatbot: ChatbotServer  # bound by ChatbotServer.__init__
    protocol_version = "HTTP/1.1"
    server_version = "fastWorkflowChatbot"
    sys_version = ""

    # -- plumbing --------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Quiet by default — the terminal belongs to the launch banner.
        # Opt-in access lines are emitted from ``_end_request`` instead so they
        # can include duration; BaseHTTPRequestHandler's log_request has none.
        pass

    def _redacted_path(self) -> str:
        """Request path with ``token`` query values scrubbed for logs."""
        split = urlsplit(self.path)
        if not split.query:
            return split.path or self.path
        parts: list[str] = []
        for key, values in parse_qs(split.query, keep_blank_values=True).items():
            if key == "token":
                parts.append("token=REDACTED")
                continue
            for value in values:
                parts.append(f"{key}={value}")
        return f"{split.path}?{'&'.join(parts)}"

    def _begin_request(self) -> float:
        self._response_status = 0
        return time.perf_counter()

    def _end_request(self, started: float) -> None:
        if os.environ.get("FASTWORKFLOW_CHATBOT_ACCESS_LOG") != "1":
            return
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "%s %s %s %.1fms",
            self.command,
            self._redacted_path(),
            getattr(self, "_response_status", 0),
            duration_ms,
        )

    def _report_internal_error(self, exc: BaseException) -> None:
        error_id = secrets.token_hex(4)
        logger.exception(
            "chatbot %s %s failed (error_id=%s)",
            self.command,
            self._redacted_path(),
            error_id,
        )
        try:
            self._error(
                500,
                f"internal error: {type(exc).__name__}",
                error_id=error_id,
            )
        except Exception:
            pass

    def _read_json_body(
        self, max_bytes: int = _MAX_JSON_BODY_BYTES
    ) -> Optional[Any]:
        """Parse a JSON body, or send an error response and return None.

        Absent Content-Length with no Transfer-Encoding is an empty body
        (``{}``), matching prior bodyless-POST behaviour. Chunked transfer
        encodings are refused with 411; oversize declared lengths with 413
        without reading the body.
        """
        transfer = (self.headers.get("Transfer-Encoding") or "").strip().lower()
        if transfer and transfer != "identity":
            self._error(411, "Content-Length required")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or raw_length == "":
            length = 0
        else:
            try:
                length = int(raw_length)
            except (TypeError, ValueError):
                self._error(400, "invalid Content-Length")
                return None
            if length < 0:
                self._error(400, "invalid Content-Length")
                return None
            if length > max_bytes:
                self._error(413, "request body too large")
                return None
        try:
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw or b"{}")
        except (ValueError, TypeError):
            self._error(400, "invalid JSON body")
            return None

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._response_status = status
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(
        self, status: int, message: str, error_id: Optional[str] = None
    ) -> None:
        payload: dict[str, Any] = {"error": message}
        if error_id is not None:
            payload["error_id"] = error_id
        self._send_json(payload, status=status)

    # -- access control [R5][R18] ---------------------------------------
    #
    # The allowlist admits any LOOPBACK authority — 127.0.0.1 / localhost /
    # [::1], any port — and nothing else. Loopback-only is what defeats DNS
    # rebinding (a rebound request arrives with the attacker's hostname in
    # Host); the port is deliberately NOT pinned, because port forwarders
    # (VS Code Remote / WSL relays) legitimately re-expose the server on a
    # different local port and the browser's Host names THAT port. The bearer
    # token remains the authentication on every request either way.

    @staticmethod
    def _is_loopback_authority(authority: str) -> bool:
        authority = authority.strip().lower()
        if not authority:
            return False
        if authority.startswith("["):  # bracketed IPv6, e.g. [::1]:8901
            hostname = authority.split("]", 1)[0].lstrip("[")
        else:
            hostname = authority.rsplit(":", 1)[0] if ":" in authority else authority
        return hostname in ("127.0.0.1", "localhost", "::1")

    def _host_origin_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not self._is_loopback_authority(host):
            logger.warning(
                f"Chatbot refused a request with non-loopback Host {host!r} [R18]"
            )
            return False
        origin = (self.headers.get("Origin") or "").strip().lower()
        if origin:
            scheme, sep, authority = origin.partition("://")
            if (
                scheme != "http"
                or not sep
                or not self._is_loopback_authority(authority)
            ):
                logger.warning(
                    f"Chatbot refused a request with non-loopback Origin {origin!r} [R18]"
                )
                return False
        return True

    def _token_valid(self, query: dict[str, list[str]]) -> bool:
        presented = ""
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            presented = auth[len("Bearer ") :].strip()
        elif query.get("token"):
            presented = query["token"][0]
        return hmac.compare_digest(
            presented.encode("utf-8"), self.chatbot.token.encode("utf-8")
        )

    def _gate(self, query: dict[str, list[str]], *, forbidden: str) -> bool:
        """Host/Origin, then the bearer token. False means 403 or 401 was sent.

        ``forbidden`` is the 403 text. GET quotes the rejected Host and Origin;
        writes keep the short refusal. The two texts stay distinct.
        """
        if not self._host_origin_allowed():
            self._error(403, forbidden)
            return False
        # EVERY request is token-gated, the page included (Jupyter pattern).
        if not self._token_valid(query):
            self._error(401, "unauthorized: missing or invalid token")
            return False
        return True

    def _review_capability(self) -> str:
        """Return the separately presented rater capability."""
        return (self.headers.get("X-Review-Capability") or "").strip()

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        started = self._begin_request()
        try:
            self._handle_get()
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._report_internal_error(exc)
        finally:
            self._end_request(started)

    def _handle_get(self) -> None:
        split = urlsplit(self.path)
        path = split.path
        query = parse_qs(split.query)

        if not self._gate(
            query,
            forbidden=(
                "forbidden: only loopback hosts (127.0.0.1 / localhost / [::1]) "
                f"may access the chatbot; got Host={self.headers.get('Host')!r}, "
                f"Origin={self.headers.get('Origin')!r}"
            ),
        ):
            return

        if path in ("/", "/index.html"):
            self._send(
                200,
                self.chatbot.index_html,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": self.chatbot.page_csp},
            )
            return
        if path == "/trace" or path.startswith("/trace/"):
            self._handle_trace_navigation(path, query)
            return
        if path.startswith("/api/"):
            self._handle_api(path, query)
            return
        self._error(404, "not found")

    # Writes: ordinary observability browsing stays read-only. The explicit
    # control-plane POSTs (select workflow, configure env, train, clear
    # conversations) carry the same host/origin + token gates as GETs.
    def _refuse_write(self) -> None:
        self._send_json(
            {"error": "method not allowed: observability data is read-only"}, 405
        )

    def _dispatch_write(self, method: str) -> None:
        """Host/Origin, then the bearer token, then the route table.

        The 405 allowlist is ``_WRITE_ROUTES``. A matched row always names a
        handler, so nothing allowlisted can fall through into clear-conversations.
        An unknown write from a bad host or a missing token is 403 or 401,
        not 405 (owner decision 2026-09-30, fix-hzux.10).
        """
        split = urlsplit(self.path)
        path = split.path
        query = parse_qs(split.query)
        if not self._gate(query, forbidden="forbidden: host/origin not allowed"):
            return
        route = _match_write_route(method, path)
        if route is None:
            self._refuse_write()
            return
        body = None
        if route.reads_body:
            body = self._read_json_body()
            if body is None:
                return
        if route.require_object_body and not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        if (
            not route.allowed_in_workspace
            and self.chatbot.workspace is not None
        ):
            self._error(403, route.workspace_refusal)
            return
        getattr(self, route.handler)(path, body, query)

    def do_POST(self) -> None:  # noqa: N802
        started = self._begin_request()
        try:
            self._dispatch_write("POST")
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._report_internal_error(exc)
        finally:
            self._end_request(started)

    def do_PUT(self) -> None:  # noqa: N802
        """Admitted PUT: the benchmark's sibling analysis file."""
        started = self._begin_request()
        try:
            self._dispatch_write("PUT")
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._report_internal_error(exc)
        finally:
            self._end_request(started)

    def do_DELETE(self) -> None:  # noqa: N802
        started = self._begin_request()
        try:
            self._dispatch_write("DELETE")
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._report_internal_error(exc)
        finally:
            self._end_request(started)

    def do_PATCH(self) -> None:  # noqa: N802
        """Two admitted PATCH surfaces: an experiment's editable annotations,
        and the author's description on a registration not yet claimed.

        The Host/Origin and bearer-token gates are applied through the shared
        `_gate` chokepoint -- `_handle_get` and every write verb call it before
        route lookup -- rather than each verb repeating its own. A do_PATCH written
        without them would be an ungated cross-origin write.
        """
        started = self._begin_request()
        try:
            self._dispatch_write("PATCH")
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._report_internal_error(exc)
        finally:
            self._end_request(started)

    # -- API endpoints ---------------------------------------------------

    def _handle_process_log(self, path: str, query: dict[str, list[str]]) -> None:
        """``GET /api/logs/server`` and ``GET /api/logs/train``.

        Reached only from :meth:`_handle_api`, which :meth:`_handle_get`
        calls after the single Host and token gate. A missing log is
        ``exists: false``, not a 404.
        """
        tail, error = _parse_log_tail(query)
        if error:
            self._error(400, error)
            return
        if path == "/api/logs/server":
            log_path = self.chatbot.resolved_server_log_path()
        else:
            log_path = self.chatbot.resolved_train_log_path()
        self._send_json(
            read_log_tail(log_path, tail, self.chatbot.token)
        )

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        # Authoring and navigation do not depend on execution evidence. In
        # particular, an incompatible selected store must not trap the user
        # by breaking session loading and the workflow picker.
        q = lambda name: query.get(name, [None])[0]  # noqa: E731
        if path == "/api/navigation":
            self._handle_navigation()
            return
        if path == "/api/feedback-notes":
            self._handle_feedback_notes(query)
            return
        if path == "/api/feedback-taxonomy":
            # One source for the composer's selectors and for a coding agent
            # that wants to know the enum values before posting.
            self._send_json(feedback.taxonomy_payload())
            return
        if path.startswith(selection_api.EXPERIMENTS_PREFIX):
            self._handle_selection_api("GET", path, query=query)
            return
        if path.startswith("/api/benchmark-experiments/"):
            self._handle_benchmark_registration(unquote(path[len("/api/benchmark-experiments/"):]))
            return
        if path == "/api/session":
            self._send_json({"session": self.chatbot.session_payload()})
            return
        if path in ("/api/logs/server", "/api/logs/train"):
            self._handle_process_log(path, query)
            return
        if path == "/api/workflows":
            self._send_json({"workflows": list_workflow_candidates()})
            return
        if path == "/api/browse":
            self._send_json(browse_directories(q("dir") or ""))
            return
        if path == "/api/experiment-setups" or path.startswith("/api/experiment-setups/"):
            self._handle_setup(path)
            return
        if path == "/api/benchmarks" or path.startswith("/api/benchmarks/"):
            self._handle_benchmarks(path)
            return

        # Per-request read-only store; never migrate incompatible evidence.
        try:
            store = self.chatbot.open_store()
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        if path.startswith("/api/review/assignments/"):
            self._handle_review_assignment(path)
        elif path == "/api/workspace" or path.startswith("/api/workspace/"):
            self._handle_workspace(path, q)
        elif self.chatbot.workspace is not None and path in {
            "/api/turns",
            "/api/experiments",
        }:
            self._error(
                400,
                "workspace reads must be scoped by store_id; unscoped search is "
                "refused. Search one store with "
                "/api/workspace/turns?store_id=<id>",
            )
        elif self.chatbot.workspace is not None and (
            path == "/api/training-runs" or path.startswith("/api/training-run/")
        ):
            # Same rule as turns: a workspace holds several stores and two of
            # them may hold the same run_id, so an unscoped read would have to
            # pick one. It names the scoped route instead of guessing.
            self._error(
                400,
                "workspace training-history reads must name their store; list "
                "one store with /api/workspace/training-runs?store_id=<id>",
            )
        elif self.chatbot.workspace is not None and (
            path.startswith("/api/turn/")
            or path.startswith("/api/spans/")
            or path.startswith("/api/experiment/")
        ):
            self._error(
                400,
                "workspace reads must use the store-aware /api/workspace routes",
            )
        elif path == "/api/meta":
            self._send_json(
                {
                    "workflow_path": self.chatbot.workflow_path,
                    "workflow_name": (
                        self.chatbot.workflow_path.rstrip("/\\").rsplit("/", 1)[-1]
                        if self.chatbot.workflow_path
                        else ""
                    ),
                    "db_path": self.chatbot.db_path,
                    "db_available": store is not None,
                    "db_size_bytes": store.db_size_bytes() if store else 0,
                }
            )
        elif store is None:
            if path == "/api/health":
                self._send_json(
                    {"writer_health": None, "db_size_bytes": 0, "db_available": False}
                )
            elif path in ("/api/channels", "/api/conversations", "/api/turns"):
                self._send_json({"channels": [], "conversations": [], "turns": []})
            elif path == "/api/experiments":
                # An empty state, not "observability DB not found": a cold start
                # has no experiments, which is a fact about the DB rather than
                # an error the operator can act on.
                self._send_json({"experiments": []})
            elif path == "/api/training-runs":
                # The same argument for the same reason: a workflow nobody has
                # trained through this store recorded no training runs.
                self._send_json({"training_runs": []})
            else:
                self._error(404, "observability DB not found")
        elif path == "/api/channels":
            self._send_json({"channels": store.list_channels()})
        elif path == "/api/conversations":
            self._send_json(
                {
                    "conversations": store.list_conversations(
                        channel_id=q("channel"),
                        limit=self._int(q("limit"), 100),
                        offset=self._int(q("offset"), 0),
                    )
                }
            )
        elif path == "/api/turns":
            # Every filter -- the store's own, the marker predicates and the
            # low-confidence test -- is applied by `search_turns` to the whole
            # authorized dataset. This route used to list one page and then drop
            # rows from it, so a turn matching on page four was reported as no
            # match at all (fix-9eg.18.1); the filtering now happens before the
            # page is cut rather than after.
            self._search_turns_response(store, q)
        elif path.startswith("/api/turn/"):
            turn_key = path[len("/api/turn/") :]
            turn = store.get_turn(turn_key)
            if turn is None:
                self._error(404, "turn not found")
                return
            try:
                turn["record"] = json.loads(turn.pop("record_json"))
            except (ValueError, KeyError):
                turn["record"] = None
            spans = store.get_spans(turn_key)
            annotate_turn_detail(turn, spans)
            try:
                annotate_turn_diagnosis(
                    turn,
                    spans,
                    store_id=diagnostic_store_id(store),
                    low_confidence_below=_wire_number(
                        q("low_confidence_below"), "low_confidence_below"
                    ),
                )
            except InvalidTurnQuery as exc:
                self._error(400, str(exc))
                return
            self._send_json({"turn": turn})
        # GET /api/feedback and /api/feedback/<turn_key> read the agent-memory
        # `feedback` table and were removed with it (fix-9eg.16). They are not
        # re-pointed at review notes: a client still calling them is asking
        # for something that no longer exists, and a 404 says so, where a
        # silently different payload would not.
        elif path == "/api/task-feedback":
            self._handle_task_feedback(store, query)
        elif path.startswith("/api/spans/"):
            trace_id = path[len("/api/spans/") :]
            spans = store.get_spans(trace_id)
            for span in spans:
                try:
                    span["attributes"] = json.loads(span["attributes"])
                except (ValueError, TypeError, KeyError):
                    pass
            self._send_json({"spans": spans})
        elif path == "/api/experiments" or path.startswith("/api/experiment/"):
            self._handle_experiments(store, path, q)
        elif path.startswith("/api/artifact/"):
            self._serve_artifact(store, path[len("/api/artifact/") :])
        elif path == "/api/training-runs" or path.startswith("/api/training-run/"):
            self._handle_training_history(store, path, q)
        elif path == "/api/health":
            self._send_json(
                {
                    "writer_health": store.writer_health(),
                    "db_size_bytes": store.db_size_bytes(),
                    "db_available": True,
                }
            )
        else:
            self._error(404, "not found")

    def _serve_artifact(
        self, store: ReadOnlyObservabilityStore, artifact_id: str
    ) -> None:
        """Offloaded artifact content, with its stored content-type.

        HTML-ish content is only ever *rendered* inside a sandboxed iframe by
        the SPA; the raw response is additionally neutralized with
        ``CSP: default-src 'none'; sandbox`` so navigating to the URL directly
        cannot run scripts either.
        """
        artifact = store.get_artifact(artifact_id)
        if artifact is None:
            self._error(404, "artifact not found")
            return
        content_type = artifact.get("content_type") or "application/octet-stream"
        value = artifact.get("inline_value") or b""
        if isinstance(value, str):
            value = value.encode("utf-8")
        base_type = content_type.split(";")[0].strip().lower()
        headers = {
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Disposition": "inline",
        }
        if base_type in _HTMLISH_TYPES:
            headers["X-FW-Artifact-Htmlish"] = "1"
        self._send(200, bytes(value), content_type, headers)

    @staticmethod
    def _float_or_none(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _int(value: Optional[str], default: Any) -> Any:
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default


def _require_write_handlers() -> None:
    for route in _WRITE_ROUTES:
        if not hasattr(_ChatbotRequestHandler, route.handler):
            raise RuntimeError(
                f"{route.method} write route names missing handler {route.handler}"
            )


_require_write_handlers()


# ----------------------------------------------------------------------
# CLI entry points (used by `fastworkflow run_chatbot`; kept import-light)
# ----------------------------------------------------------------------


def _open_in_browser(url: str) -> None:
    """Open the user's default browser; never noisy, never fatal.

    On WSL there is usually no Linux browser — stdlib ``webbrowser`` falls
    through to xdg-open, which sprays a 'not found' line per candidate and
    gives up — while the WINDOWS default browser is one hop away. Prefer
    ``wslview`` (wslu) then ``powershell.exe Start-Process``; the printed URL
    in the banner is always the fallback.
    """
    import shutil
    import subprocess

    if "PYTEST_CURRENT_TEST" in os.environ:
        return  # pytest is the only skip path; there is no --no-browser flag

    is_wsl = False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass
    if is_wsl:
        for cmd in (
            ["wslview", url],
            # The token is token_urlsafe (A-Za-z0-9_-), so the single-quoted
            # PowerShell literal cannot be escaped out of.
            ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
        ):
            if shutil.which(cmd[0]) is None:
                continue
            try:
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
            except OSError:
                continue
        return  # no opener available; the banner URL is the path
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_prune(db_path: str) -> dict[str, int]:
    """Library maintenance utility: bounded prune + vacuum.

    Not wired to any CLI flag or HTTP route — pruning runs automatically at
    sink startup; this exists for scripts/tests that need it on demand.
    """
    return ObservabilityStore(db_path).prune()


def run_forget_channel(
    db_path: str, channel_id: str, workflow_path: str = ""
) -> dict[str, int]:
    """Library erasure utility: delete one channel everywhere.

    Not wired to any CLI flag or HTTP route — the chatbot UI exposes the
    all-channel Clear-conversations action instead; this remains the
    single-channel primitive for scripts/tests (e.g. a deletion request for
    one API channel). Also deletes the LEGACY per-channel conversation DB
    (``conversations/<channel_id>.sqlite3`` + sidecars) while the Phase-A
    dual-write period lasts — without this, "forgotten" conversations remain
    fully readable in the legacy store.

    The channel's offload evidence -- the archived execute responses its
    turns produced -- lives in the same database and is erased by
    ``forget_channel`` in the same transaction as its turn records, experiment
    runs included; there is no second evidence file to sweep.
    """
    deleted = ObservabilityStore(db_path).forget_channel(channel_id)
    if workflow_path and channel_id == os.path.basename(channel_id):
        legacy_db = os.path.join(
            state_paths.conversations_dir(workflow_path), f"{channel_id}.sqlite3"
        )
        removed = 0
        for path in (legacy_db, f"{legacy_db}-wal", f"{legacy_db}-shm"):
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
        deleted["legacy_conversation_db_files"] = removed
    return deleted


PREFERRED_SPAWN_PORT = 8000


def spawn_options_from_cli_args(args) -> dict:
    """Map ``run_chatbot`` CLI flags to ChatbotServer spawn_options.

    Passing ``--server-port`` means an existing FastAPI server: do not spawn.
    Omitting it auto-spawns a loopback server (preferred port
    ``PREFERRED_SPAWN_PORT``; a busy port still moves at activate time).
    """
    external_port = getattr(args, "server_port", None)
    return {
        "no_server": external_port is not None,
        "server_port": (
            int(external_port) if external_port is not None else PREFERRED_SPAWN_PORT
        ),
        "expect_encrypted_jwt": bool(getattr(args, "expect_encrypted_jwt", False)),
    }


def run_chatbot_main(args) -> int:
    """Entry point for the `fastworkflow run_chatbot` subcommand.

    UX contract:
    The chatbot opens with a workflow picker (bundled examples + a directory
    browser). Selecting one discovers its env files or asks the developer to
    install them, then starts the FastAPI server unless ``--server-port``
    named an existing server.
    """
    workspace_manifest_path = getattr(args, "workspace_manifest", None)
    spawn_options = spawn_options_from_cli_args(args)
    if workspace_manifest_path:
        # Workspace inspection never starts or connects to a live workflow server.
        spawn_options = {"no_server": True}
    try:
        server = ChatbotServer(
            port=0,
            spawn_options=spawn_options,
            workspace_manifest_path=workspace_manifest_path,
        )
    except (OSError, WorkspaceError, ValueError) as exc:
        print(f"Error: cannot start the chatbot ({exc}).")
        return 1

    # -- banner ---------------------------------------------------------
    print("fastWorkflow Chatbot")
    if server.workspace is not None:
        print(
            "  read-only workspace: "
            + server.workspace.label
            + " ("
            + str(server.workspace.manifest_path)
            + ")"
        )
    else:
        print("  pick a workflow in the browser (bundled examples")
        print("  and local folders are listed; you can browse anywhere).")
    print(f"\n  Open in your browser:\n\n    {server.url}\n")
    print("Press Ctrl+C to stop.", flush=True)
    _open_in_browser(server.url)

    # A service-manager SIGTERM / terminal SIGHUP must run the same cleanup as
    # Ctrl+C — without this, `kill <chatbot-pid>` or closing the terminal
    # orphans the spawned FastAPI server (which may be running with unsigned
    # JWTs and loaded API keys). SIGKILL still cannot be caught; the child's
    # --parent_pid watch covers that case.
    def _raise_system_exit(_signum, _frame):
        raise SystemExit(0)

    for _sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if _sig is None:
            continue
        try:
            signal.signal(_sig, _raise_system_exit)
        except (ValueError, OSError):
            pass  # not the main thread / unsupported platform: keep going

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.server_proc is not None:
            print("Stopping the spawned FastAPI server...", flush=True)
        server.shutdown()  # also terminates the spawned server
    return 0
