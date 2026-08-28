"""fastWorkflow Chatbot debug mode: stdlib-only localhost read-only HTTP layer.

Design invariants (docs/fastworkflow_observability_studio_design.md §3.4):

- Access control [R5][R18]: binds 127.0.0.1 only; a per-launch random bearer
  token (``secrets.token_urlsafe``) embedded in the printed URL (Jupyter
  pattern) is required on EVERY request (Authorization header or ``?token=``),
  compared in constant time (``hmac.compare_digest``); a strict Host/Origin
  allowlist rejects everything non-loopback with 403. Loopback hosts
  (``127.0.0.1`` / ``localhost`` / ``[::1]``) pass on ANY port — port
  forwarders (WSL relays, IDE port forwards) legitimately re-expose the
  server on a different local port — while the loopback-only rule is what
  defeats DNS rebinding, and the token stays the authentication.
- Rendering safety [R22]: the SPA page ships with a restrictive CSP
  (inline script allowed only via its own sha256 hashes — the page is one
  self-contained file); artifact responses carry
  ``default-src 'none'; sandbox`` so direct navigation is inert; the read
  layer only calls ObservabilityStore methods (parameterized queries).
- Read discipline [R12]: per-request store reads — every ObservabilityStore
  method opens its own short-lived connection, no held cursors — so WAL
  checkpointing by the writer never starves.
- Packaging [R23]: stdlib-only. This module must never import
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
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from fastworkflow import state_paths
from fastworkflow.observability_store import (
    IncompatibleObservabilityDB,
    ObservabilityStore,
    ReadOnlyObservabilityStore,
)
from fastworkflow.run_chatbot import launcher

logger = logging.getLogger(__name__)

# Tolerates attributes (e.g. a future type="module") so adding one cannot
# silently produce an empty hash list — which would fail closed and brick the
# page. ChatbotServer.__init__ additionally asserts extraction succeeded.
_SCRIPT_RE = re.compile(rb"<script\b[^>]*>(.*?)</script>", re.DOTALL)

# Content types that may contain active content: only ever rendered inside a
# sandboxed iframe by the SPA; direct responses are additionally sandboxed via
# CSP (see _artifact_headers) [R22].
_HTMLISH_TYPES = ("text/html", "application/xhtml+xml", "image/svg+xml")


def load_index_html() -> bytes:
    """The single self-contained SPA page, shipped as package data [R23]."""
    resource = importlib.resources.files("fastworkflow.run_chatbot") / "static" / "index.html"
    return resource.read_bytes()


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


def _looks_like_workflow(path: str) -> bool:
    """A fastWorkflow workflow dir: authored commands or trained artifacts."""
    return os.path.isdir(os.path.join(path, "_commands")) or os.path.isdir(
        os.path.join(path, "___command_info")
    )


# Mirrors model_pipeline_training.GLOBAL_CONTEXT_FOLDER without importing
# that module — it pulls in torch/transformers, which the chatbot must not.
_GLOBAL_CONTEXT_FOLDER = "global"
_CME_CONTEXT_NAMES: Optional[set[str]] = None


def _cme_context_names() -> set[str]:
    """Internal command_metadata_extraction context names (cached).

    App workflow ``routing_definition.json`` lists these too; they are trained
    in the CME workflow, not per app, so they must not count as missing.
    """
    global _CME_CONTEXT_NAMES
    if _CME_CONTEXT_NAMES is not None:
        return _CME_CONTEXT_NAMES
    names: set[str] = set()
    try:
        import fastworkflow

        internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    except Exception:
        _CME_CONTEXT_NAMES = names
        return names
    json_path = os.path.join(internal, "command_context_model.json")
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            names.update(str(key) for key in data)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    commands = os.path.join(internal, "_commands")
    try:
        for entry in os.listdir(commands):
            full = os.path.join(commands, entry)
            if (
                os.path.isdir(full)
                and not entry.startswith(".")
                and entry != "__pycache__"
            ):
                names.add(entry)
    except OSError:
        pass
    _CME_CONTEXT_NAMES = names
    return names


def _workflow_is_trained(path: str) -> bool:
    """Filesystem check matching ``is_workflow_trained`` without importing torch.

    ``___command_info`` appearing is not enough: train writes that directory
    immediately, before any ``threshold.json`` exists.
    """
    command_info_root = os.path.join(path, "___command_info")
    routing_def_path = os.path.join(command_info_root, "routing_definition.json")
    if not os.path.isfile(routing_def_path):
        return False
    try:
        with open(routing_def_path, encoding="utf-8") as handle:
            routing_definition = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    contexts = routing_definition.get("contexts") or {}
    if not isinstance(contexts, dict) or not contexts:
        return False
    contexts_to_check = (set(contexts) - _cme_context_names()) | {"*"}
    for context_name in contexts_to_check:
        folder = (
            _GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        )
        threshold_path = os.path.join(command_info_root, folder, "threshold.json")
        if not os.path.isfile(threshold_path):
            return False
    return True


def _rel_under(path: str, root: str) -> str:
    """Path relative to ``root``, posix slashes, or ``""`` if outside ``root``."""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        return ""
    if rel == ".":
        return ""
    if rel == ".." or rel.startswith(".." + os.sep):
        return ""
    return rel.replace("\\", "/")


def _workflow_entry(path: str, source: str, rel: str = "") -> dict[str, Any]:
    path = os.path.abspath(path)
    if launcher.is_bundled_example_path(path):
        source = "examples"
    trained = _workflow_is_trained(path)
    training = launcher.is_train_running(path)
    bundled = source == "examples" or launcher.is_bundled_example_path(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "rel": rel or os.path.basename(path),
        "trained": trained,
        "training": training,
        "source": source,
        "trainable": (not bundled) and (not trained) and (not training),
    }


_SKIP_DIR_NAMES = {
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "venv",
    "_commands",
    "___command_info",
    "___workflow_contexts",
    "___convo_info",
}

# Nested project layouts (apps/team/workflow) sit deeper than the old
# two-level scan; five is enough to find them without walking the world.
_MAX_WF_SCAN_DEPTH = 5
_MAX_WF_CANDIDATES = 100


def list_workflow_candidates() -> list[dict[str, Any]]:
    """Workflow dirs the developer most likely wants: the bundled examples,
    plus a bounded nested scan below the launch directory.

    Each entry carries ``rel`` (path relative to the launch directory, or a
    ``Bundled examples/`` prefix when the workflow lives outside it) so the
    picker can group them under folders instead of a flat list.
    """
    seen: dict[str, dict[str, Any]] = {}
    cwd = os.getcwd()

    def add(path: str, source: str, rel: str) -> None:
        path = os.path.abspath(path)
        if path not in seen and _looks_like_workflow(path):
            seen[path] = _workflow_entry(path, source, rel)

    add(cwd, "local", _rel_under(cwd, cwd))

    def walk(current: str, depth: int) -> None:
        if depth > _MAX_WF_SCAN_DEPTH or len(seen) >= _MAX_WF_CANDIDATES:
            return
        try:
            names = sorted(os.listdir(current))
        except OSError:
            return
        for name in names:
            if len(seen) >= _MAX_WF_CANDIDATES:
                return
            if name.startswith(".") or name in _SKIP_DIR_NAMES:
                continue
            full = os.path.join(current, name)
            if not os.path.isdir(full):
                continue
            rel = _rel_under(full, cwd)
            if _looks_like_workflow(full):
                add(full, "local", rel)
            walk(full, depth + 1)

    walk(cwd, 1)
    try:
        import fastworkflow

        examples = os.path.join(
            os.path.dirname(os.path.abspath(fastworkflow.__file__)), "examples"
        )
        for entry in sorted(os.listdir(examples)):
            full = os.path.join(examples, entry)
            rel = _rel_under(full, cwd)
            if not rel:
                rel = "Bundled examples/" + entry
            add(full, "examples", rel)
    except Exception:
        pass
    # A directory that both looks like a workflow and contains other workflows
    # (the library package has _commands/ plus examples/) is a folder, not a
    # leaf the developer would pick.
    for path in list(seen):
        if any(
            other != path and other.startswith(path + os.sep) for other in seen
        ):
            seen.pop(path, None)
    candidates = list(seen.values())
    candidates.sort(
        key=lambda w: (w["source"] != "local", not w["trained"], w["name"].lower())
    )
    return candidates[:_MAX_WF_CANDIDATES]


def browse_directories(dir_path: str) -> dict[str, Any]:
    """One level of the local filesystem for the workflow picker: directories
    only, never file contents; each entry flagged when it is a workflow."""
    base = os.path.abspath(dir_path or os.getcwd())
    if not os.path.isdir(base):
        return {"error": f"not a directory: {base}"}
    entries = []
    try:
        names = sorted(os.listdir(base))
    except OSError as exc:
        return {"error": f"cannot list {base}: {exc}"}
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        is_workflow = _looks_like_workflow(full)
        entry = {
            "name": name,
            "path": full,
            "is_workflow": is_workflow,
            "trained": _workflow_is_trained(full) if is_workflow else False,
        }
        if is_workflow:
            entry["training"] = launcher.is_train_running(full)
        entries.append(entry)
        if len(entries) >= 300:
            break
    parent = os.path.dirname(base)
    return {
        "dir": base,
        "parent": parent if parent != base else None,
        "entries": entries,
    }


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

    return first_existing("fastworkflow.env"), first_existing("fastworkflow.passwords.env")


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
    ) -> None:
        self.db_path = db_path or ""
        self.workflow_path = workflow_path
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
        self._train_lock = threading.Lock()
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
        it inspects [R12] — a missing DB (e.g. test-mode cold start before the
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

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    # -- workflow activation + server lifecycle -------------------------

    def session_payload(self) -> dict[str, Any]:
        """What the SPA needs to run without asking the developer anything.
        Reflects the LIVE child state — the SPA polls this, so a server that
        dies mid-session is reported honestly, not as 'running'."""
        running = self.server_proc is not None and self.server_proc.poll() is None
        # An omitted spawn still reports server_url when --server-port named an
        # existing server, so the Advanced panel can be prefilled.
        expose_url = running or bool(
            self.spawn_options.get("no_server") and self.server_url
        )
        return {
            "workflow_path": self.workflow_path,
            "workflow_name": (
                os.path.basename(os.path.abspath(self.workflow_path))
                if self.workflow_path
                else ""
            ),
            "db_path": self.db_path,
            "server_url": self.server_url if expose_url else None,
            "server_running": running,
            "server_exit_code": (
                self.server_proc.returncode
                if self.server_proc is not None and not running
                else None
            ),
            "server_note": self.server_note,
            "env_setup_required": self.env_setup_required,
            "env_file_path": self.env_file_path or None,
            "passwords_file_path": self.passwords_file_path or None,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "jwt_mode": (
                "signed" if self.spawn_options.get("expect_encrypted_jwt") else "unsigned"
            ),
            "spawn_error": self.spawn_error,
        }

    def activate_workflow(self, workflow_path: str) -> dict[str, Any]:
        """Point the chatbot at a workflow and (unless disabled) make sure its
        FastAPI server is running. Selecting a different workflow replaces the
        spawned server. Never raises: failures land in ``spawn_error`` and the
        chatbot stays usable as a trace viewer."""
        from fastworkflow import state_paths

        with self._activate_lock:
            workflow_path = os.path.abspath(workflow_path)
            same_workflow = os.path.abspath(self.workflow_path or "") == workflow_path
            self.workflow_path = workflow_path
            self.db_path = state_paths.observability_db(workflow_path)
            self.spawn_error = None
            self.server_note = None
            if self.spawn_options.get("no_server"):
                external = self.spawn_options.get("server_port")
                if external:
                    self.server_url = f"http://127.0.0.1:{int(external)}"
                return self.session_payload()
            if (
                same_workflow
                and self.server_proc is not None
                and self.server_proc.poll() is None
            ):
                return self.session_payload()  # already serving this workflow

            from fastworkflow.run_chatbot import launcher

            if self.server_proc is not None and self.server_proc.poll() is None:
                launcher.terminate_server(self.server_proc)
            self.server_proc = None
            self.server_url = None

            env_file = self.spawn_options.get("env_file_path") or ""
            passwords_file = self.spawn_options.get("passwords_file_path") or ""
            if not env_file or not passwords_file:
                auto_env, auto_passwords = _autodetect_env_files(workflow_path)
                env_file = env_file or auto_env
                passwords_file = passwords_file or auto_passwords
            self.env_file_path = env_file
            self.passwords_file_path = passwords_file
            self.env_setup_required = not (
                env_file
                and passwords_file
                and os.path.isfile(env_file)
                and os.path.isfile(passwords_file)
            )
            if self.env_setup_required:
                self.server_proc = None
                self.server_url = None
                return self.session_payload()

            preferred_port = int(
                self.spawn_options.get("server_port") or PREFERRED_SPAWN_PORT
            )
            server_port, moved = _free_server_port(preferred_port)
            self.server_note = (
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
            plan = launcher.plan_server_spawn(
                workflow_path=workflow_path,
                env_file_path=env_file,
                passwords_file_path=passwords_file,
                chatbot_origin=f"http://127.0.0.1:{self.port}",
                server_port=server_port,
                expect_encrypted_jwt=expect_encrypted,
                # Loopback-only + loopback-pinned CORS + a chatbot that mints
                # its own tokens via /initialize: unsigned dev JWTs are the
                # default posture for the AUTO-spawned server (owner decision
                # amending R19's opt-in flag; --expect-encrypted-jwt restores
                # signed mode).
                allow_unsigned_jwt=not expect_encrypted,
            )
            if not plan.ok:
                self.spawn_error = plan.reason
                return self.session_payload()
            try:
                self.server_proc = launcher.spawn_server(plan)
            except OSError as exc:
                self.spawn_error = f"could not start the FastAPI server: {exc}"
                return self.session_payload()
            time.sleep(1.0)  # one early liveness check: died-at-startup is common
            if self.server_proc.poll() is not None:
                self.spawn_error = (
                    "the FastAPI server exited immediately "
                    f"(exit code {self.server_proc.returncode}) — its output is in "
                    "the chatbot's terminal; check that the --server-port is free "
                    "and the env files are valid"
                )
                self.server_proc = None
                return self.session_payload()
            self.server_url = plan.server_url
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
                _write_env_file(
                    env_target, _env_template_text("fastworkflow.env")
                )
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


class _ChatbotRequestHandler(BaseHTTPRequestHandler):
    """Token-gated request handler. Observability queries are GET-only;
    explicit control-plane POSTs select a workflow, configure env, start
    train, or clear recorded conversations."""

    chatbot: ChatbotServer  # bound by ChatbotServer.__init__
    protocol_version = "HTTP/1.1"
    server_version = "fastWorkflowChatbot"
    sys_version = ""

    # -- plumbing --------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # quiet; the terminal belongs to the launch banner

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
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

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

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
            if scheme != "http" or not sep or not self._is_loopback_authority(authority):
                logger.warning(
                    f"Chatbot refused a request with non-loopback Origin {origin!r} [R18]"
                )
                return False
        return True

    def _token_valid(self, query: dict[str, list[str]]) -> bool:
        presented = ""
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            presented = auth[len("Bearer "):].strip()
        elif query.get("token"):
            presented = query["token"][0]
        return hmac.compare_digest(
            presented.encode("utf-8"), self.chatbot.token.encode("utf-8")
        )

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    def _handle_get(self) -> None:
        split = urlsplit(self.path)
        path = split.path
        query = parse_qs(split.query)

        if not self._host_origin_allowed():
            self._error(
                403,
                "forbidden: only loopback hosts (127.0.0.1 / localhost / [::1]) "
                f"may access the chatbot; got Host={self.headers.get('Host')!r}, "
                f"Origin={self.headers.get('Origin')!r}",
            )
            return
        # EVERY request is token-gated, the page included (Jupyter pattern).
        if not self._token_valid(query):
            self._error(401, "unauthorized: missing or invalid token")
            return

        if path in ("/", "/index.html"):
            self._send(
                200,
                self.chatbot.index_html,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": self.chatbot.page_csp},
            )
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

    def do_POST(self) -> None:  # noqa: N802
        try:
            split = urlsplit(self.path)
            if split.path not in {
                "/api/select_workflow",
                "/api/configure_env",
                "/api/clear_conversations",
                "/api/train",
            }:
                self._refuse_write()
                return
            query = parse_qs(split.query)
            if not self._host_origin_allowed():
                self._error(403, "forbidden: host/origin not allowed")
                return
            if not self._token_valid(query):
                self._error(401, "unauthorized: missing or invalid token")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._error(400, "invalid JSON body")
                return
            if split.path == "/api/select_workflow":
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
                return
            if split.path == "/api/configure_env":
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
                return
            if split.path == "/api/train":
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
                        if (
                            "already running" in lowered
                            or "in progress" in lowered
                        )
                        else 400
                    )
                    self._error(status, reason)
                    return
                self._send_json(result)
                return

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
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    do_PUT = _refuse_write  # noqa: N815
    do_DELETE = _refuse_write  # noqa: N815
    do_PATCH = _refuse_write  # noqa: N815

    # -- API endpoints ---------------------------------------------------

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        # Per-request read-only store [R12]; None while the DB does not exist
        # yet (test-mode cold start) — serve empty views, never an error.
        store = self.chatbot.open_store()
        q = lambda name: query.get(name, [None])[0]  # noqa: E731

        if path == "/api/meta":
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
        elif path == "/api/session":
            self._send_json({"session": self.chatbot.session_payload()})
        elif path == "/api/workflows":
            self._send_json({"workflows": list_workflow_candidates()})
        elif path == "/api/browse":
            self._send_json(browse_directories(q("dir") or ""))
        elif store is None:
            if path == "/api/health":
                self._send_json(
                    {"writer_health": None, "db_size_bytes": 0, "db_available": False}
                )
            elif path in ("/api/channels", "/api/conversations", "/api/turns"):
                self._send_json(
                    {"channels": [], "conversations": [], "turns": []}
                )
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
            success = q("success")
            self._send_json(
                {
                    "turns": store.list_turns(
                        channel_id=q("channel"),
                        conversation_id=(
                            self._int(q("conversation"), None)
                            if q("conversation") is not None
                            else None
                        ),
                        status=q("status"),
                        success=(
                            None
                            if success is None
                            else success in ("1", "true", "True")
                        ),
                        command_name=q("command"),
                        context=q("context"),
                        limit=self._int(q("limit"), 100),
                        offset=self._int(q("offset"), 0),
                    )
                }
            )
        elif path.startswith("/api/turn/"):
            turn_key = path[len("/api/turn/"):]
            turn = store.get_turn(turn_key)
            if turn is None:
                self._error(404, "turn not found")
                return
            try:
                turn["record"] = json.loads(turn.pop("record_json"))
            except (ValueError, KeyError):
                turn["record"] = None
            self._send_json({"turn": turn})
        elif path.startswith("/api/spans/"):
            trace_id = path[len("/api/spans/"):]
            spans = store.get_spans(trace_id)
            for span in spans:
                try:
                    span["attributes"] = json.loads(span["attributes"])
                except (ValueError, TypeError, KeyError):
                    pass
            self._send_json({"spans": spans})
        elif path.startswith("/api/artifact/"):
            self._serve_artifact(store, path[len("/api/artifact/"):])
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

    def _serve_artifact(self, store: ReadOnlyObservabilityStore, artifact_id: str) -> None:
        """Offloaded artifact content, with its stored content-type.

        HTML-ish content is only ever *rendered* inside a sandboxed iframe by
        the SPA [R22]; the raw response is additionally neutralized with
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
    def _int(value: Optional[str], default: Any) -> Any:
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default


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
    """Library maintenance utility [R12]: bounded prune + vacuum.

    Not wired to any CLI flag or HTTP route — pruning runs automatically at
    sink startup; this exists for scripts/tests that need it on demand.
    """
    return ObservabilityStore(db_path).prune()


def run_forget_channel(
    db_path: str, channel_id: str, workflow_path: str = ""
) -> dict[str, int]:
    """Library erasure utility [R21]: delete one channel everywhere.

    Not wired to any CLI flag or HTTP route — the chatbot UI exposes the
    all-channel Clear-conversations action instead; this remains the
    single-channel primitive for scripts/tests (e.g. a deletion request for
    one API channel). Also deletes the LEGACY per-channel conversation DB
    (``conversations/<channel_id>.sqlite3`` + sidecars) while the Phase-A
    dual-write period lasts — without this, "forgotten" conversations remain
    fully readable in the legacy store (Phase-7 ruling C1).
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


def run_clear_conversations(
    db_path: str, workflow_path: str = ""
) -> dict[str, int]:
    """Erase all conversation/turn observability for one workflow."""
    deleted = ObservabilityStore(db_path).clear_conversations()
    if workflow_path:
        legacy_dir = state_paths.conversations_dir(workflow_path)
        removed = 0
        try:
            names = os.listdir(legacy_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if not name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm")):
                continue
            path = os.path.join(legacy_dir, name)
            if not os.path.isfile(path):
                continue
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
    spawn_options = spawn_options_from_cli_args(args)
    try:
        server = ChatbotServer(port=0, spawn_options=spawn_options)
    except OSError as exc:
        print(
            f"Error: cannot bind 127.0.0.1 to a free port ({exc})."
        )
        return 1

    # -- banner ---------------------------------------------------------
    print("fastWorkflow Chatbot")
    print("  pick a workflow in the browser (bundled examples")
    print("  and local folders are listed; you can browse anywhere).")
    print(f"\n  Open in your browser:\n\n    {server.url}\n")
    print("Press Ctrl+C to stop.", flush=True)
    _open_in_browser(server.url)

    # A service-manager SIGTERM must run the same cleanup as Ctrl+C — without
    # this, `kill <chatbot-pid>` orphans the spawned FastAPI server (which may
    # be running with unsigned JWTs and loaded API keys).
    def _raise_system_exit(_signum, _frame):
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _raise_system_exit)
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
