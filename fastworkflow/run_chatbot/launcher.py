"""Chatbot test mode: spawning the FastAPI server as a child process [R19].

Design invariants (docs/fastworkflow_observability_studio_design.md §3.4,
Decision Log R19/R23):

- A chatbot-started FastAPI server ALWAYS binds ``--host 127.0.0.1``. A wider
  bind is a command-line decision on ``fastworkflow run_fastapi_mcp`` itself,
  never a chatbot option (and never a chatbot UI option).
- CORS is pinned to exactly the chatbot origin via ``--cors_origin`` — no
  wildcard ever.
- The server's default JWT posture is unsigned (dev) tokens, which IS the
  chatbot's default spawn mode ([R19] as amended: loopback-only bind +
  loopback-pinned CORS + tokens minted by the chatbot itself via
  ``/initialize``); ``run_chatbot --expect-encrypted-jwt`` restores signed
  mode. ``plan_server_spawn`` itself still refuses unsigned spawns unless the
  caller opts in — the chatbot is that caller.
- The FastAPI server needs the optional ``[server]`` extra at runtime
  (fastapi/uvicorn/fastapi-mcp/pyjwt). Debug mode does not — this module is
  itself stdlib-only and never imports any of those packages; it only checks
  they are importable before spawning the subprocess [R23].

The decision is factored into :func:`plan_server_spawn`, a pure function, so
the refusal logic is unit-testable without spawning subprocesses.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

# Same package set `fastworkflow.cli._require_server_extra` guards on.
SERVER_EXTRA_PACKAGES = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("fastapi-mcp", "fastapi_mcp"),
    ("pyjwt", "jwt"),
)


def missing_server_packages() -> list[str]:
    """Names of [server]-extra packages that are not importable."""
    return [
        pkg
        for pkg, mod in SERVER_EXTRA_PACKAGES
        if importlib.util.find_spec(mod) is None
    ]


@dataclass
class ServerSpawnPlan:
    """The outcome of the spawn decision: either a refusal or a command."""

    ok: bool
    reason: str = ""  # human-readable refusal message when not ok
    cmd: list[str] = field(default_factory=list)
    server_url: str = ""


def plan_server_spawn(
    *,
    workflow_path: str,
    env_file_path: str,
    passwords_file_path: str,
    chatbot_origin: str,
    server_port: int = 8000,
    expect_encrypted_jwt: bool = False,
    allow_unsigned_jwt: bool = False,
    missing_packages: Optional[list[str]] = None,
) -> ServerSpawnPlan:
    """Decide whether the chatbot may spawn the FastAPI server, and with what args.

    Pure function [R19]: callers pass ``missing_packages`` (defaulting to a
    live probe) so tests can exercise every branch without subprocesses.
    """
    if missing_packages is None:
        missing_packages = missing_server_packages()
    if missing_packages:
        return ServerSpawnPlan(
            ok=False,
            reason=(
                "The chatbot's test mode needs the FastAPI server, which requires the "
                "optional 'server' extra. Missing packages: "
                f"{', '.join(missing_packages)}.\n"
                "Install it with one of:\n"
                '  pip install "fastworkflow[server]"\n'
                "  poetry install --extras server\n"
                "(Debug mode keeps working without it.)"
            ),
        )

    if not expect_encrypted_jwt and not allow_unsigned_jwt:
        return ServerSpawnPlan(
            ok=False,
            reason=(
                "Refusing to start the FastAPI server: it would run in "
                "unsigned-JWT development mode (anyone who can reach the port "
                "can mint a valid token). If that is what you want on this "
                "machine, re-run with --allow-unsigned-jwt; to require signed "
                "tokens instead, pass --expect-encrypted-jwt. [R19]"
            ),
        )

    cmd = [
        sys.executable,
        "-m",
        "fastworkflow.run_fastapi_mcp",
        "--workflow_path", workflow_path,
        "--env_file_path", env_file_path,
        "--passwords_file_path", passwords_file_path,
        "--port", str(server_port),
        # Always loopback [R19]. Never configurable from the chatbot.
        "--host", "127.0.0.1",
        # CORS pinned to loopback origins (any port) — no wildcard, no
        # routable origin ever; any-port because port forwarders (WSL/IDE)
        # legitimately serve the chatbot page from a different local port
        # [R19 as amended alongside R18].
        "--cors_loopback_only",
        # The chatbot exists to be inspected, and token usage / cost /
        # cache-hit status reach the trace only via a DSPy history entry.
        # Bounded, and only ever for this loopback single-developer server.
        "--keep_dspy_history",
    ]
    if expect_encrypted_jwt:
        cmd.append("--expect_encrypted_jwt")
    return ServerSpawnPlan(
        ok=True, cmd=cmd, server_url=f"http://127.0.0.1:{server_port}"
    )


def spawn_server(plan: ServerSpawnPlan) -> subprocess.Popen:
    """Start the planned server as a child process (inherits stdio/env).

    The child gets its own session/process group so (a) a terminal Ctrl+C
    cannot kill it out from under the chatbot's own shutdown sequencing and
    (b) ``terminate_server`` can signal the whole group, catching any workers
    the server itself forks.
    """
    if not plan.ok:
        raise ValueError(f"refused spawn plan cannot be executed: {plan.reason}")
    return subprocess.Popen(plan.cmd, start_new_session=(os.name == "posix"))


def terminate_server(proc: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    """Stop the child server: SIGTERM, bounded wait, then SIGKILL.

    Signals the child's process group when it leads one (see
    ``spawn_server``), so forked workers stop with it.
    """
    if proc.poll() is not None:
        return

    def _signal(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                raise OSError  # fall through to the single-process path
        except (OSError, ProcessLookupError):
            try:
                proc.send_signal(sig)
            except (OSError, ProcessLookupError):
                pass

    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        _signal(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ----------------------------------------------------------------------
# Detached `fastworkflow train` (chatbot picker Train button, fix-kw7.16)
# ----------------------------------------------------------------------
#
# Training can take an hour and must survive chatbot exit. The chatbot does
# not keep the Popen, does not terminate it on shutdown, and learns status
# from a pid file + `is_workflow_trained`-equivalent filesystem check.
# Stdio is redirected to a log under the workflow state dir: inheriting the
# chatbot's pipes would SIGPIPE the child when the chatbot process exits.

TRAIN_PID_FILENAME = "chatbot_train.pid"
TRAIN_LOG_FILENAME = "chatbot_train.log"


@dataclass
class TrainSpawnPlan:
    """Outcome of the train-spawn decision: a refusal or a command + paths."""

    ok: bool
    reason: str = ""
    cmd: list[str] = field(default_factory=list)
    workflow_path: str = ""
    pid_path: str = ""
    log_path: str = ""


def bundled_examples_root() -> str:
    """Installed ``fastworkflow/examples/`` — never train these in place (Rule 3)."""
    import fastworkflow

    return os.path.join(os.path.dirname(os.path.abspath(fastworkflow.__file__)), "examples")


def is_bundled_example_path(path: str, root: Optional[str] = None) -> bool:
    """True when *path* is the packaged examples dir or a workflow inside it."""
    root = os.path.realpath(root if root is not None else bundled_examples_root())
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def train_artifact_paths(workflow_path: str, *, create: bool = False) -> tuple[str, str]:
    """``(pid_path, log_path)`` under the workflow state dir.

    ``create=False`` (the poll path) must not mkdir: listing 100 candidates
    would otherwise stamp empty state dirs for every untrained workflow.
    """
    from fastworkflow import state_paths

    if create:
        state_dir = state_paths.workflow_state_dir(workflow_path)
    else:
        state_dir = os.path.join(
            state_paths.state_root(),
            "workflows",
            state_paths.workflow_id(workflow_path),
        )
    return (
        os.path.join(state_dir, TRAIN_PID_FILENAME),
        os.path.join(state_dir, TRAIN_LOG_FILENAME),
    )


def read_pid_file(pid_path: str) -> Optional[int]:
    return _read_pid_record(pid_path)[0]


def _read_pid_record(pid_path: str) -> tuple[Optional[int], Optional[str]]:
    """(pid, recorded process start time) from a pid file; (None, None) when
    absent/garbled. The start time is the second whitespace-separated token,
    present since the pid-reuse hardening; older single-token files parse
    with a None start time."""
    try:
        text = open(pid_path, encoding="utf-8").read().strip()
    except OSError:
        return None, None
    tokens = text.split()
    try:
        pid = int(tokens[0])
    except (IndexError, ValueError):
        return None, None
    if pid <= 0:
        return None, None
    return pid, (tokens[1] if len(tokens) > 1 else None)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists; we just cannot signal it
    except OSError:
        return False
    return True


def _proc_start_time(pid: int) -> Optional[str]:
    """The kernel's start-time ticks for *pid* (Linux /proc), or None.

    (pid, start_time) identifies a process across pid reuse; the comm field
    can contain spaces/parens, so split from the LAST ')'.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            stat = f.read()
        fields = stat.rsplit(")", 1)[1].split()
        return fields[19]  # field 22 overall; 20th after (comm)
    except (OSError, IndexError):
        return None


def _pid_is_recorded_train(pid: int, recorded_start: Optional[str]) -> bool:
    """Guard against PID reuse: pid files persist after a train finishes, and
    an unrelated process recycling the pid would otherwise block training
    forever ('another training run is already in progress' with no override).
    The pid file records the process start time at spawn; a live pid whose
    start time differs is a stranger wearing a recycled number. Records
    without a start time (older files, non-Linux) fall back to liveness."""
    if not process_is_alive(pid):
        return False
    if recorded_start is None:
        return True
    current = _proc_start_time(pid)
    return current is None or current == recorded_start


def _clear_stale_pid_file(pid_path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(pid_path)


def write_pid_file(pid_path: str, pid: int) -> None:
    parent = os.path.dirname(pid_path)
    os.makedirs(parent, exist_ok=True)
    start_time = _proc_start_time(pid)
    record = f"{pid} {start_time}" if start_time is not None else str(pid)
    tmp_path = pid_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as stream:
        stream.write(record)
    os.replace(tmp_path, pid_path)


def is_train_running(workflow_path: str) -> bool:
    pid_path, _log_path = train_artifact_paths(workflow_path, create=False)
    pid, recorded_start = _read_pid_record(pid_path)
    if pid is None:
        return False
    if _pid_is_recorded_train(pid, recorded_start):
        return True
    _clear_stale_pid_file(pid_path)  # finished train, or a recycled pid
    return False


def find_live_train() -> Optional[tuple[str, int]]:
    """``(pid_path, pid)`` of any live chatbot-spawned train, else None.

    Stale pid files (finished trains, recycled pids) are removed as they are
    encountered, so one forgotten file can never block training globally.
    """
    from fastworkflow import state_paths

    root = os.path.join(state_paths.state_root(), "workflows")
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in names:
        pid_path = os.path.join(root, name, TRAIN_PID_FILENAME)
        pid, recorded_start = _read_pid_record(pid_path)
        if pid is None:
            continue
        if _pid_is_recorded_train(pid, recorded_start):
            return pid_path, pid
        _clear_stale_pid_file(pid_path)
    return None


def missing_training_package() -> bool:
    """True when the optional ``datasets`` extra (required by ``train``) is absent."""
    return importlib.util.find_spec("datasets") is None


def plan_train_spawn(
    *,
    workflow_path: str,
    env_file_path: str,
    passwords_file_path: str,
    bundled_root: Optional[str] = None,
    live_train: Optional[tuple[str, int]] = None,
    datasets_missing: Optional[bool] = None,
    already_trained: bool = False,
) -> TrainSpawnPlan:
    """Decide whether the chatbot may spawn ``fastworkflow train``.

    Pure enough to unit-test: callers inject *bundled_root*, *live_train*,
    and *datasets_missing*. Default probes match production.
    """
    workflow_path = os.path.abspath(workflow_path)
    pid_path, log_path = train_artifact_paths(workflow_path, create=False)

    if is_bundled_example_path(workflow_path, bundled_root):
        return TrainSpawnPlan(
            ok=False,
            reason=(
                "Refusing to train a bundled package example in place. "
                "Fetch a local copy with `fastworkflow examples fetch <name>` "
                "and train that (Rule 3: never write fastworkflow/examples/"
                "*/___command_info)."
            ),
            workflow_path=workflow_path,
        )

    if already_trained:
        return TrainSpawnPlan(
            ok=False,
            reason="This workflow is already trained.",
            workflow_path=workflow_path,
            pid_path=pid_path,
            log_path=log_path,
        )

    if live_train is None:
        live_train = find_live_train()
    if live_train is not None:
        live_path, live_pid = live_train
        if os.path.realpath(live_path) == os.path.realpath(pid_path):
            reason = (
                f"Training is already running for this workflow (pid {live_pid})."
            )
        else:
            reason = (
                f"Another training run is already in progress (pid {live_pid}). "
                "Only one train at a time — intent models are memory-heavy."
            )
        return TrainSpawnPlan(
            ok=False,
            reason=reason,
            workflow_path=workflow_path,
            pid_path=pid_path,
            log_path=log_path,
        )

    if not env_file_path or not passwords_file_path:
        return TrainSpawnPlan(
            ok=False,
            reason=(
                "Missing fastworkflow.env and/or fastworkflow.passwords.env. "
                "Select this workflow in the picker to create or upload them, "
                "then click Train."
            ),
            workflow_path=workflow_path,
        )
    if not os.path.isfile(env_file_path) or not os.path.isfile(passwords_file_path):
        return TrainSpawnPlan(
            ok=False,
            reason=(
                "Env files were named but are not on disk. Select this workflow "
                "in the picker to create or upload them, then click Train."
            ),
            workflow_path=workflow_path,
        )

    if datasets_missing is None:
        datasets_missing = missing_training_package()
    if datasets_missing:
        return TrainSpawnPlan(
            ok=False,
            reason=(
                "Training needs the optional 'training' extra (the datasets "
                "package). Install it with one of:\n"
                '  pip install "fastworkflow[training]"\n'
                "  poetry install --extras training"
            ),
            workflow_path=workflow_path,
        )

    cmd = [
        sys.executable,
        "-m",
        "fastworkflow.train",
        workflow_path,
        env_file_path,
        passwords_file_path,
    ]
    return TrainSpawnPlan(
        ok=True,
        cmd=cmd,
        workflow_path=workflow_path,
        pid_path=pid_path,
        log_path=log_path,
    )


def spawn_detached_train(plan: TrainSpawnPlan) -> int:
    """Start train detached: new session, stdio to the log, pid file, reaper.

    Returns the child pid. The Popen is not returned — a daemon thread reaps
    it so a finished train does not stay a zombie (which would look `kill 0`
    alive) while the chatbot is still up. Chatbot shutdown must NOT signal
    this process.
    """
    if not plan.ok:
        raise ValueError(f"refused train plan cannot be executed: {plan.reason}")
    os.makedirs(os.path.dirname(plan.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(plan.pid_path), exist_ok=True)
    log_f = open(plan.log_path, "ab")
    try:
        proc = subprocess.Popen(
            plan.cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
            close_fds=True,
        )
    finally:
        log_f.close()
    write_pid_file(plan.pid_path, proc.pid)
    threading.Thread(
        target=proc.wait, name="chatbot-train-reaper", daemon=True
    ).start()
    return proc.pid
