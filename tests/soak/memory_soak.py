#!/usr/bin/env python
"""RSS/USS/latency soak harness for the run_fastapi_mcp server (design §16.5, §16.6).

Deliberately NOT a pytest module. The filename does not match pytest's default
``test_*.py`` / ``*_test.py`` patterns, so ``pytest tests/`` does not collect it.
That is intentional: this takes minutes, launches real server processes, and is a
measurement instrument rather than a pass/fail unit test.

Why a subprocess and real HTTP, not an in-process ASGI/TestClient harness
------------------------------------------------------------------------
The DSPy memory controls this soak exists to gate are installed in
``fastworkflow/run_fastapi_mcp/__main__.py::main()``, synchronously, *before*
uvicorn creates the event loop. An in-process harness imports ``app`` directly
and never runs ``main()``, so ``server_memory.install_policy()`` never fires and
the harness cannot observe the thing it is meant to gate (§16.5 ``[R2-22]``).
This harness therefore launches ``python -m fastworkflow.run_fastapi_mcp`` and
asserts, from the server's own startup log, that the policy was installed.

Workload
--------
Strictly sequential HTTP, one unique ~450 KB payload per request, no LLM call:
every request dispatches ``add_two_numbers`` as a *direct action*, which bypasses
intent detection and parameter extraction entirely, so no trained model and no
provider call is involved.

Arms
----
A0  unique ``channel_id`` per request (the motivating production shape).
    RECORDED, NOT GATED. Expected to sit materially below the unpatched baseline
    but still above 0.05 MB/request, because the live-session cache retains every
    unique channel at the unchanged 2000 default.
A1  one channel, several hundred request-sized direct actions.
    GATED: in-memory conversation bytes must plateau and the second-half slope
    must be <= 0.05 MB/request.

Usage
-----
    source .venv/bin/activate
    python tests/soak/memory_soak.py --arm a1 --requests 300 --replicates 3
    python tests/soak/memory_soak.py --arm a0 --requests 300 --replicates 3
    python tests/soak/memory_soak.py --arm a1 --latency --json soak.json
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.client import HTTPConnection, HTTPException
from typing import Any, Optional

try:  # optional: not a fastworkflow dependency, so /proc is the expected path
    import psutil
except ImportError:
    psutil = None

# ---------------------------------------------------------------------------
# Structural caps under test. Duplicated here rather than imported, because the
# harness must be able to catch a cap that was silently *raised* in the server;
# importing the constant would make that change invisible to this check.
# ---------------------------------------------------------------------------
CAP_RETAINED_TURNS = 20          # turns.MAX_RETAINED_STARTUP_TURNS
CAP_DSPY_CACHE_ENTRIES = 200     # server_memory.SERVER_DSPY_MEMORY_CACHE_ENTRIES
CAP_CONVERSATION_TURNS = 20      # utils.MAX_CONVERSATION_TURNS_IN_MEMORY

SLOPE_TARGET_MB_PER_REQUEST = 0.05

# §16.5 calls for "several hundred measured requests minimum". Below this the
# slope is dominated by allocator warm-up, so the gate is reported as advisory
# instead of being enforced — a smoke run must not be able to manufacture a
# green (or red) verdict.
MIN_REQUESTS_FOR_BINDING_GATE = 200

# One-sided 95% t quantiles by degrees of freedom, for the slope upper bound.
_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 15: 1.753,
        20: 1.725, 30: 1.697}


class SoakError(RuntimeError):
    """A prerequisite is missing or the server misbehaved. Never degrade silently."""


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    repo_root: str
    python: str
    env_file: str
    passwords_file: str
    workflow_path: str
    workflow_name: str


def resolve_paths() -> Paths:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # The venv interpreter explicitly, not sys.executable: DSPy configuration
    # ownership is process-global and an ambient interpreter may carry a
    # different DSPy than the one the policy was proven against (§16.4 [R15]).
    python = os.path.join(repo_root, ".venv", "bin", "python")
    if not os.path.isfile(python):
        raise SoakError(
            f"venv interpreter not found at {python}. "
            f"Run: cd {repo_root} && python -m venv .venv && poetry install"
        )

    env_file = os.path.join(repo_root, "env", ".env")
    passwords_file = os.path.join(repo_root, "passwords", ".env")
    for label, path in (("env", env_file), ("passwords", passwords_file)):
        if not os.path.isfile(path):
            raise SoakError(
                f"{label} file missing at {path}. The server reads SPEEDDICT_FOLDERNAME "
                f"and model/API settings from these; see fastworkflow/examples/fastworkflow.env."
            )

    # Both bundled hello_world workflows define add_two_numbers; prefer the tests
    # copy so the soak never writes near fastworkflow/examples/*/___command_info.
    candidates = [
        os.path.join(repo_root, "tests", "hello_world_workflow"),
        os.path.join(repo_root, "fastworkflow", "examples", "hello_world"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "_commands", "add_two_numbers.py")):
            return Paths(repo_root, python, env_file, passwords_file,
                         candidate, os.path.basename(candidate))

    raise SoakError(
        "No hello_world workflow with an add_two_numbers command found. Looked in:\n  "
        + "\n  ".join(candidates)
    )


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------

class MemoryProbe:
    """RSS/USS/cgroup for a foreign pid, from psutil when present, /proc otherwise.

    psutil is not a fastworkflow dependency, so the /proc path is the expected
    one on a stock checkout. Which source was used is reported rather than
    assumed: USS from smaps_rollup and USS from psutil are the same quantity,
    but a reader comparing two runs needs to know they were measured the same way.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._proc = None
        if psutil is not None:
            self._proc = psutil.Process(pid)
            self.source = f"psutil {psutil.__version__}"
        elif os.path.isfile(f"/proc/{pid}/smaps_rollup"):
            self.source = "/proc/<pid>/status VmRSS + /proc/<pid>/smaps_rollup Private_*"
        else:
            raise SoakError(
                "psutil is not installed and /proc/<pid>/smaps_rollup is unavailable, "
                "so USS cannot be measured. Install psutil into .venv, or run on a Linux "
                "kernel that exposes smaps_rollup."
            )

        self.cgroup_path = self._resolve_cgroup(pid)
        self.cgroup_reason = (
            None if self.cgroup_path
            else "no readable cgroup memory accounting file for this pid "
                 "(cgroup v2 root cgroups expose no memory.current)"
        )

    @staticmethod
    def _resolve_cgroup(pid: int) -> Optional[str]:
        candidates: list[str] = []
        with contextlib.suppress(OSError):
            with open(f"/proc/{pid}/cgroup", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split(":", 2)
                    if len(parts) != 3:
                        continue
                    controllers, rel = parts[1], parts[2].lstrip("/")
                    if controllers == "":  # cgroup v2 unified line
                        candidates.append(os.path.join("/sys/fs/cgroup", rel, "memory.current"))
                    elif "memory" in controllers.split(","):
                        candidates.append(
                            os.path.join("/sys/fs/cgroup/memory", rel, "memory.usage_in_bytes")
                        )
        candidates += ["/sys/fs/cgroup/memory.current",
                       "/sys/fs/cgroup/memory/memory.usage_in_bytes"]
        return next((path for path in candidates if os.access(path, os.R_OK)), None)

    def _proc_rss_uss(self) -> tuple[int, int]:
        rss = 0
        with open(f"/proc/{self.pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        uss = 0
        with open(f"/proc/{self.pid}/smaps_rollup", encoding="utf-8") as handle:
            for line in handle:
                # USS is exactly the private set, which is what psutil reports too.
                if line.startswith(("Private_Clean:", "Private_Dirty:", "Private_Hugetlb:")):
                    uss += int(line.split()[1]) * 1024
        return rss, uss

    def sample(self) -> tuple[int, int, Optional[int]]:
        if self._proc is not None:
            info = self._proc.memory_full_info()
            rss, uss = info.rss, info.uss
        else:
            rss, uss = self._proc_rss_uss()

        cgroup: Optional[int] = None
        if self.cgroup_path:
            try:
                with open(self.cgroup_path, encoding="utf-8") as handle:
                    cgroup = int(handle.read().strip())
            except (OSError, ValueError):
                cgroup = None
        return rss, uss, cgroup


# ---------------------------------------------------------------------------
# Server process
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    """A fresh ``python -m fastworkflow.run_fastapi_mcp`` on a private port and store.

    ``server_tree`` selects which checkout the *server* imports fastworkflow from,
    so the pre-Release-A baseline can be measured by the same harness binary on the
    same host with the same payloads. The harness itself always runs from the
    working tree; only the subprocess is redirected.
    """

    def __init__(self, paths: Paths, *, startup_timeout: float, graceful: bool,
                 grace_seconds: float, baseline: bool = False,
                 server_tree: Optional[str] = None):
        self.paths = paths
        self.startup_timeout = startup_timeout
        self.graceful = graceful
        self.grace_seconds = grace_seconds
        self.baseline = baseline
        self.server_tree = server_tree or paths.repo_root
        self.port = _free_port()
        self.tmpdir = tempfile.mkdtemp(prefix="fw_memory_soak_")
        self.speeddict_dir = os.path.join(self.tmpdir, "speeddict")
        self.env_file = os.path.join(self.tmpdir, "env")
        self.log_path = os.path.join(self.tmpdir, "server.log")
        self.proc: Optional[subprocess.Popen] = None
        self.probe: Optional[MemoryProbe] = None
        self.policy_line: Optional[str] = None
        self.policy_check: str = "pending"
        self.memory_metrics_available: Optional[bool] = None
        self._log_handle = None

    def _write_env_override(self) -> None:
        """Redirect every durable store into the temp dir.

        SPEEDDICT_FOLDERNAME is read from the env FILE, and fastworkflow.get_env_var
        prefers the file over os.environ, so exporting the variable would be
        silently ignored. The default value is the *relative* ``___workflow_contexts``,
        which would otherwise land RocksDB directories inside the repo's workflow
        fixtures and mix soak data into the committed test artifacts.
        """
        os.makedirs(self.speeddict_dir, exist_ok=True)
        with open(self.paths.env_file, encoding="utf-8") as handle:
            lines = [
                line for line in handle.read().splitlines()
                if not line.strip().startswith("SPEEDDICT_FOLDERNAME=")
            ]
        lines.append(f"SPEEDDICT_FOLDERNAME={self.speeddict_dir}")
        with open(self.env_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def start(self) -> None:
        self._write_env_override()
        cmd = [
            self.paths.python, "-m", "fastworkflow.run_fastapi_mcp",
            "--workflow_path", self.paths.workflow_path,
            "--env_file_path", self.env_file,
            "--passwords_file_path", self.paths.passwords_file,
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ]
        env = dict(os.environ)
        # Unbuffered, because the readiness assertion below reads the startup log
        # while the process is still running; a buffered stream would hide it.
        env["PYTHONUNBUFFERED"] = "1"
        # The venv installs fastworkflow editable via a plain-path .pth, which site
        # appends to sys.path; PYTHONPATH is inserted ahead of it, so this is what
        # decides which checkout the server imports. Verified per run by
        # resolve_server_import(), and again behaviourally by _verify_tree_identity().
        env["PYTHONPATH"] = os.pathsep.join(
            [self.server_tree] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )

        self._log_handle = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.server_tree,
            env=env,
            # To a file, not a pipe: nothing drains a pipe during the soak, and a
            # full pipe buffer would block the server mid-measurement.
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready()
        self._assert_policy_installed()
        self._verify_tree_identity()
        self.probe = MemoryProbe(self.proc.pid)

    def _wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        last_error = "no attempt made"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SoakError(
                    f"Server exited with code {self.proc.returncode} before becoming "
                    f"ready. Log tail:\n{self.log_tail()}"
                )
            try:
                conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request("GET", "/probes/readyz")
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}: {body[:200]!r}"
            except (OSError, HTTPException) as exc:
                last_error = repr(exc)
            time.sleep(0.5)
        raise SoakError(
            f"Server on port {self.port} never returned 200 from /probes/readyz within "
            f"{self.startup_timeout:.0f}s (last: {last_error}). Log tail:\n{self.log_tail()}"
        )

    def _assert_policy_installed(self) -> None:
        """Fail if the entrypoint's DSPy policy did not actually install.

        Readiness alone does not prove it: importing the app some other way also
        serves traffic, just without the memory controls this soak exists to
        gate. The startup record distinguishes the two.
        """
        if self.baseline:
            # The pre-Release-A entrypoint has no server_memory module at all, so
            # there is no policy to assert. Skipping is the whole point of the
            # baseline arm; recorded rather than silent so a treatment run can
            # never be mistaken for a baseline run in the artifact.
            self.policy_check = (
                "SKIPPED (--baseline): the baseline tree has no "
                "fastworkflow/run_fastapi_mcp/server_memory.py, so no DSPy memory "
                "policy exists to install or assert"
            )
            return

        log = self._read_log()
        match = re.search(r"memory bounds active:.*", log)
        if not match:
            raise SoakError(
                "Server became ready but never logged its memory bounds, so the DSPy "
                f"policy cannot be confirmed. Log tail:\n{self.log_tail()}"
            )
        self.policy_line = match.group(0).split(" - ")[0].strip()
        if "dspy_policy=not installed" in self.policy_line:
            raise SoakError(
                "The server did not reach main()/install_policy() — it logged "
                f"'{self.policy_line}'. This soak would measure a process without the "
                "DSPy memory controls and prove nothing."
            )
        for marker in ("dspy_history=off (asserted)", "dspy_trace=off (asserted)",
                       "dspy_policy_owner=claimed"):
            if marker not in self.policy_line:
                raise SoakError(
                    f"DSPy memory policy incomplete: expected '{marker}' in the startup "
                    f"record but got '{self.policy_line}'."
                )
        self.policy_check = "asserted from the server's own startup record"

    def _verify_tree_identity(self) -> None:
        """Prove from the RUNNING server which tree's code it is executing.

        A correct PYTHONPATH is a claim about the launch, not about the process.
        The readiness probe is the discriminator: Release A added a ``memory``
        query parameter to it, the baseline has none. If the observed shape does
        not match the requested tree, the import redirect silently failed and
        every number after this point would be attributed to the wrong tree.
        """
        conn = HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("GET", "/probes/readyz?memory=true")
        body = _decode(conn.getresponse().read())
        conn.close()
        self.memory_metrics_available = "memory" in body

        expected = not self.baseline
        if self.memory_metrics_available != expected:
            tree = "baseline" if self.baseline else "treatment"
            raise SoakError(
                f"Server launched against the {tree} tree at {self.server_tree}, but "
                f"/probes/readyz?memory=true "
                f"{'returned' if self.memory_metrics_available else 'did not return'} a "
                f"memory object, which is the opposite of what that tree's code does. "
                f"The PYTHONPATH redirect did not take effect; refusing to attribute "
                f"measurements to the wrong tree."
            )

    def _read_log(self) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""

    def log_tail(self, lines: int = 25) -> str:
        return "\n".join(self._read_log().splitlines()[-lines:])

    def durable_store_metrics(self) -> dict[str, Any]:
        """Record count and physical bytes of the durable conversation store."""
        conversations_dir = os.path.join(self.speeddict_dir, "channel_conversations")
        records = 0
        if os.path.isdir(conversations_dir):
            records = sum(name.endswith(".rdb") for name in os.listdir(conversations_dir))
        return {
            "records": records,
            "conversation_bytes": _dir_bytes(conversations_dir),
            "speeddict_bytes": _dir_bytes(self.speeddict_dir),
        }

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if self.graceful:
                # Opt-in only. The graceful path calls generate_topic_and_summary()
                # once per live session, i.e. one real provider call per channel —
                # for arm A0 that is one LLM call per request in the run.
                self._signal_group(signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=self.grace_seconds)
            if self.proc.poll() is None:
                self._signal_group(signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=15)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _signal_group(self, sig: int) -> None:
        # The whole group, because the server is started in its own session: a
        # stray uvicorn child must not outlive the harness and hold the port.
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(OSError):
                self.proc.send_signal(sig)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def __enter__(self) -> "ServerProcess":
        try:
            self.start()
        except BaseException:
            self.stop()
            self.cleanup()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
        self.cleanup()


def resolve_server_import(paths: Paths, server_tree: str, baseline: bool) -> dict[str, Any]:
    """Resolve, in a real subprocess, which fastworkflow the server will import.

    Run once per soak rather than per replicate. The editable install puts the
    working tree on sys.path unconditionally, so "I set PYTHONPATH" is not
    evidence; this asks the interpreter that will actually run the server.
    """
    if not os.path.isfile(os.path.join(server_tree, "fastworkflow", "__init__.py")):
        raise SoakError(
            f"--server-tree {server_tree} does not contain fastworkflow/__init__.py"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [server_tree] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    probe_source = (
        "import json, importlib.util, fastworkflow;"
        "print(json.dumps({"
        "'fastworkflow_file': fastworkflow.__file__,"
        "'has_server_memory': importlib.util.find_spec("
        "'fastworkflow.run_fastapi_mcp.server_memory') is not None}))"
    )
    completed = subprocess.run(
        [paths.python, "-c", probe_source],
        cwd=server_tree, env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise SoakError(
            f"Could not resolve the server's fastworkflow import under "
            f"PYTHONPATH={server_tree}: {completed.stderr[-800:]}"
        )
    resolved = json.loads(completed.stdout.strip().splitlines()[-1])

    expected_root = os.path.realpath(server_tree)
    actual_root = os.path.realpath(
        os.path.dirname(os.path.dirname(resolved["fastworkflow_file"]))
    )
    if actual_root != expected_root:
        raise SoakError(
            f"PYTHONPATH did not win over the editable-install .pth: asked for "
            f"{expected_root} but the interpreter imported fastworkflow from "
            f"{actual_root}. Refusing to measure an unverified tree."
        )
    if baseline and resolved["has_server_memory"]:
        raise SoakError(
            f"--baseline was requested but {server_tree} exposes "
            f"fastworkflow.run_fastapi_mcp.server_memory, so it is not a "
            f"pre-Release-A tree."
        )
    if not baseline and not resolved["has_server_memory"]:
        raise SoakError(
            f"A treatment run was requested but {server_tree} has no "
            f"fastworkflow.run_fastapi_mcp.server_memory."
        )
    resolved["server_tree"] = expected_root
    resolved["baseline"] = baseline
    return resolved


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            # RocksDB rotates files under us; a vanished file is not a failure.
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Client:
    """Keep-alive HTTP/1.1 client.

    Reconnecting per request would fold TCP setup into every latency sample and
    swamp the differences §16.6 is trying to resolve.
    """

    def __init__(self, port: int, timeout: float = 300.0):
        self.port = port
        self.timeout = timeout
        self.conn = HTTPConnection("127.0.0.1", port, timeout=timeout)

    def post(self, path: str, body: dict, headers: Optional[dict] = None
             ) -> tuple[int, dict, float]:
        payload = json.dumps(body).encode()
        request_headers = {"Content-Type": "application/json",
                           "Content-Length": str(len(payload))}
        request_headers |= headers or {}
        return self._send("POST", path, payload, request_headers)

    def get(self, path: str) -> tuple[int, dict, float]:
        return self._send("GET", path, None, {})

    def _send(self, method: str, path: str, payload, headers) -> tuple[int, dict, float]:
        for attempt in (0, 1):
            started = time.perf_counter()
            try:
                self.conn.request(method, path, body=payload, headers=headers)
                resp = self.conn.getresponse()
                raw = resp.read()
                elapsed = time.perf_counter() - started
                return resp.status, _decode(raw), elapsed
            except (OSError, HTTPException) as exc:
                # A keep-alive connection can be reaped between requests; one
                # reconnect is recovery, two in a row is a real server failure.
                self.close()
                self.conn = HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                if attempt == 1:
                    raise SoakError(f"{method} {path} failed twice: {exc!r}") from exc
        raise AssertionError("unreachable")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()


def _decode(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"_raw": raw[:500].decode("utf-8", "replace")}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def make_payload(kilobytes: int, fill: str = "random") -> str:
    """A fresh, unique ~N KB string. Base64 output, so JSON-safe and wire-accurate.

    ``random`` is the measurement default: nothing upstream can dedupe or cache it,
    and RocksDB's block compression cannot shrink it, so durable bytes reflect the
    payload actually written.

    ``compressible`` is an attribution instrument, not a workload. It keeps the
    string length, the uniqueness, and therefore every Python-side allocation and
    JSON encode identical, but makes the bytes trivially compressible so RocksDB
    stores a fraction of them. Running the same arm both ways separates "RSS
    tracks the durable store" from "RSS tracks allocation churn" without touching
    production code.
    """
    if fill == "random":
        return base64.b64encode(os.urandom(kilobytes * 1024 * 3 // 4)).decode("ascii")
    if fill != "compressible":
        raise SoakError(f"unknown payload fill {fill!r}")
    # Unique prefix so no layer can collapse two requests into one, then a constant
    # run that Snappy reduces to near nothing.
    prefix = uuid.uuid4().hex
    body = b"\0" * (kilobytes * 1024 * 3 // 4 - len(prefix))
    return prefix + base64.b64encode(body).decode("ascii")


def make_action(payload: str) -> dict:
    """A direct add_two_numbers action carrying the payload.

    InitializationRequest has no free-form ``context`` field, so the payload
    travels as an extra action parameter. That is the field that actually reaches
    the retention path under test: WorkflowExecutionContext._process_action puts
    ``action.parameters`` verbatim into the record it appends to conversation
    history, which is also what the ConversationStore persists. The command's
    pydantic Input model ignores the extra key, so the command still executes.
    """
    return {
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 2.0, "second_num": 3.0, "soak_payload": payload},
    }


# ---------------------------------------------------------------------------
# Samples and statistics
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    index: int
    rss_bytes: int
    uss_bytes: int
    cgroup_bytes: Optional[int]
    latency_s: float
    http_status: int


@dataclass
class Probe:
    index: int
    # Retention metrics come from the server's readiness probe, which only Release A
    # exposes; on the baseline they are genuinely unknown rather than zero.
    live_sessions: Optional[int]
    retained_turns: Optional[int]
    dspy_cache_entries: Optional[int]
    dspy_cache_bytes: Optional[int]
    conversation_turns: Optional[int]
    conversation_bytes: Optional[int]
    # Durable-store figures are read from the filesystem, so they exist for both trees.
    durable_records: int
    durable_conversation_bytes: int
    durable_speeddict_bytes: int
    metrics_available: bool = True


@dataclass
class Replicate:
    replicate: int
    pid: int
    port: int
    policy_line: str
    memory_source: str
    cgroup_path: Optional[str]
    cgroup_reason: Optional[str]
    server_tree: str = ""
    baseline: bool = False
    policy_check: str = ""
    memory_metrics_available: bool = True
    samples: list[Sample] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    cap_violations: list[str] = field(default_factory=list)
    forced_gc: Optional[dict[str, Any]] = None


def least_squares_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def second_half_slope_mb(samples: list[Sample], attr: str) -> float:
    """MB per request over the second half only, per §16.5."""
    half = samples[len(samples) // 2:]
    if len(half) < 2:
        return float("nan")
    xs = [float(s.index) for s in half]
    ys = [getattr(s, attr) / (1024 * 1024) for s in half]
    return least_squares_slope(xs, ys)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return next((value for key, value in sorted(_T95.items()) if df <= key), 1.645)


def summarize_slopes(slopes: list[float]) -> dict[str, Any]:
    """Aggregate replicate slopes with an upper bound, never a bare point estimate."""
    usable = [s for s in slopes if not math.isnan(s)]
    if not usable:
        return {"replicates": 0, "note": "no usable slopes"}
    summary: dict[str, Any] = {
        "replicates": len(usable),
        "slopes": usable,
        "mean": statistics.fmean(usable),
        "max": max(usable),
        "spread": max(usable) - min(usable),
    }
    if len(usable) >= 2:
        stdev = statistics.stdev(usable)
        stderr = stdev / math.sqrt(len(usable))
        summary |= {
            "stdev": stdev,
            "stderr": stderr,
            "upper_bound_95": statistics.fmean(usable) + t95(len(usable) - 1) * stderr,
            "upper_bound_method": f"mean + t(0.95, df={len(usable) - 1}) * stderr",
        }
    else:
        summary |= {
            "upper_bound_95": max(usable),
            "upper_bound_method": "single replicate: max slope, no interval computable",
        }
    return summary


# ---------------------------------------------------------------------------
# Probing the server's own retention metrics
# ---------------------------------------------------------------------------

def read_probe(client: Client, server: ServerProcess, index: int) -> Probe:
    status, body, _ = client.get("/probes/readyz?memory=true")
    if status != 200:
        raise SoakError(
            f"/probes/readyz?memory=true returned HTTP {status}: {str(body)[:400]}. "
            "A non-ready server invalidates every sample after this point."
        )
    # The baseline probe has no ``memory`` parameter and ignores the query string,
    # so it answers 200 with no metrics. Absent is recorded as unknown; inventing
    # zeros there would fabricate a "caps held" result for a tree that has no caps.
    memory = body.get("memory")
    if memory is None and not server.baseline:
        raise SoakError(
            "/probes/readyz?memory=true returned no memory object on a treatment "
            "server, which should always expose one. Body: " + str(body)[:400]
        )
    durable = server.durable_store_metrics()
    return Probe(
        index=index,
        live_sessions=memory["live_sessions"] if memory else None,
        retained_turns=memory["retained_turns"] if memory else None,
        dspy_cache_entries=memory["dspy_cache"]["entries"] if memory else None,
        dspy_cache_bytes=memory["dspy_cache"]["approx_bytes"] if memory else None,
        conversation_turns=memory["conversations"]["turns"] if memory else None,
        conversation_bytes=memory["conversations"]["approx_bytes"] if memory else None,
        durable_records=durable["records"],
        durable_conversation_bytes=durable["conversation_bytes"],
        durable_speeddict_bytes=durable["speeddict_bytes"],
        metrics_available=memory is not None,
    )


def check_caps(probe: Probe, arm: str) -> list[str]:
    if not probe.metrics_available:
        # Nothing to check: the baseline exposes no retention metrics, and it has
        # no caps to hold in the first place.
        return []
    violations = []
    if probe.retained_turns > CAP_RETAINED_TURNS:
        violations.append(
            f"request {probe.index}: retained_turns={probe.retained_turns} > {CAP_RETAINED_TURNS}"
        )
    if probe.dspy_cache_entries > CAP_DSPY_CACHE_ENTRIES:
        violations.append(
            f"request {probe.index}: dspy_cache.entries={probe.dspy_cache_entries} "
            f"> {CAP_DSPY_CACHE_ENTRIES}"
        )
    if arm == "a1":
        # conversations.turns is a process-wide SUM over live sessions, so it only
        # reads as a per-channel window while arm A1's single channel is the only
        # live session. Check that premise rather than assuming it, or a stray
        # second session would turn a real cap breach into a silent pass.
        if probe.live_sessions != 1:
            violations.append(
                f"request {probe.index}: arm A1 expects exactly 1 live session but saw "
                f"{probe.live_sessions}; the per-channel conversation-turn cap cannot "
                f"be read from the process-wide total"
            )
        elif probe.conversation_turns > CAP_CONVERSATION_TURNS:
            violations.append(
                f"request {probe.index}: conversations.turns={probe.conversation_turns} "
                f"> {CAP_CONVERSATION_TURNS}"
            )
    return violations


def forced_gc_availability(client: Client) -> dict[str, Any]:
    """Whether a forced-gc.collect() series can be produced against this server.

    Checked rather than asserted: the server is a separate process, so the only
    honest way to force collection is a server-side trigger. Adding one to
    production code is out of scope, so if none is exposed the forced-GC series
    is reported as not produced, with the reason.
    """
    status, body, _ = client.get("/openapi.json")
    if status != 200:
        return {"available": False,
                "reason": f"could not read /openapi.json (HTTP {status}) to look for a "
                          "server-side collection trigger"}
    paths = list((body.get("paths") or {}).keys())
    triggers = [p for p in paths if re.search(r"gc|collect|debug/mem", p, re.IGNORECASE)]
    if triggers:
        return {"available": True, "endpoints": triggers}
    return {
        "available": False,
        "endpoints_scanned": len(paths),
        "reason": (
            "The server exposes no endpoint that forces gc.collect() in its own process, "
            "and the soak drives it over HTTP from a separate process, so a forced-GC "
            "series cannot be produced without adding a debug endpoint to production "
            "code. Only the natural-GC series below was measured."
        ),
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_replicate(paths: Paths, args, replicate_index: int) -> Replicate:
    with ServerProcess(paths, startup_timeout=args.startup_timeout,
                       graceful=args.graceful_shutdown,
                       grace_seconds=args.grace_seconds,
                       baseline=args.baseline,
                       server_tree=args.server_tree) as server:
        client = Client(server.port)
        try:
            result = Replicate(
                replicate=replicate_index,
                pid=server.proc.pid,
                port=server.port,
                policy_line=server.policy_line or "",
                memory_source=server.probe.source,
                cgroup_path=server.probe.cgroup_path,
                cgroup_reason=server.probe.cgroup_reason,
                server_tree=server.server_tree,
                baseline=server.baseline,
                policy_check=server.policy_check,
                memory_metrics_available=bool(server.memory_metrics_available),
            )
            if replicate_index == 0:
                result.forced_gc = forced_gc_availability(client)

            token = _warmup(client, args)
            first_probe = read_probe(client, server, 0)
            result.probes.append(first_probe)
            result.cap_violations += check_caps(first_probe, args.arm)

            for index in range(1, args.requests + 1):
                sample = _measured_request(client, server, args, index, token)
                result.samples.append(sample)
                if index % args.probe_every == 0 or index == args.requests:
                    probe = read_probe(client, server, index)
                    result.probes.append(probe)
                    result.cap_violations += check_caps(probe, args.arm)
            return result
        finally:
            client.close()


def _warmup(client: Client, args) -> Optional[str]:
    """One excluded warm-up request; for arm A1 it also mints the channel's token."""
    channel = f"soak-{args.arm}-{uuid.uuid4().hex}"
    body = {
        "channel_id": channel,
        "user_id": "soak",
        "startup_action": make_action(make_payload(args.payload_kb, args.payload_fill)),
        # Generous, so the startup turn never defers into a 202 that would turn a
        # latency sample into a poll loop.
        "timeout_seconds": args.request_timeout,
    }
    status, response, _ = client.post("/initialize", body)
    _require_200(status, response, "/initialize (warm-up)")
    if args.arm != "a1":
        return None
    token = response.get("access_token")
    if not token:
        raise SoakError(
            f"/initialize returned 200 without an access_token, so arm A1 cannot pin "
            f"subsequent turns to one channel. Body: {str(response)[:300]}"
        )
    return token


def _measured_request(client: Client, server: ServerProcess, args, index: int,
                      token: Optional[str]) -> Sample:
    payload = make_payload(args.payload_kb, args.payload_fill)
    if args.arm == "a0":
        # Unique channel per request: the motivating production shape.
        body = {
            "channel_id": f"soak-a0-{index}-{uuid.uuid4().hex}",
            "user_id": "soak",
            "startup_action": make_action(payload),
            "timeout_seconds": args.request_timeout,
        }
        status, response, latency = client.post("/initialize", body)
        _require_200(status, response, f"/initialize (request {index})")
    else:
        status, response, latency = client.post(
            "/perform_action",
            {"action": make_action(payload), "timeout_seconds": args.request_timeout},
            {"Authorization": f"Bearer {token}"},
        )
        _require_200(status, response, f"/perform_action (request {index})")

    rss, uss, cgroup = server.probe.sample()
    return Sample(index=index, rss_bytes=rss, uss_bytes=uss, cgroup_bytes=cgroup,
                  latency_s=latency, http_status=status)


def _require_200(status: int, body: dict, what: str) -> None:
    if status == 200:
        return
    if status == 202:
        raise SoakError(
            f"{what} deferred (202) instead of completing inline. Raise "
            f"--request-timeout; a deferred turn would make this sample's latency "
            f"meaningless. Body: {str(body)[:300]}"
        )
    raise SoakError(f"{what} returned HTTP {status}: {str(body)[:400]}")


# ---------------------------------------------------------------------------
# Latency matrix (§16.6, Release A scope)
# ---------------------------------------------------------------------------

def run_latency(paths: Paths, args) -> dict[str, Any]:
    with ServerProcess(paths, startup_timeout=args.startup_timeout,
                       graceful=args.graceful_shutdown,
                       grace_seconds=args.grace_seconds,
                       baseline=args.baseline,
                       server_tree=args.server_tree) as server:
        client = Client(server.port)
        try:
            results: dict[str, Any] = {"server_tree": server.server_tree,
                                       "baseline": server.baseline}

            # First, on a still-empty registry: below MAX_RETAINED_STARTUP_TURNS
            # no retention sweep has run, which is the "without overflow" case.
            turn_latencies = []
            for index in range(CAP_RETAINED_TURNS - 1):
                status, response, latency = client.post("/initialize", {
                    "channel_id": f"lat-turn-{index}-{uuid.uuid4().hex}",
                    "user_id": "soak",
                    "startup_action": make_action(make_payload(args.payload_kb, args.payload_fill)),
                    "timeout_seconds": args.request_timeout,
                })
                _require_200(status, response, f"/initialize (turn completion {index})")
                turn_latencies.append(latency)
            overflow_probe = read_probe(client, server, CAP_RETAINED_TURNS - 1)
            results["turn_completion_without_overflow"] = _latency_block(
                turn_latencies,
                "startup turn on a fresh channel while retained turns stay below the cap",
                # A baseline server reports no retention metrics at all, which is
                # the point of comparing against it: it retains without a cap.
                extra={"retained_turns_at_end": overflow_probe.retained_turns,
                       "overflowed": (
                           None if overflow_probe.retained_turns is None
                           else overflow_probe.retained_turns >= CAP_RETAINED_TURNS
                       )},
            )

            # No-eviction live-session hit: one warm channel, far below the 2000
            # session cap, so every request is a cache hit with no eviction work.
            status, response, _ = client.post("/initialize", {
                "channel_id": f"lat-hit-{uuid.uuid4().hex}",
                "user_id": "soak",
                "startup_action": make_action(make_payload(args.payload_kb, args.payload_fill)),
                "timeout_seconds": args.request_timeout,
            })
            _require_200(status, response, "/initialize (no-eviction warm-up)")
            headers = _auth(response)
            hit_latencies = []
            for index in range(args.latency_samples):
                status, response, latency = client.post(
                    "/perform_action",
                    {"action": make_action(make_payload(args.payload_kb, args.payload_fill)),
                     "timeout_seconds": args.request_timeout},
                    headers,
                )
                _require_200(status, response, f"/perform_action (no-eviction {index})")
                hit_latencies.append(latency)
            results["no_eviction_live_session_hit"] = _latency_block(
                hit_latencies, "direct action on a resident channel, no eviction"
            )

            # Conversation append at depth: turn 1 vs turn N on ONE channel. This
            # is where the old rewrite-the-whole-conversation save was O(N^2).
            status, response, _ = client.post("/initialize", {
                "channel_id": f"lat-depth-{uuid.uuid4().hex}",
                "user_id": "soak",
                "startup_action": make_action(make_payload(args.payload_kb, args.payload_fill)),
                "timeout_seconds": args.request_timeout,
            })
            _require_200(status, response, "/initialize (depth warm-up)")
            headers = _auth(response)
            depth_latencies = []
            for index in range(args.latency_depth):
                status, response, latency = client.post(
                    "/perform_action",
                    {"action": make_action(make_payload(args.payload_kb, args.payload_fill)),
                     "timeout_seconds": args.request_timeout},
                    headers,
                )
                _require_200(status, response, f"/perform_action (depth {index})")
                depth_latencies.append(latency)
            window = max(1, min(20, args.latency_depth // 4))
            results["conversation_append_shallow"] = _latency_block(
                depth_latencies[:window],
                f"turns 1-{window} on one channel",
            )
            results["conversation_append_at_depth"] = _latency_block(
                depth_latencies[-window:],
                f"turns {args.latency_depth - window + 1}-{args.latency_depth} on the same channel",
            )
            # Both figures are process-wide, not per-channel: this run left the 19
            # overflow-free channels and the no-eviction channel resident too.
            depth_probe = read_probe(client, server, args.latency_depth)
            results["conversation_append_at_depth"].update(
                process_wide_conversation_turns=depth_probe.conversation_turns,
                process_wide_live_sessions=depth_probe.live_sessions,
                process_wide_durable_conversation_bytes=depth_probe.durable_conversation_bytes,
            )
            results["raw_depth_latencies_s"] = depth_latencies
            results["policy_line"] = server.policy_line
            return results
        finally:
            client.close()


def _auth(init_response: dict) -> dict[str, str]:
    token = init_response.get("access_token")
    if not token:
        raise SoakError(
            f"/initialize returned 200 without an access_token: {str(init_response)[:300]}"
        )
    return {"Authorization": f"Bearer {token}"}


def _latency_block(values: list[float], description: str,
                   extra: Optional[dict] = None) -> dict[str, Any]:
    block = {
        "description": description,
        "n": len(values),
        "p50_ms": percentile(values, 0.50) * 1000,
        "p95_ms": percentile(values, 0.95) * 1000,
        "min_ms": (min(values) if values else float("nan")) * 1000,
        "max_ms": (max(values) if values else float("nan")) * 1000,
        "samples_s": values,
    }
    return block | (extra or {})


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mb(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / (1024 * 1024):8.1f}"


def _num(value: Optional[int]) -> str:
    return "null" if value is None else str(value)


def print_soak_report(args, paths: Paths, replicates: list[Replicate]) -> dict[str, Any]:
    """Print the human-readable report and return the structured verdict.

    The verdict distinguishes "the slope gate was met" from "this run is allowed
    to fail the build", because below the §16.5 sample minimum the gate is
    advisory. A single boolean would let an artifact reader mistake a
    non-blocking advisory failure for a pass.
    """
    first = replicates[0]
    forced_gc = first.forced_gc or {}
    tree_label = "BASELINE (pre-Release-A)" if first.baseline else "TREATMENT (Release A)"
    print("=" * 78)
    print(f"MEMORY SOAK — arm {args.arm.upper()}  (design §16.5)")
    print(f"MEASURED TREE     : {tree_label}")
    print(f"                    {first.server_tree}")
    print("=" * 78)
    print(f"workflow          : {paths.workflow_name} ({paths.workflow_path})")
    print(f"interpreter       : {paths.python}")
    print(f"requests/replicate: {args.requests} measured (+1 warm-up, excluded)")
    print(f"replicates        : {args.replicates} fresh server processes")
    print(f"payload           : {args.payload_kb} KB, unique per request "
          f"(base64, fill={args.payload_fill})")
    print("payload field     : startup_action/action parameters['soak_payload']")
    print(f"memory source     : {first.memory_source}")
    cgroup_source = first.cgroup_path or f"unavailable — {first.cgroup_reason}"
    print(f"cgroup source     : {cgroup_source}")
    print(f"policy check      : {first.policy_check}")
    print(f"dspy policy       : {first.policy_line or 'n/a (no policy on this tree)'}")
    if not first.memory_metrics_available:
        print("retention metrics : UNAVAILABLE — /probes/readyz on this tree takes no "
              "'memory'")
        print("                    parameter, so live_sessions / retained_turns / "
              "dspy_cache /")
        print("                    conversations are recorded as null, not zero. Durable-"
              "store")
        print("                    figures below are read from the filesystem and are real.")
    print()

    print("-" * 78)
    print("GC SERIES")
    print("-" * 78)
    print("natural GC        : measured (series below)")
    if forced_gc.get("available"):
        print(f"forced gc.collect(): available via {forced_gc.get('endpoints')}")
    else:
        print("forced gc.collect(): NOT PRODUCED")
        print(f"  reason: {forced_gc.get('reason')}")
    print()

    for rep in replicates:
        print("-" * 78)
        print(f"REPLICATE {rep.replicate + 1}/{len(replicates)}  (server pid {rep.pid}, "
              f"port {rep.port})")
        print("-" * 78)
        print(f"  {'req':>6} {'RSS MB':>9} {'USS MB':>9} {'cgroup MB':>10} "
              f"{'live':>6} {'ret':>5} {'dspy':>5} {'convN':>6} {'convMB':>9} "
              f"{'durRec':>7} {'durMB':>9}")
        by_index = {s.index: s for s in rep.samples}
        for probe in rep.probes:
            sample = by_index.get(probe.index)
            print(f"  {probe.index:>6} "
                  f"{_mb(sample.rss_bytes) if sample else '      n/a':>9} "
                  f"{_mb(sample.uss_bytes) if sample else '      n/a':>9} "
                  f"{_mb(sample.cgroup_bytes) if sample else '      n/a':>10} "
                  f"{_num(probe.live_sessions):>6} {_num(probe.retained_turns):>5} "
                  f"{_num(probe.dspy_cache_entries):>5} "
                  f"{_num(probe.conversation_turns):>6} "
                  f"{_mb(probe.conversation_bytes):>9} "
                  f"{probe.durable_records:>7} {_mb(probe.durable_conversation_bytes):>9}")
        rss_slope = second_half_slope_mb(rep.samples, "rss_bytes")
        uss_slope = second_half_slope_mb(rep.samples, "uss_bytes")
        latencies = [s.latency_s for s in rep.samples]
        print(f"  second-half slope: RSS {rss_slope:+.5f} MB/req   "
              f"USS {uss_slope:+.5f} MB/req")
        print(f"  request latency  : p50 {percentile(latencies, 0.5) * 1000:.1f} ms   "
              f"p95 {percentile(latencies, 0.95) * 1000:.1f} ms")
        print()

    rss_summary = summarize_slopes([second_half_slope_mb(r.samples, "rss_bytes")
                                    for r in replicates])
    uss_summary = summarize_slopes([second_half_slope_mb(r.samples, "uss_bytes")
                                    for r in replicates])
    print("-" * 78)
    print("SLOPE (least squares over the SECOND HALF of samples, MB/request)")
    print("-" * 78)
    for label, summary in (("RSS", rss_summary), ("USS", uss_summary)):
        print(f"  {label}: " + "  ".join(f"{s:+.5f}" for s in summary.get("slopes", [])))
        print(f"       mean {summary.get('mean', float('nan')):+.5f}   "
              f"max {summary.get('max', float('nan')):+.5f}   "
              f"spread {summary.get('spread', float('nan')):.5f}   "
              f"upper bound {summary.get('upper_bound_95', float('nan')):+.5f}")
        print(f"       ({summary.get('upper_bound_method')})")
    print()

    print("-" * 78)
    print("STRUCTURAL CAPS")
    print("-" * 78)
    if not first.memory_metrics_available:
        # Not "OK" and not "VIOLATED": the caps do not exist on this tree and the
        # instrument to read them does not either. Printing zeros here would be
        # the single most misleading thing this report could do.
        print("  NOT APPLICABLE — the baseline tree implements none of these caps and "
              "exposes")
        print("  no retention metrics to read them from. Only the durable-store columns "
              "above")
        print("  and the RSS/USS slopes below are comparable across trees.")
        print()
        return _finish_verdict(args, replicates, rss_summary, [], first)
    max_retained = max((p.retained_turns for r in replicates for p in r.probes), default=0)
    max_dspy = max((p.dspy_cache_entries for r in replicates for p in r.probes), default=0)
    max_conv = max((p.conversation_turns for r in replicates for p in r.probes), default=0)
    max_live = max((p.live_sessions for r in replicates for p in r.probes), default=0)
    print(f"  retained_turns      max {max_retained:>6}  cap {CAP_RETAINED_TURNS:>6}  "
          f"{'OK' if max_retained <= CAP_RETAINED_TURNS else 'VIOLATED'}")
    print(f"  dspy_cache.entries  max {max_dspy:>6}  cap {CAP_DSPY_CACHE_ENTRIES:>6}  "
          f"{'OK' if max_dspy <= CAP_DSPY_CACHE_ENTRIES else 'VIOLATED'}")
    if args.arm == "a1":
        print(f"  conversations.turns max {max_conv:>6}  cap {CAP_CONVERSATION_TURNS:>6}  "
              f"{'OK' if max_conv <= CAP_CONVERSATION_TURNS else 'VIOLATED'}   "
              f"(process-wide total; arm A1 keeps exactly {max_live} live channel)")
    else:
        print(f"  conversations.turns max {max_conv:>6}  (process-wide total: arm A0 "
              f"holds one turn per live channel, so this tracks live_sessions)")
    print(f"  live_sessions       max {max_live:>6}  cap   2000  (unchanged in Release A)")
    violations = [v for r in replicates for v in r.cap_violations]
    for violation in violations:
        print(f"  VIOLATION: {violation}")
    print()
    return _finish_verdict(args, replicates, rss_summary, violations, first)


def _finish_verdict(args, replicates: list[Replicate], rss_summary: dict,
                    violations: list[str], first: Replicate) -> dict[str, Any]:
    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    upper = rss_summary.get("upper_bound_95", float("nan"))
    binding = args.requests >= MIN_REQUESTS_FOR_BINDING_GATE
    verdict: dict[str, Any] = {
        "arm": args.arm,
        "baseline": first.baseline,
        "server_tree": first.server_tree,
        "caps_held": None if first.baseline else not violations,
        "cap_violations": violations,
        "rss_slope_upper_bound_mb_per_request": upper,
        "slope_target_mb_per_request": SLOPE_TARGET_MB_PER_REQUEST,
        # The gate is a Release A acceptance criterion. Measuring the baseline is
        # how the slope gets attributed, not a pass/fail event for the old code.
        "slope_gate_applies": args.arm == "a1" and not first.baseline,
        "slope_gate_binding": binding,
    }

    if first.baseline:
        print("  BASELINE REFERENCE RUN — no gate is evaluated. Release A's §16.5 gate "
              "applies")
        print("  to the treatment tree; this run exists so the treatment slope can be "
              "attributed")
        print(f"  rather than guessed at. RSS slope upper bound {upper:+.5f} MB/request.")
        if args.arm == "a1":
            print(f"  in-memory conversation bytes: {_plateau_note(replicates)}")
        verdict["run_failed"] = False
        print()
        print("  These are raw samples and slopes. Do not restate this run as "
              "'survived N requests',")
        print("  and do not sum ablation deltas as independent shares (§16.5).")
        print()
        return verdict

    if args.arm == "a1":
        conv_note = _plateau_note(replicates)
        print(f"  in-memory conversation bytes: {conv_note}")
        meets = not math.isnan(upper) and upper <= SLOPE_TARGET_MB_PER_REQUEST
        label = "PASS" if meets else "FAIL"
        verdict |= {"conversation_bytes": conv_note, "slope_gate_met": meets}
        if binding:
            print(f"  GATE (Release A, arm A1): RSS slope upper bound {upper:+.5f} "
                  f"<= {SLOPE_TARGET_MB_PER_REQUEST} MB/request ... {label}")
        else:
            print(f"  GATE (Release A, arm A1): RSS slope upper bound {upper:+.5f} vs "
                  f"{SLOPE_TARGET_MB_PER_REQUEST} MB/request ... {label} (ADVISORY)")
            print(f"    ADVISORY because {args.requests} measured requests is below the "
                  f"§16.5 minimum of {MIN_REQUESTS_FOR_BINDING_GATE}; at this sample "
                  f"count the slope is dominated by allocator warm-up.")
    else:
        print(f"  Arm A0 is RECORDED, NOT GATED (§16.5). RSS slope upper bound "
              f"{upper:+.5f} MB/request.")
        print(f"    Pre-registered expectation: materially below the unpatched baseline "
              f"but ABOVE {SLOPE_TARGET_MB_PER_REQUEST} MB/request, because the "
              f"live-session cache still retains every unique channel at the unchanged "
              f"2000 default.")
        observed = ("above" if upper > SLOPE_TARGET_MB_PER_REQUEST else "at or below")
        print(f"    Observed: {observed} the {SLOPE_TARGET_MB_PER_REQUEST} MB/request "
              f"target. Quote this number verbatim in §1.2 and the release notes.")
        print("    Structural caps above ARE gated for this arm.")
        verdict["slope_vs_target"] = observed

    # A cap breach always fails the run; the slope only does so once the sample
    # count makes the gate binding.
    verdict["run_failed"] = bool(violations) or (
        args.arm == "a1" and binding and not verdict.get("slope_gate_met", False)
    )

    print()
    print("  These are raw samples and slopes. Do not restate this run as "
          "'survived N requests',")
    print("  and do not sum ablation deltas as independent shares (§16.5).")
    print()
    return verdict


def _plateau_note(replicates: list[Replicate]) -> str:
    if not replicates[0].memory_metrics_available:
        return ("not measurable on this tree — it exposes no in-memory conversation "
                "byte counter; see the durable-store columns instead")
    values = []
    for rep in replicates:
        probes = [p for p in rep.probes if p.index > 0]
        if len(probes) >= 2:
            half = probes[len(probes) // 2:]
            slope = least_squares_slope(
                [float(p.index) for p in half],
                [p.conversation_bytes / (1024 * 1024) for p in half],
            )
            values.append((probes[-1].conversation_bytes, slope))
    if not values:
        return "not enough probe points to judge"
    final = max(v[0] for v in values)
    slopes = [v[1] for v in values if not math.isnan(v[1])]
    slope_text = (f"second-half slope {max(slopes):+.5f} MB/request"
                  if slopes else "slope not computable")
    return f"final {final / (1024 * 1024):.1f} MB, {slope_text}"


def print_latency_report(args, results: dict[str, Any]) -> None:
    print("=" * 78)
    print("LATENCY MATRIX — Release A scope (design §16.6)")
    print("=" * 78)
    print(f"payload: {args.payload_kb} KB unique per request; depth "
          f"{args.latency_depth} turns on one channel")
    print()
    order = [
        ("no_eviction_live_session_hit", "no-eviction live-session hit"),
        ("turn_completion_without_overflow", "turn completion without overflow"),
        ("conversation_append_shallow", "conversation append, shallow (turn 1)"),
        ("conversation_append_at_depth", "conversation append at depth"),
    ]
    print(f"  {'measurement':<42} {'n':>4} {'p50 ms':>9} {'p95 ms':>9}")
    for key, label in order:
        if block := results.get(key):
            print(f"  {label:<42} {block['n']:>4} {block['p50_ms']:>9.1f} "
                  f"{block['p95_ms']:>9.1f}")
    print()
    shallow = results.get("conversation_append_shallow")
    deep = results.get("conversation_append_at_depth")
    if shallow and deep and shallow["p50_ms"] > 0:
        ratio = deep["p50_ms"] / shallow["p50_ms"]
        print(f"  append-at-depth / shallow p50 ratio: {ratio:.2f}x  "
              f"(a superlinear save path shows up here)")
        print(f"  at end of the depth run, process-wide (all "
              f"{deep.get('process_wide_live_sessions')} live channels, not just the "
              f"depth channel):")
        print(f"    in-memory conversation turns : "
              f"{deep.get('process_wide_conversation_turns')}  "
              f"(per-channel window is {CAP_CONVERSATION_TURNS})")
        print(f"    durable conversation bytes   : "
              f"{deep.get('process_wide_durable_conversation_bytes', 0) / (1024 * 1024):.1f} MB")
    overflow = results.get("turn_completion_without_overflow")
    if overflow:
        print(f"  retained turns at end of the overflow-free run: "
              f"{overflow.get('retained_turns_at_end')} (cap {CAP_RETAINED_TURNS}, "
              f"overflowed={overflow.get('overflowed')})")
    print()
    print("  GATE (§16.6): no-eviction p50/p95 must regress by less than 5%, or less than")
    print("  measurement noise, whichever is larger. This harness measures the current")
    print("  tree only; it cannot construct the unpatched baseline, so NO VERDICT is")
    print("  emitted here. Run the same command on the baseline tree and compare.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory/latency soak for the run_fastapi_mcp server (§16.5, §16.6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", choices=["a0", "a1"], default="a1",
                        help="a0 = unique channel per request (recorded); "
                             "a1 = one hot channel (gated)")
    parser.add_argument("--requests", type=int, default=300,
                        help="measured requests per replicate, excluding the warm-up")
    parser.add_argument("--replicates", type=int, default=3,
                        help="fresh server processes; more than one is what makes an "
                             "upper confidence bound possible")
    parser.add_argument("--payload-kb", type=int, default=450,
                        help="unique payload size per request, in KB")
    parser.add_argument("--payload-fill", choices=["random", "compressible"],
                        default="random",
                        help="'random' is the measurement default. 'compressible' keeps "
                             "the same size and uniqueness but shrinks what RocksDB "
                             "stores, to attribute RSS growth to the durable store "
                             "versus allocation churn")
    parser.add_argument("--probe-every", type=int, default=25,
                        help="how often to read /probes/readyz?memory=true")
    parser.add_argument("--latency", action="store_true",
                        help="also run the §16.6 latency matrix in its own server process")
    parser.add_argument("--latency-samples", type=int, default=60,
                        help="samples for the no-eviction latency measurement")
    parser.add_argument("--latency-depth", type=int, default=200,
                        help="turns on one channel for the append-at-depth measurement")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write raw samples, probes and slopes to this JSON file")
    parser.add_argument("--startup-timeout", type=float, default=300.0,
                        help="seconds to wait for /probes/readyz (cold import of "
                             "torch/transformers is slow)")
    parser.add_argument("--request-timeout", type=int, default=300,
                        help="timeout_seconds sent to the server, high enough that a "
                             "turn never defers into a 202")
    parser.add_argument("--graceful-shutdown", action="store_true",
                        help="SIGTERM before SIGKILL. Off by default because the "
                             "graceful path generates a topic/summary per live session, "
                             "i.e. one real LLM call per channel in the run")
    parser.add_argument("--grace-seconds", type=float, default=30.0,
                        help="grace period before SIGKILL when --graceful-shutdown is set")
    parser.add_argument("--baseline", action="store_true",
                        help="the server tree is pre-Release-A: skip the DSPy policy "
                             "assertion, accept a readiness probe with no memory object, "
                             "record retention metrics as null, and evaluate no gate")
    parser.add_argument("--server-tree", default=None,
                        help="checkout the SERVER subprocess imports fastworkflow from "
                             "(cwd + PYTHONPATH). The harness always runs from the "
                             "working tree. Defaults to the working tree.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests < 2:
        raise SoakError("--requests must be at least 2 to fit a second-half slope")
    if args.replicates < 1:
        raise SoakError("--replicates must be at least 1")

    paths = resolve_paths()
    args.server_tree = os.path.realpath(args.server_tree or paths.repo_root)
    started_at = time.time()

    # Resolve the server's import before spending minutes measuring it. The whole
    # baseline-vs-treatment comparison is worthless if the subprocess quietly
    # imported the working tree through the editable-install .pth.
    print(f"[soak] verifying the server's fastworkflow import under "
          f"PYTHONPATH={args.server_tree} ...", flush=True)
    import_check = resolve_server_import(paths, args.server_tree, args.baseline)
    print(f"[soak] server will import: {import_check['fastworkflow_file']}")
    print(f"[soak] server_memory present in that tree: "
          f"{import_check['has_server_memory']}")
    print(f"[soak] MEASURED TREE: "
          f"{'BASELINE (pre-Release-A)' if args.baseline else 'TREATMENT (Release A)'}",
          flush=True)

    replicates: list[Replicate] = []
    for index in range(args.replicates):
        print(f"[soak] replicate {index + 1}/{args.replicates}: starting a fresh server "
              f"(cold import can take ~60s)...", flush=True)
        replicates.append(run_replicate(paths, args, index))
        print(f"[soak] replicate {index + 1}/{args.replicates}: done", flush=True)

    latency_results: Optional[dict[str, Any]] = None
    if args.latency:
        print("[soak] latency matrix: starting a fresh server...", flush=True)
        latency_results = run_latency(paths, args)
        print("[soak] latency matrix: done", flush=True)

    print()
    verdict = print_soak_report(args, paths, replicates)
    if latency_results is not None:
        print_latency_report(args, latency_results)

    if args.json_out:
        artifact = {
            "arm": args.arm,
            "config": vars(args),
            "measured_tree": {
                "path": args.server_tree,
                "kind": "baseline" if args.baseline else "treatment",
                "fastworkflow_file": import_check["fastworkflow_file"],
                "has_server_memory": import_check["has_server_memory"],
                "harness_tree": paths.repo_root,
            },
            "workflow_path": paths.workflow_path,
            "started_at": started_at,
            "duration_s": time.time() - started_at,
            "gc_series": {
                "natural": "measured",
                "forced": replicates[0].forced_gc,
            },
            "caps": {
                "retained_turns": CAP_RETAINED_TURNS,
                "dspy_cache_entries": CAP_DSPY_CACHE_ENTRIES,
                "conversation_turns": CAP_CONVERSATION_TURNS,
            },
            "replicates": [
                {
                    "replicate": rep.replicate,
                    "pid": rep.pid,
                    "port": rep.port,
                    "policy_line": rep.policy_line,
                    "policy_check": rep.policy_check,
                    "server_tree": rep.server_tree,
                    "baseline": rep.baseline,
                    "memory_metrics_available": rep.memory_metrics_available,
                    "memory_source": rep.memory_source,
                    "cgroup_path": rep.cgroup_path,
                    "cgroup_reason": rep.cgroup_reason,
                    "cap_violations": rep.cap_violations,
                    "samples": [asdict(s) for s in rep.samples],
                    "probes": [asdict(p) for p in rep.probes],
                    "second_half_slope_mb_per_request": {
                        "rss": second_half_slope_mb(rep.samples, "rss_bytes"),
                        "uss": second_half_slope_mb(rep.samples, "uss_bytes"),
                    },
                }
                for rep in replicates
            ],
            "slope_summary": {
                "rss": summarize_slopes(
                    [second_half_slope_mb(r.samples, "rss_bytes") for r in replicates]),
                "uss": summarize_slopes(
                    [second_half_slope_mb(r.samples, "uss_bytes") for r in replicates]),
            },
            "latency": latency_results,
            "verdict": verdict,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            # No default= coercion: this artifact is the release gate's evidence,
            # and a measurement silently stringified is a measurement misreported.
            json.dump(artifact, handle, indent=2)
        print(f"[soak] raw samples written to {args.json_out}")

    return 1 if verdict["run_failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SoakError as error:
        print(f"\nSOAK PREREQUISITE / RUNTIME FAILURE:\n  {error}\n", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
