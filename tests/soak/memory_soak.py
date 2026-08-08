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
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Release C: checkpoint retention, and why the plateau gate needs a long run.
#
# Both constants are private to the server (utils.CHECKPOINT_REAP_INTERVAL_SECONDS
# and checkpoint_store.RetentionPolicy's defaults) with no env override, so they
# are duplicated here for the same reason the caps above are: the harness has to
# be able to notice if one changes underneath it. They are re-derived from the
# server's behaviour where possible and reported either way.
# ---------------------------------------------------------------------------
REAP_INTERVAL_SECONDS = 300.0     # utils.CHECKPOINT_REAP_INTERVAL_SECONDS
RETENTION_MAX_CHANNELS = 1000     # RetentionPolicy.max_channels default
RETENTION_MAX_AGE_SECONDS = 86_400.0   # RetentionPolicy.max_age_seconds default
# §16.5 wants the plateau to be shown repeating, not glimpsed once, so a plateau
# claim needs the run to span at least two reap passes.
REAP_PASSES_FOR_PLATEAU = 2

# Loss-adjacent lines the Release C code actually emits. §16.5 gates arm C on
# "zero context-loss sentinels" but never states the string, so the harness
# enumerates the real ones and reports each separately: collapsing them into one
# boolean would make a quarantine indistinguishable from a retry.
SENTINEL_PATTERNS = {
    "checkpoint_unreadable": (
        r"Checkpoint for channel_id \S+ could not be read"),
    "checkpoint_quarantined": (
        r"Checkpoint for channel_id \S+ could not be applied"),
    "checkpoint_write_failed": r"checkpoint write failed",
    "conversation_high_water_exceeded": r"Conversation high-water mark",
    # The precondition for a write from an executor whose caller has moved on.
    "streaming_delivery_deadline": r"passed its \d+s delivery deadline",
    # A unique channel is never revisited, so a restore means an id collided.
    "unexpected_checkpoint_restore": r"Restored checkpoint for channel_id",
    "launch_context_conflict": r"conflicts_resolved_to_launch=\[\S",
}

#: A retirement line plus the timestamp ``fastworkflow.utils.logging.LOG_FORMAT``
#: appends after the message. Arm C needs the timestamp, not just the channel id.
RETIRED_LINE = re.compile(
    r"Retired channel_id (\S+) at generation \d+ - "
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z"
)


#: Resolution of a parsed log timestamp, in seconds. See _parse_log_timestamp:
#: only the whole-second field is usable, so any ordering closer than this is
#: reported as ambiguous rather than decided.
LOG_TIMESTAMP_RESOLUTION_S = 1.0


def _parse_log_timestamp(text: str) -> Optional[float]:
    """The server's log stamp as an epoch comparable to ``time.time()``.

    Whole seconds only, and deliberately so. ``fastworkflow.utils.logging.format_ns``
    formats the seconds with ``tz=timezone.utc`` — so the trailing ``Z`` is
    accurate — but it builds the sub-second field as
    ``f"{(time_in_ns // 10**9) * 10**9}"[:9]``, which is the leading digits of the
    epoch nanosecond count rather than a fraction of a second: it is very nearly
    constant for the whole run. Parsing it would add a fixed ~0.18s bias dressed up
    as precision, so it is dropped and the caller treats the result as ±1s.

    Returns None rather than guessing if the format ever changes, so the caller can
    report the gap instead of silently mis-ordering.
    """
    head, _, _fraction_is_not_a_fraction = text.partition(".")
    try:
        stamp = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=timezone.utc).timestamp()

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
# Arm fixtures
# ---------------------------------------------------------------------------

# Renaming rather than deleting: the rename is reversible, greppable, and leaves
# the docstring in place, so anyone reading the temp fixture can see what was
# done to it and why.
DISABLED_HOOK_NAME = "get_state__disabled_by_soak_arm_b"


@dataclass
class Fixture:
    """The workflow an arm drives, and the direct action that exercises it.

    Split from ``Paths`` because arms A0/A1/A/C share one workflow while arm B
    needs a different one: the shape a soak measures is a property of the arm,
    not of the checkout.
    """

    name: str
    workflow_path: str
    command_name: str
    base_parameters: dict[str, Any]
    expect_pinned: bool
    note: str
    tempdir: Optional[str] = None

    def cleanup(self) -> None:
        if self.tempdir:
            shutil.rmtree(self.tempdir, ignore_errors=True)


def resolve_fixture(paths: Paths, arm: str) -> Fixture:
    """Pick the fixture the arm's pre-registered expectation is about.

    Arms A0, A1, A and C all want §16.5's "function-style workflow, no
    command-context object", which is exactly the hello_world shape: its
    ``_commands/`` tree has no context directory, so
    ``project_command_contexts`` returns NO_CONTEXT and the session is
    checkpointable — hence evictable. Arm A and arm A0 therefore differ only in
    the code under test, which is what makes their slopes comparable.
    """
    if arm != "b":
        return Fixture(
            name=paths.workflow_name,
            workflow_path=paths.workflow_path,
            command_name="add_two_numbers",
            base_parameters={"first_num": 2.0, "second_num": 3.0},
            expect_pinned=False,
            note="function-style workflow, no command-context object (§16.5 arms A0/A1/A/C)",
        )
    return build_pinned_fixture(paths)


def build_pinned_fixture(paths: Paths) -> Fixture:
    """Construct arm B's pinned fixture in a temp directory.

    §16.5 names ``simple_workflow_template`` as the pinned shape, but that
    premise has expired: it now ships serialization hooks, as do messaging_app_2/3/4
    and tests/todo_list_workflow, so all of them are evictable. Re-reading §16.5's
    intent — arm B exists to quantify §1.4, the *undeclared* workflow — the
    genuinely pinned shape today is a context class with no ``get_state``, which is
    what a workflow looks like before its author writes one.

    So: copy ``tests/todo_list_workflow`` and rename ``get_state`` in the COPY.
    Nothing under the repo's committed fixtures is touched. That workflow is
    chosen because it satisfies the other half of §16.5's arm B requirement that
    the bundled examples do not — ``_commands/startup.py`` assigns
    ``root_command_context``, so the session pins from turn 1 (§1.4, and the
    §16.5 note that arm B "must pin from turn 1 to be the adversarial case").
    Its ``___command_info`` is copied too, so no training step is needed.
    """
    source = os.path.join(paths.repo_root, "tests", "todo_list_workflow")
    hook_relpath = os.path.join("_commands", "TodoListManager", "_TodoListManager.py")
    startup_relpath = os.path.join("_commands", "startup.py")
    for relpath in (hook_relpath, startup_relpath,
                    os.path.join("___command_info", "routing_definition.json")):
        if not os.path.isfile(os.path.join(source, relpath)):
            raise SoakError(
                f"Arm B needs {source}/{relpath}, which is missing. That workflow is "
                f"the harness's pinned fixture donor; pick another workflow whose "
                f"_commands/startup.py assigns root_command_context and adjust "
                f"build_pinned_fixture()."
            )

    with open(os.path.join(source, startup_relpath), encoding="utf-8") as handle:
        if "root_command_context" not in handle.read():
            raise SoakError(
                f"{source}/{startup_relpath} no longer assigns root_command_context, so "
                f"the fixture would not pin from turn 1 and arm B would measure the "
                f"wrong thing (§16.5 arm B, §1.4)."
            )

    tempdir = tempfile.mkdtemp(prefix="fw_soak_pinned_")
    dest = os.path.join(tempdir, "todo_list_workflow_undeclared")
    # Runtime artifacts are excluded: they are another server's RocksDB state and
    # copying them would seed the arm with someone else's channels.
    shutil.copytree(
        source, dest,
        ignore=shutil.ignore_patterns("__pycache__", "___workflow_contexts",
                                      "___convo_info"),
    )

    hook_path = os.path.join(dest, hook_relpath)
    with open(hook_path, encoding="utf-8") as handle:
        original = handle.read()
    patched = original.replace("def get_state(", f"def {DISABLED_HOOK_NAME}(", 1)
    if patched == original:
        raise SoakError(
            f"Could not find 'def get_state(' in {source}/{hook_relpath}. Arm B "
            f"depends on removing that hook to make the context unprojectable; the "
            f"donor workflow's hook must have been renamed."
        )
    with open(hook_path, "w", encoding="utf-8") as handle:
        handle.write(patched)
    # Belt and braces: `hasattr(hooks, "get_state")` is what decides pinning, and a
    # second definition further down the file would defeat the rename above.
    if "def get_state(" in patched:
        raise SoakError(
            f"{hook_path} still defines get_state after patching, so the fixture "
            f"would remain evictable and arm B would silently measure arm A."
        )

    return Fixture(
        name="todo_list_workflow_undeclared",
        workflow_path=dest,
        command_name="startup",
        # startup has no Signature/Input class, so CommandExecutor.perform_action
        # calls the response generator without parameters — the payload still lands
        # in the conversation record via _process_action, which is the retention
        # path this measures.
        base_parameters={},
        expect_pinned=True,
        note=(f"copy of tests/todo_list_workflow with get_state renamed to "
              f"{DISABLED_HOOK_NAME}; root_command_context is assigned in startup, "
              f"so every channel pins from turn 1 (§16.5 arm B, §1.4)"),
        tempdir=tempdir,
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


def _entries(root: str, depth: int) -> list[str]:
    """Paths exactly ``depth`` levels below ``root``, directories or not."""
    level = [root] if os.path.isdir(root) else []
    for _ in range(depth):
        nxt = []
        for path in level:
            with contextlib.suppress(OSError):
                nxt += [entry.path for entry in os.scandir(path)]
        level = nxt
    return level


def _subdirectories(root: str, depth: int) -> list[str]:
    return [path for path in _entries(root, depth) if os.path.isdir(path)]


def _dir_usage(path: str) -> tuple[int, int, int]:
    """(files, apparent bytes, physical bytes) for one subtree.

    A faithful reimplementation of ``checkpoint_store._dir_usage``, including the
    two things that are easy to get wrong and that the cross-check caught: the
    blocks of *descendant directories* count toward physical bytes (a directory
    occupies space), and the subtree root's own blocks do not. Reimplemented rather
    than imported so the harness stays independent of the tree it measures; the
    cross-check is what keeps the copy honest.
    """
    files = apparent = physical = 0
    for directory, subdirs, names in os.walk(path, followlinks=False):
        for name in list(subdirs) + names:
            try:
                info = os.lstat(os.path.join(directory, name))
            except OSError:
                continue
            blocks = getattr(info, "st_blocks", None)
            if stat.S_ISDIR(info.st_mode):
                physical += int(blocks) * 512 if blocks else 0
                continue
            files += 1
            apparent += info.st_size
            physical += int(blocks) * 512 if blocks is not None else info.st_size
    return (files, apparent, physical)


class ServerProcess:
    """A fresh ``python -m fastworkflow.run_fastapi_mcp`` on a private port and store.

    ``server_tree`` selects which checkout the *server* imports fastworkflow from,
    so the pre-Release-A baseline can be measured by the same harness binary on the
    same host with the same payloads. The harness itself always runs from the
    working tree; only the subprocess is redirected.
    """

    def __init__(self, paths: Paths, *, startup_timeout: float, graceful: bool,
                 grace_seconds: float, baseline: bool = False,
                 server_tree: Optional[str] = None,
                 workflow_path: Optional[str] = None,
                 log_level: Optional[str] = None):
        self.paths = paths
        self.startup_timeout = startup_timeout
        self.graceful = graceful
        self.grace_seconds = grace_seconds
        self.baseline = baseline
        self.server_tree = server_tree or paths.repo_root
        # Arm B drives a different workflow than the rest, so the fixture — not the
        # checkout — decides what the server serves.
        self.workflow_path = workflow_path or paths.workflow_path
        self.log_level = log_level
        # Release C's own record of the resolved cap, read back from the server's
        # startup line rather than assumed: resolve_max_live_sessions() takes the OS
        # environment first, so what the harness set and what the server used can
        # differ, and only the server can say which won.
        self.max_live_sessions: Optional[int] = None
        self.max_live_sessions_source: Optional[str] = None
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
            "--workflow_path", self.workflow_path,
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
        if self.log_level:
            # fastworkflow.utils.logging reads LOG_LEVEL from the OS environment at
            # import, not from the env file.
            env["LOG_LEVEL"] = self.log_level

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
        self._read_live_session_cap()

    def _read_live_session_cap(self) -> None:
        """The live-session cap the server actually resolved, from its own record.

        Release C's gate is ``live_sessions <= MAX_LIVE_SESSIONS``, and the resolver
        takes the OS environment ahead of the env file, so the only authority on the
        effective value is the server. A tree that does not log it is a
        pre-Release-C tree, which is recorded rather than guessed at.
        """
        match = re.search(
            r"max_live_sessions=(\d+)\s*\(source=([^)]*)\)", self.policy_line or ""
        )
        if not match:
            return
        self.max_live_sessions = int(match.group(1))
        self.max_live_sessions_source = match.group(2)

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

    def eviction_diagnostics(self) -> dict[str, Any]:
        """Eviction and loss facts that only the server's own log carries.

        §16.5 gates arm C on "zero context-loss sentinels" and arm B on a pinned
        count, but the readiness probe exposes neither: it reports
        ``live_sessions`` and nothing about why a session is still there. The
        eviction path does state both, in the log — the over-target warning carries
        the server's own pinned count, and each pin reason is logged once per
        (workflow, reason).

        The design names a "context-loss sentinel" without giving a string, so
        SENTINEL_PATTERNS below enumerates the loss-adjacent lines the Release C
        code actually emits, and each is reported with its count rather than
        collapsed into one boolean.
        """
        log = self._read_log()
        pinned = [int(n) for n in re.findall(
            r"no candidate could be retired \((\d+) pinned\)", log)]
        return {
            "retired_channels": len(re.findall(r"Retired channel_id \S+ at generation", log)),
            "over_target_warnings": len(pinned),
            "pinned_reported_max": max(pinned) if pinned else 0,
            "pinned_reported_last": pinned[-1] if pinned else 0,
            # Non-greedy and anchored on the log formatter's " - " separator, which
            # otherwise lands the timestamp and module inside the reason.
            "pinned_reported_series": pinned,
            "pin_reasons": sorted(set(re.findall(
                r"cannot be checkpointed and will not be evicted: (.+?)(?: - |$)",
                log, re.MULTILINE))),
            "reap_passes_reclaiming": [
                int(n) for n in re.findall(r"Reclaimed (\d+) abandoned checkpoint", log)
            ],
            "reap_pass_failures": len(re.findall(r"Checkpoint reap pass failed", log)),
            "sentinels": {
                name: len(re.findall(pattern, log))
                for name, pattern in SENTINEL_PATTERNS.items()
            },
        }

    def retired_channel_ids(self) -> set[str]:
        """Channels the server says it evicted. Requires --server-log-level DEBUG.

        This is the only direct evidence for arm C's "no streaming turn's context
        closed": the pop-and-close is logged at DEBUG with the channel id, so at
        INFO the assertion would silently have nothing to look at.
        """
        return set(re.findall(
            r"Retired channel_id (\S+) at generation", self._read_log()))

    def retired_channel_events(self) -> list[dict[str, Any]]:
        """Every retirement as (channel, when), so arm C can order it against a turn.

        Arm C's assertion is temporal — "not evicted *while* its turn is
        registered" — and a set of channel ids cannot answer it. Retiring a
        streaming channel a moment after its turn ends is not a violation but the
        designed behaviour (``trim_live_sessions`` exists to do exactly that), so
        without the timestamp the two cases are indistinguishable and the arm
        reports a violation for correct behaviour.
        """
        return [
            {"channel_id": channel_id, "at": _parse_log_timestamp(stamp)}
            for channel_id, stamp in RETIRED_LINE.findall(self._read_log())
        ]

    def checkpoint_metrics(self) -> dict[str, Any]:
        """The checkpoint namespace, in the units §16.5's plateau gate reads.

        Measured by walking the store's documented layout rather than by importing
        ``ChannelCheckpointStore`` into the harness. Three reasons: importing
        fastworkflow here costs ~7 s and would configure DSPy in the *measuring*
        process; the harness must keep working against trees that predate the
        module (``--baseline``); and a walk cannot be fooled by a drifting
        counter, which is the failure mode the store's own docstring warns about.
        ``verify_checkpoint_accounting()`` cross-checks this against the store's
        own ``stats()`` once per run, so the walk is corroborated rather than
        trusted.

        physical uses st_blocks (the gated number, since it is what the filesystem
        actually occupies); apparent uses st_size (exactly reproducible, hence
        assertable).
        """
        base = os.path.join(self.speeddict_dir, "channel_checkpoints")
        files = apparent = physical = channels = generations = 0

        # The unit of measurement is one directory per record, matching
        # ChannelCheckpointStore._scan_* : a channel directory, a quarantine entry,
        # or a reclaim tree. The namespace scaffolding above those (channels/,
        # <deployment>/, <fingerprint>/) is deliberately not counted, because the
        # store does not count it either and the cross-check has to be exact.
        # Quarantine and reclaim debris ARE inside the totals: excluding them would
        # let an interrupted reap look like space that came back.
        for record_dir in self._checkpoint_record_dirs(base):
            record_files, record_apparent, record_physical = _dir_usage(record_dir)
            files += record_files
            apparent += record_apparent
            physical += record_physical
        for channel_dir in _subdirectories(
            os.path.join(base, "channels"), depth=3
        ):
            channels += 1
            generations += len(_subdirectories(os.path.join(channel_dir, "gen"),
                                               depth=1))
        return {
            "total_files": files,
            "total_bytes_apparent": apparent,
            "total_bytes_physical": physical,
            "channels": channels,
            "generations": generations,
        }

    @staticmethod
    def _checkpoint_record_dirs(base: str) -> list[str]:
        """Every directory the store measures as one record.

        Layout, from ChannelCheckpointStore's own docstring::

            <base>/channels/<dep>/<fp>/<channel_key>/
            <base>/__quarantine__/<dep>/<fp>/<channel_key>/<entry>/
            <base>/__reclaim__/<dep>/<fp>/<name>
        """
        return (
            _subdirectories(os.path.join(base, "channels"), depth=3)
            + _subdirectories(os.path.join(base, "__quarantine__"), depth=4)
            # Reclaim debris can be a file as well as a tree, so entries are taken
            # whole rather than filtered to directories.
            + _entries(os.path.join(base, "__reclaim__"), depth=3)
        )

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


def make_action(fixture: Fixture, payload: str) -> dict:
    """A direct action on the arm's fixture, carrying the payload.

    InitializationRequest has no free-form ``context`` field, so the payload
    travels as an extra action parameter. That is the field that actually reaches
    the retention path under test: WorkflowExecutionContext._process_action puts
    ``action.parameters`` verbatim into the record it appends to conversation
    history, which is also what the ConversationStore persists. The command
    itself ignores the extra key — hello_world's pydantic Input model drops it,
    and startup has no Input class at all — so the command still executes.
    """
    return {
        "command_name": fixture.command_name,
        "command": "",
        "parameters": fixture.base_parameters | {"soak_payload": payload},
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
    # §16.5's durable-storage plateau gate reads these. Physical is the gated
    # number; apparent is carried alongside because it is exactly reproducible.
    checkpoint_bytes_physical: int
    checkpoint_bytes_apparent: int
    checkpoint_channels: int
    checkpoint_generations: int
    checkpoint_files: int
    elapsed_s: float
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
    fixture: str = ""
    fixture_note: str = ""
    max_live_sessions: Optional[int] = None
    max_live_sessions_source: Optional[str] = None
    wall_clock_s: float = 0.0
    samples: list[Sample] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    cap_violations: list[str] = field(default_factory=list)
    forced_gc: Optional[dict[str, Any]] = None
    eviction: Optional[dict[str, Any]] = None
    streaming: Optional[dict[str, Any]] = None
    checkpoint_accounting: Optional[dict[str, Any]] = None


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

def read_probe(client: Client, server: ServerProcess, index: int,
               started_at: float) -> Probe:
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
    checkpoints = server.checkpoint_metrics()
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
        checkpoint_bytes_physical=checkpoints["total_bytes_physical"],
        checkpoint_bytes_apparent=checkpoints["total_bytes_apparent"],
        checkpoint_channels=checkpoints["channels"],
        checkpoint_generations=checkpoints["generations"],
        checkpoint_files=checkpoints["total_files"],
        elapsed_s=time.time() - started_at,
        metrics_available=memory is not None,
    )


def retained_turn_allowance(args) -> int:
    """The ceiling ``retained_turns`` may legitimately reach for this arm.

    ``retained_turns`` is ``len(turn_registry._by_key)``, which holds retained
    *terminal* executions AND every currently active one. MAX_RETAINED_STARTUP_TURNS
    bounds only the first group (``_evict_overflow`` sorts terminal executions and
    drops the excess), and the probe reports no breakdown. Every arm but C has
    exactly one turn in flight at a time, so for them the total and the cap coincide.
    Arm C deliberately holds ``--streams`` more, so its ceiling is that much higher —
    calling those a violation would fail the gate for doing what the arm asks.
    """
    return CAP_RETAINED_TURNS + (args.streams if args.arm == "c" else 0)


def check_caps(probe: Probe, arm: str, max_live_sessions: Optional[int] = None,
               retained_cap: int = CAP_RETAINED_TURNS) -> list[str]:
    if not probe.metrics_available:
        # Nothing to check: the baseline exposes no retention metrics, and it has
        # no caps to hold in the first place.
        return []
    violations = []
    if arm in ("a", "c") and max_live_sessions is not None:
        # Release C's own cap, and the one it gates on. Not checked for arm A0
        # (whose pre-registered expectation is that the cache retains everything),
        # for arm A1 (one channel), or for arm B (pinned by construction — its
        # unbounded growth is the recorded result, not a violation).
        if probe.live_sessions > max_live_sessions:
            violations.append(
                f"request {probe.index}: live_sessions={probe.live_sessions} > "
                f"max_live_sessions={max_live_sessions}"
            )
    if probe.retained_turns > retained_cap:
        violations.append(
            f"request {probe.index}: retained_turns={probe.retained_turns} > {retained_cap}"
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


def verify_checkpoint_accounting(paths: Paths, server_tree: str,
                                 checkpoint_dir: str) -> dict[str, Any]:
    """Cross-check the harness's directory walk against the store's own ``stats()``.

    The plateau gate rests entirely on one number, so that number should not rest
    entirely on the harness's reading of an undocumented layout. Run in a
    subprocess against the SERVER's tree, so the accounting being compared is the
    accounting the server itself would report — and so a tree without the module
    degrades to "unavailable" instead of breaking the run.
    """
    script = (
        "import json, sys\n"
        "from fastworkflow.checkpoint_store import ChannelCheckpointStore\n"
        "s = ChannelCheckpointStore(sys.argv[1]).stats()\n"
        "print(json.dumps({'total_bytes_physical': s.total_bytes_physical,\n"
        "                  'total_bytes_apparent': s.total_bytes_apparent,\n"
        "                  'total_files': s.total_files,\n"
        "                  'channels': s.channels,\n"
        "                  'generations': s.generations,\n"
        "                  'describe': s.describe()}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [server_tree] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    completed = subprocess.run(
        [paths.python, "-c", script, checkpoint_dir],
        cwd=server_tree, env=env, capture_output=True, text=True, timeout=300,
    )
    if completed.returncode != 0:
        return {"available": False,
                "reason": f"ChannelCheckpointStore.stats() could not be read from "
                          f"{server_tree}: {completed.stderr.strip()[-300:]}"}
    try:
        return {"available": True, "store": json.loads(completed.stdout.splitlines()[-1])}
    except (ValueError, IndexError):
        return {"available": False,
                "reason": f"unparsable stats() output: {completed.stdout[-300:]}"}


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

#: Arms whose workload is "a unique channel per request", i.e. the motivating
#: production shape. A0 is Release A's recorded arm and A/C are Release C's; they
#: differ in the code under test and in what is asserted, not in the requests sent.
UNIQUE_CHANNEL_ARMS = ("a0", "a", "b", "c")

#: Concurrency for arm C's pressure burst. Enough to finish inside one agent turn,
#: small enough that the server is not the thing being stress-tested.
PRESSURE_WORKERS = 8


class StreamingCohort:
    """Arm C: a few long-lived ``/invoke_agent_stream`` turns held across eviction.

    §16.5 arm C is "arm A plus concurrent ``/invoke_agent_stream`` turns spanning
    more than ``MAX_LIVE_SESSIONS`` subsequent creations". Two consequences shape
    this class:

    * It is the only arm that is not strictly sequential, because "concurrent" is
      the hypothesis. The sequential creation loop stays on the main thread and
      keeps producing the RSS series; each stream gets its own thread and its own
      socket, so the measured series is still one request at a time on the main
      connection.
    * ``/invoke_agent_stream`` strips the leading ``/`` from ``user_query`` before
      dispatch, so there is no deterministic route through it: the turn runs the
      agent and therefore makes real LLM calls. That is what makes it slow enough
      to still be registered 50+ creations later, and it is also why the cohort is
      small by default.

    The "more than MAX_LIVE_SESSIONS subsequent creations" span cannot be left to
    the measured loop. A 450 KB measured request costs ~200 ms, so covering 51 of
    them takes ~10s — about how long one agent turn lasts, which makes the arm's
    own premise a coin flip that was observed landing at 40 and 49 creations
    against a cap of 50. So the cohort issues its own burst of cheap unique
    channel creations immediately after opening the streams. They are real
    creations that force real retirements, they are excluded from the measured
    series, and the burst stops as soon as the span clears the cap, so the
    perturbation is bounded and lands between two measured requests.

    The assertion is ordering, not just outcome: a streaming channel must not be
    retired *before its own turn finished*. It is compared per stream against that
    stream's own completion time, because a retirement after the turn ends is the
    designed behaviour rather than a violation — ``trim_live_sessions`` exists to
    retire exactly those channels — and with 60 unique channels against a cap of
    50 it is the expected outcome for every stream.
    """

    def __init__(self, client: Client, args, fixture: Fixture):
        self.args = args
        self.fixture = fixture
        self.port = client.port
        self.streams: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []
        self.open_at = max(1, args.stream_after)
        self.opened = False
        # Plain ints, written by the main thread and read by the stream threads. The
        # only thing they must be is monotone, so an int assignment under CPython is
        # the whole synchronisation story; a lock here would serialise the
        # measurement loop against the streams for no benefit.
        self.creations = 0
        self.pressure_creations = 0
        self.creations_at_open = 0

    def note_creation(self) -> None:
        """One measured request created one channel. Called by the measured loop."""
        self.creations += 1

    def maybe_open(self, index: int, server: ServerProcess) -> None:
        """Open the cohort once, early enough that the rest of the run outlasts it."""
        if self.opened or index != self.open_at:
            return
        self.opened = True
        self.creations_at_open = self.creations
        cap = server.max_live_sessions
        remaining = self.args.requests - index
        if cap is not None and remaining <= cap and self.args.stream_pressure_kb <= 0:
            raise SoakError(
                f"Arm C needs more than MAX_LIVE_SESSIONS ({cap}) channel creations "
                f"after the streams open, but only {remaining} requests remain after "
                f"request {index} and --stream-pressure-kb 0 disabled the burst that "
                f"would otherwise supply them. Use --requests {index + cap + 1} or "
                f"more, or lower --stream-after."
            )
        client = Client(self.port)
        try:
            for slot in range(self.args.streams):
                channel = f"soak-c-stream-{slot}-{uuid.uuid4().hex}"
                status, response, _ = client.post("/initialize", {
                    "channel_id": channel,
                    "user_id": "soak",
                    "startup_action": make_action(
                        self.fixture, make_payload(self.args.payload_kb,
                                                   self.args.payload_fill)),
                    "timeout_seconds": self.args.request_timeout,
                })
                _require_200(status, response, f"/initialize (stream {slot})")
                record: dict[str, Any] = {
                    "slot": slot,
                    "channel_id": channel,
                    "opened_at_request": index,
                    "events": {},
                    "error": None,
                    "http_status": None,
                    "completed": False,
                    "duration_s": None,
                    "completed_at": None,
                    "requests_at_completion": None,
                }
                self.streams.append(record)
                thread = threading.Thread(
                    target=self._run_stream,
                    args=(record, response["access_token"]),
                    daemon=True,
                    name=f"soak-stream-{slot}",
                )
                self.threads.append(thread)
                thread.start()
        finally:
            client.close()
        self._apply_pressure(cap)

    def _apply_pressure(self, cap: Optional[int]) -> None:
        """Create cheap unique channels until the span clears the cap.

        These are the "subsequent channel creations" the arm is about, and they are
        issued concurrently rather than one at a time. Sequentially they are not
        fast enough to be useful: a channel creation costs ~180 ms almost
        independently of payload size, so 51 of them take ~9s — the same order as
        the agent turn they are supposed to outlast, which is what left the observed
        span one creation short of the cap at 49/50. A small pool collapses the
        burst to a second or two, comfortably inside one turn.

        Kept out of ``samples``, so the RSS and latency series stay a function of
        the measured requests alone.
        """
        if cap is None or self.args.stream_pressure_kb <= 0:
            return
        payload = make_payload(self.args.stream_pressure_kb, self.args.payload_fill)
        # One more than the cap: the premise is "more than MAX_LIVE_SESSIONS", and
        # every one of these evicts something, so the streaming channels are offered
        # for retirement cap+1 times over.
        wanted = cap + 1 - (self.creations - self.creations_at_open)
        if wanted <= 0:
            return
        with ThreadPoolExecutor(max_workers=PRESSURE_WORKERS) as pool:
            for created in pool.map(lambda _: self._create_pressure_channel(payload),
                                    range(wanted)):
                self.creations += created
                self.pressure_creations += created

    def _create_pressure_channel(self, payload: str) -> int:
        """One throwaway channel on its own connection. Returns 1 so callers can sum."""
        client = Client(self.port)
        try:
            status, response, _ = client.post("/initialize", {
                "channel_id": f"soak-c-pressure-{uuid.uuid4().hex}",
                "user_id": "soak",
                "startup_action": make_action(self.fixture, payload),
                "timeout_seconds": self.args.request_timeout,
            })
            _require_200(status, response, "/initialize (arm C pressure channel)")
            return 1
        finally:
            client.close()

    def _run_stream(self, record: dict[str, Any], token: str) -> None:
        started = time.time()
        try:
            record["http_status"], record["events"] = stream_ndjson(
                self.port, token, self.args.stream_query, self.args.stream_timeout
            )
            record["completed"] = True
        except Exception as exc:  # recorded, never raised into the main thread
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["completed_at"] = time.time()
            record["duration_s"] = record["completed_at"] - started
            # How many channels were created while this turn was still registered.
            # This — not the total request count — is what "spanning more than
            # MAX_LIVE_SESSIONS subsequent creations" means.
            record["creations_at_completion"] = self.creations

    def finish(self, server: ServerProcess) -> dict[str, Any]:
        """Join the cohort, then assert nothing evicted a stream while it ran."""
        deadline = time.time() + self.args.stream_timeout + 60
        for thread in self.threads:
            thread.join(timeout=max(1.0, deadline - time.time()))
        still_running = [t.name for t in self.threads if t.is_alive()]

        events = server.retired_channel_events()
        retired = {event["channel_id"] for event in events}
        evicted_while_streaming, evicted_after_completion, unordered = (
            self._classify_retirements(events))

        spans = [
            (record["creations_at_completion"] or 0) - self.creations_at_open
            for record in self.streams
        ]
        creations_during = min(spans) if spans else 0
        cap = server.max_live_sessions
        violations: list[str] = []
        if cap is not None and creations_during <= cap:
            # Not a defect in the server — a defect in the run. Without the overlap
            # the arm never put a streaming channel at risk, so "nothing was evicted"
            # would be evidence of nothing.
            shortest = min((r["duration_s"] or 0.0) for r in self.streams)
            rate = creations_during / shortest if shortest else 0.0
            violations.append(
                f"the shortest-lived stream saw only {creations_during} channel "
                f"creations while it was registered, which does not exceed "
                f"MAX_LIVE_SESSIONS ({cap}); the arm's premise was not met, so "
                f"'nothing was evicted' is evidence of nothing. The span is bounded "
                f"by throughput, not by --requests: this server created "
                f"{rate:.1f} channels/s and the shortest agent turn lasted "
                f"{shortest:.1f}s, so ~{creations_during} is the most this workflow "
                f"can span. Rerun with --max-live-sessions well below that (e.g. "
                f"--max-live-sessions 10), which both satisfies the premise and "
                f"raises the eviction pressure on the streaming channels."
            )
        if evicted_while_streaming:
            violations.append(
                "streaming channels were retired BEFORE their own turn completed: "
                + ", ".join(evicted_while_streaming)
            )
        if unordered:
            violations.append(
                f"{len(unordered)} retirement(s) of a streaming channel could not be "
                f"ordered against its turn, so the assertion is indeterminate rather "
                f"than met: {', '.join(unordered)}"
            )
        for record in self.streams:
            if record["error"]:
                violations.append(
                    f"stream {record['slot']} failed: {record['error']}")
            elif record["http_status"] != 200:
                violations.append(
                    f"stream {record['slot']} returned HTTP {record['http_status']}")
            elif not record["events"].get("output"):
                violations.append(
                    f"stream {record['slot']} produced no output event, so its turn "
                    f"did not complete; events seen: {record['events']}")
            if record["events"].get("error"):
                violations.append(
                    f"stream {record['slot']} emitted {record['events']['error']} "
                    f"error event(s)")
        if still_running:
            violations.append(
                f"streams still running after the join deadline: {still_running}")
        if not server.log_level or server.log_level.upper() != "DEBUG":
            violations.append(
                "server log level is not DEBUG, so 'Retired channel_id' is not "
                "logged and the no-eviction assertion had nothing to read; rerun "
                "with --server-log-level DEBUG"
            )

        return {
            "streams_requested": self.args.streams,
            "streams_opened": len(self.streams),
            "opened_at_request": self.open_at,
            "creations_after_open": self.creations - self.creations_at_open,
            "pressure_creations": self.pressure_creations,
            "pressure_payload_kb": self.args.stream_pressure_kb,
            "creations_while_registered_min": creations_during,
            "creations_while_registered_per_stream": spans,
            "max_live_sessions": cap,
            "spans_more_than_cap": (
                None if cap is None else creations_during > cap
            ),
            "eviction_assertion_readable": (
                bool(server.log_level) and server.log_level.upper() == "DEBUG"),
            "channels_retired_total": len(retired),
            "streaming_channels_retired_during_turn": evicted_while_streaming,
            "streaming_channels_retired_after_turn": evicted_after_completion,
            "streaming_retirements_unordered": unordered,
            "streams": self.streams,
            "violations": violations,
        }

    def _classify_retirements(
        self, events: list[dict[str, Any]]
    ) -> tuple[list[str], list[str], list[str]]:
        """Split streaming-channel retirements by when they happened.

        Three buckets: mid-turn (the violation), at-or-after completion (expected —
        ``trim_live_sessions`` retires a channel the moment its turn frees it, so
        this is the designed outcome and with 60 unique channels against a cap of 50
        it is the one to expect), and unorderable because no timestamp parsed
        (indeterminate, and never silently treated as a pass).

        The mid-turn threshold is one second rather than zero because the log
        carries whole seconds only. The cost is stated rather than hidden: an
        eviction in the final second of a turn is not distinguished from one just
        after it. It does not weaken the arm, because the pressure burst spans the
        whole ~10s turn, so a guard that failed to skip a busy channel would evict
        it seconds before completion and land in the mid-turn bucket — and would
        additionally show up as a missing output event or a context-loss sentinel.
        """
        during: list[str] = []
        after: list[str] = []
        unordered: list[str] = []
        for record in self.streams:
            completed_at = record["completed_at"]
            for event in events:
                if event["channel_id"] != record["channel_id"]:
                    continue
                if event["at"] is None or completed_at is None:
                    unordered.append(f"{event['channel_id']} (no usable timestamp)")
                    continue
                delta = event["at"] - completed_at
                if delta < -LOG_TIMESTAMP_RESOLUTION_S:
                    during.append(
                        f"{event['channel_id']} ({-delta:.0f}s before it completed)")
                elif delta < LOG_TIMESTAMP_RESOLUTION_S:
                    after.append(
                        f"{event['channel_id']} (at completion, within the log's "
                        f"{LOG_TIMESTAMP_RESOLUTION_S:.0f}s resolution)")
                else:
                    after.append(
                        f"{event['channel_id']} (+{delta:.0f}s after it completed)")
        return sorted(during), sorted(after), sorted(unordered)

    def abandon(self) -> None:
        """Threads are daemons and hold only their own sockets, so nothing to undo.

        Present so the caller's ``finally`` has something honest to call on the
        error path, where ``finish`` never ran.
        """
        return None


def stream_ndjson(port: int, token: str, query: str,
                  timeout: float) -> tuple[int, dict[str, int]]:
    """Drive one ``/invoke_agent_stream`` turn and tally its NDJSON event types.

    Read incrementally rather than with ``Client``: the whole point is to hold the
    turn open, and a response fully buffered before the first assertion would
    collapse the window the arm exists to create.
    """
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST", "/invoke_agent_stream",
            body=json.dumps({"user_query": query,
                             "timeout_seconds": int(timeout)}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"},
        )
        response = conn.getresponse()
        counts: dict[str, int] = {}
        if response.status != 200:
            response.read()
            return response.status, counts
        buffer = b""
        while chunk := response.read(4096):
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                _tally_ndjson_line(line, counts)
        _tally_ndjson_line(buffer, counts)
        return response.status, counts
    finally:
        conn.close()


def _tally_ndjson_line(line: bytes, counts: dict[str, int]) -> None:
    if not line.strip():
        return
    try:
        kind = json.loads(line).get("type", "unparsed")
    except ValueError:
        kind = "unparsed"
    counts[kind] = counts.get(kind, 0) + 1


def run_replicate(paths: Paths, fixture: Fixture, args, replicate_index: int) -> Replicate:
    with ServerProcess(paths, startup_timeout=args.startup_timeout,
                       graceful=args.graceful_shutdown,
                       grace_seconds=args.grace_seconds,
                       baseline=args.baseline,
                       server_tree=args.server_tree,
                       workflow_path=fixture.workflow_path,
                       log_level=args.server_log_level) as server:
        client = Client(server.port)
        streams: Optional[StreamingCohort] = None
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
                fixture=fixture.name,
                fixture_note=fixture.note,
                max_live_sessions=server.max_live_sessions,
                max_live_sessions_source=server.max_live_sessions_source,
            )
            if replicate_index == 0:
                result.forced_gc = forced_gc_availability(client)

            streams = StreamingCohort(client, args, fixture) if args.arm == "c" else None
            token = _warmup(client, fixture, args)
            started_at = time.time()
            first_probe = read_probe(client, server, 0, started_at)
            result.probes.append(first_probe)
            retained_cap = retained_turn_allowance(args)
            result.cap_violations += check_caps(first_probe, args.arm,
                                                server.max_live_sessions, retained_cap)

            for index in range(1, args.requests + 1):
                if streams is not None:
                    streams.maybe_open(index, server)
                sample = _measured_request(client, server, fixture, args, index, token)
                result.samples.append(sample)
                if streams is not None:
                    streams.note_creation()
                if index % args.probe_every == 0 or index == args.requests:
                    probe = read_probe(client, server, index, started_at)
                    result.probes.append(probe)
                    result.cap_violations += check_caps(
                        probe, args.arm, server.max_live_sessions, retained_cap)
            if streams is not None:
                result.streaming = streams.finish(server)
            result.wall_clock_s = time.time() - started_at
            result.eviction = server.eviction_diagnostics()
            if replicate_index == 0:
                # While the server is still up, so the directory being compared is
                # the one the samples above were read from.
                result.checkpoint_accounting = verify_checkpoint_accounting(
                    paths, server.server_tree,
                    os.path.join(server.speeddict_dir, "channel_checkpoints"),
                )
                result.checkpoint_accounting["walk"] = server.checkpoint_metrics()
            return result
        finally:
            if streams is not None:
                streams.abandon()
            client.close()


def _warmup(client: Client, fixture: Fixture, args) -> Optional[str]:
    """One excluded warm-up request; for arm A1 it also mints the channel's token."""
    channel = f"soak-{args.arm}-{uuid.uuid4().hex}"
    body = {
        "channel_id": channel,
        "user_id": "soak",
        "startup_action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
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


def _measured_request(client: Client, server: ServerProcess, fixture: Fixture,
                      args, index: int, token: Optional[str]) -> Sample:
    payload = make_payload(args.payload_kb, args.payload_fill)
    if args.arm in UNIQUE_CHANNEL_ARMS:
        body = {
            "channel_id": f"soak-{args.arm}-{index}-{uuid.uuid4().hex}",
            "user_id": "soak",
            "startup_action": make_action(fixture, payload),
            "timeout_seconds": args.request_timeout,
        }
        status, response, latency = client.post("/initialize", body)
        _require_200(status, response, f"/initialize (request {index})")
    else:
        status, response, latency = client.post(
            "/perform_action",
            {"action": make_action(fixture, payload), "timeout_seconds": args.request_timeout},
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

def run_latency(paths: Paths, fixture: Fixture, args) -> dict[str, Any]:
    with ServerProcess(paths, startup_timeout=args.startup_timeout,
                       graceful=args.graceful_shutdown,
                       grace_seconds=args.grace_seconds,
                       baseline=args.baseline,
                       server_tree=args.server_tree) as server:
        client = Client(server.port)
        try:
            latency_started_at = time.time()
            results: dict[str, Any] = {"server_tree": server.server_tree,
                                       "baseline": server.baseline,
                                       "fixture": fixture.name}

            # First, on a still-empty registry: below MAX_RETAINED_STARTUP_TURNS
            # no retention sweep has run, which is the "without overflow" case.
            turn_latencies = []
            for index in range(CAP_RETAINED_TURNS - 1):
                status, response, latency = client.post("/initialize", {
                    "channel_id": f"lat-turn-{index}-{uuid.uuid4().hex}",
                    "user_id": "soak",
                    "startup_action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
                    "timeout_seconds": args.request_timeout,
                })
                _require_200(status, response, f"/initialize (turn completion {index})")
                turn_latencies.append(latency)
            overflow_probe = read_probe(client, server, CAP_RETAINED_TURNS - 1,
                                        latency_started_at)
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
                "startup_action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
                "timeout_seconds": args.request_timeout,
            })
            _require_200(status, response, "/initialize (no-eviction warm-up)")
            headers = _auth(response)
            hit_latencies = []
            for index in range(args.latency_samples):
                status, response, latency = client.post(
                    "/perform_action",
                    {"action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
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
                "startup_action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
                "timeout_seconds": args.request_timeout,
            })
            _require_200(status, response, "/initialize (depth warm-up)")
            headers = _auth(response)
            depth_latencies = []
            for index in range(args.latency_depth):
                status, response, latency = client.post(
                    "/perform_action",
                    {"action": make_action(fixture, make_payload(args.payload_kb, args.payload_fill)),
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
            depth_probe = read_probe(client, server, args.latency_depth,
                                     latency_started_at)
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
    print(f"workflow          : {first.fixture} ({args.fixture_path})")
    print(f"fixture rationale : {first.fixture_note}")
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
              f"{'durRec':>7} {'durMB':>9} {'ckPhyMB':>9} {'ckAppMB':>9} {'ckCh':>6}")
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
                  f"{probe.durable_records:>7} {_mb(probe.durable_conversation_bytes):>9} "
                  f"{_mb(probe.checkpoint_bytes_physical):>9} "
                  f"{_mb(probe.checkpoint_bytes_apparent):>9} "
                  f"{probe.checkpoint_channels:>6}")
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
    retained_cap = retained_turn_allowance(args)
    print(f"  retained_turns      max {max_retained:>6}  cap {retained_cap:>6}  "
          f"{'OK' if max_retained <= retained_cap else 'VIOLATED'}")
    if retained_cap != CAP_RETAINED_TURNS:
        print(f"    cap is MAX_RETAINED_STARTUP_TURNS ({CAP_RETAINED_TURNS}) + "
              f"{args.streams} concurrent stream(s): retained_turns counts active "
              f"executions too, and only the terminal ones are what the cap bounds.")
    print(f"  dspy_cache.entries  max {max_dspy:>6}  cap {CAP_DSPY_CACHE_ENTRIES:>6}  "
          f"{'OK' if max_dspy <= CAP_DSPY_CACHE_ENTRIES else 'VIOLATED'}")
    if args.arm == "a1":
        print(f"  conversations.turns max {max_conv:>6}  cap {CAP_CONVERSATION_TURNS:>6}  "
              f"{'OK' if max_conv <= CAP_CONVERSATION_TURNS else 'VIOLATED'}   "
              f"(process-wide total; arm A1 keeps exactly {max_live} live channel)")
    else:
        print(f"  conversations.turns max {max_conv:>6}  (process-wide total: the "
              f"unique-channel arms hold one turn per live channel, so this tracks "
              f"live_sessions)")
    if args.arm in ("a", "c") and first.max_live_sessions is not None:
        print(f"  live_sessions       max {max_live:>6}  cap {first.max_live_sessions:>6}  "
              f"{'OK' if max_live <= first.max_live_sessions else 'VIOLATED'}   "
              f"(Release C, source={first.max_live_sessions_source})")
    elif args.arm == "b":
        print(f"  live_sessions       max {max_live:>6}  cap "
              f"{_num(first.max_live_sessions):>6}  RECORDED, NOT GATED   "
              f"(arm B pins by construction; unbounded growth IS the result)")
    elif first.max_live_sessions in (None, 2000):
        print(f"  live_sessions       max {max_live:>6}  cap   2000  "
              f"(unchanged in Release A)")
    elif args.arm not in UNIQUE_CHANNEL_ARMS:
        # A1 drives one channel, so it is never near any cap and a changed default
        # cannot have touched its numbers. Flagging a broken premise here would be
        # a false alarm on the one arm the cap is irrelevant to.
        print(f"  live_sessions       max {max_live:>6}  cap "
              f"{first.max_live_sessions:>6}  OK   (immaterial to arm "
              f"{args.arm.upper()}, which reuses a single channel)")
    else:
        # A0's pre-registered expectation is "the session cache still retains every
        # unique channel at the unchanged 2,000 default". On a Release C tree the
        # default is 50, so the arm no longer measures what §16.5.1 recorded, and
        # saying "cap 2000" here would misattribute the resulting number.
        print(f"  live_sessions       max {max_live:>6}  cap "
              f"{first.max_live_sessions:>6}  PREMISE BROKEN")
        print(f"    Arm {args.arm.upper()} assumes the Release A default of 2,000, but "
              f"this server resolved {first.max_live_sessions} "
              f"(source={first.max_live_sessions_source}). Eviction is therefore active "
              f"and this run is NOT comparable to §16.5.1. To reproduce that number, "
              f"point --server-tree at a Release A checkout, or pass "
              f"--max-live-sessions 2000.")
    violations = [v for r in replicates for v in r.cap_violations]
    for violation in violations:
        print(f"  VIOLATION: {violation}")
    print()
    return _finish_verdict(args, replicates, rss_summary, violations, first)


#: Which arms the design's shipping targets actually gate the slope on.
#: Release A gates arm A1; Release C gates arms A and C. A0 and B are recorded and
#: quoted verbatim (§1.2 and §1.4 respectively), so a failing slope there is the
#: pre-registered expectation rather than a build break.
SLOPE_GATED_ARMS = ("a1", "a", "c")


def _finish_verdict(args, replicates: list[Replicate], rss_summary: dict,
                    violations: list[str], first: Replicate) -> dict[str, Any]:
    plateau = analyze_durable_plateau(args, replicates)
    eviction = print_eviction_section(args, replicates)
    print_durable_plateau(plateau)
    print()
    streaming = _print_streaming_section(replicates) if args.arm == "c" else None

    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    upper = rss_summary.get("upper_bound_95", float("nan"))
    binding = args.requests >= MIN_REQUESTS_FOR_BINDING_GATE
    verdict: dict[str, Any] = {
        "arm": args.arm,
        "baseline": first.baseline,
        "server_tree": first.server_tree,
        "fixture": first.fixture,
        "fixture_note": first.fixture_note,
        "caps_held": None if first.baseline else not violations,
        "cap_violations": violations,
        "rss_slope_upper_bound_mb_per_request": upper,
        "slope_target_mb_per_request": SLOPE_TARGET_MB_PER_REQUEST,
        "slope_gate_applies": args.arm in SLOPE_GATED_ARMS and not first.baseline,
        "slope_gate_binding": binding,
        "durable_plateau": plateau,
        "eviction": eviction,
        "streaming": streaming,
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

    observed = "above" if upper > SLOPE_TARGET_MB_PER_REQUEST else "at or below"
    if args.arm == "a1":
        conv_note = _plateau_note(replicates)
        print(f"  in-memory conversation bytes: {conv_note}")
        verdict["conversation_bytes"] = conv_note

    if args.arm in SLOPE_GATED_ARMS:
        release = "A" if args.arm == "a1" else "C"
        meets = not math.isnan(upper) and upper <= SLOPE_TARGET_MB_PER_REQUEST
        label = "PASS" if meets else "FAIL"
        verdict["slope_gate_met"] = meets
        suffix = "" if binding else " (ADVISORY)"
        print(f"  GATE (Release {release}, arm {args.arm.upper()}): RSS slope upper "
              f"bound {upper:+.5f} <= {SLOPE_TARGET_MB_PER_REQUEST} MB/request "
              f"... {label}{suffix}")
        if not binding:
            print(f"    ADVISORY because {args.requests} measured requests is below the "
                  f"§16.5 minimum of {MIN_REQUESTS_FOR_BINDING_GATE}; at this sample "
                  f"count the slope is dominated by allocator warm-up.")
        # Unlike the slope, a cap breach is a single observed sample above a fixed
        # ceiling, so it is binding at any request count.
        print(f"  GATE (Release {release}, arm {args.arm.upper()}): structural caps held "
              + ("... PASS" if not violations else "... FAIL — see VIOLATION above"))
        if args.arm == "c":
            print("  GATE (Release C, arm C): streaming invariants ... "
                  + ("PASS" if not (streaming or {}).get("violations")
                     else "FAIL — see STREAMING above"))
    elif args.arm == "a0":
        print(f"  Arm A0 is RECORDED, NOT GATED (§16.5). RSS slope upper bound "
              f"{upper:+.5f} MB/request.")
        print(f"    Pre-registered expectation: materially below the unpatched baseline "
              f"but ABOVE {SLOPE_TARGET_MB_PER_REQUEST} MB/request, because the "
              f"live-session cache still retains every unique channel at the unchanged "
              f"2000 default.")
        print(f"    Observed: {observed} the {SLOPE_TARGET_MB_PER_REQUEST} MB/request "
              f"target. Quote this number verbatim in §1.2 and the release notes.")
        print("    Structural caps above ARE gated for this arm.")
        verdict["slope_vs_target"] = observed
    else:
        max_live = max((p.live_sessions or 0 for r in replicates for p in r.probes),
                       default=0)
        pinned = max((e.get("pinned_reported_max", 0)
                      for e in eviction["per_replicate"]), default=0)
        pinned_monotone = _monotone_non_decreasing(
            e.get("pinned_reported_series") or [] for e in eviction["per_replicate"])
        monotone = _live_sessions_monotone(replicates)
        print("  Arm B is RECORDED, NOT GATED (§16.5); quote it verbatim in §1.4.")
        print("    Pre-registered expectation: slope materially above target, live "
              "sessions grow without bound, pinned count rises monotonically.")
        print(f"    RSS slope upper bound {upper:+.5f} MB/request — {observed} the "
              f"{SLOPE_TARGET_MB_PER_REQUEST} target.")
        print(f"    live_sessions reached {max_live} over {args.requests} requests "
              f"(cap {_num(first.max_live_sessions)}), monotone non-decreasing: "
              f"{monotone}.")
        print(f"    Server-reported pinned count peaked at {pinned}, monotone "
              f"non-decreasing: {pinned_monotone}.")
        verdict |= {
            "slope_vs_target": observed,
            "max_live_sessions_observed": max_live,
            "live_sessions_monotone": monotone,
            "pinned_reported_max": pinned,
            "pinned_monotone": pinned_monotone,
        }

    # A cap breach always fails the run; the slope only does so once the sample count
    # makes the gate binding. Arm C's streaming invariants are not sample-size
    # dependent — a closed context is a defect at any run length — so they always bind.
    verdict["run_failed"] = bool(violations) or bool(
        (streaming or {}).get("violations")
    ) or (
        args.arm in SLOPE_GATED_ARMS and binding
        and not verdict.get("slope_gate_met", False)
    )

    print()
    print("  These are raw samples and slopes. Do not restate this run as "
          "'survived N requests',")
    print("  and do not sum ablation deltas as independent shares (§16.5).")
    print()
    return verdict


def _monotone_non_decreasing(serieses) -> bool:
    return all(
        all(after >= before for before, after in zip(series, series[1:]))
        for series in serieses
    )


def _live_sessions_monotone(replicates: list[Replicate]) -> bool:
    """Arm B's "live sessions grow without bound", checked rather than asserted."""
    return _monotone_non_decreasing(
        [p.live_sessions for p in rep.probes if p.live_sessions is not None]
        for rep in replicates
    )


def _print_streaming_section(replicates: list[Replicate]) -> dict[str, Any]:
    print("-" * 78)
    print("STREAMING (§16.5 arm C)")
    print("-" * 78)
    merged: dict[str, Any] = {"violations": [], "per_replicate": []}
    for rep in replicates:
        block = rep.streaming or {}
        merged["per_replicate"].append(block)
        merged["violations"] += block.get("violations", [])
        print(f"  replicate {rep.replicate + 1}: {block.get('streams_opened')} stream(s) "
              f"opened at request {block.get('opened_at_request')}; while registered "
              f"they spanned {block.get('creations_while_registered_min')} channel "
              f"creations (per stream "
              f"{block.get('creations_while_registered_per_stream')}), "
              f"cap {_num(block.get('max_live_sessions'))}, "
              f"spans more than cap: {block.get('spans_more_than_cap')}")
        print(f"    of {block.get('creations_after_open')} creations after the open, "
              f"{block.get('pressure_creations')} were the "
              f"{block.get('pressure_payload_kb')} KB pressure burst (excluded from "
              f"the measured series)")
        print(f"    channels retired during the run: "
              f"{block.get('channels_retired_total')}")
        print(f"    streaming channels retired MID-TURN (violation): "
              f"{block.get('streaming_channels_retired_during_turn') or 'none'}")
        print(f"    streaming channels retired at/after their turn (expected): "
              f"{block.get('streaming_channels_retired_after_turn') or 'none'}")
        for stream in block.get("streams", []):
            print(f"    stream {stream['slot']}: HTTP {stream['http_status']}, "
                  f"{stream['duration_s']:.1f}s, events {stream['events']}"
                  + (f", ERROR {stream['error']}" if stream["error"] else ""))
    print("  Assertion: a streaming turn's channel is never retired while its turn is "
          "registered.")
    print("  Evidence: each 'Retired channel_id' line's own log timestamp is compared "
          "against")
    print("  the completion time of that stream, so a retirement after the turn ended — "
          "which")
    print("  is what trim_live_sessions is for, and is expected here — is not counted "
          "against")
    print("  the assertion. The log carries whole seconds only, so an eviction inside "
          "the")
    print("  final second of a turn is not distinguished from one just after it; the "
          "burst")
    print("  spans the whole turn, so a broken busy-channel guard would evict seconds "
          "early.")
    print("  'No detached executor writes' is observed as the ABSENCE of the delivery-"
          "deadline")
    print("  warning plus a completed output event for every stream: the streaming path "
          "awaits")
    print("  its executor future to completion by construction, so a detached write has "
          "no")
    print("  code path to occur on — only the deadline warning would precede one.")
    for violation in merged["violations"]:
        print(f"  VIOLATION: {violation}")
    print()
    return merged


def analyze_durable_plateau(args, replicates: list[Replicate]) -> dict[str, Any]:
    """Did total physical checkpoint bytes stop growing, and if not, why not.

    §16.5 gates Release C on total physical bytes per namespace reaching a plateau
    and explicitly refuses a bytes-per-1,000-requests rate as a substitute. So this
    reports the plateau or reports that it was not observed — it never converts a
    growth rate into a pass.

    Reclamation needs BOTH of two things, and a short soak has neither:

    1. **A reap pass must fire.** The server's lifespan task sleeps
       ``CHECKPOINT_REAP_INTERVAL_SECONDS`` (300 s) *before* its first pass, so the
       first is at t=300 s and the second at t=600 s. A 300-request arm-A soak
       finishes in well under a minute.
    2. **Something must be reclaimable.** ``RetentionPolicy``'s defaults are
       ``max_age_seconds=86400`` and ``max_channels=1000``. Nothing in a soak is 24 h
       old, so the age window never fires; the count cap is the only live mechanism,
       and below 1,000 channels on disk it reclaims nothing. A reap pass that fires
       at 300 channels is a no-op.

    Neither constant has an environment override and the harness will not reach into
    the store to reap on the server's behalf — a second writer per channel is
    precisely what ``ChannelCheckpointStore`` is documented not to have. So the
    required run length is stated instead: get past the count cap, then span two
    passes.
    """
    per_request_s = statistics.fmean(
        [statistics.fmean([s.latency_s for s in r.samples])
         for r in replicates if r.samples]
    ) if any(r.samples for r in replicates) else float("nan")

    series = []
    for rep in replicates:
        points = [(p.elapsed_s, p.checkpoint_bytes_physical, p.checkpoint_bytes_apparent,
                   p.checkpoint_channels, p.index) for p in rep.probes]
        decreases = sum(
            1 for before, after in zip(points, points[1:]) if after[1] < before[1]
        )
        series.append({
            "replicate": rep.replicate,
            "wall_clock_s": rep.wall_clock_s,
            "reap_passes_elapsed": int(rep.wall_clock_s // REAP_INTERVAL_SECONDS),
            "reap_passes_reclaiming": (rep.eviction or {}).get(
                "reap_passes_reclaiming", []),
            "final_bytes_physical": points[-1][1] if points else 0,
            "final_bytes_apparent": points[-1][2] if points else 0,
            "final_channels_on_disk": points[-1][3] if points else 0,
            "bytes_physical_decreases": decreases,
            "second_half_slope_bytes_per_request": (
                least_squares_slope(
                    [float(p[4]) for p in points[len(points) // 2:]],
                    [float(p[1]) for p in points[len(points) // 2:]],
                ) if len(points) >= 4 else float("nan")
            ),
        })

    passes = min((s["reap_passes_elapsed"] for s in series), default=0)
    max_channels_seen = max((s["final_channels_on_disk"] for s in series), default=0)
    reclaimed = any(s["bytes_physical_decreases"] for s in series) or any(
        s["reap_passes_reclaiming"] for s in series)

    blockers = []
    if args.arm not in UNIQUE_CHANNEL_ARMS:
        # Arm A1 drives one channel forever. Its namespace holds a single record, so
        # the count cap can never bind and no run length would make a plateau
        # observable. Saying so beats printing a required-request count that is a
        # projection from a workload the arm does not run.
        return {
            "gate": "total physical bytes per namespace reach a plateau (§16.5)",
            "accounting_cross_check": replicates[0].checkpoint_accounting,
            "observed": "not applicable",
            "plateau_observable_at_this_run_length": None,
            "blockers": [
                f"arm {args.arm.upper()} drives a single channel, so the checkpoint "
                f"namespace holds one record and cannot reach "
                f"RetentionPolicy.max_channels={RETENTION_MAX_CHANNELS} at any run "
                f"length. The plateau gate is a property of the unique-channel "
                f"workload; measure it on arm A or C."
            ],
            "reclamation_observed": reclaimed,
            "mean_request_seconds": per_request_s,
            "required_requests_for_plateau_measurement": None,
            "required_wall_clock_seconds": None,
            "projected_durable_gib_at_required_length": None,
            "per_replicate": series,
        }
    if max_channels_seen == 0 and not any(
            s["final_bytes_physical"] for s in series):
        # Arm B pins every session, so nothing is ever evicted and nothing is ever
        # checkpointed. Durable bytes are flat at zero, which satisfies the letter
        # of a plateau and tests none of its substance; the growth that arm records
        # is in memory. Reporting the required-run-length projection here would
        # invite reading a vacuous zero as a pass.
        return {
            "gate": "total physical bytes per namespace reach a plateau (§16.5)",
            "accounting_cross_check": replicates[0].checkpoint_accounting,
            "observed": "flat at zero — vacuous",
            "plateau_observable_at_this_run_length": None,
            "blockers": [
                f"arm {args.arm.upper()} wrote no checkpoints at all: every session "
                f"pinned, so nothing was evicted and the namespace stayed empty. "
                f"Bytes are constant at zero, which is not the bounded-after-growth "
                f"plateau the gate is about. Measure the plateau on arm A or C."
            ],
            "reclamation_observed": reclaimed,
            "mean_request_seconds": per_request_s,
            "required_requests_for_plateau_measurement": None,
            "required_wall_clock_seconds": None,
            "projected_durable_gib_at_required_length": None,
            "per_replicate": series,
        }
    if passes < REAP_PASSES_FOR_PLATEAU:
        blockers.append(
            f"the run spanned {passes} reap interval(s) of "
            f"{REAP_INTERVAL_SECONDS:.0f}s; a plateau claim needs at least "
            f"{REAP_PASSES_FOR_PLATEAU}"
        )
    if max_channels_seen <= RETENTION_MAX_CHANNELS:
        blockers.append(
            f"only {max_channels_seen} channels reached disk, at or below "
            f"RetentionPolicy.max_channels={RETENTION_MAX_CHANNELS}, so a reap pass "
            f"would have had nothing to reclaim even if one had fired "
            f"(max_age_seconds={RETENTION_MAX_AGE_SECONDS:.0f}s cannot fire in a soak)"
        )

    if math.isnan(per_request_s) or per_request_s <= 0:
        required_requests = None
        required_wall_s = None
    else:
        # Both conditions at once, not one after the other: the count cap is passed
        # at request 1,001 whatever the clock says, so the two requirements overlap
        # and the larger one governs. At any rate faster than
        # 2 x 300 s / 1,000 = 600 ms/request the clock is what binds.
        required_requests = max(
            RETENTION_MAX_CHANNELS + 1,
            math.ceil(REAP_PASSES_FOR_PLATEAU * REAP_INTERVAL_SECONDS / per_request_s),
        )
        required_wall_s = required_requests * per_request_s

    # Disk, not memory, is what actually stops a plateau run: the conversation store
    # is append-only and is not what retention reaps, so it keeps every payload.
    observed_durable = max(
        (p.durable_speeddict_bytes for r in replicates for p in r.probes), default=0)
    observed_requests = max((len(r.samples) for r in replicates), default=0)
    projected_gib = (
        observed_durable / observed_requests * required_requests / (1024 ** 3)
        if observed_requests and required_requests else None
    )

    return {
        "gate": "total physical bytes per namespace reach a plateau (§16.5)",
        "accounting_cross_check": replicates[0].checkpoint_accounting,
        "observed": "plateau" if (reclaimed and not blockers) else "no plateau",
        "plateau_observable_at_this_run_length": not blockers,
        "blockers": blockers,
        "reclamation_observed": reclaimed,
        "reap_interval_seconds": REAP_INTERVAL_SECONDS,
        "reap_passes_required_for_claim": REAP_PASSES_FOR_PLATEAU,
        "retention_max_channels": RETENTION_MAX_CHANNELS,
        "retention_max_age_seconds": RETENTION_MAX_AGE_SECONDS,
        "mean_request_seconds": per_request_s,
        "required_requests_for_plateau_measurement": required_requests,
        "required_wall_clock_seconds": required_wall_s,
        "projected_durable_gib_at_required_length": projected_gib,
        "per_replicate": series,
    }


def print_durable_plateau(analysis: dict[str, Any]) -> None:
    print("-" * 78)
    print("DURABLE STORAGE PLATEAU (§16.5: total physical bytes per namespace)")
    print("-" * 78)
    for rep in analysis["per_replicate"]:
        print(f"  replicate {rep['replicate'] + 1}: "
              f"final physical {rep['final_bytes_physical'] / (1024 * 1024):.1f} MB, "
              f"apparent {rep['final_bytes_apparent'] / (1024 * 1024):.1f} MB, "
              f"{rep['final_channels_on_disk']} channels, "
              f"{rep['wall_clock_s']:.1f}s wall, "
              f"{rep['reap_passes_elapsed']} reap interval(s) spanned, "
              f"{rep['bytes_physical_decreases']} byte decrease(s)")
    check = analysis.get("accounting_cross_check") or {}
    if check.get("available"):
        store, walk = check["store"], check["walk"]
        agree = all(store[key] == walk[key] for key in
                    ("total_bytes_physical", "total_bytes_apparent", "total_files",
                     "channels", "generations"))
        print(f"  accounting cross-check vs ChannelCheckpointStore.stats(): "
              f"{'AGREES' if agree else 'DISAGREES'}")
        print(f"    store: {store['describe']}")
        if not agree:
            print(f"    walk : {walk}")
            print("    The gated number is the store's, not the walk's; treat the walk's "
                  "per-sample series as indicative until they agree.")
    else:
        print(f"  accounting cross-check: UNAVAILABLE — {check.get('reason')}")
    if analysis["observed"] == "plateau":
        print("  PLATEAU OBSERVED — physical bytes stopped growing under the stated "
              "retention policy.")
        return
    if analysis["observed"] in ("not applicable", "flat at zero — vacuous"):
        print(f"  PLATEAU NOT MEASURABLE ON THIS ARM ({analysis['observed']}).")
        for blocker in analysis["blockers"]:
            print(f"    - {blocker}")
        return
    print("  PLATEAU NOT OBSERVED. This is a measurement-window result, not a "
          "finding about")
    print("  the retention policy: the run was too short for reclamation to be "
          "possible at all.")
    for blocker in analysis["blockers"]:
        print(f"    - {blocker}")
    required = analysis["required_requests_for_plateau_measurement"]
    if required is not None:
        print(f"  A real plateau measurement needs about {required} requests "
              f"(~{analysis['required_wall_clock_seconds'] / 60:.0f} min at the "
              f"observed {analysis['mean_request_seconds'] * 1000:.0f} ms/request):")
        print(f"    whichever is larger of {RETENTION_MAX_CHANNELS + 1} requests (to "
              f"pass RetentionPolicy.max_channels) and "
              f"{math.ceil(REAP_PASSES_FOR_PLATEAU * REAP_INTERVAL_SECONDS / analysis['mean_request_seconds'])} "
              f"requests (to span {REAP_PASSES_FOR_PLATEAU} reap intervals).")
        if projected := analysis.get("projected_durable_gib_at_required_length"):
            print(f"    Budget ~{projected:.1f} GiB of free space under "
                  f"SPEEDDICT_FOLDERNAME for that run: the conversation store keeps "
                  f"every payload and only the checkpoint namespace is reaped.")
    print("  The observed byte growth is deliberately NOT reported as a "
          "bytes-per-1,000-requests")
    print("  rate: §16.5 rejects any positive rate as a bound, because a rate grows "
          "forever.")


def print_eviction_section(args, replicates: list[Replicate]) -> dict[str, Any]:
    """Eviction, pinning and loss-sentinel counts from the servers' own logs."""
    print("-" * 78)
    print("EVICTION AND PINNING (from the server's own log)")
    print("-" * 78)
    merged: dict[str, Any] = {"per_replicate": []}
    sentinels: dict[str, int] = {}
    for rep in replicates:
        diag = rep.eviction or {}
        merged["per_replicate"].append(diag)
        for name, count in (diag.get("sentinels") or {}).items():
            sentinels[name] = sentinels.get(name, 0) + count
        print(f"  replicate {rep.replicate + 1}: "
              f"max_live_sessions={_num(rep.max_live_sessions)} "
              f"(source={rep.max_live_sessions_source or 'not reported by this tree'}), "
              f"retired={diag.get('retired_channels', 0)}, "
              f"over-target warnings={diag.get('over_target_warnings', 0)}, "
              f"pinned reported max={diag.get('pinned_reported_max', 0)}")
        for reason in diag.get("pin_reasons") or []:
            print(f"    pin reason: {reason}")
    merged["sentinels"] = sentinels
    merged["sentinels_total"] = sum(sentinels.values())
    print(f"  context-loss sentinels: {merged['sentinels_total']} total"
          + (f" -> {[k for k, v in sentinels.items() if v]}"
             if merged["sentinels_total"] else " (none)"))
    if args.arm in ("a", "c") and not any(
        (rep.eviction or {}).get("retired_channels") for rep in replicates
    ):
        print("  NOTE: zero retirements were logged. Retirement is logged at DEBUG, so "
              "either")
        print("  the cap was never exceeded or --server-log-level is not DEBUG.")
    print()
    return merged


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
    parser.add_argument("--arm", choices=["a0", "a1", "a", "b", "c"], default="a1",
                        help="Release A: a0 = unique channel per request (recorded), "
                             "a1 = one hot channel (gated). Release C: a = evictable "
                             "unique-channel (gated), b = pinned/undeclared workflow "
                             "(recorded, quoted in §1.4), c = arm A plus concurrent "
                             "/invoke_agent_stream turns (gated)")
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
    parser.add_argument("--max-live-sessions", type=int, default=None,
                        help="override MAX_LIVE_SESSIONS in the server's process "
                             "environment. resolve_max_live_sessions() reads the OS "
                             "environment first, so this wins over the env file. Left "
                             "unset the server's own default (50) applies; the effective "
                             "value is always read back from its startup record.")
    parser.add_argument("--server-log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="LOG_LEVEL for the server process. Arm C defaults this to "
                             "DEBUG because 'Retired channel_id' is logged at DEBUG and "
                             "is the only direct evidence for its no-eviction assertion")
    parser.add_argument("--streams", type=int, default=3,
                        help="arm C only: concurrent /invoke_agent_stream turns. Each "
                             "runs the agent, so each is a real LLM call")
    parser.add_argument("--stream-after", type=int, default=5,
                        help="arm C only: the request index at which the streams open. "
                             "Everything after it is the creation burst they must "
                             "survive, so --requests must exceed it by more than "
                             "MAX_LIVE_SESSIONS")
    parser.add_argument("--stream-pressure-kb", type=int, default=1,
                        help="arm C only: payload for the burst of cheap channel "
                             "creations issued right after the streams open, which is "
                             "what supplies the required span of more than "
                             "MAX_LIVE_SESSIONS creations. Excluded from the measured "
                             "series. 0 disables the burst and leaves the span to the "
                             "measured loop, where it is not reliably reached")
    parser.add_argument("--stream-timeout", type=float, default=180.0,
                        help="arm C only: per-stream deadline, seconds")
    parser.add_argument("--stream-query", default="add 2 and 3",
                        help="arm C only: the natural-language query each stream sends. "
                             "/invoke_agent_stream strips a leading '/', so there is no "
                             "deterministic route through it")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests < 2:
        raise SoakError("--requests must be at least 2 to fit a second-half slope")
    if args.replicates < 1:
        raise SoakError("--replicates must be at least 1")

    paths = resolve_paths()
    args.server_tree = os.path.realpath(args.server_tree or paths.repo_root)
    if args.arm == "c" and args.server_log_level is None:
        # Not silently: the alternative is an assertion with nothing to read, which
        # would look like a pass.
        args.server_log_level = "DEBUG"
        print("[soak] arm C: defaulting --server-log-level to DEBUG so the "
              "no-eviction assertion has evidence to read", flush=True)
    if args.max_live_sessions is not None:
        os.environ["MAX_LIVE_SESSIONS"] = str(args.max_live_sessions)
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

    fixture = resolve_fixture(paths, args.arm)
    args.fixture_path = fixture.workflow_path
    print(f"[soak] arm {args.arm.upper()} fixture: {fixture.name} at "
          f"{fixture.workflow_path}")
    print(f"[soak]   {fixture.note}", flush=True)

    try:
        replicates: list[Replicate] = []
        for index in range(args.replicates):
            print(f"[soak] replicate {index + 1}/{args.replicates}: starting a fresh "
                  f"server (cold import can take ~60s)...", flush=True)
            replicates.append(run_replicate(paths, fixture, args, index))
            print(f"[soak] replicate {index + 1}/{args.replicates}: done", flush=True)

        latency_results: Optional[dict[str, Any]] = None
        if args.latency:
            print("[soak] latency matrix: starting a fresh server...", flush=True)
            latency_results = run_latency(paths, fixture, args)
            print("[soak] latency matrix: done", flush=True)
    finally:
        fixture.cleanup()

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
            "workflow_path": fixture.workflow_path,
            "fixture": {"name": fixture.name, "note": fixture.note,
                        "expect_pinned": fixture.expect_pinned},
            "retention": {
                "reap_interval_seconds": REAP_INTERVAL_SECONDS,
                "max_channels": RETENTION_MAX_CHANNELS,
                "max_age_seconds": RETENTION_MAX_AGE_SECONDS,
            },
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
                    "fixture": rep.fixture,
                    "max_live_sessions": rep.max_live_sessions,
                    "max_live_sessions_source": rep.max_live_sessions_source,
                    "wall_clock_s": rep.wall_clock_s,
                    "eviction": rep.eviction,
                    "streaming": rep.streaming,
                    "checkpoint_accounting": rep.checkpoint_accounting,
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
