"""Structural tests for the server-only DSPy memory policy (design §16.4, steps 1-9).

``fastworkflow/run_fastapi_mcp/server_memory.py`` turns off DSPy's global
history, per-LM history and predictor trace, and bounds its response cache.
``Settings.configure`` is a dict assignment with no key validation, so
configuring proves nothing on its own: these tests establish the baseline
(defaults really do retain a copy of every call), then prove the policy removes
it, that a later async ``configure()`` cannot silently undo it, that drift makes
the pod unready, and that the response cache honours its cap without touching
disk.

Every test runs in an isolated subprocess launched as the repo's pinned
``.venv/bin/python``. Two reasons, both load-bearing:

* DSPy configuration ownership is process-global and irreversible — the first
  thread to call ``configure()`` owns it for the life of the process — so tests
  cannot share an interpreter with each other or with the pytest session.
* The ambient interpreter on the development host resolves to DSPy 2.6.27 while
  the pinned virtualenv carries 3.2.1, so ``sys.executable`` would silently test
  a different DSPy than the one the server runs.

No test makes a network LLM call. The stub is placed at ``dspy.LM.forward`` (via
``server_memory._ProbeLM``) or below it at ``litellm_completion``, so the real
``BaseLM`` history path and ``Predict._forward_postprocess`` trace path execute;
a stub any higher would make the whole exercise vacuous. Steps 1-5 additionally
run in the production call shape — ``Predict`` inside ``dspy.context(lm=...,
adapter=...)`` inside a ``ThreadPoolExecutor`` worker — because a main-thread
call would pass even if ``Settings.context`` did not inherit
``main_thread_config``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import tomllib

import pytest

# Import only: this module installs nothing at import time, so the parent test
# process stays free of DSPy global configuration.
from fastworkflow.run_fastapi_mcp import server_memory


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "bin", "python")
_SUBPROCESS_TIMEOUT_SECONDS = 300

if not os.path.exists(_VENV_PYTHON):
    pytest.skip(
        f"the DSPy memory policy can only be proven against the pinned interpreter, "
        f"which is missing at {_VENV_PYTHON}",
        allow_module_level=True,
    )


def _last_json_line(completed: subprocess.CompletedProcess) -> dict:
    # Interpreter teardown (speedict, torch, atexit hooks) can emit trailing
    # noise after the observation line, so scan back to the last JSON object.
    for line in reversed(completed.stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    raise AssertionError(
        "probe subprocess printed no JSON observation line\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


def _run_in_venv(script_source: str, python: str = _VENV_PYTHON) -> dict:
    """Run a probe script in a fresh interpreter and return its JSON observations.

    Assertions stay in the parent process: the script only reports what it saw,
    so a failure reads as a pytest diff rather than a subprocess traceback.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, "dspy_memory_probe.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(script_source)
        completed = subprocess.run(
            [python, script_path],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            # A private, empty DSPy cache per probe. The cache-bound probe
            # asserts that every prompt reached the provider, which is only true
            # against a COLD cache -- and once the server policy began enabling
            # the disk cache (v2.29.1) the shared ~/.dspy_cache served the fixed
            # probe strings instead, so the test passed exactly once per machine.
            # DSPY_CACHEDIR is read at import time, which is why it is set here
            # rather than in the script: install_policy calls configure_cache
            # without a directory, so anything the script set would be replaced.
            # It also stops these probes writing into the developer's real cache.
            env={
                **os.environ,
                "PYTHONPATH": _REPO_ROOT,
                "DSPY_CACHEDIR": os.path.join(tmp_dir, "dspy_cache"),
            },
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )

    assert completed.returncode == 0, (
        f"probe subprocess exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return _last_json_line(completed)


@pytest.fixture
def workflow_path():
    path = os.path.join(_REPO_ROOT, "tests", "hello_world_workflow")
    if not os.path.isdir(path):
        pytest.skip(f"hello_world_workflow not found at {path}")
    return path


@pytest.fixture
def env_files():
    env_file = os.path.join(_REPO_ROOT, "env", ".env")
    passwords_file = os.path.join(_REPO_ROOT, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI server tests")
    return env_file, passwords_file


@pytest.fixture
def server_argv_preamble(workflow_path, env_files):
    """Source that sets ``sys.argv`` before ``__main__`` parses it at import time."""
    env_file, passwords_file = env_files
    return (
        "import sys\n"
        "sys.argv = [\n"
        "    'fw-server-memory-test',\n"
        f"    '--workflow_path', {workflow_path!r},\n"
        f"    '--env_file_path', {env_file!r},\n"
        f"    '--passwords_file_path', {passwords_file!r},\n"
        "]\n"
    )


_DEFAULTS_BASELINE_SCRIPT = """
import json
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.clients import base_lm

from fastworkflow.run_fastapi_mcp.server_memory import _ProbeLM, _ProbeSignature


def call_in_worker():
    lm = _ProbeLM()
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        dspy.Predict(_ProbeSignature)(question="baseline default settings probe")
    return lm


with ThreadPoolExecutor(max_workers=1) as pool:
    probe_lm = pool.submit(call_in_worker).result()

print(json.dumps({
    "dspy_version": dspy.__version__,
    "disable_history": dspy.settings.disable_history,
    "max_trace_size": dspy.settings.max_trace_size,
    "global_history": len(base_lm.GLOBAL_HISTORY),
    "lm_history": len(probe_lm.history),
    "trace": len(dspy.settings.trace or []),
}))
"""


_POLICY_SCRIPT = """
import json
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.clients import base_lm

from fastworkflow.run_fastapi_mcp import server_memory

status = server_memory.install_policy()


def call_in_worker(index):
    lm = server_memory._ProbeLM()
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        dspy.Predict(server_memory._ProbeSignature)(
            question=f"unique server policy question {index}"
        )
    return len(lm.history)


with ThreadPoolExecutor(max_workers=4) as pool:
    lm_history_lengths = list(pool.map(call_in_worker, range(8)))

print(json.dumps({
    "dspy_version": status.dspy_version,
    "asserted": status.asserted,
    "history_off": status.history_off,
    "trace_off": status.trace_off,
    "disk_cache_off": status.disk_cache_off,
    "details": status.details,
    "memory_cache_entries": status.memory_cache_entries,
    "policy_settings": server_memory.POLICY_SETTINGS,
    "effective_disable_history": dspy.settings.disable_history,
    "effective_max_trace_size": dspy.settings.max_trace_size,
    "calls": len(lm_history_lengths),
    "lm_history_lengths": lm_history_lengths,
    "global_history": len(base_lm.GLOBAL_HISTORY),
    "trace": len(dspy.settings.trace or []),
    "drift": server_memory.check_policy_in_force(),
}))
"""


_ASYNC_OWNER_SCRIPT = """
import asyncio
import json

import dspy

from fastworkflow.run_fastapi_mcp import server_memory

server_memory.install_policy()


async def intruder_task():
    try:
        dspy.settings.configure(disable_history=False, max_trace_size=10000)
    except RuntimeError as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}
    return {"exception": None, "message": None}


async def owning_task():
    claimed = server_memory.claim_async_owner()
    intruder = await asyncio.create_task(intruder_task())
    return claimed, intruder


claimed, intruder = asyncio.run(owning_task())

print(json.dumps({
    "claimed": claimed,
    "intruder": intruder,
    "disable_history_after": dspy.settings.disable_history,
    "max_trace_size_after": dspy.settings.max_trace_size,
    "drift_after": server_memory.check_policy_in_force(),
}))
"""


_LIFESPAN_SCRIPT = """
import asyncio
import json

from fastworkflow.run_fastapi_mcp import server_memory

# What main() does before uvicorn.run(): install synchronously, before the
# event loop exists and before any LM does.
server_memory.install_policy()

import fastworkflow.run_fastapi_mcp.__main__ as main_module

observations = {"claimed_before_lifespans": server_memory.policy_status().async_owner_claimed}
errors = []


async def run_lifespan():
    async with main_module.lifespan(main_module.app):
        return server_memory.policy_status().async_owner_claimed


for label in ("first", "second"):
    try:
        observations[f"claimed_inside_{label}"] = asyncio.run(run_lifespan())
    except Exception as exc:
        errors.append(f"{label} lifespan: {type(exc).__name__}: {exc}")
        observations[f"claimed_inside_{label}"] = None

observations["errors"] = errors
observations["reclaim_returns"] = server_memory.claim_async_owner()
observations["drift"] = server_memory.check_policy_in_force()

print(json.dumps(observations))
"""


_READINESS_DRIFT_SCRIPT = """
import json

import dspy
from fastapi.testclient import TestClient

from fastworkflow.run_fastapi_mcp import server_memory
import fastworkflow.run_fastapi_mcp.__main__ as main_module

observations = {}

with TestClient(main_module.app) as client:
    # No policy yet: check_policy_in_force() is a no-op, so readiness must not
    # fail merely because the app was imported outside the server entrypoint.
    observations["ready_before_policy"] = client.get("/probes/readyz").status_code

    server_memory.install_policy()
    observations["drift_after_install"] = server_memory.check_policy_in_force()
    observations["ready_with_policy"] = client.get("/probes/readyz").status_code

    # Attribute assignment on dspy.settings routes through configure(), so drift
    # needs no explicit opt-in to happen in application code.
    dspy.settings.configure(max_trace_size=10000)
    observations["drift_after_override"] = server_memory.check_policy_in_force()
    drifted = client.get("/probes/readyz")
    observations["ready_after_drift"] = drifted.status_code
    observations["body_after_drift"] = drifted.json()

print(json.dumps(observations))
"""


_CACHE_BOUND_SCRIPT = """
import json

import dspy
from dspy.clients import base_lm
from dspy.clients import lm as dspy_lm_module
from litellm import ModelResponse
from litellm.types.utils import Choices, Message

from fastworkflow.run_fastapi_mcp import server_memory

CAP = 5
CALLS = CAP * 4

provider_calls = []


def stub_litellm_completion(request, num_retries, cache=None):
    provider_calls.append(request["messages"][-1]["content"])
    return ModelResponse(
        choices=[
            Choices(
                message=Message(
                    content="[[ ## answer ## ]]\\nok\\n\\n[[ ## completed ## ]]",
                    role="assistant",
                )
            )
        ],
        model=request["model"],
    )


# Below LM.forward, so the real request_cache wrapper runs. It keys on
# f"{fn.__module__}.{fn.__qualname__}", which a Mock does not carry.
dspy_lm_module.litellm_completion = stub_litellm_completion

status = server_memory.install_policy(memory_cache_entries=CAP)

lm = dspy.LM(model="fastworkflow/server-memory-cache-probe", cache=True)
observed_entries = []
with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
    predict = dspy.Predict(server_memory._ProbeSignature)
    for index in range(CALLS):
        predict(question=f"cache bound probe {index}")
        observed_entries.append(server_memory.dspy_cache_metrics()["entries"])
    provider_calls_before_repeat = len(provider_calls)
    predict(question=f"cache bound probe {CALLS - 1}")
    provider_calls_after_repeat = len(provider_calls)

metrics = server_memory.dspy_cache_metrics()
print(json.dumps({
    "cap": CAP,
    "calls": CALLS,
    "default_cap": server_memory.SERVER_DSPY_MEMORY_CACHE_ENTRIES,
    "provider_calls_before_repeat": provider_calls_before_repeat,
    "provider_calls_after_repeat": provider_calls_after_repeat,
    "max_observed_entries": max(observed_entries),
    "entries": metrics["entries"],
    "max_entries": metrics["max_entries"],
    "approx_bytes": metrics["approx_bytes"],
    "disk_cache_enabled": metrics["disk_cache_enabled"],
    "cache_enable_disk_cache": bool(getattr(dspy.cache, "enable_disk_cache", False)),
    "disk_cache_truthy": bool(dspy.cache.disk_cache),
    "disk_cache_len": len(dspy.cache.disk_cache),
    "lm_history": len(lm.history),
    "global_history": len(base_lm.GLOBAL_HISTORY),
    "trace": len(dspy.settings.trace or []),
}))
"""


def test_dspy_defaults_retain_history_trace_and_lm_history():
    """Steps 1-2: a real Predict call under DEFAULT settings fills all three retainers.

    This is the baseline the policy is measured against. Without it, "the
    structures are empty" is equally consistent with the probe never reaching
    the code that fills them.
    """
    observed = _run_in_venv(_DEFAULTS_BASELINE_SCRIPT)

    assert observed["dspy_version"].startswith("3."), (
        f"expected the pinned DSPy 3.x, got {observed['dspy_version']}"
    )
    assert observed["disable_history"] is False
    assert observed["max_trace_size"] == 10000
    assert observed["global_history"] == 1, "dspy.clients.base_lm.GLOBAL_HISTORY did not grow"
    assert observed["lm_history"] == 1, "the LM's own .history did not grow"
    assert observed["trace"] == 1, "dspy.settings.trace did not grow"


def test_server_policy_keeps_history_and_trace_empty():
    """Steps 3-5: a fresh policy subprocess makes 8 unique calls and retains nothing.

    Each question is unique so a cache hit cannot be mistaken for a policy that
    works, and the calls run through a thread pool because that is the shape the
    server uses.
    """
    observed = _run_in_venv(_POLICY_SCRIPT)

    assert observed["calls"] == 8
    assert observed["lm_history_lengths"] == [0] * 8, "a per-LM history retained entries"
    assert observed["global_history"] == 0, "GLOBAL_HISTORY retained entries"
    assert observed["trace"] == 0, "dspy.settings.trace retained entries"
    assert observed["drift"] is None


def test_install_policy_asserts_dspy_actually_reads_both_keys():
    """Step 7: install_policy() proves the release honours both policy keys.

    ``Settings.configure`` never rejects a key it does not read, so on a DSPy
    release that dropped ``disable_history`` or ``max_trace_size`` the policy
    would be a silent no-op. This is the same assertion the server entrypoint
    runs, which is what keeps the test and the runtime guarantee from drifting.
    """
    observed = _run_in_venv(_POLICY_SCRIPT)

    assert observed["asserted"] is True
    assert observed["history_off"] is True
    assert observed["trace_off"] is True
    assert observed["disk_cache_off"] is False
    # The cap is read back off the live cache, not echoed from the request, so
    # a release that ignored memory_max_entries could not be logged as bounded.
    assert observed["details"] == {
        "global_history_entries": 0,
        "lm_history_entries": 0,
        "trace_entries": 0,
        "memory_cache_maxsize": server_memory.SERVER_DSPY_MEMORY_CACHE_ENTRIES,
    }
    assert observed["policy_settings"] == {"disable_history": True, "max_trace_size": 0}
    assert observed["effective_disable_history"] is True
    assert observed["effective_max_trace_size"] == 0


def test_repeated_lifespans_claim_the_async_owner_exactly_once(server_argv_preamble):
    """Step 6: two lifespans on one installed policy, one owner-task configure.

    A synchronous install claims the owning thread but leaves DSPy's async owner
    unset, so the first async task to configure would take over and undo the
    policy. Claiming it in the lifespan closes that hole — but only if the second
    lifespan does not configure again, which from a different task is a
    RuntimeError.
    """
    observed = _run_in_venv(server_argv_preamble + _LIFESPAN_SCRIPT)

    assert observed["errors"] == []
    assert observed["claimed_before_lifespans"] is False
    # Nothing else flips the flag, so the first lifespan's claim returned True.
    assert observed["claimed_inside_first"] is True
    assert observed["claimed_inside_second"] is True
    assert observed["reclaim_returns"] is False
    assert observed["drift"] is None


def test_post_readiness_async_configure_is_rejected():
    """Step 8 (first half): a non-owning async task cannot configure the policy away."""
    observed = _run_in_venv(_ASYNC_OWNER_SCRIPT)

    assert observed["claimed"] is True
    assert observed["intruder"]["exception"] == "RuntimeError", (
        f"a second async task configured DSPy unopposed: {observed['intruder']}"
    )
    assert observed["disable_history_after"] is True
    assert observed["max_trace_size_after"] == 0
    assert observed["drift_after"] is None


def test_readiness_probe_reports_unready_once_the_policy_drifts(server_argv_preamble):
    """Step 8 (second half): drift turns into an unready pod, not a quiet leak.

    The startup probe proves the policy was in force once. Readiness is where
    losing it later gets caught, so the 503 is the whole mechanism.
    """
    observed = _run_in_venv(server_argv_preamble + _READINESS_DRIFT_SCRIPT)

    assert observed["ready_before_policy"] == 200
    assert observed["drift_after_install"] is None
    assert observed["ready_with_policy"] == 200
    assert observed["drift_after_override"] == "max_trace_size=10000 (expected 0)"
    assert observed["ready_after_drift"] == 503
    assert observed["body_after_drift"]["status"] == "not_ready"
    assert "drifted" in observed["body_after_drift"]["checks"]["dspy_memory_policy"]


def test_response_cache_is_bounded_and_disk_cache_is_enabled():
    """Step 9: the response cache honours its entry cap; disk cache stays on.

    Installed with a cap of 5 and driven with 20 unique requests so the bound is
    observable. The repeat call at the end must not reach the provider, which is
    what makes the entry count evidence of a live cache rather than a dead one.
    Disk cache is intentionally enabled (size-capped) for repeat-prompt hits.
    """
    observed = _run_in_venv(_CACHE_BOUND_SCRIPT)

    assert observed["provider_calls_before_repeat"] == observed["calls"]
    assert observed["provider_calls_after_repeat"] == observed["calls"], (
        "the repeat request reached the provider, so nothing was actually cached"
    )
    assert observed["max_entries"] == observed["cap"]
    assert observed["max_observed_entries"] <= observed["cap"]
    assert observed["entries"] <= observed["max_entries"]
    assert observed["default_cap"] == 200
    assert observed["disk_cache_enabled"] is True
    assert observed["cache_enable_disk_cache"] is True
    assert observed["disk_cache_truthy"] is True
    # The bound is worthless if the calls that filled the cache also filled the
    # retainers the policy is supposed to have closed.
    assert observed["lm_history"] == 0
    assert observed["global_history"] == 0
    assert observed["trace"] == 0


def _declared_dspy_minimum() -> str:
    with open(os.path.join(_REPO_ROOT, "pyproject.toml"), "rb") as handle:
        pyproject = tomllib.load(handle)
    constraint = pyproject["tool"]["poetry"]["dependencies"]["dspy"]
    if isinstance(constraint, dict):
        constraint = constraint.get("version", "")
    return constraint.lstrip("^~><= ").strip()


def _interpreter_carrying_dspy(version: str):
    candidates = [
        os.environ.get("FASTWORKFLOW_DSPY_MIN_PYTHON"),
        os.path.join(_REPO_ROOT, f".venv-dspy-{version}", "bin", "python"),
        os.path.join(_REPO_ROOT, ".venv-dspy-min", "bin", "python"),
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        probe = subprocess.run(
            [candidate, "-c", "import dspy; print(dspy.__version__)"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if probe.returncode == 0 and probe.stdout.strip() == version:
            return candidate
    return None


def test_policy_holds_on_the_minimum_supported_dspy():
    """The policy is only as good as the floor of the declared dependency range.

    ``dspy = "^3.0.1"`` lets a downstream install resolve to 3.0.1, which the
    pinned virtualenv (3.2.1) says nothing about. This runs the two assertions
    that would break first — that both policy keys are read, and that the cache
    cap holds — against an interpreter carrying exactly that version.
    """
    minimum = _declared_dspy_minimum()
    interpreter = _interpreter_carrying_dspy(minimum)
    if interpreter is None:
        pytest.skip(
            f"no interpreter carrying the declared minimum dspy {minimum} "
            f"(pyproject.toml declares dspy ^{minimum}); point "
            f"FASTWORKFLOW_DSPY_MIN_PYTHON at one, or create "
            f"{os.path.join(_REPO_ROOT, f'.venv-dspy-{minimum}')}"
        )

    policy = _run_in_venv(_POLICY_SCRIPT, python=interpreter)
    assert policy["dspy_version"] == minimum
    assert policy["asserted"] is True
    assert policy["global_history"] == 0
    assert policy["lm_history_lengths"] == [0] * policy["calls"]
    assert policy["trace"] == 0

    cache = _run_in_venv(_CACHE_BOUND_SCRIPT, python=interpreter)
    assert cache["entries"] <= cache["cap"]
    assert cache["disk_cache_enabled"] is True
