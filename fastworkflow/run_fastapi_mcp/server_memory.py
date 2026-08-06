"""Server-only memory policy for the run_fastapi_mcp process.

DSPy keeps three things across calls that grow with request size: a global
diagnostic history, a per-LM history, and the predictor trace. It also ships a
response cache defaulting to a million in-memory entries and a 30 GB disk cache.
None of that is useful to a long-lived server, and all of it retains prompts.

This module turns them off (or bounds them), and — because configuring is not
evidence that a policy is in force — proves it structurally before the server
accepts traffic, then keeps owning it:

* ``install_policy()`` runs synchronously from the server entrypoint, BEFORE
  uvicorn creates the event loop and before any LM exists. DSPy's global config
  belongs to the first thread that configures it, so claiming it here is what
  makes the policy the process's policy.
* ``claim_async_owner()`` runs once from the lifespan task. A synchronous
  configure leaves DSPy's *async* owner unset, so the first async task to call
  ``dspy.configure()`` would silently take over and undo the policy while the
  startup probe stayed green in the logs. Claiming the async owner turns that
  into DSPy's own ownership error.
* ``check_policy_in_force()`` runs on the readiness probe, so drift becomes an
  unready pod rather than a quiet leak.

Nothing here applies outside this server: training, build, refine and the CLI
keep DSPy's defaults. Application code uses ``dspy.context(...)`` for temporary
overrides, never process-global ``configure()``, which is what makes claiming
global ownership here safe.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import dspy
from dspy.clients import base_lm
from litellm import ModelResponse
from litellm.types.utils import Choices, Message

from fastworkflow.utils.logging import logger


# Bounded response cache. The retainer under scrutiny is process memory, so the
# entry count is the load-bearing knob; a million entries of request-sized
# prompts is unbounded for practical purposes. 200 entries is ~90 MB worst case
# at a 450 KB payload, sized to keep worst-case retention comparable to the
# live-session budget. The disk cache is off: 30 GB is not a sane container
# default and DSPy's disk cache is pickle-backed.
SERVER_DSPY_MEMORY_CACHE_ENTRIES = 200

# The settings that constitute the policy. Re-applied when the async owner is
# claimed, and compared against on every readiness probe.
POLICY_SETTINGS: dict[str, Any] = {
    "disable_history": True,
    "max_trace_size": 0,
}

# Deep-size walk guard: the cache holds arbitrary provider objects.
_MAX_SIZE_WALK_NODES = 200_000


class DSPyPolicyError(RuntimeError):
    """The DSPy memory policy could not be installed or proven to be in force."""


@dataclass
class PolicyStatus:
    """What was installed, and what was actually proven about it."""

    dspy_version: str
    history_off: bool = False
    trace_off: bool = False
    memory_cache_entries: int = SERVER_DSPY_MEMORY_CACHE_ENTRIES
    cache_bounded: bool = False
    disk_cache_off: bool = False
    async_owner_claimed: bool = False
    asserted: bool = False
    details: dict[str, Any] = field(default_factory=dict)


_status: Optional[PolicyStatus] = None
_reported_drift: Optional[str] = None


def policy_status() -> Optional[PolicyStatus]:
    """The installed policy, or None when this process never installed one."""
    return _status


class _ProbeSignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _ProbeLM(dspy.LM):
    """An LM whose provider call is replaced, and nothing else.

    Only ``forward`` is stubbed. Everything above it — ``_process_lm_response``,
    the history append, ``Predict._forward_postprocess`` and its trace append —
    is the real code, which is the entire point: a probe that skipped them would
    prove nothing. The response must be a real litellm ``ModelResponse``; a
    plain dict raises inside ``_process_completion``.
    """

    def __init__(self) -> None:
        super().__init__(model="fastworkflow/server-memory-probe", cache=False)

    def forward(self, prompt=None, messages=None, **kwargs):
        return ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]",
                        role="assistant",
                    )
                )
            ],
            model=self.model,
        )


def _settings_module():
    """DSPy's settings *module* (``dspy.dsp.utils.settings`` resolves to the singleton)."""
    return sys.modules["dspy.dsp.utils.settings"]


def _effective(key: str) -> Any:
    """The process-wide value of a setting, ignoring any thread-local override."""
    return _settings_module().main_thread_config.get(key)


def install_policy(
    memory_cache_entries: int = SERVER_DSPY_MEMORY_CACHE_ENTRIES,
) -> PolicyStatus:
    """Install the policy, prove it structurally, and claim the owning thread.

    Call this synchronously from the server entrypoint before the event loop
    exists. Idempotent: a second call re-proves the policy rather than fighting
    over it.

    Raises DSPyPolicyError if the policy cannot be installed or if the installed
    DSPy does not actually honour it.
    """
    global _status

    dspy_version = getattr(dspy, "__version__", "unknown")

    # Order matters. The helper cannot find arbitrary pre-existing per-LM
    # histories, so clear the global one first; and configure_cache REPLACES the
    # global cache object, so it has to happen before any LM is created.
    base_lm.GLOBAL_HISTORY.clear()

    previous_cache = getattr(dspy, "cache", None)
    try:
        dspy.configure_cache(
            enable_memory_cache=True,
            memory_max_entries=memory_cache_entries,
            enable_disk_cache=False,
        )
        dspy.settings.configure(**POLICY_SETTINGS, trace=[])
    except Exception as exc:
        raise DSPyPolicyError(
            f"Could not install the DSPy memory policy on dspy {dspy_version}: {exc}"
        ) from exc

    _close_replaced_disk_cache(previous_cache)

    status = PolicyStatus(
        dspy_version=dspy_version,
        memory_cache_entries=memory_cache_entries,
        disk_cache_off=not getattr(dspy.cache, "enable_disk_cache", True),
    )
    _assert_policy_structurally(status)
    _status = status
    return status


def _close_replaced_disk_cache(previous_cache: Any) -> None:
    """Release the disk cache configure_cache orphaned.

    ``configure_cache`` rebinds ``dspy.cache``; the cache built at import time
    still holds its sqlite shards open for the life of the process.
    """
    disk_cache = getattr(previous_cache, "disk_cache", None)
    if disk_cache is None or not hasattr(disk_cache, "close"):
        return
    try:
        disk_cache.close()
    except Exception as exc:  # best effort; never block startup on a cache handle
        logger.debug(f"Could not close the replaced DSPy disk cache: {exc}")


def _assert_policy_structurally(status: PolicyStatus) -> None:
    """Prove the policy by running a real call through it and finding nothing retained.

    ``Settings.configure`` does no key validation — it is a dict assignment — so
    on a DSPy release that does not read ``disable_history`` or
    ``max_trace_size`` the policy would be a silent no-op while startup happily
    logged "off". This is the difference between configuring and knowing.
    """
    probe_lm = _ProbeLM()
    try:
        with dspy.context(lm=probe_lm, adapter=dspy.ChatAdapter()):
            dspy.Predict(_ProbeSignature)(question="memory policy probe")
    except Exception as exc:
        raise DSPyPolicyError(
            f"DSPy memory policy probe failed on dspy {status.dspy_version}: {exc}"
        ) from exc

    global_history = len(base_lm.GLOBAL_HISTORY)
    lm_history = len(probe_lm.history)
    trace = len(dspy.settings.trace or [])
    cache_maxsize = _memory_cache_maxsize()

    status.history_off = global_history == 0 and lm_history == 0
    status.trace_off = trace == 0
    # The cap is read back off the live cache rather than echoed from the
    # request: a release that ignored memory_max_entries would otherwise be
    # logged as bounded while holding a million entries.
    status.cache_bounded = cache_maxsize == status.memory_cache_entries
    status.asserted = status.history_off and status.trace_off and status.cache_bounded
    status.details = {
        "global_history_entries": global_history,
        "lm_history_entries": lm_history,
        "trace_entries": trace,
        "memory_cache_maxsize": cache_maxsize,
    }

    if not status.asserted:
        raise DSPyPolicyError(
            f"DSPy {status.dspy_version} did not honour the memory policy "
            f"{POLICY_SETTINGS} / {status.memory_cache_entries} cache entries: "
            f"global_history={global_history}, lm_history={lm_history}, "
            f"trace={trace}, memory_cache_maxsize={cache_maxsize}. This DSPy "
            f"release retains request-sized data per call; pin a release that "
            f"reads these keys."
        )

    base_lm.GLOBAL_HISTORY.clear()


def claim_async_owner() -> bool:
    """Claim DSPy's async config owner so later async ``configure()`` calls are refused.

    Call once from the lifespan startup, before readiness. A synchronous
    configure claims the owning *thread* but leaves the async owner unset, so
    the first async task to configure would take ownership and undo the policy.
    Returns True if this call claimed it.

    No-op when no policy was installed (i.e. not running under the server
    entrypoint), so importing the app in tests does not touch global DSPy state.
    """
    if _status is None or _status.async_owner_claimed:
        return False

    try:
        dspy.settings.configure(**POLICY_SETTINGS)
    except RuntimeError as exc:
        raise DSPyPolicyError(
            f"Another owner already holds DSPy's global settings, so the server "
            f"memory policy cannot be guaranteed: {exc}"
        ) from exc

    _status.async_owner_claimed = True
    return True


def check_policy_in_force() -> Optional[str]:
    """Re-assert the policy cheaply; return a description of the drift, or None.

    A one-time probe proves the policy was in force at startup, not that it
    stayed there — attribute assignment on ``dspy.settings`` routes through
    ``configure()``, so drift needs no explicit call to happen. Readiness is
    where that gets caught.
    """
    if _status is None:
        return None

    drift = [
        f"{key}={_effective(key)!r} (expected {expected!r})"
        for key, expected in POLICY_SETTINGS.items()
        if _effective(key) != expected
    ]
    if getattr(dspy.cache, "enable_disk_cache", False):
        drift.append("disk cache re-enabled")
    if _memory_cache_maxsize() != _status.memory_cache_entries:
        drift.append(
            f"memory cache maxsize={_memory_cache_maxsize()} "
            f"(expected {_status.memory_cache_entries})"
        )
    return ", ".join(drift) or None


def log_policy_drift(drift: Optional[str]) -> None:
    """Report drift once per transition.

    Readiness is probed every few seconds, so logging on every check would bury
    the surrounding diagnostics under thousands of identical lines a day.
    """
    global _reported_drift
    if drift == _reported_drift:
        return
    _reported_drift = drift
    if drift is None:
        logger.info("DSPy memory policy is back in force")
    else:
        logger.error(f"DSPy memory policy drift detected after readiness: {drift}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _memory_cache_maxsize() -> Optional[int]:
    return getattr(getattr(dspy.cache, "memory_cache", None), "maxsize", None)


def _deep_size(value: Any) -> int:
    """Approximate retained bytes, following containers and plain objects once each."""
    seen: set[int] = set()
    total = 0
    pending = [value]
    while pending and len(seen) < _MAX_SIZE_WALK_NODES:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        try:
            total += sys.getsizeof(item)
        except TypeError:
            continue
        if isinstance(item, (str, bytes, bytearray)):
            continue
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
        elif hasattr(item, "__dict__"):
            pending.append(vars(item))
    return total


def dspy_cache_metrics() -> dict[str, Any]:
    """Entry count, cap and approximate bytes of the DSPy response cache."""
    memory_cache = getattr(dspy.cache, "memory_cache", None)
    entries = len(memory_cache) if memory_cache is not None else 0
    return {
        "entries": entries,
        "max_entries": _memory_cache_maxsize(),
        "approx_bytes": _deep_size(dict(memory_cache)) if entries else 0,
        "disk_cache_enabled": bool(getattr(dspy.cache, "enable_disk_cache", False)),
    }


def conversation_memory_metrics(runtimes) -> dict[str, Any]:
    """Turn count and approximate bytes held by in-memory conversation history.

    Counts only the turn payloads, which is what actually scales with requests.
    """
    turns = 0
    total_bytes = 0
    for runtime in runtimes:
        for message in runtime.execution_context.conversation_history.messages:
            turns += 1
            for value in message.values():
                if isinstance(value, str):
                    total_bytes += len(value)
                elif value is not None:
                    total_bytes += _deep_size(value)
    return {"turns": turns, "approx_bytes": total_bytes}
