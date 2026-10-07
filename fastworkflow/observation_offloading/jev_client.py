"""The plumbing shared by every decision-model feature, and the Jev (TypeSafe) provider.

The finish-time execution check (``finish_check``) and search routing
(``search_router``) each turn on with their own flag set to ``jev`` plus a
``JEV_API_KEY``, or set to the name of a provider registered from code
(``decision.register_decision_provider``). This module owns what they have in
common: the optional SDK import, the key's env name and the default model, the
endpoint, redaction of what is sent, client construction (one attempt, no
retries), the Jev adapter (``JevProvider``: the SDK's question objects built
from ``decision``'s neutral questions, its errors mapped to ``decision``'s), the
build of a registered provider, a per-process client cache, the warning logged
when a flag is set but its feature cannot activate, and how a failed call is
described and warned about.

An unset flag, or one set to an explicit off value (``off``, ``0``, ``false``,
``no``, ``none``), is silent. A set flag that cannot take effect -- an unrecognised
value, the SDK not installed, no key, an endpoint that may not be used, or
``FW_OFFLOAD_EVIDENCE_REDACTION=off`` -- logs one warning per process per cause
and the feature stays off. A registered provider passes the same redaction
gate; one whose factory fails, or returns no ``DecisionProvider``, warns once
and the feature stays off. Selecting one warns once per flag that the feature's
thresholds were calibrated with Jev. Every value sent passes ``egress`` first: the
credential-scrubbed form the archive would store, or None -- send nothing --
when redaction is off. The raw key is
never stored or logged: caches are keyed by a short SHA-256 fingerprint, so a
rotated key builds a new client.

The endpoint is ``FW_JEV_BASE_URL`` (default ``https://api.typesafe.ai``) and
nothing else: it receives the key and every payload, so it must be https,
except plain http to a loopback address (a local stand-in). The SDK's own
``TYPESAFE_BASE_URL`` is ignored, with one warning.

The SDK logs whole request and response bodies, unredacted, at DEBUG; a filter
on its logger drops those records whatever the log levels, keeping its INFO
line (status, latency, request id).

Every call has a hard wall-clock bound (``bounded_call``). The SDK's timeout is
per network phase -- a body trickled in slices, or a slow resolver, outlasts it
many times over -- so each request runs on its own daemon worker thread, at
most ``VENDOR_WORKERS`` slots held process-wide, and the caller waits at most ``min(cap,
time left)``; past that the request is abandoned and the caller fails open. An
abandoned request may still complete, and be billed, vendor-side; it is never
retried, and it never delays interpreter exit. Every call passes the SDK a
timeout of its cutoff plus ``CUTOFF_MARGIN_SECONDS`` (or the client's own, if
shorter), and an abandoned request gives its worker slot back at the latest
``CUTOFF_MARGIN_SECONDS`` after its cutoff even if a trickled body keeps it
running (``calls_orphaned``), so abandoned requests cannot hold every slot.
When every worker is busy, or ``ORPHANED_CALLS_MAX`` abandoned requests still
run, a call fails at once (``VendorBusy``) rather than queueing, and sends
nothing. The finish check's events and the router's records carry
``vendor_calls_in_flight`` (``calls_in_flight``) when any slot is held and
``vendor_calls_orphaned`` (``calls_orphaned``) when any orphan still runs
(``vendor_pressure``). The time left is the smaller of the caller's own
deadline and the turn's ``TurnBudget``: ``TURN_VENDOR_SECONDS`` of time
actually spent waiting on vendor calls, all features together, and at most
``ROUTER_CALLS_PER_TURN`` routing calls.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import ipaddress
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Hashable, Optional
from urllib.parse import urlsplit

try:
    from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeAPIError, TypeSafeClient
except ImportError:  # optional dependency: without it no Jev-backed feature activates
    Choice = Noul = RetryPolicy = TypeSafeAPIError = TypeSafeClient = None

from fastworkflow import context_budget
from fastworkflow.observation_offloading import decision
from fastworkflow.observation_offloading.archive import (
    REDACTION_ENV,
    REDACTION_OFF,
    capture_record_for,
    redaction_enabled,
)
from fastworkflow.utils.logging import logger

JEV = "jev"
KEY_ENV = "JEV_API_KEY"
DEFAULT_MODEL = "jev-1.13.0"
INSTALL_HINT = "pip install 'fastworkflow[jev]'"
#: Flag values (stripped, any case) that say "off" and so turn it off silently.
OFF_VALUES = decision.OFF_VALUES
_FINGERPRINT_CHARS = 16

BASE_URL_ENV = "FW_JEV_BASE_URL"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
#: The SDK's own endpoint variable, which it would read if no base_url were passed.
SDK_BASE_URL_ENV = "TYPESAFE_BASE_URL"
SDK_LOGGER = "typesafe_sdk"

#: The code the API puts in an error body when the state is too long for the model.
STATE_TOO_LARGE_CODE = "max_tokens_exceeded"
#: A failure of one kind (error type, HTTP status) is warned about at most once per this.
FAILURE_WARN_INTERVAL_SECONDS = 300.0
#: What counts as a machine-readable error code: one token, never a sentence.
_CODE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_CODE_KEYS = ("code", "type", "error_type")
_REQUEST_ID_CHARS = 128

#: Vendor requests in flight at once, process-wide; one more fails at once (``VendorBusy``).
VENDOR_WORKERS = 4
#: Abandoned requests still running after their slot was given back (``calls_orphaned``);
#: at this many every new call fails at once (``VendorBusy``), so a trickling
#: endpoint cannot pile up threads and sockets without bound.
ORPHANED_CALLS_MAX = 8
#: What one turn may spend waiting on vendor calls, every feature together.
TURN_VENDOR_SECONDS = 10.0
#: Routing calls one turn may make; later searches use the search model.
ROUTER_CALLS_PER_TURN = 3
#: A call with less time than this left is not sent.
MIN_CALL_SECONDS = 0.05
#: The SDK's own timeout is set this much past the hard cutoff, so the cutoff
#: always decides and the SDK only frees the abandoned worker.
CUTOFF_MARGIN_SECONDS = 0.5

_WARNED: set[tuple[str, ...]] = set()
_WARNED_LOCK = threading.Lock()


class _DropWireBodies(logging.Filter):
    """Drops the SDK's DEBUG records: they carry whole request and response bodies, unredacted."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno > logging.DEBUG


def _guard_sdk_logger() -> None:
    sdk_logger = logging.getLogger(SDK_LOGGER)
    if not any(isinstance(existing, _DropWireBodies) for existing in sdk_logger.filters):
        sdk_logger.addFilter(_DropWireBodies())


if TypeSafeClient is not None:
    _guard_sdk_logger()


def redacted(text: str) -> str:
    """*text* as the archive would store it; what may be sent."""
    stored, _record = capture_record_for(text)
    return stored


def egress(text: str) -> Optional[str]:
    """*text* as it may leave the process, or None when it may not leave at all.

    None when redaction is off: unscrubbed text never leaves the process.
    """
    stored, record = capture_record_for(text)
    if record.get("redaction") == REDACTION_OFF:
        return None
    return stored


def withholding_cause() -> Optional[str]:
    """Why what the Jev features send would be left unredacted, or None when it would not."""
    if not redaction_enabled():
        return f"{REDACTION_ENV}={REDACTION_OFF}, so nothing sent would be redacted"
    return None


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def _log_once(cause: tuple[str, ...], level: int, message: str, *args: Any) -> None:
    with _WARNED_LOCK:
        if cause in _WARNED:
            return
        _WARNED.add(cause)
    logger.log(level, message, *args)


def _warn_once(cause: tuple[str, ...], message: str, *args: Any) -> None:
    _log_once(cause, logging.WARNING, message, *args)


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def base_url() -> Optional[str]:
    """The endpoint every Jev feature uses, or None when ``FW_JEV_BASE_URL`` names one it may not.

    Unset means ``DEFAULT_BASE_URL``. A set value must be https, or plain http to
    a loopback address, with no credentials, query or fragment; anything else
    warns once and every Jev feature stays off. A non-default endpoint's host is
    logged once. ``TYPESAFE_BASE_URL`` is never used: ``make_client`` always
    passes the endpoint, and a set one warns once.
    """
    if os.environ.get(SDK_BASE_URL_ENV, "").strip():
        _warn_once((SDK_BASE_URL_ENV,),
                   "%s is set but ignored; fastWorkflow's Jev features use %s (default %s)",
                   SDK_BASE_URL_ENV, BASE_URL_ENV, DEFAULT_BASE_URL)
    raw = (context_budget.env_value(BASE_URL_ENV) or "").strip()
    if not raw:
        return DEFAULT_BASE_URL
    url = raw.rstrip("/")
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        _port = parts.port  # raises on a malformed port
        allowed = bool(host) and not (parts.username or parts.password or parts.query or parts.fragment) and (
            parts.scheme == "https" or (parts.scheme == "http" and _is_loopback(host)))
    except ValueError:
        parts, host, allowed = None, "", False
    if not allowed:
        # The value itself is not logged: it may carry credentials.
        _warn_once((BASE_URL_ENV, "rejected", key_fingerprint(url)),
                   "%s must be an https URL (plain http only to a loopback address) without "
                   "credentials, query or fragment; got scheme %r, host %r. The Jev features stay off",
                   BASE_URL_ENV, parts.scheme if parts is not None else "", host)
        return None
    if url != DEFAULT_BASE_URL:
        _log_once((BASE_URL_ENV, "host", url), logging.INFO,
                  "%s: the Jev features send their key and payloads to %s", BASE_URL_ENV, parts.netloc)
    return url


def flag_value(flag_env: str) -> str:
    """*flag_env* as a feature reads it: stripped, lower case, ``""`` when unset."""
    return (context_budget.env_value(flag_env) or "").strip().lower()


def _redaction_allows(flag_env: str, value: str, feature: str) -> bool:
    """False, with one warning per cause, when what *feature* sends would be left unredacted."""
    cause = withholding_cause()
    if cause is not None:
        _warn_once((flag_env, "redaction", cause),
                   "%s=%s but %s; %s stays off", flag_env, value, cause, feature)
        return False
    return True


def requested_key(flag_env: str, feature: str) -> Optional[str]:
    """The API key when *flag_env* asks for Jev and the feature can activate, else None.

    The flag is read before the SDK is looked for, so a set flag without the
    SDK warns rather than staying silently off. Last, ``withholding_cause`` is
    asked: redaction switched off keeps the feature off with one warning.
    The endpoint is checked separately (``base_url``), after this has returned
    a key. A value naming a registered provider returns None without a warning:
    the caller asks ``decision.registration`` first.
    """
    value = flag_value(flag_env)
    if value in OFF_VALUES:
        return None
    if value != JEV:
        if decision.registration(value) is None:
            _warn_once((flag_env, "value", value),
                       "%s=%r is not recognised; %s stays off (set %s=%s, or the name of a "
                       "registered decision provider, to turn it on)",
                       flag_env, value, feature, flag_env, JEV)
        return None
    if TypeSafeClient is None:
        _warn_once((flag_env, "sdk"),
                   "%s=%s but typesafe-sdk is not installed; %s stays off (%s)",
                   flag_env, JEV, feature, INSTALL_HINT)
        return None
    key = context_budget.env_value(KEY_ENV)
    if not key:
        _warn_once((flag_env, "key"),
                   "%s=%s but %s is not set; %s stays off (set %s)",
                   flag_env, JEV, KEY_ENV, feature, KEY_ENV)
        return None
    if not _redaction_allows(flag_env, JEV, feature):
        return None
    return key


def registered_provider(flag_env: str, entry: decision.Registration, feature: str) -> Optional[Any]:
    """The provider *entry* registers, for *feature* selected by *flag_env*; None when it cannot activate.

    The redaction gate applies as it does to Jev. A factory that raises, or
    returns something that is not a ``decision.DecisionProvider``, warns once
    per registration and the feature stays off. The first activation per flag
    and name warns that the feature was calibrated with Jev.
    """
    if not _redaction_allows(flag_env, entry.name, feature):
        return None
    try:
        provider = decision.build_provider(entry)
    except Exception as error:  # noqa: BLE001 - a broken provider leaves the feature off, not the agent
        _warn_once((flag_env, "factory", entry.name, str(entry.generation)),
                   "%s=%s but the decision provider registered under that name could not be built (%s); "
                   "%s stays off", flag_env, entry.name, type(error).__name__, feature)
        return None
    _warn_once((flag_env, "uncalibrated", entry.name),
               "%s=%s: %s's thresholds and questions were calibrated with Jev (%s), not with this "
               "provider; its precision and recall are unmeasured", flag_env, entry.name, feature, DEFAULT_MODEL)
    return provider


def make_client(key: str, model: str, timeout: float, url: str) -> Any:
    """One attempt per call, *timeout* seconds: these features may never hold a turn up for long.

    *url* is always passed, so the SDK never falls back to ``TYPESAFE_BASE_URL``.
    The SDK's timeout is per network phase; the wall-clock bound is ``bounded_call``'s.
    """
    return TypeSafeClient(api_key=key, model=model, timeout=timeout, base_url=url,
                          retry=RetryPolicy(max_retries=0, timeout=timeout))


# ---------------------------------------------------------------------------
# Bounded calls and the per-turn budget
# ---------------------------------------------------------------------------

class VendorBusy(Exception):
    """Every vendor worker is already waiting on a call; nothing was sent."""

    def __init__(self) -> None:
        super().__init__("vendor busy")


class CallTimedOut(Exception):
    """The call's own cap ran out before it answered; it was abandoned, not retried."""


#: The time left (the caller's deadline or the turn's budget) ran out; the call
#: was abandoned or never sent. Neutral: any provider raises it.
OutOfTime = decision.OutOfTime


#: Each request runs on its own daemon thread, at most ``VENDOR_WORKERS`` holding a slot at once:
#: an abandoned request must never hold up interpreter exit, which a
#: ``ThreadPoolExecutor`` (joined at exit) would.
_in_flight = 0
_IN_FLIGHT_LOCK = threading.Lock()
#: Abandoned requests whose worker slot was given back while they still run.
_orphaned = 0


def calls_in_flight() -> int:
    """Vendor worker slots held: requests still running, abandoned ones included
    until ``CUTOFF_MARGIN_SECONDS`` past their cutoff (see ``call_within``)."""
    with _IN_FLIGHT_LOCK:
        return _in_flight


def calls_orphaned() -> int:
    """Abandoned requests still running after their worker slot was given back."""
    with _IN_FLIGHT_LOCK:
        return _orphaned


def vendor_pressure() -> dict[str, int]:
    """``vendor_calls_in_flight`` and ``vendor_calls_orphaned``, each only when above 0:
    what a finish check event or a routing record carries."""
    with _IN_FLIGHT_LOCK:
        counts = {"vendor_calls_in_flight": _in_flight, "vendor_calls_orphaned": _orphaned}
    return {name: count for name, count in counts.items() if count}


class _Slot:
    """One worker slot, given back exactly once: when the request ends, or
    ``CUTOFF_MARGIN_SECONDS`` after it was abandoned, whichever is first."""

    def __init__(self) -> None:
        self._held = True
        self._orphaned = False

    def release(self) -> None:
        global _in_flight, _orphaned
        with _IN_FLIGHT_LOCK:
            if self._held:
                self._held = False
                _in_flight -= 1
            elif self._orphaned:
                self._orphaned = False
                _orphaned -= 1

    def reclaim(self) -> None:
        """The abandoned request still runs past its margin: free its slot, count it orphaned."""
        global _in_flight, _orphaned
        with _IN_FLIGHT_LOCK:
            if self._held:
                self._held = False
                self._orphaned = True
                _in_flight -= 1
                _orphaned += 1


def call_within(client: Any, seconds: float, *, expired: type[Exception] = CallTimedOut,
                sdk_timeout: Optional[float] = None, **kwargs: Any) -> Any:
    """``client.system_one(**kwargs)``, waited on for at most *seconds* of wall clock.

    Raises *expired* when the time runs out (the request is abandoned),
    ``VendorBusy`` at once, sending nothing, when all ``VENDOR_WORKERS`` are
    taken or ``ORPHANED_CALLS_MAX`` abandoned requests still run. *sdk_timeout*,
    when given, is passed to the call as its own timeout. The worker is a daemon
    thread, so an abandoned request never delays interpreter exit.

    An abandoned request keeps its worker slot until it ends by itself or
    ``CUTOFF_MARGIN_SECONDS`` have passed, whichever is first. The SDK's
    timeout is per network phase, so a body trickled in slices may run far
    longer; past the margin its slot is given back and it is counted in
    ``calls_orphaned`` instead, so abandoned requests cannot hold every slot;
    the orphans' threads and sockets are bounded by ``ORPHANED_CALLS_MAX``
    instead, released as each orphan ends.
    """
    global _in_flight
    with _IN_FLIGHT_LOCK:
        if _in_flight >= VENDOR_WORKERS or _orphaned >= ORPHANED_CALLS_MAX:
            raise VendorBusy()
        _in_flight += 1
    slot = _Slot()
    if sdk_timeout is not None:
        kwargs["timeout"] = sdk_timeout
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def work() -> None:
        if not future.set_running_or_notify_cancel():
            slot.release()
            return
        try:
            value = client.system_one(**kwargs)
        except BaseException as error:  # noqa: BLE001 - handed to the waiting caller
            # The slot is free before the caller wakes, so what it records next
            # does not count this call.
            slot.release()
            future.set_exception(error)
        else:
            slot.release()
            future.set_result(value)

    try:
        threading.Thread(target=work, name="jev", daemon=True).start()
    except BaseException:
        slot.release()
        raise
    # Not future.result(timeout=...): its TimeoutError is the builtin, which the
    # SDK's own timeout error also is, so the two could not be told apart.
    done, _pending = concurrent.futures.wait([future], timeout=seconds)
    if not done:
        future.cancel()
        reclaimer = threading.Timer(CUTOFF_MARGIN_SECONDS, slot.reclaim)
        reclaimer.daemon = True
        reclaimer.start()
        raise expired()
    return future.result()


class TurnBudget:
    """One turn's vendor allowance: seconds actually spent waiting on calls, and routing calls made.

    Wall-clock time between calls is not counted. Thread-safe; replaced at the
    start of every turn, kept across an ask_user resume.
    """

    def __init__(self, seconds: float = TURN_VENDOR_SECONDS,
                 router_calls: int = ROUTER_CALLS_PER_TURN) -> None:
        self.seconds = float(seconds)
        self.router_calls_max = int(router_calls)
        self.router_calls = 0
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.seconds - self._spent)

    @property
    def vendor_ms(self) -> int:
        """Milliseconds this turn has spent waiting on vendor calls so far."""
        with self._lock:
            return round(self._spent * 1000)

    def spend(self, seconds: float) -> None:
        with self._lock:
            self._spent += max(0.0, seconds)

    def take_router_call(self) -> bool:
        """Count one routing call; False, counting nothing, once the cap or the time is used up."""
        with self._lock:
            if self.router_calls >= self.router_calls_max or self.seconds - self._spent < MIN_CALL_SECONDS:
                return False
            self.router_calls += 1
            return True


def _configured_timeout(client: Any) -> Optional[float]:
    """The SDK client's own numeric timeout (``make_client``'s *timeout*), or None when unknown."""
    timeout = getattr(getattr(client, "_config", None), "timeout", None)
    return float(timeout) if isinstance(timeout, (int, float)) else None


def bounded_call(client: Any, *, cap: float, deadline: Optional[float] = None,
                 budget: Optional[TurnBudget] = None, **kwargs: Any) -> Any:
    """One ``system_one`` call, cut off after ``min(cap, deadline - now, budget.remaining)`` seconds.

    Raises ``CallTimedOut`` when *cap* was the binding limit, ``OutOfTime`` when
    the deadline or the budget was (or less than ``MIN_CALL_SECONDS`` was left,
    in which case nothing is sent), ``VendorBusy`` when the pool is full. The
    time waited is charged to *budget*.
    """
    left = float("inf")
    if deadline is not None:
        left = deadline - time.monotonic()
    if budget is not None:
        left = min(left, budget.remaining)
    if left < MIN_CALL_SECONDS:
        raise OutOfTime()
    seconds = min(cap, left)
    out_of_time = left <= cap
    started = time.monotonic()
    try:
        # Every call gets its own SDK timeout just past its cutoff, a call cut at
        # its own cap included: a client built with a longer timeout must not
        # keep an abandoned request's worker waiting on it. A shorter one is kept.
        sdk_timeout = seconds + CUTOFF_MARGIN_SECONDS
        configured = _configured_timeout(client)
        if configured is not None:
            sdk_timeout = min(sdk_timeout, configured)
        return call_within(client, seconds, expired=OutOfTime if out_of_time else CallTimedOut,
                           sdk_timeout=sdk_timeout, **kwargs)
    finally:
        if budget is not None:
            budget.spend(time.monotonic() - started)


# ---------------------------------------------------------------------------
# Failed calls
# ---------------------------------------------------------------------------

def is_api_error(error: BaseException) -> bool:
    """Whether *error* is the SDK's HTTP-response failure, which carries status, body and request id."""
    return TypeSafeAPIError is not None and isinstance(error, TypeSafeAPIError)


def state_too_large(error: BaseException) -> bool:
    """Whether the API refused a call because its state was too long for the model.

    Read from the error body, not ``str(error)``: when the server words the
    error for humans the SDK shows that message instead, and the code is gone.
    """
    if not is_api_error(error):
        return False
    try:
        return STATE_TOO_LARGE_CODE in json.dumps(error.body, default=str)
    except (TypeError, ValueError):
        return False


def error_code(body: Any) -> Optional[str]:
    """The machine-readable code in an API error body, if it has one -- never its free text."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, str) and _CODE_RE.fullmatch(error):
        return error
    for holder in (error, body.get("detail"), body):
        if isinstance(holder, dict):
            for key in _CODE_KEYS:
                value = holder.get(key)
                if isinstance(value, str) and _CODE_RE.fullmatch(value):
                    return value
    return None


def describe_failure(error: BaseException) -> dict[str, Any]:
    """``{error_type, status, request_id, code}`` for a failed call: what a log line or event may carry.

    The class name, the HTTP status, the vendor's request id and the body's
    machine-readable code -- never the message, which may echo what was sent.
    """
    api = is_api_error(error)
    request_id = error.request_id if api else None
    return {"error_type": type(error).__name__,
            "status": error.status if api else None,
            "request_id": str(request_id)[:_REQUEST_ID_CHARS] if request_id else None,
            "code": error_code(error.body) if api else None}


def failure_summary(failure: dict[str, Any], stage: Optional[str] = None) -> str:
    parts = [str(failure.get("error_type") or "error")]
    for label, key in (("status", "status"), ("request", "request_id"), ("code", "code")):
        if failure.get(key) is not None:
            parts.append(f"{label} {failure[key]}")
    if stage:
        parts.append(f"stage {stage}")
    return ", ".join(parts)


class FailureWarner:
    """One feature's failure warnings: at most one per interval per (error type, HTTP status).

    A sustained outage is not silent -- each interval's first failure is logged
    again, with how many like it went unlogged since -- and not a flood either.
    """

    def __init__(self, feature: str, consequence: str,
                 interval_seconds: float = FAILURE_WARN_INTERVAL_SECONDS) -> None:
        self.feature = feature
        self.consequence = consequence
        self.interval_seconds = interval_seconds
        self._last: dict[tuple[Any, Any], float] = {}
        self._suppressed: dict[tuple[Any, Any], int] = {}
        self._lock = threading.Lock()

    def warn(self, failure: dict[str, Any], *, stage: Optional[str] = None,
             exc_info: Optional[BaseException] = None) -> bool:
        """Log *failure* unless one like it was logged within the interval; True when logged."""
        kind = (failure.get("error_type"), failure.get("status"))
        now = time.monotonic()
        with self._lock:
            last = self._last.get(kind)
            if last is not None and now - last < self.interval_seconds:
                self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
                return False
            self._last[kind] = now
            suppressed = self._suppressed.pop(kind, 0)
        logger.warning("%s unavailable (%s); %s%s", self.feature, failure_summary(failure, stage),
                       self.consequence,
                       f" ({suppressed} more like it since the last warning)" if suppressed else "",
                       exc_info=exc_info)
        return True


# ---------------------------------------------------------------------------
# The Jev provider
# ---------------------------------------------------------------------------

def sdk_question(question: Any) -> Any:
    """*question* as the SDK sends it: ``YesNo`` a ``Noul``, ``OneOf`` a ``Choice``, an SDK question as is.

    Byte for byte what the features built before the provider interface: the
    measured precision and recall hold for exactly these requests.
    """
    if isinstance(question, decision.YesNo):
        return Noul(instructions=question.instructions,
                    criteria={"true": question.true, "false": question.false})
    if isinstance(question, decision.OneOf):
        return Choice(instructions=question.instructions, criteria=dict(question.choices))
    return question


def _answer(raw: Any) -> decision.Answer:
    if hasattr(raw, "choice"):
        return raw.choice, {label: float(p) for label, p in (raw.probabilities or {}).items()}
    return float(raw.noul)


def _usage(usage: Any) -> Optional[dict[str, Any]]:
    if usage is None:
        return None
    if callable(getattr(usage, "model_dump", None)):
        return usage.model_dump()
    return {name: getattr(usage, name) for name in ("input_tokens", "output_tokens") if hasattr(usage, name)}


class JevProvider:
    """TypeSafe's Jev as a ``decision.DecisionProvider``, over an SDK client (``make_client``).

    Each ``ask`` is one ``bounded_call``. ``OutOfTime`` passes through; a state
    too long for the model (``state_too_large``) is ``decision.StateTooLarge``,
    every other failure -- ``CallTimedOut`` and ``VendorBusy`` included -- a
    ``decision.ProviderFailure``, each carrying ``describe_failure``'s fields.
    """

    name = JEV
    third_party = True

    def __init__(self, client: Any) -> None:
        self.client = client

    def ask(self, state: Any, questions: Any, *, cap: float, deadline: Optional[float] = None,
            budget: Optional["TurnBudget"] = None) -> decision.Answers:
        try:
            response = bounded_call(self.client, cap=cap, deadline=deadline, budget=budget, state=state,
                                    questions={key: sdk_question(q) for key, q in questions.items()})
            return decision.Answers({key: _answer(raw) for key, raw in response.answers.items()},
                                    usage=_usage(getattr(response, "usage", None)))
        except decision.OutOfTime:
            raise
        except Exception as error:  # noqa: BLE001 - every failure leaves as the neutral error
            failure = describe_failure(error)
            if state_too_large(error):
                raise decision.StateTooLarge(failure) from error
            raise decision.ProviderFailure(failure) from error


def as_provider(client_or_provider: Any) -> Any:
    """*client_or_provider* when it is a ``decision.DecisionProvider``, else a ``JevProvider`` over it."""
    if isinstance(client_or_provider, decision.DecisionProvider):
        return client_or_provider
    return JevProvider(client_or_provider)


class ClientCache:
    """Per-process, locked: one object per *slot*, rebuilt when the key's fingerprint changes."""

    def __init__(self) -> None:
        self._entries: dict[Hashable, tuple[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, slot: Hashable, key: str, build: Callable[[], Any]) -> Any:
        fingerprint = key_fingerprint(key)
        with self._lock:
            entry = self._entries.get(slot)
            if entry is None or entry[0] != fingerprint:
                entry = (fingerprint, build())
                self._entries[slot] = entry
            return entry[1]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
