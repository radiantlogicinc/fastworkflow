"""The decision-provider seam: what a decision-model feature asks, independent of who answers.

The finish-time execution check (``finish_check``) and search routing
(``search_router``) ask classification questions about a state -- yes/no
(``YesNo``) or one of several named choices (``OneOf``) -- of a
``DecisionProvider``. The built-in provider is TypeSafe's Jev
(``jev_client.JevProvider``, flag value ``jev``); a deployment may register
others from code with ``register_decision_provider(name, factory)`` and select
one by setting a feature's flag (``FW_FINISH_CHECK``, ``FW_SEARCH_ROUTER``) to
its name. A module path is never read from the environment: only code already
running in the process can add a provider.

A provider's ``ask(state, questions, *, cap, deadline, budget)`` answers every
question it was given -- a ``YesNo`` with the probability of yes, a ``OneOf``
with ``(choice, {label: probability})`` -- or raises. It must stop within
``min(cap, deadline - time.monotonic(), budget.remaining)`` seconds, charge the
time it waited to *budget* (``budget.spend``), and raise ``OutOfTime`` when the
deadline or the budget ran out. A state too long for its model is
``StateTooLarge`` (the caller may halve it and ask again); any other failure is
a ``ProviderFailure`` carrying ``describe(error)``'s fields: error type, HTTP
status, request id and machine-readable code -- never a message, which may echo
what was sent.

Everything a feature sends passes the archive's credential scrub first, and the
features' redaction gate applies to every provider today. ``third_party``
says whether what is sent leaves the deployment, so that gate may later be
relaxed for a self-hosted provider.

The features' thresholds (``finish_check.FLAG_MIN`` and ``ASK_MIN``,
``search_router.ROUTE_ALL_ROWS_MIN``), their questions' wording and their
published precision and recall were calibrated with Jev; with another provider
they are unmeasured.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Union, runtime_checkable

#: Flag values (stripped, any case) that say "off"; never a provider's name.
OFF_VALUES = frozenset({"", "off", "0", "false", "no", "none"})
#: The built-in provider's flag value, which a registration may not take.
BUILTIN_NAMES = frozenset({"jev"})
_NAME_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class YesNo:
    """A yes/no question; *true* and *false* describe the two outcomes."""

    instructions: str
    true: str
    false: str


@dataclass(frozen=True)
class OneOf:
    """A question picking one of *choices*: label -> description, in the order asked."""

    instructions: str
    choices: Mapping[str, str]


Question = Union[YesNo, OneOf]
#: A ``YesNo``'s probability of yes, or a ``OneOf``'s ``(choice, {label: probability})``.
Answer = Union[float, tuple[str, dict[str, float]]]


class Answers(dict):
    """``{key: Answer}`` for every question asked; ``usage`` is the provider's token counts, if it reports any."""

    def __init__(self, answers: Mapping[str, Answer], usage: Optional[dict[str, Any]] = None) -> None:
        super().__init__(answers)
        self.usage = usage


class OutOfTime(Exception):
    """The time left (the caller's deadline or the turn's budget) ran out; the call was abandoned or never sent."""


class ProviderFailure(Exception):
    """A decision call failed; ``failure`` is ``{error_type, status, request_id, code}`` of the underlying error."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = {"error_type": failure.get("error_type") or "ProviderFailure",
                        "status": failure.get("status"), "request_id": failure.get("request_id"),
                        "code": failure.get("code")}
        super().__init__(self.failure["error_type"])


class StateTooLarge(ProviderFailure):
    """The provider refused the state as too long for its model; a smaller state may be accepted."""


def describe(error: BaseException) -> dict[str, Any]:
    """``{error_type, status, request_id, code}`` for a failed call: a ``ProviderFailure``'s own, else the type alone."""
    if isinstance(error, ProviderFailure):
        return dict(error.failure)
    return {"error_type": type(error).__name__, "status": None, "request_id": None, "code": None}


@runtime_checkable
class DecisionProvider(Protocol):
    """Answers ``YesNo`` and ``OneOf`` questions about a JSON-serialisable state."""

    #: Whether what is sent leaves the deployment (a vendor), rather than a self-hosted model.
    third_party: bool

    def ask(self, state: Any, questions: Mapping[str, Question], *, cap: float,
            deadline: Optional[float] = None, budget: Any = None) -> Mapping[str, Answer]:
        ...


# ---------------------------------------------------------------------------
# Registered providers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Registration:
    """One registered provider; *generation* changes whenever the name is registered again."""

    name: str
    factory: Callable[[], DecisionProvider]
    generation: int


_REGISTRY: dict[str, Registration] = {}
_BUILT: dict[str, tuple[int, DecisionProvider]] = {}
_GENERATION = 0
_REGISTRY_LOCK = threading.Lock()


def _normalised(name: Any) -> str:
    return str(name or "").strip().lower()


def register_decision_provider(name: str, factory: Callable[[], DecisionProvider]) -> None:
    """Make *name* a flag value selecting the provider *factory* builds.

    Call from code, before the agent is built. *factory* takes no arguments and
    is called once, the first time a feature asks for the provider (again after
    the name is re-registered; two features asking at the same moment may each
    call it, see ``build_provider``); the provider it returns may serve several
    features at once. *name* is matched case-insensitively and may be neither an
    off value nor ``jev``.
    """
    global _GENERATION
    key = _normalised(name)
    if key in OFF_VALUES or key in BUILTIN_NAMES or not _NAME_RE.fullmatch(key):
        raise ValueError(f"{name!r} cannot name a decision provider")
    if not callable(factory):
        raise TypeError("factory must be callable")
    with _REGISTRY_LOCK:
        _GENERATION += 1
        _REGISTRY[key] = Registration(key, factory, _GENERATION)
        _BUILT.pop(key, None)


def unregister_decision_provider(name: str) -> None:
    """Forget *name*; a flag set to it is then an unrecognised value again."""
    key = _normalised(name)
    with _REGISTRY_LOCK:
        _REGISTRY.pop(key, None)
        _BUILT.pop(key, None)


def registration(name: Any) -> Optional[Registration]:
    """The registration of flag value *name*, or None when no provider is registered under it."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(_normalised(name))


def build_provider(entry: Registration) -> DecisionProvider:
    """*entry*'s provider, built by its factory once per registration.

    Raises whatever the factory raises, and ``TypeError`` when what it returned
    is not a ``DecisionProvider``; a failed build is retried on the next call.
    The factory runs outside the registry lock, so it may itself call
    ``registration`` or ``register_decision_provider``; two features asking at
    once may each run it, and the first provider stored is the one both get.
    """
    with _REGISTRY_LOCK:
        built = _BUILT.get(entry.name)
        if built is not None and built[0] == entry.generation:
            return built[1]
    provider = entry.factory()
    if not isinstance(provider, DecisionProvider):
        raise TypeError(f"the factory registered as {entry.name!r} did not return a DecisionProvider")
    with _REGISTRY_LOCK:
        built = _BUILT.get(entry.name)
        if built is not None and built[0] == entry.generation:
            return built[1]
        if _REGISTRY.get(entry.name) == entry:
            _BUILT[entry.name] = (entry.generation, provider)
    return provider
