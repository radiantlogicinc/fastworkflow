"""Optional routing of search_memory requests by a decision model.

Whether a search wants every row of a listing is a classification, so it is
asked of a decision model (TypeSafe's Jev by default) rather than the search model, and
only when a listing was found in the observation. Measured on 180
hand-labelled recorded questions (fix-xg1a): with the neutral wording below the
all-rows decision had precision 1.00 at every threshold tried and recall
0.95-0.98 at 0.5.

OFF unless a deployment turns it on: ``FW_SEARCH_ROUTER=jev`` AND a
``JEV_API_KEY``. A key present for some other purpose does not enable it,
because routing sends the agent's question, its reasoning and the first lines
of the observation to a third party. What is sent is passed through the
archive's credential scrub first (``jev_client.egress``), and if that fails
nothing is sent. Routing stays off (one warning) when
``FW_OFFLOAD_EVIDENCE_REDACTION=off``; a value ``egress`` refuses at call time
sends nothing and returns the error ``policy_withheld``.

Routing fails open: no SDK, no key, a timeout or any error means ``route``
returns an error record and the search model answers exactly as before. One
attempt, a short timeout, no retries -- routing is an optimisation in front of
a search and may never hold one up for long. A set ``FW_SEARCH_ROUTER`` that
cannot take effect (an unrecognised value, no SDK, no key, a rejected
``FW_JEV_BASE_URL``) logs one warning per cause; see ``jev_client``. A failure
is warned about at most once per five minutes per (error type, HTTP status),
counting the ones not logged; the error record carries ``error_status``,
``error_request_id``, ``error_code`` and ``error_stage`` ("redaction",
"request", or "budget" when the turn's vendor time ran out mid-call), and
lands on the ``search_memory`` event as its ``router`` field. A
``policy_withheld`` or ``router_budget`` record is not a failure and is not
warned about.

Each call is cut off after ``ROUTER_TIMEOUT_SECONDS`` of wall clock, or less
when the turn's vendor budget has less left (``jev_client.bounded_call``). A
turn routes at most ``jev_client.ROUTER_CALLS_PER_TURN`` searches; after that,
or once the turn's vendor time is spent, ``route`` returns the error
``router_budget`` without a call or a span, and the search model answers. Every
record carries ``vendor_ms``, the turn's vendor time so far (None when the
router runs without a turn budget), and ``provider``: the flag value that
selected who answers.

The questions go through the vendor-neutral ``decision.DecisionProvider``
interface; Jev is its built-in implementation (``jev_client.JevProvider``).
``FW_SEARCH_ROUTER`` may instead name a provider registered from code
(``decision.register_decision_provider``) -- never a module path -- under the
same redaction gate and per-value filter. ``ROUTE_ALL_ROWS_MIN``, the
wording and the precision and recall above were measured with Jev: with any
other provider they are unmeasured (one warning per process).

A workflow may add examples in its own vocabulary in
``<workflow>/search_router_examples.json``:
``{"all_rows": ["..."], "rows_about_named_item": ["..."]}``.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading import decision, jev_client
from fastworkflow.observation_offloading.decision import OneOf, YesNo
from fastworkflow.utils.logging import logger

ROUTER_ENV = "FW_SEARCH_ROUTER"
ROUTER_JEV = jev_client.JEV
ROUTER_KEY_ENV = jev_client.KEY_ENV
ROUTER_MODEL_ENV = "FW_SEARCH_ROUTER_MODEL"
DEFAULT_ROUTER_MODEL = jev_client.DEFAULT_MODEL
ROUTE_ALL_ROWS_MIN = 0.5
ROUTER_HEAD_LINES = 4
ROUTER_TIMEOUT_SECONDS = 2.0
EXAMPLES_FILE = "search_router_examples.json"
ALL_ROWS = "all_rows"
#: ``route``'s error when ``jev_client.egress`` refused a value it would have sent.
POLICY_WITHHELD = "policy_withheld"
#: ``route``'s error once the turn's routing calls or vendor time are used up.
ROUTER_BUDGET = "router_budget"

_INSTRUCTIONS = (
    "An agent asked a question about one earlier command output (the observation). "
    "A command output is often a list that belongs to one subject, such as the items "
    "of one person, one account or one record. What does the question ask to get back "
    "from that output?"
)
_CRITERIA = {
    ALL_ROWS: ("The whole list: every row or item it contains, e.g. 'list the items for "
               "Jane Doe', 'what entries are listed for this account', 'who are the "
               "members', 'the actions suggested for this record', 'the uids and labels "
               "for X'. Naming the subject the list belongs to (a person, an account, a "
               "record) still means all rows."),
    "rows_by_position": ("Only some rows chosen by position: the first one, the first N, "
                         "rows 11 to 20, the next few."),
    "rows_about_named_item": ("Only the row or rows for one particular item inside the "
                              "list, whether such an item is present, or the rows that "
                              "match a stated condition, e.g. 'is Jane Doe among them', "
                              "'does item X appear', 'entries whose label contains Y'. The "
                              "subject the whole list belongs to is not such an item."),
    "count": "Only how many rows or items there are.",
    "other": "A field value, a property record, a score or a judgement: not rows of a list.",
}
_FOR_REPORT = YesNo(
    instructions=("Is the agent gathering this only so it can include it in its final "
                  "answer or report, rather than to decide its next action (which item "
                  "to open, which command to run)?"),
    true="Only for the final answer or report.",
    false="To decide or perform a next action, or not stated.",
)


def _questions(examples: dict[str, list[str]]) -> dict[str, Any]:
    criteria = dict(_CRITERIA)
    for choice, extra in examples.items():
        if choice in criteria and extra:
            criteria[choice] += " Also, in this workflow: " + "; ".join(
                f"'{e}'" for e in extra[:8]) + "."
    return {"wants": OneOf(instructions=_INSTRUCTIONS, choices=criteria),
            "for_report": _FOR_REPORT}


def _load_examples(workflow_path: str) -> dict[str, list[str]]:
    path = Path(workflow_path) / EXAMPLES_FILE if workflow_path else None
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: [str(e) for e in v] for k, v in data.items() if isinstance(v, list)}
    except (OSError, ValueError) as error:
        logger.warning("ignoring %s: %s", path, type(error).__name__)
        return {}


class SearchRouter:
    """One configured routing client. ``route`` never raises.

    *client* is the ``decision.DecisionProvider`` that answers, or a Jev SDK
    client (``jev_client.make_client``), which is wrapped in a
    ``jev_client.JevProvider``. *provider_name* (default: the provider's
    ``name``) is recorded on every route record as ``provider``.
    """

    def __init__(self, client: Any, questions: dict[str, Any], model: str = "", *,
                 call_seconds: float = ROUTER_TIMEOUT_SECONDS,
                 warner: Optional[jev_client.FailureWarner] = None,
                 provider_name: Optional[str] = None) -> None:
        self._provider = jev_client.as_provider(client)
        self._provider_name = provider_name or str(getattr(self._provider, "name", "") or "")
        self._questions = questions
        self._model = model
        self._call_seconds = call_seconds
        self._warner = warner or jev_client.FailureWarner("search router", "searches use the model path")
        self._budget_source: Optional[Callable[[], Optional[jev_client.TurnBudget]]] = None

    def within_budget(self, budget_source: Callable[[], Optional[jev_client.TurnBudget]]) -> "SearchRouter":
        """This router, drawing on the turn budget *budget_source* returns at each ``route``.

        A copy: the configured router is shared per process, the budget is one agent's turn.
        """
        bound = copy.copy(self)
        bound._budget_source = budget_source
        return bound

    def route(self, question: str, reasoning: str, text: str,
              host: Any = None, budget: Optional[jev_client.TurnBudget] = None) -> dict[str, Any]:
        """``{"choice", "p_all_rows", "for_report", "latency_ms", "vendor_ms"}`` or ``{"choice": None, "error"}``.

        With a trace *host*, the call is also an ``fw.search.route`` span under
        the current agent step; without one, or with no active sink, it is not.
        *budget* (default: this router's budget source, if any) is the turn's;
        once it refuses a routing call nothing is sent and no span is started.
        """
        if budget is None and self._budget_source is not None:
            budget = self._budget_source()
        started = time.monotonic()
        state, refused = self._state(question, reasoning, text, started)
        # A route that sends nothing (refused by ``egress``, or failed
        # redaction) does not use up one of the turn's routing calls.
        if refused is None and budget is not None and not budget.take_router_call():
            return self._with_in_flight(
                {"choice": None, "error": ROUTER_BUDGET, "error_status": None,
                 "error_request_id": None, "error_code": None, "error_stage": "budget",
                 "latency_ms": 0, "vendor_ms": budget.vendor_ms, "provider": self._provider_name})
        span = tracing.start_span(host, tracing.SPAN_SEARCH_ROUTE, kind=tracing.KIND_LLM,
                                  attributes={"model": self._model}) if host is not None else None
        result = refused if refused is not None else self._decide(state, budget, started)
        result["vendor_ms"] = budget.vendor_ms if budget is not None else None
        result["provider"] = self._provider_name
        self._with_in_flight(result)
        usage = result.get("usage") or {}
        tracing.end_span(
            host, span,
            status=tracing.STATUS_ERROR if result.get("error") else tracing.STATUS_OK,
            attributes={"choice": result.get("choice"),
                        "p_all_rows": result.get("p_all_rows"),
                        "for_report": result.get("for_report"),
                        "latency_ms": result.get("latency_ms"),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "error_type": result.get("error")},
        )
        return result

    @staticmethod
    def _with_in_flight(record: dict[str, Any]) -> dict[str, Any]:
        """*record*, with ``vendor_calls_in_flight`` when any vendor worker slot is held and
        ``vendor_calls_orphaned`` when any abandoned request still runs (``jev_client.vendor_pressure``)."""
        record.update(jev_client.vendor_pressure())
        return record

    def _failed(self, error: Exception, stage: str, started: float) -> dict[str, Any]:
        if isinstance(error, decision.OutOfTime):
            stage = "budget"
        failure = decision.describe(error)
        self._warner.warn(failure, stage=stage)
        return {"choice": None, "error": failure["error_type"],
                "error_status": failure["status"], "error_request_id": failure["request_id"],
                "error_code": failure["code"], "error_stage": stage,
                "latency_ms": round((time.monotonic() - started) * 1000)}

    def _state(self, question: str, reasoning: str, text: str,
               started: float) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """``(state to send, None)``, or ``(None, error record)`` when nothing may be sent."""
        stage = "redaction"
        try:
            state = {"question": jev_client.egress(question),
                     "agent_reasoning": jev_client.egress(reasoning),
                     "observation_start": jev_client.egress(
                         "\n".join(text.splitlines()[:ROUTER_HEAD_LINES]))}
        except Exception as error:  # noqa: BLE001 - routing must never stop a search
            return None, self._failed(error, stage, started)
        if any(value is None for value in state.values()):
            return None, {"choice": None, "error": POLICY_WITHHELD, "error_status": None,
                          "error_request_id": None, "error_code": None, "error_stage": stage,
                          "latency_ms": round((time.monotonic() - started) * 1000)}
        return state, None

    def _decide(self, state: dict[str, Any], budget: Optional[jev_client.TurnBudget],
                started: float) -> dict[str, Any]:
        stage = "request"
        try:
            answers = self._provider.ask(state, self._questions, cap=self._call_seconds, budget=budget)
            choice, probabilities = answers["wants"]
            return {"choice": choice,
                    "p_all_rows": float(probabilities.get(ALL_ROWS, 0.0)),
                    "for_report": float(answers["for_report"]),
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "usage": getattr(answers, "usage", None)}
        except Exception as error:  # noqa: BLE001 - routing must never stop a search
            return self._failed(error, stage, started)

    def wants_all_rows(self, route: Optional[dict[str, Any]]) -> bool:
        return bool(route and route.get("choice") == ALL_ROWS
                    and route.get("p_all_rows", 0.0) >= ROUTE_ALL_ROWS_MIN)


_ROUTERS = jev_client.ClientCache()
#: The cache slot of a router answered by a registered provider, keyed by its registration.
REGISTERED_SLOT = "registered"


def router_for_workflow(workflow_path: str = "") -> Optional[SearchRouter]:
    """The workflow's router, built once per workflow path, model, endpoint and key; None when routing is off.

    A set ``FW_SEARCH_ROUTER`` that cannot take effect warns once per cause
    (``jev_client.requested_key``, ``jev_client.base_url``). A value naming a
    provider registered from code (``decision.register_decision_provider``)
    selects it instead of Jev, built once per workflow path and registration;
    ``ROUTE_ALL_ROWS_MIN`` and the wording were calibrated with Jev, so
    selecting one warns once.
    """
    entry = decision.registration(jev_client.flag_value(ROUTER_ENV))
    if entry is not None:
        provider = jev_client.registered_provider(ROUTER_ENV, entry, "search routing")
        if provider is None:
            return None
        return _ROUTERS.get((workflow_path, REGISTERED_SLOT, entry.name), f"{entry.name}#{entry.generation}",
                            lambda: SearchRouter(provider, _questions(_load_examples(workflow_path)),
                                                 model=str(getattr(provider, "model", "") or ""),
                                                 provider_name=entry.name))
    key = jev_client.requested_key(ROUTER_ENV, "search routing")
    if key is None:
        return None
    url = jev_client.base_url()
    if url is None:
        return None
    model = context_budget.env_value(ROUTER_MODEL_ENV) or DEFAULT_ROUTER_MODEL
    return _ROUTERS.get((workflow_path, model, url), key, lambda: SearchRouter(
        jev_client.make_client(key, model, ROUTER_TIMEOUT_SECONDS + jev_client.CUTOFF_MARGIN_SECONDS, url),
        _questions(_load_examples(workflow_path)), model=model))
