"""Optional routing of search_memory requests by a decision model.

Whether a search wants every row of a listing is a classification, so it is
asked of a decision model (TypeSafe's Jev) rather than the search model, and
only when a listing was found in the observation. Measured on 180
hand-labelled recorded questions (fix-xg1a): with the neutral wording below the
all-rows decision had precision 1.00 at every threshold tried and recall
0.95-0.98 at 0.5.

OFF unless a deployment turns it on: ``FW_SEARCH_ROUTER=jev`` AND a
``JEV_API_KEY``. A key present for some other purpose does not enable it,
because routing sends the agent's question, its reasoning and the first lines
of the observation to a third party. What is sent is passed through the
archive's capture policy first (``capture_record_for``), and if that fails
nothing is sent.

Routing fails open: no SDK, no key, a timeout or any error means ``route``
returns an error record and the search model answers exactly as before. One
attempt, a short timeout, no retries -- routing is an optimisation in front of
a search and may never hold one up for long.

A workflow may add examples in its own vocabulary in
``<workflow>/search_router_examples.json``:
``{"all_rows": ["..."], "rows_about_named_item": ["..."]}``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient
except ImportError:  # optional dependency: without it there is no routing
    Choice = Noul = RetryPolicy = TypeSafeClient = None

from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading.archive import capture_record_for
from fastworkflow.utils.logging import logger

ROUTER_ENV = "FW_SEARCH_ROUTER"
ROUTER_JEV = "jev"
ROUTER_KEY_ENV = "JEV_API_KEY"
ROUTER_MODEL_ENV = "FW_SEARCH_ROUTER_MODEL"
DEFAULT_ROUTER_MODEL = "jev-1.13.0"
ROUTE_ALL_ROWS_MIN = 0.5
ROUTER_HEAD_LINES = 4
ROUTER_TIMEOUT_SECONDS = 2.0
EXAMPLES_FILE = "search_router_examples.json"
ALL_ROWS = "all_rows"

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
_FOR_REPORT = {
    "instructions": ("Is the agent gathering this only so it can include it in its final "
                     "answer or report, rather than to decide its next action (which item "
                     "to open, which command to run)?"),
    "criteria": {"true": "Only for the final answer or report.",
                 "false": "To decide or perform a next action, or not stated."},
}


def _questions(examples: dict[str, list[str]]) -> dict[str, Any]:
    criteria = dict(_CRITERIA)
    for choice, extra in examples.items():
        if choice in criteria and extra:
            criteria[choice] += " Also, in this workflow: " + "; ".join(
                f"'{e}'" for e in extra[:8]) + "."
    return {"wants": Choice(instructions=_INSTRUCTIONS, criteria=criteria),
            "for_report": Noul(**_FOR_REPORT)}


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


def _redacted(text: str) -> str:
    stored, _record = capture_record_for(text)
    return stored


class SearchRouter:
    """One configured routing client. ``route`` never raises."""

    def __init__(self, client: Any, questions: dict[str, Any], model: str = "") -> None:
        self._client = client
        self._questions = questions
        self._model = model
        self._warned: set[str] = set()

    def route(self, question: str, reasoning: str, text: str,
              host: Any = None) -> dict[str, Any]:
        """``{"choice", "p_all_rows", "for_report", "latency_ms"}`` or ``{"choice": None, "error"}``.

        With a trace *host*, the call is also an ``fw.search.route`` span under
        the current agent step; without one, or with no active sink, it is not.
        """
        span = tracing.start_span(host, tracing.SPAN_SEARCH_ROUTE, kind=tracing.KIND_LLM,
                                  attributes={"model": self._model}) if host is not None else None
        result = self._decide(question, reasoning, text)
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

    def _decide(self, question: str, reasoning: str, text: str) -> dict[str, Any]:
        started = time.monotonic()
        try:
            state = {"question": _redacted(question),
                     "agent_reasoning": _redacted(reasoning),
                     "observation_start": _redacted(
                         "\n".join(text.splitlines()[:ROUTER_HEAD_LINES]))}
            response = self._client.system_one(state=state, questions=self._questions)
            wants = response.answers["wants"]
            usage = getattr(response, "usage", None)
            return {"choice": wants.choice,
                    "p_all_rows": float(wants.probabilities.get(ALL_ROWS, 0.0)),
                    "for_report": float(response.answers["for_report"].noul),
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "usage": usage.model_dump() if usage is not None else None}
        except Exception as error:  # noqa: BLE001 - routing must never stop a search
            name = type(error).__name__
            if name not in self._warned:
                self._warned.add(name)
                logger.warning("search router unavailable (%s); searches use the model path", name)
            return {"choice": None, "error": name,
                    "latency_ms": round((time.monotonic() - started) * 1000)}

    def wants_all_rows(self, route: Optional[dict[str, Any]]) -> bool:
        return bool(route and route.get("choice") == ALL_ROWS
                    and route.get("p_all_rows", 0.0) >= ROUTE_ALL_ROWS_MIN)


_ROUTERS: dict[str, Optional[SearchRouter]] = {}
_ROUTERS_LOCK = threading.Lock()


def router_for_workflow(workflow_path: str = "") -> Optional[SearchRouter]:
    """The workflow's router, built once per workflow path; None when routing is off."""
    if TypeSafeClient is None:
        return None
    if (context_budget.env_value(ROUTER_ENV) or "").strip().lower() != ROUTER_JEV:
        return None
    key = context_budget.env_value(ROUTER_KEY_ENV)
    if not key:
        return None
    with _ROUTERS_LOCK:
        if workflow_path not in _ROUTERS:
            model = context_budget.env_value(ROUTER_MODEL_ENV) or DEFAULT_ROUTER_MODEL
            client = TypeSafeClient(
                api_key=key,
                model=model,
                timeout=ROUTER_TIMEOUT_SECONDS,
                retry=RetryPolicy(max_retries=0, timeout=ROUTER_TIMEOUT_SECONDS))
            _ROUTERS[workflow_path] = SearchRouter(
                client, _questions(_load_examples(workflow_path)), model=model)
        return _ROUTERS[workflow_path]
