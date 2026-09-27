"""Finish-time execution check (fix-4dsr): was every step of the turn's plan executed?

When the agent chooses ``finish``, a decision model (TypeSafe's Jev) is asked, for
every step of the turn's initial plan and every subject the request names,
whether the turn's record shows the step carried out for that subject. Steps it
judges unexecuted are named in one note that replaces the finish observation, and
the agent goes back to the loop -- once per turn, only with iterations left, and
the agent may finish anyway. It checks EXECUTION, not whether the request was
answered: a step whose command ran counts, whatever it returned.

Measured offline on 82 recorded ido attempts none of it was tuned on (910
hand-labelled step x subject pairs, pre-registered): precision 0.77, recall 0.89,
F1 0.82, against 0.75 for the plain one-question-per-pair check. Every
unexecuted subject-less step (9 of 129) was flagged and nothing else was.

OFF unless a deployment turns it on: ``FW_FINISH_CHECK=jev`` AND a
``JEV_API_KEY``. A key present for another purpose does not enable it, because
the check sends the request, the plan and a summary of every step (command,
context, the first bytes of its output) to a third party. All of it passes the
archive's capture policy (``capture_record_for``) first.

It fails open: no SDK, no key, no plan, a timeout, an error or an exhausted time
budget all mean no note, and the turn finishes exactly as it would without the
check. It never raises.

What the model sees, and why:

* the plan state -- the request and each step's text -- for "does step k need a
  command?", "does step k (or its part j) concern subject S?";
* the ledger state -- one row per step: the command, the context it ran in, the
  context it left the agent in (``acted_on``), every identifier in them resolved
  to the label a retrieved listing gave it (``refers_to``), the request's
  subjects found anywhere in its full output (``names_in_output``), whether it
  errored or came back empty, and the first bytes of the output -- for "was step
  k (part j) executed for S?" and "was step k executed at all?".

A step's parts are checked separately so a partly executed step is not scored
as executed. Optional steps and steps that wait on the user are not checked.
Question batches are chunked; a ledger too large for one request is halved
until it fits, and a step counts as executed if any half shows it.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient
except ImportError:  # optional dependency: without it there is no check
    Noul = RetryPolicy = TypeSafeClient = None

from fastworkflow import context_budget, tracing
from fastworkflow.observation_offloading.archive import capture_record_for
from fastworkflow.observation_offloading.compact import EXECUTE_TOOL_NAME, step_indexes
from fastworkflow.observation_offloading.labels import command_response
from fastworkflow.observation_offloading.state import context_clause_of, record_event
from fastworkflow.turn_plan import TurnPlan, command_parts, is_checked, step_text
from fastworkflow.utils.logging import logger

CHECK_ENV = "FW_FINISH_CHECK"
CHECK_JEV = "jev"
KEY_ENV = "JEV_API_KEY"
MODEL_ENV = "FW_FINISH_CHECK_MODEL"
DEFAULT_MODEL = "jev-1.13.0"

#: A pair or step is named in the note when its unmet score reaches this.
FLAG_MIN = 0.5
#: Execution questions are only asked where scope and applicability reach this.
ASK_MIN = 0.3
QUESTIONS_PER_REQUEST = 100
#: One call's timeout, and the whole check's budget. Measured: two sequential
#: calls, median 0.53 s, p90 0.62 s, max 0.72 s over 118 recorded turns.
CALL_TIMEOUT_SECONDS = 4.0
CHECK_BUDGET_SECONDS = 8.0
HEAD_BYTES = 200
NOTE_MAX_BYTES = 1024
#: The note costs an iteration and is worthless unless the agent can act on it.
MIN_ITERS_LEFT = 2

_ROW_RE = re.compile(r"^\s*([A-Za-z0-9_]{16,})\s{2,}(\S.*?)\s*$")
_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{16,}")
_EMPTY_RE = re.compile(r"^\s*(0 \w|no \w|none\b|not found)", re.IGNORECASE)

_ROW_RULE = (
    'A row is for "{name}" if its command, context or acted_on names "{name}", or if refers_to '
    'resolves one of its identifiers to "{name}" or to what the request calls "{name}" (a label '
    'may be worded differently). names_in_output only says "{name}" appears somewhere in the '
    'row\'s output listing: that alone does not make the row about "{name}", but it can tie a '
    'finding, listing or context to "{name}".'
)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def _redacted(text: str) -> str:
    stored, _record = capture_record_for(text)
    return stored


def _outcome(response: str) -> str:
    if response.startswith("Execution error"):
        return "error"
    if not response.strip() or _EMPTY_RE.match(response):
        return "empty"
    return "result"


def _head(text: str) -> str:
    return text.encode("utf-8")[:HEAD_BYTES].decode("utf-8", "ignore")


def build_ledger(agent: Any, subject_names: list[str]) -> list[dict[str, Any]]:
    """One row per non-finish step of the turn's full trajectory, redacted for sending."""
    trajectory = getattr(agent, "current_trajectory", None) or {}
    scope = getattr(agent, "continuation_scope", None)
    store = getattr(agent, "observation_archive", None)
    pairs_of = getattr(agent, "execute_ordinal_pairs", None)
    alias_of = {index: f"O{ordinal}" for index, ordinal in (pairs_of(trajectory) if callable(pairs_of) else [])}
    archived: dict[str, str] = {}
    if scope is not None and store is not None:
        try:
            archived = {str(h["alias"]): str(h.get("text") or "") for h in store.list(scope)}
        except Exception:  # noqa: BLE001 - an unreadable archive leaves the inline text
            archived = {}

    steps = []
    for index in step_indexes(trajectory):
        tool = str(trajectory.get(f"tool_name_{index}") or "")
        if tool == "finish":
            continue
        args = trajectory.get(f"tool_args_{index}") or {}
        observation = str(trajectory.get(f"observation_{index}") or "")
        alias = alias_of.get(index)
        response = archived.get(alias) if alias in archived else (
            command_response(observation, alias) if alias else observation)
        clause = ""
        if alias and scope is not None:
            clause = context_clause_of(scope, alias, selected_archive=store) or ""
        command = (str(args.get("command") or "") if tool == EXECUTE_TOOL_NAME and isinstance(args, dict)
                   else json.dumps(args, ensure_ascii=False, default=str)[:HEAD_BYTES])
        steps.append({"tool": tool, "command": command or tool, "context": clause,
                      "response": response, "alias": alias})

    labels: dict[str, dict[str, str]] = {}
    for step in steps:
        for line in step["response"].splitlines():
            match = _ROW_RE.match(line)
            if match and match.group(1) not in labels:
                labels[match.group(1)] = {"label": match.group(2)[:80], "listed_by": step["command"][:90]}

    rows = []
    for position, step in enumerate(steps):
        acted_on = ""
        if step["tool"] == EXECUTE_TOOL_NAME:
            acted_on = next((later["context"] for later in steps[position + 1:] if later["context"]), "")
        refers_to = {}
        for token in dict.fromkeys(_ID_TOKEN_RE.findall(" ".join([step["command"], step["context"], acted_on]))):
            if token in labels:
                listed_by = labels[token]["listed_by"]
                for inner in _ID_TOKEN_RE.findall(listed_by):
                    if inner in labels and inner != token:
                        listed_by += f" [{inner} = {labels[inner]['label']}]"
                refers_to[token] = _redacted(f"{labels[token]['label']} (listed by {listed_by})")
        rows.append({
            "n": position + 1,
            "command": _redacted(step["command"]),
            "context": _redacted(step["context"]),
            "acted_on": _redacted(acted_on),
            "refers_to": refers_to,
            "names_in_output": [name for name in subject_names if name and name.casefold() in step["response"].casefold()],
            "outcome": _outcome(step["response"]),
            "head": _redacted(_head(step["response"])),
        })
    return rows


# ---------------------------------------------------------------------------
# The questions (wording as measured; changing it invalidates the measurement)
# ---------------------------------------------------------------------------

def _action_question(k: int) -> Any:
    return Noul(
        instructions=(f"Does plan step {k} require running a workflow command to retrieve or change something? "
                      f"Answer no if step {k} only reasons over results already obtained (compare, intersect, "
                      f"summarise, report) or asks the user."),
        criteria={"true": f"Step {k} runs at least one command.",
                  "false": f"Step {k} runs no command: it reasons, reports or asks the user."})


def _applies_question(k: int, name: str, kind: str) -> Any:
    return Noul(
        instructions=(f'Does plan step {k} call for work on the {kind} "{name}"? Answer yes if step {k} names '
                      f'"{name}", or if step {k} applies to every {kind} named in the request and "{name}" is one of them.'),
        criteria={"true": f'Step {k} requires an action about "{name}".',
                  "false": f'Step {k} is about other things, not "{name}".'})


def _part_applies_question(k: int, j: int, part: str, name: str, kind: str) -> Any:
    return Noul(
        instructions=(f'Part {j} of plan step {k}: "{part}"\nDoes this part call for work on the {kind} "{name}"? '
                      f'Answer yes if it names "{name}", or if it applies to every {kind} it is about and "{name}" is one of them.'),
        criteria={"true": f'This part requires an action about "{name}".',
                  "false": f'This part is about other things, not "{name}".'})


def _exec_question(step: str, part: Optional[str], name: str) -> Any:
    what = f'Plan step: "{step}"' + (f'\nPart of that step: "{part}"' if part else "")
    target = "this part" if part else "this step"
    return Noul(
        instructions=(f"{what}\nWas {target} executed for \"{name}\"? Yes if some ledger row ran the command "
                      f"{target} names, or another command doing the same kind of action, for \"{name}\", "
                      f"and the row is not an execution error. What the command returned does not matter. "
                      + _ROW_RULE.format(name=name)),
        criteria={"true": f'Some row ran {target}\'s action for "{name}" without an execution error.',
                  "false": f'No row ran {target}\'s action for "{name}", or every such row was an execution error.'})


def _any_question(k: int, step: str) -> Any:
    return Noul(
        instructions=(f'Plan step {k}: "{step}"\nWas this step executed at all? Yes if some ledger row ran '
                      f"the command it names, or another command doing the same kind of action, and the row "
                      f"is not an execution error. What the command returned does not matter."),
        criteria={"true": "Some row ran this step's action without an execution error.",
                  "false": "No row ran this step's action, or every such row was an execution error."})


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

class _OutOfTime(Exception):
    """The whole check's budget ran out before its questions were answered."""


@dataclass
class CheckResult:
    """What one check found. ``error`` set means it produced no verdicts."""

    flagged: list[dict[str, Any]] = field(default_factory=list)
    questions: int = 0
    requests: int = 0
    splits: int = 0
    input_tokens: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


class FinishChecker:
    """One configured check. ``note`` and ``check`` never raise."""

    def __init__(self, client: Any, model: str = "", *, budget_seconds: float = CHECK_BUDGET_SECONDS) -> None:
        self._client = client
        self._model = model
        self._budget_seconds = budget_seconds
        self._warned: set[str] = set()

    # -- one decision-model request, chunked and halved -----------------------

    def _call(self, state: Any, questions: dict[str, Any], result: CheckResult, deadline: float) -> dict[str, float]:
        if time.monotonic() > deadline:
            raise _OutOfTime()
        response = self._client.system_one(state=state, questions=questions)
        result.requests += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            result.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        return {key: float(response.answers[key].noul) for key in questions}

    def _ask(self, rows: list[Any], questions: dict[str, Any], wrap: Callable[[list[Any]], Any],
             result: CheckResult, deadline: float) -> dict[str, float]:
        items = list(questions.items())
        answers: dict[str, float] = {}
        for start in range(0, len(items), QUESTIONS_PER_REQUEST):
            chunk = dict(items[start:start + QUESTIONS_PER_REQUEST])
            try:
                answers.update(self._call(wrap(rows), chunk, result, deadline))
            except _OutOfTime:
                raise
            except Exception as error:  # noqa: BLE001 - only an over-long state is split
                if "max_tokens_exceeded" not in str(error) or len(rows) < 2:
                    raise
                result.splits += 1
                middle = len(rows) // 2
                left = self._ask(rows[:middle], chunk, wrap, result, deadline)
                right = self._ask(rows[middle:], chunk, wrap, result, deadline)
                answers.update({key: max(left[key], right[key]) for key in chunk})
        return answers

    # -- the verdicts ---------------------------------------------------------

    def check(self, plan: TurnPlan, request: str, ledger: list[dict[str, Any]]) -> CheckResult:
        started = time.monotonic()
        deadline = started + self._budget_seconds
        result = CheckResult()
        try:
            self._check(plan, request, ledger, result, deadline)
        except Exception as error:  # noqa: BLE001 - the check must never stop a turn
            name = "OutOfTime" if isinstance(error, _OutOfTime) else type(error).__name__
            if name not in self._warned:
                self._warned.add(name)
                logger.warning("finish check unavailable (%s); the turn finishes unchecked", name)
            result.flagged = []
            result.error = name
        result.latency_ms = round((time.monotonic() - started) * 1000)
        return result

    def _check(self, plan: TurnPlan, request: str, ledger: list[dict[str, Any]],
               result: CheckResult, deadline: float) -> None:
        steps = {k: step for k, step in enumerate(plan.steps, 1) if is_checked(step)}
        subjects = [(s.name, (s.kind or "item").strip() or "item") for s in plan.subjects if s.name.strip()]
        if not steps:
            return
        plan_state = {"request": _redacted(request),
                      "plan_steps": {str(k): _redacted(step_text(step)) for k, step in enumerate(plan.steps, 1)}}
        parts = {k: command_parts(step) for k, step in steps.items()}

        scope_questions: dict[str, Any] = {}
        for k in steps:
            scope_questions[f"x{k}"] = _action_question(k)
            for si, (name, kind) in enumerate(subjects):
                scope_questions[f"a{k}_{si}"] = _applies_question(k, name, kind)
                for j, part in enumerate(parts[k], 1):
                    scope_questions[f"p{k}_{j}_{si}"] = _part_applies_question(k, j, _redacted(part.text), name, kind)
        result.questions += len(scope_questions)
        scope = self._ask([plan_state], scope_questions, lambda rows: rows[0], result, deadline)

        exec_questions: dict[str, Any] = {}
        for k, step in steps.items():
            if scope[f"x{k}"] < ASK_MIN:
                continue
            text = _redacted(step_text(step))
            exec_questions[f"g{k}"] = _any_question(k, text)
            for si, (name, _kind) in enumerate(subjects):
                if scope[f"a{k}_{si}"] < ASK_MIN:
                    continue
                if not parts[k]:
                    exec_questions[f"e{k}_{si}"] = _exec_question(text, None, name)
                    continue
                for j, part in enumerate(parts[k], 1):
                    if scope[f"p{k}_{j}_{si}"] >= ASK_MIN:
                        exec_questions[f"e{k}_{j}_{si}"] = _exec_question(text, _redacted(part.text), name)
        result.questions += len(exec_questions)
        executed = (self._ask(ledger, exec_questions, lambda rows: {"ledger": rows}, result, deadline)
                    if exec_questions else {})

        for k, step in steps.items():
            x = scope[f"x{k}"]
            best_applies = max((scope[f"a{k}_{si}"] for si in range(len(subjects))), default=0.0)
            for si, (name, _kind) in enumerate(subjects):
                applies = x * scope[f"a{k}_{si}"]
                if parts[k]:
                    worst = max((scope[f"p{k}_{j}_{si}"] * (1 - executed[f"e{k}_{j}_{si}"])
                                 for j in range(1, len(parts[k]) + 1) if f"e{k}_{j}_{si}" in executed), default=0.0)
                    unmet = applies * worst
                else:
                    unmet = applies * (1 - executed[f"e{k}_{si}"]) if f"e{k}_{si}" in executed else 0.0
                if unmet >= FLAG_MIN:
                    result.flagged.append({"step": k, "subject": name, "text": step.text, "p_unmet": round(unmet, 3)})
            if f"g{k}" in executed:
                unmet = x * (1 - best_applies) * (1 - executed[f"g{k}"])
                if unmet >= FLAG_MIN:
                    result.flagged.append({"step": k, "subject": None, "text": step.text, "p_unmet": round(unmet, 3)})

    # -- what the agent is told -----------------------------------------------

    def note(self, agent: Any, input_args: dict[str, Any], *, iterations_left: int) -> str:
        """The note for this finish, or ``""``. Records one event either way."""
        plan_source = getattr(agent, "plan_source", None)
        plan = plan_source() if callable(plan_source) else None
        scope = getattr(agent, "continuation_scope", None)
        event: dict[str, Any] = {"kind": "finish_check", "scope_id": getattr(scope, "scope_id", None),
                                 "model": self._model, "iterations_left": iterations_left, "fired": False}
        if plan is None or not plan.steps:
            record_event({**event, "reason": "no plan"})
            return ""
        if iterations_left < MIN_ITERS_LEFT:
            record_event({**event, "reason": "no room to act"})
            return ""
        host = tracing.current_host()
        span = tracing.start_span(host, tracing.SPAN_FINISH_CHECK, kind=tracing.KIND_LLM,
                                  attributes={"model": self._model})
        try:
            ledger = build_ledger(agent, [s.name for s in plan.subjects])
            result = self.check(plan, str(input_args.get("user_query") or ""), ledger)
        except Exception as error:  # noqa: BLE001 - building the ledger must not stop a turn either
            result = CheckResult(error=type(error).__name__)
        text = compose_note(result.flagged, iterations_left) if result.flagged else ""
        attributes = {"plan_source": plan.source, "steps": len(plan.steps), "subjects": len(plan.subjects),
                      "questions": result.questions, "requests": result.requests, "splits": result.splits,
                      "input_tokens": result.input_tokens, "latency_ms": result.latency_ms,
                      "flagged": len(result.flagged), "fired": bool(text), "error_type": result.error}
        tracing.end_span(host, span, status=tracing.STATUS_ERROR if result.error else tracing.STATUS_OK,
                         attributes=attributes)
        record_event({**event, **attributes, "fired": bool(text),
                      "reason": "error" if result.error else ("unexecuted steps" if text else "every step executed"),
                      "flagged_steps": result.flagged})
        return text


NOTE_HEAD = "Before you finish: this turn's record shows no command carrying out these steps of the plan:\n"
NOTE_TAIL = ("Run them if they are still needed, or say in your answer why they were not. "
             "You have {left} steps left.")
NOTE_MORE = "- and {count} more\n"


def compose_note(flagged: list[dict[str, Any]], iterations_left: int) -> str:
    """The note, capped at ``NOTE_MAX_BYTES``; every flagged step is counted even when cut."""
    tail = NOTE_TAIL.format(left=max(0, int(iterations_left)))
    lines = []
    for item in sorted(flagged, key=lambda f: (f["step"], f["subject"] or "")):
        who = f" (for {item['subject']})" if item["subject"] else ""
        lines.append(f"- step {item['step']}{who}: {item['text'].strip()[:160]}\n")
    budget = NOTE_MAX_BYTES - len(NOTE_HEAD.encode("utf-8")) - len(tail.encode("utf-8"))
    kept, used = [], 0
    for index, line in enumerate(lines):
        more = NOTE_MORE.format(count=len(lines) - index - 1) if index + 1 < len(lines) else ""
        size = len(line.encode("utf-8"))
        if used + size + len(more.encode("utf-8")) > budget:
            kept.append(NOTE_MORE.format(count=len(lines) - index))
            break
        kept.append(line)
        used += size
    return NOTE_HEAD + "".join(kept) + tail


_CHECKERS: dict[str, Optional[FinishChecker]] = {}
_CHECKERS_LOCK = threading.Lock()


def checker_from_env() -> Optional[FinishChecker]:
    """The configured check, built once per model; None when the check is off."""
    if TypeSafeClient is None:
        return None
    if (context_budget.env_value(CHECK_ENV) or "").strip().lower() != CHECK_JEV:
        return None
    key = context_budget.env_value(KEY_ENV)
    if not key:
        return None
    model = context_budget.env_value(MODEL_ENV) or DEFAULT_MODEL
    with _CHECKERS_LOCK:
        if model not in _CHECKERS:
            client = TypeSafeClient(
                api_key=key, model=model, timeout=CALL_TIMEOUT_SECONDS,
                retry=RetryPolicy(max_retries=0, timeout=CALL_TIMEOUT_SECONDS))
            _CHECKERS[model] = FinishChecker(client, model=model)
        return _CHECKERS[model]
