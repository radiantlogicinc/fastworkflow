"""Finish-time execution check (fix-4dsr): was every step of the turn's plan executed?

When the agent chooses ``finish``, a decision model (TypeSafe's Jev by default) is asked, for
every step of the turn's initial plan and every subject the request names,
whether the turn's record shows the step carried out for that subject. Steps it
judges unexecuted are named in one note that replaces the finish observation, and
the agent goes back to the loop -- once per turn, only with iterations left, and
the agent may finish anyway. It checks EXECUTION, not whether the request was
answered: a step whose command ran counts, whatever it returned.

About one flag in four names a step that did run, and the plan is the turn's
initial one even after the user answered an ``ask_user`` question. So the note
tells the agent not to repeat a change the record shows was already made, not
to make a change the user has not confirmed (ask instead), and to skip any step
the user has since declined or changed.

Measured offline on 82 recorded ido attempts none of it was tuned on (910
hand-labelled step x subject pairs, pre-registered): precision 0.77, recall 0.89,
F1 0.82, against 0.75 for the plain one-question-per-pair check. Every
unexecuted subject-less step (9 of 129) was flagged and nothing else was.
``FLAG_MIN``, ``ASK_MIN`` and the questions' wording were calibrated there, on
ido with ``DEFAULT_MODEL``, and nowhere else: those figures need not transfer to
another workflow or model. Every event records ``calibration``
(``CALIBRATION``) -- events only: it is not an attribute of the
``fw.finish_check`` span, whose contract is unchanged -- and a ``FW_FINISH_CHECK_MODEL`` other than the default warns
once per process.

OFF unless a deployment turns it on: ``FW_FINISH_CHECK=jev`` AND a
``JEV_API_KEY``. A key present for another purpose does not enable it, because
the check sends the request, the plan and a summary of every step (command,
context, the first bytes of its output) to a third party. All of it -- the
subjects' names and kinds, the ``refers_to`` identifiers and ``names_in_output``
included -- passes the archive's credential scrub (``jev_client.egress``) first.

The questions go through the vendor-neutral ``decision.DecisionProvider``
interface; Jev is its built-in implementation (``jev_client.JevProvider``).
``FW_FINISH_CHECK`` may instead name a provider registered from code
(``decision.register_decision_provider``) -- never a module path. The same
redaction gate and per-value filter apply to it. ``FLAG_MIN``,
``ASK_MIN``, the questions' wording and the precision and recall above were
calibrated with Jev: with any other provider they are unmeasured (one warning
per process).

It fails open: no SDK, no key, no plan, a timeout, an error, every vendor worker
busy or an exhausted time budget all mean no note, and the turn finishes exactly
as it would without the check. It never raises. The whole check has
``CHECK_BUDGET_SECONDS`` of wall clock, counted from before the ledger is
built, each call at most ``CALL_TIMEOUT_SECONDS`` of it; for the calls both
are hard cutoffs (``jev_client.bounded_call``). Building the ledger is not cut
off: it is local work plus archive reads that each wait at most
``LEDGER_READ_TIMEOUT_SECONDS`` for a locked database, the first failed read
ending the check (reason "error", stage "ledger"); its calls also draw on the turn's shared vendor
budget (``jev_client.TurnBudget``), whose running total lands on the event as
``vendor_ms``. A set ``FW_FINISH_CHECK`` that cannot take effect (an
unrecognised value, no SDK, no key, a rejected ``FW_JEV_BASE_URL``,
``FW_OFFLOAD_EVIDENCE_REDACTION=off``)
logs one warning per cause; see ``jev_client``. A failure is warned about at
most once per five minutes per (error type, HTTP status), counting the ones not
logged. A value ``jev_client.egress`` refuses at call time sends nothing: the
check is skipped with reason "policy_withheld", unwarned.

Every finish of an agent with a check attached records one ``finish_check``
event, with a ``reason`` even when nothing was asked ("disabled", "cap reached",
"no plan", "nothing to check", "no read-only steps", "no room to act", "error",
"policy_withheld", "subjects capped", "ledger incomplete"), ``provider`` (the flag value that
selected who answers: ``jev`` or a registered name), and ``user_replies``: how many
``ask_user`` steps the turn's trajectory holds, and ``vendor_calls_in_flight``
(``jev_client.calls_in_flight``) when any vendor worker slot is held and
``vendor_calls_orphaned`` (``jev_client.calls_orphaned``) when any abandoned
request still runs past its slot. A "no plan" event also carries
``no_plan_cause``: "planner_empty" (the planner returned nothing),
"plan_unreadable" (no steps could be read from what it returned),
"lost_on_resume" (the turn resumed from a session state written without its
plan) or "not_planned". "ledger incomplete" is a turn resumed in another process
from a suspension the context-window fallback had already cut: its record lacks
the cut steps, so it is not checked. An event recorded once the plan is read
also carries ``unchecked_for_effect`` (steps skipped as not provably read-only).
A checked finish's event also carries ``subjects_capped``
(how many subjects were left unchecked individually, 0 when none),
``ledger_bytes`` (the ledger as sent) and ``ledger_rows_trimmed``. An agent with no
check attached -- the check is off -- records none. Stored events name steps and
subjects by index, never by text. An "error" event also carries
``error_status``, ``error_request_id``, ``error_code`` (the body's
machine-readable code, never its text) and ``error_stage``: "ledger" (building
the ledger failed), "request" (a decision-model call or its answer), "budget"
(the check's or the turn's time budget ran out: ``OutOfTime``), "check" (the
check's own logic) or "note". A call cut off at its own cap is ``CallTimedOut``
and a call refused because every vendor worker was busy is ``VendorBusy``, both
at stage "request".

What the model sees, and why:

* the plan state -- the request and each step's text -- for "does step k need a
  command?", "does step k (or its part j) concern subject S?";
* the ledger state -- one row per step: the command, the context it ran in, the
  context it left the agent in (``acted_on``), every identifier in them resolved
  to the label a retrieved listing gave it (``refers_to``: an ``id  label``
  row, or the rest of a row of any listing ``listing.parse_table`` reads in its
  label mode -- a page or one group of a longer listing included -- keyed
  by its first cell; an identifier is a long token, a parameter value of the
  command, or a listed first cell a context clause names), the request's
  subjects found anywhere in its full output (``names_in_output``), whether it
  errored or came back empty (a step the NLU stage stopped before any command
  ran -- parameter extraction, an ambiguous or misunderstood command -- is an
  error, recorded at dispatch by ``record_dispatch``, not read from its text),
  and the first bytes of the output -- for "was step
  k (part j) executed for S?" and "was step k executed at all?".

A step's parts are checked separately so a partly executed step is not scored
as executed. Optional steps and steps that wait on the user are not checked.

Only a step that is provably read-only is checked (``provably_read_only``):
every command it and its parts name must be declared ``read_only`` in the
workflow's runtime manifest (``command_effects``). A step naming a ``write``
command, or one the manifest does not declare (``unknown``), is never asked
about, its text is not sent, and it is never named in the note, so the note
cannot drive a repeated or unconfirmed change. A step naming no command is still
checked: "does step k need a command?" decides it. Undeclared counts as not
read-only here -- the safe direction for a note that asks the agent to act; the
owner's "treat undeclared commands as read-only" decision was for a
framework-enforced approval gate before write commands, which does not exist.
So a workflow without a manifest, or with one that cannot be read, has only its
command-less steps checked. Replayed on the same ido labels with ido's
manifest (cached answers, no new calls): no flag left on a write step (7
before, after user-gated steps were already skipped), precision 0.80, recall
0.87 (0.88 before). How many steps were skipped this way is the event's
``unchecked_for_effect``; when it is every step the check would hold the agent
to, nothing is asked (reason "no read-only steps").

The size is bounded. A plan naming more than ``SUBJECTS_MAX`` subjects is
checked step by step only -- "does step k need a command?" and "was step k
executed at all?", at most two questions per step, both measured wordings --
with reason "subjects capped": a step missed for one subject among many goes
unflagged, a step missed for all of them does not. The subject list is not
sent and ``names_in_output`` is left empty, but the request, the plan's step
texts and the ledger (commands, contexts, output heads) still are, and they
usually name the subjects. Question batches are chunked; a ledger over ``LEDGER_MAX_BYTES`` loses
its rows' ``head`` and then their ``refers_to``, oldest rows first, until it
fits. A ledger still too large for one request is halved once, and a step
counts as executed if either half shows it; a half still too large is an
error.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastworkflow import context_budget, runtime_manifest, tracing
from fastworkflow.observation_offloading import decision, jev_client
from fastworkflow.observation_offloading.archive import SUMMARY_READ_TIMEOUT_SECONDS
from fastworkflow.observation_offloading.compact import EXECUTE_TOOL_NAME, step_indexes
from fastworkflow.observation_offloading.decision import YesNo
from fastworkflow.observation_offloading.labels import command_response
from fastworkflow.observation_offloading.listing import parse_table
from fastworkflow.observation_offloading.state import context_clause_of, record_event
from fastworkflow.turn_plan import PlanStep, TurnPlan, command_parts, is_checked, step_commands, step_text
from fastworkflow.utils.logging import logger

CHECK_ENV = "FW_FINISH_CHECK"
CHECK_JEV = jev_client.JEV
KEY_ENV = jev_client.KEY_ENV
MODEL_ENV = "FW_FINISH_CHECK_MODEL"
DEFAULT_MODEL = jev_client.DEFAULT_MODEL
#: What ``FLAG_MIN``, ``ASK_MIN`` and the questions' wording were calibrated on:
#: the ido workflow's recorded attempts, with ``DEFAULT_MODEL``. Recorded on
#: every ``finish_check`` event. Another workflow or model has not been
#: measured: the published precision and recall need not hold for it.
CALIBRATION = "ido-v7-2026-09"

#: A pair or step is named in the note when its unmet score reaches this.
#: Calibrated on ido (``CALIBRATION``).
FLAG_MIN = 0.5
#: Execution questions are only asked where scope and applicability reach this.
#: Calibrated on ido (``CALIBRATION``).
ASK_MIN = 0.3
QUESTIONS_PER_REQUEST = 100
#: Above this many named subjects the check asks step-level questions only.
#: Recorded turns (118, fix-4dsr) named at most 9.
SUBJECTS_MAX = 12
#: The ledger's sent size (UTF-8 JSON), bounded by trimming before the first call.
#: Every chunk resends the whole ledger. A full chunk of per-subject execution
#: questions is ~105 KB (~1 KB each, ~26k tokens at ~4 bytes per token); with a
#: ledger at this cap (~24k tokens) one request stays under the 54k input
#: tokens the model was seen to accept on recorded turns. Recorded turns (118)
#: built ledgers of at most 52 KB (p95 43 KB): none is trimmed.
LEDGER_MAX_BYTES = 96 * 1024
#: One call's timeout, and the whole check's budget, both wall clock and hard.
#: Measured: two sequential calls, median 0.53 s, p90 0.62 s, max 0.72 s over
#: 118 recorded turns.
CALL_TIMEOUT_SECONDS = 4.0
CHECK_BUDGET_SECONDS = 8.0
#: How long one archive read made while building the ledger waits for the
#: database lock; the first read that fails ends the check (stage "ledger").
#: With the archive's rows listed once and at most one clause read timing out,
#: a locked database costs the check about two of these, not the evidence
#: writes' 30 s.
LEDGER_READ_TIMEOUT_SECONDS = SUMMARY_READ_TIMEOUT_SECONDS
HEAD_BYTES = 200
NOTE_MAX_BYTES = 1024
#: The note costs an iteration and is worthless unless the agent can act on it.
MIN_ITERS_LEFT = 2

#: Bytes of one step's output searched for listings (``listing.parse_table``);
#: ``id  label`` rows (``_ROW_RE``) are read from all of it.
LISTING_PARSE_MAX_BYTES = 64 * 1024

_ROW_RE = re.compile(r"^\s*([A-Za-z0-9_]{16,})\s{2,}(\S.*?)\s*$")
_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{16,}")
_PARAM_RE = re.compile(r"<([A-Za-z_][\w.-]*)>(.*?)</\1>", re.DOTALL)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_ALIGNED_SPLIT_RE = re.compile(r" {2,}")
#: An output is empty when its first sentence opens by saying so ("No accounts
#: found for Alan.", "0 rows", "Identity not found") -- unless that sentence
#: goes on to a counted clause ("0 errors, 12 updated", "No errors; 12 accounts
#: updated") or picks from a counted set ("None of the 5 are overdue"): those
#: are results.
_EMPTY_RE = re.compile(
    r"\s*(?:(?:no|0|zero)\s+\w|none\b(?!\s+of\b)|nothing\b|(?:[\w()'-]+\s+){0,3}not\s+found\b)",
    re.IGNORECASE)
_COUNTED_CLAUSE_RE = re.compile(r"[,;][^,;]*\d")
_SENTENCE_END_RE = re.compile(r"[.!?\n]")
#: Entries kept in a finish_check event's score table.
SCORES_MAX = 200
#: The agent's tool for asking the user; its steps are counted as ``user_replies``.
ASK_USER_TOOL_NAME = "ask_user"
#: The event reason when ``jev_client.egress`` refused a value the check would send.
POLICY_WITHHELD = "policy_withheld"
#: The event reason when the plan named more than ``SUBJECTS_MAX`` subjects: fired
#: and ``flagged_steps`` say what the step-level check found.
SUBJECTS_CAPPED = "subjects capped"
#: The event reason when the turn resumed from a suspension whose steps the
#: context-window fallback had already cut: the ledger would miss them.
LEDGER_INCOMPLETE = "ledger incomplete"
#: ``dispatch_outcomes`` values: the step's command ran, or the NLU stage
#: stopped it first (parameter extraction, an ambiguous or misunderstood
#: command). A not-run step's ledger row has outcome "error".
DISPATCH_RAN = "ran"
DISPATCH_NOT_RUN = "not_run"
#: A "no plan" event's ``no_plan_cause`` when the agent cannot say why
#: (``plan_status``): no plan was made for the check.
NO_PLAN_DEFAULT_CAUSE = "not_planned"
#: The event reason when every step the check would hold the agent to names a
#: command not declared read-only (``provably_read_only``): nothing is asked.
NO_READ_ONLY_STEPS = "no read-only steps"
#: ``runtime_manifest.EffectKind`` values, and how much caution each demands
#: (the manifest's own order): a name declared twice takes the more severe.
READ_ONLY = "read_only"
UNKNOWN_EFFECT = "unknown"
_EFFECT_SEVERITY = {READ_ONLY: 0, UNKNOWN_EFFECT: 1, "write": 2}

#: A command name -> its declared ``runtime_manifest.EffectKind``.
CommandEffect = Callable[[str], str]


class PolicyWithheld(Exception):
    """``jev_client.egress`` refused a value the check would send: nothing is sent."""


def _redacted(text: str) -> str:
    """*text* as it may be sent (``jev_client.egress``); raises ``PolicyWithheld`` when it may not."""
    sent = jev_client.egress(text)
    if sent is None:
        raise PolicyWithheld()
    return sent

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

def record_dispatch(agent: Any, *, ran: bool) -> None:
    """Record whether the execute step being dispatched reached a command.

    Called from the dispatch of every ``execute_workflow_query`` call; a no-op
    unless a check is attached and active (not switched off by
    ``FW_EVAL_FINISH_REMINDERS=0``: ``workflow_agent.finish_check_active``,
    which imports this module). The step is the latest execute step of the
    mirror still waiting for its observation, so a nested call made for it (a
    clarified command retried) overwrites what the outer call recorded.
    """
    if (getattr(agent, "finish_checker", None) is None
            or not bool(getattr(agent, "finish_reminders_enabled", True))):
        return
    trajectory = getattr(agent, "current_trajectory", None) or {}
    pending = [index for index in step_indexes(trajectory)
               if str(trajectory.get(f"tool_name_{index}") or "") == EXECUTE_TOOL_NAME
               and f"observation_{index}" not in trajectory]
    if not pending:
        return
    outcomes = getattr(agent, "dispatch_outcomes", None)
    if not isinstance(outcomes, dict):
        outcomes = agent.dispatch_outcomes = {}
    outcomes[str(pending[-1])] = DISPATCH_RAN if ran else DISPATCH_NOT_RUN


def _outcome(response: str) -> str:
    if response.startswith("Execution error"):
        return "error"
    first_sentence = _SENTENCE_END_RE.split(response.strip(), maxsplit=1)[0].strip()
    if not response.strip() or (_EMPTY_RE.match(first_sentence)
                                and not _COUNTED_CLAUSE_RE.search(first_sentence)):
        return "empty"
    return "result"


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)


def _head(text: str) -> str:
    return text.encode("utf-8")[:HEAD_BYTES].decode("utf-8", "ignore")


def _row_cells(shape: str, row: str) -> list[str]:
    if shape == "markdown":
        return [cell.strip() for cell in row.strip().strip("|").split("|")]
    if shape == "tabbed":
        return [cell.strip() for cell in row.split("\t")]
    return _ALIGNED_SPLIT_RE.split(row.strip())


def _listing_labels(response: str) -> list[tuple[str, str]]:
    """``(first cell, rest of row)`` for every row of every listing in *response*.

    Only the first ``LISTING_PARSE_MAX_BYTES`` are read, cut at a line. Each
    blank-line-separated block is parsed on its own, and within a block each
    listing after the one before it. Listings are read in ``parse_table``'s
    label mode (``require_complete=False``): a page of a longer listing, or one
    group of several, still labels its rows; a malformed row still refuses the
    rest of its block.
    """
    text = response.encode("utf-8")[:LISTING_PARSE_MAX_BYTES].decode("utf-8", "ignore")
    if len(text) < len(response):
        text = text.rsplit("\n", 1)[0]
    found = []
    for block in _BLANK_LINE_RE.split(text):
        lines = block.splitlines()
        while lines:
            table = parse_table("\n".join(lines), require_complete=False)
            if table is None:
                break
            for row in table["rows"]:
                cells = _row_cells(table["shape"], row)
                if len(cells) >= 2 and cells[0]:
                    found.append((cells[0], " ".join(cell for cell in cells[1:] if cell)))
            lines = lines[table["end_line"]:]
    return found


def _clause_key(key: str) -> bool:
    """Whether label key *key* may be recognised as a token of a context clause: a short number may not."""
    return len(key) >= 4 or not key.isdigit()


def _token_edge(char: str) -> bool:
    return char.isalnum() or char in "_-"


def _whole_token_in(key: str, text: str) -> bool:
    """Whether *key* occurs in *text* with no word character or ``-`` touching either end."""
    start = text.find(key)
    while start >= 0:
        end = start + len(key)
        if (start == 0 or not _token_edge(text[start - 1])) and (end == len(text) or not _token_edge(text[end])):
            return True
        start = text.find(key, start + 1)
    return False


def _ledger_clause(scope: Any, alias: str, store: Any) -> str:
    """*alias*'s context clause: process memory first, then the archive, waiting at most
    ``LEDGER_READ_TIMEOUT_SECONDS``. An archive that cannot be read raises.

    Through ``context_clause_of``, so an archive answer -- a clause, or "no
    subject" -- is cached like any other: a later check on the turn reads no
    alias from the archive twice."""
    return context_clause_of(scope, alias, selected_archive=store,
                             timeout=LEDGER_READ_TIMEOUT_SECONDS, strict=True) or ""


def build_ledger(agent: Any, subject_names: list[str]) -> list[dict[str, Any]]:
    """One row per non-finish step of the turn's full trajectory, redacted for sending.

    Every value in a row passes ``jev_client.egress``, the ``refers_to`` keys
    and the ``names_in_output`` names included; *subject_names* are matched
    raw, locally. Raises ``PolicyWithheld`` when ``egress`` refuses a value.
    """
    trajectory = getattr(agent, "current_trajectory", None) or {}
    scope = getattr(agent, "continuation_scope", None)
    store = getattr(agent, "observation_archive", None)
    dispatched = getattr(agent, "dispatch_outcomes", None) or {}
    pairs_of = getattr(agent, "execute_ordinal_pairs", None)
    alias_of = {index: f"O{ordinal}" for index, ordinal in (pairs_of(trajectory) if callable(pairs_of) else [])}
    archived: dict[str, str] = {}
    if scope is not None and store is not None:
        # History: an unreadable archive used to leave the inline text (the
        # error was swallowed here). Every archive read now waits at most
        # ``LEDGER_READ_TIMEOUT_SECONDS`` and a failure raises, so a locked
        # database costs the check its ledger (reason "error", stage "ledger")
        # rather than the 30 s evidence wait, or a ledger of offload labels.
        archived = {str(h["alias"]): str(h.get("text") or "")
                    for h in store.list(scope, timeout=LEDGER_READ_TIMEOUT_SECONDS)}

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
            clause = _ledger_clause(scope, alias, store)
        command = (str(args.get("command") or "") if tool == EXECUTE_TOOL_NAME and isinstance(args, dict)
                   else json.dumps(args, ensure_ascii=False, default=str)[:HEAD_BYTES])
        steps.append({"tool": tool, "command": command or tool, "context": clause,
                      "response": response, "alias": alias,
                      "not_run": dispatched.get(str(index)) == DISPATCH_NOT_RUN})

    # ``id  label`` rows first, so a listing's parse never relabels one of them.
    labels: dict[str, dict[str, str]] = {}
    for step in steps:
        for line in step["response"].splitlines():
            match = _ROW_RE.match(line)
            if match and match.group(1) not in labels:
                labels[match.group(1)] = {"label": match.group(2)[:80], "listed_by": step["command"][:90]}
    for step in steps:
        for key, label in _listing_labels(step["response"]):
            if key not in labels:
                labels[key] = {"label": label[:80], "listed_by": step["command"][:90]}
    clause_keys = [key for key in labels if _clause_key(key)]
    keys_in_clause: dict[str, list[str]] = {}

    def in_clause(clause: str) -> list[str]:
        if clause not in keys_in_clause:
            keys_in_clause[clause] = [key for key in clause_keys if _whole_token_in(key, clause)] if clause else []
        return keys_in_clause[clause]

    name_patterns = [(name, _name_pattern(name)) for name in subject_names if name]
    rows = []
    for position, step in enumerate(steps):
        acted_on = ""
        if step["tool"] == EXECUTE_TOOL_NAME:
            acted_on = next((later["context"] for later in steps[position + 1:] if later["context"]), "")
        refers_to = {}
        # Long tokens anywhere, then the command's parameter values, then any
        # listed identifier the context clauses name.
        candidates = [*_ID_TOKEN_RE.findall(" ".join([step["command"], step["context"], acted_on])),
                      *(value.strip() for _name, value in _PARAM_RE.findall(step["command"])),
                      *in_clause(step["context"]), *in_clause(acted_on)]
        for token in dict.fromkeys(candidates):
            if token in labels:
                listed_by = labels[token]["listed_by"]
                for inner in _ID_TOKEN_RE.findall(listed_by):
                    if inner in labels and inner != token:
                        listed_by += f" [{inner} = {labels[inner]['label']}]"
                refers_to[_redacted(token)] = _redacted(f"{labels[token]['label']} (listed by {listed_by})")
        rows.append({
            "n": position + 1,
            "command": _redacted(step["command"]),
            "context": _redacted(step["context"]),
            "acted_on": _redacted(acted_on),
            "refers_to": refers_to,
            "names_in_output": [_redacted(name) for name, pattern in name_patterns
                                if pattern.search(step["response"])],
            "outcome": "error" if step["not_run"] else _outcome(step["response"]),
            "head": _redacted(_head(step["response"])),
        })
    return rows


def user_replies(agent: Any) -> int:
    """How many ``ask_user`` steps the turn's full trajectory holds."""
    trajectory = getattr(agent, "current_trajectory", None) or {}
    return sum(1 for index in step_indexes(trajectory)
               if str(trajectory.get(f"tool_name_{index}") or "") == ASK_USER_TOOL_NAME)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def bound_ledger(rows: list[dict[str, Any]], max_bytes: int = LEDGER_MAX_BYTES) -> tuple[list[dict[str, Any]], int]:
    """*rows* within *max_bytes* of JSON, and how many rows were trimmed to get there.

    Empties ``head``, oldest row first, then ``refers_to``, until the ledger
    fits; the rest of a row is kept, so a ledger of many rows may still exceed
    *max_bytes*. *rows* is not changed: trimmed rows are copies.
    """
    size = _json_bytes(rows)
    if size <= max_bytes:
        return rows, 0
    rows = [dict(row) for row in rows]
    trimmed: set[int] = set()
    for key, empty in (("head", ""), ("refers_to", {})):
        for index, row in enumerate(rows):
            if size <= max_bytes:
                break
            if row.get(key):
                size -= _json_bytes(row[key]) - _json_bytes(empty)
                row[key] = empty
                trimmed.add(index)
    return rows, len(trimmed)


def _named_subjects(plan: TurnPlan) -> list[tuple[str, str]]:
    return [(s.name, (s.kind or "item").strip() or "item") for s in plan.subjects if s.name.strip()]


def subjects_capped(plan: TurnPlan) -> bool:
    """Whether *plan* names more than ``SUBJECTS_MAX`` subjects, and so is checked step by step only."""
    return len(_named_subjects(plan)) > SUBJECTS_MAX


# ---------------------------------------------------------------------------
# Which steps may be checked: the commands' declared effects
# ---------------------------------------------------------------------------

def _all_unknown(_command: str) -> str:
    return UNKNOWN_EFFECT


def command_effects(workflow_path: str) -> CommandEffect:
    """The declared effect of each of the workflow's commands, by name; never raises.

    Reads the metadata registered at startup (``runtime_manifest.get_runtime_metadata``),
    else the workflow's ``workflow_runtime.json`` (``runtime_manifest.load_manifest``,
    merged over the core manifest with no deployment features). No manifest, one
    that cannot be read or merged, or any other failure makes every command
    ``unknown``. A plan's command is matched by its last ``/`` segment, and a name
    several qualified keys share takes the most severe of their kinds. A name the
    manifest does not declare is ``unknown`` -- never ``read_only``.
    """
    try:
        metadata = runtime_manifest.get_runtime_metadata(workflow_path) if workflow_path else None
        if metadata is None and workflow_path:
            manifest = runtime_manifest.load_manifest(workflow_path)
            if manifest is not None:
                metadata = runtime_manifest.merge_and_gate(manifest, deployment_features={})
        if metadata is None or not metadata.has_workflow_manifest:
            return _all_unknown
        kinds: dict[str, str] = {}
        for key, declaration in metadata.commands.items():
            name = str(key).split("/")[-1]
            kind = str(declaration.effect_kind())
            severity = _EFFECT_SEVERITY.get(kind, _EFFECT_SEVERITY[UNKNOWN_EFFECT])
            if name not in kinds or severity > _EFFECT_SEVERITY.get(kinds[name], _EFFECT_SEVERITY[UNKNOWN_EFFECT]):
                kinds[name] = kind
    except Exception as error:  # noqa: BLE001 - an unreadable manifest only narrows the check
        logger.warning("finish check: the runtime manifest of %s could not be read (%s); "
                       "only steps naming no command are checked", workflow_path, type(error).__name__)
        return _all_unknown

    def effect(command: str) -> str:
        return kinds.get(str(command).strip().split("/")[-1], UNKNOWN_EFFECT)

    return effect


def provably_read_only(step: PlanStep, command_effect: Optional[CommandEffect]) -> bool:
    """Whether every command *step* and its parts name is declared ``read_only``.

    True for a step naming no command. *command_effect* None, or one that
    raises, makes every command ``unknown``.
    """
    names = step_commands(step)
    if not names:
        return True
    effect = command_effect or _all_unknown
    try:
        return all(effect(name) == READ_ONLY for name in names)
    except Exception:  # noqa: BLE001 - an effect lookup that fails proves nothing
        return False


def checked_steps(plan: TurnPlan, command_effect: Optional[CommandEffect]) -> tuple[dict[int, PlanStep], int]:
    """The steps the check asks about, by 1-based number, and how many ``is_checked``
    steps were left out as not ``provably_read_only``."""
    held = {k: step for k, step in enumerate(plan.steps, 1) if is_checked(step)}
    steps = {k: step for k, step in held.items() if provably_read_only(step, command_effect)}
    return steps, len(held) - len(steps)


# ---------------------------------------------------------------------------
# The questions (wording as measured; changing it invalidates the measurement)
# ---------------------------------------------------------------------------

def _action_question(k: int) -> YesNo:
    return YesNo(
        instructions=(f"Does plan step {k} require running a workflow command to retrieve or change something? "
                      f"Answer no if step {k} only reasons over results already obtained (compare, intersect, "
                      f"summarise, report) or asks the user."),
        true=f"Step {k} runs at least one command.",
        false=f"Step {k} runs no command: it reasons, reports or asks the user.")


def _applies_question(k: int, name: str, kind: str) -> YesNo:
    return YesNo(
        instructions=(f'Does plan step {k} call for work on the {kind} "{name}"? Answer yes if step {k} names '
                      f'"{name}", or if step {k} applies to every {kind} named in the request and "{name}" is one of them.'),
        true=f'Step {k} requires an action about "{name}".',
        false=f'Step {k} is about other things, not "{name}".')


def _part_applies_question(k: int, j: int, part: str, name: str, kind: str) -> YesNo:
    return YesNo(
        instructions=(f'Part {j} of plan step {k}: "{part}"\nDoes this part call for work on the {kind} "{name}"? '
                      f'Answer yes if it names "{name}", or if it applies to every {kind} it is about and "{name}" is one of them.'),
        true=f'This part requires an action about "{name}".',
        false=f'This part is about other things, not "{name}".')


def _exec_question(step: str, part: Optional[str], name: str) -> YesNo:
    what = f'Plan step: "{step}"' + (f'\nPart of that step: "{part}"' if part else "")
    target = "this part" if part else "this step"
    return YesNo(
        instructions=(f"{what}\nWas {target} executed for \"{name}\"? Yes if some ledger row ran the command "
                      f"{target} names, or another command doing the same kind of action, for \"{name}\", "
                      f"and the row is not an execution error. What the command returned does not matter. "
                      + _ROW_RULE.format(name=name)),
        true=f'Some row ran {target}\'s action for "{name}" without an execution error.',
        false=f'No row ran {target}\'s action for "{name}", or every such row was an execution error.')


def _any_question(k: int, step: str) -> YesNo:
    return YesNo(
        instructions=(f'Plan step {k}: "{step}"\nWas this step executed at all? Yes if some ledger row ran '
                      f"the command it names, or another command doing the same kind of action, and the row "
                      f"is not an execution error. What the command returned does not matter."),
        true="Some row ran this step's action without an execution error.",
        false="No row ran this step's action, or every such row was an execution error.")


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

#: The whole check's budget, or the turn's, ran out before its questions were answered.
_OutOfTime = decision.OutOfTime


@dataclass
class CheckResult:
    """What one check found. ``error`` set means it produced no verdicts.

    ``scores`` is one entry per checked step x subject (x part), subjects by
    index into the plan's subjects, at most ``SCORES_MAX`` of them.
    """

    flagged: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    scores_truncated: int = 0
    questions: int = 0
    requests: int = 0
    splits: int = 0
    input_tokens: int = 0
    latency_ms: int = 0
    #: The plan's subject count when over ``SUBJECTS_MAX`` (checked step by step only), else 0.
    subjects_capped: int = 0
    #: The ledger as sent (``bound_ledger``), and how many of its rows were trimmed.
    ledger_bytes: int = 0
    ledger_rows_trimmed: int = 0
    #: Steps ``is_checked`` holds the agent to that were not asked about: not ``provably_read_only``.
    unchecked_for_effect: int = 0
    error: Optional[str] = None
    #: ``decision.describe`` of the error, and where it happened.
    failure: Optional[dict[str, Any]] = None
    error_stage: Optional[str] = None
    #: ``jev_client.egress`` refused a value the check would send; nothing was sent.
    withheld: bool = False


def _failure_fields(failure: Optional[dict[str, Any]], stage: Optional[str]) -> dict[str, Any]:
    if failure is None:
        return {}
    return {"error_status": failure.get("status"), "error_request_id": failure.get("request_id"),
            "error_code": failure.get("code"), "error_stage": stage}


class FinishChecker:
    """One configured check. ``note`` and ``check`` never raise.

    *client* is the ``decision.DecisionProvider`` that answers, or a Jev SDK
    client (``jev_client.make_client``), which is wrapped in a
    ``jev_client.JevProvider``. *provider_name* (default: the provider's
    ``name``) is recorded on every event as ``provider``.
    """

    def __init__(self, client: Any, model: str = "", *, budget_seconds: float = CHECK_BUDGET_SECONDS,
                 call_seconds: float = CALL_TIMEOUT_SECONDS,
                 ledger_max_bytes: int = LEDGER_MAX_BYTES,
                 warner: Optional[jev_client.FailureWarner] = None,
                 provider_name: Optional[str] = None) -> None:
        self._provider = jev_client.as_provider(client)
        self._provider_name = provider_name or str(getattr(self._provider, "name", "") or "")
        self._model = model
        self._budget_seconds = budget_seconds
        self._call_seconds = call_seconds
        self._ledger_max_bytes = ledger_max_bytes
        self._warner = warner or jev_client.FailureWarner("finish check", "the turn finishes unchecked")

    # -- one decision-model request, chunked and halved at most once ----------

    def _call(self, state: Any, questions: dict[str, Any], result: CheckResult, deadline: float,
              budget: Optional[jev_client.TurnBudget]) -> dict[str, float]:
        result.error_stage = "request"
        response = self._provider.ask(state, questions, cap=self._call_seconds, deadline=deadline, budget=budget)
        result.requests += 1
        usage = getattr(response, "usage", None)
        if usage:
            result.input_tokens += int(usage.get("input_tokens", 0) or 0)
        answers = {key: float(response[key]) for key in questions}
        result.error_stage = None
        return answers

    def _ask(self, rows: list[Any], questions: dict[str, Any], wrap: Callable[[list[Any]], Any],
             result: CheckResult, deadline: float,
             budget: Optional[jev_client.TurnBudget]) -> dict[str, float]:
        items = list(questions.items())
        answers: dict[str, float] = {}
        # Once one chunk needed the halves, every later chunk goes straight to them.
        pieces = [rows]
        for start in range(0, len(items), QUESTIONS_PER_REQUEST):
            chunk = dict(items[start:start + QUESTIONS_PER_REQUEST])
            if len(pieces) == 1:
                try:
                    answers.update(self._call(wrap(rows), chunk, result, deadline, budget))
                    continue
                except decision.StateTooLarge:  # only an over-long state is split
                    if len(rows) < 2:
                        raise
                    result.splits += 1
                    middle = len(rows) // 2
                    pieces = [rows[:middle], rows[middle:]]
            halves = [self._call(wrap(piece), chunk, result, deadline, budget) for piece in pieces]
            answers.update({key: max(half[key] for half in halves) for key in chunk})
        return answers

    # -- the verdicts ---------------------------------------------------------

    def check(self, plan: TurnPlan, request: str, ledger: list[dict[str, Any]], *,
              command_effect: Optional[CommandEffect] = None,
              deadline: Optional[float] = None,
              budget: Optional[jev_client.TurnBudget] = None) -> CheckResult:
        """The verdicts, within *deadline* (default: ``budget_seconds`` from now) and the turn's *budget*.

        Only ``checked_steps`` are asked about, by *command_effect* (None: every
        command ``unknown``, so only steps naming no command).
        """
        started = time.monotonic()
        if deadline is None:
            deadline = started + self._budget_seconds
        result = CheckResult()
        try:
            self._check(plan, request, ledger, result, deadline, budget, command_effect)
        except PolicyWithheld:
            result = CheckResult(withheld=True)
        except Exception as error:  # noqa: BLE001 - the check must never stop a turn
            failure = decision.describe(error)
            if isinstance(error, _OutOfTime):
                failure["error_type"] = "OutOfTime"
                result.error_stage = "budget"
            else:
                result.error_stage = result.error_stage or "check"
            self._warner.warn(failure, stage=result.error_stage)
            result.flagged = []
            result.scores = []
            result.scores_truncated = 0
            result.error = failure["error_type"]
            result.failure = failure
        result.latency_ms = round((time.monotonic() - started) * 1000)
        return result

    def _check(self, plan: TurnPlan, request: str, ledger: list[dict[str, Any]],
               result: CheckResult, deadline: float, budget: Optional[jev_client.TurnBudget],
               command_effect: Optional[CommandEffect] = None) -> None:
        steps, result.unchecked_for_effect = checked_steps(plan, command_effect)
        # A step left out for its effect is not sent at all; optional and
        # user-gated steps still are, as context for the others.
        sent_steps = {k: step for k, step in enumerate(plan.steps, 1)
                      if k in steps or not is_checked(step)}
        subjects = _named_subjects(plan)
        if not steps:
            return
        if len(subjects) > SUBJECTS_MAX:
            result.subjects_capped = len(subjects)
            subjects = []
        ledger, result.ledger_rows_trimmed = bound_ledger(ledger, self._ledger_max_bytes)
        result.ledger_bytes = _json_bytes(ledger)
        # Everything sent is filtered before the first call, so a withheld value sends nothing.
        plan_state = {"request": _redacted(request),
                      "plan_steps": {str(k): _redacted(step_text(step)) for k, step in sent_steps.items()}}
        sent_subjects = [(_redacted(name), _redacted(kind)) for name, kind in subjects]
        parts = {k: command_parts(step) for k, step in steps.items()}
        sent_parts = {k: [_redacted(part.text) for part in parts[k]] for k in steps}

        scope_questions: dict[str, Any] = {}
        for k in steps:
            scope_questions[f"x{k}"] = _action_question(k)
            for si, (name, kind) in enumerate(sent_subjects):
                scope_questions[f"a{k}_{si}"] = _applies_question(k, name, kind)
                for j, part in enumerate(sent_parts[k], 1):
                    scope_questions[f"p{k}_{j}_{si}"] = _part_applies_question(k, j, part, name, kind)
        result.questions += len(scope_questions)
        scope = self._ask([plan_state], scope_questions, lambda rows: rows[0], result, deadline, budget)

        exec_questions: dict[str, Any] = {}
        for k, step in steps.items():
            if scope[f"x{k}"] < ASK_MIN:
                continue
            text = plan_state["plan_steps"][str(k)]
            exec_questions[f"g{k}"] = _any_question(k, text)
            for si, (name, _kind) in enumerate(sent_subjects):
                if scope[f"a{k}_{si}"] < ASK_MIN:
                    continue
                asked_parts = [(j, part) for j, part in enumerate(sent_parts[k], 1)
                               if scope[f"p{k}_{j}_{si}"] >= ASK_MIN]
                if not asked_parts:
                    # The step concerns S but no single part does: ask about the whole step.
                    exec_questions[f"e{k}_{si}"] = _exec_question(text, None, name)
                    continue
                for j, part in asked_parts:
                    exec_questions[f"e{k}_{j}_{si}"] = _exec_question(text, part, name)
        result.questions += len(exec_questions)
        executed = (self._ask(ledger, exec_questions, lambda rows: {"ledger": rows}, result, deadline, budget)
                    if exec_questions else {})

        def score(k: int, si: Optional[int], part: Optional[int], applies: float,
                  was_executed: Optional[float], unmet: float) -> None:
            if len(result.scores) >= SCORES_MAX:
                result.scores_truncated += 1
                return
            result.scores.append({"step": k, "subject": si, "part": part, "applies": round(applies, 2),
                                  "executed": None if was_executed is None else round(was_executed, 2),
                                  "unmet": round(unmet, 2)})

        for k, step in steps.items():
            x = scope[f"x{k}"]
            best_applies = max((scope[f"a{k}_{si}"] for si in range(len(subjects))), default=0.0)
            for si, (name, kind) in enumerate(subjects):
                applies = x * scope[f"a{k}_{si}"]
                part_keys = [j for j in range(1, len(parts[k]) + 1) if f"e{k}_{j}_{si}" in executed]
                if part_keys:
                    worst = 0.0
                    for j in part_keys:
                        part_unmet = scope[f"p{k}_{j}_{si}"] * (1 - executed[f"e{k}_{j}_{si}"])
                        worst = max(worst, part_unmet)
                        score(k, si, j, applies * scope[f"p{k}_{j}_{si}"], executed[f"e{k}_{j}_{si}"],
                              applies * part_unmet)
                    unmet = applies * worst
                elif f"e{k}_{si}" in executed:
                    unmet = applies * (1 - executed[f"e{k}_{si}"])
                    score(k, si, None, applies, executed[f"e{k}_{si}"], unmet)
                else:
                    unmet = 0.0
                    score(k, si, None, applies, None, unmet)
                if unmet >= FLAG_MIN:
                    result.flagged.append({"step": k, "subject": name, "subject_index": si, "kind": kind,
                                           "text": step.text, "p_unmet": round(unmet, 3)})
            if f"g{k}" in executed:
                unmet = x * (1 - best_applies) * (1 - executed[f"g{k}"])
                score(k, None, None, x * (1 - best_applies), executed[f"g{k}"], unmet)
                if unmet >= FLAG_MIN:
                    result.flagged.append({"step": k, "subject": None, "subject_index": None, "kind": None,
                                           "text": step.text, "p_unmet": round(unmet, 3)})
            else:
                score(k, None, None, x, None, 0.0)

    # -- what the agent is told -----------------------------------------------

    def _event(self, agent: Any, iterations_left: int) -> dict[str, Any]:
        scope = getattr(agent, "continuation_scope", None)
        event = {"kind": "finish_check", "scope_id": getattr(scope, "scope_id", None),
                 "provider": self._provider_name, "model": self._model, "calibration": CALIBRATION,
                 "iterations_left": iterations_left, "fired": False,
                 "user_replies": user_replies(agent)}
        event.update(jev_client.vendor_pressure())
        return event

    def record_skip(self, agent: Any, *, reason: str, iterations_left: int,
                    error_type: Optional[str] = None, failure: Optional[dict[str, Any]] = None,
                    error_stage: Optional[str] = None) -> None:
        """Record a finish that produced no check: switched off, cap reached, or a failure."""
        try:
            event = {**self._event(agent, iterations_left), "reason": reason}
            if error_type is not None:
                event["error_type"] = error_type
            event.update(_failure_fields(failure, error_stage))
            record_event(event)
        except Exception as error:  # noqa: BLE001 - recording must not stop a turn
            logger.warning("finish check event not recorded: %s", type(error).__name__)

    def note(self, agent: Any, input_args: dict[str, Any], *, iterations_left: int,
             shown_iterations_left: Optional[int] = None) -> str:
        """The note for this finish, or ``""``. Records one event either way; never raises.

        ``iterations_left`` decides whether the note may fire; ``shown_iterations_left``
        (default: the same) is the count the note states, which for a segmented
        agent includes the later segments' room.
        """
        try:
            return self._note(agent, input_args, iterations_left, shown_iterations_left)
        except Exception as error:  # noqa: BLE001 - the check must never stop a turn
            failure = decision.describe(error)
            self._warner.warn(failure, stage="note")
            self.record_skip(agent, reason="error", iterations_left=iterations_left,
                             error_type=failure["error_type"], failure=failure, error_stage="note")
            return ""

    def _note(self, agent: Any, input_args: dict[str, Any], iterations_left: int,
              shown_iterations_left: Optional[int] = None) -> str:
        event = self._event(agent, iterations_left)
        if shown_iterations_left is not None:
            event["shown_iterations_left"] = shown_iterations_left
        plan_source = getattr(agent, "plan_source", None)
        plan = plan_source() if callable(plan_source) else None
        if plan is None or not plan.steps:
            plan_status = getattr(agent, "plan_status", None)
            cause = plan_status() if callable(plan_status) else None
            record_event({**event, "reason": "no plan", "no_plan_cause": cause or NO_PLAN_DEFAULT_CAUSE})
            return ""
        if getattr(agent, "ledger_incomplete", False):
            record_event({**event, "reason": LEDGER_INCOMPLETE})
            return ""
        command_effect = getattr(agent, "command_effect", None)
        steps, event["unchecked_for_effect"] = checked_steps(plan, command_effect)
        if not any(is_checked(step) for step in plan.steps):
            record_event({**event, "reason": "nothing to check"})
            return ""
        if not steps:
            record_event({**event, "reason": NO_READ_ONLY_STEPS})
            return ""
        if iterations_left < MIN_ITERS_LEFT:
            record_event({**event, "reason": "no room to act"})
            return ""
        host = tracing.current_host()
        span = tracing.start_span(host, tracing.SPAN_FINISH_CHECK, kind=tracing.KIND_LLM,
                                  attributes={"model": self._model})
        attributes: dict[str, Any] = {
            "plan_source": None, "steps": None, "subjects": None, "questions": None, "requests": None,
            "splits": None, "input_tokens": None, "latency_ms": None, "flagged": None, "fired": False,
            "error_type": None}
        status = tracing.STATUS_ERROR
        # The check's wall clock starts before the ledger is built; an agent
        # without a turn budget gets one for this check alone.
        deadline = time.monotonic() + self._budget_seconds
        budget = getattr(agent, "vendor_budget", None)
        if not isinstance(budget, jev_client.TurnBudget):
            budget = jev_client.TurnBudget()
        try:
            try:
                # Step-level questions never read names_in_output: it is not filled, so the
                # subject list itself is not sent (the request and step texts still are).
                ledger = build_ledger(agent, [] if subjects_capped(plan) else [s.name for s in plan.subjects])
            except PolicyWithheld:
                result = CheckResult(withheld=True)
            except Exception as error:  # noqa: BLE001 - building the ledger must not stop a turn either
                failure = decision.describe(error)
                self._warner.warn(failure, stage="ledger", exc_info=error)
                result = CheckResult(error=failure["error_type"], failure=failure, error_stage="ledger")
            else:
                result = self.check(plan, str(input_args.get("user_query") or ""), ledger,
                                    command_effect=command_effect, deadline=deadline, budget=budget)
            shown = iterations_left if shown_iterations_left is None else shown_iterations_left
            text = compose_note(result.flagged, shown) if result.flagged else ""
            attributes = {"plan_source": plan.source, "steps": len(plan.steps), "subjects": len(plan.subjects),
                          "questions": result.questions, "requests": result.requests, "splits": result.splits,
                          "input_tokens": result.input_tokens, "latency_ms": result.latency_ms,
                          "flagged": len(result.flagged), "fired": bool(text), "error_type": result.error}
            status = tracing.STATUS_ERROR if result.error else tracing.STATUS_OK
        except BaseException as error:
            attributes = {**attributes, "fired": False, "error_type": type(error).__name__}
            raise
        finally:
            tracing.end_span(host, span, status=status, attributes=attributes)
        if result.withheld:
            reason = POLICY_WITHHELD
        elif result.error:
            reason = "error"
        elif result.subjects_capped:
            reason = SUBJECTS_CAPPED
        else:
            reason = "unexecuted steps" if text else "every step executed"
        record_event({**event, **attributes, **_failure_fields(result.failure, result.error_stage),
                      "fired": bool(text),
                      "reason": reason,
                      "vendor_ms": budget.vendor_ms,
                      "subjects_capped": result.subjects_capped,
                      "ledger_bytes": result.ledger_bytes,
                      "ledger_rows_trimmed": result.ledger_rows_trimmed,
                      "flagged_steps": [{"step": f["step"], "subject": f["subject_index"],
                                         "kind": jev_client.redacted(f["kind"]) if f["kind"] else None,
                                         "p_unmet": f["p_unmet"]} for f in result.flagged],
                      "scores": result.scores, "scores_truncated": result.scores_truncated})
        return text


NOTE_HEAD = "Before you finish: this turn's record shows no command carrying out these steps of the plan:\n"
NOTE_TAIL = ("Run any that are still needed, or say in your answer why not. Do not repeat a change "
             "this turn's record shows was already made, and do not make a change the user has not "
             "confirmed: ask them instead. Skip any step the user has since declined or changed. "
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


_CHECKERS = jev_client.ClientCache()
#: The cache slot of a check answered by a registered provider, keyed by its registration.
REGISTERED_SLOT = "registered"
_MODEL_OVERRIDE_WARNED: set[str] = set()
_MODEL_OVERRIDE_LOCK = threading.Lock()


def _warn_uncalibrated_model(model: str) -> None:
    if model == DEFAULT_MODEL:
        return
    with _MODEL_OVERRIDE_LOCK:
        if _MODEL_OVERRIDE_WARNED:
            return
        _MODEL_OVERRIDE_WARNED.add(model)
    logger.warning("%s=%s: the finish check's thresholds and questions were calibrated with %s "
                   "(calibration %s), not with this model; its precision and recall are unmeasured",
                   MODEL_ENV, model, DEFAULT_MODEL, CALIBRATION)


def checker_from_env() -> Optional[FinishChecker]:
    """The configured check, built once per model, endpoint and key; None when the check is off.

    A set ``FW_FINISH_CHECK`` that cannot take effect warns once per cause
    (``jev_client.requested_key``, ``jev_client.base_url``). A
    ``FW_FINISH_CHECK_MODEL`` other than ``DEFAULT_MODEL`` warns once per
    process: the calibration (``CALIBRATION``) was not made with it.

    A value naming a provider registered from code
    (``decision.register_decision_provider``) selects it instead of Jev: built
    once per registration, with the provider's ``model`` attribute (if any) as
    the event's ``model``; ``FW_FINISH_CHECK_MODEL``, ``JEV_API_KEY`` and
    ``FW_JEV_BASE_URL`` are not read. ``FLAG_MIN``, ``ASK_MIN``, the questions'
    wording and the published precision and recall were calibrated with Jev
    (``CALIBRATION``): with another provider they are unmeasured, and selecting
    one warns once.
    """
    entry = decision.registration(jev_client.flag_value(CHECK_ENV))
    if entry is not None:
        provider = jev_client.registered_provider(CHECK_ENV, entry, "the finish check")
        if provider is None:
            return None
        return _CHECKERS.get((REGISTERED_SLOT, entry.name), f"{entry.name}#{entry.generation}",
                             lambda: FinishChecker(provider, model=str(getattr(provider, "model", "") or ""),
                                                   provider_name=entry.name))
    key = jev_client.requested_key(CHECK_ENV, "the finish check")
    if key is None:
        return None
    url = jev_client.base_url()
    if url is None:
        return None
    model = context_budget.env_value(MODEL_ENV) or DEFAULT_MODEL
    _warn_uncalibrated_model(model)
    return _CHECKERS.get((model, url), key, lambda: FinishChecker(
        jev_client.make_client(key, model, CALL_TIMEOUT_SECONDS + jev_client.CUTOFF_MARGIN_SECONDS, url),
        model=model))
