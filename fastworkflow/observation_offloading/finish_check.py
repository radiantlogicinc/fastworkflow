"""Finish check (fix-3xts): did each command step of the run's plan run?

When the agent chooses ``finish``, the run's initial plan is read step by step
against the outputs the run archived (``O<step>`` rows of ``offload_evidence``).
Steps found unmet are named in one short note that replaces the finish
observation, and the agent goes back to the loop -- once per agent run, only
with iterations left; a second ``finish`` is not checked.

No model is called. Each numbered plan step is read for the workflow command
it names (the first token that is a declared command). Conditional steps
("only when ...", "if needed", "... (skip when ...)"), negated steps ("do not run X"), answer steps
("present ...", "say ...") and navigation-only steps (the framework's core
commands) are not held to a command. A command step with no successful
archived run of its command in this run's range is unmet: "not executed".
An archived row counts for the command the router resolved its text to (from
the session's action log) when that is known, else for the command its text
names. A step that writes an argument right after its command
(``command key=<...>``) is not satisfied by a run whose text has no arguments.
In a skill phase's check (``values=True``, set by the phase runner) a step
that gives an argument a concrete value (``find_x query=Ann``) is also
satisfied only by a run that carries that value (``step_arguments``,
``satisfies``); a placeholder (``key=<...>``) or prose ("with the uid of the
match") names no value, so any run of the command with arguments does. Other
runs' checks do not compare values.

On whenever the run has a plan; there is no switch. It fails open: any
error -- no plan, no archive -- means no note and the turn finishes as it
would without the check.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: The note costs an iteration and is worthless unless the agent can act on it.
MIN_ITERS_LEFT = 2
NOTE_MAX_BYTES = 1024

KIND_COMMAND = "command"
KIND_CONDITIONAL = "conditional"
KIND_OTHER = "other"
KIND_SKIPPED = "skipped"

VERDICT_UNMET = "unmet"
NOT_EXECUTED = "not executed"

#: Responses the framework itself writes when a command did not run: an
#: exception, a refused or unknown command, a parameter-extraction stop, an
#: ambiguous or misunderstood command. A row whose text carries one is not a run
#: of its command.
FAILURE_MARKERS = (
    "Execution error in ",
    "PARAMETER EXTRACTION ERROR",
    "is not available in the current context",
    "with these arguments is unavailable for this step",
    "Ambiguous intent error for command",
    "I couldn't determine which available command matches",
)

_STEP_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:step\s*)?(\d{1,3})\s*[.):]\s+(.*\S)?\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s+(\S.*)$")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Words that, before a step's command or in its chain (after it, up to the first
#: ``(`` or ``;``), make the step conditional.
_CONDITION_RE = re.compile(
    r"\b(?:if|only|when|whenever|unless|otherwise|optional(?:ly)?|in case|as needed|"
    r"where (?:needed|applicable)|else)\b", re.IGNORECASE)
#: Anywhere in a step, these make it conditional.
_ANYWHERE_CONDITION_RE = re.compile(
    r"\b(?:if needed|if necessary|if required|as needed|optional(?:ly)?|"
    r"skip (?:it )?(?:when|if|unless))\b", re.IGNORECASE)
#: A step's command followed directly by an argument: ``key=<...>`` or ``key=value``.
_STEP_ARGUMENT_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")
#: An argument name and its ``=`` in a step: where one argument's value ends.
_ARGUMENT_KEY_RE = re.compile(r"(?:^|(?<=\s))([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")
#: A value that is a placeholder, not a value: ``<...>``, ``{...}``, ``[...]``, an ellipsis.
_PLACEHOLDER_RE = re.compile(r"[<>{}\[\]]|\.\.\.|\u2026")
#: A value's first token that is an identifier, not a word of a name: at least
#: 16 characters of letters, digits, ``_`` or ``-``, three or more of them digits.
_UID_TOKEN_RE = re.compile(r"^(?=(?:[^0-9]*[0-9]){3})[A-Za-z0-9_-]{16,}$")
#: Words that, before a step's command, negate it ("do not run X", "none of ...").
_NEGATION_RE = re.compile(
    r"\b(?:do not|don't|dont|never|none|not|no|nor|without|avoid|skip)\b", re.IGNORECASE)
#: A step opening with one of these is about the answer, not a command.
_ANSWER_VERBS = frozenset("""
answer report present return say give summarize summarise state tell reply respond
list_nothing compare note identify end stop finish confirm verify conclude explain_answer
offer mention describe include combine format output provide write
""".split())

@dataclass
class PlanStep:
    number: int
    text: str
    kind: str
    command: str = ""
    #: The step writes an argument right after its command; a run of the
    #: command with no arguments at all does not satisfy it.
    wants_arguments: bool = False
    #: The concrete argument values the step gives (``key=value`` right after
    #: its command), normalised (``plain_value``); placeholders are left out.
    #: A run satisfies the step only when it carries each of them.
    arguments: dict = field(default_factory=dict)
    #: Archive rows attributed to this step (successful runs of its command).
    rows: list = field(default_factory=list)


@dataclass
class Verdict:
    step: int
    verdict: str
    missing: str = ""
    next_command: str = ""


@dataclass
class CheckResult:
    steps: list
    unmet: list
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# The plan's steps, their commands, and what ran
# ---------------------------------------------------------------------------

def plain_command_name(name: str) -> str:
    """A routing name (``Context/command``) as the agent types it."""
    return str(name or "").strip().split("/")[-1]


def command_token(command_text: str, names: Iterable[str] = ()) -> str:
    """The command an archived command string ran, unqualified.

    With *names* (lower-case plain command names), the first of the string's
    first three tokens that is one of them (``Context/command args``,
    ``command<arg>``, ``command open_x``); otherwise, or when none is, its
    first token.
    """
    known = set(names)
    tokens = _TOKEN_RE.findall(str(command_text or ""))[:3]
    for token in tokens:
        if token.lower() in known:
            return token
    return tokens[0] if tokens else ""


def parse_plan(plan_text: str, commands: Iterable[str],
               core_commands: Iterable[str] = ()) -> list[PlanStep]:
    """The numbered steps of *plan_text*, each classified.

    *commands* are the workflow's declared command names (plain or qualified);
    *core_commands* the framework's navigation commands, which a step naming
    only them is not held to. Text before the first numbered line and lines
    after a step that are not numbered are part of that step.
    """
    names = {plain_command_name(c).lower() for c in commands if c}
    names.discard("wildcard")
    core = {plain_command_name(c).lower() for c in core_commands if c}
    lines = str(plan_text or "").splitlines()
    # Numbered steps; a plan written only as a bullet list is numbered in order.
    bulleted = not any(_STEP_RE.match(line) for line in lines)
    numbered: list[tuple[int, str]] = []
    for line in lines:
        match = _STEP_RE.match(line)
        bullet = _BULLET_RE.match(line) if bulleted else None
        if match:
            numbered.append((int(match.group(1)), (match.group(2) or "").strip()))
        elif bullet:
            numbered.append((len(numbered) + 1, bullet.group(1).strip()))
        elif numbered and line.strip():
            number, text = numbered[-1]
            numbered[-1] = (number, f"{text} {line.strip()}".strip())
    steps = []
    for number, text in numbered:
        steps.append(_classify(number, text, names, core))
    return steps


def _classify(number: int, text: str, names: set, core: set) -> PlanStep:
    clean = text.replace("`", " ").replace("*", " ")
    tokens = list(_TOKEN_RE.finditer(clean))
    command, position = "", -1
    # The first declared command the step names; a core (navigation) command
    # only when it names no other ("go_up, then open_x" is a step about open_x).
    for wanted in (names - core, core):
        for token in tokens:
            if token.group(0).lower() in wanted:
                command, position = token.group(0), token.start()
                break
        if command:
            break
    before = clean[:position] if position >= 0 else clean
    first_word = tokens[0].group(0).lower() if tokens else ""
    if not command:
        if first_word in _ANSWER_VERBS:
            return PlanStep(number, text, KIND_SKIPPED)
        if _CONDITION_RE.search(clean):
            return PlanStep(number, text, KIND_CONDITIONAL)
        return PlanStep(number, text, KIND_OTHER)
    if _NEGATION_RE.search(before):
        return PlanStep(number, text, KIND_SKIPPED, command)
    if position > 0 and first_word in _ANSWER_VERBS:
        return PlanStep(number, text, KIND_SKIPPED, command)
    if command.lower() in core:
        return PlanStep(number, text, KIND_SKIPPED, command)
    rest = clean[position + len(command):]
    wants_arguments = bool(_STEP_ARGUMENT_RE.match(rest))
    arguments = step_arguments(rest)
    # A condition word in the command's chain (up to the first "(" or ";") makes the
    # step conditional too, so "find_x query=A only when needed" is not run with that
    # prose as its value; a note in parentheses never changes what runs.
    chain = re.split(r"[(;]", rest, maxsplit=1)[0]
    if (_CONDITION_RE.search(before) or _CONDITION_RE.search(chain)
            or _ANYWHERE_CONDITION_RE.search(clean)):
        return PlanStep(number, text, KIND_CONDITIONAL, command, wants_arguments, arguments)
    return PlanStep(number, text, KIND_COMMAND, command, wants_arguments, arguments)


def plain_value(value: Any) -> str:
    """A value as runs and steps are compared: words single-spaced, outer quotes and
    trailing punctuation dropped, case folded. ``None`` is ``""``."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text.strip("\"'`").strip(" .,:").strip("\"'`").casefold()


def step_arguments(rest: str) -> dict[str, str]:
    """The concrete ``key=value`` arguments written right after a step's command.

    *rest* is the step's text after its command. Only a chain of ``key=value``
    that starts there counts; it ends at a ``(`` or ``;`` (prose about the step).
    A value runs to the next ``key=``, and also ends at ``", "`` (prose after
    it: ``scope_uid=X, no filter``). A value whose first token is an
    identifier (``_UID_TOKEN_RE``) is that token alone: it ends at the first
    ``,`` or whitespace. A name of several words (``query=Ann Lee``) is kept whole. A placeholder (``<...>``, ``{...}``, an
    ellipsis) or an empty value is not a value and is left out. Values are
    normalised with ``plain_value``.
    """
    if not _STEP_ARGUMENT_RE.match(rest or ""):
        return {}
    chain = re.split(r"[(;]", rest, maxsplit=1)[0]
    keys = list(_ARGUMENT_KEY_RE.finditer(chain))
    out: dict[str, str] = {}
    for i, match in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(chain)
        raw = _value_text(chain[match.end():end])
        value = plain_value(raw)
        if value and not _PLACEHOLDER_RE.search(raw):
            out[match.group(1)] = value
    return out


def _value_text(raw: str) -> str:
    """The text of one step value: up to ``", "``, or its first token when that
    token is an identifier (see ``step_arguments``)."""
    text = raw.strip()
    first = re.split(r"[\s,]", text, maxsplit=1)[0]
    if _UID_TOKEN_RE.match(first.strip("\"'`")):
        return first
    return re.split(r",\s", text, maxsplit=1)[0]


def run_failed(text: str) -> bool:
    """Whether an archived response says its command did not run."""
    body = str(text or "")
    return any(marker in body for marker in FAILURE_MARKERS)


def _normalized(command_text: str) -> str:
    return " ".join(str(command_text or "").split())


def resolved_command_names(action_log: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Raw command text -> the plain command name it resolved to, from an action log.

    Records carry ``command`` (the text as sent) and ``command_name`` (what the
    router resolved it to). A text seen resolving to more than one command is
    left out: its rows fall back to the name their text gives.
    """
    resolved: dict[str, str] = {}
    conflicting: set[str] = set()
    for record in action_log or ():
        try:
            raw = _normalized(record.get("command"))
            name = plain_command_name(record.get("command_name") or "")
        except AttributeError:
            continue
        if not raw or not name:
            continue
        if resolved.setdefault(raw, name).lower() != name.lower():
            conflicting.add(raw)
    for raw in conflicting:
        resolved.pop(raw, None)
    return resolved


def has_arguments(command_text: str, names: Iterable[str] = ()) -> bool:
    """Whether a command string carries anything after its command name.

    When the command name cannot be found in the text, it is assumed to
    (the conservative reading: the run is not ruled out).
    """
    text = str(command_text or "")
    token = command_token(text, names)
    if not token:
        return False
    match = re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text)
    if match is None:
        return True
    return bool(text[match.end():].strip(" \t\r\n.,;:"))


def _runs_by_command(rows: Sequence[Mapping[str, Any]], names: set,
                     resolved: Optional[Mapping[str, str]] = None) -> dict[str, list]:
    runs: dict[str, list] = {}
    resolved = {_normalized(k): v for k, v in (resolved or {}).items()}
    for row in sorted(rows, key=lambda r: int(r.get("step_index") or 0)):
        if run_failed(row.get("text", "")):
            continue
        raw = row.get("command", "")
        name = resolved.get(_normalized(raw)) or command_token(raw, names)
        runs.setdefault(name.lower(), []).append(row)
    return runs


def _given_inputs(run_input: Any) -> dict[str, Any]:
    """The parameters a command run was given, as ``facts`` reads "given"."""
    if not isinstance(run_input, Mapping):
        return {}
    from fastworkflow.observation_offloading.facts import _given, _not_given_sentinels

    sentinels = _not_given_sentinels()
    return {name: value for name, value in run_input.items() if _given(value, sentinels)}


def _carries(value: str, row: Mapping[str, Any], names: set) -> bool:
    """Whether *row* ran with the step argument *value* (normalised)."""
    if "input" in row:
        # A command run: its structured Input.
        given = _given_inputs(row.get("input"))
        return any(plain_value(v) == value for v in given.values())
    text = plain_value(row.get("command", ""))
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])",
                     text) is not None


def satisfies(step: PlanStep, row: Mapping[str, Any], names: Iterable[str] = ()) -> bool:
    """Whether *row* is a run that does *step* (a run of the step's command).

    *row* is an archive row (``command`` holds the command text as sent) or a
    command run (``command_runs``: ``input`` holds the parameters it ran with).

    - A step that writes an argument right after its command needs a run with
      arguments: a bare run does not satisfy it.
    - Each concrete value the step gives (``step.arguments``) must be one the
      run carries: the parameter of that name when the run has it, else any of
      its parameters (a command run); the value as a word of its text (an
      archive row). Values compare after ``plain_value``. Paging parameters the
      step does not name do not matter, so each page of the step's query
      satisfies it.
    - A placeholder or prose argument names no value: any run with arguments does.
    """
    names = {str(n).lower() for n in names}
    if "input" in row:
        given = _given_inputs(row.get("input"))
        if step.wants_arguments and not given:
            return False
        for key, value in step.arguments.items():
            if key in given:
                if plain_value(given[key]) != value:
                    return False
            elif isinstance(row.get("input"), Mapping) and key in row["input"]:
                return False  # the parameter the step names was not given
            elif not _carries(value, row, names):
                return False
        return True
    if step.wants_arguments and not has_arguments(row.get("command", ""), names):
        return False
    return all(_carries(value, row, names) for value in step.arguments.values())


def _satisfies(step: PlanStep, row: Mapping[str, Any], names: set, values: bool = False) -> bool:
    """Whether *row* does *step*. By default only the bare-run rule: a run with no
    arguments does not satisfy a step that writes one. With *values* (a skill phase's
    check), also the step's concrete values (``satisfies``)."""
    if values:
        return satisfies(step, row, names)
    return not step.wants_arguments or has_arguments(row.get("command", ""), names)


def attribute_rows(steps: Sequence[PlanStep], rows: Sequence[Mapping[str, Any]],
                   commands: Iterable[str] = (),
                   earlier_rows: Sequence[Mapping[str, Any]] = (),
                   resolved: Optional[Mapping[str, str]] = None,
                   values: bool = False) -> None:
    """Give each command (and conditional command) step the successful rows of its command.

    Steps naming the same command take its runs in order: the k-th such step
    gets the k-th run, and the last such step also gets any runs left over (a
    paged listing is one step run several times). Rows are archive rows:
    ``command`` (the command text that ran, tool arguments folded in; see
    ``compact.archive_step``), ``step_index``, ``text``.

    *resolved* maps a row's command text to the command it resolved to (see
    ``resolved_command_names``); a row whose text it lacks counts for the
    command its text names. A step that writes an argument after its command
    takes only runs that carry arguments -- with *values* (a skill phase's
    check), only runs that carry each concrete value it gives (``satisfies``):
    those steps pick first, in order, then the others take what is left, in order.

    *earlier_rows* are the turn's rows from before this run (an earlier
    sub-task). A step left without a row that comes before the run's first
    attributed step may take one of them: a run that starts where the previous
    one left off has its setup steps (entering a context, finding a record)
    already done.
    """
    names = {plain_command_name(c).lower() for c in commands if c}
    runs = _runs_by_command(rows, names, resolved)
    by_command: dict[str, list[PlanStep]] = {}
    for step in steps:
        if step.kind in (KIND_COMMAND, KIND_CONDITIONAL) and step.command:
            by_command.setdefault(step.command.lower(), []).append(step)
    for command, group in by_command.items():
        available = runs.get(command, [])
        used: set[int] = set()
        # Steps that write an argument first, so a bare run never takes the
        # place of a run with arguments they need; with none, this is the
        # k-th step / k-th run pairing.
        for picky in (True, False):
            for step in group:
                if step.wants_arguments != picky:
                    continue
                for position, row in enumerate(available):
                    if position not in used and _satisfies(step, row, names, values):
                        step.rows = [row]
                        used.add(position)
                        break
        if group:
            last = group[-1]
            last.rows.extend(row for position, row in enumerate(available)
                             if position not in used and _satisfies(last, row, names, values))
    first_done = next((s.number for s in steps if s.rows), None)
    if earlier_rows and first_done is not None:
        earlier = _runs_by_command(earlier_rows, names, resolved)
        for step in steps:
            if step.number >= first_done:
                break
            if step.kind in (KIND_COMMAND, KIND_CONDITIONAL) and step.command and not step.rows:
                previous = [row for row in earlier.get(step.command.lower(), [])
                            if _satisfies(step, row, names, values)]
                if previous:
                    step.rows = [previous[-1]]


def unmet_steps(steps: Sequence[PlanStep]) -> list[Verdict]:
    """Command steps with no successful run of their command: unmet, not executed."""
    return [Verdict(step.number, VERDICT_UNMET, missing=NOT_EXECUTED,
                    next_command=step.command)
            for step in steps if step.kind == KIND_COMMAND and not step.rows]


# ---------------------------------------------------------------------------
# The note and the whole check
# ---------------------------------------------------------------------------

def build_note(unmet: Sequence[Verdict]) -> Optional[str]:
    """At most ``NOTE_MAX_BYTES``: the unmet steps and what to do about each; None when none."""
    if not unmet:
        return None
    lines = ["Finish check: not every plan step is done yet."]
    for verdict in sorted(unmet, key=lambda v: v.step):
        lines.append(f"- Step {verdict.step} ({verdict.next_command}) was not run: run it now.")
    lines.append("Do not repeat a change already made. Then finish; if a step cannot be done,"
                 " say so instead of claiming it.")
    note = "\n".join(lines)
    while len(note.encode("utf-8")) > NOTE_MAX_BYTES and len(lines) > 3:
        lines.pop(-2)
        note = "\n".join(lines[:-1] + ["- (more steps unmet)", lines[-1]])
    return note.encode("utf-8")[:NOTE_MAX_BYTES].decode("utf-8", "ignore")


def check(plan_text: str, rows: Sequence[Mapping[str, Any]], commands: Iterable[str],
          *, core_commands: Iterable[str] = (),
          earlier_rows: Sequence[Mapping[str, Any]] = (),
          resolved: Optional[Mapping[str, str]] = None,
          values: bool = False) -> CheckResult:
    """The plan's steps, the command steps left unmet, and the note naming them.

    *resolved*: raw command text -> resolved command name (``resolved_command_names``).
    """
    commands = list(commands)
    core_commands = list(core_commands)
    steps = parse_plan(plan_text, commands, core_commands)
    attribute_rows(steps, rows, commands + core_commands, earlier_rows, resolved, values)
    unmet = unmet_steps(steps)
    return CheckResult(steps=steps, unmet=unmet, note=build_note(unmet))


# ---------------------------------------------------------------------------
# Runtime: the archive read
# ---------------------------------------------------------------------------

def review_finish(*, plan_text: str, floor: int, iters_left: int,
                  commands: Iterable[str], core_commands: Iterable[str] = (),
                  resolved: Optional[Mapping[str, str]] = None,
                  values: bool = False) -> Optional[str]:
    """The note for this run's finish, or None. Reads the run's archived rows; never raises.

    *resolved* maps raw command text to the command it resolved to; rows whose
    text it lacks count for the command their text names. *values*: a step's
    concrete argument values must match (a skill phase's check; ``satisfies``).
    """
    from fastworkflow.observation_offloading.state import (
        current_scope, durable_archive, record_event)

    began = time.monotonic()
    scope = store = None
    event: dict[str, Any] = {"kind": "finish_check"}
    try:
        if iters_left < MIN_ITERS_LEFT:
            event["reason"] = "no room to act"
            return None
        if not str(plan_text or "").strip():
            event["reason"] = "no plan"
            return None
        scope, store = current_scope(), durable_archive()
        if store is None:
            # Nothing to read the runs from: every step would look not run.
            event["reason"] = "no archive"
            return None
        stored = store.list(scope, timeout=2.0)
        rows = [r for r in stored if int(r.get("step_index") or 0) >= int(floor or 0)]
        earlier = [r for r in stored if int(r.get("step_index") or 0) < int(floor or 0)]
        result = check(plan_text, rows, commands, core_commands=core_commands,
                       earlier_rows=earlier, resolved=resolved, values=values)
        event.update({
            "reason": "checked",
            "plan_sha256": hashlib.sha256(str(plan_text).encode("utf-8")).hexdigest()[:16],
            "steps": len(result.steps),
            "unmet_not_executed": [v.step for v in result.unmet],
            "note_bytes": len((result.note or "").encode("utf-8")),
        })
        return result.note
    except Exception as error:  # noqa: BLE001 - the check never fails a turn
        event.update({"reason": "error", "error": f"{type(error).__name__}"})
        logger.warning("finish check skipped: %s: %s", type(error).__name__, error)
        return None
    finally:
        event["duration_ms"] = round((time.monotonic() - began) * 1000)
        try:
            record_event(event, scope=scope, store=store)
        except Exception:  # noqa: BLE001
            pass
