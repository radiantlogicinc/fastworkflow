"""Finish check (fix-3xts): was each step of the run's plan satisfied, not only run?

When the agent chooses ``finish``, the run's initial plan is read step by step
against the outputs the run archived (``O<step>`` rows of ``offload_evidence``).
Steps found unmet are named in one short note that replaces the finish
observation, and the agent goes back to the loop -- once per agent run, only
with iterations left; a second ``finish`` is not checked.

Two passes, both here as pure functions so an offline measurement runs exactly
this code:

A. No model. Each numbered plan step is read for the workflow command it names
   (the first token that is a declared command). Conditional steps ("only when
   ...", "if needed"), negated steps ("do not run X"), answer steps ("present
   ...", "say ...") and navigation-only steps (the framework's core commands)
   are not held to a command. A command step with no successful archived run
   of its command in this run's range is unmet: "not executed".

B. One model call. The executed command steps whose text asks for something
   beyond running the command (each with the output(s) attributed to it, cut
   to head and tail plus the lines that name the step's words) and the
   conditional and command-less steps (against a one-line-per-command ledger)
   are judged ``satisfied`` / ``unmet`` / ``not_applicable``. The model must
   quote what is missing from the step's own text; code then refuses an
   ``unmet`` whose quote is not in the step text, names the step's command,
   opens with an answer verb, names none of the step's asked-for words, or
   whose asked-for words mostly (``OUTPUT_COVERAGE_VETO``) appear in the
   step's full output (it is probably there), and blanks a suggested command
   that is not a declared command or that already ran for that step (no write
   is ever suggested twice).

Measured offline (fix-3xts, one workflow, its agent model as the judge): the
filters were tuned on one third of the recorded turns and scored once on the
rest. Those figures need not transfer to another workflow or model.

OFF unless ``FW_FINISH_CHECK`` is set: ``1``/``on``/``true``/``full`` runs both
passes, ``commands`` runs pass A alone (no model call). The model is
``LLM_FINISH_CHECK`` (with ``LITELLM_API_KEY_FINISH_CHECK``) when set, else the
agent's ``LLM_AGENT``. The prompt is bounded by ``context_budget.FINISH_CHECK``.
It fails open: any error -- no plan, no archive, a model or parse failure --
means no note and the turn finishes as it would without the check.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

CHECK_ENV = "FW_FINISH_CHECK"
MODEL_ENV = "LLM_FINISH_CHECK"
KEY_ENV = "LITELLM_API_KEY_FINISH_CHECK"
#: The note costs an iteration and is worthless unless the agent can act on it.
MIN_ITERS_LEFT = 2
NOTE_MAX_BYTES = 1024
#: One output's share of the prompt before it is cut to head + tail.
OBS_MAX_BYTES = 4096
#: Lines of a cut output that name one of the step's words, kept beside head and tail.
GREP_LINES_MAX = 12
LEDGER_LINE_BYTES = 160
#: A command step is put to the model only when its text names at least this
#: many content words beyond its command and arguments.
ASKED_WORDS_MIN = 2
#: An unmet whose quoted words appear in the step's output at least this often
#: (as a share) is refused: what it calls missing is probably there.
OUTPUT_COVERAGE_VETO = 0.5
MODEL_MAX_TOKENS = 4096
MODEL_TIMEOUT_SECONDS = 60

KIND_COMMAND = "command"
KIND_CONDITIONAL = "conditional"
KIND_OTHER = "other"
KIND_SKIPPED = "skipped"

VERDICT_SATISFIED = "satisfied"
VERDICT_UNMET = "unmet"
VERDICT_NOT_APPLICABLE = "not_applicable"
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
_WORD_RE = re.compile(r"[a-z0-9]+")
#: Words that, before a step's command, make the step conditional.
_CONDITION_RE = re.compile(
    r"\b(?:if|only|when|whenever|unless|otherwise|optional(?:ly)?|in case|as needed|"
    r"where (?:needed|applicable)|else)\b", re.IGNORECASE)
#: Anywhere in a step, these make it conditional.
_ANYWHERE_CONDITION_RE = re.compile(
    r"\b(?:if needed|if necessary|if required|as needed|optional(?:ly)?)\b", re.IGNORECASE)
#: Words that, before a step's command, negate it ("do not run X", "none of ...").
_NEGATION_RE = re.compile(
    r"\b(?:do not|don't|dont|never|none|not|no|nor|without|avoid|skip)\b", re.IGNORECASE)
#: A step opening with one of these is about the answer, not a command.
_ANSWER_VERBS = frozenset("""
answer report present return say give summarize summarise state tell reply respond
list_nothing compare note identify end stop finish confirm verify conclude explain_answer
offer mention describe include combine format output provide write
""".split())
_STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the
their them then there these this those to was were which with its own each every
any all one first next last same other its step steps show list get open run use
using uid uids id ids match matches item items record records value values
would could should will can may might what why how whether about before after behind
against available full whole read now them they its it's his her him whose which
""".split())


@dataclass
class PlanStep:
    number: int
    text: str
    kind: str
    command: str = ""
    #: Archive rows attributed to this step (successful runs of its command).
    rows: list = field(default_factory=list)


@dataclass
class Verdict:
    step: int
    verdict: str
    missing: str = ""
    evidence: str = ""
    next_command: str = ""
    source: str = "model"  # "pass_a" or "model"


@dataclass
class CheckResult:
    steps: list
    unmet: list
    vetoes: list = field(default_factory=list)
    note: Optional[str] = None
    prompt: str = ""
    raw_response: str = ""
    error: str = ""
    model_called: bool = False


# ---------------------------------------------------------------------------
# Pass A: the plan's steps, their commands, and what ran
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
    if _CONDITION_RE.search(before) or _ANYWHERE_CONDITION_RE.search(clean):
        return PlanStep(number, text, KIND_CONDITIONAL, command)
    return PlanStep(number, text, KIND_COMMAND, command)


def run_failed(text: str) -> bool:
    """Whether an archived response says its command did not run."""
    body = str(text or "")
    return any(marker in body for marker in FAILURE_MARKERS)


def _runs_by_command(rows: Sequence[Mapping[str, Any]], names: set) -> dict[str, list]:
    runs: dict[str, list] = {}
    for row in sorted(rows, key=lambda r: int(r.get("step_index") or 0)):
        if run_failed(row.get("text", "")):
            continue
        runs.setdefault(command_token(row.get("command", ""), names).lower(), []).append(row)
    return runs


def attribute_rows(steps: Sequence[PlanStep], rows: Sequence[Mapping[str, Any]],
                   commands: Iterable[str] = (),
                   earlier_rows: Sequence[Mapping[str, Any]] = ()) -> None:
    """Give each command (and conditional command) step the successful rows of its command.

    Steps naming the same command take its runs in order: the k-th such step
    gets the k-th run, and the last such step also gets any runs left over (a
    paged listing is one step run several times). Rows are archive rows:
    ``command`` (the command text as sent), ``step_index``, ``text``.

    *earlier_rows* are the turn's rows from before this run (an earlier
    sub-task). A step left without a row that comes before the run's first
    attributed step may take one of them: a run that starts where the previous
    one left off has its setup steps (entering a context, finding a record)
    already done.
    """
    names = {plain_command_name(c).lower() for c in commands if c}
    runs = _runs_by_command(rows, names)
    by_command: dict[str, list[PlanStep]] = {}
    for step in steps:
        if step.kind in (KIND_COMMAND, KIND_CONDITIONAL) and step.command:
            by_command.setdefault(step.command.lower(), []).append(step)
    for command, group in by_command.items():
        available = runs.get(command, [])
        for position, step in enumerate(group):
            if position < len(available):
                step.rows = [available[position]]
        if len(available) > len(group) and group:
            group[-1].rows.extend(available[len(group):])
    first_done = next((s.number for s in steps if s.rows), None)
    if earlier_rows and first_done is not None:
        earlier = _runs_by_command(earlier_rows, names)
        for step in steps:
            if step.number >= first_done:
                break
            if step.kind in (KIND_COMMAND, KIND_CONDITIONAL) and step.command and not step.rows:
                previous = earlier.get(step.command.lower())
                if previous:
                    step.rows = [previous[-1]]


def pass_a(steps: Sequence[PlanStep]) -> list[Verdict]:
    """Command steps with no successful run of their command: unmet, not executed."""
    return [Verdict(step.number, VERDICT_UNMET, missing=NOT_EXECUTED,
                    next_command=step.command, source="pass_a")
            for step in steps if step.kind == KIND_COMMAND and not step.rows]


# ---------------------------------------------------------------------------
# Pass B: the model's satisfaction question
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You check an agent's work before it gives its answer. The agent was given a numbered plan and ran commands; each command's output is shown. For each plan step listed, decide:
- "satisfied": the output(s) for the step contain what the step asks for, or clearly show it does not exist (an empty result, "none", 0 found, or a total is an answer).
- "unmet": the step asks for a specific piece of information or action, and the outputs neither contain it nor say it is absent.
- "not_applicable": the step's condition did not hold, or the step only says how to answer.
Judge only what the step's own words ask for. Do not require anything the step does not name. An output that has the field, section or list the step names (even under another label) satisfies it. Do not mark a step unmet because the output is long, or because only a page of a list is shown when the step does not ask for all of it.
For every "unmet" step, copy into "missing" the exact words of the step text that name the missing information (copied, not paraphrased; not the command name or its arguments), and into "next_command" one command name from the command list that would get it, or "" if none would.
Reply with JSON only: {"steps": [{"step": <number>, "verdict": "satisfied|unmet|not_applicable", "missing": "", "evidence": "<a few words quoted from the output>", "next_command": ""}]}"""


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(str(text or "").lower())
            if len(w) >= 3 and w not in _STOPWORDS]


def _normalized(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text or "").lower()))


def _cut_bytes(text: str, limit: int, *, tail: bool = False) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    piece = data[-limit:] if tail else data[:limit]
    return piece.decode("utf-8", "ignore")


def output_excerpt(text: str, step_text: str, max_bytes: int = OBS_MAX_BYTES) -> str:
    """*text* whole when it fits *max_bytes*; else its head, the lines naming the step's words, and its tail."""
    body = str(text or "")
    if len(body.encode("utf-8")) <= max_bytes:
        return body
    half = max_bytes // 2
    head = _cut_bytes(body, half * 3 // 4)
    tail = _cut_bytes(body, half // 2, tail=True)
    words = {w for w in _content_words(step_text) if len(w) >= 4}
    middle = body[len(head):len(body) - len(tail)]
    hits = []
    for line in middle.splitlines():
        lowered = line.lower()
        if words and any(w in lowered for w in words):
            hits.append(_cut_bytes(line.strip(), 200))
            if len(hits) >= GREP_LINES_MAX:
                break
    parts = [head, "[... cut ...]"]
    if hits:
        parts += ["[lines naming the step's words:]", *hits, "[... cut ...]"]
    parts.append(tail)
    return "\n".join(parts)


def ledger(rows: Sequence[Mapping[str, Any]]) -> str:
    """One line per archived command of the run: alias, command, first line of its output."""
    lines = []
    for row in sorted(rows, key=lambda r: int(r.get("step_index") or 0)):
        first = str(row.get("text", "")).strip().splitlines()[:1]
        status = "FAILED: " if run_failed(row.get("text", "")) else ""
        line = f"O{row.get('step_index')} {row.get('command', '')} -> {status}{first[0] if first else ''}"
        lines.append(_cut_bytes(line, LEDGER_LINE_BYTES))
    return "\n".join(lines)


_ARG_RE = re.compile(r"<[^>]*>|\S+=(?:<[^>]*>|\S*)|`[^`]*`")
_NAME_RE = re.compile(r"(?<=[\w,;:)] )[A-Z][\w.'-]*")


def asked_words(step: PlanStep) -> list[str]:
    """The content words a step asks for beyond naming its command, its arguments
    and the records it names (capitalised words after the first)."""
    text = _ARG_RE.sub(" ", step.text)
    if step.command:
        text = re.sub(rf"\b{re.escape(step.command)}\b", " ", text, flags=re.IGNORECASE)
    return _content_words(_NAME_RE.sub(" ", text))


def judged_steps(steps: Sequence[PlanStep]) -> list[PlanStep]:
    """The steps pass B asks about.

    An executed command step only when its text asks for something beyond
    running the command (a step that just names a command is done when the
    command ran); every conditional and command-less step.
    """
    return [s for s in steps
            if (s.kind == KIND_COMMAND and s.rows and len(asked_words(s)) >= ASKED_WORDS_MIN)
            or s.kind in (KIND_CONDITIONAL, KIND_OTHER)]


def build_prompt(request: str, steps: Sequence[PlanStep], rows: Sequence[Mapping[str, Any]],
                 commands: Iterable[str], max_bytes: int) -> str:
    """The user prompt of pass B, at most about *max_bytes* (outputs shrink to fit)."""
    to_judge = judged_steps(steps)
    names = sorted({plain_command_name(c) for c in commands if c} - {"wildcard"})
    fixed = [
        "Request:", _cut_bytes(str(request or "").strip(), 1500), "",
        "Plan:", *[f"{s.number}. {s.text}" for s in steps], "",
        "Commands run (O<step> command -> first line of output):",
        _cut_bytes(ledger(rows), max(512, max_bytes // 6)), "",
        "Command list: " + ", ".join(names), "",
    ]
    fixed_text = "\n".join(fixed)
    evidence_rows = sum(len(s.rows) for s in to_judge if s.rows) or 1
    room = max(max_bytes - len(fixed_text.encode("utf-8")) - 400, 512)
    per_output = max(min(OBS_MAX_BYTES, room // evidence_rows), 256)
    blocks = []
    for step in to_judge:
        blocks.append(f"### Step {step.number}: {step.text}")
        if step.rows:
            for row in step.rows:
                blocks.append(f"Output O{row.get('step_index')} ({row.get('command', '')}):")
                blocks.append(output_excerpt(str(row.get("text", "")), step.text, per_output))
        else:
            blocks.append("(no single command: judge it from the commands run above)")
        blocks.append("")
    numbers = ", ".join(str(s.number) for s in to_judge)
    return (fixed_text + "\n".join(blocks)
            + f"\nJudge these steps: {numbers}. Reply with the JSON only.")


def parse_response(text: str) -> list[dict]:
    """The step objects of the model's JSON reply (the last JSON object that has them)."""
    body = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    candidates = []
    for match in re.finditer(r"\{", body):
        depth = 0
        for end in range(match.start(), len(body)):
            if body[end] == "{":
                depth += 1
            elif body[end] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(body[match.start():end + 1])
                    break
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            return [item for item in parsed["steps"] if isinstance(item, dict)]
    raise ValueError("finish check reply has no steps JSON")


def filter_verdicts(items: Sequence[Mapping[str, Any]], steps: Sequence[PlanStep],
                    rows: Sequence[Mapping[str, Any]], commands: Iterable[str]
                    ) -> tuple[list[Verdict], list[dict]]:
    """``(unmet verdicts kept, vetoes)``: the anti-invention rules, in code.

    An ``unmet`` is kept only for a judged step, only when its ``missing`` quote
    is (normalized) part of the step's text, and only when not every content
    word of that quote appears in the step's full output(s) -- for a command-less
    step, in the run's outputs. A ``next_command`` that is not a declared
    command, or that already ran for the step, is blanked.
    """
    judged = {s.number: s for s in judged_steps(steps)}
    names = {plain_command_name(c).lower(): plain_command_name(c) for c in commands if c}
    names.pop("wildcard", None)
    all_text = "\n".join(str(r.get("text", "")) for r in rows).lower()
    kept, vetoes = [], []
    seen = set()
    for item in items:
        if str(item.get("verdict", "")).strip().lower() != VERDICT_UNMET:
            continue
        try:
            number = int(str(item.get("step", "")).strip().rstrip("."))
        except ValueError:
            vetoes.append({"step": item.get("step"), "reason": "bad step number"})
            continue
        step = judged.get(number)
        if step is None or number in seen:
            vetoes.append({"step": number, "reason": "not a judged step"})
            continue
        missing = str(item.get("missing", "") or "").strip()
        if not missing or _normalized(missing) not in _normalized(step.text):
            vetoes.append({"step": number, "reason": "quote not in step", "missing": missing[:200]})
            continue
        opening = _TOKEN_RE.match(missing.strip("\"'` "))
        if opening and opening.group(0).lower() in _ANSWER_VERBS:
            vetoes.append({"step": number, "reason": "quote is about the answer", "missing": missing[:200]})
            continue
        if step.command and re.search(rf"\b{re.escape(step.command)}\b", missing, re.IGNORECASE):
            vetoes.append({"step": number, "reason": "quote names the command", "missing": missing[:200]})
            continue
        asked = set(asked_words(step)) if step.command else set(_content_words(step.text))
        words = [w for w in _content_words(_ARG_RE.sub(" ", missing)) if w in asked]
        if not words:
            vetoes.append({"step": number, "reason": "quote names nothing asked", "missing": missing[:200]})
            continue
        output = ("\n".join(str(r.get("text", "")) for r in step.rows).lower()
                  if step.rows else all_text)
        present = sum(1 for w in words if w in output)
        if present >= OUTPUT_COVERAGE_VETO * len(words):
            vetoes.append({"step": number, "reason": "words in output", "missing": missing[:200]})
            continue
        suggestion = names.get(plain_command_name(str(item.get("next_command", "") or "")).lower(), "")
        if suggestion and step.rows and suggestion.lower() in {
                command_token(r.get("command", ""), names).lower() for r in step.rows}:
            suggestion = ""
        seen.add(number)
        kept.append(Verdict(number, VERDICT_UNMET, missing=missing[:200],
                            evidence=str(item.get("evidence", "") or "")[:200],
                            next_command=suggestion))
    return kept, vetoes


# ---------------------------------------------------------------------------
# The note
# ---------------------------------------------------------------------------

def build_note(unmet: Sequence[Verdict], steps: Sequence[PlanStep]) -> Optional[str]:
    """At most ``NOTE_MAX_BYTES``: the unmet steps and what to do about each; None when none."""
    if not unmet:
        return None
    by_number = {s.number: s for s in steps}
    lines = ["Finish check: not every plan step is done yet."]
    for verdict in sorted(unmet, key=lambda v: v.step):
        step = by_number.get(verdict.step)
        where = ", ".join(f"O{r.get('step_index')}" for r in (step.rows if step else [])[:3])
        if verdict.missing == NOT_EXECUTED:
            lines.append(f"- Step {verdict.step} ({verdict.next_command}) was not run: run it now.")
        else:
            todo = (f"run {verdict.next_command}" if verdict.next_command
                    else "use another command that gets it")
            lines.append(f"- Step {verdict.step} asks for \"{verdict.missing}\""
                         f"{' and ' + where + ' does not show it' if where else ', not found in the outputs'}"
                         f": {todo}, or say in the answer that it is unavailable.")
    lines.append("Do not repeat a change already made. Then finish; if a step cannot be done,"
                 " say so instead of claiming it.")
    note = "\n".join(lines)
    while len(note.encode("utf-8")) > NOTE_MAX_BYTES and len(lines) > 3:
        lines.pop(-2)
        note = "\n".join(lines[:-1] + ["- (more steps unmet)", lines[-1]])
    return _cut_bytes(note, NOTE_MAX_BYTES)


# ---------------------------------------------------------------------------
# The whole check, pure: the model is a callable
# ---------------------------------------------------------------------------

def check(plan_text: str, rows: Sequence[Mapping[str, Any]], commands: Iterable[str],
          *, core_commands: Iterable[str] = (), request: str = "",
          earlier_rows: Sequence[Mapping[str, Any]] = (),
          ask: Optional[Callable[[str, str], str]] = None,
          max_bytes: int = 24_576) -> CheckResult:
    """Pass A, then pass B through *ask(system, user) -> reply* when given.

    Never raises for a model failure: the result's ``error`` says what went
    wrong and only pass A's verdicts stand.
    """
    commands = list(commands)
    core_commands = list(core_commands)
    steps = parse_plan(plan_text, commands, core_commands)
    attribute_rows(steps, rows, commands + core_commands, earlier_rows)
    unmet = pass_a(steps)
    result = CheckResult(steps=steps, unmet=unmet)
    if ask is not None and judged_steps(steps):
        prompt = build_prompt(request, steps, rows, commands, max_bytes)
        result.prompt = prompt
        try:
            result.model_called = True
            reply = ask(SYSTEM_PROMPT, prompt)
            result.raw_response = str(reply or "")
            kept, vetoes = filter_verdicts(parse_response(reply), steps, rows, commands)
            result.unmet = unmet + kept
            result.vetoes = vetoes
        except Exception as error:  # noqa: BLE001 - the model pass fails open
            result.error = f"{type(error).__name__}: {error}"[:300]
    result.note = build_note(result.unmet, steps)
    return result


# ---------------------------------------------------------------------------
# Runtime: the switch, the model, the archive read
# ---------------------------------------------------------------------------

MODE_FULL = "full"
MODE_COMMANDS = "commands"


def mode() -> Optional[str]:
    """``MODE_FULL`` (both passes), ``MODE_COMMANDS`` (pass A only) or None (off).

    ``FW_FINISH_CHECK``: ``1``/``on``/``true``/``full`` for both passes,
    ``commands`` for the command pass alone (no model call); anything else is off.
    """
    from fastworkflow import context_budget

    value = context_budget.env_value(CHECK_ENV).lower()
    if value in ("1", "on", "true", "yes", MODE_FULL):
        return MODE_FULL
    if value == MODE_COMMANDS:
        return MODE_COMMANDS
    return None


def model_ask(system: str, user: str, *, record: Optional[dict] = None) -> str:
    """The configured finish-check model's reply to one prompt.

    *record*, when given, receives the model id, token usage, cost and latency.
    """
    from fastworkflow import context_budget
    from fastworkflow.utils.dspy_utils import get_lm

    model_env, key_env = ((MODEL_ENV, KEY_ENV) if context_budget.env_value(MODEL_ENV)
                          else (context_budget.AGENT_MODEL_ENV, "LITELLM_API_KEY_AGENT"))
    lm = get_lm(model_env, key_env, temperature=0, max_tokens=MODEL_MAX_TOKENS,
                timeout=MODEL_TIMEOUT_SECONDS, num_retries=1, cache=False)
    began = time.monotonic()
    replies = lm(messages=[{"role": "system", "content": system},
                           {"role": "user", "content": user}])
    if record is not None:
        history = lm.history[-1] if lm.history else {}
        usage = history.get("usage") or {}
        record.update({
            "model": lm.model,
            "latency_ms": round((time.monotonic() - began) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": history.get("cost"),
        })
    reply = replies[0] if replies else ""
    if isinstance(reply, dict):
        reply = reply.get("text") or ""
    return str(reply or "")


def review_finish(*, plan_text: str, request: str, floor: int, iters_left: int,
                  commands: Iterable[str], core_commands: Iterable[str] = (),
                  use_model: bool = True,
                  ask: Optional[Callable[[str, str], str]] = None) -> Optional[str]:
    """The note for this run's finish, or None. Reads the run's archived rows; never raises."""
    from fastworkflow import context_budget
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
        stored = store.list(scope, timeout=2.0) if store is not None else []
        rows = [r for r in stored if int(r.get("step_index") or 0) >= int(floor or 0)]
        earlier = [r for r in stored if int(r.get("step_index") or 0) < int(floor or 0)]
        usage: dict[str, Any] = {}
        result = check(plan_text, rows, commands, core_commands=core_commands,
                       earlier_rows=earlier, request=request,
                       ask=None if not use_model else ask if ask is not None else (
                           lambda system, user: model_ask(system, user, record=usage)),
                       max_bytes=context_budget.finish_check_max_bytes())
        event.update({
            "reason": "checked",
            "plan_sha256": hashlib.sha256(str(plan_text).encode("utf-8")).hexdigest()[:16],
            "steps": len(result.steps),
            "unmet_not_executed": [v.step for v in result.unmet if v.source == "pass_a"],
            "unmet_by_model": [v.step for v in result.unmet if v.source == "model"],
            "vetoes": [{"step": v.get("step"), "reason": v.get("reason")} for v in result.vetoes],
            "model_called": result.model_called,
            "prompt_bytes": len(result.prompt.encode("utf-8")),
            "error": result.error or None,
            "note_bytes": len((result.note or "").encode("utf-8")),
            **({"usage": usage} if usage else {}),
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
