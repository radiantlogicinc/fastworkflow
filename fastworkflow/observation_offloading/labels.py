"""Useful pointers to persisted observations, without payload-sized metadata."""
from __future__ import annotations

import math
import re

CHARS_PER_TOKEN = 4
OFFLOAD_MARK = "Offloaded observation "
#: The earlier label wording opened with an instruction to search, read at the
#: exact moment the agent decides whether to; it is still matched so a
#: trajectory recorded before the change resumes.
LABEL_RE = re.compile(
    r"^(?:Offloaded observation |Use search_memory tool to search inside Observation )"
    r"(O[1-9]\d*) returned by ")
#: The A1 handle line, with the ido-8ps.13 context clause optional. The clause
#: can never contain a parenthesis or a newline (``context_clause`` removes
#: both), so the closing ``)`` is unambiguous and a line printed before the
#: clause existed still matches.
ALIAS_LINE_RE = re.compile(
    r"^Observation (O[1-9]\d*) \(execute_workflow_query(?:, in ([^()\n]*))?\)\n")
#: Longest instance identity printed. A uid plus a display name, not a payload.
MAX_INSTANCE_LABEL_CHARS = 80
#: Longest context name printed, for the same reason.
MAX_CONTEXT_NAME_CHARS = 60
SEARCH_ANSWER_KEY_RE = re.compile(r"^(O[1-9]\d*)#a([1-9]\d*)$")
#: What a command response is quoted with when its own first line is shaped
#: like a line this module prints (``ido-cku``). Two characters, visible, and
#: self-escaping: see ``escape_response``.
RESPONSE_ESCAPE = "> "


def search_answer_key(alias: str, sequence: int) -> str:
    r"""Archive key for one complete search answer, under the searched handle.

    Deliberately NOT an O alias. The agent-visible ``O`` namespace is execute
    ordinals only, and ``search_memory`` validates its ``alias`` argument
    against ``O[1-9]\d*``, so this key can never be passed back as a handle: a
    bounded answer's marking names a record, not a searchable observation. The
    searched alias is kept as the prefix so the archived answer is filed under
    the observation that produced it, and ``sequence`` separates repeated
    searches of the same observation within one scope.
    """
    if not sequence >= 1:
        raise ValueError("search answer sequence must be a positive integer")
    return f"{alias}#a{sequence}"


def is_search_answer_key(key: str) -> bool:
    return SEARCH_ANSWER_KEY_RE.match(key) is not None


def _clipped(value: str, limit: int) -> str:
    """*value* with no parenthesis, no newline, collapsed spaces, capped."""
    cleaned = " ".join(str(value or "").replace("(", " ").replace(")", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def context_clause(context_name: str, instance_label: str = "") -> str:
    """The ``<ContextName> <instance label>`` clause, or ``""``.

    The single place a context identity is made printable, so the format, the
    character rules the regex depends on and the length cap have one
    definition. An empty context name yields an empty clause (the root context
    prints no clause at all); a context with no declared identity yields its
    name alone -- an identifier is never invented to fill the gap.
    """
    name = _clipped(context_name, MAX_CONTEXT_NAME_CHARS)
    if not name:
        return ""
    label = _clipped(instance_label, MAX_INSTANCE_LABEL_CHARS)
    return f"{name} {label}" if label else name


def alias_line(alias: str, context: str = "") -> str:
    """The canonical handle line printed above an inline execute observation.

    This is the only identifier the agent is ever asked to pass to
    search_memory, so it must read the same here and in an offload label. It is
    presentation only: archived text never carries it (see strip_alias_line).

    ``context`` is the clause from ``context_clause`` -- the context the command
    RAN IN and, where the workflow declares one, that context's instance
    identity. It is empty at the root context, and the line is then
    byte-for-byte the bare alias line.
    """
    clause = _clipped(context, MAX_CONTEXT_NAME_CHARS + MAX_INSTANCE_LABEL_CHARS + 1)
    suffix = f", in {clause}" if clause else ""
    return f"Observation {alias} (execute_workflow_query{suffix})\n"


def _escape_depth(text: str) -> int | None:
    """Escape markers standing between the start of *text* and a line of ours.

    ``0`` for a text that already opens with a handle line or an offload label,
    ``n`` for one behind ``n`` markers, ``None`` for anything else -- which is
    the overwhelmingly common case and the one that is never escaped at all.
    Counting the markers is what makes the escape reversible: a response that
    genuinely begins with the marker is escaped once more, so removing exactly
    one marker always lands back on the response that was given.
    """
    depth = 0
    rest = text
    while rest.startswith(RESPONSE_ESCAPE):
        rest = rest[len(RESPONSE_ESCAPE):]
        depth += 1
    if ALIAS_LINE_RE.match(rest) or LABEL_RE.match(rest):
        return depth
    return None


def escape_response(text: str) -> str:
    """*text* made safe to print underneath a handle line.

    A command response whose own first line is shaped like a handle line or an
    offload label would otherwise be read back as one: the framework would
    trust a name the backend printed, and a response opening "Observation O7
    (execute_workflow_query)" would be taken for observation seven. Only this
    module may name an alias, so such a response is quoted with a marker. The
    marker is visible on purpose -- the agent should see that the line below
    the handle line is part of the output, not a second header -- and the quote
    is undone by ``strip_alias_line`` before anything is stored or hashed.
    """
    return RESPONSE_ESCAPE + text if _escape_depth(text) is not None else text


def unescape_response(text: str) -> str:
    """The response ``escape_response`` was given back, exactly."""
    depth = _escape_depth(text)
    return text[len(RESPONSE_ESCAPE):] if depth else text


def annotated_observation(alias: str, context: str = "", text: str = "") -> str:
    """The observation as the agent sees it: our handle line, then the response.

    The one place the two are joined, so the compaction hook that prints the
    line on a completed step and the rehydration that re-prints it over an
    archived response escape a shape-colliding response identically.
    """
    return alias_line(alias, context) + escape_response(text)


def printed_alias(text: str) -> str | None:
    """The alias already printed on this observation, or None."""
    match = ALIAS_LINE_RE.match(text)
    return match.group(1) if match else None


def printed_context(text: str) -> str | None:
    """The context clause printed on this observation, or None.

    ``None`` both when there is no alias line and when the line carries no
    clause: a root-context observation and an unannotated one are separated by
    ``printed_alias``, not by this.
    """
    match = ALIAS_LINE_RE.match(text)
    if match is None:
        return None
    return match.group(2) or None


#: Which line of ours named an observation's alias. An alias is evidence only
#: with its provenance: a reader that cannot tell the printed handle line from
#: an offload label cannot tell a resident observation from a pointer to one.
ALIAS_SOURCE_HEADER = "header"
ALIAS_SOURCE_LABEL = "label"


def observation_alias(text: str) -> tuple[str | None, str | None]:
    """The alias an observation carries, and which line of ours named it.

    The two shapes this module prints are the only two an alias can come from:
    the handle line ``annotated_observation`` puts above a resident response,
    and the offload label that replaces a response entirely. Readers that knew
    only the label reported no alias at all for the normal, inline case,
    which is every execute observation since the handle line became
    unconditional.

    Both reads are anchored at byte zero and agree with the writer: our line
    is always the FIRST line of an annotated observation,
    and a response whose own first line has either shape is printed under it
    behind ``RESPONSE_ESCAPE`` -- which neither ``ALIAS_LINE_RE`` nor
    ``LABEL_RE`` matches. So a backend's text can never be read as an alias
    here, and the header is checked first because an escaped label-shaped
    response stands UNDER a header naming the step's real ordinal.

    ``(None, None)`` for an observation carrying neither shape.
    """
    alias = printed_alias(text)
    if alias is not None:
        return alias, ALIAS_SOURCE_HEADER
    alias = label_alias(text)
    return (alias, ALIAS_SOURCE_LABEL) if alias is not None else (None, None)


def owns_line(text: str, alias: str) -> bool:
    """Whether *text* opens with a line of ours naming *alias*, the step's own.

    A step's alias comes from the agent's execute ledger, never from its text.
    Once a step is annotated, a backend line shaped like ours is always escaped,
    so this is true of every handle line and label the framework wrote. It is
    false for such a line on a step that was never annotated -- a trajectory
    built outside the loop, or one whose compaction failed before the handle
    line was printed -- where the line is the backend's text and names nothing.
    Every reader that strips a handle line or resolves a label asks this first.
    """
    return observation_alias(text)[0] == alias


def command_response(text: str, alias: str) -> str:
    """The exact command response in the observation slot of step *alias*.

    Only a handle line naming *alias* is presentation, so only that line is
    removed; any other first line is the response's own and stays in the
    archive, its digest and every search of it.
    """
    return strip_alias_line(text) if printed_alias(text) == alias else text


def canonical_response(text: str) -> str | None:
    """The command response an observation slot carries, byte for byte, or None.

    The counterpart of ``observation_alias`` for evidence rather than naming:
    what a digest of this observation has to be taken over for it to be
    comparable with the raw tool return recorded on ``fw.agent.step``.
    Our handle line and the escape underneath it are
    presentation added after that record was closed, so both come off.

    ``None`` for an offload label, which holds no response at all -- its bytes
    stand for evidence that lives in the archive, and comparing them with a
    response would be comparing a pointer with the thing pointed at.
    """
    if printed_alias(text) is not None:
        return strip_alias_line(text)
    return None if is_offload_label(text) else text


def strip_alias_line(text: str) -> str:
    """The original command response, without OUR printed alias line.

    The inverse of ``annotated_observation``: the presentation line goes, and
    the escape that protected a response of the same shape is undone, so what
    comes back is the command response byte for byte. A response
    that was never annotated, or one carrying a line somebody else printed, is
    returned untouched -- ``unescape_response`` only ever runs on the text that
    stood under a line this module wrote.
    """
    match = ALIAS_LINE_RE.match(text)
    return unescape_response(text[match.end():]) if match else text


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def output_description(response: str) -> str:
    """Truthful fallback when a command has no authored Output description."""
    heading = next((line.strip() for line in response.splitlines() if line.strip()), "")
    return f"command output beginning with: {heading[:200]}" if heading else "an empty command result"


def offload_label(*, alias: str, command_name: str, response: str,
                  description: str = "") -> str:
    description = description.strip() or output_description(response)
    return (
        f"{OFFLOAD_MARK}{alias} returned by {command_name}. "
        f"It contains {description.rstrip('.')}. "
        f"It is restored in full when the final answer is written, so search it with "
        f"search_memory only for a value you need for your next step."
    ).rstrip()


def is_offload_label(text: str) -> bool:
    return LABEL_RE.match(text) is not None


def label_alias(text: str) -> str | None:
    match = LABEL_RE.match(text)
    return match.group(1) if match else None


def replacement_saves_space(original: str, replacement: str) -> bool:
    return (len(replacement) < len(original)
            and len(replacement.encode('utf-8')) < len(original.encode('utf-8')))


def offload_saving_bytes(original: str, replacement: str) -> int:
    """UTF-8 bytes the trajectory loses by swapping a response for its label.

    This is the quantity offloading exists to buy, so it is what eligibility is
    decided on: a token estimate of the response alone cannot tell a 3 KB page
    worth replacing from a 300 B fact whose label is bigger than it is. Negative
    when the label is the larger of the two.
    """
    return len(original.encode("utf-8")) - len(replacement.encode("utf-8"))
