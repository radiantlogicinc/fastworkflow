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
    r"(O(?:0|[1-9]\d*)) returned by ")
#: The A1 handle line, with the context clause optional. The clause
#: can never contain a parenthesis or a newline (``context_clause`` removes
#: both), so the closing ``)`` is unambiguous and a line printed before the
#: clause existed still matches.
#: The clause is printed as " ran in context '<ContextName>' and <instance
#: label>" (" and ..." only when there is a label). A command that moved the
#: context prints " ran in prior context ..." and is followed by
#: "; now in context '<ContextName>' and <instance label>", or by
#: ``CONTEXT_CHANGE_SUFFIX`` when the destination was not recorded. The earlier
#: " ran in <clause>" and ", in <clause>" forms are still matched. A printed
#: clause never contains a semicolon either, so it cannot end in something the
#: suffix would be mistaken for.
ALIAS_LINE_RE = re.compile(
    r"^Observation (?P<alias>O(?:0|[1-9]\d*)) \(execute_workflow_query"
    r"(?: ran in (?:prior )?context '(?P<name>[^'()\n]*)'(?: and (?P<label>[^()\n]*?))?"
    r"|(?:, in| ran in) (?P<clause>[^()\n]*?))?"
    r"(?P<changed>; and resulted in a context change"
    r"|; now in context '[^'()\n]*'(?: and [^()\n]*?)?)?\)\n")
#: What the handle line names the root context as. The root's clause is empty.
ROOT_CONTEXT_LABEL = "global"
#: Appended to the handle line when the command moved the context. The new
#: context is not named here: the command's own response already says it.
CONTEXT_CHANGE_SUFFIX = "; and resulted in a context change"
#: Longest instance identity printed. A uid plus a display name, not a payload.
MAX_INSTANCE_LABEL_CHARS = 80
#: Longest context name printed, for the same reason.
MAX_CONTEXT_NAME_CHARS = 60


def _clipped(value: str, limit: int) -> str:
    """*value* with no parenthesis, semicolon or newline, collapsed spaces, capped."""
    cleaned = " ".join(
        str(value or "").replace("(", " ").replace(")", " ").replace(";", " ").split())
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


def _printed_context(clause: str) -> str:
    """``context '<ContextName>' and <instance label>`` for a ``context_clause``."""
    # A context name never contains a space, so the first word is the name.
    name, _, label = clause.partition(" ")
    printed = f"context '{name or ROOT_CONTEXT_LABEL}'"
    return f"{printed} and {label}" if label else printed


def alias_line(alias: str, context: str | None = None, *, context_changed: bool = False,
               now_in: str | None = None) -> str:
    """The canonical handle line printed above an inline execute observation.

    This is the only identifier the agent is ever asked to pass to
    search_memory, so it must read the same here and in an offload label. It is
    presentation only: archived text never carries it (see strip_alias_line).

    ``context`` is the clause from ``context_clause`` -- the context the command
    RAN IN and, where the workflow declares one, that context's instance
    identity. It is empty at the root context, which is printed as
    ``ROOT_CONTEXT_LABEL``. ``None`` means no clause was recorded, and the line
    is then byte-for-byte the bare alias line.

    ``context_changed`` says the command moved the context, so the agent does
    not read the context it ran in as the one it is now in. ``now_in`` is the
    ``context_clause`` the command left the agent in; it is named when given,
    and is not recorded with an archived observation, so a reprint says only
    that the context changed.
    """
    if context is None:
        return f"Observation {alias} (execute_workflow_query)\n"
    limit = MAX_CONTEXT_NAME_CHARS + MAX_INSTANCE_LABEL_CHARS + 1
    clause = _clipped(context, limit)
    if not context_changed:
        suffix = f" ran in {_printed_context(clause)}"
    elif now_in is None:
        suffix = f" ran in prior {_printed_context(clause)}{CONTEXT_CHANGE_SUFFIX}"
    else:
        suffix = (f" ran in prior {_printed_context(clause)}"
                  f"; now in {_printed_context(_clipped(now_in, limit))}")
    return f"Observation {alias} (execute_workflow_query{suffix})\n"


def annotated_observation(alias: str, context: str | None = None, text: str = "",
                          *, context_changed: bool = False, now_in: str | None = None) -> str:
    """The observation as the agent sees it: our handle line, then the response.

    The one place the two are joined, so the step that prints the line and the
    rehydration that re-prints it build the same observation.
    """
    return alias_line(alias, context, context_changed=context_changed, now_in=now_in) + text


def printed_alias(text: str) -> str | None:
    """The alias already printed on this observation, or None."""
    match = ALIAS_LINE_RE.match(text)
    return match.group("alias") if match else None


def _matched_clause(match: re.Match) -> str | None:
    """The clause a matched handle line printed, in ``context_clause`` form."""
    if match.group("name") is not None:
        label = match.group("label")
        return f"{match.group('name')} {label}" if label else match.group("name")
    return match.group("clause")


def printed_context(text: str) -> str | None:
    """The context clause printed on this observation, or None.

    ``None`` both when there is no alias line and when the line carries no
    clause: a root-context observation and an unannotated one are separated by
    ``printed_alias``, not by this.
    """
    match = ALIAS_LINE_RE.match(text)
    if match is None:
        return None
    clause = _matched_clause(match)
    return None if clause == ROOT_CONTEXT_LABEL else (clause or None)


def printed_subject(text: str) -> tuple[str | None, bool]:
    """The (context clause, changed) an observation's handle line records, as the archive stores them.

    ``(None, False)`` when no clause was printed; ``""`` for the root context.
    """
    match = ALIAS_LINE_RE.match(text)
    clause = None if match is None else _matched_clause(match)
    if clause is None:
        return None, False
    clause = "" if clause == ROOT_CONTEXT_LABEL else clause
    return clause, match.group("changed") is not None


#: A context an alias line names, in either "ran in context" or "now in context" form.
CONTEXT_IN_RE = re.compile(
    r"(?:ran in (?:prior )?|now in )context '(?P<name>[^'()\n]*)'(?: and (?P<label>[^()\n;]*))?")


def printed_instances(text: str) -> list[tuple[str, str, str]]:
    """(context name, instance id, instance label) the handle line of *text* names, in printed order.

    Only contexts that printed an instance are listed. The instance clause is
    ``<instance id> <label>``: its first word is the id, the rest the label.
    """
    match = ALIAS_LINE_RE.match(text)
    if match is None:
        return []
    found = []
    for named in CONTEXT_IN_RE.finditer(match.group(0)):
        instance = (named.group("label") or "").strip()
        if instance:
            instance_id, _, label = instance.partition(" ")
            found.append((named.group("name"), instance_id, label.strip()))
    return found


def command_response(text: str, alias: str) -> str:
    """The exact command response in the observation slot of step *alias*.

    Only a handle line naming *alias* is presentation, so only that line is
    removed. Any other first line is the response's own.
    """
    return strip_alias_line(text) if printed_alias(text) == alias else text


def strip_alias_line(text: str) -> str:
    """The command response without the handle line printed above it."""
    match = ALIAS_LINE_RE.match(text)
    return text[match.end():] if match else text


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def output_description(response: str) -> str:
    """Truthful fallback when a command has no authored Output description."""
    heading = next((line.strip() for line in response.splitlines() if line.strip()), "")
    return f"command output beginning with: {heading[:200]}" if heading else "an empty command result"


#: The per-label reminder of the restore promise. The promise and what follows
#: from it (search only for a next-step value) are stated ONCE, in the agent
#: signature and the search_memory tool description; repeating them in every
#: label cost ~100 bytes per label and made fewer observations worth
#: offloading (``offload_saving_bytes``). Labels written with the longer
#: sentence still parse: ``LABEL_RE`` reads only the prefix. "Normally", because
#: the restore is bounded by the answer's evidence budget: when it binds, the
#: oldest observations are not restored and the answer step is told which
#: (``answer_rehydration.NOT_REHYDRATED_PREFIX``). Labels written with the
#: earlier "Restored in full for the final answer." parse the same way.
LABEL_RESTORE_MARK = "Normally restored for the final answer."


def offload_label(*, alias: str, command_name: str, response: str,
                  description: str = "") -> str:
    description = description.strip() or output_description(response)
    return (
        f"{OFFLOAD_MARK}{alias} returned by {command_name}. "
        f"It contains {description.rstrip('.')}. "
        f"{LABEL_RESTORE_MARK}"
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
