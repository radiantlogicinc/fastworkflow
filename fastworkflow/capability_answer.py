"""Answer a capability question ("what can you do?") in code, in agent mode.

When the app workflow has skills (``_skills/<name>.py``) and the whole user message asks
what the assistant can do, the turn is answered here, before the split and the planner,
with no model call: each skill in plain words with its shortest ``plain_utterance`` as the
example, then the current context's commands, one line each, then one line for the commands
that more than half of the other contexts add, then one line per other context with the
commands only it (or a minority) adds. Each other-context line names commands in order while
it fits ``LINE_WIDTH`` characters (the shared line two lines' worth) and counts the rest
("and 3 more"); it always names at least one, and never leaves out just one (it names that
one too, past the width).

A capability question is one of the ``plain_utterances`` of the CME command
``IntentDetection/what_can_i_do``, up to minor wording deviations. Both sides are
normalised (``_words``: lowercase, contractions expanded, punctuation dropped, filler and
politeness words dropped, then a few plain rewrites such as "help me with" -> "do"), and the
message then matches an utterance when they have the same number of words and each word
pair is equal or one typo apart (``_close_word``). Matching word by word, rather than on
the whole string, keeps a message that adds a topic ("what can I do about the logins?")
from ever matching: it has more words than any utterance. The leading-window matcher of
``utils.fuzzy_match.find_best_matches`` does not fit here, because it scores a candidate
on its first ``len(message)`` characters, so "what can" would match "what can you do?".

Workflows without skills, the executor's ``what_can_i_do`` tool, the CME command itself
and assistant mode ('/' commands) are untouched.
"""
from __future__ import annotations

import functools
import re
from typing import Iterable, Optional

import Levenshtein

import fastworkflow
from fastworkflow.command_metadata_api import CommandMetadataAPI

#: The CME command whose ``plain_utterances`` define a capability question.
WHAT_CAN_I_DO = "IntentDetection/what_can_i_do"

#: The characters an other-context line of the answer fits in (the shared line gets twice
#: that); the commands that do not fit are counted ("and 3 more"), unless only one is left.
LINE_WIDTH = 90

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "what're": "what are", "i'd": "i would",
    "i'm": "i am", "you're": "you are", "u": "you", "ya": "you", "r": "are",
    "we": "i", "our": "my", "could": "can", "pls": "please", "plz": "please",
}

#: Words that carry no topic, dropped wherever they stand.
_FILLER = frozenset(
    "please so hey hi hello ok okay well um uh oh then just actually exactly really again "
    "anyway right all else here the things stuff".split())

#: Phrases rewritten after the filler is dropped (the message is lowercase and unpunctuated).
_REWRITES = (
    (re.compile(r"^(?:can you tell me|tell me|i want to know|i would like to know) "), ""),
    (re.compile(r"\b(?:kinds?|sorts?) of\b"), ""),
    (re.compile(r"\bhelp(?: me)?(?: with)?\b"), "do"),
    (re.compile(r"\bfor me$"), ""),
    (re.compile(r"^what (you|i) can\b"), r"what can \1"),
)


def _words(text: str) -> tuple[str, ...]:
    """*text* as the words a capability question is compared on (see the module docstring)."""
    words = re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'")).split()
    words = " ".join(_CONTRACTIONS.get(word, word) for word in words).replace("'", "").split()
    text = " ".join(word for word in words if word not in _FILLER)
    for pattern, replacement in _REWRITES:
        text = pattern.sub(replacement, text).strip()
    return tuple(text.split())


def _close_word(typed: str, expected: str) -> bool:
    """Equal, or (for a word of 3+ letters) one edit or one swap of neighbours apart."""
    if typed == expected:
        return True
    if len(expected) < 3:
        return False
    if Levenshtein.distance(typed, expected) <= 1:
        return True
    diff = [i for i, (a, b) in enumerate(zip(typed, expected)) if a != b]
    return (len(typed) == len(expected) and len(diff) == 2 and diff[1] == diff[0] + 1
            and typed[diff[0]] == expected[diff[1]] and typed[diff[1]] == expected[diff[0]])


@functools.lru_cache(maxsize=8)
def _utterance_words(utterances: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(words for words in map(_words, utterances) if words)


def is_capability_question(message: str, utterances: Iterable[str]) -> bool:
    """True when the whole *message* is one of *utterances* up to minor wording deviations."""
    typed = _words(message)
    return bool(typed) and any(
        len(words) == len(typed) and all(map(_close_word, typed, words))
        for words in _utterance_words(tuple(utterances)))


def capability_utterances(cme_workflow_folderpath: str) -> tuple[str, ...]:
    """The ``plain_utterances`` of the CME command ``IntentDetection/what_can_i_do``."""
    crd = fastworkflow.RoutingRegistry.get_definition(cme_workflow_folderpath)
    return tuple(crd.command_directory.map_command_2_utterance_metadata[WHAT_CAN_I_DO].plain_utterances)


def _plain_name(name: str, capitalize: bool = True) -> str:
    text = name.replace("_", " ").strip().lower()
    text = text.capitalize() if capitalize else text
    return re.sub(r"\bi\b", "I", text)


def _plain_context_name(name: str) -> str:
    return _plain_name(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name))


def _command_list(prefix: str, names: list[str], width: int) -> str:
    """*prefix* then *names* in plain words, as many in order as keep the whole line, with
    " and K more" for the rest, within *width* characters; at least one name. When just one
    would be left out it is named too (never "and 1 more"), and the line may pass *width*."""
    plain = [_plain_name(name, capitalize=False) for name in names]

    def line(shown: int) -> str:
        rest = len(plain) - shown
        return prefix + ", ".join(plain[:shown]) + (f" and {rest} more" if rest else "")

    shown = next((n for n in range(len(plain), 1, -1) if len(line(n)) <= width), min(1, len(plain)))
    return line(len(plain) if len(plain) - shown == 1 else shown)


def _context_lines(other_contexts: dict[str, list[str]]) -> list[str]:
    """One line (up to ``2 * LINE_WIDTH`` characters) for the commands that more than half of
    the other contexts add, then one line (up to ``LINE_WIDTH``) per context with the rest of
    its commands; a context left with none gets no line. Order is kept."""
    seen: dict[str, int] = {}
    for names in other_contexts.values():
        for name in dict.fromkeys(names):
            seen[name] = seen.get(name, 0) + 1
    shared = [name for name, count in seen.items() if count * 2 > len(other_contexts)]
    lines = [_command_list("- In several places: ", shared, 2 * LINE_WIDTH)] if shared else []
    for context, names in other_contexts.items():
        own = [name for name in names if name not in shared]
        if own:
            lines.append(_command_list(f"- In {_plain_context_name(context)}: ", own, LINE_WIDTH))
    return lines


def render_capability_answer(skills: dict[str, tuple[str, ...]], commands: list[str],
                             other_contexts: Optional[dict[str, list[str]]] = None) -> str:
    """Markdown: each skill with its shortest example request, then one line per command,
    then one line for the commands more than half of the other contexts add, then one line per
    other context with the rest of its commands (each of these lines cut to ``LINE_WIDTH``
    characters, the shared line to twice that, with the rest counted: "and 3 more", never
    "and 1 more").

    *skills* maps a skill's name to its ``plain_utterances`` (written as users talk, unlike
    the ``Signature`` docstring, which is written for the matcher; the shortest by length is
    shown, the first of equals); *commands* are the names of the current context's commands;
    *other_contexts* maps each other context's name to the names of the commands it adds
    (``CommandMetadataAPI.get_other_context_commands``).
    """
    lines = ["Here is what I can do. Ask in your own words, for example:", ""]
    for name, utterances in skills.items():
        examples = [u.strip() for u in utterances if u.strip()]
        example = f': "{min(examples, key=len)}"' if examples else ""
        lines.append(f"- **{_plain_name(name)}**{example}")
    lines += ["", "Or ask for a single action:", ""]
    lines += [f"- {_plain_name(name)}" for name in commands]
    lines += _context_lines(other_contexts or {})
    return "\n".join(lines)


def context_commands(app_workflow, cme_workflow_folderpath: str) -> list[str]:
    """The names of the app commands of the current context (no parameters, no docstrings:
    a command's docstring is written for the agent)."""
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=app_workflow.folderpath,
        cme_workflow_path=cme_workflow_folderpath,
        active_context_name=app_workflow.current_command_context_name,
    )
    return [cmd["name"] for cmd in meta.get("commands", [])
            if not cmd["qualified_name"].startswith("IntentDetection/")]


def other_context_commands(app_workflow) -> dict[str, list[str]]:
    """The commands each other context adds, by context name; the CME navigation commands are
    in every context, so the current one already lists (and skips) them. Empty on failure."""
    try:
        return CommandMetadataAPI.get_other_context_commands(
            app_workflow.folderpath, app_workflow.current_command_context_name)
    except Exception:
        return {}


def capability_answer(message: str, app_workflow, cme_workflow_folderpath: str) -> Optional[str]:
    """The answer to *message* when it is a capability question and the workflow has skills, else None."""
    from fastworkflow.workflow_agent import skill_utterances

    skills = skill_utterances(app_workflow.folderpath)
    if not skills or not is_capability_question(message, capability_utterances(cme_workflow_folderpath)):
        return None
    return render_capability_answer(skills, context_commands(app_workflow, cme_workflow_folderpath),
                                    other_context_commands(app_workflow))
