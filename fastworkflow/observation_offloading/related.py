"""Which other observations of a turn a question more likely concerns.

The relatedness score, the related-handle suggestions, and the verbatim answer
``search_memory`` gives for an observation too short to search. They read only
the archive's summaries and recorded subjects, never stored text.
"""
from __future__ import annotations

from typing import Any, Optional
import re

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
    is_broad_scope,
)
from fastworkflow.observation_offloading.labels import (
    is_search_answer_key,
    quote_marker_lines,
)
from fastworkflow.observation_offloading.state import context_clause_of


#: An archived observation at or under this many UTF-8 bytes is returned
#: verbatim instead of searched. It is lossless: the search model would read
#: exactly these bytes. In recorded runs (fix-cj7t) such handles were
#: navigation replies ("Context is now 'DirectoryExplorer'") or one-line
#: failures picked by mistake, 21 of 23 times.
SHORT_OBSERVATION_BYTES = 256
SHORT_OBSERVATION_MARK = "[search_memory SHORT OBSERVATION:"
RELATED_HANDLES_SHOWN = 3
_RELATED_STOPWORDS = frozenset({
    "the", "and", "for", "of", "a", "an", "to", "in", "on", "is", "are", "what", "which",
    "with", "list", "all", "their", "its", "this", "that", "from", "by", "show", "provide",
    "give", "label", "uid", "listed", "returned", "output", "observation", "corresponding",
    "value", "was", "were", "get", "find", "open"})


def _fold_word(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _related_words(text: str) -> set[str]:
    # Unicode letters and digits, split at underscores so ``list_entitlements``
    # in a question still meets the command's words; casefolded on both sides.
    return {_fold_word(w) for w in re.findall(r"[^\W_]+", (text or "").casefold())
            if len(w) > 1 and w not in _RELATED_STOPWORDS}


def command_verb(command: str) -> str:
    """The command name at the start of a recorded command line."""
    return re.split(r"[\s<(]", command.strip(), maxsplit=1)[0]


def relatedness(wanted: set[str], verb: str, clause: Optional[str]) -> int:
    """How strongly a handle's command and subject mention the question's words."""
    return (3 * len(wanted & _related_words(verb.replace("_", " ")))
            + 2 * len(wanted & _related_words(clause or "")))


def related_handles(
    question: str,
    exclude: str,
    scope: RuntimeHandleScope,
    store: RuntimeHandleArchive,
    limit: int = RELATED_HANDLES_SHOWN,
) -> list[tuple[str, str, Optional[str]]]:
    """``(alias, command, clause)`` of the turn's handles that best match *question*.

    See ``scored_related_handles``, which also returns each handle's score.
    """
    return [(alias, verb, clause) for _, alias, verb, clause in
            scored_related_handles(question, exclude, scope, store, limit)]


def scored_related_handles(
    question: str,
    exclude: str,
    scope: RuntimeHandleScope,
    store: RuntimeHandleArchive,
    limit: int = RELATED_HANDLES_SHOWN,
    *,
    above: int = 0,
) -> list[tuple[int, str, str, Optional[str]]]:
    """``(score, alias, command, clause)`` of the turn's handles that best match *question*.

    Scored on the question's words against each handle's command name and
    recorded subject clause (``relatedness``), most recent first on ties; only
    handles scoring strictly more than ``above`` are offered. Short handles and
    search-answer records are never offered: they are not something to search.
    Chosen from the archive's summaries, so no stored text is loaded; subjects
    come from the process cache after their first read. Raises when the
    summaries cannot be read.
    """
    return _scored_summaries(_related_words(question), store.list_summaries(scope),
                             exclude, scope, store, limit, above)


def _scored_summaries(
    wanted: set[str],
    summaries: list[dict[str, Any]],
    exclude: str,
    scope: RuntimeHandleScope,
    store: RuntimeHandleArchive,
    limit: int,
    above: int,
) -> list[tuple[int, str, str, Optional[str]]]:
    scored = []
    for row in summaries:
        alias = str(row.get("alias") or "")
        if alias == exclude or is_search_answer_key(alias):
            continue
        if int(row.get("utf8_bytes") or 0) <= SHORT_OBSERVATION_BYTES:
            continue
        verb = command_verb(str(row.get("command") or ""))
        clause = context_clause_of(scope, alias, selected_archive=store)
        score = relatedness(wanted, verb, clause)
        if score > above:
            scored.append(((score, int(row.get("offload_order") or 0)), alias, verb, clause))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(rank[0], alias, verb, clause) for rank, alias, verb, clause in scored[:limit]]


#: Why a short observation's answer names no other observation, beyond "none
#: mentions the question's words more". None when the turn's handles were read.
RELATED_SCOPE_REFUSED = "scope_refused"
RELATED_LOOKUP_FAILED = "lookup_failed"


def short_observation_lookup(
    question: str,
    alias: str,
    verb: str,
    scope: RuntimeHandleScope,
    store: RuntimeHandleArchive,
) -> dict[str, Any]:
    """The short observation's own subject and score, and the handles that outscore it.

    ``{"clause", "own_score", "scored", "unlisted", "error"}``. The searched
    handle is scored with the same formula as the others, so another handle is
    offered only when it mentions the question's words MORE -- a short answer
    about the right subject is not undercut by a longer one about another.

    Never raises and never waits long: a broad scope is not enumerated at all
    (``is_broad_scope``), and an unavailable or failing archive leaves nothing
    listed, the subject then read from process memory only. Either way
    ``unlisted`` says why, so the hint does not claim nothing else matches.
    """
    wanted = _related_words(question)
    found: dict[str, Any] = {"scored": [], "unlisted": None, "error": None}
    summaries: list[dict[str, Any]] = []
    if is_broad_scope(scope):
        found["unlisted"] = RELATED_SCOPE_REFUSED
    elif not getattr(store, "available", True):
        found["unlisted"], found["error"] = RELATED_LOOKUP_FAILED, "archive_unavailable"
    else:
        try:
            summaries = store.list_summaries(scope)
        except Exception as error:  # noqa: BLE001 - a suggestion must never fail a search
            found["unlisted"], found["error"] = RELATED_LOOKUP_FAILED, type(error).__name__
    found["clause"] = context_clause_of(
        scope, alias, selected_archive=store,
        durable=found["unlisted"] != RELATED_LOOKUP_FAILED)
    found["own_score"] = relatedness(wanted, verb, found["clause"])
    if summaries:
        found["scored"] = _scored_summaries(wanted, summaries, alias, scope, store,
                                            RELATED_HANDLES_SHOWN, found["own_score"])
    return found


def short_observation_answer(
    *, alias: str, command: str, text: str,
    related: list[tuple[str, str, Optional[str]]],
    clause: Optional[str] = None,
    unlisted: Optional[str] = None,
) -> str:
    """The whole of a short observation, and where the answer more likely is.

    The observation is the backend's text, printed between framework lines, so
    any line of it shaped like one is quoted (``quote_marker_lines``).
    """
    ran = f" of {command}" if command else ""
    if clause:
        ran += f", in {clause}"
    elif clause is not None:
        ran += ", at the workflow root"
    if related:
        where = "; ".join(f"{a} ({c}{', in ' + cl if cl else ''})" for a, c, cl in related)
        hint = (f"If it does not answer the question, other observations of this turn "
                f"that mention the question's words more are: {where}. Search one of "
                f"those instead.")
    elif unlisted == RELATED_LOOKUP_FAILED:
        hint = ("Other observations of this turn could not be listed, so none is "
                "suggested here.")
    elif unlisted == RELATED_SCOPE_REFUSED:
        hint = ("Other observations are not listed in this scope, so none is "
                "suggested here.")
    else:
        hint = ("No other observation in this turn matches the question's words more "
                "than this one; if it does not answer the question, run the command "
                "that produces what you need.")
    return (f"{SHORT_OBSERVATION_MARK} {alias} is the complete response{ran}, shown "
            f"verbatim because it is too short to search]\n{quote_marker_lines(text)}\n{hint}")
