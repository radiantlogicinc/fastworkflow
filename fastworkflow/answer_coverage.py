"""Roster nudge: the agent is told which named items of the request it never
looked at.

A request that names people, uids or email addresses is a request about those
items, and an agent that finishes without having made one of them the subject
of a command has not covered the request. This module detects that case and
composes one bounded note to say so, at the point the loop tries to finish.

The whole pipeline is deterministic and reads only what the turn itself
produced:

* named items come out of the request text by regex (``named_entities``), with
  bounds on span length and count so a long request cannot become the prompt;
* only names, uids and email addresses are ever instructed on
  (``INSTRUCTED_KINDS``) -- a quoted phrase is narrative framing, extracted and
  measured but never turned into an instruction;
* presence is a normalised substring test (``split_by_presence``) against the
  context clauses the handle archive recorded for this scope
  (``subject_corpus``), not against anything a model said;
* the note itself is capped at ``NUDGE_MAX_BYTES`` and only fires when the loop
  still has iterations left to act on it (``NUDGE_MIN_ITERS_LEFT``).

Rules this module does not bend:

* **No model chooses anything.** Extraction and presence are both mechanical.
* **Nothing is invented.** An item named in the note is quoted verbatim from
  the request; nothing is said about what its value would have been.
* **Absence of evidence is never asserted.** The note says an item was never a
  subject, which is a fact about the run. It never says data about the item
  does not exist, nor that nothing about it was retrieved: a name can arrive
  as a row of somebody else's listing without ever being a subject. When no
  clause was recorded at all, the note says nothing, because "never a subject"
  would then be a statement about the archive rather than about the run. The
  same holds when clauses were recorded but none of them names an instance:
  a workflow that declares no instance identity prints context names alone,
  so no named item could ever be found in them.
* **It is a note, not a gate.** ``build_nudge`` returns text and a
  ``NudgeReport``; the caller decides. Nothing here edits, rejects or retries
  an answer.

What is no longer here: the answer-time coverage statement (the prepended
unobserved/observed instruction block and its post-check) has been removed, so
nothing in this module is injected into the extract input. The finish reminders
the ReAct loop emits -- of which this nudge is one -- remain on by default and
can only be turned off, by setting ``FW_EVAL_FINISH_REMINDERS=0``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import is_search_answer_key
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    default_scope,
    stored_handles,
)

logger = logging.getLogger(__name__)

#: The key the answer-time coverage statement was stored under in the
#: extractor's trajectory copy. Nothing writes this key any more. The constant
#: survives only so that ``utils/react.py`` can go on excluding the key when it
#: truncates a trajectory recorded while the statement still existed.
COVERAGE_KEY = "coverage_statement"

#: ``build_query_with_next_steps`` hands the agent ``<request>\n\nExecute these
#: next steps:\n<plan>``. The plan is a model's text about the request, not the
#: request, so entities are taken from the part before this marker. Splitting is
#: literal and deterministic; a query without the marker is used whole.
PLAN_MARKER = "\n\nExecute these next steps:\n"
#: The prefix the same builder adds when it passes agent inputs and trajectory.
REQUEST_PREFIX = "User Query:\n"

#: Bounds on what counts as one named item, so a runaway span cannot become a
#: paragraph-long "entity" and a 300-name request cannot become the prompt.
MIN_ENTITY_CHARS = 3
MAX_ENTITY_CHARS = 120
MAX_ENTITIES = 64

#: Only these kinds are ever INSTRUCTED as "not retrieved". A quoted phrase is
#: extracted and measured, never instructed, and the offline replay
#: (``evaluation/artifacts/result-search/honesty-replay.md``) is why: the pinned
#: card quotes its control as 'Active contractor identities whom manager left'
#: while the catalogue's label is ``Contractor whom manager left``, so the exact
#: phrase appears in no observation of 83 of 88 stored attempts -- INCLUDING the
#: 28 in which that branch was judged CORRECT. A quoted phrase in a request is
#: narrative framing; a name, a uid and an address are handles the workflow
#: itself prints. Instructing "not retrieved" on framing would manufacture the
#: very false absence ``ido-8ps.22`` is about.
INSTRUCTED_KINDS = frozenset({"name", "uid", "email"})

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

#: ``ido-c7m``/F36. A zero-width space is a break opportunity -- the character
#: is literally named a space -- so it folds to one rather than being removed.
#: Deleting it would turn ``Alan<ZWSP>Cooper`` into ``alancooper`` and leave
#: the request's ``Alan Cooper`` unobserved, which is the very false absence
#: this is about. The cost is the other reading: a zero-width space INSIDE a
#: token splits it, so ``Al<ZWSP>an Cooper`` still does not match. That reading
#: is what the soft hyphen is for and what every other invisible character here
#: gets, and a fold to a space can only split a run, never join two -- which is
#: the side a presence test may err on.
_IGNORABLE_TO_SPACE = "\u200b"


def _strip_format_characters(text: str) -> str:
    """Drop the format (``Cf``) characters, folding a zero-width space to a space.

    Soft hyphen (U+00AD), byte-order mark (U+FEFF), the zero-width joiner and
    non-joiner, the word joiner and the bidi controls are default-ignorable:
    they render as nothing, so a row carrying one inside a name prints exactly
    the name the request wrote, while the request's spelling reads as
    unobserved and the run is credited with never having looked at that item.
    They are removed here, BEFORE the compatibility fold, so a fold that spans
    one still composes.

    Removal is by Unicode category, not by the list of characters one review
    happened to try: any other ``Cf`` character is invisible for the same
    reason. It cannot make two visibly different names collide -- every
    character it removes renders as nothing -- and the one character that could
    have joined two visible runs into a third spelling is folded to a space
    instead.
    """
    if not any(unicodedata.category(ch) == "Cf" for ch in text):
        return text
    return "".join(
        " " if ch in _IGNORABLE_TO_SPACE
        else "" if unicodedata.category(ch) == "Cf"
        else ch
        for ch in text
    )


def normalise(text: Any) -> str:
    """NFKC, casefolded, whitespace collapsed -- the one comparison form.

    Presence is decided on this form and nothing else, so a name broken across a
    line in a rendered row still matches the name written on one line in the
    request, and a full-width or ligature variant matches its plain spelling.
    Invisible format characters go first (:func:`_strip_format_characters`), so
    a soft hyphen or a byte-order mark inside a name in a row cannot hide that
    name from the presence test.
    """
    stripped = _strip_format_characters(str(text or ""))
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    return " ".join(folded.split())


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------

def request_text(user_query: Any) -> str:
    """The user's request, without the planner's todo list.

    The agent's ``user_query`` is the refined request with a generated plan
    appended under a fixed marker. The plan is a model's paraphrase: a name it
    invents is not a named item of the request, and telling the extractor to
    report "not retrieved" for one would be this module inventing work. So the
    text before the first marker is the request, and a query that never went
    through the planner is its own request.
    """
    text = str(user_query or "")
    if text.startswith(REQUEST_PREFIX):
        text = text[len(REQUEST_PREFIX):]
    marker = text.find(PLAN_MARKER)
    return text if marker < 0 else text[:marker]


# ---------------------------------------------------------------------------
# Named entities
# ---------------------------------------------------------------------------

#: A uid as this workflow prints them: a long unbroken hex run.
_UID_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{16,64}(?![0-9A-Za-z])")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]")
#: ``'...'`` where the opening quote follows a space or an opener and the closing
#: quote precedes a space or punctuation. That rule is what keeps the apostrophe
#: of "this quarter's" from opening a quotation.
_SQUOTE_RE = re.compile(r"(?<![^\s(\[{])'([^'\n]{2,120})'(?=[\s.,;:!?)\]}]|$)")
_DQUOTE_RE = re.compile(r"[\"“]([^\"”\n]{2,120})[\"”]")
_CURLY_SQUOTE_RE = re.compile(r"‘([^’\n]{2,120})’")

#: Stripped off the ends of a token; never from inside it, so
#: ``Active Directory_Cloud`` and ``O'Brien`` survive intact.
_LEAD_STRIP = "\"'“”‘’([{"
_TRAIL_STRIP = "\"'“”‘’)]}"
#: A token ending in one of these ends the run it is in: a comma or a full stop
#: separates two names, it does not join them.
_BREAK_AFTER = ",;:.!?—–"
_SENTENCE_END = ".!?"
#: ``ido-jf6``/F29. These separate two named items with no space around them, so
#: ``Alan Cooper/Brandon Miller`` is two people and ``Anna Garcia—contractor``
#: is a person and a word. They never belong to a token and always end a run.
_SEPARATORS = "/—–"
_TOKEN_RE = re.compile(r"[^\s/—–]+")
#: Taken off the end of a token BEFORE the possessive test. The apostrophes are
#: deliberately absent: a trailing apostrophe is stripped as a closing quote,
#: which would make a plural possessive invisible to that test.
_TRAIL_BEFORE_POSSESSIVE = "\"“”)]}" + _BREAK_AFTER
#: ``ido-jf6``/F29. "Cooper's" is a name plus a grammatical marker, not a
#: two-word name, and the marker ENDS the name: "Alan Cooper's Active Directory
#: rights" is Alan Cooper AND Active Directory, never one item. Straight and
#: curly apostrophes, and the upper-case spelling of a shouted request.
_POSSESSIVE_SUFFIXES = ("'s", "’s", "'S", "’S")
#: ``ido-jf6``/F29. A sentence that OPENS with one of these opens with a verb,
#: so the capital is grammar and not a name: "List Identities whose manager
#: left" and "Compare Alan and Brandon" name nobody. The list is deliberately
#: small, explicit, and holds only words that are not also ordinary given names
#: or surnames -- no "Mark", "Bill", "Grant", "Will", "Rose", "May" -- because
#: dropping a real first name would cost the run a handle, the more expensive
#: mistake of the two. Only the FIRST token of a sentence is tested against it,
#: so "Alan Cooper met Barbara List" is untouched.
_IMPERATIVE_VERBS = frozenset({
    "audit", "check", "compare", "count", "describe", "display", "explain",
    "fetch", "find", "get", "identify", "list", "report", "retrieve", "review",
    "search", "show", "summarise", "summarize", "tell", "verify",
})


def _clean_token(raw: str) -> tuple[str, bool]:
    """The token itself, and whether it carried a possessive marker.

    The possessive comes off before the capitalisation test, because "Cooper's"
    ends in a letter and would otherwise read as an ordinary capitalised token
    and join its run to whatever capitalised word follows it.
    """
    trimmed = raw.lstrip(_LEAD_STRIP).rstrip(_TRAIL_BEFORE_POSSESSIVE)
    possessive = False
    for suffix in _POSSESSIVE_SUFFIXES:
        if len(trimmed) > len(suffix) and trimmed.endswith(suffix):
            trimmed, possessive = trimmed[: -len(suffix)], True
            break
    else:
        # Plural possessive: a bare apostrophe after an s ("the Hendersons'
        # rights"). Only that shape, so a closing quote on any other word stays
        # an ordinary closing quote.
        if len(trimmed) > 2 and trimmed[-1] in "'’" and trimmed[-2] in "sS":
            trimmed, possessive = trimmed[:-1], True
    cleaned = trimmed.strip(_LEAD_STRIP).rstrip(_TRAIL_STRIP + _BREAK_AFTER)
    return cleaned.strip(_LEAD_STRIP), possessive


def _is_capitalised(token: str) -> bool:
    """First letter uppercase, and there is a letter. Digits alone are not names."""
    for char in token:
        if char.isalpha():
            return char.isupper()
    return False


@dataclass(frozen=True)
class Entity:
    """One named item of the request, with where it was found."""

    text: str
    kind: str  # "name" | "quoted" | "uid" | "email"
    start: int

    @property
    def key(self) -> str:
        return normalise(self.text)


def _name_spans(text: str) -> list[Entity]:
    """Maximal runs of two or more consecutive capitalised tokens.

    A run is broken by a lowercase token and by the punctuation that ends a
    token, so "Alan Cooper, Alisha Ochoa" is two names rather than one. A run
    that starts a sentence loses its first token when two or more remain --
    "Two Active Directory rights" is about *Active Directory*, and the capital on
    "Two" is grammar, not a name. "Christopher Hubbard is one of the people it
    names" keeps both of its tokens: dropping the sentence capital there would
    leave a bare surname, which is a worse handle than the name itself.

    A run also ends at a possessive marker and at a ``/`` or a dash, and a run
    that opens a sentence with an imperative verb loses that verb however short
    the run is. Without those two rules "List Alan Cooper's Active Directory
    rights" comes out as a single named item that appears in no observation
    ever retrieved, and the run is then reported as never having turned to a
    person it had in fact opened.
    """
    tokens: list[tuple[str, int, bool, bool]] = []  # cleaned, start, sentence_start, breaks
    previous_end = 0
    previous_raw = ""
    matches = list(_TOKEN_RE.finditer(text))
    for index, match in enumerate(matches):
        raw = match.group(0)
        gap = text[previous_end:match.start()]
        sentence_start = (
            not tokens
            or "\n" in gap
            or previous_raw.rstrip(_TRAIL_STRIP).endswith(tuple(_SENTENCE_END))
        )
        cleaned, possessive = _clean_token(raw)
        following = (
            text[match.end():matches[index + 1].start()]
            if index + 1 < len(matches)
            else text[match.end():]
        )
        breaks = (
            possessive
            or raw.rstrip(_TRAIL_STRIP).endswith(tuple(_BREAK_AFTER))
            or any(char in _SEPARATORS for char in following)
        )
        tokens.append((cleaned, match.start(), sentence_start, breaks))
        previous_end, previous_raw = match.end(), raw

    spans: list[Entity] = []
    run: list[tuple[str, int, bool]] = []

    def flush() -> None:
        body = run
        if body and body[0][2] and (
            len(body) >= 3 or body[0][0].casefold() in _IMPERATIVE_VERBS
        ):
            body = body[1:]
        if len(body) >= 2:
            spans.append(
                Entity(" ".join(part[0] for part in body), "name", body[0][1])
            )
        run.clear()

    for cleaned, start, sentence_start, breaks in tokens:
        if cleaned and _is_capitalised(cleaned):
            run.append((cleaned, start, sentence_start))
            if breaks:
                flush()
        else:
            flush()
    flush()
    return spans


def named_entities(request: str) -> list[Entity]:
    """Every named item of *request*, in the order it is written.

    Proper names (capitalised multi-token spans), quoted strings, uids and email
    addresses. Deduplicated on the normalised form, first spelling kept, capped.
    Nothing here consults a model, a dictionary or the workflow: the same request
    always yields the same list.
    """
    found: list[Entity] = list(_name_spans(request))
    for pattern in (_SQUOTE_RE, _DQUOTE_RE, _CURLY_SQUOTE_RE):
        found.extend(
            Entity(match.group(1).strip(), "quoted", match.start(1))
            for match in pattern.finditer(request)
        )
    found.extend(
        Entity(match.group(0), "uid", match.start()) for match in _UID_RE.finditer(request)
    )
    found.extend(
        Entity(match.group(0), "email", match.start())
        for match in _EMAIL_RE.finditer(request)
    )

    ordered: list[Entity] = []
    seen: set[str] = set()
    for entity in sorted(found, key=lambda item: (item.start, item.kind)):
        key = entity.key
        if not key or key in seen:
            continue
        if not (MIN_ENTITY_CHARS <= len(entity.text) <= MAX_ENTITY_CHARS):
            continue
        seen.add(key)
        ordered.append(entity)
        if len(ordered) >= MAX_ENTITIES:
            break
    return ordered


# ---------------------------------------------------------------------------
# What the run retrieved
# ---------------------------------------------------------------------------


def subject_corpus(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
) -> str:
    """The CONTEXT CLAUSES of this turn's observations, normalised, as one haystack.

    "Does this name appear anywhere in what the run retrieved" is the wrong
    question for a finish-time check: ``find_identity <name>`` puts a name into
    the retrieved text, so an agent that audits one person and never opens the
    other four still looks as though it covered all five.

    The context clause answers the right question. Each archived observation
    records the context INSTANCE the command ran against -- "Identity
    28c5aeb5... Alan Cooper", "Account e8a0c3a1... Alan Cooper". A name in a
    clause is a name the run made the SUBJECT of a command; a name only in
    observation text is a row in somebody else's listing.

    Clauses only. No observation text, no stored rows: this haystack may never
    be the reason a name is called retrieved, and it never is -- it is read only
    to decide whether the agent has yet turned to that subject.
    """
    return normalise("\n".join(subject_clauses(scope=scope, archive=archive)))


def clause_names_instance(clause: str) -> bool:
    """Whether *clause* carries an instance identity, not just a context name.

    ``labels.context_clause`` prints ``<ContextName> <instance label>`` and a
    context that declares no identity as its name alone, and context names are
    class names, so a space is what separates the two shapes -- the same test
    the ``context_line`` event records as ``has_instance``.
    """
    return " " in " ".join(str(clause or "").split())


def subject_clauses(
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
) -> list[str]:
    """The non-empty context clauses recorded for this turn's observations."""
    selected = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()
    try:
        rows = archive.list(selected)
    except Exception:  # noqa: BLE001 - an unreadable archive is an empty one
        logger.debug("roster nudge could not list the archive", exc_info=True)
        rows = []
    aliases: list[str] = []
    for handle in rows:
        alias = str(handle.get("alias") or "")
        if alias and alias not in aliases and not is_search_answer_key(alias):
            aliases.append(alias)
    for alias in stored_handles(selected):
        if alias and alias not in aliases and not is_search_answer_key(alias):
            aliases.append(alias)
    parts = [
        context_clause_of(selected, alias, selected_archive=archive) or ""
        for alias in aliases
    ]
    return [part for part in parts if part.strip()]


#: ``ido-mng``. A rendered result page states the query in its own header --
#: ``result_handle=O2 filter="Christopher Hubbard" page 1 rows 0 of 12 ...`` --
#: and a backend that finds nothing may say so in the same words the agent typed
#: ("No identity matching 'Christopher Hubbard' was found."). Both are the
#: agent's OWN text quoted back, never a retrieved row, and both are in the
#: archived response, which is why ``retrieved_corpus``'s "the archive keeps the
#: command response, never the command" was not by itself enough to keep the
#: contract. A filtered miss that echoes its literal would otherwise make the
#: typed name look retrieved and leave the writer instructed AGAINST the one
#: true statement -- that the filtered listing returned nothing.
_FILTER_ECHO_RE = re.compile(r"""filter=(?:"[^"\n]*"|'[^'\n]*')""", re.IGNORECASE)

#: A miss, however the backend words it: "no rows matched", "no identity
#: matching", "no accounts were found", "0 results returned".
_MISS_CUE = (
    r"no(?:t)?\s+(?:[\w'\u2019\-]+\s+){0,4}?"
    r"(?:match(?:ed|es|ing)?|found|returned|exist(?:s|ed)?)"
)

#: The literal a miss quotes back, and ONLY that literal: the cue, a short gap
#: that crosses no sentence boundary, then one quoted span. Nothing else on the
#: line is touched, because removing more than the echo is how a haystack starts
#: reporting that something retrieved was never retrieved, which is the error
#: this module exists to prevent.
#: The quote pairs a backend may use. Typographic quotes survive NFKC, so a
#: miss written with them echoes just as loudly as one written with ASCII.
_QUOTED_SPAN = (
    r"\"[^\"\n]*\"|'[^'\n]*'|\u201c[^\u201d\n]*\u201d|\u2018[^\u2019\n]*\u2019"
)

_MISS_ECHO_RE = re.compile(
    _MISS_CUE + r"(?:[^\"'\u201c\u2018\n.;]{0,24}?)(" + _QUOTED_SPAN + r")",
    re.IGNORECASE,
)


def strip_query_echoes(text: str) -> str:
    """*text* without the run's own query quoted back at it.

    Two spans go, both of them the agent's typed literal and neither of them a
    retrieved value: the ``filter="..."`` echo a result page prints in its
    header, and the quoted literal a "nothing matched" sentence repeats. What is
    removed is the echo itself, never the line around it: a page that DID match
    rows still carries the name in its rows, so this cannot turn a retrieved
    name into an absent one. Idempotent, and safe on text whose line structure
    normalisation has already collapsed, because neither pattern spans a line.
    """
    out = _FILTER_ECHO_RE.sub(" ", str(text or ""))
    return _MISS_ECHO_RE.sub(lambda match: match.group(0)[:match.start(1) - match.start(0)], out)


def split_by_presence(
    entities: Iterable[Entity], haystack: str
) -> tuple[list[Entity], list[Entity]]:
    """``(observed, unobserved)``: a normalised substring test, nothing more.

    Nothing more, on a haystack of what the run RETRIEVED: the run's own query
    echoes are stripped first (:func:`strip_query_echoes`), so a name the agent
    merely typed into a filter can never come back as observed even when the
    backend quotes it in a header or a "nothing matched" sentence.
    """
    hunted = strip_query_echoes(haystack)
    observed: list[Entity] = []
    unobserved: list[Entity] = []
    for entity in entities:
        (observed if entity.key and entity.key in hunted else unobserved).append(entity)
    return observed, unobserved


# ---------------------------------------------------------------------------
# The roster nudge (ido-8ps.27)
# ---------------------------------------------------------------------------

#: The nudge is bounded like the observed list and for the same reason: it is
#: built from a regex over one request, so this is a backstop, not a budget. A
#: list that does not fit is cut and counted, never dropped silently.
NUDGE_MAX_BYTES = 1024

#: The nudge costs one iteration and is worthless unless the agent can act on
#: it. Below this many further actions the loop ends as it always did.
NUDGE_MIN_ITERS_LEFT = 2

NUDGE_HEAD = (
    "Harness check before this turn ends. You selected finish, and this run "
    "has not made the following named items of the request the subject of any "
    "command: "
)
NUDGE_TAIL = (
    ". You have {left} more actions available before this turn ends - the "
    "budget is not spent. Retrieve for each of them what the request asks, or "
    "select finish again and say in your final answer why you could not. This "
    "note is from the harness, not from the user: do not ask the user about "
    "it, and it is shown once per turn."
)
NUDGE_MORE = "; and {count} more"


def nudge_block(names: Iterable[str], iterations_left: int) -> tuple[str, int]:
    """``(text, named)`` -- the nudge, capped at ``NUDGE_MAX_BYTES``.

    Deterministic: the same missing set and the same budget produce the same
    bytes. Names are taken in order until the cap is reached and the remainder
    is counted, so a request naming forty people still yields one bounded note.
    """
    wanted = [str(name).strip() for name in names if str(name).strip()]
    if not wanted:
        return "", 0
    tail = NUDGE_TAIL.format(left=max(0, int(iterations_left)))
    frame = len(NUDGE_HEAD.encode("utf-8")) + len(tail.encode("utf-8"))
    room = NUDGE_MAX_BYTES - frame
    taken: list[str] = []
    for name in wanted:
        candidate = "; ".join([*taken, name])
        dropped = len(wanted) - len(taken) - 1
        suffix = NUDGE_MORE.format(count=dropped) if dropped else ""
        if len(candidate.encode("utf-8")) + len(suffix.encode("utf-8")) > room:
            break
        taken.append(name)
    if not taken:
        return "", 0
    listed = "; ".join(taken)
    if len(taken) < len(wanted):
        listed += NUDGE_MORE.format(count=len(wanted) - len(taken))
    return f"{NUDGE_HEAD}{listed}{tail}", len(taken)


@dataclass
class NudgeReport:
    """What the finish-time check saw, whether or not it fired."""

    fired: bool = False
    reason: str = ""
    entities_total: int = 0
    subjects_total: int = 0
    subjects_missing: list[str] = field(default_factory=list)
    subjects_named: int = 0
    iterations_left: int = 0
    text_bytes: int = 0
    clause_bytes: int = 0

    def as_event(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "reason": self.reason,
            "entities_total": self.entities_total,
            "subjects_total": self.subjects_total,
            "subjects_missing": list(self.subjects_missing),
            "subjects_named": self.subjects_named,
            "iterations_left": self.iterations_left,
            "text_bytes": self.text_bytes,
            "clause_bytes": self.clause_bytes,
        }


def build_nudge(
    *,
    user_query: Any,
    iterations_left: int,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    clauses: Optional[str | Iterable[str]] = None,
) -> tuple[str, NudgeReport]:
    """``(text, report)``. ``text`` is ``""`` when nothing should be injected.

    The whole decision, in one deterministic place, so the loop's own code is a
    call and a branch. Nothing here calls a model, reads a backend or consults
    the user; it reads the request by the same regex ``build_statement`` uses
    and the same context clauses the archive already holds.

    *clauses* overrides the archive: a list of recorded clauses, or one string
    taken as a single clause.
    """
    report = NudgeReport(iterations_left=int(iterations_left))
    if int(iterations_left) < NUDGE_MIN_ITERS_LEFT:
        report.reason = "no room to act"
        return "", report
    entities = named_entities(request_text(user_query))
    instructed = [e for e in entities if e.kind in INSTRUCTED_KINDS]
    report.entities_total = len(entities)
    report.subjects_total = len(instructed)
    if not instructed:
        report.reason = "the request names no items"
        return "", report
    if clauses is None:
        recorded = subject_clauses(scope=scope, archive=archive)
    elif isinstance(clauses, str):
        recorded = [clauses] if clauses.strip() else []
    else:
        recorded = [str(clause) for clause in clauses if str(clause).strip()]
    haystack = normalise("\n".join(recorded))
    report.clause_bytes = len(haystack.encode("utf-8"))
    if not haystack:
        # No clause was recorded at all, so "never the subject of a command" is
        # a statement about the archive rather than about the run. Say nothing,
        # exactly as build_statement names nothing on an incomplete archive.
        report.reason = "no context clauses recorded"
        return "", report
    if not any(clause_names_instance(clause) for clause in recorded):
        # Clauses exist but every one is a bare context name: this workflow
        # declares no instance identity, so no named item can ever appear in
        # them, and reporting them all missing would be a false note.
        report.reason = "no recorded clause names an instance"
        return "", report
    _, missing = split_by_presence(instructed, haystack)
    report.subjects_missing = [entity.text for entity in missing]
    if not missing:
        report.reason = "every named item was already a subject"
        return "", report
    text, named = nudge_block(report.subjects_missing, iterations_left)
    if not text:
        report.reason = "the note would not fit its byte cap"
        return "", report
    report.fired = True
    report.subjects_named = named
    report.text_bytes = len(text.encode("utf-8"))
    return text, report


__all__ = [
    "COVERAGE_KEY",
    "Entity",
    "INSTRUCTED_KINDS",
    "MAX_ENTITIES",
    "MAX_ENTITY_CHARS",
    "MIN_ENTITY_CHARS",
    "NUDGE_MAX_BYTES",
    "NUDGE_MIN_ITERS_LEFT",
    "NudgeReport",
    "PLAN_MARKER",
    "build_nudge",
    "clause_names_instance",
    "named_entities",
    "normalise",
    "nudge_block",
    "request_text",
    "split_by_presence",
    "strip_query_echoes",
    "subject_clauses",
    "subject_corpus",
]
