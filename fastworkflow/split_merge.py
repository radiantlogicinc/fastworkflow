"""Deterministic clean-up of a split: code only, no model, no I/O.

``merge_subtasks(user_query, subtasks)`` takes the sub-tasks the splitter's
model returned and joins the ones that are not independent jobs. The original
request is the ground truth: when everything merges back into one job, the
request itself is returned verbatim (which also restores any qualifier the
model dropped, such as "Do not apply anything").

A split is meant to be one sub-task per subject (a person, a unit, a right, a
feed, a control, a tag). The rules, applied in order:

1. **Meta lines are dropped**: a sub-task wholly inside parentheses or square
   brackets ("(No tag action is requested.)") is commentary, not a job.
2. **Contained duplicates are dropped**: a sub-task whose content words and
   subjects all occur in another sub-task repeats part of it.
3. **No new subject, no new sub-task**: a sub-task that names no subject, or
   exactly the same subjects as the sub-task before it, is joined to the one
   before it (a leading subject-less sub-task is joined to the one after it).
   "How risky is Ann Lee?" + "Which findings are driving that?" is one
   job; "Confirm whether Raj Patel is among its holders" names a person
   of its own and stays separate. A name that extends a name of the group
   before it ("Payroll_Approver" after "Payroll": the shorter name
   followed by ``_ - / . :`` or a space) is no new subject either when the
   sub-task, or the request sentence naming it, points back ("among them",
   "of it", "its", "their").
   A sub-task naming only some of the subjects before it joins them when it
   checks a change ("before and after", "the change"): "Show me the tags on
   Ann Lee before and after" after "Put `on-leave` on Ann Lee".
   Before that, **a statement is context, not a job** (3a): a request sentence that asks nothing
   ("Ann Lee leaves on Friday.", "I am closing out Payroll.") is written before the job it is about,
   and **a sub-task with the wrong record is dropped** (3b, see ``_restore_cut_subjects``).
4. **A comparison is one job**: consecutive sub-tasks that each name one
   different subject and say the same thing about it ("Open Billing ...",
   "Open Shipping ...") are joined when the request sentence they come from
   compares ("the riskier of", "compare", "side by side", ...). The joined
   sub-task is the request's own text: the sentences from the one naming the
   subjects together to the one the sub-tasks come from.

5. **Limits travel with the sub-tasks**: a request sentence that limits how
   the work is done rather than asking for a job ("Do not apply anything.",
   "Apply none of them.", "I do not need the full population.", "The matches
   are enough.") is appended to every sub-task it applies to that does not
   already hold its content words: all sub-tasks, or, when it points back
   ("them", "those", "it"), the ones carrying the job sentence before it. A
   limit clause ending a job sentence (", without applying any of them")
   goes to the sub-tasks carrying that sentence only.
6. **Dropped jobs are put back**: a job sentence whose record no sub-task names, or whose records only
   other jobs' sub-tasks name, is put back verbatim (``_restore_dropped``).
7. **One job per record**: a sub-task asking the same thing of each record of a list ("each", "both")
   is one sub-task per record (``_one_per_record``); a comparison is not cut.

Fewer than two sub-tasks left means the request is one job: ``[user_query]``.

Subjects are found without any catalog: back-ticked and quoted spans,
uids, the identifiers the workflow declares (``identifiers``: compiled patterns, e.g. from
``identifier_patterns`` of its ``_skills/_split.py``), and capitalised name spans (common words that open a
sentence, weekdays and months are not names). "X and Y Ltd", "X and Sons":
a chain of names joined by "and" or commas whose last word is a company
suffix is one name (a business unit). A sub-task mentions a subject when it
holds its full name, or for a name of three or more words, two or more of its
capitalised words in a row ("Hart Mills" of "Hart Mills and Grey
Ltd"). Names a sub-task holds that the request does not are subjects too.
"""
from __future__ import annotations

import functools
import re
import unicodedata

__all__ = ["merge_subtasks", "request_subjects", "subjects_in"]

# ------------------------------------------------------------------ text helpers
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ", " ": " ",
})


def _typographic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


#: ``_norm`` and the word-boundary patterns are cached: ``subjects_in`` tests every request
#: subject against the same text, so without the caches a long, name-dense request is
#: normalised and a pattern compiled once per subject (quadratic; fix-ajir.7).
_CACHE_SIZE = 4096


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _norm(text: str) -> str:
    """Lowercase, punctuation (but ``-`` and ``_``) to spaces, whitespace collapsed."""
    text = _typographic(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"(?<![\w])-+|-+(?![\w])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(haystack: str, needle: str) -> bool:
    """``needle`` in ``haystack`` after ``_norm``, on word boundaries (``-`` and ``_`` are word characters)."""
    n = _norm(needle)
    if not n:
        return False
    return _word(n).search(_norm(haystack)) is not None


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _word(text: str) -> re.Pattern:
    """``text`` as a whole word (``-`` and ``_`` are word characters)."""
    return re.compile(r"(?<![\w-])" + re.escape(text) + r"(?![\w-])")


_STOP = frozenset("""a an and are as at be by did do does for from give has have how i in is it its
me my of on or show tell that the their them then these this those to was what when which who whom
whose why with you your""".split())


def _content(text: str) -> set[str]:
    return {t for t in _norm(text).split() if t not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ------------------------------------------------------------------ subjects
#: Words that, capitalised at the start of a sentence, open it rather than a name
#: (question words, request verbs, function words). Stripped from a name span only there.
_COMMON = frozenset("""
a about after again all also an and another any are as at be before both but by can could did do does
each either every few first for from had has have how i if in into is it its just last let may me might
more most my next no none not now of on once only or other our please second should so some such than
that the their then there these they third this those though to two three four five until very was we
were what when where whether which while who whom whose why will with within without would yes you your
add apply audit carry check compare confirm count explain find flag get give go grab help keep list look
make mark name open pull put rank remove review run see show sweep tag take tell triage walk work
""".split())
#: Capitalised words that are never a name on their own, wherever they stand (weekdays, months).
_NOT_NAMES = _COMMON | frozenset("""monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december""".split())
#: A chain of names joined by "and" or commas that ends on one of these is one company name.
_COMPANY_SUFFIX = frozenset("ltd plc inc llc llp sons group co corp company partners associates gmbh".split())

_TICKED = re.compile(r"`([^`]+)`")
_QUOTED = re.compile(r"(?<![\w])'([^']{2,}?)'(?![\w])|\"([^\"]{2,}?)\"")
_UID = re.compile(r"\b(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|"
                  r"[0-9a-fA-F]{24,})\b")
_TOKEN = re.compile(r"[A-Za-z0-9][\w-]*(?:'s)?|[,.;:!?()\[\]\"]")


def _is_cap(word: str) -> bool:
    return word[:1].isupper() and word != "I"


def _name_like(word: str) -> bool:
    """A single word that is a name even when it opens a sentence: PayPal, HR2, Billing_Admin."""
    return ("_" in word or any(c.isdigit() for c in word) or any(c.isupper() for c in word[1:])) \
        and word[:1].isalnum()


def _strip_possessive(word: str) -> str:
    return word[:-2] if word.endswith("'s") else word


def _name_spans(text: str) -> list[str]:
    """Capitalised name spans of ``text`` (quoted, ticked and code spans already blanked)."""
    tokens = _TOKEN.findall(text)
    # (words, opens a sentence, "and" or "," joining it to the span before, else None)
    spans: list[tuple[list[str], bool, str | None]] = []
    i, sentence_start, link = 0, True, None
    while i < len(tokens):
        tok = tokens[i]
        if _is_cap(tok) and tok[:1].isalnum():
            words = [tok]
            i += 1
            while i < len(tokens) and (_is_cap(tokens[i]) or tokens[i].isdigit()) \
                    and not words[-1].endswith("'s"):
                words.append(tokens[i])
                i += 1
            spans.append((words, sentence_start, link))
            sentence_start = False
            link = None
            if i + 1 < len(tokens) and tokens[i] in ("and", ",") and _is_cap(tokens[i + 1]) \
                    and not words[-1].endswith("'s"):
                link = tokens[i]
                i += 1
            continue
        sentence_start = tok in ".;:!?()[]\""
        link = None
        i += 1

    out: list[str] = []

    def emit(words: list[str], opened: bool) -> None:
        while opened and words and words[0].lower() in _COMMON \
                and not (len(words[0]) > 1 and words[0].isupper()):  # "IT Support" keeps "IT"
            words, opened = words[1:], False
        if not words:
            return
        words = words[:-1] + [_strip_possessive(words[-1])]
        if len(words) == 1 and (words[0].lower() in _NOT_NAMES - _COMMON
                                or (opened and not _name_like(words[0]))):
            return
        out.append(" ".join(words))

    chain: list[tuple[list[str], bool, str | None]] = []

    def flush() -> None:
        # "Hart Mills and Grey Ltd", "Lewis and Sons": a chain ending on a company suffix is one name
        last = [w for w in chain[-1][0] if not w.isdigit()] if chain else []
        if len(chain) > 1 and last and _strip_possessive(last[-1]).lower() in _COMPANY_SUFFIX:
            words = list(chain[0][0])
            for part, _, joint in chain[1:]:
                if joint == ",":
                    words[-1] += ","
                else:
                    words.append("and")
                words += part
            emit(words, chain[0][1])
        else:
            for part, opened, _ in chain:
                emit(part, opened)
        chain.clear()

    for words, opened, linked in spans:
        if not linked:
            flush()
        chain.append((words, opened, linked))
    flush()
    return out


def request_subjects(text: str, identifiers: tuple[re.Pattern, ...] = ()) -> list[str]:
    """The subjects ``text`` names, longest first, without duplicates.

    ``identifiers`` are the workflow's identifier patterns: each match is a subject.
    """
    t = _typographic(text)
    found = [m for m in _TICKED.findall(t)]
    found += [a or b for a, b in _QUOTED.findall(t)]
    blank = _TICKED.sub(" . ", t)
    blank = _QUOTED.sub(" . ", blank)
    for pattern in identifiers:
        found += [m.group(0) for m in pattern.finditer(blank)]
        blank = pattern.sub(" ", blank)
    found += _UID.findall(blank)
    blank = _UID.sub(" ", blank)
    found += _name_spans(blank)
    out: dict[str, str] = {}
    for name in found:
        name = name.strip()
        if name and _norm(name) not in out:
            out[_norm(name)] = name
    return sorted(out.values(), key=lambda s: -len(_norm(s)))


def _partial_runs(subject: str) -> list[str]:
    """Parts of a name that stand for it: its tail after the first word ("Ledger" of a
    sentence-opening "Treat Ledger", "Pike" of "Owen Pike"), and for a name of three or
    more words, runs of two or more words holding at least two capitalised words."""
    return list(_cached_partial_runs(subject))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _cached_partial_runs(subject: str) -> tuple[str, ...]:
    words = subject.split()
    runs = []
    tail = [w for w in words[1:]]
    if tail and _is_cap(tail[0]) and not all(w.lower() in _COMPANY_SUFFIX or w.isdigit() for w in tail) \
            and not (len(tail) == 1 and tail[0].lower() in _NOT_NAMES):
        runs.append(" ".join(tail))
    if len(words) < 3:
        return tuple(runs)
    for a in range(len(words)):
        for b in range(a + 2, len(words) + 1):
            run = words[a:b]
            if len(run) == len(words):
                continue
            caps = [w for w in run if _is_cap(w)]
            if len(caps) >= 2 and not all(w.lower() in _COMPANY_SUFFIX for w in caps) \
                    and _is_cap(run[0]) and _is_cap(run[-1]):
                runs.append(" ".join(run))
    return tuple(sorted(set(runs), key=lambda r: -len(r)))


def _blank(text: str, name: str) -> str:
    return _word(_norm(name)).sub(" ", _norm(text))


def subjects_in(text: str, request_subjects_: list[str],
                identifiers: tuple[re.Pattern, ...] = ()) -> frozenset[str]:
    """The subjects ``text`` mentions, as normalised names.

    A request subject counts when its full name occurs, or a part of it that no
    other request subject shares (see ``_partial_runs``). A name of the text
    that is no request subject counts as a subject of its own.
    """
    found: set[str] = set()
    rest = _norm(text)
    for subject in request_subjects_:
        if _contains(rest, subject):
            found.add(_norm(subject))
            rest = _blank(rest, subject)
    # a part stands for its name only when no other request subject holds it or is it
    parts = {s: _partial_runs(s) for s in request_subjects_}
    owners: dict[str, int] = {}
    for s, runs in parts.items():
        for run in runs:
            owners[_norm(run)] = owners.get(_norm(run), 0) + 1
    for s in request_subjects_:
        owners[_norm(s)] = owners.get(_norm(s), 0) + 1
    written = _typographic(text)
    for subject in request_subjects_:
        if _norm(subject) in found:
            continue
        for run in parts[subject]:
            # a part stands for the name only written as a name ("Administrator", not "administrator rights")
            if owners[_norm(run)] == 1 and _contains(rest, run) and _written_as_is(written, run):
                found.add(_norm(subject))
                rest = _blank(rest, run)
    for name in request_subjects(text, identifiers):
        if _contains(rest, name):
            found.add(_fragment_owner(name, request_subjects_) or _norm(name))
            rest = _blank(rest, name)
    return frozenset(found)


def _written_as_is(text: str, run: str) -> bool:
    """``run`` occurs in ``text`` with its own capitals, on word boundaries."""
    return re.search(r"(?<![\w-])" + re.escape(run) + r"(?![\w-])", text) is not None


def _fragment_owner(name: str, request_subjects_: list[str]) -> str | None:
    """The one request subject ``name`` is a run of words of ("Sons" of "Hart and Sons and Sons"), else None.

    A name of a sub-task that the request does not name on its own is that subject cut short.
    """
    words = _norm(name).split()
    owners = []
    for subject in request_subjects_:
        whole = _norm(subject).split()
        if words != whole and any(whole[i:i + len(words)] == words for i in range(len(whole) - len(words) + 1)):
            owners.append(_norm(subject))
    return owners[0] if len(owners) == 1 else None


# ------------------------------------------------------------------ rules
_META = re.compile(r"^\s*(?:\(.*\)|\[.*\])\s*[.]?\s*$", re.DOTALL)


def _is_meta(text: str) -> bool:
    return bool(_META.match(text))


def _invents_value(text: str, user_query: str, known: list[str]) -> bool:
    """A sub-task quoting a value (`x`, 'x', "x") that the request does not give is the model's invention.

    The request gives the value when it quotes it too, when the value is or holds one of its
    subjects ("Tag 'Ann Lee'" quotes a name the request gives unquoted), or when the value is a
    phrase of several words the request uses. A one-word value the request only uses as a word
    ("offboarding" of "an offboarding sweep") is not given.
    """
    t = _typographic(text)
    request_quoted = {_norm(q) for q in _TICKED.findall(_typographic(user_query))} | {
        _norm(a or b) for a, b in _QUOTED.findall(_typographic(user_query))}
    for value in _TICKED.findall(t) + [a or b for a, b in _QUOTED.findall(t)]:
        v = _norm(value)
        if not v or v in request_quoted:
            continue
        if any(_norm(k) == v or _contains(value, k) for k in known):
            continue
        if len(v.split()) > 1 and _contains(user_query, value):
            continue
        return True
    return False


def _source_sentence(text: str, sentences: list[str], subj: frozenset[str], sole: bool = False) -> int | None:
    """The request sentence ``text`` is taken from: the one holding the largest share (at least half)
    of its content words once the subjects are blanked out of both; None when no sentence does (or,
    with ``sole``, when another sentence holds as large a share)."""
    words = _template(text, subj)
    if not words:
        return None
    cover = [len(words & _template(s, subj)) / len(words) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: cover[i])
    if sole and sum(c == cover[best] for c in cover) > 1:
        return None
    return best if cover[best] >= SOURCE_COVERAGE else None


def _restore_cut_subjects(user_query: str, texts: list[str], known: list[str],
                          identifiers: tuple[re.Pattern, ...], wrong_records_only: bool = False) -> list[str]:
    """Rule 6a: the sub-tasks taken from a job sentence carry all of that sentence's subjects.

    "List the logins nobody owns on NORTHFEED and on SOUTHFEED" cut to "... on NORTHFEED" loses the
    SOUTHFEED job even when a later sub-task ("Open the riskier of NORTHFEED and SOUTHFEED") names
    SOUTHFEED, so a request-wide check does not see it. Per sentence: when a subject of the sentence is
    named by none of the sub-tasks taken from it (``_source_sentence``), the first of them becomes the
    sentence verbatim and the others are dropped. Only sub-tasks made of the sentence's own words are
    replaced (one that also carries another sentence's words is left alone, as is a limit sentence),
    and a sub-task that is already the whole sentence is never touched, so nothing is duplicated.

    A sub-task taken from a sentence that names none of the sentence's subjects but a record named
    elsewhere in the request has the wrong record ("Audit Billing" made from "Audit Payroll_Admin and
    Payroll_Clerk", Billing named in another sentence): it is the sentence verbatim when the other
    sub-tasks taken from it leave some of its subjects out, else it is dropped. ``wrong_records_only``
    does only this (rule 3b, before rule 3 can join such a sub-task to the one naming its record).
    """
    sentences = _sentences(user_query)
    subj = [subjects_in(t, known, identifiers) for t in texts]
    sources = [_source_sentence(t, sentences, s) for t, s in zip(texts, subj)]
    replace: dict[int, str] = {}
    drop: set[int] = set()
    for k, sentence in enumerate(sentences):
        if _is_limit(sentence):
            continue
        own = subjects_in(sentence, known, identifiers)
        taken = [i for i, src in enumerate(sources) if src == k]
        if not own or not taken:
            continue
        elsewhere = frozenset().union(*(subjects_in(s, known, identifiers) for j, s in enumerate(sentences) if j != k))
        strays = [i for i in taken if subj[i] and not subj[i] & own and subj[i] <= elsewhere
                  and _source_sentence(texts[i], sentences, subj[i], sole=True) == k]
        kept = [i for i in taken if i not in strays]
        missing = own - frozenset().union(*(subj[i] for i in kept))
        text = _LEADING_LINK.sub("", sentence.strip())
        text = text[:1].upper() + text[1:]
        if strays:
            if missing:             # the sentence takes the place of all the sub-tasks taken from it
                replace[taken[0]] = text
                strays = taken[1:]
            drop |= set(strays)
            continue
        if wrong_records_only or not missing or not all(_content(texts[i]) <= _content(sentence) for i in taken):
            continue
        replace[taken[0]] = text
        drop |= set(taken[1:])
    return [replace.get(i, t) for i, t in enumerate(texts) if i not in drop]


def _restore_dropped(user_query: str, texts: list[str], known: list[str],
                     identifiers: tuple[re.Pattern, ...]) -> list[str]:
    """Rule 6: a request sentence naming a subject that no sub-task names is a job the model dropped.

    It is put back, verbatim, after the last sub-task that comes from an earlier sentence. A limit
    sentence is not a job (rule 5 carries it). Only subjects of the request count. First, a sub-task
    cut from a sentence that left out some of that sentence's subjects is given the whole sentence
    back (``_restore_cut_subjects``), even when another sub-task names the subjects it left out.
    A sentence that asks for something (``_asks``) is put back as well when its records are named only
    by other jobs' sub-tasks (``_carried_job``): "Audit the right X across the systems it appears in"
    is not done by "Confirm whether Ann Lee holds X".
    """
    texts = _restore_cut_subjects(user_query, texts, known, identifiers)
    named = set().union(*(subjects_in(t, known, identifiers) for t in texts))
    sentences = _sentences(user_query)
    out = list(texts)
    # last sentence first: a sentence naming the records of later ones ("the picture of X and Y") is not
    # put back when a later sentence about them is
    for k, sentence in reversed(list(enumerate(sentences))):
        if _is_limit(sentence):
            continue
        own = subjects_in(sentence, known, identifiers)
        if not own or (own & named and not (_asks(sentence) and not _carried_job(sentence, own, out, known,
                                                                                   identifiers))):
            continue
        before = " ".join(sentences[:k])
        position = 0
        for i, t in enumerate(out):
            if _content(t) & _content(before) and not (_content(t) <= _content(" ".join(sentences[k:]))):
                position = i + 1
        out.insert(position, sentence.strip())
        named |= own
    return out


def _drop_contained(items: list[str], known: list[str], identifiers: tuple[re.Pattern, ...]) -> list[str]:
    """Drop a sub-task whose content words and subjects all occur in another (the earlier one wins a tie)."""
    words = [_content(t) for t in items]
    subj = [subjects_in(t, known, identifiers) for t in items]
    keep = []
    for i, w in enumerate(words):
        contained = any(j != i and w <= words[j] and subj[i] <= subj[j] and (w != words[j] or j < i)
                        for j in range(len(items)))
        if not contained or not w:
            keep.append(i)
    return [items[i] for i in keep]


def _join(texts: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for t in texts:
        w = _content(t)
        if out and w and w <= seen:
            continue  # nothing this part adds
        t = t.strip()
        if out and not re.search(r"[.?!]$", out[-1]):
            out[-1] += "."
        out.append(t[:1].upper() + t[1:] if out else t)
        seen |= w
    return " ".join(out)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z`'\"])", _typographic(text))
    return [p for p in parts if p.strip()]


_COMPARES = re.compile(
    r"\b(?:compar\w*|side by side|versus|vs|between|which of|whichever|"
    r"riskier|riskiest|safer|safest|higher|highest|lower|lowest|larger|largest|smaller|smallest|"
    r"bigger|biggest|fewer|fewest)\b", re.IGNORECASE)
_LEADING_LINK = re.compile(r"^(?:then|and|also|next|finally|so)\b[\s,]*", re.IGNORECASE)
#: Two sub-tasks say the same thing about different subjects at or above this content-word overlap.
SAME_TEMPLATE = 0.75


def _template(text: str, subjects: frozenset[str]) -> set[str]:
    rest = _norm(text)
    for s in sorted(subjects, key=len, reverse=True):
        rest = _blank(rest, s)
    return _content(rest)


def _comparison_text(user_query: str, run: list[tuple[str, frozenset[str]]]) -> str | None:
    """The request's own text for a run of per-subject sub-tasks it asks to compare; None when it does not."""
    sentences = _sentences(user_query)
    subjects = frozenset().union(*(s for _, s in run))
    templates = [_template(t, s) for t, s in run]
    if any(_jaccard(templates[0], t) < SAME_TEMPLATE for t in templates[1:]):
        return None
    scores = [_jaccard(templates[0], _template(s, subjects)) for s in sentences]
    source = max(range(len(sentences)), key=lambda i: scores[i])
    if not _COMPARES.search(sentences[source]):
        return None
    together = [i for i, s in enumerate(sentences) if all(_contains(s, x) for x in subjects)]
    if not together:
        return None
    start = max((i for i in together if i <= source), default=None)
    if start is None:
        start = min(together)
        lo, hi = source, start
    else:
        lo, hi = start, source
    text = " ".join(sentences[lo:hi + 1])
    text = _LEADING_LINK.sub("", text)
    return text[:1].upper() + text[1:]


# ------------------------------------------------------------------ related names (rule 3)
#: Characters that join a name to a longer name built on it: "Payroll" -> "Payroll_Approvers",
#: "Finance" -> "Finance/Payroll", "core" -> "core.db". (``_norm`` keeps ``_`` and ``-`` and turns
#: ``/ . :`` into spaces, so on normalised names the separators are ``_``, ``-`` and a space.)
_NAME_SEPARATORS = "_-/.: "
#: Prepositions that place one thing among or within others ("among them", "is ann lee among ...").
_PLACING = r"among|amongst|of|in|on|from|within|under|inside|across|between"
#: A back-reference to something named earlier: "among them", "of it", "its", "their".
_BACK_REFERENCE = re.compile(
    r"\b(?:(?:" + _PLACING + r")\s+(?:them|those|these|it)\b"
    r"|its|their)\b", re.IGNORECASE)


def _extends(longer: str, shorter: str) -> bool:
    """``longer`` is ``shorter`` followed by a separator and more ("gitlab_infrastructure manager", "gitlab")."""
    return len(longer) > len(shorter) + 1 and longer.startswith(shorter) and longer[len(shorter)] in _NAME_SEPARATORS


def _extends_group(subj: frozenset[str], group: frozenset[str]) -> bool:
    """Every subject of ``subj`` is one of ``group`` or a name extending one of them, and one extends."""
    if not subj or not group:
        return False
    related = [s in group or any(_extends(s, g) for g in group) for s in subj]
    return all(related) and any(s not in group for s in subj)


#: A sub-task that asks to see the effect of a change ("before and after", "so I can see the change
#: landed"): it checks the job before it rather than being a job of its own.
_OBSERVES_CHANGE = re.compile(
    r"\bbefore\s+and\s+after\b|\b(?:the|that|this)\s+change\b"
    r"|\bafter\s+(?:the|that|this)\s+(?:change|update|edit)\b", re.IGNORECASE)


def _observes_change(text: str) -> bool:
    return _OBSERVES_CHANGE.search(_typographic(text)) is not None


def _refers_back(text: str) -> bool:
    return _BACK_REFERENCE.search(_typographic(text)) is not None


def _request_refers_back(user_query: str, names: frozenset[str]) -> bool:
    """A request sentence naming one of ``names`` points back ("among them"): the model may have dropped it."""
    return any(_refers_back(sentence) for sentence in _sentences(user_query)
               if any(name in _norm(sentence) for name in names))


# ------------------------------------------------------------------ limits (rule 5)
#: A sentence that limits how the work is done instead of asking for a job of its own. Generic English:
#: a negated imperative ("Do not apply anything", "Never open them"), "<verb> none/nothing" ("Apply none
#: of them"), "I do not need ...", "... is enough / will do / suffices", an "Only/Just <noun phrase>"
#: fragment ("Only the first five"), "Without ...ing ...", "Leave ... as is / alone / untouched".
_LIMIT_SENTENCE = re.compile(r"""^(?:(?:but|and|so|also|then|please)\b[\s,]*)*(?:
      (?:do\s+not|don't|dont|never|no\s+need\s+to|there\s+is\s+no\s+need\s+to|avoid)\b
          (?!\s+(?:forget|fail|miss|skip|overlook|neglect|omit|leave\s+out)\b)
    | [a-z]+\s+(?:none|nothing)\b
    | (?:i|we)\s+(?:do\s+not|don't|dont|won't|will\s+not)\s+need\b
    | .*\b(?:is|are|will\s+be|would\s+be)\s+(?:enough|sufficient|plenty)\b
    | .*\b(?:suffices?|will\s+do)\s*[.!]?$
    | (?:only|just)\s+(?:the|a|an|one|two|three|four|five|\d+|those|these|them|that|this)\b
    | without\s+[a-z]+ing\b
    | leave\b.*\b(?:as\s+is|as\s+they\s+are|as\s+it\s+is|alone|untouched|unchanged)\b
)""", re.IGNORECASE | re.VERBOSE)
#: The same kinds of limit as a clause at the end of a job sentence: ", without applying any of them",
#: ", with none applied", ", but do not open them", ", but I do not need the full list".
_LIMIT_CLAUSE = re.compile(r"""[,;]\s*(
      (?:but\s+|and\s+)?without\s+[a-z]+ing\b
    | (?:but\s+|and\s+)?with\s+(?:none|nothing)\s+[a-z]+ed\b
    | (?:but\s+|and\s+)(?:do\s+not|don't|never)\s+(?!forget|fail|miss|skip|overlook|neglect|omit)[a-z]+
    | (?:but\s+)?(?:i|we)\s+(?:do\s+not|don't|won't|will\s+not)\s+need\b
)[^,;.?!]*""", re.IGNORECASE | re.VERBOSE)
#: A limit that points back at what came before it ("Apply none of them") qualifies that sentence's job.
_POINTS_BACK = re.compile(r"\b(?:it|its|them|they|their|those|these|that|this|each|both|either)\b", re.IGNORECASE)
#: A sentence's job is carried by a sub-task holding at least this share of its content words.
SOURCE_COVERAGE = 0.5


def _is_limit(sentence: str) -> bool:
    s = sentence.strip()
    return "?" not in s and _LIMIT_SENTENCE.match(s) is not None


# ------------------------------------------------------------------ asks and statements (rules 3a, 6)
#: Verbs that open a request or a clause of one ("Audit ...", ": work it through", ", so triage ...").
_REQUEST_VERBS = frozenset("""add answer apply assess audit carry check clear close compare confirm count describe
explain find finish flag get give go grab inspect list look make mark open pull put rank recertify remove review
run see show start summarise summarize sweep tag take tell treat triage untag verify walk work""".split())
#: Words that ask wherever they stand: a question, "I want", "can you", "please", "show me", "needs auditing".
_ASK_WORDS = re.compile(
    r"\?|\b(?:i|we)\s+(?:want|need|would\s+like|'d\s+like)\b|\b(?:can|could|would|will)\s+you\b|\bplease\b"
    r"|\b(?:show|give|tell|send|get|let)\s+me\b|\bneeds?\s+(?:to|[a-z]+ing)\b", re.IGNORECASE)
#: A clause opening on one of these asks for it: ": what Ann Lee can reach", ", who holds it".
_WH = frozenset("what which who whom whose how why where when whether".split())
_CLAUSE = re.compile(r"\s*[:;,]\s*|\s+-\s+|\s+(?:so|then|and|but)\s+", re.IGNORECASE)
_CLAUSE_LINK = re.compile(r"^(?:(?:then|and|also|so|finally|next|separately|but|now|please|first|just)\b[\s,]*)+",
                          re.IGNORECASE)
#: A finite verb in a sentence's first clause makes it a statement when it asks nothing ("Ann Lee leaves
#: on Friday.", "I am closing out Payroll."); a noun phrase ("Quarterly review of Payroll: ...") is not one.
_FINITE = re.compile(r"\b(?:is|are|am|was|were|has|have|had|will|keeps?|leaves?|left|starts?|comes?|closes?|"
                     r"holds?|joins?|moves?|names|asked)\b", re.IGNORECASE)


def _asks(sentence: str) -> bool:
    """The sentence asks for something: a question, a request phrase, or a clause opening on a request verb
    or a question word."""
    s = _typographic(sentence)
    if _ASK_WORDS.search(s):
        return True
    for clause in _CLAUSE.split(s):
        words = _CLAUSE_LINK.sub("", clause).split()
        if words and (words[0].lower() in _REQUEST_VERBS or words[0].lower() in _WH):
            return True
    return False


def _is_statement(sentence: str) -> bool:
    """The sentence says something and asks nothing ("Ann Lee leaves on Friday."): context, not a job."""
    s = _typographic(sentence)
    return not _asks(s) and not _is_limit(s) and _FINITE.search(re.split(r"[:;,]|\s-\s", s)[0]) is not None


def _carried_job(sentence: str, own: frozenset[str], texts: list[str], known: list[str],
                 identifiers: tuple[re.Pattern, ...]) -> bool:
    """Some sub-task does the sentence's job: it names a subject of the sentence and either names no record
    of another job or holds SOURCE_COVERAGE of the sentence's words other than names. "Confirm whether Ann
    Lee holds X" names X but is Ann Lee's job: it does not carry "Audit the right X across the systems"."""
    words = _template(sentence, frozenset(_norm(k) for k in known))
    for text in texts:
        subj = subjects_in(text, known, identifiers)
        if not subj & own:
            continue
        if subj <= own or not words or len(words & _content(text)) / len(words) >= SOURCE_COVERAGE:
            return True
    return False


def _attach_context(user_query: str, items: list[str], known: list[str],
                    identifiers: tuple[re.Pattern, ...]) -> list[str]:
    """Rule 3a: a statement of the request is context for the jobs, never a job of its own.

    - A sub-task that is a statement of the request ("Ann Lee's login is a standard account.") is written
      before the sub-task after it when that one names no record or all of the statement's records.
    - A statement naming a record no sub-task names ("I am closing out Payroll for the year.") is written
      before the sub-tasks taken from the next sentence that has any ("Walk the whole unit: ..."), so they
      keep the record they point back to. With none, rule 6 puts it back.
    """
    out: list[str] = []
    for text in items:
        if out and _is_statement(out[-1]) and _content(out[-1]) <= _content(user_query):
            context = subjects_in(out[-1], known, identifiers)
            subj = subjects_in(text, known, identifiers)
            if not subj or context <= subj:
                out[-1] = _join([out[-1], text])
                continue
        out.append(text)
    sentences = _sentences(user_query)
    named = set().union(*(subjects_in(t, known, identifiers) for t in out))
    for k, sentence in enumerate(sentences):
        own = subjects_in(sentence, known, identifiers)
        if not own or own & named or not _is_statement(sentence):
            continue
        sources = [_source_sentence(t, sentences, subjects_in(t, known, identifiers)) for t in out]
        later = sorted({s for s in sources if s is not None and s > k})
        if not later:
            continue
        out = [_join([sentence, t]) if src == later[0] else t for t, src in zip(out, sources)]
        named |= own
    return out


# ------------------------------------------------------------------ one job per record (rule 7)
#: Words that ask the same thing of each record of a list: "each", "both", "every one".
_DISTRIBUTIVE = re.compile(r"\b(?:each|both|every\s+one)\b", re.IGNORECASE)
#: Words that point at the whole list; in a record's own sub-task they become its name.
_EACH_OF = re.compile(r"\b(?:each|both|all)\s+of\s+(?:them|those|these)\b|\beach\s+one\b", re.IGNORECASE)
#: What stands between two records of a list: ", ", " and ", ", and ", " and on ", "', '".
_LIST_GAP = re.compile(r"^['\"`]?(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?:on\s+|the\s+)?['\"`]?$", re.IGNORECASE)


def _listed(text: str, known: list[str]) -> tuple[int, int, list[str]] | None:
    """``(start, end, names)`` of the one list of two or more request subjects in ``text``, else None."""
    spans: list[tuple[int, int, str]] = []
    for name in known:                                   # longest first: "X and Sons Ltd" before "X"
        for m in re.finditer(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text, re.IGNORECASE):
            if not any(a < m.end() and m.start() < b for a, b, _ in spans):
                spans.append((m.start(), m.end(), m.group(0)))
    spans.sort()
    runs: list[list[tuple[int, int, str]]] = []
    for span in spans:
        if runs and _LIST_GAP.match(text[runs[-1][-1][1]:span[0]]):
            runs[-1].append(span)
        else:
            runs.append([span])
    lists = [r for r in runs if len(r) > 1]
    if len(lists) != 1:
        return None
    return lists[0][0][0], lists[0][-1][1], [name for _, _, name in lists[0]]


def _one_per_record(texts: list[str], known: list[str]) -> list[str]:
    """Rule 7: a sub-task asking the same thing of each record of a list is one sub-task per record.

    "Audit Payroll_Admin and Payroll_Clerk wherever they appear, each with its feed" becomes "Audit
    Payroll_Admin wherever they appear, each with its feed" and the same for Payroll_Clerk: the list is
    replaced by one record's name, and "each of them" by that name too. Only a sub-task with one such
    list and a word that distributes ("each", "both", "every one") is cut, and never a comparison.
    """
    out: list[str] = []
    for text in texts:
        t = _typographic(text)
        found = _listed(t, known) if _DISTRIBUTIVE.search(t) and not _COMPARES.search(t) else None
        if found is None:
            out.append(text)
            continue
        start, end, names = found
        out += [_EACH_OF.sub(name, t[:start] + name + t[end:]) for name in names]
    return out


def _carried(text: str, limit: str) -> bool:
    words = _content(limit)
    return bool(words) and words <= _content(text)


def _holders(sentence: str, texts: list[str]) -> list[int]:
    """The sub-tasks carrying ``sentence``'s job: those holding SOURCE_COVERAGE of its content words,
    else the one holding the most (none when no sub-task shares a word)."""
    words = _content(sentence)
    if not words:
        return []
    cover = [len(words & _content(t)) / len(words) for t in texts]
    held = [i for i, c in enumerate(cover) if c >= SOURCE_COVERAGE]
    if held:
        return held
    best = max(range(len(texts)), key=lambda i: cover[i])
    return [best] if cover[best] > 0 else []


def _append_sentence(text: str, sentence: str) -> str:
    text = text.rstrip()
    if not re.search(r"[.?!]$", text):
        text += "."
    return f"{text} {sentence}"


def _append_clause(text: str, clause: str) -> str:
    text = text.rstrip()
    end = text[-1] if text and text[-1] in ".?!" else "."
    return f"{text.rstrip('.?!').rstrip(',;')}, {clause}{end}"


def _carry_limits(user_query: str, texts: list[str], known: list[str],
                  identifiers: tuple[re.Pattern, ...]) -> list[str]:
    """Rule 5: give every sub-task the request's limits that apply to it and that it does not carry.

    - A limit sentence ("Do not apply anything.", "I do not need the full population.") that points
      back at nothing applies to the whole request: every sub-task. One that points back ("Apply
      none of them.") qualifies the job of the nearest job sentence before it: the sub-tasks carrying
      that sentence (see ``_holders``).
    - A limit clause at the end of a job sentence (", without applying any of them") applies to the
      sub-tasks carrying that sentence only. When no sub-task carries the rest of its sentence
      ("Answer per person; I do not need the full population."), it is a limit sentence of its own.
    - A limit that names subjects applies only to sub-tasks naming all of them; one naming a subject
      no sub-task names is left alone (it may be a job).
    A sub-task already carries a limit when it holds all of its content words.
    """
    out = list(texts)
    sentences = _sentences(user_query)
    limits = [_is_limit(s) for s in sentences]
    subj = [subjects_in(t, known, identifiers) for t in out]

    def scoped(limit: str, targets: list[int]) -> list[int]:
        named = subjects_in(limit, known, identifiers)
        if named:
            targets = [i for i in targets if named <= subj[i]]
        return [i for i in targets if not _carried(out[i], limit)]

    def as_sentence(limit: str) -> str:
        text = _LEADING_LINK.sub("", re.sub(r"^(?:but|please)\b[\s,]*", "", limit.strip(), flags=re.I))
        text = text[:1].upper() + text[1:]
        return text if re.search(r"[.?!]$", text) else text + "."

    def whole_request_limit(k: int, limit: str) -> None:
        antecedent = next((j for j in range(k - 1, -1, -1) if not limits[j]), None)
        if _POINTS_BACK.search(limit) and antecedent is not None:
            targets = _holders(sentences[antecedent], out)
        else:
            targets = list(range(len(out)))
        text = as_sentence(limit)
        for i in scoped(text, targets):
            out[i] = _append_sentence(out[i], text)

    for k, sentence in enumerate(sentences):
        if limits[k]:
            whole_request_limit(k, sentence)
            continue
        for m in _LIMIT_CLAUSE.finditer(_typographic(sentence)):
            clause = m.group(0).lstrip(",; ").strip()
            job = _typographic(sentence).replace(m.group(0), " ")
            holders = _holders(job, out)
            if not holders:   # the rest of the sentence is no sub-task's job ("Answer per person; I do not need ...")
                whole_request_limit(k, clause)
                continue
            for i in scoped(clause, holders):
                out[i] = _append_clause(out[i], clause)
    return out


def merge_subtasks(user_query: str, subtasks: list[str],
                   identifiers: tuple[re.Pattern, ...] = ()) -> list[str]:
    """``subtasks`` with the parts that are not independent jobs joined; ``[user_query]`` when one job is left.

    Pure: no model and no I/O. Fewer than two sub-tasks are returned unchanged. ``identifiers``
    are the workflow's identifier patterns (``identifier_patterns``); each match is a subject.
    """
    items = [t for t in subtasks if t and t.strip()]
    if len(items) < 2:
        return list(subtasks)
    known = request_subjects(user_query, identifiers)
    items = [t for t in items if not _is_meta(t) and not _invents_value(t, user_query, known)]
    items = _drop_contained(items, known, identifiers)
    if len(items) < 2:
        return [user_query]
    # rule 3a: a statement is context for the jobs, not a job
    items = _attach_context(user_query, items, known, identifiers)
    # rule 3b: a sub-task that put another sentence's record into a sentence's job is not a job
    items = _restore_cut_subjects(user_query, items, known, identifiers, wrong_records_only=True)

    # rule 3: no new subject, no new sub-task (a name extending one of the group's names, with a
    # back-reference, is no new subject either; nor is part of the group's subjects checking a change)
    groups: list[tuple[list[str], frozenset[str]]] = []
    for text in items:
        subj = subjects_in(text, known, identifiers)
        if groups and (not subj or subj == groups[-1][1]):
            groups[-1][0].append(text)
        elif groups and subj < groups[-1][1] and _observes_change(text):
            groups[-1][0].append(text)
        elif groups and _extends_group(subj, groups[-1][1]) and (
                _refers_back(text) or _request_refers_back(user_query, subj - groups[-1][1])):
            groups[-1] = (groups[-1][0] + [text], groups[-1][1] | subj)
        elif groups and not groups[-1][1]:          # a leading subject-less part joins the next one
            groups[-1] = (groups[-1][0] + [text], subj)
        else:
            groups.append(([text], subj))
    merged = [(_join(texts), subj) for texts, subj in groups]
    if len(merged) < 2:
        return [user_query]

    # rule 4: a comparison is one job
    out: list[tuple[str, frozenset[str]]] = []
    i = 0
    while i < len(merged):
        best = None
        if len(merged[i][1]) == 1:
            j = i + 1
            while j < len(merged) and len(merged[j][1]) == 1 and \
                    merged[j][1] not in [m[1] for m in merged[i:j]]:
                j += 1
            for end in range(j, i + 1, -1):           # the longest run the request compares
                text = _comparison_text(user_query, merged[i:end])
                if text is not None:
                    best = (end, text)
                    break
        if best:
            end, text = best
            out.append((text, frozenset().union(*(m[1] for m in merged[i:end]))))
            i = end
        else:
            out.append(merged[i])
            i += 1
    if len(out) < 2:
        return [user_query]
    # rule 6: a job sentence of the request whose record no sub-task names is put back
    texts = _restore_dropped(user_query, [t for t, _ in out], known, identifiers)
    # rule 7: the same thing asked of each record of a list is one sub-task per record
    texts = _one_per_record(texts, known)
    # rule 5: limits on the whole request travel with every sub-task they apply to
    return _carry_limits(user_query, texts, known, identifiers)


# ------------------------------------------------------------------ referents (the caller resolves)
_DEMONSTRATIVE = re.compile(r"\b(?:that|this)\s+(?!(?:it|its|they|them|their|he|she|his|her|is|are|was|"
                            r"were|has|have|the|a|an)\b)([a-z][\w-]*)\b")
_ITS = re.compile(r"\bits\b")


def unnamed_subjects(user_query: str, text: str, identifiers: tuple[re.Pattern, ...] = ()) -> list[str]:
    """The request's subjects that ``text`` does not name: what a sub-task's "it", "its" or
    "that team" may stand for. Choosing among them is the caller's (a model); this is code only.

    Left out too: the records listed in one request sentence with a record ``text`` names ("Ann Lee
    and Bob Kerr leave, so each of them ..."). A sibling in that list is never the referent of the
    sub-task cut for the other one."""
    known = request_subjects(user_query, identifiers)
    named = subjects_in(text, known, identifiers)
    siblings = set()
    for sentence in _sentences(user_query):
        found = _listed(sentence, known)
        if found and any(_norm(name) in named for name in found[2]):
            siblings |= {_norm(name) for name in found[2]}
    return [k for k in known if _norm(k) not in named and _norm(k) not in siblings]


#: "the <word>": a definite description that may stand for a record named earlier ("the customer").
_DEFINITE = re.compile(r"\b[Tt]he\s+([a-z][a-z-]*)\b(?!\s+of\b)")
#: Words after "the" that open a longer description rather than name a kind of record ("the full list").
_NOT_KINDS = frozenset("""same whole full entire rest latter former following above below current latest new old
main top bottom one ones""".split())


def definite_descriptions(text: str) -> list[str]:
    """The "the <word>" phrases of ``text`` that may stand for one record named earlier, in order.

    Which of them, if any, stands for the sub-task's referent is the caller's choice (a model):
    "the customer" in "Tag the customer `x`" stands for a person, "the label" in "Put the label `x`
    on her" does not, and nothing in the words alone tells the two apart. Left out, as they cannot
    stand for one record named earlier: a plural ("the logins"), a common word ("the first", "the
    other", "the full"), "the <word> of ..." ("the riskier of"), and a phrase followed by a name
    ("the unit Payroll").
    """
    out: list[str] = []
    for m in _DEFINITE.finditer(text):
        word = m.group(1)
        if word in _COMMON or word in _STOP or word in _NOT_KINDS or (word.endswith("s") and not word.endswith("ss")):
            continue
        after = text[m.end():].lstrip()
        if after[:1].isupper() or _UID.match(after):
            continue
        phrase = f"the {word}"
        if phrase not in out:
            out.append(phrase)
    return out


def _write_after(text: str, description: str, subject: str) -> str | None:
    """``text`` with ``subject`` written right after the first ``description`` ("the customer Ann Lee")."""
    m = re.search(r"\b" + re.escape(description) + r"\b(?!'s\b)", text, flags=re.IGNORECASE)
    if m is None:
        m = re.search(r"\b" + re.escape(description) + r"'s\b", text, flags=re.IGNORECASE)
        if m is None:
            return None
        # "the customer's tags" -> "the customer Ann Lee's tags"
        return f"{text[:m.end() - 2]} {subject}'s{text[m.end():]}"
    return f"{text[:m.end()]} {subject}{text[m.end():]}"


def with_referent(text: str, subject: str, description: str | None = None) -> str:
    """``text`` with the record its pointing-back words stand for written in.

    ``description`` is the definite description standing for the record, when the caller found one
    (see ``definite_descriptions``): the subject is written right after it, "Tag the customer Ann Lee
    `x`", so a value after it is not read as the record's name. When the sub-task has exactly one
    pointing phrase, it is replaced as well: "that team" becomes the subject, "its holders" becomes
    "<subject>'s holders" (a plural "those rights" may mean something else and does not count).
    With no description and none or several pointing phrases, which one means the subject is
    unknown, so the subject is appended instead: "Triage that request and that team's roster (for
    Ann Lee)."
    """
    if _contains(text, subject):
        return text
    described = _write_after(text, description, subject) if description else None
    demonstratives = _DEMONSTRATIVE.findall(text)
    possessives = _ITS.findall(text)
    if len(demonstratives) + len(possessives) == 1:
        text = described or text
        if demonstratives:
            return _DEMONSTRATIVE.sub(subject, text, count=1)
        return _ITS.sub(f"{subject}'s", text, count=1)
    if described:
        return described
    body = text.rstrip()
    end = body[-1] if body and body[-1] in ".?!" else ""
    return f"{body[:len(body) - len(end)]} (for {subject}){end or '.'}"
