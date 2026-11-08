"""Resolve step numbers and goal phrases against the proposal under review."""
from __future__ import annotations

import re
from typing import Optional

import fastworkflow

_STOPWORDS = frozenset({
    "a", "an", "the", "step", "steps", "item", "items", "of", "to", "for", "and", "or",
})

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
}

_STEP_PREFIX = re.compile(r"(?i)^(?:step|item)\s+(\d+)\s*$")
_HASH_STEP = re.compile(r"^#(\d+)\s*$")
_INTEGER = re.compile(r"^\d+\s*$")
_STEP_PHRASE = re.compile(
    r"^(?:the\s+)?(?:(?:step|item|part)\s+)?"
    r"(?P<tok>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
    r"seventh|7th|eighth|8th|last|#?\d+)"
    r"(?:\s+(?:step|item|part|one))?$",
    re.IGNORECASE,
)
_STEP_LEADING = re.compile(
    r"^(?:the\s+)?(?:(?:step|item|part)\s+#?(?P<num>\d+)|"
    r"(?P<ord>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|"
    r"seventh|7th|eighth|8th)\s+(?:step|item|part|one))\b",
    re.IGNORECASE,
)


def _token_to_index(tok: str) -> Optional[int]:
    cleaned = tok.strip().lower().lstrip("#")
    if cleaned == "last":
        return -1
    if cleaned in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[cleaned]
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _proposal_item_goals(proposal) -> list[str]:
    if proposal is None:
        return []
    return [getattr(item, "goal", "") or "" for item in getattr(proposal, "items", ())]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token and token not in _STOPWORDS}


def _strip_surrounding_quotes(text: str) -> str:
    stripped = (text or "").strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1].strip()
    return stripped


def _unique_goal_index(goals: list[str], raw: str) -> Optional[int]:
    lowered = _strip_surrounding_quotes(raw).lower()
    if not lowered:
        return None
    substring_hits = [index + 1 for index, goal in enumerate(goals) if goal and lowered in goal.lower()]
    if len(substring_hits) == 1:
        return substring_hits[0]
    raw_tokens = _tokenize(lowered)
    if not raw_tokens:
        return None
    token_hits: list[int] = []
    for index, goal in enumerate(goals):
        goal_tokens = _tokenize(goal)
        if goal_tokens and raw_tokens & goal_tokens:
            token_hits.append(index + 1)
    if len(token_hits) == 1:
        return token_hits[0]
    return None


def _parse_explicit_index(raw: str) -> Optional[int]:
    text = raw.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[lowered]
    if lowered == "last":
        return -1
    if _INTEGER.fullmatch(text):
        return int(text)
    if match := _STEP_PREFIX.fullmatch(text):
        return int(match.group(1))
    if match := _HASH_STEP.fullmatch(text):
        return int(match.group(1))
    if match := _STEP_PHRASE.fullmatch(text):
        return _token_to_index(match.group("tok"))
    if match := _STEP_LEADING.match(lowered):
        if match.group("num"):
            return int(match.group("num"))
        ord_tok = match.group("ord")
        if ord_tok:
            return _ORDINAL_WORDS.get(ord_tok.lower())
    return None


_ALL_SLOT = re.compile(
    r"\b(?:both(?:\s+steps)?|all(?:\s+(?:three|four|five|six|seven|eight))?\s*(?:steps)?|"
    r"all\s+of\s+them|every\s+step|each\s+step)\b",
    re.I,
)
_STEP_RANGE = re.compile(
    r"\bsteps?\s+(\d+)\s*(?:through|to|[-–—])\s*(\d+)\b",
    re.I,
)
_COUNT_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
}
_STEP_RANGE_INLINE = re.compile(
    r"(?:steps?|items?)?\s*#?(\d+)\s*(?:to|through|[-–—])\s*#?(\d+)",
    re.I,
)
_STEP_PAIR_INLINE = re.compile(
    r"(?:steps?|items?)\s+#?(\d+)\s+and\s+#?(\d+)",
    re.I,
)
_STEP_REF_INLINE = re.compile(
    r"\b(?:step|item)\s+#?(\d+)\b",
    re.I,
)


def extract_step_numbers(segment: str, count: int) -> list[int]:
    """Collect 1-based step numbers named in *segment* within ``1..count``."""
    if not segment or count <= 0:
        return []
    found: list[int] = []

    def add(value: int) -> None:
        if 1 <= value <= count and value not in found:
            found.append(value)

    for lo_s, hi_s in _STEP_RANGE_INLINE.findall(segment):
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        for index in range(lo, hi + 1):
            add(index)
    for first, second in _STEP_PAIR_INLINE.findall(segment):
        add(int(first))
        add(int(second))
    for match in re.finditer(
        r"(?:steps?|items?)\s+#?(\d+)(?:\s*(?:,|and)\s*#?(\d+))+",
        segment,
        re.I,
    ):
        add(int(match.group(1)))
        for group in match.groups()[1:]:
            if group:
                add(int(group))
    for ref in _STEP_REF_INLINE.findall(segment):
        add(int(ref))
    return found


def resolve_slot_item(proposal, raw: str, *, field_name: str = "item") -> str:
    """Resolve a slot answer to ``all``, comma-separated indices, or one index string."""
    count = len(proposal.items) if proposal is not None else 0
    text = (raw or "").strip()
    if not text:
        raise fastworkflow.Refused(
            f"Which of the {count} steps do you mean?" if count else "Which step do you mean?",
            field=field_name,
        )
    lowered = text.lower()
    if _ALL_SLOT.search(text) or lowered in ("both", "both steps"):
        if lowered in ("both", "both steps") and count != 2:
            raise fastworkflow.Refused(f"Which of the {count} steps do you mean?", field=field_name)
        return "all"
    if match := re.match(r"\ball\s+(three|four|five|six|seven|eight)\b", lowered):
        if _COUNT_WORDS.get(match.group(1)) == count:
            return "all"
    if match := _STEP_RANGE.search(text):
        lo, hi = int(match.group(1)), int(match.group(2))
        if 1 <= lo <= hi <= count:
            return ",".join(str(index) for index in range(lo, hi + 1))
    nums = [int(num) for num in re.findall(r"\b(\d+)\b", text) if 1 <= int(num) <= count]
    if len(nums) >= 2:
        ordered = sorted(set(nums))
        if set(ordered) == set(range(1, count + 1)):
            return "all"
        return ",".join(str(num) for num in ordered)
    if re.search(r"\b(?:all|every|each|both)\b", text, re.I) and count > 0:
        return "all"
    prose = extract_step_numbers(text, count)
    if prose:
        return ",".join(str(num) for num in prose) if len(prose) > 1 else str(prose[0])
    return resolve_proposal_step(proposal, text, field_name=field_name, slot_retry=True)


def resolve_proposal_step(
    proposal,
    raw: str,
    *,
    field_name: str = "item",
    slot_retry: bool = False,
) -> str:
    """Resolve *raw* to a 1-based step number string for the proposal's items."""
    count = len(proposal.items) if proposal is not None else 0
    text = (raw or "").strip()
    if not text:
        raise fastworkflow.Refused(
            f"Which of the {count} steps do you mean?" if count else "Which step do you mean?",
            field=field_name,
        )

    explicit = _parse_explicit_index(text)
    if explicit is not None:
        if explicit == -1:
            if count < 1:
                raise fastworkflow.Refused(f"Which of the {count} steps do you mean?", field=field_name)
            return str(count)
        if count > 0 and (explicit < 1 or explicit > count):
            message = (
                f"There are only {count} step(s)."
                if not slot_retry
                else f"Which of the {count} steps do you mean?"
            )
            raise fastworkflow.Refused(message, field=field_name)
        if explicit >= 1:
            return str(explicit)

    goals = _proposal_item_goals(proposal)
    matched = _unique_goal_index(goals, text)
    if matched is not None:
        return str(matched)

    if count > 0:
        raise fastworkflow.Refused(f"Which of the {count} steps do you mean?", field=field_name)
    raise fastworkflow.Refused("Which step do you mean?", field=field_name)


def explicit_out_of_range_step(text: str, count: int) -> bool:
    """True when *text* names an explicit step number outside 1..count."""
    if count <= 0:
        return False
    explicit = _parse_explicit_index((text or "").strip())
    if explicit is None or explicit == -1:
        return False
    return explicit < 1 or explicit > count
