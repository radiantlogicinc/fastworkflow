"""Argument checks of an execution command's LLM parameter extraction (fix-qjf4).

A planning command's typed call has its arguments checked against the request
(``planning_commands.check_arguments``). An execution command's parameters, when the LLM
extractor fills them, get the same kinds of checks here, built on the same helpers:

- **grounding**: a ``str`` value must be written in the request, verbatim up to case and
  Unicode dashes (any dash matches ``-``), after surrounding quotes, angle brackets and a
  trailing ``.;,`` are removed. The value then takes the request's own spelling.
- **placeholders**: punctuation alone (``...``), a template token or an invented example value (``planning_commands.
  _placeholder_like``: ``{name}``, ``NAME_1``, "unspecified", "John Doe" the request does not
  write), a bracketed placeholder (``<item_uid>``), and one of the field's own description
  ``examples`` the request does not write.
- **allowed values**: a ``str`` field whose allowed values are declared outside its type, in
  ``json_schema_extra={"enum": [...]}`` (the JSON Schema keyword) or in its description as
  ``One of: a, b, c``, must hold one of them (case-insensitive; the declared spelling is taken).
  A ``Literal`` or ``Enum`` field is already checked by its type in ``validate_parameters``. The rule
  (``allowed_member``) is shared: ``planning_commands.check_arguments`` applies it to a planning command's
  ``One of:`` list (fix-48np.3).
- **numbers**: an ``int`` value must be a number the request writes, when the request writes any
  number outside the text of its ``str`` values. Unlike the planning rule, a value the request does
  not write is refused, never replaced by the one number the request writes.

**What "the request" is.** The command text the extractor read (``InputForParamExtraction.command``:
the user's message in assistant mode, the command the agent wrote in agent mode). It is the
extractor's only input, so a value it does not write came from somewhere else: the field's
description, its few-shot examples, or the model. The conversation's earlier user messages are not
counted: the extractor never sees them, and the offline replay (the app) found that adding them
changes no verdict on a ``str`` value.

**What is not checked.**

- A value that is not given: ``None``, empty, ``NOT_FOUND``, the ``<UNKNOWN>`` marker, the
  extractor's "not given" number, the field's not-given value or its declared default (the
  extractor is told to answer the default for a value the request does not give, so a default is
  indistinguishable from "not given"). An absent ``Optional[int] = None`` is never refused.
- Types other than ``str`` and ``int`` (``float``, ``bool``, dates, lists, models): their values are
  converted on purpose, or (lists) are not checked by design, as on the planning side.
- A field exempted from grounding (its value is legitimately converted from the request: a date
  written "next Monday", an amount, a code the request spells another way). The marker is the
  ``converted_from_command_text`` option: ``json_schema_extra={"converted_from_command_text": True}``. A field with a
  declared ``pattern`` is exempt too: it declares a format the value is converted into (``#W123`` of
  "order W123"). An exempt field still gets the placeholder and allowed-values checks, but not the
  description-example check: a converted value can equal an example legitimately (tau-bench airline:
  "New York" on "May 20" gives ``JFK`` and ``2024-05-20``, the search fields' own examples). On an ``int``
  field the same marker turns the number check off; it needs a plain default (``int = 0``), since a
  check option cannot sit on an ``Optional[int]`` (``planning_commands.option_params`` refuses it).
- A planning command's ``Input``: ``check_arguments`` checks it already.

A refused value is treated as a value that was not given: the extractor is asked once more with the
refusal appended to the command (``REFUSED_EXTRACTION_NOTE``, as ``typed_planning_call`` does with
``REFUSED_CALL_NOTE``), the new values are checked against the original command text, and a value
still refused takes the field's not-given value (a required field is then asked for; an optional one
takes its default). ``FW_EXTRACTION_ARGUMENT_CHECKS=off`` turns the checks off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, get_origin

from pydantic import BaseModel

import fastworkflow
from fastworkflow import planning_commands
from fastworkflow.field_options import CONVERTED_FROM_COMMAND_TEXT
from fastworkflow.utils.input_validation import set_field_unvalidated
from fastworkflow.utils.signatures import (
    NOT_FOUND,
    _is_filled,
    _numeric_kind,
    classify_bracketed_value,
    is_optional_str_field,
    is_sentinel_number,
    is_unknown_marker,
    not_given_value,
    unwrap_quoted_value,
)

#: The switch. On unless set to off, 0 or false (case-insensitive), like the other extraction guards.
ARGUMENT_CHECKS_ENV_VAR = "FW_EXTRACTION_ARGUMENT_CHECKS"
_OFF = ("off", "0", "false")

#: Appended to the command text when the extractor is asked once more after a refusal.
REFUSED_EXTRACTION_NOTE = (
    "(A first extraction from this command was refused: {refusal}. Extract the values again from the "
    "command above, correcting that: copy each value exactly as the command writes it, and answer "
    f"{NOT_FOUND} for a value the command does not give.)")

#: Appended to the request when a call's first phase found nothing for its lookup, and the arguments are extracted
#: once more (``planning_commands.reextracted_arguments``): the phase's observation, in the refusal note's style.
PHASE_FAILED_EXTRACTION_NOTE = (
    "(A first run of this command found nothing for its lookup: {observation}. Extract the values again from the "
    "command above, correcting that: copy each value exactly as the command writes it, and answer "
    f"{NOT_FOUND} for a value the command does not give.)")

NOT_WRITTEN_WHY = "is not written in the request"
EXAMPLE_WHY = "is an example from the field's description, not a value the request writes"
BRACKET_PLACEHOLDER_WHY = "is a placeholder, not a value"
NO_VALUE_WHY = "is punctuation, not a value"

#: ``One of: a, b, c`` (or ``a, b or c``) in a field description: the allowed values, identifier-shaped.
_ONE_OF = re.compile(r"\bone of\s*:\s*(?P<items>[^.;\n]+)", re.IGNORECASE)
_ONE_OF_SEPARATOR = re.compile(r"\s*(?:,|\bor\b)\s*")
_ONE_OF_ITEM = re.compile(r"[A-Za-z0-9][\w-]*")
#: A whole number the request writes: digits (with thousands separators), not part of a word or code.
_DIGITS = re.compile(r"(?<![\w.])(-?)(\d{1,3}(?:,\d{3})+|\d+)(?![\w])")
_NUMBER_WORD = re.compile(r"\b(" + "|".join(planning_commands.NUMBER_WORDS) + r")\b", re.IGNORECASE)


def checks_enabled() -> bool:
    """False when FW_EXTRACTION_ARGUMENT_CHECKS is off, 0 or false."""
    value = str(fastworkflow.get_env_var(ARGUMENT_CHECKS_ENV_VAR, default="on")).strip().lower()
    return value not in _OFF


@dataclass
class ExtractionCheck:
    """What ``check_extracted_values`` found: ``refused`` maps a field to ``(value, why)``,
    ``respelled`` maps a field to ``(value, the request's or the declared spelling)``."""

    refused: Dict[str, Tuple[Any, str]] = dataclass_field(default_factory=dict)
    respelled: Dict[str, Tuple[Any, Any]] = dataclass_field(default_factory=dict)

    def refusal_texts(self) -> List[str]:
        """``name='value' why`` for each refused field, in field order."""
        return [f"{name}={value!r} {why}" for name, (value, why) in self.refused.items()]

    def note(self) -> str:
        """The refusal, as appended to the command for the second extraction."""
        return REFUSED_EXTRACTION_NOTE.format(refusal="; ".join(self.refusal_texts()))

    def apply(self, params: BaseModel) -> None:
        """Write the respellings onto *params*, and give each refused field its not-given value."""
        fields = type(params).model_fields
        for name, (_, spelled) in self.respelled.items():
            if name not in self.refused:
                set_field_unvalidated(params, name, spelled)
        for name in self.refused:
            set_field_unvalidated(params, name, not_given_value(fields[name]))


def _extra(field_info: Any) -> dict:
    extra = getattr(field_info, "json_schema_extra", None)
    return extra if isinstance(extra, dict) else {}


def allowed_values(field_info: Any) -> Tuple[str, ...]:
    """The values a ``str`` field allows when its type does not say: ``json_schema_extra["enum"]`` (a
    list of text), else a ``One of: a, b, c`` list in its description (two or more identifier-shaped
    items; anything else there is prose, not a list). Empty when the field declares neither."""
    declared = _extra(field_info).get("enum")
    if isinstance(declared, (list, tuple)) and declared and all(isinstance(v, str) for v in declared):
        return tuple(declared)
    match = _ONE_OF.search(getattr(field_info, "description", None) or "")
    if not match:
        return ()
    items = [item.strip().strip("'\"`") for item in _ONE_OF_SEPARATOR.split(match["items"].strip())]
    items = [item for item in items if item]
    if len(items) < 2 or not all(_ONE_OF_ITEM.fullmatch(item) for item in items):
        return ()
    return tuple(items)


def allowed_member(allowed: Tuple[str, ...], text: str) -> Optional[str]:
    """The allowed-values rule, shared by both checkers (fix-48np.3): the member of *allowed* that *text* is,
    ignoring case, in its declared spelling; None when *text* is none of them (the value is refused). The
    execution checker reads *allowed* from ``allowed_values``; the planning checker
    (``planning_commands.check_arguments``) from ``allowed_values`` too, where its loader admits only the
    ``One of:`` description list (a planning command's choice is otherwise a ``Literal``)."""
    folded = text.lower()
    return next((v for v in allowed if v.lower() == folded), None)


def not_allowed_why(allowed: Tuple[str, ...]) -> str:
    """The refusal of a value outside *allowed*, on either path."""
    return f"is not one of {', '.join(allowed)}"


def is_converted(field_info: Any) -> bool:
    """Whether a field carries ``json_schema_extra={"converted_from_command_text": True}``: its value is converted
    from the command text, so the text is not checked against it (spec 2.0a, the converted option)."""
    return _extra(field_info).get(CONVERTED_FROM_COMMAND_TEXT) is True


def is_grounding_exempt(field_info: Any) -> bool:
    """Whether a ``str`` field's value need not be written in the request: it carries
    ``json_schema_extra={"converted_from_command_text": True}``, or declares a ``pattern`` (a format the value is
    converted into)."""
    if is_converted(field_info):
        return True
    if "pattern" in _extra(field_info):
        return True
    return any(getattr(meta, "pattern", None) for meta in getattr(field_info, "metadata", None) or ())


def _is_str_field(field_info: Any) -> bool:
    return field_info.annotation is str or is_optional_str_field(field_info)


def _is_int_field(field_info: Any) -> bool:
    annotation = field_info.annotation
    if annotation is bool or (isinstance(annotation, type) and issubclass(annotation, Enum)):
        return False
    if get_origin(annotation) is Literal:
        return False
    return _numeric_kind(annotation) is int


def _given(field_info: Any, value: Any) -> bool:
    """Whether *value* is a value the extractor gave, not its way of saying "not given"."""
    if isinstance(value, str) and (is_unknown_marker(value) or value.strip() in ("", "None", NOT_FOUND)):
        return False
    return _is_filled(field_info, value)


def written_numbers(request: str, texts: List[str]) -> set:
    """The whole numbers *request* writes (digits, with thousands separators and a minus sign, or number
    words), outside the text of *texts*. Shared by the execution check and the planning check
    (``planning_commands._request_numbers``)."""
    lowered = request.lower()
    taken: List[Tuple[int, int]] = []
    for text in texts:
        text = text.lower()
        at = lowered.find(text) if text else -1
        while at >= 0:
            taken.append((at, at + len(text)))
            at = lowered.find(text, at + 1)
    found = set()
    for match in _DIGITS.finditer(request):
        if any(a <= match.start(2) < b for a, b in taken):
            continue
        parts = [match[2].replace(",", "")] + (match[2].split(",") if "," in match[2] else [])
        for part in parts:
            number = int(part)
            found.add(number)
            if match[1]:
                found.add(-number)
    for match in _NUMBER_WORD.finditer(request):
        if not any(a <= match.start() < b for a, b in taken):
            found.add(planning_commands.NUMBER_WORDS[match[1].lower()])
    return found


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _str_check(name: str, field_info: Any, value: Any, request: str, folded: str,
               check: ExtractionCheck) -> Optional[str]:
    """Check one given ``str`` value; returns the text it stands for when it passes, else None."""
    single = planning_commands._single(value)
    if not isinstance(single, str) or isinstance(single, Enum):
        return None
    _, text = unwrap_quoted_value(single, request)
    kind, inner = classify_bracketed_value(text, name)
    if kind == "placeholder":
        check.refused[name] = (value, BRACKET_PLACEHOLDER_WHY)
        return None
    text = planning_commands._clean_str(inner)
    if not text.strip():
        # only quotes or punctuation (``...``): nothing of the request, and no value
        check.refused[name] = (value, NO_VALUE_WHY)
        return None
    if text.strip() in ("None", NOT_FOUND):
        return None

    allowed = allowed_values(field_info)
    if allowed:
        member = allowed_member(allowed, text)
        if member is None:
            check.refused[name] = (value, not_allowed_why(allowed))
            return None
        if member != value:
            check.respelled[name] = (value, member)
        return member

    if planning_commands._placeholder_like(text, folded):
        check.refused[name] = (value, planning_commands.PLACEHOLDER_WHY)
        return None
    if is_grounding_exempt(field_info):
        # A converted value can equal a description example legitimately: "New York" on "May 20"
        # gives origin JFK and date 2024-05-20, the airline search fields' own examples. So an
        # exempt field gets no example check.
        return text
    exact = planning_commands._written(text, folded, request)
    examples = [str(e).lower() for e in (getattr(field_info, "examples", None) or ())]
    if exact is None and text.lower() in examples:
        check.refused[name] = (value, EXAMPLE_WHY)
        return None
    if exact is None:
        check.refused[name] = (value, NOT_WRITTEN_WHY)
        return None
    if exact != value:
        check.respelled[name] = (value, exact)
    return exact


def check_extracted_values(params: BaseModel, request: Optional[str]) -> ExtractionCheck:
    """Check the values the LLM extractor gave in *params* against *request* (see the module doc).

    Reads *params*, changes nothing: ``ExtractionCheck.apply`` writes the outcome. With no request
    text there is nothing to check against, and nothing is refused.
    """
    check = ExtractionCheck()
    if not request or not request.strip():
        return check
    folded = planning_commands._fold(request).lower()
    fields = type(params).model_fields
    texts: List[str] = []
    numbers: List[Tuple[str, Any, int]] = []
    for name, field_info in fields.items():
        value = getattr(params, name, None)
        if not _given(field_info, value):
            continue
        if _is_str_field(field_info):
            text = _str_check(name, field_info, value, request, folded, check)
            if text:
                texts.append(text)
        elif _is_int_field(field_info) and not is_converted(field_info):
            number = _as_int(value)
            if number is not None and not is_sentinel_number(number):
                numbers.append((name, value, number))
    if numbers:
        written = written_numbers(request, texts)
        for name, value, number in numbers:
            if written and number not in written:
                check.refused[name] = (
                    value, f"is not a number the request writes ({', '.join(map(str, sorted(written)))})")
    return check
