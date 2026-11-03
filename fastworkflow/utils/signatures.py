import sys
import ast
import inspect
import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
import os
from contextlib import suppress
from dataclasses import dataclass, field as dataclass_field
from typing import Literal, Optional, Tuple, Union, Dict, Any, Type, List, get_args, get_origin
from enum import Enum
from datetime import date
import re
from difflib import get_close_matches
import json

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.utils.logging import logger
from fastworkflow.model_pipeline_training import get_route_layer_filepath_model
from fastworkflow.utils.fuzzy_match import find_best_matches
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.dspy_logger import DSPyForward
from fastworkflow.command_directory import CommandDirectory
from fastworkflow import planning_commands
from fastworkflow.utils.workflow_vocabulary import (
    KIND_WORD_WHY, WorkflowVocabulary, is_kind_word, is_pronoun_phrase, load_workflow_vocabulary)

MISSING_INFORMATION_ERRMSG = None
INVALID_INFORMATION_ERRMSG = None
PARAMETER_EXTRACTION_ERROR_MSG = None
NOT_FOUND = None

LLM_PARAM_EXTRACTION = None
LITELLM_API_KEY_PARAM_EXTRACTION = None

INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max


# --- Angle-bracketed parameter values ------------------------------------------
#
# A command's usage text writes arguments as ``<field_name>``, and an agent copies
# that shape: sometimes around a real value it read from an earlier output
# (``<9a1b2c3d...>``), sometimes with no value at all (``<item_uid>``). The first
# is the right value in the wrong wrapper and is unwrapped; the second is a
# placeholder the agent never filled in, and running the command with it would
# only produce a not-found or an empty result that reads like an answer.

# The whole value is one bracketed token, possibly doubled (``<<x>>``). A value
# with brackets inside it, or text outside them, is left alone.
_BRACKETED_VALUE = re.compile(r"^\s*<+\s*([^<>]*?)\s*>+\s*$")

# Lowercase words joined by ``_`` or ``-``, no digits: the shape of a variable
# name, which is what a usage-text placeholder is.
_IDENTIFIER_SHAPE = re.compile(r"[a-z]+(?:[_\-][a-z]+)*")

# Words a placeholder is made of when it does not repeat the field's own name.
_PLACEHOLDER_WORDS = frozenset({
    "uid", "uuid", "id", "ids", "key", "code", "value", "values", "name",
    "placeholder", "param", "parameter", "arg", "argument",
})


def _is_placeholder_text(inner: str, field_name: str) -> bool:
    """Whether *inner* (the text between the brackets) names a value instead of
    being one.

    Placeholder: empty, punctuation only, or variable-name shaped (lowercase
    words, no digits, all one case) and either snake_case or made of the field's
    own words / generic placeholder words: ``repo_uid``, ``uid``,
    ``first_order_uid``, ``&_catalog_uid_``, ``filter`` for a ``filter``
    field. Anything carrying a digit, a space, or mixed case is a value:
    ``9a1b2c3d...``, ``Ann Lee``, ``PayHub``.
    """
    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", inner)
    if not core:
        return True
    if not (core.islower() or core.isupper()):
        return False
    lowered = core.lower()
    if not _IDENTIFIER_SHAPE.fullmatch(lowered):
        return False
    if "_" in lowered:
        return True
    words = set(re.split(r"[_\-]", lowered))
    field_words = set(field_name.lower().split("_"))
    return bool(words & (field_words | _PLACEHOLDER_WORDS))


def classify_bracketed_value(value: Any, field_name: str) -> Tuple[str, Any]:
    """Classify one parameter value for angle brackets around it.

    Returns ``("unchanged", value)``, ``("unwrapped", inner_value)`` for a real
    value inside brackets, or ``("placeholder", value)`` for bracketed text that
    holds no value. Only strings are inspected; a list is checked item by item
    and is a placeholder if any item is.
    """
    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            kind, new_item = classify_bracketed_value(item, field_name)
            if kind == "placeholder":
                return "placeholder", value
            changed = changed or kind == "unwrapped"
            items.append(new_item)
        return ("unwrapped", items) if changed else ("unchanged", value)
    if not isinstance(value, str) or isinstance(value, Enum):
        return "unchanged", value
    match = _BRACKETED_VALUE.match(value)
    if not match:
        return "unchanged", value
    inner = match[1]
    if _is_placeholder_text(inner, field_name):
        return "placeholder", value
    return "unwrapped", inner


def placeholder_value_message(field_name: str, field_value: Any) -> str:
    """What the caller is told when *field_name* was given a placeholder."""
    return (
        f"{field_name} {field_value!r} is a placeholder, not a value: pass the real "
        f"{field_name} copied from an earlier command's output (run the command "
        f"that lists it first if no output has shown it yet)."
    )


# --- Quoted parameter values -----------------------------------------------------
#
# A request often sets a value off in quotes (filter by "Ann Lee"), and an
# extractor sometimes keeps them: the command then searches for the nine
# characters ``"Ann Lee"`` and finds nothing, which reads like an answer. A value
# that is wholly one matching quote pair is unwrapped to the text inside it. It
# is kept when the request itself writes the quotes as part of a quoted value
# (the quoted text appears in the request right next to another quote mark,
# as in ``'"Ann Lee"'``), or when the inner text holds the same quote mark,
# where the outer pair may not be a pair at all (``"a" or "b"``).

_QUOTE_PAIRS = (
    ('"', '"'), ("'", "'"), ("`", "`"),
    ("“", "”"), ("‘", "’"), ("„", "“"),
    ("«", "»"),
)
_QUOTE_CHARS = frozenset(ch for pair in _QUOTE_PAIRS for ch in pair)


def _request_writes_quotes(quoted: str, request_text: Optional[str]) -> bool:
    """Whether *request_text* writes *quoted* (quotes included) inside quotes."""
    if not request_text:
        return False
    start = request_text.find(quoted)
    while start != -1:
        end = start + len(quoted)
        before = request_text[start - 1] if start > 0 else ""
        after = request_text[end] if end < len(request_text) else ""
        if before in _QUOTE_CHARS or after in _QUOTE_CHARS:
            return True
        start = request_text.find(quoted, start + 1)
    return False


def unwrap_quoted_value(value: Any, request_text: Optional[str] = None) -> Tuple[bool, Any]:
    """Strip quote pairs wrapped around a whole str value.

    Returns ``(changed, value)``. Straight, typographic, guillemet and backtick
    pairs are recognised, nested pairs are removed one layer at a time, and
    surrounding whitespace goes with them. A list is unwrapped item by item.
    Non-str values come back unchanged. *request_text*, when given, is the text
    the value was taken from; a layer the request itself writes inside quotes is
    kept (see the comment above).
    """
    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            item_changed, new_item = unwrap_quoted_value(item, request_text)
            changed = changed or item_changed
            items.append(new_item)
        return (True, items) if changed else (False, value)
    if not isinstance(value, str) or isinstance(value, Enum):
        return False, value
    current = value.strip()
    unwrapped = False
    while len(current) >= 2:
        pair = next(
            ((left, right) for left, right in _QUOTE_PAIRS
             if current[0] == left and current[-1] == right),
            None,
        )
        if pair is None:
            break
        inner = current[1:-1]
        if not inner.strip() or pair[0] in inner or pair[1] in inner:
            break
        if _request_writes_quotes(current, request_text):
            break
        current = inner.strip()
        unwrapped = True
    return (True, current) if unwrapped else (False, value)


# --- Placeholder-shaped values the request never writes -----------------------------
#
# Without brackets a placeholder still has a recognisable shape: a qualifier
# word plus the field's own name (``desired-tag``, ``your_uid``, ``new-name``),
# or nothing but the field's name / generic placeholder words (``uid`` for an
# ``item_uid`` field). Such a value is a placeholder only if the request does
# not write it: a request that says "add the tag new-name" means it.

_PLACEHOLDER_QUALIFIERS = frozenset({
    "desired", "your", "my", "new", "some", "example", "sample", "placeholder",
    "target", "actual", "real", "specific", "chosen", "selected", "appropriate",
    "relevant", "correct", "given", "the", "a", "an", "any",
})


def _is_declared_choice(annotation: Any, value: Any) -> bool:
    """Whether str *value* is one of the choices *annotation* declares: a member of a ``Literal``,
    or the value of an ``Enum``, alone or inside an ``Optional``. A declared choice is a value
    whatever word it is: ``code`` for ``Literal["code", "name"]`` is never a placeholder."""
    if not isinstance(value, str):
        return False
    options = get_args(annotation) if get_origin(annotation) is Union else (annotation,)
    for option in options:
        if get_origin(option) is Literal and value in get_args(option):
            return True
        if isinstance(option, type) and issubclass(option, Enum) and value in {m.value for m in option}:
            return True
    return False


def is_ungrounded_placeholder(value: Any, field_name: str,
                              request_text: Optional[str]) -> bool:
    """Whether str *value* is placeholder-shaped and *request_text* never writes it.

    Placeholder-shaped: lowercase words joined by ``-`` or ``_`` (no digits, no
    spaces), and either a qualifier word followed by words that include one of
    the field's own words or a generic placeholder word (``desired-tag`` for
    ``tag``, ``your_item_uid`` for ``item_uid``), or made only of such words
    (``uid`` for ``item_uid``). The request is searched case-insensitively for
    the value as written. With no request text there is nothing to ground
    against and the value is left alone. A list is a placeholder if any item is.
    """
    if isinstance(value, list):
        return any(is_ungrounded_placeholder(v, field_name, request_text) for v in value)
    if not isinstance(value, str) or isinstance(value, Enum) or not request_text:
        return False
    text = value.strip()
    if not text or not _IDENTIFIER_SHAPE.fullmatch(text):
        return False
    if text.lower() in request_text.lower():
        return False
    words = re.split(r"[_\-]", text)
    known = set(field_name.lower().split("_")) | _PLACEHOLDER_WORDS
    if len(words) >= 2 and words[0] in _PLACEHOLDER_QUALIFIERS:
        return bool(set(words[1:]) & known)
    return set(words) <= known


# --- A command's name written as a value -----------------------------------------------
#
# An extractor sometimes copies a command's name into a parameter: a planning command's own name as
# its ``application``, or another command's name as the ``query`` of a lookup, because it
# has just been reading that name. No real value is a command name, so such a value is
# treated as not given. Only the identifier form counts (``orphan_login_review``,
# ``Directory/find_group``, in any case); a name written with spaces is a value, so an
# entity called "Tag Record" is never refused. A declared Literal or enum choice is never
# refused, whatever word it is.

def command_name_index(app_workflow, subject_command_name: str) -> Dict[str, str]:
    """Identifier form (lowercase) of each command name of *app_workflow* -> the name as written.

    Names come from the workflow's command directory: each qualified key (``Directory/find_group``)
    and its bare name (``find_group``). Empty when the workflow has no folder path.
    """
    names = set()
    folderpath = getattr(app_workflow, "folderpath", None)
    if folderpath:
        directory = fastworkflow.RoutingRegistry.get_definition(folderpath).command_directory
        for key in directory.map_command_2_metadata:
            names.add(key)
            if "/" in key:
                names.add(key.split("/", 1)[1])
    index: Dict[str, str] = {}
    for name in sorted(names):
        index.setdefault(name.lower(), name)
    return index


def _is_declared_choice_of(annotation: Any, value: Any) -> bool:
    """``_is_declared_choice`` for a field, and for the items of a list field."""
    if _is_declared_choice(annotation, value):
        return True
    options = get_args(annotation) if get_origin(annotation) is Union else (annotation,)
    return any(
        get_origin(option) in (list, List) and get_args(option)
        and _is_declared_choice(get_args(option)[0], value)
        for option in options
    )


def command_name_refusals(cmd_parameters: BaseModel, subject_command_name: str,
                          index: Dict[str, str], skip: Any) -> Dict[str, Tuple[Any, str]]:
    """Fields whose str value (or one str item of a list) is a command name.

    Returns ``{field: (value, wording)}``. The name is the called command's own (its
    qualified or bare name) or any other command's name in *index*. Fields in *skip* were
    already refused by ``screen_parameter_values`` and are left alone.
    """
    own_forms = {name.strip().lower() for name in
                 (subject_command_name or "", (subject_command_name or "").split("/")[-1]) if name}
    refused: Dict[str, Tuple[Any, str]] = {}
    for field_name, field_info in type(cmd_parameters).model_fields.items():
        if field_name in skip:
            continue
        value = getattr(cmd_parameters, field_name, None)
        for item in (value if isinstance(value, list) else [value]):
            if not isinstance(item, str) or isinstance(item, Enum):
                continue
            if _is_declared_choice_of(field_info.annotation, item):
                continue
            form = item.strip().lower()
            if form in own_forms:
                wording = (f"{field_name} {item!r} is the name of this command, not a value: "
                           f"pass the real {field_name} the request names.")
            elif form in index:
                wording = (f"{field_name} {item!r} is the name of the command {index[form]!r}, "
                           f"not a value: pass the real {field_name} the request names.")
            else:
                continue
            refused[field_name] = (item, wording)
            break
    return refused


# --- A free-text value that is only a word of the workflow ---------------------------
#
# A planner that fills a free-text (``str``) input can write a word that names the kind of
# record or a field name rather than a record (``query='right_name'``, ``query='department'``),
# or a pronoun (``query='Her'``). The workflow's own vocabulary
# (``fastworkflow.utils.workflow_vocabulary``) and the closed pronoun list decide. Only a value
# that is exactly such a word is refused; a value that merely contains one passes. A declared
# choice (a Literal or an Enum field) is never screened, whatever word it is. A kind-selector
# field (its description or labeled examples name the kinds it takes, ``kind_options``) is exempt
# from the kind-word refusal for the values it lists; the pronoun refusal still applies to it.
# The text ``None`` is the missing-value marker and is never a kind word (``default_unfilled_optional_strs``).

PRONOUN_WHY = "is a pronoun, not a value"


def _workflow_vocabulary(app_workflow) -> Optional[WorkflowVocabulary]:
    """The workflow's vocabulary (the stored file when fresh, else derived in memory); None when
    the workflow has no folder path or the derivation fails. The pronoun screen still runs then."""
    folderpath = getattr(app_workflow, "folderpath", None)
    if not folderpath:
        return None
    try:
        return load_workflow_vocabulary(folderpath)
    except Exception:  # noqa: BLE001 - the workflow's own code: the screens are skipped, not the call
        logger.warning(f"could not derive the vocabulary of {folderpath}", exc_info=True)
        return None


def word_screen_refusals(cmd_parameters: BaseModel, vocabulary: Optional[WorkflowVocabulary],
                         skip: Any, command_name: Optional[str]) -> Dict[str, Tuple[Any, str]]:
    """Free-text ``str`` fields whose value is only a pronoun or only a kind word or field name.

    Returns ``{field: (value, wording)}``. A ``str`` or ``Optional[str]`` field only; the
    fields in *skip* were already refused by an earlier screen and are left alone. A kind-selector
    field (``WorkflowVocabulary.kind_options``) is exempt from the kind-word refusal only: a value
    it lists is a kind it takes. The pronoun refusal still applies.
    """
    refused: Dict[str, Tuple[Any, str]] = {}
    for field_name, field_info in type(cmd_parameters).model_fields.items():
        if field_name in skip:
            continue
        if field_info.annotation is not str and not is_optional_str_field(field_info):
            continue
        value = getattr(cmd_parameters, field_name, None)
        if not isinstance(value, str) or isinstance(value, Enum):
            continue
        text = value.strip()
        if not text:
            continue
        options = vocabulary.kind_options(command_name, field_name) if vocabulary is not None else frozenset()
        if is_pronoun_phrase(text):
            why = PRONOUN_WHY
        elif is_kind_word(text, field_name, vocabulary, options):
            why = KIND_WORD_WHY
        else:
            continue
        refused[field_name] = (value, f"{field_name} {value!r} {why}: pass the real {field_name} the request names.")
    return refused


# --- Numbers out of any sane range ---------------------------------------------------
#
# The extractor's own "not given" marker for an int is ``INVALID_INT_VALUE``
# (``-sys.maxsize``), and the signature tells the model to return it for an
# optional int the request does not mention. For a field that has a default (or
# is Optional) that marker used to reach the command as the value; it is now
# replaced by the field's default. A value of the same kind the model makes up
# (int32/int64 limits, anything beyond 2**53, a non-finite float) is treated the
# same way: absent when the field can be absent, a parameter error otherwise.
#
# Declared bounds (``Field(ge=..., gt=..., le=..., lt=...)``) are enforced, since
# the parameters are built with ``model_construct`` and pydantic never checks
# them. A field with no declared lower bound whose name says it is a position,
# a count or a size (``start``, ``offset``, ``limit``, ``page_size``...) may not
# be negative; a field that does take negative values declares ``ge=``.

_SENTINEL_INT_MAGNITUDES = frozenset({2**31 - 1, 2**31, 2**32 - 1, 2**63 - 1, 2**63, 2**64 - 1})
_SAFE_INT_MAGNITUDE = 2**53
_NON_NEGATIVE_NAME_WORDS = frozenset({
    "start", "skip", "offset", "limit", "count", "page", "size",
})
_INT_TEXT = re.compile(r"^\s*[+-]?\d+\s*$")


def _numeric_kind(annotation: Any) -> Optional[Type[Any]]:
    """``int`` or ``float`` when the field is a plain (optional) number, else None."""
    candidates = [annotation]
    if get_origin(annotation) is Union:
        candidates = [t for t in get_args(annotation) if t is not type(None)]
    if int in candidates and bool not in candidates:
        return int
    if float in candidates:
        return float
    return None


def _numeric_view(value: Any, kind: Type[Any]) -> Optional[Union[int, float]]:
    """*value* as a number of *kind*, or None if it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if kind is int and _INT_TEXT.match(text):
            return int(text)
        if kind is float:
            with suppress(ValueError):
                return float(text)
    return None


def is_sentinel_number(value: Union[int, float]) -> bool:
    """Whether *value* is a marker or a made-up extreme rather than a real number."""
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf")) or value == INVALID_FLOAT_VALUE
    return abs(value) in _SENTINEL_INT_MAGNITUDES or abs(value) > _SAFE_INT_MAGNITUDE


def _declared_bounds(field_info: Any) -> Dict[str, Any]:
    bounds: Dict[str, Any] = {}
    for meta in getattr(field_info, "metadata", []) or []:
        for name in ("ge", "gt", "le", "lt"):
            bound = getattr(meta, name, None)
            if bound is not None:
                bounds[name] = bound
    return bounds


def numeric_range_problem(field_name: str, field_info: Any,
                          value: Union[int, float]) -> Optional[str]:
    """Why *value* is out of range for the field, or None if it is in range."""
    bounds = _declared_bounds(field_info)
    if "ge" in bounds and value < bounds["ge"]:
        return f"must be at least {bounds['ge']}"
    if "gt" in bounds and value <= bounds["gt"]:
        return f"must be greater than {bounds['gt']}"
    if "le" in bounds and value > bounds["le"]:
        return f"must be at most {bounds['le']}"
    if "lt" in bounds and value >= bounds["lt"]:
        return f"must be less than {bounds['lt']}"
    if (
        value < 0
        and "ge" not in bounds and "gt" not in bounds
        and set(field_name.lower().split("_")) & _NON_NEGATIVE_NAME_WORDS
    ):
        return "must not be negative"
    return None


def out_of_range_value_message(field_name: str, field_value: Any, problem: str) -> str:
    """What the caller is told when *field_name* was given an out-of-range number."""
    return (
        f"{field_name} {field_value} is out of range ({problem}): pass the number "
        f"the request asks for, or leave {field_name} out if it does not give one."
    )


# --- "Not given" text in an Optional str ---------------------------------------------
#
# The extraction signature tells the model to answer ``NOT_FOUND`` (or ``None``, the
# displayed default of an Optional field) for a value the request does not give.
# For a required field ``NOT_FOUND`` means "ask the user", and that stays. For an
# ``Optional[str]`` field the text used to reach the command as the value, while
# the key=value and XML paths give the field's default; it is now replaced by the
# default, as the "not given" int marker is (fix-3t6p, the str analogue of fix-7t0q).

def is_optional_str_field(field_info: Any) -> bool:
    """Whether the field is ``Optional[str]`` (``str`` and ``None`` only)."""
    args = get_args(field_info.annotation)
    return type(None) in args and [t for t in args if t is not type(None)] == [str]


def default_unfilled_optional_strs(cmd_parameters: BaseModel, not_found: Optional[str]) -> List[str]:
    """Give each ``Optional[str]`` field holding ``not_found`` or ``"None"`` its default.

    Returns the names of the fields changed. The default is the field's declared one
    (a factory is called), or None when it declares none. A required ``str`` field holding
    ``"None"`` is set to None, the missing value: the literal text ``None`` is the extractor's
    "not given" marker, not a value, so the missing check reports the field as missing. A
    required field holding ``not_found`` is left alone (that check already counts it missing).
    """
    markers = {"None"} | ({not_found} if not_found else set())
    changed: List[str] = []
    for field_name, field_info in type(cmd_parameters).model_fields.items():
        value = getattr(cmd_parameters, field_name, None)
        if not isinstance(value, str) or isinstance(value, Enum) or value.strip() not in markers:
            continue
        if is_optional_str_field(field_info):
            default = field_info.get_default(call_default_factory=True)
            setattr(cmd_parameters, field_name, None if default is PydanticUndefined else default)
        elif value.strip() == "None":
            setattr(cmd_parameters, field_name, None)
        else:
            continue
        changed.append(field_name)
    return changed


# --- One pass over a call's values, shared by commands and planning command calls -----------------


@dataclass
class ParameterScreen:
    """What ``screen_parameter_values`` found and changed, per field name.

    ``invalid_fields`` lists ``"<field> '<value>'"`` for each refused field, in field
    order; a refused field is a placeholder or an out-of-range number. The other dicts map
    a field to the value it had before the change (``defaulted_fields``: a made-up number
    replaced by the field's default; the framework's own "not given" marker is replaced
    without being recorded).
    """

    placeholder_fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    unwrapped_fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    quoted_fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    defaulted_fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    out_of_range_fields: Dict[str, Any] = dataclass_field(default_factory=dict)
    out_of_range_problems: Dict[str, str] = dataclass_field(default_factory=dict)
    invalid_fields: List[str] = dataclass_field(default_factory=list)

    @property
    def refused_fields(self) -> Dict[str, Any]:
        return {**self.placeholder_fields, **self.out_of_range_fields}


def screen_parameter_values(cmd_parameters: BaseModel, quote_texts: Optional[str],
                            grounding_text: Optional[str],
                            field_names: Optional[Any] = None) -> ParameterScreen:
    """Check each value of *cmd_parameters* for wrappers, placeholders and made-up numbers.

    Writes the corrected values back onto *cmd_parameters*: a quote pair around a whole
    str value is removed (``unwrap_quoted_value``; *quote_texts* is the text the value was
    taken from), a real value inside angle brackets is unwrapped, and a sentinel or made-up
    number (``is_sentinel_number``) becomes the field's default when it has one. A
    bracketed placeholder, a placeholder-shaped value *grounding_text* does not write
    (``is_ungrounded_placeholder``) and a number out of its declared or implied range are
    refused (left in place, listed in ``invalid_fields``). A required number holding the
    "not given" marker is left as it is, for the caller to report as missing.

    *field_names*, when given, limits the pass to those fields. ``validate_parameters``
    runs this before any other check of a command's parameters; a planning command call runs it on its
    ``str`` and ``int`` arguments (``fastworkflow.planning_commands.check_arguments``).
    """
    screen = ParameterScreen()
    placeholder_fields = screen.placeholder_fields
    unwrapped_fields = screen.unwrapped_fields
    quoted_fields = screen.quoted_fields
    defaulted_fields = screen.defaulted_fields
    out_of_range_fields = screen.out_of_range_fields
    out_of_range_problems = screen.out_of_range_problems
    invalid_fields = screen.invalid_fields
    for field_name, field_info in type(cmd_parameters).model_fields.items():
        if field_names is not None and field_name not in field_names:
            continue
        field_value = getattr(cmd_parameters, field_name, None)
        original_value = field_value

        quoted, field_value = unwrap_quoted_value(field_value, quote_texts)
        kind, new_value = classify_bracketed_value(field_value, field_name)
        if kind == "unwrapped":
            unwrapped_fields[field_name] = original_value
            inner_quoted, new_value = unwrap_quoted_value(new_value, quote_texts)
            quoted = quoted or inner_quoted
            field_value = new_value
        if quoted:
            quoted_fields[field_name] = original_value
        if field_value is not original_value:
            setattr(cmd_parameters, field_name, field_value)
        # A declared Literal or Enum choice is never ungrounded: it is the field's own vocabulary,
        # and a choice that looks like a placeholder word (``code``) is not one. A bracketed
        # placeholder (``<code>``) is still refused: it is not the choice itself.
        if kind == "placeholder" or (
            kind == "unchanged"
            and not _is_declared_choice(field_info.annotation, field_value)
            and is_ungrounded_placeholder(field_value, field_name, grounding_text)
        ):
            placeholder_fields[field_name] = field_value
            invalid_fields.append(f"{field_name} '{field_value}'")
            continue

        numeric_kind = _numeric_kind(field_info.annotation)
        number = _numeric_view(field_value, numeric_kind) if numeric_kind else None
        if number is None:
            continue
        if is_sentinel_number(number):
            is_optional_field = type(None) in get_args(field_info.annotation)
            default = field_info.get_default(call_default_factory=True)
            if default is PydanticUndefined and is_optional_field:
                default = None
            if default is not PydanticUndefined:
                setattr(cmd_parameters, field_name, default)
                # The framework's own "not given" marker is routine; only a
                # number the model made up is worth recording.
                if number not in (INVALID_INT_VALUE, INVALID_FLOAT_VALUE):
                    defaulted_fields[field_name] = original_value
                continue
            if number in (INVALID_INT_VALUE, INVALID_FLOAT_VALUE):
                continue  # a required field not given: the caller reports it missing
            problem = "not a real value"
        else:
            problem = numeric_range_problem(field_name, field_info, number)
            if problem is None:
                continue
        out_of_range_fields[field_name] = field_value
        out_of_range_problems[field_name] = problem
        invalid_fields.append(f"{field_name} '{field_value}'")
    return screen


def get_trainset(subject_command_name,workflow_folderpath) -> List[Dict[str, Any]]:
        """Load labeled trainset from the command specific JSON file for LabeledFewShot"""

        trainset = []

        if not subject_command_name:
            return trainset
        
        try:
            trainset_file = f"{subject_command_name}_param_labeled.json"
            trainset_path=get_route_layer_filepath_model(workflow_folderpath,trainset_file)
            
            if os.path.exists(trainset_path):
                with open(trainset_path, "r") as f:
                    trainset_data = json.load(f)

                if 'valid_examples' in trainset_data:
                    for example_dict in trainset_data['valid_examples']:
                        try:
                            example = dspy.Example(**example_dict["fields"])
                            example = example.with_inputs(*example_dict["inputs"])
                            trainset.append(example)
                        except Exception as e:
                            logger.warning(f"Failed to parse example {example_dict}: {e}")                    
            
        except Exception as e:
            logger.warning(f"Error loading trainset for {subject_command_name}: {e}")
            
        return trainset


class DatabaseValidator:
    """Generic validator for database lookups with fuzzy matching"""
    
    @staticmethod
    def fuzzy_match(value: str, key_values: list[str], threshold: float = 0.6,
                    suggest_threshold: float = 0.7,
                    auto_apply_threshold: float = 0.2) -> Tuple[bool, Optional[str], List[str]]:
        """Find the closest matching value in the specified database.

        Three stages run in order, and each one short-circuits: a
        case-insensitive exact match, then Levenshtein, then ``difflib``.

        Parameters
        ----------
        threshold
            ``difflib`` cutoff for the final suggestion-only stage. Only values
            that no candidate came within ``suggest_threshold`` of reach it, so
            a loose cutoff here yields unrelated suggestions rather than useful
            ones — and because any non-empty suggestion list fails validation,
            those unrelated suggestions reject values that are merely absent
            from this particular candidate list.
        suggest_threshold
            Maximum Levenshtein distance at which a candidate is considered at
            all. Beyond this the value falls through to ``difflib``.
        auto_apply_threshold
            Maximum Levenshtein distance at which a *unique* match is reported
            as matched, which makes the framework overwrite the user's value.
            Kept strict deliberately: the distance at which a real typo lands
            and the distance at which an unrelated value lands are separable,
            and being generous here silently substitutes non-members (``Batman``
            scores 0.333 against a customer list). Note the metric is
            edits/length, so a one-edit typo on a value shorter than about five
            characters exceeds this and is offered as a suggestion instead.
            Raise it per command if an extra clarification turn costs more than
            a wrong substitution.
        """
        global NOT_FOUND
        if not NOT_FOUND:
            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

        if not value or value in [None, NOT_FOUND]:
            return False, None, []

        if not key_values:
            return False, None, []     

        match = next((v for v in key_values if value.lower() == v.lower()), None)
        if match is not None:
            return True, match, []

        best_matches, best_distance = find_best_matches(
            value, key_values, threshold = suggest_threshold, best_window = True)
        if best_matches:
            if len(best_matches) == 1 and best_distance <= auto_apply_threshold:
                return True, best_matches[0], []
            # Ambiguous, or close enough to offer but too far to apply unannounced.
            return False, None, best_matches

        lowercase_matches = get_close_matches(
            value.lower(), 
            [val.lower() for val in key_values], 
            n = 3, 
            cutoff = threshold
        )

        original_matches = [
            value 
            for match in lowercase_matches 
            for value in key_values 
            if value.lower() == match
        ]

        if original_matches:
            return False, None, original_matches

        return False, None, []


def _planning_command_of(app_workflow, subject_command_name: str):
    """The loaded planning command *subject_command_name* of *app_workflow*, or None."""
    folderpath = getattr(app_workflow, "folderpath", None)
    if not folderpath:
        return None
    directory = fastworkflow.RoutingRegistry.get_definition(folderpath).command_directory
    metadata = directory.map_command_2_metadata.get(subject_command_name)
    if metadata is None or not metadata.is_planning:
        return None
    return planning_commands.planning_command(folderpath, subject_command_name)


def _unfilled_required(planning, cmd_parameters: BaseModel) -> List[str]:
    """The required arguments of a planning command that hold no value (missing)."""
    unfilled = []
    for param in planning.params:
        value = getattr(cmd_parameters, param.name, None)
        if param.required and (value is None or value == NOT_FOUND
                               or (isinstance(value, str) and not value.strip())
                               or (isinstance(value, list) and not value)):
            unfilled.append(param.name)
    return unfilled


class InputForParamExtraction(BaseModel):
    """Extract parameters from a user query using step-by-step reasoning.
Today's date is {today}.
"""  
    command: str
    input_for_param_extraction_class: Optional[Type[Any]] = None
    input_for_param_extraction: Optional[Any] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create(cls, app_workflow: fastworkflow.Workflow, subject_command_name: str, subject_command: str):
        """
        Create an instance of InputForParamExtraction with a command string.
        
        Arguments:
            command: The user's request
        
        Returns:
            An instance of InputForParamExtraction   
        """
        subject_workflow_folderpath = app_workflow.folderpath
        subject_command_routing_definition = fastworkflow.RoutingRegistry.get_definition(subject_workflow_folderpath)
        
        input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
            subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
        
        input_for_param_extraction = None
        if input_for_param_extraction_class and input_for_param_extraction_class is not cls and hasattr(input_for_param_extraction_class, 'create'):
            try:
                input_for_param_extraction = input_for_param_extraction_class.create(
                    app_workflow, subject_command_name, subject_command)
            except Exception as e:
                logger.warning(f"Failed to create input_for_param_extraction: {e}")
        
        today=date.today()
        cls.__doc__ = cls.__doc__.format(today=today)
        
        return cls(
            command=subject_command,
            input_for_param_extraction_class=input_for_param_extraction_class,
            input_for_param_extraction=input_for_param_extraction
        )
    
    @staticmethod
    def create_signature_from_pydantic_model(
        pydantic_model: Type[BaseModel],
        sentences: Optional[Dict[str, Tuple[str, ...]]] = None,
    ) -> Type[dspy.Signature]:
        """
        Create a DSPy Signature class from a Pydantic model with type annotations.
        
        Args:
            pydantic_model: The Pydantic model class to convert
            sentences: ``{field: rule sentences}`` appended after that field's description
                (``planning_commands.extraction_sentences_for``); nothing else reads them
            
        Returns:
            A DSPy Signature class
        """
        signature_components = {
            "statement": (str, dspy.InputField(desc="Statement according to Dhar"))
        }

        steps = []
        field_num = 1

        for attribute_name, attribute_metadata in pydantic_model.model_fields.items():
            is_optional = False
            attribute_type = attribute_metadata.annotation

            if hasattr(attribute_type, "__origin__") and attribute_type.__origin__ is Union:
                union_elements = get_args(attribute_type)
                if type(None) in union_elements:
                    is_optional = True
                    attribute_type = next((elem for elem in union_elements if elem is not type(None)), str)

            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
            if attribute_type is str:            
                default_value = NOT_FOUND
            elif attribute_type is int:
                default_value = INVALID_INT_VALUE
            elif attribute_type is float:
                default_value = -sys.float_info.max
            else:
                default_value = None

            if (
                attribute_metadata.default is not PydanticUndefined and 
                attribute_metadata.default is not None and 
                attribute_metadata.default != Ellipsis
            ):
                default_value = attribute_metadata.default

            info_text = attribute_metadata.description or f"The {attribute_name}"
            if rule_sentences := (sentences or {}).get(attribute_name):
                # the rule sentences of the field's options (3.4, Descriptions), after its description;
                # the ". This field is ..." below ends the last one
                info_text = f"{info_text.rstrip('. ')}. {' '.join(s.rstrip('.') for s in rule_sentences)}"

            if attribute_name != "query": 
                steps.append(f"Step {field_num}: Identify the {attribute_name} ({info_text}).")
                field_num += 1

            if isinstance(attribute_type, type) and issubclass(attribute_type, Enum):
                possible_values = [f"'{option.value}'" for option in attribute_type]
                info_text += f". Valid values: {', '.join(possible_values)}"

            if attribute_metadata.examples:
                sample_values = ", ".join([f"'{sample}'" for sample in attribute_metadata.examples])
                info_text += f". Examples: {sample_values}"

            requirement_status = "Optional" if is_optional else "Required"
            info_text += f". This field is {requirement_status}."

            if is_optional:
                default_display = default_value if default_value is not None else 'None'
                info_text += f" If not mentioned in the query, use: '{default_display}'."
            elif default_value is not None:
                info_text += f" Default value: '{default_value}'."

            field_definition = dspy.OutputField(desc=info_text, default=default_value)
            signature_components[attribute_name] = (attribute_metadata.annotation, field_definition)

            steps.extend((
                f"Step {field_num}: ",
                "For missing parameter values - return the default if it is specified otherwise return None",
            ))
            
        generated_docstring = (
            f"Extract parameter values from the statement according to Dhar. Today's date is {date.today()}.\n"
            f"{chr(10).join(steps)}"
        )

        return dspy.Signature(signature_components, generated_docstring)
    
    def extract_parameters(self, CommandParameters: Type[BaseModel] = None, subject_command_name: str = None, workflow_folderpath: str = None) -> BaseModel:
        """
        Extract parameters from the command using DSPy.
        
        Returns:
            The extracted parameters
        """
        global PARAMETER_EXTRACTION_ERROR_MSG, LLM_PARAM_EXTRACTION, LITELLM_API_KEY_PARAM_EXTRACTION

        if not PARAMETER_EXTRACTION_ERROR_MSG:
            PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

        if not LLM_PARAM_EXTRACTION:
            LLM_PARAM_EXTRACTION = fastworkflow.get_env_var("LLM_PARAM_EXTRACTION")
            LITELLM_API_KEY_PARAM_EXTRACTION = fastworkflow.get_env_var("LITELLM_API_KEY_PARAM_EXTRACTION")

        lm = dspy_utils.get_lm("LLM_PARAM_EXTRACTION", "LITELLM_API_KEY_PARAM_EXTRACTION")
        
        model_class = CommandParameters 
        if model_class is None:
            raise ValueError("No model class provided")
        
        params_signature = self.create_signature_from_pydantic_model(
            model_class,
            planning_commands.option_sentences(model_class))
        
        class ParamExtractor(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.predictor = dspy.ChainOfThought(signature)

            @DSPyForward.intercept
            def forward(self, command=None):
                return self.predictor(statement=command)
        
        param_extractor = ParamExtractor(params_signature)
        
        trainset = get_trainset(subject_command_name,workflow_folderpath)
        length = len(trainset)

        param_dict = {}
        field_names = list(model_class.model_fields.keys())
        with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
            optimizer = dspy.LabeledFewShot(k=length)
            compiled_model = optimizer.compile(
                student=param_extractor,
                trainset=trainset
            )

            def basic_checks(args, pred):
                for field_name in field_names:
                    # return 0 if it extracts an example value instead of correct value | None
                    extracted_param_value = getattr(pred, field_name)
                    examples = model_class.model_fields[field_name].examples
                    if examples and extracted_param_value in examples:
                        return 0.0
                return 1.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.BestOfN(
                module=compiled_model, 
                N=3, 
                reward_fn=basic_checks, 
                threshold=1.0)

            try:
                dspy_result = best_of_3(command=self.command)
                for field_name in field_names:
                    default = model_class.model_fields[field_name].default
                    param_dict[field_name] = getattr(dspy_result, field_name, default)
            except Exception as exc:
                logger.warning(PARAMETER_EXTRACTION_ERROR_MSG.format(error=exc))

        # IMPORTANT: Do *not* instantiate the original model via the regular
        # constructor – that would invoke full validation including regex
        # pattern checks.  Instead use `model_construct`, which builds the
        # object in-place without any validation.
        return model_class.model_construct(**param_dict)  # type: ignore[arg-type]
    
    def _option_command(self, app_workflow, subject_command_name: str, cmd_parameters: BaseModel):
        """The checks of the command's ``Input`` options as a ``PlanningCommandCommand`` (2.46): its fields
        that carry an option, for an execution command; a planning command's own description otherwise."""
        from fastworkflow.workflow_agent import identifier_patterns

        folderpath = getattr(app_workflow, "folderpath", None)
        identifiers = identifier_patterns(folderpath) if folderpath else ()
        return planning_commands.option_command(type(cmd_parameters), subject_command_name, identifiers)

    def _check_option_arguments(self, command, cmd_parameters: BaseModel,
                                diagnostics: Optional[Dict[str, Any]]) -> Optional[str]:
        """The arguments of the command *command*'s check options, checked against the request (self.command),
        for a command of either kind (2.46, 3.4).

        ``check_verdicts`` (one classifier call per ``verify`` field, when a Jev LM is set) and then
        ``check_arguments``' guards; the guarded values are written back onto *cmd_parameters*.
        Returns the refusal's text, or None when every guard passes.
        """
        # workflow_agent imports this module (through command_executor), so it is read at call time.
        from fastworkflow.workflow_agent import _jev_lm

        raw = {param.name: getattr(cmd_parameters, param.name, None) for param in command.params}
        try:
            verdicts = planning_commands.check_verdicts(_jev_lm(), command, self.command)
            args, notes = planning_commands.check_arguments(command, raw, self.command, verdicts)
        except planning_commands.PlanningCommandCallError as refusal:
            if diagnostics is not None:
                diagnostics["argument_refusal"] = str(refusal)
            return str(refusal)
        if diagnostics is not None and notes:
            diagnostics["argument_notes"] = list(notes)
        for name, value in args.items():
            setattr(cmd_parameters, name, value)
        return None

    def _check_input_rules(self, command, cmd_parameters: BaseModel, app_workflow) -> Optional[str]:
        """The refusal text of the ``Input``'s own validators, for a command of either kind, or None.

        The validators read the request (self.command) and the workflow from the validation context
        (``planning_commands.check_input``, design 3.4).
        """
        args = {name: getattr(cmd_parameters, name, None) for name in type(cmd_parameters).model_fields}
        try:
            planning_commands.check_input(command, args, self.command, app_workflow)
        except planning_commands.PlanningCommandCallError as refusal:
            return str(refusal)
        return None

    def validate_parameters(self,
                            app_workflow: fastworkflow.Workflow,
                            subject_command_name: str,
                            cmd_parameters: BaseModel,
                            diagnostics: Optional[Dict[str, Any]] = None
                            ) -> Tuple[bool, str, Dict[str, List[str]], List[str]]:
        """
        Check if the parameters are valid in the current context, including database lookups.

        ``diagnostics`` (optional, additive): a caller-owned dict this method
        fills with STRUCTURED validation outcomes — db_lookup events (field,
        input value, correction applied / suggestions offered), the
        validate_extracted_parameters hook's verdict, and the missing/invalid
        field lists. The observability layer attaches it to the
        fw.nlu.param_extraction span so debugging agents read fields instead
        of parsing the prose error message. Passing None changes nothing.
        """
        global MISSING_INFORMATION_ERRMSG, INVALID_INFORMATION_ERRMSG, NOT_FOUND
        if not MISSING_INFORMATION_ERRMSG:
            MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
        if not INVALID_INFORMATION_ERRMSG:
            INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")
        if not NOT_FOUND:
            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

        is_valid = True
        missing_fields = []
        invalid_fields = []
        all_suggestions = {}

        # Wrappers and placeholders first, before type coercion, patterns,
        # db_lookup and the workflow's hook see the value: every extraction
        # method (stored merge, XML regex, LLM) and the direct-action path reach
        # the command through here, so this is where a copied usage-text
        # wrapper, kept quotes, a placeholder or a made-up number is caught.
        context = getattr(app_workflow, "context", None) or {}
        user_request = context.get("raw_user_message") if isinstance(context, dict) else None
        # A value the agent wrote is always in its own command text, so in agent
        # mode a placeholder is grounded against the user's request instead.
        grounding_text = (
            user_request
            if user_request and isinstance(context, dict) and context.get("run_as_agent")
            else self.command
        )
        quote_texts = " \n ".join(t for t in (self.command, user_request) if t)

        default_unfilled_optional_strs(cmd_parameters, NOT_FOUND)
        screen = screen_parameter_values(cmd_parameters, quote_texts, grounding_text)
        placeholder_fields = screen.placeholder_fields
        unwrapped_fields = screen.unwrapped_fields
        quoted_fields = screen.quoted_fields
        defaulted_fields = screen.defaulted_fields
        out_of_range_fields = screen.out_of_range_fields
        out_of_range_problems = screen.out_of_range_problems
        if screen.invalid_fields:
            invalid_fields.extend(screen.invalid_fields)
            is_valid = False

        refused_fields = screen.refused_fields
        # A command's name written as a value is treated as not given: the field is cleared
        # (to its default when it has one), so the missing check reports a required one.
        command_name_fields = command_name_refusals(
            cmd_parameters, subject_command_name, command_name_index(app_workflow, subject_command_name),
            refused_fields)
        # A free-text value that is only a pronoun, or only a word of the workflow's vocabulary, is
        # treated the same way (see word_screen_refusals).
        word_fields = word_screen_refusals(
            cmd_parameters, _workflow_vocabulary(app_workflow), {**refused_fields, **command_name_fields},
            subject_command_name)
        for field_name in {**command_name_fields, **word_fields}:
            field_info = type(cmd_parameters).model_fields[field_name]
            default = field_info.get_default(call_default_factory=True)
            setattr(cmd_parameters, field_name, None if default is PydanticUndefined else default)
        if diagnostics is not None:
            def _as_text(fields: Dict[str, Any]) -> Dict[str, str]:
                return {name: str(value) for name, value in fields.items()}
            if unwrapped_fields:
                diagnostics["unwrapped_bracketed_values"] = _as_text(unwrapped_fields)
            if quoted_fields:
                diagnostics["unwrapped_quoted_values"] = _as_text(quoted_fields)
            if placeholder_fields:
                diagnostics["placeholder_values"] = _as_text(placeholder_fields)
            if defaulted_fields:
                diagnostics["defaulted_sentinel_values"] = _as_text(defaulted_fields)
            if out_of_range_fields:
                diagnostics["out_of_range_values"] = _as_text(out_of_range_fields)
            if command_name_fields:
                diagnostics["command_name_values"] = {
                    name: str(value) for name, (value, _) in command_name_fields.items()}
            if word_fields:
                diagnostics["word_screen_values"] = {
                    name: str(value) for name, (value, _) in word_fields.items()}

        for field_name, field_info in type(cmd_parameters).model_fields.items():
                if field_name in refused_fields:
                    continue
                field_value = getattr(cmd_parameters, field_name, None)

                if field_value not in [NOT_FOUND, None, INVALID_INT_VALUE, INVALID_FLOAT_VALUE]:
                        annotation = field_info.annotation

                        # Build list of candidate concrete types (exclude NoneType from Union)
                        candidate_types: List[Type[Any]] = []
                        if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                            for t in get_args(annotation):
                                if t is not type(None):  # noqa: E721
                                    candidate_types.append(t)  # type: ignore[arg-type]
                        else:
                            candidate_types = [annotation]  # type: ignore[list-item]

                        def build_type_suggestion() -> List[str]:
                            examples = getattr(field_info, "examples", []) or []
                            example = examples[0] if examples else None
                            # Enum and Literal suggestions list valid values
                            enum_types = [t for t in candidate_types if isinstance(t, type) and issubclass(t, Enum)]
                            literal_values = [v for t in candidate_types if get_origin(t) is Literal for v in get_args(t)]
                            if enum_types or literal_values:
                                opts = [f"'{opt.value}'" for t in enum_types for opt in t] + [f"'{v}'" for v in literal_values]
                                return [f"Please provide a value matching the expected type/format. Valid values: {', '.join(opts)}"]
                            # List suggestions
                            def _is_list_type(tt):
                                try:
                                    return hasattr(tt, "__origin__") and tt.__origin__ in (list, List)
                                except Exception:
                                    return False
                            list_types = [t for t in candidate_types if _is_list_type(t)]
                            if list_types:
                                inner_args = get_args(list_types[0])
                                inner = inner_args[0] if inner_args else str
                                inner_name = inner.__name__ if isinstance(inner, type) else str(inner)
                                hint = (
                                    f"Please provide a list of {inner_name} values. Accepted formats: "
                                    f"JSON list (e.g., [\"a\", \"b\"]), Python list (e.g., ['a', 'b']), "
                                    f"or comma-separated (e.g., a,b)."
                                )
                                return [hint]
                            # Fallback: show expected type names (handles unions)
                            name_list: List[str] = []
                            for t in candidate_types:
                                if isinstance(t, type):
                                    name_list.append(t.__name__)
                                else:
                                    name_list.append(str(t))
                            base = f"Please provide a value matching the expected type/format: {' or '.join(name_list)}"
                            if example is not None:
                                base = f"{base} (e.g., {example})"
                            return [base]

                        valid_by_type = False
                        corrected_value: Optional[Any] = None
                        def _is_list_type(tt):
                            try:
                                return hasattr(tt, "__origin__") and tt.__origin__ in (list, List)
                            except Exception:
                                return False

                        def _parse_list_like_string(s: str) -> Optional[list]:
                                if not isinstance(s, str):
                                    return None
                                text = s.strip()
                                if text.startswith("[") and text.endswith("]"):
                                        with suppress(Exception):
                                                parsed = json.loads(text)
                                                if isinstance(parsed, list):
                                                    return parsed
                                # Try Python literal list
                                with suppress(Exception):
                                        parsed = ast.literal_eval(text)
                                        if isinstance(parsed, list):
                                            return parsed
                                # Comma-separated values
                                if "," in text:
                                    parts = [p.strip() for p in text.split(",")]
                                    cleaned = [
                                        (p[1:-1] if len(p) >= 2 and ((p[0] == p[-1] == '"') or (p[0] == p[-1] == "'")) else p)
                                        for p in parts
                                    ]
                                    return cleaned
                                # Single value - treat as a list with one element
                                if text:
                                    # Remove quotes if present
                                    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
                                        return [text[1:-1]]
                                    return [text]
                                return None

                        def _coerce_scalar(expected_type: Type[Any], val: Any) -> Tuple[bool, Optional[Any]]:
                                # str
                                if expected_type is str:
                                    return True, str(val)
                                # bool
                                if expected_type is bool:
                                    if isinstance(val, bool):
                                        return True, val
                                    elif isinstance(val, str):
                                        lower_val = val.lower().strip()
                                        if lower_val in ('true', 'false'):
                                            return True, lower_val == 'true'
                                        # Also handle string representations of integers
                                        elif lower_val in ('0', '1'):
                                            return True, lower_val == '1'
                                    elif isinstance(val, int):
                                        return True, bool(val)
                                    return False, None
                                # int
                                if expected_type is int:
                                    if isinstance(val, bool):
                                        return False, None
                                    if isinstance(val, int):
                                        return True, val
                                    if isinstance(val, str):
                                        with suppress(Exception):
                                            return True, int(val.strip())
                                    return False, None
                                # float
                                if expected_type is float:
                                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                                        return True, float(val)
                                    if isinstance(val, str):
                                        with suppress(Exception):
                                            return True, float(val.strip())
                                    return False, None
                                # Enum
                                if isinstance(expected_type, type) and issubclass(expected_type, Enum):
                                    ok, enum_val = _try_coerce_enum(expected_type, val)
                                    return (ok, enum_val if ok else None)
                                # Pydantic BaseModel: construct from dict
                                if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                                    if isinstance(val, expected_type):
                                        return True, val
                                    if isinstance(val, dict):
                                        with suppress(Exception):
                                            return True, expected_type(**val)
                                    return False, None
                                # Unknown: accept if already instance
                                return (True, val) if isinstance(val, expected_type) else (False, None)

                        def _try_coerce_list(list_type: Any, value: Any) -> Tuple[bool, Optional[list]]:
                            inner_args = get_args(list_type)
                            inner_type = inner_args[0] if inner_args else str
                            raw_list: Optional[list] = None
                            if isinstance(value, list):
                                raw_list = value
                            elif isinstance(value, str):
                                raw_list = _parse_list_like_string(value)
                            if raw_list is None:
                                return False, None
                            coerced_list = []
                            for item in raw_list:
                                ok, coerced = _coerce_scalar(inner_type, item)
                                if not ok:
                                    return False, None
                                coerced_list.append(coerced)
                            return True, coerced_list

                        def _try_coerce_enum(enum_cls: Type[Enum], val: Any) -> Tuple[bool, Optional[Enum]]:
                            if isinstance(val, enum_cls):
                                return True, val
                            if isinstance(val, str):
                                for member in enum_cls:
                                    if val == member.value or val.lower() == str(member.name).lower():
                                        return True, member
                            return False, None

                        def _try_coerce_literal(literal_type: Any, val: Any) -> Tuple[bool, Optional[str]]:
                            # A member as written, else the one member it matches in any case (as an enum's name)
                            members = get_args(literal_type)
                            if val in members:
                                return True, val
                            if isinstance(val, str):
                                folded = [m for m in members if isinstance(m, str) and m.lower() == val.lower()]
                                if len(folded) == 1:
                                    return True, folded[0]
                            return False, None

                        for t in candidate_types:
                            # list[...] and typing.List[...] support
                            if _is_list_type(t):
                                ok, coerced_list = _try_coerce_list(t, field_value)
                                if ok:
                                    corrected_value = coerced_list
                                    valid_by_type = True
                                    break
                            # str, bool, int, float - use _coerce_scalar for consistency
                            if t in (str, bool, int, float):
                                ok, coerced = _coerce_scalar(t, field_value)
                                if ok:
                                    corrected_value = coerced
                                    valid_by_type = True
                                    break                            # Enum
                            if isinstance(t, type) and issubclass(t, Enum):
                                ok, enum_val = _try_coerce_enum(t, field_value)
                                if ok:
                                    corrected_value = enum_val
                                    valid_by_type = True
                                    break
                            # Literal (a planning command's choice of text values): a member, as an enum's value
                            if get_origin(t) is Literal:
                                ok, literal_val = _try_coerce_literal(t, field_value)
                                if ok:
                                    corrected_value = literal_val
                                    valid_by_type = True
                                    break

                        if valid_by_type:
                            if corrected_value is not None:
                                pattern = next(
                                    (meta.pattern
                                        for meta in getattr(field_info, "metadata", [])
                                        if hasattr(meta, "pattern")),
                                    None,
                                )
                                if pattern and field_value is not None and field_value != NOT_FOUND:
                                    invalid_value = None
                                    if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                                        invalid_value = field_info.json_schema_extra.get("invalid_value")

                                    # if invalid_value and field_value == invalid_value:
                                    #     invalid_fields.append(f"{field_name} '{field_value}'")
                                    #     pattern_str = str(pattern)
                                    #     examples = getattr(field_info, "examples", [])
                                    #     example = examples[0] if examples else ""
                                    #     all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                                    #     is_valid = False

                                    # else:
                                    pattern_regex = re.compile(pattern)
                                    if not pattern_regex.fullmatch(str(field_value)):
                                        invalid_fields.append(f"{field_name} '{field_value}'")
                                        pattern_str = str(pattern)
                                        examples = getattr(field_info, "examples", [])
                                        example = examples[0] if examples else ""

                                        invalid_fields.append(f"{field_name} '{field_value}'")
                                        all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                                        is_valid = False
                                    else:
                                        try:
                                            setattr(cmd_parameters, field_name, corrected_value)
                                        except Exception as e:
                                            logger.critical(f"Failed to set attribute {field_name} with value {corrected_value}")
                                            raise e
                                else:
                                    try:
                                        setattr(cmd_parameters, field_name, corrected_value)
                                    except Exception as e:
                                        logger.critical(f"Failed to set attribute {field_name} with value {corrected_value}")
                                        raise e
                        else:
                            invalid_fields.append(f"{field_name} '{field_value}'")
                            all_suggestions[field_name] = build_type_suggestion()
                            is_valid = False

                is_optional = False
                attribute_type = field_info.annotation
                if hasattr(attribute_type, "__origin__") and attribute_type.__origin__ is Union:
                    union_elements = get_args(attribute_type)
                    if type(None) in union_elements:
                        is_optional = True

                is_required=True
                if is_optional:
                        is_required=False

                # Check if this is a list field
                is_list_field = False
                check_type = field_info.annotation
                check_origin = get_origin(check_type)
                if check_origin is Union:
                    check_args = get_args(check_type)
                    non_none = [t for t in check_args if t is not type(None)]
                    if non_none:
                        check_type = non_none[0]
                        check_origin = get_origin(check_type)
                if check_origin in (list, List):
                    is_list_field = True

                # Only add to missing fields if it's required AND has no value
                # For list fields, empty list [] is considered missing
                if is_required:
                    is_missing = False
                    if is_list_field:
                        # For list fields, treat empty list as missing
                        if field_value in [NOT_FOUND, None] or (isinstance(field_value, list) and len(field_value) == 0):
                            is_missing = True
                    else:
                        # For non-list fields, use standard sentinel values
                        if field_value in [NOT_FOUND, None, INVALID_INT_VALUE, INVALID_FLOAT_VALUE]:
                            is_missing = True

                    if is_missing:
                        missing_fields.append(field_name)
                        is_valid = False

        for field_name, field_info in type(cmd_parameters).model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)

            if field_value in [NOT_FOUND, None] or field_name in refused_fields:
                continue

            is_db_lookup = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                is_db_lookup = field_info.json_schema_extra.get("db_lookup")

            if is_db_lookup:
                if not self.input_for_param_extraction_class:
                    raise ValueError("input_for_param_extraction_class is not set.")
                matched, corrected_value, field_suggestions = (
                    self.input_for_param_extraction_class.db_lookup(
                        app_workflow, field_name, field_value)
                )
                # matched, corrected_value, field_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)

                if diagnostics is not None:
                    # The three-state contract, recorded structurally:
                    # applied (value possibly rewritten), rejected (suggestions
                    # offered), declined (no opinion — value left alone).
                    diagnostics.setdefault("db_lookup", []).append({
                        "field": field_name,
                        "input_value": str(field_value),
                        "outcome": (
                            "applied" if matched
                            else "rejected" if field_suggestions
                            else "declined"
                        ),
                        "corrected_value": (
                            str(corrected_value) if matched else None
                        ),
                        "corrected": bool(matched) and corrected_value != field_value,
                        "suggestions": [str(s) for s in (field_suggestions or [])],
                    })

                if matched:
                    if corrected_value != field_value:
                        logger.debug(
                            f"db_lookup substituted {subject_command_name}.{field_name}: "
                            f"'{field_value}' -> '{corrected_value}'"
                        )
                    setattr(cmd_parameters, field_name, corrected_value)
                elif field_suggestions:
                    invalid_fields.append(f"{field_name} '{field_value}'")
                    all_suggestions[field_name] = field_suggestions
                    is_valid = False

        planning = _planning_command_of(app_workflow, subject_command_name)
        if is_valid and planning is not None:
            unfilled = _unfilled_required(planning, cmd_parameters)
            if unfilled:
                missing_fields.extend(unfilled)
                is_valid = False
        # One validation step for every command (3.4, The argument path): the options' checks, then the
        # hook, then the Input's validators. A command's checks are its option fields (2.46).
        command = planning if planning is not None else self._option_command(
            app_workflow, subject_command_name, cmd_parameters)
        if is_valid and command.params:
            guard_refusal = self._check_option_arguments(command, cmd_parameters, diagnostics)
            if guard_refusal is not None:
                return (False, guard_refusal, {}, [])

        if is_valid and self.input_for_param_extraction_class and \
                hasattr(self.input_for_param_extraction_class, 'validate_extracted_parameters'):
            try:
                hook = self.input_for_param_extraction_class.validate_extracted_parameters
                # The command text goes only to a hook that declares `command_text`,
                # so every existing (workflow, command, cmd_parameters) hook is
                # called exactly as before.
                hook_kwargs = (
                    {"command_text": self.command}
                    if "command_text" in inspect.signature(hook).parameters
                    else {}
                )
                try:
                    is_valid, message = hook(app_workflow, subject_command_name, cmd_parameters,
                                             **hook_kwargs)
                except fastworkflow.Refused as refusal:
                    is_valid, message = False, str(refusal)
                if diagnostics is not None:
                    diagnostics["validation_hook"] = {
                        "ran": True,
                        "is_valid": bool(is_valid),
                        "message": None if is_valid else str(message),
                    }
            except Exception as e:
                message = f"Exception in {subject_command_name}'s validate_extracted_parameters function: {str(e)}"
                logger.critical(message)
                if diagnostics is not None:
                    diagnostics["validation_hook"] = {
                        "ran": True,
                        "is_valid": False,
                        "message": message,
                        "raised": type(e).__name__,
                    }
                return (False, message, {}, [])

            if not is_valid:
                return (False, message, {}, [])

        if is_valid:
            input_refusal = self._check_input_rules(command, cmd_parameters, app_workflow)
            if input_refusal is not None:
                return (False, input_refusal, {}, [])
        if is_valid:
            return (True, "All required parameters are valid.", {}, [])

        if diagnostics is not None:
            diagnostics["missing_fields"] = list(missing_fields)
            diagnostics["invalid_fields"] = [
                field.split(" '")[0].strip() for field in invalid_fields
            ]

        message = ''
        if missing_fields:
            message += f"{MISSING_INFORMATION_ERRMSG}" + ", ".join(missing_fields) + "\n"

            for missing_field in missing_fields:
                is_available_from = None
                if hasattr(type(cmd_parameters).model_fields.get(missing_field), "json_schema_extra") and type(cmd_parameters).model_fields.get(missing_field).json_schema_extra:
                    is_available_from = type(cmd_parameters).model_fields.get(missing_field).json_schema_extra.get("available_from")
                if is_available_from:
                    msg_prefix = "abort and "
                    if "run_as_agent" in app_workflow.context:
                        msg_prefix = ""
                    bare_commands = [
                        str(entry).split("/")[-1].split(".", 1)[0]
                        for entry in is_available_from
                    ]
                    message += f"{msg_prefix}use the {' or '.join(bare_commands)} command(s) to get {missing_field} information. OR...\n"

        if invalid_fields:
            message += f"{INVALID_INFORMATION_ERRMSG}" + ", ".join(invalid_fields) + "\n"

        for field, value in placeholder_fields.items():
            message += placeholder_value_message(field, value) + "\n"

        for _, wording in command_name_fields.values():
            message += wording + "\n"

        for _, wording in word_fields.values():
            message += wording + "\n"

        for field, value in out_of_range_fields.items():
            message += out_of_range_value_message(field, value, out_of_range_problems[field]) + "\n"

        for field, suggestions in all_suggestions.items():
            if suggestions:
                is_format_instruction = any(("format" in str(s).lower() or "pattern" in str(s).lower()) for s in suggestions)

                if is_format_instruction:
                    message += f"\n{field}: {', '.join(suggestions)}"
                else:
                    message += f"Did you mean one of these {field}s? {', '.join(suggestions)}\n"

        combined_fields = []
        if missing_fields:
            combined_fields.extend(missing_fields)
        if invalid_fields:
            invalid_field_names = [field.split(" '")[0].strip() for field in invalid_fields]
            combined_fields.extend(invalid_field_names)

        if combined_fields:
            combined_fields_str = ", ".join(combined_fields)
            message += f"\nProvide corrected parameter values in the exact order specified below, separated by commas:\n{combined_fields_str}"
            if "run_as_agent" not in app_workflow.context:
                message += "\nFor parameter values that include a comma, provide separately from other values, and one at a time."

        return (False, message, all_suggestions, combined_fields)