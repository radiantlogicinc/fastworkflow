"""The author-facing field options of a planning command or execution command ``Input`` (spec section 2).

An author writes plain pydantic ``Field(..., json_schema_extra={...})``. Each key names one behaviour; its
value is ``True`` or a dict of that key's own options. The loader (``planning_commands.option_params``,
``planning_commands.check_option_spec``) reads these keys and writes the same internal options as the
spellings they replace, so the runtime behaves as before.

``FieldOptions`` is a plain dict at runtime. Its TypedDicts give IDE completion and typo detection:

    json_schema_extra=FieldOptions(copied_from_command_text={"ignore": ["the"]})
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

#: ``copied_from_command_text``: the value is written in the command text (a ``str``, an ``int`` or a
#: ``list[str]``). Its ``ignore`` lists words stripped from either end of a ``str`` before the match.
COPIED_FROM_COMMAND_TEXT = "copied_from_command_text"
#: ``record_named_in_command_text``: a ``str`` naming a record the command text names (a subject).
RECORD_NAMED_IN_COMMAND_TEXT = "record_named_in_command_text"
#: ``classifier_confirms``: a classifier checks the value again from the command text.
CLASSIFIER_CONFIRMS = "classifier_confirms"
#: ``classifier_decides``: a bool the classifier alone decides from the command text.
CLASSIFIER_DECIDES = "classifier_decides"
#: ``command_text_after_command_name``: a ``str`` that takes the text after the command name (execution commands).
COMMAND_TEXT_AFTER_COMMAND_NAME = "command_text_after_command_name"
#: ``from_command_output``: the command (and field) whose output supplies the value.
FROM_COMMAND_OUTPUT = "from_command_output"
#: ``converted_from_command_text``: a ``str`` or ``int`` the command text is converted into, so it is not
#: checked against the command text (execution commands; the field keeps its placeholder checks).
CONVERTED_FROM_COMMAND_TEXT = "converted_from_command_text"

#: Every author-facing option key.
OPTION_KEYS = (
    COPIED_FROM_COMMAND_TEXT,
    RECORD_NAMED_IN_COMMAND_TEXT,
    CLASSIFIER_CONFIRMS,
    CLASSIFIER_DECIDES,
    COMMAND_TEXT_AFTER_COMMAND_NAME,
    FROM_COMMAND_OUTPUT,
    CONVERTED_FROM_COMMAND_TEXT,
)


class CopiedOptions(TypedDict, total=False):
    """Options of ``copied_from_command_text`` (a dict form)."""

    ignore: list[str]


class RecordNamedOptions(TypedDict, total=False):
    """Options of ``record_named_in_command_text`` (a dict form)."""

    ignore: list[str]


class ConfirmsOptions(TypedDict, total=False):
    """Options of ``classifier_confirms`` (a dict form, on a ``Literal`` only): each value's meaning."""

    meanings: dict[str, str]


class FieldOptions(TypedDict, total=False):
    """The ``json_schema_extra`` keys of a field of a planning command or execution command ``Input``.

    Optional: a plain dict with the same keys works. Leave a key out to turn its behaviour off.
    """

    copied_from_command_text: Union[Literal[True], CopiedOptions]
    record_named_in_command_text: Union[Literal[True], RecordNamedOptions]
    classifier_confirms: Union[Literal[True], ConfirmsOptions]
    classifier_decides: Literal[True]
    command_text_after_command_name: Literal[True]
    from_command_output: Union[str, list[str]]
    converted_from_command_text: Literal[True]
