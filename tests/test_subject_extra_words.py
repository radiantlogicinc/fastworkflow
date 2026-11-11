"""A value that is a record the request names with extra words run on is the record, and nothing else changes.

Real components throughout: ``planning_commands.option_command`` (a command with one ``record_named_in_command_text``
field), the real ``check_arguments`` and the real ``split_merge.request_subjects``. No model is called. The drafts are
recorded ones: the extractor's values for sub-tasks of an internal benchmark, each with the value
the reviewer ended with (``fixed``) or the value that was right and must stay (``kept``).
"""
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from fastworkflow import planning_commands as pc

DRAFTS = json.loads((Path(__file__).parent / "fixtures" / "recorded_subject_drafts.json").read_text())


class Named(BaseModel):
    value: str = Field(description="the record's name exactly as the request writes it",
                       json_schema_extra={"record_named_in_command_text": True})


COMMAND = pc.option_command(Named, "Library/show_record")


def _check(request: str, value: str) -> str:
    args, _ = pc.check_arguments(COMMAND, {"value": value}, request)
    return args["value"]


@pytest.mark.parametrize("draft", [d for d in DRAFTS["fixed"] if d["field"] != "unit"],
                         ids=lambda d: f"{d['value'][:30]}->{d['expected'][:30]}")
def test_a_value_with_extra_words_run_on_becomes_the_record_the_request_names(draft):
    assert _check(draft["request"], draft["value"]) == draft["expected"]


@pytest.mark.parametrize("draft", DRAFTS["kept"], ids=lambda d: f"{d['value'][:40]}")
def test_a_right_value_is_kept(draft):
    assert _check(draft["request"], draft["value"]) == draft["expected"]


@pytest.mark.parametrize("request_text, value, expected", [
    # one lower-case word after a name is part of it ("admins"): kept, as the tag_record field says
    ("Bamboo HR admins no longer confers anything privileged. Remove its `privileged` tag.",
     "Bamboo HR admins", "Bamboo HR admins"),
    # a subject that is only the start of a longer identifier is not the value ("GitLab" of "GitLab_Admin")
    ("Who holds GitLab_Admin? Also tell me about GitLab.", "GitLab_Admin", "GitLab_Admin"),
    # the snapped subject is the first one the value holds, with only lower-case words before it
    ("Ian Morales has access to Confluence: what it exposes, who already holds it.",
     "access to Confluence: what it exposes, who already holds it", "Confluence"),
    # an unquoted sentence-case title is one name: its opening capitalized word is not snapped to
    ("Why was Active identities who are contractors with past ending date flagged?",
     "Active identities who are contractors with past ending date",
     "Active identities who are contractors with past ending date"),
    # a value the request quotes whole is kept as written
    ("Explain the control 'Snapshot of the quarterly access review'.",
     "Snapshot of the quarterly access review", "Snapshot of the quarterly access review"),
])
def test_the_limits_of_the_extra_words_rule(request_text, value, expected):
    assert _check(request_text, value) == expected
