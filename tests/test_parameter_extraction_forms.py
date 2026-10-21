"""Deterministic parameter forms, end to end through ParameterExtraction.

Covers the ``"<command_name>: "`` prefix the caller adds (fix-hia1.11), the
positional form (fix-hia1.16), the refusal of a lone value that binds to nothing
(fix-hia1.14), the ``deterministic_form`` diagnostic (fix-hia1.15), and a pending
parameter round met by a new full command or by an answer (fix-hia1.17). No LLM:
the LLM extractor is mocked, and asserted not to run where code should bind.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from fastworkflow import NLUPipelineStage
from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe_module
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    NOT_FOUND,
    ParameterExtraction,
)

FIND = "Shelf/find_book"
OPEN = "Shelf/open_book"


class _FindParams(BaseModel):
    query: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


class _MarkedFindParams(BaseModel):
    query: Optional[str] = Field(default=None, json_schema_extra={"positional": True})
    start: Optional[int] = Field(default=None)


class _OpenParams(BaseModel):
    book_uid: str = Field(description="uid")


class _TwoStrings(BaseModel):
    book_uid: Optional[str] = Field(default=None)
    filter: Optional[str] = Field(default=None)


class _MarkedTwoStrings(BaseModel):
    book_uid: Optional[str] = Field(default=None, json_schema_extra={"positional": True})
    filter: Optional[str] = Field(default=None)


def _pos(text, cls, name=FIND):
    return ParameterExtraction._extract_parameters_from_positional(text, cls, name)


# ---------------------------------------------------------------------------
# The extractor, with routing and the LLM mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def harness():
    """Run ParameterExtraction for a parameters class; the LLM must not be needed."""
    state = {"cls": None, "llm": None}

    definition = MagicMock()
    definition.get_command_class.side_effect = lambda name, kind: state["cls"]
    validator = MagicMock()
    validator.validate_parameters.return_value = (True, "", {}, [])

    def llm(cls, command_name, folder):
        if state["llm"] is None:
            raise AssertionError("the LLM extractor ran")
        return state["llm"](cls)

    validator.extract_parameters.side_effect = llm

    def run(cls, text, command_name=FIND, agent=True, cme_context=None, llm_result=None):
        state["cls"], state["llm"] = cls, llm_result
        cme = MagicMock()
        cme.context = {} if cme_context is None else cme_context
        app = MagicMock()
        app.context = {"run_as_agent": True} if agent else {}
        extractor = ParameterExtraction(cme, app, command_name, text)
        diagnostics: dict = {}
        output = extractor._extract_impl(diagnostics)
        return output, diagnostics, extractor

    run.use = lambda cls, llm_result=None: state.update(cls=cls, llm=llm_result)

    with patch("fastworkflow.RoutingRegistry") as registry, \
            patch.object(pe_module, "InputForParamExtraction") as input_cls:
        registry.get_definition.return_value = definition
        input_cls.create.return_value = validator
        yield run


def test_prefixed_key_value_binds_without_the_llm(harness):
    # fix-hia1.11: the caller sends "Ctx/cmd: cmd k=v"; stripping only the name
    # left ": cmd k=v", which the key=value parser rejected.
    output, diagnostics, extractor = harness(
        _FindParams, f"{FIND}: find_book query=Ann Lee 2 start=5")
    assert output.parameters_are_valid
    assert (output.cmd_parameters.query, output.cmd_parameters.start) == ("Ann Lee 2", 5)
    assert extractor.command == "find_book query=Ann Lee 2 start=5"
    assert diagnostics["extraction_method"] == "xml_regex"
    assert diagnostics["deterministic_form"] == "key_value"


def test_prefix_strip_leaves_a_value_naming_the_command_alone(harness):
    output, _, _ = harness(_FindParams, f"{FIND}: find_book query=Shelf/find_book notes")
    assert output.cmd_parameters.query == "Shelf/find_book notes"


def test_extract_wraps_the_same_path(harness):
    # The public entry point (span wrapper included), not just _extract_impl.
    harness.use(_FindParams)
    cme = MagicMock()
    cme.context = {}
    app = MagicMock()
    app.context = {"run_as_agent": True}
    result = ParameterExtraction(cme, app, FIND, f"{FIND}: {FIND} query=Ann Lee").extract()
    assert result.parameters_are_valid
    assert result.cmd_parameters.query == "Ann Lee"


def test_positional_form_binds_and_is_recorded(harness):
    output, diagnostics, _ = harness(_MarkedFindParams, f"{FIND}: find_book Ann Lee 2")
    assert (output.cmd_parameters.query, output.cmd_parameters.start) == ("Ann Lee 2", None)
    assert diagnostics["extraction_method"] == "xml_regex"
    assert diagnostics["deterministic_form"] == "positional"


def test_xml_records_no_deterministic_form(harness):
    _, diagnostics, _ = harness(_OpenParams, f"{OPEN}: open_book <book_uid>b1</book_uid>",
                                command_name=OPEN)
    assert diagnostics["extraction_method"] == "xml_regex"
    assert "deterministic_form" not in diagnostics


def test_assistant_mode_does_not_parse_in_code(harness):
    output, diagnostics, _ = harness(
        _FindParams, f"{FIND}: find_book query=Ann", agent=False,
        llm_result=lambda cls: cls(query="from llm"))
    assert output.cmd_parameters.query == "from llm"
    assert diagnostics["extraction_method"] == "llm"
    assert "deterministic_form" not in diagnostics


# ---------------------------------------------------------------------------
# a4: a lone value that binds to nothing is refused
# ---------------------------------------------------------------------------

def test_lone_unbound_value_is_refused_with_the_parameter_names(harness):
    output, diagnostics, _ = harness(
        _TwoStrings, f"{FIND}: find_book d397a1", llm_result=lambda cls: cls())
    assert not output.parameters_are_valid
    assert "'d397a1' was not bound to any parameter" in output.error_msg
    assert "book_uid=d397a1" in output.error_msg and "filter=d397a1" in output.error_msg
    assert diagnostics["unbound_value"] == "d397a1"
    assert diagnostics["extraction_method"] == "llm"


def test_lone_value_the_llm_bound_runs(harness):
    output, diagnostics, _ = harness(
        _TwoStrings, f"{FIND}: find_book d397a1",
        llm_result=lambda cls: cls(book_uid="d397a1"))
    assert output.parameters_are_valid
    assert "unbound_value" not in diagnostics


def test_lone_value_with_a_positional_mark_binds_in_code(harness):
    output, _, _ = harness(_MarkedTwoStrings, f"{FIND}: find_book d397a1")
    assert output.parameters_are_valid
    assert output.cmd_parameters.book_uid == "d397a1"


def test_several_words_or_no_value_are_not_lone_values(harness):
    output, _, _ = harness(_TwoStrings, f"{FIND}: find_book", llm_result=lambda cls: cls())
    assert output.parameters_are_valid
    output, _, _ = harness(_TwoStrings, f"{FIND}: find_book all of them",
                           llm_result=lambda cls: cls())
    assert output.parameters_are_valid


def test_unbound_message_names_only_fields_that_take_the_value():
    message = ParameterExtraction._unbound_value_message("abc", _FindParams)
    assert "query=abc" in message and "start=" not in message


# ---------------------------------------------------------------------------
# a3: the guarded positional form
# ---------------------------------------------------------------------------

def test_positional_single_token_to_the_only_string_field():
    assert _pos("find_book Atlas", _FindParams).query == "Atlas"
    assert _pos("Shelf/find_book Atlas", _FindParams).query == "Atlas"
    assert _pos("open_book a6c42ac0", _OpenParams, OPEN).book_uid == "a6c42ac0"


def test_positional_unmarked_field_takes_one_token_only():
    assert _pos("find_book Ann Lee archived-shelf", _FindParams) is None


def test_positional_marked_field_takes_words_and_keeps_a_trailing_number():
    result = _pos("find_book Reading Room Clerk 1", _MarkedFindParams)
    assert (result.query, result.start) == ("Reading Room Clerk 1", None)


def test_positional_marked_field_wins_over_other_strings():
    assert _pos("find_book d397a1", _MarkedTwoStrings).book_uid == "d397a1"
    assert _pos("find_book d397a1", _TwoStrings) is None


def test_positional_strips_a_leading_name_of_the_target():
    assert _pos("open_book book_uid f4d0", _OpenParams, OPEN).book_uid == "f4d0"
    assert _pos("open_book uid 203b", _OpenParams, OPEN).book_uid == "203b"
    assert _pos("find_book query Ann Lee", _MarkedFindParams).query == "Ann Lee"
    assert _pos("open_book uid", _OpenParams, OPEN) is None


def test_positional_declines_a_leading_name_of_another_field():
    assert _pos("open_book shelf_uid 2b55", _OpenParams, OPEN) is None
    assert _pos("find_book start 5", _MarkedFindParams) is None


def test_positional_declines_doubtful_values():
    for text in ("find_book **Ann**", "find_book * Ann", "find_book `Atlas`",
                 "find_book Ann?", "find_book a\\b", "find_book a / b",
                 "find_book <name>", "find_book {x}", "find_book .", "find_book --"):
        assert _pos(text, _MarkedFindParams) is None, text


def test_positional_strips_matching_quotes():
    assert _pos('find_book "Ann Lee"', _MarkedFindParams).query == "Ann Lee"
    assert _pos("find_book 'Atlas'", _FindParams).query == "Atlas"


def test_positional_needs_this_command_as_lead_and_a_plain_text():
    assert _pos("open_book Atlas", _FindParams) is None  # another command's name
    assert _pos("please find Atlas", _FindParams) is None
    assert _pos("find_book", _FindParams) is None
    assert _pos("find_book query=Atlas", _FindParams) is None
    assert _pos("find_book <query>Atlas</query>", _FindParams) is None


def test_positional_declines_two_marks():
    class _TwoMarks(BaseModel):
        a: Optional[str] = Field(default=None, json_schema_extra={"positional": True})
        b: Optional[str] = Field(default=None, json_schema_extra={"positional": True})

    assert _pos("find_book x", _TwoMarks) is None


# ---------------------------------------------------------------------------
# b2: a pending parameter round meets a new full command, or an answer
# ---------------------------------------------------------------------------

class _ListParams(BaseModel):
    scope: str = Field(description="scope")
    scope_uid: str = Field(description="uid")


LIST = "Shelf/list_books"


def _pending_round(stored, command=LIST):
    workflow = MagicMock()
    app_workflow = MagicMock()
    app_workflow.current_command_context_name = "Shelf"
    workflow.context = {
        "app_workflow": app_workflow,
        "NLU_Pipeline_Stage": NLUPipelineStage.PARAMETER_EXTRACTION,
        "command_name": LIST,
        "command": command,
        "stored_parameters": stored,
        "stored_parameter_round": 1,
    }
    return workflow, app_workflow


@pytest.fixture
def no_prediction():
    predictor = MagicMock()
    predictor.predict.return_value = CommandNamePrediction.Output()
    with patch.object(wildcard, "CommandNamePrediction", return_value=predictor):
        yield


@pytest.fixture
def recorded(no_prediction):
    """Record what the extractor is given and what is stored when it runs."""
    calls = []

    class _Recording(ParameterExtraction):
        def extract(self):
            calls.append({
                "text": self.command,
                "stored": self.cme_workflow.context.get("stored_parameters"),
            })
            return ParameterExtraction.Output(parameters_are_valid=False, error_msg="x")

    with patch.object(wildcard, "ParameterExtraction", _Recording):
        yield calls


def test_new_full_command_in_a_pending_round_is_a_fresh_extraction(recorded):
    stale = _ListParams.model_construct(scope="other", scope_uid=NOT_FOUND)
    workflow, _ = _pending_round(stale)
    new = "list_books scope=shelf scope_uid=3c85"
    wildcard.ResponseGenerator()(workflow, new)

    assert recorded == [{"text": f"{LIST}: {new}", "stored": None}]
    assert workflow.context["command"] == new
    assert "stored_parameter_round" not in workflow.context


def test_answer_in_a_pending_round_merges_the_new_text(recorded):
    stale = _ListParams.model_construct(scope="shelf", scope_uid=NOT_FOUND)
    workflow, _ = _pending_round(stale)
    wildcard.ResponseGenerator()(workflow, "3c85")

    assert recorded == [{"text": f"{LIST}: 3c85", "stored": stale}]
    assert workflow.context["command"] == LIST  # the pending command is kept


def test_assistant_mode_retry_round_binds_the_answer(no_prediction, harness):
    # A real extractor in assistant mode (no run_as_agent): the user's answer to
    # the missing-parameter prompt fills the missing field. It used to merge the
    # stored command text instead (": list_books" landed in scope_uid).
    harness.use(_ListParams)
    stored = _ListParams.model_construct(scope="shelf", scope_uid=NOT_FOUND)
    workflow, app_workflow = _pending_round(stored)
    app_workflow.context = {}
    output = wildcard.ResponseGenerator()(workflow, "3c85")

    params = output.command_response.artifacts["cmd_parameters"]
    assert (params.scope, params.scope_uid) == ("shelf", "3c85")
    assert output.command_response.artifacts["command"] == f"{LIST}: {LIST}"


# ---------------------------------------------------------------------------
# fix-6llp.1: the bare form -- nothing after the command name, every field defaulted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [f"{FIND}: find_book", f"{FIND}: {FIND}", f"{FIND}:"])
def test_a_bare_command_binds_its_defaults_without_the_llm(harness, text):
    output, diagnostics, _ = harness(_FindParams, text)
    assert output.parameters_are_valid
    assert (output.cmd_parameters.query, output.cmd_parameters.start) == (None, None)
    assert diagnostics["extraction_method"] == "xml_regex"
    assert diagnostics["deterministic_form"] == "bare"


def test_a_bare_command_with_a_required_field_takes_the_existing_path(harness):
    output, diagnostics, _ = harness(
        _OpenParams, f"{OPEN}: open_book", command_name=OPEN,
        llm_result=lambda cls: cls.model_construct(book_uid="from llm"))
    assert diagnostics["extraction_method"] == "llm"
    assert "deterministic_form" not in diagnostics
    assert output.cmd_parameters.book_uid == "from llm"


def test_text_after_the_name_is_not_bare():
    assert ParameterExtraction._extract_parameters_from_bare(
        "find_book everything", _FindParams, FIND) is None
    assert ParameterExtraction._extract_parameters_from_bare(
        "other_command", _FindParams, FIND) is None
