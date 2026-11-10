"""The Input's model validator is the one author hook (spec 3.1-3.4). Real components, no mocks."""
import pytest

import fastworkflow
from pydantic import BaseModel, Field, model_validator

from fastworkflow import planning_commands as pc
from fastworkflow.utils.signatures import InputForParamExtraction, removed_validate_hook_message
from tests.test_planning_command_files import _write as write_files


def _extraction(command="show Ann"):
    return InputForParamExtraction(command=command, input_for_param_extraction_class=None)


def _command(tmp_path, name, text):
    folder = write_files(tmp_path, {name: text})
    from fastworkflow import workflow_agent
    return workflow_agent.load_planning_command_map(folder)[name]


PLANNING = '''"""{name}: show one record, for the test."""
from pydantic import BaseModel, Field, model_validator

import pytest

import fastworkflow


class Signature:
    """Show one record."""

    template = ["say the record"]

    class Input(BaseModel):
        name: str = Field(description="the record")
        note: str = Field(default="", description="a note")

{body}


class PlanGenerator:
    def steps(self, input):
        return ["say the record"]
'''


def test_a_planning_command_with_the_removed_hook_fails_the_load_naming_the_validator(tmp_path):
    text = PLANNING.format(name="old", body='''
    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        return None''')
    folder = write_files(tmp_path, {"old": text})
    from fastworkflow import command_preview
    problems = command_preview.load_problems(str(folder))
    assert problems and "validate_extracted_parameters is removed" in problems[0]
    assert '@model_validator(mode="after") on Input' in problems[0]
    assert 'info.context["command_text"]' in problems[0]


def test_an_execution_signature_with_the_removed_hook_is_a_load_error():
    class Signature:
        class Input(BaseModel):
            name: str = Field(description="x")

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            return None

    message = removed_validate_hook_message(Signature, "_commands/x.py: x")
    assert message == ('_commands/x.py: x: validate_extracted_parameters is removed: write '
                       '@model_validator(mode="after") on Input (read info.context["command_text"], '
                       '["workflow"], ["command"])')
    assert removed_validate_hook_message(type("S", (), {}), "w") is None


def test_an_execution_command_gets_command_text_workflow_and_command_in_the_validation_context():
    seen = {}

    class Params(BaseModel):
        name: str = Field(description="the record")

        @model_validator(mode="after")
        def look(self, info):
            seen.update(info.context)
            return self

    is_valid, *_ = InputForParamExtraction(command="show Ann", input_for_param_extraction_class=None
                                           ).validate_parameters(None, "Library/show", Params.model_construct(name="Ann"))
    assert is_valid
    assert seen == {"command_text": "show Ann", "workflow": None, "command": "Library/show"}



def test_refused_with_a_field_from_a_validator_keeps_its_fields(tmp_path):
    command = _command(tmp_path, "pick", PLANNING.format(name="pick", body='''
        @model_validator(mode="after")
        def closed(self, info):
            if self.name == "attic":
                raise fastworkflow.Refused("the attic is closed", field="name")
            return self
'''))
    try:
        pc.check_input(command, {"name": "attic", "note": ""}, "pick the attic", None)
    except pc.PlanningCommandInputError as refusal:
        assert refusal.fields == ("name",)
        assert "the attic is closed" in str(refusal)
    else:
        raise AssertionError("expected a refusal")


def test_refused_without_a_field_refuses_the_call_with_no_field(tmp_path):
    command = _command(tmp_path, "nofit", PLANNING.format(name="nofit", body='''
        @model_validator(mode="after")
        def nofit(self, info):
            raise fastworkflow.Refused("this command does not fit the request")
'''))
    try:
        pc.check_input(command, {"name": "x", "note": ""}, "x", None)
    except pc.PlanningCommandInputError as refusal:
        assert refusal.fields == ()
        assert str(refusal) == "validate_extracted_parameters: this command does not fit the request"
    else:
        raise AssertionError("expected a refusal")


def test_a_plain_value_error_is_a_refusal_with_no_field(tmp_path):
    command = _command(tmp_path, "plain", PLANNING.format(name="plain", body='''
        @model_validator(mode="after")
        def plain(self, info):
            raise ValueError("name and note are one record")
'''))
    # a plain ValueError stays a refusal with no field (spec 3.3), in the hook's words
    try:
        pc.check_input(command, {"name": "x", "note": ""}, "x", None)
    except pc.PlanningCommandInputError as refusal:
        assert refusal.fields == ()
        assert str(refusal) == "validate_extracted_parameters: name and note are one record"
    else:
        raise AssertionError("expected a refusal")


def test_a_correction_is_written_back_with_a_note_attributed_to_the_validator(tmp_path):
    command = _command(tmp_path, "fix", PLANNING.format(name="fix", body='''
        @model_validator(mode="after")
        def upper(self, info):
            self.name = self.name.upper()
            return self
'''))
    notes: list[str] = []
    args = pc._validate_arguments(command, {"name": "ann", "note": ""}, "show ann", None, notes)
    assert args["name"] == "ANN"
    assert notes == ["name='ann' -> 'ANN' (validate_extracted_parameters)"]


def test_a_placeholder_combination_a_validator_refuses_is_left_out_of_the_template_and_preview(tmp_path):
    body = '''
        @model_validator(mode="after")
        def no_placeholder(self, info):
            if self.name == "x":
                raise fastworkflow.Refused("x is refused")
            return self
'''
    command = _command(tmp_path, "grid", PLANNING.format(name="grid", body=body).replace(
        'template = ["say the record"]', ''))
    from fastworkflow import command_preview
    grid = pc._template_grid(command)
    assert all(env["name"] != "x" for env in grid)
    assert command_preview._validated(command, {"name": "x", "note": ""})[0] is False
    assert command_preview._validated(command, {"name": "y", "note": ""})[0] is True


def test_diagnostics_still_record_the_validation_step(tmp_path):
    class Params(BaseModel):
        name: str = Field(description="the record")

        @model_validator(mode="after")
        def fix(self, info):
            self.name = self.name.upper()
            return self

    diagnostics: dict = {}
    is_valid, *_ = InputForParamExtraction(command="show ann", input_for_param_extraction_class=None
                                           ).validate_parameters(None, "Library/show",
                                                                 Params.model_construct(name="ann"),
                                                                 diagnostics=diagnostics)
    assert is_valid
    record = diagnostics["validation_hook"]
    assert record["ran"] is True and record["is_valid"] is True and record["message"] is None
    assert record["changed_fields"] == ["name"] and record["refused_fields"] == []


def test_an_execution_command_with_the_removed_hook_fails_load_problems_naming_the_validator(tmp_path):
    import shutil
    from pathlib import Path

    from fastworkflow import command_preview

    source = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "hello_world"
    folder = tmp_path / "hello_world"
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    command_file = folder / "_commands" / "add_two_numbers.py"
    text = command_file.read_text(encoding="utf-8")
    text = text.replace('''    plain_utterances = [''', '''    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters):
        return None

    plain_utterances = [''', 1)
    command_file.write_text(text, encoding="utf-8")

    problems = command_preview.load_problems(str(folder))
    assert any("validate_extracted_parameters is removed" in p and "add_two_numbers" in p for p in problems), problems


def test_an_input_validator_that_reads_the_context_names_it_when_built_without_one():
    """A command's Input loaded from a real workflow: direct construction with no validation context gets
    a message naming the validator and the fix (not 'NoneType' object is not subscriptable); with a context
    the validator runs unchanged, and the guard is applied once however often the command is loaded."""
    import os
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.utils.signatures import ModuleType

    folder = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "simple_workflow_template")
    routing = RoutingRegistry.get_definition(folder)
    signature = routing.get_command_class("WorkItem/go_to_workitem", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    Input = signature.Input

    with pytest.raises(ValueError) as raised:
        Input(path="/Feature[index=0]")
    message = str(raised.value)
    assert "Input.path_from_workitem_type needs the validation context" in message
    assert "Signature.Input.model_construct(...)" in message

    checked = Input.model_validate({"path": "/Feature[index=0]"}, context={"workflow": None})
    assert checked.path == "/Feature[index=0]"
    assert Input.model_construct(path="/Feature[index=0]").path == "/Feature[index=0]"

    # a second load of the same command does not wrap the validator again
    again = routing.get_command_class("WorkItem/go_to_workitem", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS).Input
    from fastworkflow.utils.input_validation import _CONTEXT_GUARD
    dec = again.__pydantic_decorators__.model_validators["path_from_workitem_type"]
    assert getattr(dec.func, _CONTEXT_GUARD, False)
    assert not getattr(dec.func.__wrapped__, _CONTEXT_GUARD, False)


# ----------------------------------------------------------- the validation context (fix-o8j3, "request" removed)

REMOVED = "'request' is removed from the validation context: read info.context['command_text']"


def _reading_validator(read):
    return f'''
        @model_validator(mode="after")
        def look(self, info):
            {read}
            return self'''


def test_a_validator_reading_the_removed_request_key_fails_through_check_input_naming_command_text(tmp_path):
    command = _command(tmp_path, "reads_request", PLANNING.format(
        name="reads_request", body=_reading_validator('self.note = info.context["request"]')))
    with pytest.raises(pc.PlanningCommandCallError) as raised:
        pc.check_input(command, {"name": "Ann", "note": ""}, "show Ann", None)
    assert raised.value.raised == "KeyError"
    assert str(raised.value) == f"validate_extracted_parameters raised KeyError: \"{REMOVED}\""


def test_a_validator_calling_get_on_the_removed_request_key_fails_the_same_way(tmp_path):
    command = _command(tmp_path, "gets_request", PLANNING.format(
        name="gets_request", body=_reading_validator('self.note = info.context.get("request")')))
    with pytest.raises(pc.PlanningCommandCallError) as raised:
        pc.check_input(command, {"name": "Ann", "note": ""}, "show Ann", None)
    assert raised.value.raised == "KeyError"
    assert REMOVED in str(raised.value)


def test_the_three_real_keys_still_reach_the_validators_through_check_input(tmp_path):
    command = _command(tmp_path, "reads_keys", PLANNING.format(name="reads_keys", body=_reading_validator(
        'self.note = "|".join([info.context["command_text"], str(info.context["workflow"]), '
        'info.context["command"]])')))
    written = pc.check_input(command, {"name": "Ann", "note": ""}, "show Ann", "Library")
    assert written == {"note": "show Ann|Library|reads_keys"}


def test_a_different_missing_key_is_a_plain_key_error(tmp_path):
    command = _command(tmp_path, "reads_other", PLANNING.format(
        name="reads_other", body=_reading_validator('self.note = info.context["nope"]')))
    with pytest.raises(pc.PlanningCommandCallError) as raised:
        pc.check_input(command, {"name": "Ann", "note": ""}, "show Ann", None)
    assert raised.value.raised == "KeyError"
    assert str(raised.value) == "validate_extracted_parameters raised KeyError: 'nope'"


def test_validation_context_is_a_dict_except_for_the_removed_key():
    from fastworkflow.utils.input_validation import ValidationContext
    import copy
    import pickle

    context = ValidationContext({"command_text": "show Ann", "workflow": None, "command": "Library/show"})
    assert isinstance(context, dict)
    assert context == {"command_text": "show Ann", "workflow": None, "command": "Library/show"}
    assert list(context) == ["command_text", "workflow", "command"]
    assert context.get("command") == "Library/show" and context.get("nope") is None
    assert copy.copy(context) == context and pickle.loads(pickle.dumps(context)) == context
    with pytest.raises(KeyError) as missing:
        context["nope"]
    assert missing.value.args == ("nope",)
    with pytest.raises(KeyError) as removed:
        context["request"]
    assert removed.value.args == (REMOVED,)
    with pytest.raises(KeyError, match="removed from the validation context"):
        context.get("request")


def test_a_refusal_with_no_field_is_put_in_the_hook_words_and_does_not_ask_the_user(tmp_path):
    """The replan case: a validator's Refused naming no input is "validate_extracted_parameters: <message>"
    (HEAD's hook words), so it is not read as an argument refusal ("name=... is ...") and no question names an
    argument; the planner plans the call instead (refusal_names_an_input is False)."""
    command = _command(tmp_path, "record_fact", PLANNING.format(name="record_fact", body='''
        @model_validator(mode="after")
        def owner_fits_kind(self, info):
            if self.name == "GitLab":
                raise fastworkflow.Refused("fact='owner' does not fit kind='person': owner is read only with kind='application'")
            return self
'''))
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_input(command, {"name": "GitLab", "note": ""}, "record owner of GitLab", None)
    text = str(refused.value)
    assert text == ("validate_extracted_parameters: fact='owner' does not fit kind='person': "
                    "owner is read only with kind='application'")
    assert refused.value.fields == ()
    assert pc.refusal_names_an_input(command, refused.value) is False
    assert pc.refusal_question(command, refused.value) == (
        f"I can't run {command.name.replace('_', ' ')} yet: fact='owner' does not fit kind='person': "
        "owner is read only with kind='application'. What should I change?")


def test_a_write_a_validator_makes_before_refusing_is_kept_on_the_execution_path():
    """HEAD's hook changed the caller's parameters in place, so a write before a refusal stayed, and the
    diagnostics named it in changed_fields."""
    class Rewrites(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"copied_from_command_text": True})

        @model_validator(mode="after")
        def upper_then_refuse(self, info):
            self.name = self.name.upper()
            if self.name == "REFUSE":
                raise fastworkflow.Refused("refused after the rewrite")
            return self

    instance = Rewrites.model_construct(name="refuse")
    diagnostics: dict = {}
    is_valid, message, *_ = _extraction("Show refuse").validate_parameters(
        None, "Library/show_group", instance, diagnostics=diagnostics)
    assert not is_valid
    assert message == "refused after the rewrite"
    assert instance.name == "REFUSE"
    assert diagnostics["validation_hook"]["changed_fields"] == ["name"]


class Signature:
    """A command's Signature shape, so its Input is recognised as a command Input (qualname Signature.Input)."""

    class Input(BaseModel):
        uid: str
        size: int = 1

        @model_validator(mode="after")
        def reads_the_command_text(self, info):
            assert "show" in info.context["command_text"]
            return self


class StoredFacts(BaseModel):
    """A phase's facts model holding a command Input, as a planning command's phase output does."""
    called_with: Signature.Input
    count: int
    earlier: list[Signature.Input] = []


def test_a_stored_phase_output_holding_a_command_input_is_rebuilt_without_its_validators():
    """Regression (pass 8 benchmark): a phase's stored output was re-validated with no context, so the
    Input's validator read info.context["command_text"] on None and the whole turn failed."""
    from fastworkflow.typed_phase_facts import validate_phase_output_value

    stored = {"called_with": {"uid": "u1", "size": "2"}, "count": "3", "earlier": [{"uid": "u0"}]}
    facts = validate_phase_output_value(StoredFacts, stored)
    assert isinstance(facts.called_with, Signature.Input)
    assert (facts.called_with.uid, facts.called_with.size, facts.count) == ("u1", 2, 3)
    assert [e.uid for e in facts.earlier] == ["u0"]


def test_a_stored_command_input_still_has_its_fields_checked():
    from pydantic import ValidationError

    from fastworkflow.utils.input_validation import stored_value

    with pytest.raises(ValidationError):
        stored_value(StoredFacts, {"called_with": {"uid": "u1", "size": "big"}, "count": 1})
    with pytest.raises(ValidationError):
        stored_value(StoredFacts, {"called_with": {"uid": "u1"}, "count": "many"})


def test_a_stored_model_holding_no_command_input_is_validated_as_before():
    from fastworkflow.utils.input_validation import stored_value

    class Plain(BaseModel):
        n: int

        @model_validator(mode="after")
        def doubles(self):
            self.n = self.n * 2
            return self

    assert stored_value(Plain, {"n": "2"}).n == 4
