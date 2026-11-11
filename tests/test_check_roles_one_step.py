"""The roles of two inputs given one value, in the one validation step (fix-3rsq.check-roles-one-step).

``validate_parameters`` (the turn-frame path) now runs ``planning_commands._check_roles`` for a planning
command, on every required ``str`` input, as the typed planning call does: one rule on both paths, with or
without an ``in_request`` option. An execution command is not affected.

Integration only: a temporary copy of ``planning_tutorial_workflow`` with three commands added (two
planning commands, one execution command), the real loader, ``check_arguments`` and ``validate_parameters``.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
import fastworkflow
from fastworkflow import planning_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.utils.signatures import InputForParamExtraction

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: Two name inputs (``in_request``): the same value for both is refused on the turn-frame path.
NAME_PAIR = '''
from pydantic import BaseModel, Field


class Signature:
    """Move one member's loan of a book to a branch."""

    class Input(BaseModel):
        member: str = Field(description="the member whose loan moves", json_schema_extra={"copied_from_command_text": True})
        branch: str = Field(description="the branch the loan moves to", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"move the loan to the {input.branch} branch"]
'''

#: One name input and one plain ``str`` input (no ``in_request``): still checked for roles.
NAME_AND_FREE_TEXT = '''
from pydantic import BaseModel, Field


class Signature:
    """File a note about one member."""

    class Input(BaseModel):
        member: str = Field(description="the member the note is about", json_schema_extra={"copied_from_command_text": True})
        title: str = Field(description="a title for the note")


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"file the note titled {input.title}"]
'''

#: An execution command with two required ``str`` inputs: never checked for roles.
PAIR_NAMES = '''
from pydantic import BaseModel, Field


class Signature:
    """Pair two names."""

    class Input(BaseModel):
        first: str = Field(description="a name")
        second: str = Field(description="a name")

    class Output(BaseModel):
        pass


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return Signature.Output()
'''

REQUEST = "Move the loan of member Ann Lee to branch Ann Lee"


@pytest.fixture
def workflow(tmp_path):
    """The tutorial copy with the three commands, loaded for ``validate_parameters``."""
    folder = tmp_path / "roles_wf"
    shutil.copytree(TUTORIAL, folder,
                    ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (folder / "_commands" / "name_pair.py").write_text(NAME_PAIR, encoding="utf-8")
    (folder / "_commands" / "name_and_free_text.py").write_text(NAME_AND_FREE_TEXT, encoding="utf-8")
    (folder / "_commands" / "pair_names.py").write_text(PAIR_NAMES, encoding="utf-8")
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
    app_workflow = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"roles_{uuid.uuid4().hex[:8]}")
    yield app_workflow, str(folder)
    fastworkflow.RoutingRegistry.clear_registry()


def _validate(app_workflow, command_name: str, request: str, **values):
    """``validate_parameters`` for *command_name* with the given values, as the turn-frame path calls it."""
    planning = workflow_agent.load_planning_command_map(app_workflow.folderpath).get(command_name)
    if planning is not None:
        model = planning.input_model
    else:
        definition = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
        response_generator = definition.get_command_class(
            command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        model = sys.modules[response_generator.__module__].Signature.Input
    extraction = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    return extraction.validate_parameters(app_workflow, command_name, model(**values))


def test_two_name_inputs_with_one_value_are_refused_on_the_turn_frame_path(workflow):
    app_workflow, _ = workflow

    is_valid, message, _, _ = _validate(app_workflow, "name_pair", REQUEST, member="Ann Lee", branch="Ann Lee")

    assert not is_valid
    assert message == "branch='Ann Lee' is also the value of member"


def test_the_turn_frame_refusal_matches_the_typed_call(workflow):
    app_workflow, folder = workflow
    command = workflow_agent.load_planning_command_map(folder)["name_pair"]
    args, _ = sc.check_arguments(command, {"member": "Ann Lee", "branch": "Ann Lee"}, REQUEST)

    with pytest.raises(sc.PlanningCommandInputError) as typed:
        sc._validate_arguments(command, args, REQUEST, None)

    is_valid, message, _, _ = _validate(app_workflow, "name_pair", REQUEST, member="Ann Lee", branch="Ann Lee")
    assert not is_valid
    assert message == str(typed.value)


def test_a_plain_str_input_equal_to_a_name_is_refused_as_the_typed_call_refuses_it(workflow):
    app_workflow, folder = workflow
    request = "File a note about member Ann Lee titled Ann Lee"
    command = workflow_agent.load_planning_command_map(folder)["name_and_free_text"]
    args, _ = sc.check_arguments(command, {"member": "Ann Lee", "title": "Ann Lee"}, request)

    with pytest.raises(sc.PlanningCommandInputError) as typed:
        sc._validate_arguments(command, args, request, None)

    is_valid, message, _, _ = _validate(app_workflow, "name_and_free_text", request,
                                        member="Ann Lee", title="Ann Lee")

    assert not is_valid
    assert message == str(typed.value)


def test_an_execution_command_with_two_equal_required_strings_is_not_refused(workflow):
    app_workflow, _ = workflow

    is_valid, message, _, _ = _validate(app_workflow, "pair_names", "Pair Ann Lee with Ann Lee",
                                        first="Ann Lee", second="Ann Lee")

    assert is_valid, message
