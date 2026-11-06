"""A sub-task draft's inputs from the shared parameter extractor (fix-48np, stage 1).

``draft_round1_proposal`` fills each sub-task's inputs with the values the shared extractor gives on the
sub-task's text (``parameter_extraction.extract_arguments``: the execution commands' LLM extractor and
its checks, and for a planning command ``check_arguments``, the checks ``typed_planning_call`` runs). A
value that a declared producer (``from_``) an earlier sub-task runs is ``from sub-task N``, not extracted.
Stage 1 was display only; since fix-48np.1 the drafted inputs are run (``resolve_inputs``), and a planning
command whose required value the extractor leaves missing gets the typed fill fallback in the draft too.

Gate 1a of the issue: the shared extractor runs outside the CME workflow, with no CME workflow object at
all, and leaves no state behind on the app workflow.

Integration: real workflows (``tests/planning_tutorial_workflow``, a copy of the library example with its
trained heads), the real routing, the real DSPy extractor and the real checks. The model boundary is a
scripted ``DummyLM`` that records each request. No test calls a model.
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import review_proposals
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments
from fastworkflow.review_proposals import Means, SubTask, SubTaskInput, draft_inputs, resolve_inputs, run_text
from fastworkflow.run_listing import has_trained_heads
from fastworkflow.utils import signatures
from tests.test_approval_binding import annotated_library  # noqa: F401 - the library copy with its trained heads

TUTORIAL = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())
NOTE_START = "(A first extraction from this command was refused:"
EMMA = "who wrote Emma and how many copies do we have"    # heads: book_by_title in the top-k
JOKE = "tell Ann a joke about cats"                        # heads: no command fits


class _ScriptedLM(DummyLM):
    """Answers the extractor's calls in order and records the text of each request."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class _PlannerLM(DummyLM):
    """The planner's answers (the typed fill fallback), in the planner adapter's own format; records each request."""

    def __init__(self, answers):
        super().__init__(answers)
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def _framework(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    # Jev's "is this field given" check is a model call: off, so the scripted LM answers alone.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    # Drafting is on by default; set explicitly so a developer's environment cannot turn it off here.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_DRAFT_ARGUMENTS", "on")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {k: v for k, v in fastworkflow._env_vars.items() if k != "JEV_API_KEY"})
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def scripted(monkeypatch):
    """Make the extractor answer *answers* in order; returns the LM (its ``prompts`` are the requests)."""
    def script(*answers: dict) -> _ScriptedLM:
        lm = _ScriptedLM(list(answers))
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        return lm
    return script


def _workflow(folder: str) -> fastworkflow.Workflow:
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    return fastworkflow.Workflow.create(folder, workflow_id_str=f"draft_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def library(annotated_library):  # noqa: F811
    from fastworkflow.examples.library.application.library import Library

    wf = _workflow(str(annotated_library))
    root = Library()
    wf.root_command_context = root
    wf.current_command_context = root
    return wf


@pytest.fixture
def library_heads(library):
    """The library copy, for a test that needs its trained heads to pick each sub-task's means."""
    if not has_trained_heads(library):
        pytest.skip("the library example's trained heads are not on this machine (fastworkflow train it first)")
    return library


# ---------------------------------------------------------------- gate 1a: outside the CME workflow


def test_the_shared_extractor_fills_a_planning_command_with_no_cme_workflow(scripted):
    wf = _workflow(TUTORIAL)
    before = copy.deepcopy(dict(wf.context))
    lm = scripted({"reasoning": "the title", "title": "Dune"})

    result = extract_arguments(wf, "book_by_title", "who wrote Dune")

    assert result.valid and result.refusal is None and not result.retried
    assert result.params.title == "Dune"
    assert len(lm.prompts) == 1 and "who wrote Dune" in lm.prompts[0]
    assert dict(wf.context) == before          # nothing stored on the app workflow (no stored_parameters, no stage)


def test_a_planning_value_not_in_the_request_is_refused_by_check_arguments_retried_once_and_refused(scripted):
    wf = _workflow(TUTORIAL)
    lm = scripted({"reasoning": "a guess", "title": "Dune Messiah"},
                  {"reasoning": "a guess again", "title": "Dune Messiah"})

    result = extract_arguments(wf, "book_loans", "who has Foundation on loan")

    assert result.retried and not result.valid
    assert result.refusal == "title='Dune Messiah' is not written in the request"
    assert NOTE_START not in lm.prompts[0]
    # the retry reads the refusal after the text; the checks read the text alone (so the note's own words never pass)
    assert NOTE_START in lm.prompts[1] and "title='Dune Messiah' is not written in the request" in lm.prompts[1]


def test_the_retry_can_fix_a_refused_planning_value(scripted):
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "a guess", "title": "Dune Messiah"}, {"reasoning": "copied", "title": "Foundation"})

    result = extract_arguments(wf, "book_loans", "who has Foundation on loan")

    assert result.retried and result.valid and result.params.title == "Foundation"


def test_a_missing_required_planning_value_is_named(scripted):
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "no author", "author": "NOT_FOUND"}, {"reasoning": "still none", "author": "NOT_FOUND"})

    result = extract_arguments(wf, "author_books", "show every book by this author in the library")

    assert not result.valid and result.missing == ["author"]
    assert result.refusal == "missing required argument author"


def test_an_execution_command_gets_the_execution_checks(scripted):
    """The same entry point on an execution command: argument_checks refuses an invented value and retries."""
    wf = _workflow(TUTORIAL)
    lm = scripted({"reasoning": "name", "query": "Bob Jones"}, {"reasoning": "copied", "query": "Ann Lee"})

    result = extract_arguments(wf, "find_member", "find the member Ann Lee")

    assert result.valid and result.params.query == "Ann Lee"
    assert len(lm.prompts) == 2 and NOTE_START in lm.prompts[1]


def test_two_extractions_in_a_row_share_no_state(scripted):
    """No hidden state between calls: the second command's extraction sees only its own text and answer."""
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "t", "title": "Dune"}, {"reasoning": "a", "author": "Octavia Butler"})

    first = extract_arguments(wf, "book_by_title", "who wrote Dune")
    second = extract_arguments(wf, "author_books", "what titles did Octavia Butler write")

    assert first.params.title == "Dune" and second.params.author == "Octavia Butler"
    assert not hasattr(second.params, "title")


# ---------------------------------------------------------------- the draft (stage 1)


def test_the_draft_shows_the_extracted_planning_arguments(library_heads, scripted):
    lm = scripted({"reasoning": "the title", "title": "Emma"})

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    first = draft.items[0]
    assert first.means == Means(kind="planning", name="Library/book_by_title")
    assert first.inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]
    assert first.check == ""
    assert "inputs: title=Emma" in draft.rendering()
    assert EMMA in lm.prompts[0]                  # each sub-task's means is extracted from its own text


def test_the_draft_shows_a_refusal_and_no_values(library_heads, scripted):
    scripted({"reasoning": "a guess", "title": "Persuasion"}, {"reasoning": "again", "title": "Persuasion"})
    # then the typed fill fallback on the planner (fix-48np.1) gives a title the request does not write either
    planner = _PlannerLM([{"reasoning": "typed fill", "title": "Persuasion", "steps_before": [], "steps_after": []}])

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE], planner_lm=planner)

    assert draft.items[0].inputs == []
    assert draft.items[0].check == "title='Persuasion' is not written in the request"
    assert EMMA in planner.prompts[0]                    # the typed fill read the sub-task's text


def test_with_the_switch_off_the_draft_makes_no_extraction(library_heads, scripted, monkeypatch):
    lm = scripted()
    monkeypatch.setitem(fastworkflow._env_vars, review_proposals.DRAFT_ARGUMENTS_ENV_VAR, "off")

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    assert lm.prompts == [] and draft.items[0].inputs == []
    assert draft.items[0].check                    # the empty-argument check, as before


def test_drafting_is_on_by_default(library_heads, scripted, monkeypatch):
    """On unless switched off (fix-48np.1, after the round-2 measurement): with the variable unset, the draft extracts."""
    lm = scripted({"reasoning": "the title", "title": "Emma"}, *[{"reasoning": "the member", "member": "Ann"}] * 3)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {k: v for k, v in fastworkflow._env_vars.items()
                         if k != review_proposals.DRAFT_ARGUMENTS_ENV_VAR})
    monkeypatch.delenv(review_proposals.DRAFT_ARGUMENTS_ENV_VAR, raising=False)

    assert review_proposals.drafting_on() is True
    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    assert lm.prompts and draft.items[0].inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "OFF"])
def test_drafting_is_off_only_when_switched_off(monkeypatch, value):
    monkeypatch.setitem(fastworkflow._env_vars, review_proposals.DRAFT_ARGUMENTS_ENV_VAR, value)
    assert review_proposals.drafting_on() is False


def test_a_value_an_earlier_sub_task_produces_is_marked_from_that_sub_task(library, scripted):
    """open_book's book_uid is declared from find_book: after a find_book sub-task it is 'from sub-task 1', not guessed."""
    scripted({"reasoning": "no uid in the text", "book_uid": "NOT_FOUND"})
    earlier = [SubTask(goal="find the book Dune", means=Means(kind="plain", name="Library/find_book"))]
    item = SubTask(goal="open it", means=Means(kind="plain", name="Library/open_book"))

    drafted = draft_inputs(library, item, "open it", earlier)

    assert drafted.refusal is None
    assert list(drafted.inputs) == [SubTaskInput(name="book_uid", from_subtask=1, drafted=True)]


def test_with_no_producer_before_it_the_value_is_not_marked(library, scripted):
    scripted({"reasoning": "no uid in the text", "book_uid": "NOT_FOUND"})
    item = SubTask(goal="open it", means=Means(kind="plain", name="Library/open_book"))

    drafted = draft_inputs(library, item, "open it", [])

    assert drafted.inputs == () and drafted.refusal == "missing required argument book_uid"


def test_drafted_inputs_are_run_as_the_review_shows_them():
    """Stage 1 showed drafted inputs and did not run them; fix-48np.1 runs them: a drafted value is resolved as a given
    one, and a drafted 'from sub-task N' takes N's output once N has run."""
    item = SubTask(goal="who wrote Emma", means=Means(kind="planning", name="Library/book_by_title"),
                   inputs=[SubTaskInput(name="title", value="Emma", drafted=True),
                           SubTaskInput(name="n", from_subtask=1, drafted=True)])

    before = resolve_inputs(item, {})
    assert before.named == {"title": "Emma"} and before.missing == (1,)

    after = resolve_inputs(item, {1: "3"})
    assert after.named == {"title": "Emma", "n": "3"} and after.missing == ()
    assert run_text(item, after) == "who wrote Emma (inputs: title=Emma, n=3)"


def test_a_drafted_input_round_trips_through_the_review_json():
    item = SubTask(goal="g", inputs=[SubTaskInput(name="title", value="Emma", drafted=True)])
    proposal = review_proposals.SubTaskProposal(items=[item])
    assert review_proposals.SubTaskProposal.from_text(proposal.model_dump_json()) == proposal


# ---------------------------------------------------------------- the copy-it-exactly sentence (fix-48np)


def _signature_text(folder: str, name: str) -> str:
    from fastworkflow import ModuleType, planning_commands
    from fastworkflow.utils.signatures import InputForParamExtraction, planning_verbatim_sentences

    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    model = fastworkflow.RoutingRegistry.get_definition(folder).get_command_class(
        name, ModuleType.COMMAND_PARAMETERS_CLASS)
    sentences = planning_verbatim_sentences(model, name, folder, planning_commands.option_sentences(model))
    signature = InputForParamExtraction.create_signature_from_pydantic_model(model, sentences)
    return "\n".join(f"{n}: {f.json_schema_extra.get('desc', '')}" for n, f in signature.output_fields.items())


def test_a_planning_commands_str_field_is_copied_exactly_as_the_typed_call_was_told(annotated_library):  # noqa: F811
    from fastworkflow.planning_commands import IN_REQUEST_STR_SENTENCE

    text = _signature_text(str(annotated_library), "Library/note_member")
    # text carries no in_request option: the extractor is told to copy it all the same
    assert text.count(IN_REQUEST_STR_SENTENCE.rstrip(".")) == 2           # member (in_request) and text, once each


def test_an_execution_commands_fields_are_told_nothing_new():
    from fastworkflow.planning_commands import IN_REQUEST_STR_SENTENCE

    assert IN_REQUEST_STR_SENTENCE.rstrip(".") not in _signature_text(TUTORIAL, "find_member")


def test_a_failed_draft_extraction_is_logged_and_shows_no_inputs(library, monkeypatch, caplog):
    def refuse(self, *args, **kwargs):
        raise AssertionError("the parameter-extraction model was called")

    monkeypatch.setattr(signatures.InputForParamExtraction, "extract_parameters", refuse)
    item = SubTask(goal="who wrote Emma", means=Means(kind="planning", name="Library/book_by_title"))

    with caplog.at_level("WARNING"):
        assert draft_inputs(library, item, "who wrote Emma", []) is None
    assert any("could not draft the inputs of Library/book_by_title (AssertionError" in r.getMessage()
               for r in caplog.records)
