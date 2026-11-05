"""A refused skill call in an attended turn is put to the user, and the reply runs the call.

No stand-ins here: the skills are the tutorial workflow's own (``tests/planning_tutorial_workflow``),
the argument guards and the question are the real functions, and the tests that need the
planner's typed call run against the configured planner LLM. Those skip without a usable key,
as the other ``requires_llm_key`` tests do.
"""
from __future__ import annotations

import queue
from pathlib import Path

import dspy
import pytest
from dotenv import dotenv_values
from dspy.utils.exceptions import LMTransportError

import fastworkflow
from fastworkflow import planning_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.llm_credentials import syndata_llm_available
from tests.scripted_extractor import script_extractor
from tests.test_text_planner_and_split import RecordingTraceSink, _session, workflow_env  # noqa: F401 - workflow_env is a fixture

ROOT = Path(__file__).resolve().parent.parent
PLANNING_COMMANDS = str(Path(__file__).resolve().parent / "planning_tutorial_workflow")
EXAMPLES = ROOT / "fastworkflow" / "examples"
REQUEST = "Which books does that member have on loan?"
AUTHOR_REQUEST = "Which books did the author of that one write?"  # names no author, no proper noun
MEMBER_QUESTION = ("I can't run member loans yet: your request does not say which member to use. "
                   "Which member should I use?")


def _planning_command(name: str) -> sc.PlanningCommandCommand:
    return workflow_agent.load_planning_command_map(PLANNING_COMMANDS)[name]


def _raw(command: sc.PlanningCommandCommand, **values) -> dict:
    """The planner's raw values for every param of the skill: the given ones, None for the rest."""
    return {p.name: values.get(p.name) for p in command.params}


def _planner_settings() -> dict:
    """The model settings of a real run: the examples' files, then the repo's own env/ and
    passwords/ files override them (the resolution order test_personas uses)."""
    settings = {**dotenv_values(EXAMPLES / "fastworkflow.env"),
                **dotenv_values(EXAMPLES / "fastworkflow.passwords.env")}
    for name in ("env/.env", "passwords/.env"):
        settings.update(dotenv_values(ROOT / name))
    return {key: value for key, value in settings.items() if value is not None}


def _planner_llm_configured() -> bool:
    # The credential gate the other requires_llm_key tests use, applied to the planner role.
    settings = _planner_settings()
    return syndata_llm_available({"LLM_SYNDATA_GEN": settings.get("LLM_PLANNER"),
                                  "LITELLM_API_KEY_SYNDATA_GEN": settings.get("LITELLM_API_KEY_PLANNER")})


needs_planner_llm = pytest.mark.skipif(
    not _planner_llm_configured(),
    reason="No usable LLM credentials for LLM_PLANNER; cannot run the planner's typed call.")


# ---------------------------------------------------------------- no model: real guards, real question

@pytest.mark.parametrize("planning_command, values, asked_in, question", [
    ("member_loans", {"member": ""}, REQUEST, MEMBER_QUESTION),
    ("find_record", {"name": "Dune", "kind": "film"}, "Is Dune a film?",
     "I can't run find record yet: the kind must be one of: book, member. Which kind should I use?"),
    ("find_books", {"query": "Dune"}, "which books by Asimov",
     "I can't run find books yet: the query 'Dune' is not in your request. Which query should I use?"),
], ids=["missing", "not-one-of", "not-written"])
def test_a_refused_call_of_a_tutorial_planning_command_is_put_as_a_question_naming_its_input(planning_command, values, asked_in, question):
    command = _planning_command(planning_command)
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc.check_arguments(command, _raw(command, **values), asked_in)

    assert sc.refusal_question(command, refused.value) == question


def test_a_refusal_from_a_planning_commands_own_rule_is_put_in_its_own_words():
    command = _planning_command("member_loans")
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc._validate_arguments(command, {"member": "everyone"}, "Which books does everyone have on loan?", None)

    # The hook names the field it refuses (Refused(..., field="member")): the question asks for it.
    assert refused.value.fields == ("member",)
    assert sc.refusal_question(command, refused.value) == (
        "I can't run member loans yet: member is 'everyone', which names no one member; this skill "
        "reads one member's loans: give that member's name. What should the member be?")


def _refusal(command: sc.PlanningCommandCommand, values: dict, asked_in: str) -> sc.PlanningCommandInputError:
    """The real refusal a call of ``command`` with ``values`` gets: the guards, then the hook."""
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        args, _notes = sc.check_arguments(command, _raw(command, **values), asked_in)
        sc._validate_arguments(command, args, asked_in, None)
    return refused.value


#: A refusal naming no input, worded as the hook path words a ``Refused(message)`` without ``field``
#: (``_run_validate``), and as an ``Input`` rule is worded (``check_input``): the command does not
#: fit the request, which no answer from the user fixes.
NOT_THIS_JOB = sc.PlanningCommandInputError(
    "validate_extracted_parameters: fact='owner' does not fit kind='login': the owner of any other "
    "kind is not this job")
INPUT_RULE = sc.PlanningCommandInputError("the arguments break a rule of member_loans: the two names are one record")


@pytest.mark.parametrize("planning_command, values, asked_in", [
    ("member_loans", {"member": ""}, REQUEST),                                  # missing
    ("find_books", {"query": "Dune"}, "which books by Asimov"),                 # not written in the request
    ("member_loans", {"member": "everyone"}, "Which books does everyone have on loan?"),  # Refused(field=)
], ids=["missing", "not-written", "hook-names-the-field"])
def test_a_refusal_naming_an_input_is_one_to_ask(planning_command, values, asked_in):
    command = _planning_command(planning_command)

    assert sc.refusal_names_an_input(command, _refusal(command, values, asked_in)) is True


@pytest.mark.parametrize("refusal", [NOT_THIS_JOB, INPUT_RULE], ids=["hook-names-no-field", "input-rule"])
def test_a_refusal_naming_no_input_is_not_one_to_ask(refusal):
    assert sc.refusal_names_an_input(_planning_command("member_loans"), refusal) is False


def test_an_attended_turn_asks_about_a_refusal_naming_an_input(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)
    command = _planning_command("member_loans")

    question = workflow_agent.planning_command_refusal_to_ask(ctx, command, _refusal(command, {"member": ""}, REQUEST))

    assert question == MEMBER_QUESTION


@pytest.mark.parametrize("refusal", [NOT_THIS_JOB, INPUT_RULE], ids=["hook-names-no-field", "input-rule"])
def test_an_attended_turn_does_not_ask_about_a_refusal_naming_no_input(workflow_env, refusal):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)  # attended, nothing run, nothing asked: a question would be allowed

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is True
    assert workflow_agent.planning_command_refusal_to_ask(ctx, _planning_command("member_loans"), refusal) is None


def test_an_unattended_turn_does_not_ask_about_a_refusal_naming_an_input(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST, unattended=True)
    command = _planning_command("member_loans")

    assert workflow_agent.planning_command_refusal_to_ask(ctx, command, _refusal(command, {"member": ""}, REQUEST)) is None


def test_refused_planning_command_call_gets_the_refusal_question(request):
    request.getfixturevalue("workflow_env")()
    command = workflow_agent.load_planning_command_map(PLANNING_COMMANDS)["member_loans"]

    question = sc.missing_argument_question(command, "member")

    assert question == MEMBER_QUESTION
    assert question == sc.refusal_question(command, sc.PlanningCommandInputError("missing required argument member"))


def test_a_refusal_is_asked_in_an_attended_turn_that_has_not_asked_or_run(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is True


def test_a_refusal_is_not_asked_in_an_unattended_turn(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST, unattended=True)

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is False


def test_a_refusal_is_not_asked_when_a_message_queue_blocks_the_turn(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx.set_transport_queues(user_message_queue=queue.Queue())  # the CLI driver's wiring
    ctx._begin_turn(REQUEST)

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is False


def test_a_refusal_is_not_asked_once_the_turn_has_asked_one(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)
    ctx._planning_command_refusal_answered = True  # the reply to the first question has been run

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is False


def test_a_refusal_is_not_asked_once_a_command_has_run_in_the_turn(workflow_env):
    workflow_env()
    ctx, _wf = _session(RecordingTraceSink(), folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)
    ctx._turn_outputs.append(fastworkflow.CommandOutput(
        command_name="find_member", command_response=fastworkflow.CommandResponse(response="Ann Lee")))

    assert workflow_agent._planning_command_refusal_may_ask(ctx) is False


def test_the_refusal_flags_survive_a_save_and_restore(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, _wf = _session(sink, folder=PLANNING_COMMANDS)
    ctx._planning_command_refusal_pending = True
    ctx._planning_command_refusal_answered = True
    state = ctx.serialize_state(channel_id="refusal-flags")

    restored, _wf = _session(sink, folder=PLANNING_COMMANDS)
    restored.apply_serialized_state(state)

    assert restored._planning_command_refusal_pending is True
    assert restored._planning_command_refusal_answered is True


def test_a_save_from_before_the_refusal_flags_restores_them_as_unset(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, _wf = _session(sink, folder=PLANNING_COMMANDS)
    older = {key: value for key, value in ctx.serialize_state(channel_id="older-save").items()
             if key not in ("skill_refusal_pending", "skill_refusal_answered")}

    restored, _wf = _session(sink, folder=PLANNING_COMMANDS)
    restored._planning_command_refusal_pending = True
    restored._planning_command_refusal_answered = True
    restored.apply_serialized_state(older)

    assert restored._planning_command_refusal_pending is False
    assert restored._planning_command_refusal_answered is False


def test_a_failure_that_is_not_a_refusal_is_never_put_to_the_user(workflow_env, monkeypatch):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink, folder=PLANNING_COMMANDS)
    ctx._begin_turn(REQUEST)  # attended: a refusal would be asked
    # The planner's model is unreachable: the typed call fails with a connection error, which is
    # not a refusal of the arguments. The text planner uses the same model, so the turn raises.
    unreachable = dspy.LM("openai/unreachable-planner", api_base="http://127.0.0.1:9/v1",
                          api_key="unused", num_retries=0)
    # The extractor (fix-48np) finds no member in the request, twice: the typed fill fallback then asks the
    # planner's model, which is unreachable. That connection error is not turned into a refusal either.
    script_extractor(monkeypatch, {"member": "NOT_FOUND"}, {"member": "NOT_FOUND"})
    ctx.push_active_workflow(wf)
    try:
        with pytest.raises(LMTransportError):
            workflow_agent.build_query_with_next_steps(REQUEST, ctx, planner_lm=unreachable,
                                                       planning_command=_planning_command("member_loans"))
    finally:
        ctx.pop_active_workflow()

    assert ctx._planning_command_refusal is None
    assert ctx._planning_command_refusal_pending is False


# ---------------------------------------------------------------- planner LLM: the typed call is real

@pytest.mark.slow
@pytest.mark.requires_llm_key
@needs_planner_llm
@pytest.mark.xfail(strict=False, reason=(
    "Depends on the planner model's typed extraction: with 'that one' the planner refused the author "
    "in 2 of 5 fresh-cache runs. The second turn also needs the tutorial's trained intent models "
    "(threshold.json is absent in this environment)."))
def test_an_attended_turn_asks_for_the_missing_author_and_the_reply_completes_the_call(workflow_env):
    workflow_env(**_planner_settings())
    sink = RecordingTraceSink()
    ctx, _wf = _session(sink, folder=PLANNING_COMMANDS)

    first = ctx.process_turn(AUTHOR_REQUEST)

    assert first.status == TurnStatus.AWAITING_USER
    assert first.answer.startswith("I can't run author books yet:")
    assert first.answer.endswith("Which author should I use?")

    # The question survives a save and restore, as it does between two requests in production.
    state = ctx.serialize_state(channel_id="missing-author")
    restored = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    restored.bind_app_workflow(fastworkflow.Workflow.create(PLANNING_COMMANDS, workflow_id_str="missing-author-restored"))
    restored.apply_serialized_state(state)

    second = restored.process_turn("The author is Frank Herbert.")

    assert second.status == TurnStatus.COMPLETED
    call = sink.named("fw.planner.plan")[-1]
    assert call.attributes["skill"] == "author_books"
    assert call.attributes["skill_args"]["author"] == "Frank Herbert"


@pytest.mark.slow
@pytest.mark.requires_llm_key
@needs_planner_llm
def test_an_unattended_turn_never_asks_about_a_refused_call(workflow_env):
    workflow_env(**_planner_settings())
    sink = RecordingTraceSink()
    ctx, wf = _session(sink, folder=PLANNING_COMMANDS)
    ctx._begin_turn(AUTHOR_REQUEST, unattended=True)

    ctx.push_active_workflow(wf)
    try:
        workflow_agent.build_query_with_next_steps(AUTHOR_REQUEST, ctx, planning_command=_planning_command("author_books"))
    finally:
        ctx.pop_active_workflow()

    # Whether the model refuses or guesses a value the request writes, an unattended turn is never
    # put a question: a refusal falls back to the text planner, and no question is recorded.
    span = sink.named("fw.planner.plan")[-1]
    assert span.attributes["skill_refusal_asked"] is None
    assert ctx._planning_command_refusal is None
    assert span.attributes["plan_source"] in ("text", "skill_call")
