"""Jev's reading of a reply to a missing-value prompt (fix-jd9o.6).

The classifier and the dispatch are exercised with hand-written stand-ins (no unittest.mock):
``workflow_agent._jev_lm`` returns a DummyLM, ``dspy.Predict`` answers with a fixed Jev intent,
CommandNamePrediction and CommandExecutor.perform_action are replaced by small recorders. The
dispatch tests drive ``ResponseGenerator.__call__`` in the PARAMETER_EXTRACTION stage, so the code
paths under test are the real ones; the parts that need a trained intent model, an LLM or a running
executor are the stand-ins. The real-Jev test skips without JEV_API_KEY.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils import DummyLM
from dotenv import dotenv_values
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import NLUPipelineStage
from fastworkflow import workflow_agent
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    MISSING_VALUE_REPLY_THRESHOLD,
    classify_missing_value_reply,
    missing_value_reply_enabled,
)
from fastworkflow.utils.signatures import NOT_FOUND

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PE = NLUPipelineStage.PARAMETER_EXTRACTION
PENDING_COMMAND = "ChatRoom/set_current_user"
QUESTION_PREFIX = "PARAMETER EXTRACTION ERROR FOR COMMAND"


class Params(BaseModel):
    user_name: str = Field(description="User name")


class _Predict:
    """Stand-in for the object dspy.Predict(signature) returns: one fixed Jev intent, every call recorded."""

    def __init__(self, value: str, probability: float, probabilities: dict | None | str = "same"):
        self.value = value
        self.probability = probability
        self.probabilities = {value: probability} if probabilities == "same" else probabilities
        self.calls: list[tuple[str, str]] = []

    def __call__(self, question, reply):
        self.calls.append((question, reply))
        intent = SimpleNamespace(value=self.value, probabilities=self.probabilities,
                                 confidence=self.probability)
        return SimpleNamespace(intent=intent)


class _Failing:
    def __call__(self, question, reply):
        raise RuntimeError("provider down")


class _Workflow:
    """A session context with the real end_command_processing effects (stored parameters and command go)."""

    def __init__(self, context: dict):
        self.context = context

    def end_command_processing(self) -> None:
        self.context.pop("command", None)
        self.context.pop("stored_parameters", None)
        self.context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_DETECTION


def _app_workflow(run_as_agent: bool = False) -> SimpleNamespace:
    context = {"run_as_agent": True} if run_as_agent else {}
    return SimpleNamespace(context=context, folderpath="/nonexistent-workflow",
                           _current_command_context_name="ChatRoom", _current_command_context=None,
                           root=None)


def _pending(app_workflow=None, stored=None) -> _Workflow:
    """A session in the PARAMETER_EXTRACTION stage with one missing field, as the prompt left it."""
    context = {
        "app_workflow": app_workflow or _app_workflow(),
        "NLU_Pipeline_Stage": PE,
        "command_name": PENDING_COMMAND,
        "command": f"{PENDING_COMMAND} for the team",
        "stored_parameters": stored if stored is not None else Params.model_construct(user_name=NOT_FOUND),
    }
    return _Workflow(context)


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_MISSING_VALUE_REPLY", "on")


@pytest.fixture
def jev_on(monkeypatch):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))


def _classifier(monkeypatch, stand_in) -> None:
    """dspy.Predict(signature) returns the stand-in, which answers the call with the Jev intent."""
    monkeypatch.setattr(dspy, "Predict", lambda signature: stand_in)


# ---- the classifier -------------------------------------------------------------------------

def test_the_question_and_reply_reach_jev_as_given(monkeypatch, jev_on):
    predictor = _Predict("answer", 0.97)
    _classifier(monkeypatch, predictor)
    assert classify_missing_value_reply(
        "PARAMETER EXTRACTION ERROR FOR COMMAND 'x'\nMissing parameter values: user_name", "dana.k@corp.io"
    ) == "answer"
    question, reply = predictor.calls[0]
    assert question.startswith(QUESTION_PREFIX)
    assert reply == "dana.k@corp.io"


@pytest.mark.parametrize("intent", ["different_command", "misunderstood", "abort", "wait"])
def test_a_non_answer_intent_applies_at_the_threshold(monkeypatch, jev_on, intent):
    assert MISSING_VALUE_REPLY_THRESHOLD == 0.5
    _classifier(monkeypatch, _Predict(intent, 0.5))
    assert classify_missing_value_reply("q", "reply") == intent


def test_a_non_answer_intent_below_the_threshold_is_an_answer(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("wait", 0.49))
    assert classify_missing_value_reply("q", "reply") == "answer"


def test_an_answer_is_an_answer_at_any_probability(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("answer", 0.99))
    assert classify_missing_value_reply("q", "reply") == "answer"


def test_the_probability_is_read_from_the_probabilities_map(monkeypatch, jev_on):
    intent = SimpleNamespace(value="wait", probabilities={"wait": 0.8}, confidence=0.1)
    monkeypatch.setattr(dspy, "Predict", lambda sig: (lambda question, reply: SimpleNamespace(intent=intent)))
    assert classify_missing_value_reply("q", "hold on") == "wait"


def test_the_confidence_is_the_probability_when_no_map_is_given(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("abort", 0.7, probabilities=None))
    assert classify_missing_value_reply("q", "stop") == "abort"


def test_no_jev_lm_makes_no_call_and_is_an_answer(monkeypatch):
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
    predictor = _Predict("wait", 0.99)
    _classifier(monkeypatch, predictor)
    assert classify_missing_value_reply("q", "hmm") == "answer"
    assert predictor.calls == []


def test_a_failed_call_is_an_answer(monkeypatch, jev_on):
    _classifier(monkeypatch, _Failing())
    assert classify_missing_value_reply("q", "hmm") == "answer"


def test_the_off_switch_is_read_case_insensitively(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_MISSING_VALUE_REPLY", "OFF")
    assert missing_value_reply_enabled() is False
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_MISSING_VALUE_REPLY", "on")
    assert missing_value_reply_enabled() is True


# ---- when the check runs --------------------------------------------------------------------

def test_agent_mode_makes_no_call(monkeypatch, jev_on):
    predictor = _Predict("wait", 0.99)
    _classifier(monkeypatch, predictor)
    workflow = _pending(app_workflow=_app_workflow(run_as_agent=True))
    intent, _ = wildcard._missing_value_reply_intent(
        workflow, workflow.context["app_workflow"], PENDING_COMMAND, "hmm")
    assert intent == "answer"
    assert predictor.calls == []


def test_the_off_switch_makes_no_call(monkeypatch, jev_on):
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_MISSING_VALUE_REPLY", "false")
    predictor = _Predict("wait", 0.99)
    _classifier(monkeypatch, predictor)
    workflow = _pending()
    assert wildcard._missing_value_reply_intent(
        workflow, workflow.context["app_workflow"], PENDING_COMMAND, "hmm")[0] == "answer"
    assert predictor.calls == []


def test_nothing_pending_makes_no_call(monkeypatch, jev_on):
    predictor = _Predict("wait", 0.99)
    _classifier(monkeypatch, predictor)
    workflow = _pending(stored=None)
    del workflow.context["stored_parameters"]
    assert wildcard._missing_value_reply_intent(
        workflow, workflow.context["app_workflow"], PENDING_COMMAND, "hmm")[0] == "answer"
    assert predictor.calls == []


# ---- the dispatch in ResponseGenerator.__call__ ---------------------------------------------

class _CommandNamePrediction:
    """Stand-in for the CME intent model: the pending command is what the stage already holds."""

    def __init__(self, workflow):
        pass

    def predict(self, context_name, command, stage):
        return SimpleNamespace(error_msg=None, is_cme_command=False, command_name=PENDING_COMMAND,
                               routing_hint=None, answered_bare_value=False)


class _NoMatchPrediction(_CommandNamePrediction):
    """Matches the pending command while it is pending, and matches nothing in intent detection,
    where a reply that is a different request is read again."""

    def predict(self, context_name, command, stage):
        if stage == PE:
            return super().predict(context_name, command, stage)
        return SimpleNamespace(error_msg="no command matches", is_cme_command=False, command_name=None,
                               routing_hint=None, answered_bare_value=False)


def _executor(monkeypatch, recorded: list):
    """Stand-in for CommandExecutor.perform_action: records the framework action it was asked to run."""
    def perform_action(workflow, action):
        recorded.append(action)
        return SimpleNamespace(success=True, command_response=SimpleNamespace(
            response=f"ran {action.command_name}", output=None))
    monkeypatch.setattr(wildcard.CommandExecutor, "perform_action", perform_action)


def _reply(monkeypatch, workflow, text: str):
    return wildcard.ResponseGenerator()(workflow, text, None)


def test_wait_keeps_the_pending_round(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("wait", 0.9))
    monkeypatch.setattr(wildcard, "CommandNamePrediction", _CommandNamePrediction)
    recorded: list = []
    _executor(monkeypatch, recorded)
    workflow = _pending()
    stored = workflow.context["stored_parameters"]

    output = _reply(monkeypatch, workflow, "hold on, checking")

    assert output.handled is False and output.routed is False
    assert output.response.startswith("OK, take your time.\nMissing parameter values: ")
    assert "user_name" in output.response
    assert workflow.context["NLU_Pipeline_Stage"] == PE
    assert workflow.context["stored_parameters"] is stored
    assert workflow.context["command_name"] == PENDING_COMMAND
    assert recorded == []


def test_the_next_real_answer_after_a_wait_completes_the_command(monkeypatch, jev_on):
    """After a wait the pending round is untouched, so the next answer reaches the merge as before.

    The merge itself needs the extraction LLM, so the stand-in here records that the answer was
    routed to ParameterExtraction with the pending command's text, and nothing else."""
    _classifier(monkeypatch, _Predict("answer", 0.97))
    monkeypatch.setattr(wildcard, "CommandNamePrediction", _CommandNamePrediction)
    merged: list = []

    class _Extraction(wildcard.ParameterExtraction):
        def __init__(self, workflow, app_workflow, command_name, command):
            merged.append((command_name, command))

        def extract(self):
            # The merge itself needs the extraction LLM: the stand-in ends the turn here.
            return SimpleNamespace(parameters_are_valid=False, error_msg="stop here", cmd_parameters=None)

    monkeypatch.setattr(wildcard, "ParameterExtraction", _Extraction)
    workflow = _pending()
    output = _reply(monkeypatch, workflow, "dana.k@corp.io")
    assert merged == [(PENDING_COMMAND, f"{PENDING_COMMAND}: dana.k@corp.io")]
    assert output.routed is False


def test_abort_runs_the_abort_command_and_clears_the_round(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("abort", 0.9))
    monkeypatch.setattr(wildcard, "CommandNamePrediction", _CommandNamePrediction)
    recorded: list = []
    _executor(monkeypatch, recorded)
    workflow = _pending()

    output = _reply(monkeypatch, workflow, "cancel that")

    assert [a.command_name for a in recorded] == ["ErrorCorrection/abort"]
    assert output.handled is True and output.routed is True
    assert output.response == "ran ErrorCorrection/abort"
    assert "stored_parameters" not in workflow.context
    assert workflow.context["NLU_Pipeline_Stage"] == NLUPipelineStage.INTENT_DETECTION


def test_misunderstood_runs_you_misunderstood_in_its_stage(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("misunderstood", 0.9))
    monkeypatch.setattr(wildcard, "CommandNamePrediction", _CommandNamePrediction)
    recorded: list = []
    _executor(monkeypatch, recorded)
    workflow = _pending()

    output = _reply(monkeypatch, workflow, "that is not what I meant")

    assert [a.command_name for a in recorded] == ["ErrorCorrection/you_misunderstood"]
    assert output.handled is False and output.routed is True
    assert workflow.context["NLU_Pipeline_Stage"] == NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
    assert workflow.context["command"] == "that is not what I meant"


class _ReachedIntentDetection(Exception):
    """Raised by the intent-model stand-in: the reply has reached intent detection."""


class _IntentStage(_CommandNamePrediction):
    """Matches the pending command while it is pending; in intent detection it stops the turn and
    reports the reply it was given. Routing a fresh request past this point needs the real command
    registry, which these tests do not load."""

    def predict(self, context_name, command, stage):
        if stage == PE:
            return super().predict(context_name, command, stage)
        raise _ReachedIntentDetection(command)


def test_different_command_leaves_the_pending_round_and_reads_the_reply_as_a_request(monkeypatch, jev_on):
    _classifier(monkeypatch, _Predict("different_command", 0.9))
    monkeypatch.setattr(wildcard, "CommandNamePrediction", _IntentStage)
    # The fresh request is read from its start: no registry lookups before the intent model is asked.
    from fastworkflow import workflow_agent as wa
    from fastworkflow import context_navigation
    monkeypatch.setattr(wa, "core_command_names", lambda: [])
    monkeypatch.setattr(context_navigation, "command_on_path", lambda app_workflow, name: None)
    recorded: list = []
    _executor(monkeypatch, recorded)
    workflow = _pending()

    with pytest.raises(_ReachedIntentDetection) as reached:
        _reply(monkeypatch, workflow, "show me the users")

    assert str(reached.value.args[0]) == "show me the users"
    assert "stored_parameters" not in workflow.context
    assert workflow.context["NLU_Pipeline_Stage"] == NLUPipelineStage.INTENT_DETECTION
    assert recorded == []


@pytest.fixture
def jev_env():
    """Load the env files into fastworkflow and skip when JEV_API_KEY is absent; restore afterwards."""
    previous = dict(fastworkflow._env_vars)
    env_vars = {
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env")),
        **dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.passwords.env")),
    }
    override = os.path.join(ROOT, "passwords", ".env")
    if os.path.exists(override):
        env_vars.update(dotenv_values(override))
    if not env_vars.get("JEV_API_KEY"):
        pytest.skip("JEV_API_KEY not set in passwords/.env: the reply check needs a real Jev call")
    fastworkflow.init(env_vars=env_vars)
    workflow_agent._jev_lm.cache_clear()
    try:
        yield
    finally:
        fastworkflow.init(env_vars=previous)
        # The live Jev LM this test cached must not outlive it: _jev_lm() is process-wide, and a cached
        # LM makes every later turn take the jev_tiebreak path (fix-5voz).
        workflow_agent._jev_lm.cache_clear()


@pytest.mark.parametrize("reply, expected", [
    ("hmm, hold on a sec", "wait"),
    ("dana.k@corp.io", "answer"),
])
def test_real_jev_reads_hesitation_and_answers(jev_env, reply, expected):
    question = (f"{QUESTION_PREFIX} 'ChatRoom/set_current_user'\n"
                "Missing parameter values: user_name")
    assert classify_missing_value_reply(question, reply) == expected


# ---------------------------------------------------------------------------
# End to end, through a real assistant-mode session on the trained hello_world workflow: the real
# CME pipeline, the real wildcard dispatch, the real executor. Only Jev's decision is a stand-in.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction
from fastworkflow.utils.signatures import INVALID_FLOAT_VALUE
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

_HELLO = str(_Path(__file__).parent.joinpath("hello_world_workflow").resolve())
_PENDING = "add_two_numbers"


@pytest.fixture
def _assistant_session(tmp_path, monkeypatch):
    """An assistant-mode session left waiting for add_two_numbers' second number."""
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="jd9o6-e2e")
    ctx.bind_app_workflow(fastworkflow.Workflow.create(_HELLO, workflow_id_str="jd9o6-e2e"))
    params_class = fastworkflow.RoutingRegistry.get_definition(ctx.app_workflow.folderpath).get_command_class(
        _PENDING, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "add 5 and something"
    cme["command_name"] = _PENDING
    ParameterExtraction._store_parameters(
        ctx._cme_workflow, params_class.model_construct(first_num=5.0, second_num=INVALID_FLOAT_VALUE))
    yield ctx
    RoutingRegistry.clear_registry()


def _jev_says(monkeypatch, intent: str, probability: float = 0.95):
    """Jev's decision for the missing-value reply; every other dspy.Predict is the real one."""
    real_predict = dspy.Predict
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))

    def predict(signature, *args, **kwargs):
        if "intent" in getattr(signature, "output_fields", {}):
            return lambda **_: SimpleNamespace(intent=SimpleNamespace(
                value=intent, probabilities={intent: probability}, confidence=probability))
        return real_predict(signature, *args, **kwargs)

    monkeypatch.setattr(dspy, "Predict", predict)


def _cme_state(ctx):
    cme = ctx._cme_workflow.context
    return cme.get("NLU_Pipeline_Stage"), ParameterExtraction._get_stored_parameters(ctx._cme_workflow)


def test_e2e_a_hesitation_keeps_the_pending_round(_assistant_session, monkeypatch):
    _jev_says(monkeypatch, "wait")
    out = _assistant_session.process_turn("hmm, hold on a sec")
    text = str(out.model_dump())
    assert "OK, take your time." in text and "second_num" in text, text
    stage, stored = _cme_state(_assistant_session)
    assert stage == NLUPipelineStage.PARAMETER_EXTRACTION
    assert stored is not None and stored.first_num == 5.0


def test_e2e_an_abort_leaves_the_pending_round(_assistant_session, monkeypatch):
    _jev_says(monkeypatch, "abort")
    _assistant_session.process_turn("never mind, forget it")
    stage, stored = _cme_state(_assistant_session)
    assert stage == NLUPipelineStage.INTENT_DETECTION
    assert stored is None


def test_e2e_a_different_request_runs_as_a_fresh_request(_assistant_session, monkeypatch):
    _jev_says(monkeypatch, "different_command")
    out = _assistant_session.process_turn("what can i do?")
    stage, stored = _cme_state(_assistant_session)
    assert stored is None
    assert stage == NLUPipelineStage.INTENT_DETECTION
    assert "PARAMETER EXTRACTION ERROR" not in str(out.model_dump())
