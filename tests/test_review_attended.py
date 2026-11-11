"""An attended sub-task review that the user accepts (the round-1 gate's review, replied to with 'accept').

Two things the review's accept path needs and that the review tests above do not reach:

- The framework command class of review_subtasks (its Input and Output) is routed by its plain module name, so the
  facts of its phase can read the review's typed output (typed_phase_facts._command_qname_from_class).
- After the code steps of the review phase, a framework planning command with a result (__result__) takes no
  model step: the accepted proposal is its result, and the agent does not re-issue review_proposal and ask the
  user about it (workflow_execution_context._run_planning_command_phase).

Integration tests on the annotated library copy with the real agent and the real gate. The model is scripted (the
approval fixtures' Script); no test calls a model, and the parameter-extraction guard stays on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import review_engine, review_proposals
from fastworkflow.review_engine import CONFIRMED
from fastworkflow.turn import TurnStatus
from fastworkflow.typed_phase_facts import _command_qname_from_class
from fastworkflow.utils.python_utils import get_module
from fastworkflow._framework_commands.review_proposal import Signature as ReviewProposal
from fastworkflow._framework_commands.review_subtasks import Signature as ReviewSubtasks
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import (  # noqa: F401 - the gate fixture and the draft's two sub-tasks
    EMMA, JOKE, _open_review, _split_into, gated)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
STOPPED = "stopped at its part"          # the turn's message when a phase fails (the phase's output names the part)


# ----------------------------------------------------------------------
# (a) Routing: a framework command class by its plain name; an app command class by its qualified name
# ----------------------------------------------------------------------


def test_a_framework_command_class_is_routed_by_its_plain_module_name():
    assert _command_qname_from_class(ReviewSubtasks.Output) == "review_subtasks"
    assert _command_qname_from_class(ReviewSubtasks.Input) == "review_subtasks"
    assert _command_qname_from_class(ReviewProposal.Output) == "review_proposal"


def test_an_app_command_class_keeps_its_qualified_name():
    module = get_module(str(LIBRARY / "_commands" / "Record" / "add_note.py"), search_root=str(LIBRARY))
    assert _command_qname_from_class(module.Signature.Output) == "Record/add_note"
    assert _command_qname_from_class(module.Signature.Input) == "Record/add_note"


# ----------------------------------------------------------------------
# (b) The attended accept: the review completes, its accepted sub-tasks run, the review is not issued again
# ----------------------------------------------------------------------


def test_an_attended_accept_completes_the_review_and_runs_its_sub_tasks(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    opened = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert opened.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    gated.script.steps = [("finish", {}), ("finish", {})]          # one per accepted sub-task (no review step)

    out = gated.ctx.process_turn("accept")

    assert record.state == CONFIRMED
    assert out.status is TurnStatus.COMPLETED
    responses = [output.command_response.response or "" for output in out.command_outputs
                 if getattr(output, "command_response", None) is not None]
    assert not any(STOPPED in text or "could not be read" in text for text in responses)
    assert "could not be read" not in (out.answer or "")
    assert "### 1. " + EMMA in (out.answer or "") and "### 2. " + JOKE in (out.answer or "")
    accepted = review_proposals.accepted_subtask_proposal(gated.wf)
    assert accepted is not None and [item.goal for item in accepted.items] == [EMMA, JOKE]


def test_an_accepted_review_takes_no_model_step_after_its_code_steps_and_is_not_issued_again(gated, monkeypatch):
    issued: list = []
    original = review_engine.approved_review_proposal

    def counted(workflow, spec):
        issued.append(spec)
        return original(workflow, spec)

    monkeypatch.setattr(review_engine, "approved_review_proposal", counted)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    _open_review(gated)
    issued.clear()                                                  # the opening call is not the accept
    gated.script.steps = [("finish", {}), ("finish", {})]          # the review phase's own step must not be asked

    gated.ctx.process_turn("accept")

    assert gated.script.steps == []                                 # every scripted step was used, none left over
    assert issued == ["review_subtasks"]                            # the accept runs review_proposal once, not again
