"""The policy verifier's engine and loader (fastworkflow.policy_decision, policy_loader).

Pure Python: no model, no workflow run. The loop wiring is in
tests/test_policy_loop.py and the executor guard in tests/test_policy_guard.py.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from fastworkflow.policy_decision import (
    NO_OP_TABLE,
    AfterObservationInput,
    BeforeFinishInput,
    BeforeSideEffectInput,
    CommandRecord,
    ContractFacts,
    DecisionPhase,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyError,
    PolicyMode,
    PolicyOutcome,
    PolicyRow,
    PolicyState,
    RequestedOperation,
)
from fastworkflow.policy_loader import load_policy

SIDE = frozenset({DecisionPhase.BEFORE_SIDE_EFFECT})
FINISH = frozenset({DecisionPhase.BEFORE_FINISH})


def _row(row_id, predicate, *, phases=SIDE, outcome=PolicyOutcome.REWRITE, rewrite="no"):
    return PolicyRow(row_id, phases, outcome, f"reason {row_id}", predicate, rewrite)


# ----------------------------------------------------------------------
# Table invariants
# ----------------------------------------------------------------------


def test_a_table_must_be_versioned():
    with pytest.raises(PolicyError, match="versioned"):
        DecisionTable(" ", [])


def test_row_ids_must_be_unique():
    with pytest.raises(PolicyError, match="duplicate"):
        DecisionTable("t/1", [_row("A", lambda i: True), _row("A", lambda i: True)])


def test_a_rewrite_row_must_say_something():
    with pytest.raises(PolicyError, match="nothing to say"):
        _row("A", lambda i: True, rewrite=None)


def test_a_row_needs_a_real_phase():
    with pytest.raises(PolicyError, match="no phase"):
        _row("A", lambda i: True, phases=frozenset())
    with pytest.raises(PolicyError, match="not a DecisionPhase"):
        _row("A", lambda i: True, phases=frozenset({"before-finish"}))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def test_first_matching_row_wins_and_is_cited():
    table = DecisionTable("t/1", [
        _row("A", lambda i: i.command_name == "delete", rewrite="read it first"),
        _row("B", lambda i: True, rewrite="never"),
    ])
    decision = table.evaluate(BeforeSideEffectInput(command_name="delete"))
    assert (decision.source_policy, decision.rewrite, decision.table_version) == (
        "A", "read it first", "t/1")
    assert decision.proposed_action == "delete"
    assert table.evaluate(BeforeSideEffectInput(command_name="other")).source_policy == "B"


def test_a_row_only_fires_at_its_phase():
    table = DecisionTable("t/1", [_row("A", lambda i: True, phases=FINISH)])
    assert table.evaluate(BeforeSideEffectInput(command_name="x")) is None
    assert table.evaluate(BeforeFinishInput(answer="done")).source_policy == "A"


def test_a_rewrite_can_name_what_was_wrong():
    table = DecisionTable("t/1", [_row(
        "A", lambda i: True, rewrite=lambda i: f"{i.command_name} needs a read first")])
    assert table.evaluate(BeforeSideEffectInput(command_name="close")).rewrite == (
        "close needs a read first")


def test_off_decides_nothing_and_an_allow_row_is_not_actionable():
    table = DecisionTable("t/1", [
        _row("EXEMPT", lambda i: i.command_name == "safe", outcome=PolicyOutcome.ALLOW,
             rewrite=None),
        _row("BLOCK", lambda i: True),
    ])
    inputs = BeforeSideEffectInput(command_name="safe")
    assert PolicyDecisionPoint(table, PolicyMode.OFF).decide(inputs) is None
    point = PolicyDecisionPoint(table, PolicyMode.ENFORCE)
    assert point.decide(inputs) is None
    assert point.decide(BeforeSideEffectInput(command_name="other")).source_policy == "BLOCK"
    assert [d.source_policy for d in point.enforced_log] == ["BLOCK"]
    assert not PolicyDecisionPoint().enabled and PolicyDecisionPoint().table is NO_OP_TABLE


# ----------------------------------------------------------------------
# Inputs and facts
# ----------------------------------------------------------------------


def test_identifier_arguments_are_flattened_and_matched_whole():
    facts = ContractFacts(id_patterns=(r"T-\d+",))
    inputs = BeforeSideEffectInput(
        facts=facts, command_name="merge",
        command_args={"ids": ["T-1", "T-22"], "note": "about T-3", "target": "T-4"},
    )
    assert sorted(inputs.id_arguments()) == [("ids", "T-1"), ("ids", "T-22"), ("target", "T-4")]


def test_phase_input_helpers():
    commands = (
        CommandRecord("@user", {"consents": False}, "close both tickets"),
        CommandRecord("list_tickets", {}, "T-1, T-2"),
        CommandRecord("close_ticket", {"id": "T-1"}, "Error: locked", success=False),
        CommandRecord("ask_user", {"clarification_request": "Close T-2?"}, "yes"),
    )
    inputs = AfterObservationInput(
        commands_run=commands,
        observations=tuple(c.response for c in commands),
        intent=(RequestedOperation("close", scope="all"),
                RequestedOperation("reopen", conditional=True)),
    )
    assert inputs.ran("close_ticket") == ()
    assert inputs.ran("list_tickets")[0].response == "T-1, T-2"
    assert inputs.last_user_utterance().response == "yes"
    assert inputs.requested("close", conditional=False)
    assert inputs.requested("reopen") and not inputs.requested("reopen", conditional=False)
    assert "T-2" in inputs.observation_text()


def test_failure_pattern_marks_text_errors_only_when_declared():
    assert not ContractFacts().reports_failure("Error: not found")
    facts = ContractFacts(failure_pattern=r"\b(error|not found)\b")
    assert facts.reports_failure("Result: Error: not found")
    # Only the start of a response is read: a word deep inside a result is data.
    assert not facts.reports_failure('{"id": "T-1", "status": "open"}' + "x" * 200 + " error")


def test_policy_state_round_trips_as_plain_json():
    state = PolicyState(
        commands_run=[CommandRecord("close_ticket", {"id": "T-1"}, "ok"),
                      CommandRecord("@user", {"consents": True}, "yes")],
        dialogue=[("user", "close it"), ("assistant", "Close T-1?")],
        requests=[{"id": "R1", "op": "close", "scope": "all", "status": "firm", "detail": ""}],
        intent=[RequestedOperation("close", scope="all")],
        block_counts={"A|execute_workflow_query|close_ticket": 1},
    )
    restored = PolicyState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

TABLE_MODULE = """
from fastworkflow.policy_decision import (
    ContractFacts, DecisionPhase, DecisionTable, PolicyOutcome, PolicyRow)

TABLE = DecisionTable("tmp/1", [PolicyRow(
    "R1", frozenset({DecisionPhase.BEFORE_SIDE_EFFECT}), PolicyOutcome.REWRITE,
    "always", lambda i: True, "blocked")])
OTHER = TABLE
FACTS = ContractFacts(write_commands=frozenset({"close_ticket"}))
"""


def _workflow_with_table(tmp_path, source=TABLE_MODULE):
    (tmp_path / "_policy").mkdir()
    (tmp_path / "_policy" / "table.py").write_text(textwrap.dedent(source))
    return str(tmp_path)


def test_a_workflow_without_a_table_loads_nothing(tmp_path):
    point, facts = load_policy(str(tmp_path))
    assert not point.enabled and facts == ContractFacts()
    point, _ = load_policy(None)
    assert not point.enabled


def test_a_workflow_with_a_table_is_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)  # not the workflow folder
    point, facts = load_policy(_workflow_with_table(tmp_path))
    assert point.enabled and point.mode is PolicyMode.ENFORCE
    assert point.table.version == "tmp/1"
    assert facts.is_write("close_ticket")


def test_a_broken_workflow_table_is_an_error(tmp_path):
    folder = _workflow_with_table(tmp_path, "TABLE = None\n")
    with pytest.raises(ValueError, match="not a DecisionTable"):
        load_policy(folder)


def test_another_table_can_be_named_relative_to_the_workflow(tmp_path):
    point, _ = load_policy(
        _workflow_with_table(tmp_path), reference="_policy/table.py:OTHER")
    assert point.enabled and point.table.version == "tmp/1"


def test_a_named_table_that_is_missing_or_wrong_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_policy(str(tmp_path), reference="nope.py")
    (tmp_path / "table.py").write_text(textwrap.dedent(TABLE_MODULE))
    with pytest.raises(ValueError, match="not a DecisionTable"):
        load_policy(str(tmp_path), reference="table.py:FACTS")


def test_identifier_patterns_come_from_the_workflow_when_the_table_has_none(tmp_path):
    (tmp_path / "table.py").write_text(textwrap.dedent(TABLE_MODULE))
    (tmp_path / "_commands").mkdir()
    (tmp_path / "_commands" / "_split.py").write_text(
        "import re\nidentifier_patterns = [re.compile(r'T-\\d+')]\n")
    _, facts = load_policy(str(tmp_path), reference="table.py")
    assert facts.id_patterns == (r"T-\d+",)
    assert facts.looks_like_id("T-12") and not facts.looks_like_id("x T-12")
