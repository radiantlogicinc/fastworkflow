"""Deleting an unused experiment and the winner pointer (`fix-jfy5`).

Creation registers an experiment and elects it when its group is empty;
deletion used to tombstone only the JSON registration. A group could therefore
name a winner nobody could open: `workflow_winner` reported the deleted id,
the members list still contained it, and duplicating the winner -- the one
action the winner screen offers -- failed on the tombstone.

The policy, which is deliberately small:

- The CURRENT WINNER is not deleted while its group has other members. Refused
  with a conflict (409 at the HTTP edge), automatic first winner included,
  because the alternatives are filing a "nobody won" nobody decided or electing
  a successor during an unrelated deletion. Select a different winner first.
- The winner that is its group's ONLY member IS deleted (`fix-65ik`): there is
  nobody to select instead, so the refusal had no way out. The group is left
  empty and winner-less, and its next experiment is elected automatically.
- Any other unused registration is deleted AND withdrawn from the contest, in
  one control transaction, so a promotion arriving afterwards cannot name it.
- Nothing rewrites history: withdrawal appends a `retire` row.

Integration throughout, per `.cursor/rules/testing_rules.mdc`: real
registrations and the control tables of the workflow's real live DB, a real
`ObservabilityStore`
written through a real `ExperimentController`, and the
real `ChatbotServer` over a real socket. No Mock fixtures, and nothing here runs
a model or spends anything -- every experiment below is an identity and a
declaration.
"""

from __future__ import annotations

import os
import threading

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import control as control_module
from fastworkflow.observability import selection
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request

HUMAN = {"actor": "dhar", "actor_kind": "human"}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A live workflow whose live DB lives under a temp state root."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "roster_workflow"
    workflow.mkdir()
    (workflow / "_commands").mkdir()
    obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    return workflow


@pytest.fixture
def benchmark_id(folder):
    return setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )["benchmark_id"]


def _create(folder, benchmark_id):
    return setup.create_experiment(folder, benchmark_id, "v1")["experiment_id"]


def _control(folder):
    return setup.workflow_control(folder, write=True)


def _winner_id(folder, experiment_id):
    winner = setup.workflow_winner(folder, experiment_id)
    return None if winner is None else winner["experiment_id"]


def _group_id(folder, experiment_id):
    group = _control(folder).group_for_experiment(experiment_id)
    return None if group is None else str(group["group_id"])


def _member_ids(folder, group_id):
    return {str(row["experiment_id"]) for row in _control(folder).group_members(group_id)}


def _history(folder, group_id):
    return [
        (int(row["seq"]), str(row["decision"]), row["candidate_experiment_id"],
         row["new_experiment_id"])
        for row in _control(folder).decision_history(group_id)
    ]


def _promote(folder, candidate_id, *, control=None):
    """Move the winner to `candidate_id`, the way the decisions API does."""
    control = control or _control(folder)
    winner = control.winner_for_experiment(candidate_id)
    return control.record_decision(
        str(winner["group_id"]),
        "promote",
        expected_selection_id=str(winner["selection_id"]),
        candidate_experiment_id=candidate_id,
        provenance="human",
        **HUMAN,
    )


def _declare(folder, db_path, record):
    """Hand a registration to a real runner, the way `run` does."""
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        str(folder), store.store_identity(), external=False,
    )
    controller.create_experiment(
        record["experiment_id"],
        record.get("description", ""),
        declared_tasks=len(record["task_ids"]),
        declared_attempts=1,
        declarations=[
            (task_id, 1, f"ch-{record['experiment_id'][-4:]}-{task_id}")
            for task_id in record["task_ids"]
        ],
        workflow_name=setup.workflow_name_for(folder),
    )
    return store


# ----------------------------------------------------------------------
# The current winner is not deleted
# ----------------------------------------------------------------------


class TestTheCurrentWinnerIsNotDeleted:
    def test_the_reported_reproduction_can_no_longer_happen(
        self, folder, benchmark_id
    ):
        """`fix-jfy5` as filed: create A, delete unused A, create B.

        The state it produced -- the group naming a tombstoned A -- is now
        unreachable. A is the group's only member, so deleting it takes the
        pointer with it (`fix-65ik`), and B is elected the way A was, rather
        than joining a group that still names A.
        """
        first = _create(folder, benchmark_id)
        assert _winner_id(folder, first) == first
        group_id = _group_id(folder, first)

        setup.delete_empty_experiment(folder, first)

        second = _create(folder, benchmark_id)
        winner = setup.workflow_winner(folder, second)
        assert winner["experiment_id"] == second
        assert winner["automatic"] is True
        assert _member_ids(folder, group_id) == {second}
        # Nothing was refused, so this is what was recorded instead: the
        # withdrawal of A and the pointer with it, then B's own election.
        assert _history(folder, group_id) == [
            (3, "initial", second, second),
            (2, "retire", first, None),
            (1, "initial", first, first),
        ]
        # And the thing the user came to do still works.
        assert setup.duplicate_experiment(folder, second)["source_experiment_id"] == second

    def test_the_refusal_is_the_conflict_the_http_edge_already_answers_with_409(self):
        """A refusal that was not a `BenchmarkSetupConflict` would reach the
        request handler's catch-all and become a 500 saying "refresh and try
        again", which is not something a user can act on."""
        assert issubclass(setup.ExperimentSelected, setup.BenchmarkSetupConflict)

    def test_a_promoted_winner_is_refused_too(self, folder, benchmark_id):
        first = _create(folder, benchmark_id)
        second = _create(folder, benchmark_id)
        _promote(folder, second)

        with pytest.raises(setup.ExperimentSelected):
            setup.delete_empty_experiment(folder, second)

        assert setup.load_experiment(folder, second)["experiment_id"] == second
        assert _winner_id(folder, first) == second

    def test_selecting_a_different_winner_first_makes_the_deletion_work(
        self, folder, benchmark_id
    ):
        """The way out of the refusal, and the reason it is acceptable."""
        first = _create(folder, benchmark_id)
        second = _create(folder, benchmark_id)
        _promote(folder, second)

        setup.delete_empty_experiment(folder, first)

        assert _winner_id(folder, second) == second
        assert _member_ids(folder, _group_id(folder, second)) == {second}
        with pytest.raises(setup.ExperimentDeleted):
            setup.load_experiment(folder, first)

    def test_the_last_registration_of_a_group_is_deleted_and_leaves_it_empty(
        self, folder, benchmark_id
    ):
        """No dead end (`fix-65ik`): the last one standing can go.

        It used to stay, because it was the winner -- and promoting a second
        experiment to free the first only made the second the last one
        standing. Deleting it leaves a group with no members and no pointer.
        """
        only = _create(folder, benchmark_id)
        group_id = _group_id(folder, only)

        setup.delete_empty_experiment(folder, only)

        assert _member_ids(folder, group_id) == set()
        assert _control(folder).current_winner(group_id) is None
        with pytest.raises(setup.ExperimentDeleted):
            setup.load_experiment(folder, only)


# ----------------------------------------------------------------------
# The sole winner can be deleted (`fix-65ik`)
# ----------------------------------------------------------------------


class TestTheSoleWinnerCanBeDeleted:
    def test_history_records_the_winner_leaving_and_rewrites_nothing(
        self, folder, benchmark_id
    ):
        """An ordinary system `retire`, not a new kind of decision: it names
        the winner before and nobody after."""
        only = _create(folder, benchmark_id)
        group_id = _group_id(folder, only)
        elected = _history(folder, group_id)

        setup.delete_empty_experiment(folder, only)

        history = _control(folder).decision_history(group_id)
        assert [(int(r["seq"]), r["decision"]) for r in history] == [
            (2, "retire"), (1, "initial"),
        ]
        retirement = history[0]
        assert retirement["previous_experiment_id"] == only
        assert retirement["candidate_experiment_id"] == only
        assert retirement["new_experiment_id"] is None
        assert retirement["new_selection_id"] is None
        assert retirement["actor_kind"] == selection.SYSTEM_ACTOR_KIND
        assert _history(folder, group_id)[-1:] == elected

    def test_the_next_experiment_of_the_group_is_its_automatic_winner(
        self, folder, benchmark_id
    ):
        only = _create(folder, benchmark_id)
        group_id = _group_id(folder, only)
        setup.delete_empty_experiment(folder, only)

        successor = _create(folder, benchmark_id)
        later = _create(folder, benchmark_id)

        assert _group_id(folder, successor) == group_id
        winner = setup.workflow_winner(folder, later)
        assert winner["experiment_id"] == successor
        assert winner["automatic"] is True
        assert int(winner["decision_seq"]) == 3

    def test_an_automatic_winner_with_company_is_still_refused(
        self, folder, benchmark_id
    ):
        """Only the SOLE member qualifies: one more member and it is the
        ordinary refusal, with nothing changed."""
        first = _create(folder, benchmark_id)
        second = _create(folder, benchmark_id)
        group_id = _group_id(folder, first)
        before = _history(folder, group_id)

        with pytest.raises(setup.ExperimentSelected) as refusal:
            setup.delete_empty_experiment(folder, first)

        assert _winner_id(folder, first) == first
        assert _member_ids(folder, group_id) == {first, second}
        assert _history(folder, group_id) == before
        assert setup.load_experiment(folder, first)["experiment_id"] == first
        # The message says why and what to do about it.
        message = str(refusal.value)
        assert "current winner" in message
        assert "other experiments" in message
        assert "promote" in message

    def test_the_control_still_refuses_a_sole_winner_unless_asked(
        self, folder, benchmark_id
    ):
        """`allow_sole_winner` is opt-in; the control's default is unchanged."""
        only = _create(folder, benchmark_id)

        with pytest.raises(selection.SelectionRetirementRefused) as refusal:
            _control(folder).retire_experiment(only)

        assert refusal.value.reason == "is_current_winner"
        assert _winner_id(folder, only) == only

    def test_a_sole_winner_deletion_racing_a_creation_agrees_either_way(
        self, folder
    ):
        """Unsynchronised. The two orders are "withdrawn, then the newcomer is
        elected" and "the newcomer joined, so the deletion is refused".
        Neither leaves the group without a winner or naming a tombstone."""
        for n in range(6):
            benchmark_id = setup.save_benchmark(
                folder, {"title": f"Race {n}", "tasks": [{"prompt": "Review"}]}
            )["benchmark_id"]
            only = _create(folder, benchmark_id)
            outcome = {}

            def delete():
                try:
                    setup.delete_empty_experiment(folder, only)
                    outcome["deleted"] = True
                except setup.ExperimentSelected:
                    outcome["deleted"] = False

            def create():
                outcome["created"] = _create(folder, benchmark_id)

            threads = [threading.Thread(target=delete), threading.Thread(target=create)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
                assert not thread.is_alive()

            newcomer = outcome["created"]
            winner = _winner_id(folder, newcomer)
            if outcome["deleted"]:
                assert winner == newcomer
                assert _member_ids(folder, _group_id(folder, newcomer)) == {newcomer}
            else:
                assert winner == only
                assert _member_ids(folder, _group_id(folder, newcomer)) == {only, newcomer}


# ----------------------------------------------------------------------
# A non-winner leaves the contest with its registration
# ----------------------------------------------------------------------


class TestANonWinnerLeavesTheContest:
    def test_a_late_promotion_cannot_target_a_deleted_registration(
        self, folder, benchmark_id
    ):
        """The guarantee the membership removal exists for.

        Without it the deleted id stays a valid candidate, and a promotion
        recorded minutes later moves the winner pointer onto a tombstone --
        `fix-jfy5` reached from the other end.
        """
        winner = _create(folder, benchmark_id)
        doomed = _create(folder, benchmark_id)
        control = _control(folder)
        current = control.winner_for_experiment(winner)

        setup.delete_empty_experiment(folder, doomed)

        with pytest.raises(selection.ExperimentNotInGroup):
            control.record_decision(
                str(current["group_id"]), "promote",
                expected_selection_id=str(current["selection_id"]),
                candidate_experiment_id=doomed, provenance="human", **HUMAN,
            )
        assert _winner_id(folder, winner) == winner

    def test_a_promotion_that_lands_first_turns_the_deletion_into_a_refusal(
        self, folder, benchmark_id
    ):
        """The other order of the same race, run deterministically.

        The check and the removal are one control transaction, so these two
        orders are the only two outcomes there are.
        """
        winner = _create(folder, benchmark_id)
        doomed = _create(folder, benchmark_id)
        _promote(folder, doomed)

        with pytest.raises(setup.ExperimentSelected):
            setup.delete_empty_experiment(folder, doomed)

        assert setup.load_experiment(folder, doomed)["experiment_id"] == doomed
        assert _winner_id(folder, winner) == doomed

    def test_the_winner_pointer_is_untouched_not_re_elected(
        self, folder, benchmark_id
    ):
        winner = _create(folder, benchmark_id)
        doomed = _create(folder, benchmark_id)
        group_id = _group_id(folder, winner)
        before = _control(folder).current_winner(group_id)

        setup.delete_empty_experiment(folder, doomed)

        after = _control(folder).current_winner(group_id)
        assert after["experiment_id"] == winner
        # The same selection, not a re-election that happens to agree.
        assert after["selection_id"] == before["selection_id"]
        assert after["decision_seq"] == before["decision_seq"]
        assert _member_ids(folder, group_id) == {winner}

    def test_history_gains_a_retire_row_and_rewrites_nothing(
        self, folder, benchmark_id
    ):
        winner = _create(folder, benchmark_id)
        doomed = _create(folder, benchmark_id)
        group_id = _group_id(folder, winner)
        elected = _history(folder, group_id)

        setup.delete_empty_experiment(folder, doomed)

        history = _history(folder, group_id)
        assert history[-1:] == elected  # the election, byte for byte
        assert [row[1] for row in history] == ["retire", "initial"]
        retirement = history[0]
        assert retirement[0] == 2  # appended, not renumbered
        assert retirement[2] == doomed  # candidate: what was withdrawn
        assert retirement[3] == winner  # the winner, unchanged by it

    def test_a_workflow_with_no_live_db_has_nothing_to_delete(self, tmp_path):
        """The registration lives in the live DB, so there is none without it,
        and deleting it brings no database into existence."""
        workflow = tmp_path / "unrecorded_workflow"
        workflow.mkdir()

        with pytest.raises(KeyError):
            setup.delete_empty_experiment(workflow, "exp-nobody")
        assert not os.path.exists(state_paths.observability_db(str(workflow)))


# ----------------------------------------------------------------------
# Evidence is never withdrawn
# ----------------------------------------------------------------------


class TestEvidenceIsNeverRetired:
    def test_a_bound_registration_is_refused_by_the_control_itself(
        self, folder, benchmark_id, tmp_path
    ):
        """Belt and braces: `delete_empty_experiment` refuses a bound
        registration before it gets here, but the control must not depend on a
        caller two layers up having checked."""
        _create(folder, benchmark_id)  # the winner, so this one is not
        record = setup.create_experiment(folder, benchmark_id, "v1")
        _declare(folder, state_paths.observability_db(str(folder)), record)
        experiment_id = record["experiment_id"]

        with pytest.raises(selection.SelectionRetirementRefused) as refusal:
            _control(folder).retire_experiment(experiment_id)

        assert refusal.value.reason == "has_evidence"
        assert experiment_id in _member_ids(folder, _group_id(folder, experiment_id))

    def test_deleting_a_running_experiment_is_still_refused(
        self, folder, benchmark_id, tmp_path
    ):
        _create(folder, benchmark_id)
        record = setup.create_experiment(folder, benchmark_id, "v1")
        store = _declare(folder, state_paths.observability_db(str(folder)), record)

        with pytest.raises(setup.BenchmarkSetupConflict):
            setup.delete_empty_experiment(folder, record["experiment_id"])

        assert store.get_experiment(record["experiment_id"]) is not None
        assert record["experiment_id"] in _member_ids(
            folder, _group_id(folder, record["experiment_id"])
        )

    def test_an_empty_registration_beside_a_recorded_one_still_deletes(
        self, folder, benchmark_id, tmp_path
    ):
        """The winner has evidence; the thing being deleted never ran."""
        recorded = setup.create_experiment(folder, benchmark_id, "v1")
        _declare(folder, state_paths.observability_db(str(folder)), recorded)
        empty = _create(folder, benchmark_id)

        setup.delete_empty_experiment(folder, empty)

        winner = setup.workflow_winner(folder, recorded["experiment_id"])
        assert winner["experiment_id"] == recorded["experiment_id"]
        assert winner["experiment_resolved"] is True
        assert _member_ids(folder, str(winner["group_id"])) == {
            recorded["experiment_id"]
        }


# ----------------------------------------------------------------------
# A group can retire before it elects
# ----------------------------------------------------------------------


class TestAnElectionAfterARetirementKeepsItsSequence:
    def test_a_group_that_retired_before_electing_still_elects(self, folder):
        """The `seq` the initial election used to hardcode.

        Members registered without an election, one withdrawn in the
        meantime: `retire` takes seq 1, and the election that finally arrives
        used to collide on the UNIQUE (scope, group, scope_key, seq), leaving
        the group permanently winner-less.
        """
        control = setup.workflow_control(folder, write=True)
        older = selection.ExperimentReference(
            experiment_id="exp-older", workflow_name="w", benchmark_id="b",
            created_at="2026-01-01T00:00:00Z",
        )
        newer = selection.ExperimentReference(
            experiment_id="exp-newer", workflow_name="w", benchmark_id="b",
            created_at="2026-01-02T00:00:00Z",
        )
        for reference in (older, newer):
            control.register_experiment_reference(
                reference, allow_initial_winner=False
            )
        group_id = str(control.group_for_experiment("exp-older")["group_id"])
        assert control.current_winner(group_id) is None

        retired = control.retire_experiment("exp-newer")
        assert retired["seq"] == 1

        elected = control.register_experiment_reference(older)

        assert elected["initialized"] is True
        assert control.current_winner(group_id)["experiment_id"] == "exp-older"
        assert [int(row["seq"]) for row in control.decision_history(group_id)] == [2, 1]


# ----------------------------------------------------------------------
# Over HTTP, as a client meets it
# ----------------------------------------------------------------------


@pytest.fixture
def server(folder):
    srv = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path=str(folder),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _create_over_http(server, benchmark_id):
    status, payload = _request(
        server, f"/api/benchmarks/{benchmark_id}/experiments", "POST", {"version": "v1"}
    )
    assert status == 201
    return payload["experiment"]["experiment_id"]


def _promote_over_http(server, experiment_id, candidate_id):
    expected = _request(server, f"/api/experiments/{experiment_id}/winner")[1][
        "expected_selection_id"
    ]
    status, payload = _request(
        server, f"/api/experiments/{experiment_id}/winner/decisions", "POST",
        {**HUMAN, "decision": "promote", "expected_selection_id": expected,
         "candidate_experiment_id": candidate_id},
    )
    assert status in (200, 201), payload
    return payload


class TestOverHttp:
    def test_deleting_the_current_winner_is_a_409_that_says_what_to_do(
        self, server, benchmark_id
    ):
        first = _create_over_http(server, benchmark_id)
        _create_over_http(server, benchmark_id)

        status, payload = _request(
            server, f"/api/benchmark-experiments/{first}", "DELETE"
        )

        assert status == 409
        assert "winner" in payload["error"]
        assert "promote one of the other experiments" in payload["error"]
        # Refused means unchanged, on both sides.
        assert _request(server, f"/api/benchmark-experiments/{first}")[0] == 200
        assert _request(server, f"/api/experiments/{first}/winner")[1]["is_winner"]

    def test_the_winner_endpoint_never_names_a_deleted_experiment(
        self, server, benchmark_id
    ):
        first = _create_over_http(server, benchmark_id)
        second = _create_over_http(server, benchmark_id)
        _promote_over_http(server, first, second)

        assert _request(
            server, f"/api/benchmark-experiments/{first}", "DELETE"
        )[0] == 200

        status, payload = _request(server, f"/api/experiments/{second}/winner")
        assert status == 200
        assert payload["is_winner"] is True
        assert payload["winner"]["experiment_id"] == second
        assert [row["experiment_id"] for row in payload["members"]] == [second]
        # The deleted experiment is not a member of anything any more.
        assert _request(server, f"/api/experiments/{first}/winner")[0] == 404

    def test_a_deleted_experiment_cannot_be_promoted_afterwards(
        self, server, benchmark_id
    ):
        """A stale screen promoting a since-deleted candidate gets 404.

        The id is held by nobody now -- `ExperimentNotInGroup` is what the
        membership removal turns a late promotion into, and the API already
        answers that with "no such candidate here" rather than moving anything.
        """
        first = _create_over_http(server, benchmark_id)
        second = _create_over_http(server, benchmark_id)
        third = _create_over_http(server, benchmark_id)
        expected = _request(server, f"/api/experiments/{first}/winner")[1][
            "expected_selection_id"
        ]
        assert _request(
            server, f"/api/benchmark-experiments/{third}", "DELETE"
        )[0] == 200

        status, payload = _request(
            server, f"/api/experiments/{first}/winner/decisions", "POST",
            {**HUMAN, "decision": "promote", "expected_selection_id": expected,
             "candidate_experiment_id": third},
        )

        assert status == 404 and third in payload["error"]
        assert _request(server, f"/api/experiments/{second}/winner")[1][
            "winner"
        ]["experiment_id"] == first

    def test_can_delete_tracks_the_winner_across_a_promotion(
        self, server, benchmark_id
    ):
        """The detail screen offers Delete for exactly what DELETE accepts.

        `can_delete` and `is_winner` come from one calculation, so the button
        cannot appear on the one experiment whose deletion is refused. It is
        the honest answer and not the guard: the winner can move between this
        read and the DELETE, which is what the 409 is still there for.
        """
        first = _create_over_http(server, benchmark_id)
        second = _create_over_http(server, benchmark_id)

        def flags(experiment_id):
            detail = _request(server, f"/api/benchmark-experiments/{experiment_id}")[1]
            return detail["is_winner"], detail["can_delete"]

        assert flags(first) == (True, False)
        assert flags(second) == (False, True)

        _promote_over_http(server, first, second)

        assert flags(first) == (False, True)
        assert flags(second) == (True, False)
        # And the offer is true: the one now marked deletable deletes.
        assert _request(
            server, f"/api/benchmark-experiments/{first}", "DELETE"
        )[0] == 200
        # That leaves the winner alone in its group, which makes it deletable
        # too (`fix-65ik`) -- and the offer is true for it as well.
        assert flags(second) == (True, True)
        assert _request(
            server, f"/api/benchmark-experiments/{second}", "DELETE"
        )[0] == 200

    def test_the_sole_winner_deletes_over_http_and_its_successor_wins(
        self, server, benchmark_id
    ):
        only = _create_over_http(server, benchmark_id)

        assert _request(
            server, f"/api/benchmark-experiments/{only}", "DELETE"
        )[0] == 200

        assert _request(server, f"/api/benchmark-experiments/{only}")[0] == 404
        successor = _create_over_http(server, benchmark_id)
        status, payload = _request(server, f"/api/experiments/{successor}/winner")
        assert status == 200
        assert payload["is_winner"] is True
        assert payload["winner"]["experiment_id"] == successor

    def test_a_decision_in_a_group_with_no_winner_yet_says_why(
        self, server, benchmark_id, folder
    ):
        """`NoCurrentSelection`, still a 409, saying what it means.

        The group is winner-less the way a pre-existing experiment's is: it
        was recorded before contests lived in the live DB and never enrolled.
        """
        store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
        store.create_experiment(
            "exp-older", "recorded before", declared_tasks=1, declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark_id, benchmark_version="v1",
            benchmark_digest_sha256="a" * 64, initialize_winner=False,
        )

        status, payload = _request(
            server, "/api/experiments/exp-older/winner/decisions", "POST",
            {**HUMAN, "decision": "promote", "expected_selection_id": "anything",
             "candidate_experiment_id": "exp-older"},
        )

        assert status == 409, payload
        assert "no current winner yet" in payload["error"]
        assert "enrols it" in payload["error"]

    def test_a_promotion_over_http_enrols_and_a_read_enrols_nothing(
        self, server, benchmark_id, folder
    ):
        """Explicit enrolment of a pre-existing experiment, end to end (§2.3).

        Reading its winner and its runs leaves it outside the contest; the
        promote that says "I saw no winner" is what enrols and elects it.
        """
        store = obs.ObservabilityStore(state_paths.observability_db(str(folder)))
        store.create_experiment(
            "exp-older", "recorded before", declared_tasks=1, declared_attempts=1,
            workflow_name=setup.workflow_name_for(folder),
            benchmark_id=benchmark_id, benchmark_version="v1",
            benchmark_digest_sha256="a" * 64, initialize_winner=False,
        )

        status, read = _request(server, "/api/experiments/exp-older/winner")
        assert status == 200, read
        assert read["winner"] is None
        assert _request(server, "/api/experiments/exp-older/winner/history")[0] == 200
        assert control_module.rows(
            store, "SELECT * FROM comparison_group_members"
        ) == []

        status, payload = _request(
            server, "/api/experiments/exp-older/winner/decisions", "POST",
            {**HUMAN, "decision": "promote", "expected_selection_id": None,
             "candidate_experiment_id": "exp-older"},
        )

        assert status == 201, payload
        assert _control(folder).group_for_experiment("exp-older") is not None
        assert _request(server, "/api/experiments/exp-older/winner")[1][
            "winner"]["experiment_id"] == "exp-older"

    def test_a_recorded_experiment_is_not_deletable_winner_or_not(
        self, server, benchmark_id, folder, tmp_path
    ):
        """The pre-existing reasons to refuse are untouched by the winner one."""
        recorded = setup.create_experiment(folder, benchmark_id, "v1")
        _declare(folder, state_paths.observability_db(str(folder)), recorded)
        empty = _create_over_http(server, benchmark_id)

        detail = _request(
            server, f"/api/benchmark-experiments/{recorded['experiment_id']}"
        )[1]

        assert detail["is_winner"] is True and detail["can_delete"] is False
        assert _request(server, f"/api/benchmark-experiments/{empty}")[1][
            "can_delete"
        ] is True

    def test_the_registration_detail_agrees_with_the_winner_endpoint(
        self, server, benchmark_id
    ):
        first = _create_over_http(server, benchmark_id)
        second = _create_over_http(server, benchmark_id)
        _promote_over_http(server, first, second)
        _request(server, f"/api/benchmark-experiments/{first}", "DELETE")

        detail = _request(server, f"/api/benchmark-experiments/{second}")[1]

        assert detail["is_winner"] is True
        assert detail["winner"]["experiment_id"] == second
