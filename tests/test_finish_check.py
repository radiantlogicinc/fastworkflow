"""The finish-time execution check (fix-4dsr): ledger, verdicts, note, failure modes.

Real archive, real scope, real context clauses; the decision model is replaced
by a scripted client with fixed answers, the same stand-in the search router's
tests use, because what is under test is what the check sends and what it does
with the answers -- the model's own accuracy was measured offline (fix-4dsr).
"""
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastworkflow import tracing
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.finish_check import (
    CHECK_ENV,
    KEY_ENV,
    MIN_ITERS_LEFT,
    NOTE_MAX_BYTES,
    FinishChecker,
    build_ledger,
    checker_from_env,
    compose_note,
)
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.turn_plan import (
    PlanPart,
    PlanStep,
    PlanSubject,
    TurnPlan,
    command_parts,
    parse_text_plan,
    render,
)

UID = "28c5aeb5b64e4ac6c40c57b0235980e2"
LISTING = ("1 identity(s). Each line below is `identity_uid  label`.\n"
           "identity_uid  label\n"
           f"{UID}  Alan Cooper\n")


class _ScriptedClient:
    """Stands in for the decision-model client: answers by question key, records what it was sent."""

    def __init__(self, answer=None, error=None, delay=0.0, max_rows=None):
        self.answer = answer or (lambda key: 0.9)
        self.error, self.delay, self.max_rows = error, delay, max_rows
        self.sent = []

    def system_one(self, *, state, questions):
        self.sent.append((state, dict(questions)))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.max_rows is not None and len(state.get("ledger", [])) > self.max_rows:
            raise RuntimeError('400 {"detail":{"error_type":"max_tokens_exceeded"}}')
        return SimpleNamespace(
            answers={key: SimpleNamespace(noul=self.answer(key)) for key in questions},
            usage=SimpleNamespace(input_tokens=100))


class _RecordingSink:
    def __init__(self):
        self.spans = []

    def emit_span(self, span):
        self.spans.append(span)

    def emit_turn_record(self, record):
        pass

    def record_conversation_label(self, *args):
        pass


def _plan(**overrides):
    steps = overrides.pop("steps", None) or [
        PlanStep(text="Find Alan Cooper", commands=["find_identity"]),
        PlanStep(text="Open Alan Cooper's identity and list his accounts",
                 commands=["open_identity_by_uid", "list_accounts"]),
    ]
    subjects = overrides.pop("subjects", None) or [PlanSubject(name="Alan Cooper", kind="person")]
    return TurnPlan(steps=steps, subjects=subjects, **overrides)


class LedgerFromTheTurn(unittest.TestCase):
    """What the model is shown: one row per step, identifiers resolved, finish left out."""

    def setUp(self):
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / "archive.sqlite3"))
        self.scope = RuntimeHandleScope("store", "channel", "experiment", "task", 1, "turn")
        self.archive.persist(self.scope, alias="O1", offload_order=1, command_name="find_identity",
                             step_index=0, text=LISTING,
                             text_sha256=hashlib.sha256(LISTING.encode()).hexdigest())
        record_context_clause(self.scope, "O1", "DirectoryExplorer", selected_archive=self.archive)
        record_context_clause(self.scope, "O2", "DirectoryExplorer", selected_archive=self.archive)
        record_context_clause(self.scope, "O3", f"Identity {UID} Alan Cooper", selected_archive=self.archive)
        self.agent = SimpleNamespace(
            current_trajectory={
                "tool_name_0": "execute_workflow_query",
                "tool_args_0": {"command": "find_identity <name>Alan Cooper</name>"},
                "observation_0": "Offloaded observation O1 returned by find_identity. It contains 1 row.",
                "tool_name_1": "execute_workflow_query",
                "tool_args_1": {"command": f"open_identity_by_uid <uid>{UID}</uid>"},
                "observation_1": "Entered Identity context.",
                "tool_name_2": "execute_workflow_query",
                "tool_args_2": {"command": "list_accounts"},
                "observation_2": "No accounts found.",
                "tool_name_3": "finish",
                "tool_args_3": {},
                "observation_3": "Completed.",
            },
            execute_ordinal_pairs=lambda trajectory: [(0, 1), (1, 2), (2, 3)],
            continuation_scope=self.scope,
            observation_archive=self.archive,
        )

    def test_rows_resolve_identifiers_and_leave_finish_out(self):
        rows = build_ledger(self.agent, ["Alan Cooper"])
        self.assertEqual([row["n"] for row in rows], [1, 2, 3])
        find, open_identity, accounts = rows
        # The archived text, not the label, is what names_in_output reads.
        self.assertEqual(find["names_in_output"], ["Alan Cooper"])
        self.assertEqual(open_identity["acted_on"], f"Identity {UID} Alan Cooper")
        self.assertIn("Alan Cooper (listed by find_identity", open_identity["refers_to"][UID])
        self.assertEqual(accounts["outcome"], "empty")
        self.assertEqual(accounts["context"], f"Identity {UID} Alan Cooper")

    def test_what_is_sent_passes_the_capture_policy(self):
        secret = "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
        self.agent.current_trajectory["observation_1"] = f"Entered. {secret}"
        rows = build_ledger(self.agent, ["Alan Cooper"])
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", json.dumps(rows))
        self.assertIn("[REDACTED]", rows[1]["head"])

    def test_an_agent_without_offloading_still_gets_a_ledger(self):
        bare = SimpleNamespace(current_trajectory=self.agent.current_trajectory)
        rows = build_ledger(bare, ["Alan Cooper"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["context"], "")


class VerdictsAndNote(unittest.TestCase):
    def setUp(self):
        reset_runtime_state()
        self.ledger = [{"n": 1, "command": "find_identity <name>Alan Cooper</name>", "context": "",
                        "acted_on": "", "refers_to": {}, "names_in_output": ["Alan Cooper"],
                        "outcome": "result", "head": "1 identity(s)."}]

    @staticmethod
    def _step_two_unexecuted(key):
        return 0.1 if key.startswith(("e2_", "g2")) else 0.9

    def test_an_unexecuted_step_is_flagged_for_its_subject(self):
        checker = FinishChecker(_ScriptedClient(self._step_two_unexecuted), model="jev-test")
        result = checker.check(_plan(), "Audit Alan Cooper", self.ledger)
        self.assertIsNone(result.error)
        self.assertEqual([(f["step"], f["subject"]) for f in result.flagged], [(2, "Alan Cooper")])
        self.assertEqual(result.requests, 2)
        note = compose_note(result.flagged, iterations_left=10)
        self.assertIn("step 2 (for Alan Cooper)", note)
        self.assertIn("You have 10 steps left.", note)

    def test_every_step_executed_means_no_flag(self):
        result = FinishChecker(_ScriptedClient()).check(_plan(), "Audit Alan Cooper", self.ledger)
        self.assertEqual(result.flagged, [])

    def test_a_partly_executed_step_is_flagged(self):
        step = PlanStep(text="For Alan Cooper: open the identity and list accounts",
                        parts=[PlanPart(text="open_identity_by_uid", commands=["open_identity_by_uid"]),
                               PlanPart(text="list_accounts", commands=["list_accounts"])])
        plan = _plan(steps=[step])
        answers = lambda key: 0.1 if key.startswith("e1_2_") else 0.9  # noqa: E731
        result = FinishChecker(_ScriptedClient(answers)).check(plan, "Audit Alan Cooper", self.ledger)
        self.assertEqual([(f["step"], f["subject"]) for f in result.flagged], [(1, "Alan Cooper")])

    def test_optional_and_user_gated_steps_are_not_asked_about(self):
        steps = [PlanStep(text="Find Alan Cooper", commands=["find_identity"]),
                 PlanStep(text="Show his portrait", commands=["open_portrait"], optional=True),
                 PlanStep(text="Apply the fix the user chooses", commands=["apply_remediation"], needs_user=True)]
        client = _ScriptedClient()
        FinishChecker(client).check(_plan(steps=steps), "Audit Alan Cooper", self.ledger)
        asked = {key for _state, questions in client.sent for key in questions}
        self.assertIn("x1", asked)
        self.assertFalse({key for key in asked if key[1:2] in "23" and key[0] in "xagep"})

    def test_a_plan_without_subjects_checks_each_step_ran_at_all(self):
        plan = TurnPlan(steps=_plan().steps, subjects=[], source="text")
        answers = lambda key: 0.1 if key == "g2" else 0.9  # noqa: E731
        result = FinishChecker(_ScriptedClient(answers)).check(plan, "Audit", self.ledger)
        self.assertEqual([(f["step"], f["subject"]) for f in result.flagged], [(2, None)])

    def test_an_over_long_ledger_is_halved_and_reduced_with_max(self):
        ledger = [dict(self.ledger[0], n=n) for n in range(1, 9)]
        client = _ScriptedClient(self._step_two_unexecuted, max_rows=2)
        result = FinishChecker(client).check(_plan(), "Audit Alan Cooper", ledger)
        self.assertIsNone(result.error)
        self.assertGreater(result.splits, 0)
        self.assertEqual([(f["step"], f["subject"]) for f in result.flagged], [(2, "Alan Cooper")])

    def test_the_note_is_capped_and_counts_what_it_cuts(self):
        flagged = [{"step": k, "subject": "Someone", "text": "x" * 150, "p_unmet": 0.9} for k in range(1, 30)]
        note = compose_note(flagged, iterations_left=5)
        self.assertLessEqual(len(note.encode("utf-8")), NOTE_MAX_BYTES)
        self.assertIn("more", note)


class FailsOpen(unittest.TestCase):
    """No SDK, no key, no plan, an error or a slow model: no note, the turn finishes unchecked."""

    def setUp(self):
        reset_runtime_state()
        self.agent = SimpleNamespace(current_trajectory={}, plan_source=lambda: _plan())

    def test_a_failing_model_gives_no_note_and_records_why(self):
        checker = FinishChecker(_ScriptedClient(error=TimeoutError()))
        self.assertEqual(checker.note(self.agent, {"user_query": "q"}, iterations_left=10), "")
        event = [e for e in snapshot_events() if e["kind"] == "finish_check"][-1]
        self.assertEqual((event["reason"], event["error_type"]), ("error", "TimeoutError"))

    def test_an_exhausted_time_budget_gives_no_note(self):
        checker = FinishChecker(_ScriptedClient(delay=0.05), budget_seconds=0.01)
        result = checker.check(_plan(), "q", [])
        self.assertEqual((result.error, result.flagged), ("OutOfTime", []))

    def test_no_plan_or_no_room_declines_without_calling_the_model(self):
        client = _ScriptedClient()
        checker = FinishChecker(client)
        self.agent.plan_source = lambda: None
        self.assertEqual(checker.note(self.agent, {}, iterations_left=10), "")
        self.agent.plan_source = lambda: _plan()
        self.assertEqual(checker.note(self.agent, {}, iterations_left=MIN_ITERS_LEFT - 1), "")
        self.assertEqual(client.sent, [])
        reasons = [e["reason"] for e in snapshot_events() if e["kind"] == "finish_check"]
        self.assertEqual(reasons, ["no plan", "no room to act"])

    def test_the_check_is_opt_in(self):
        with patch.dict(os.environ, {CHECK_ENV: "", KEY_ENV: "a-key-present-for-something-else"}), \
                patch.dict("fastworkflow._env_vars", {}, clear=True):
            self.assertIsNone(checker_from_env())
        with patch.dict(os.environ, {CHECK_ENV: "jev", KEY_ENV: ""}), \
                patch.dict("fastworkflow._env_vars", {}, clear=True):
            self.assertIsNone(checker_from_env())

    def test_a_check_is_an_fw_finish_check_span_without_the_request(self):
        sink = _RecordingSink()
        host = SimpleNamespace(trace_sink=sink, current_turn_key="turn-1")
        checker = FinishChecker(_ScriptedClient(lambda key: 0.1 if key.startswith("e2_") else 0.9),
                                model="jev-test")
        with tracing.host_scope(host):
            note = checker.note(self.agent, {"user_query": "Audit the secret list"}, iterations_left=10)
        self.assertIn("step 2", note)
        span = [s for s in sink.spans if s.name == tracing.SPAN_FINISH_CHECK][-1]
        self.assertEqual((span.status, span.attributes["model"], span.attributes["fired"]),
                         (tracing.STATUS_OK, "jev-test", True))
        self.assertNotIn("secret list", json.dumps(span.attributes, default=str))


class StructuredPlan(unittest.TestCase):
    def test_render_is_a_numbered_list_with_sub_steps(self):
        plan = TurnPlan(steps=[
            PlanStep(text="Find the permission", commands=["find_permission"]),
            PlanStep(text="Walk each person", parts=[PlanPart(text="list accounts", commands=["list_accounts"]),
                                                     PlanPart(text="list groups", commands=["list_groups"], optional=True)]),
            PlanStep(text="Apply the fix", commands=["apply_remediation"], needs_user=True),
        ])
        self.assertEqual(render(plan.steps).splitlines(), [
            "1. Find the permission", "2. Walk each person", "   a. list accounts",
            "   b. list groups (optional)", "3. Apply the fix (needs the user)"])
        # One required command sub-step is not a multi-part step.
        self.assertEqual(command_parts(plan.steps[1]), [])

    def test_a_text_plan_is_recovered_without_field_names_or_optional_parts(self):
        text = ("1. **Find Alan** - `find_identity` with his name.\n"
                "2. For each person: a. `open_identity_by_uid` using `identity_uid`. "
                "b. `list_accounts`. c. Optionally `list_groups`.\n"
                "3. **Apply remediation (after confirmation)** - `apply_remediation`.\n")
        plan = parse_text_plan(text, {"find_identity", "open_identity_by_uid", "list_accounts",
                                      "list_groups", "apply_remediation"})
        self.assertEqual((plan.source, len(plan.steps), plan.subjects), ("text", 3, []))
        walk = plan.steps[1]
        self.assertEqual([p.commands for p in walk.parts], [["open_identity_by_uid"], ["list_accounts"], ["list_groups"]])
        self.assertEqual([p.optional for p in walk.parts], [False, False, True])
        self.assertEqual(len(command_parts(walk)), 2)
        self.assertTrue(plan.steps[2].needs_user)


if __name__ == "__main__":
    unittest.main()
