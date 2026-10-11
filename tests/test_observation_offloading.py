"""Arm D observation offloading: compact, search_memory, continuation, span manifest."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow.observation_offloading.agent import (
    build_compacting_step,
    build_tool_agent,
)
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import (
    MIN_OFFLOAD_SAVING_BYTES,
    MIN_OFFLOAD_SAVING_BYTES_ENV,
    PACKED_TARGET_BYTES,
    archive_step,
    compact_trajectory,
    execute_step_indexes,
    min_offload_saving_bytes_from_env,
    packed_target_bytes_from_env,
    step_indexes,
)
from fastworkflow.observation_offloading.labels import (
    LABEL_RESTORE_MARK,
    alias_line,
    estimated_tokens,
    is_offload_label,
    label_alias,
    offload_label,
    offload_saving_bytes,
    printed_alias,
    strip_alias_line,
)
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    observation_row,
    uninstall_span_policy,
)
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.observation_offloading.search import search_memory

from fastworkflow.observation_offloading.state import (
    record_event,
    reset_observation_state,
    snapshot_events,
)
from fastworkflow.answer_rehydration import rehydrated_label
from fastworkflow.observation_offloading.compact import RECENT_OBSERVATIONS_PROTECTED
from fastworkflow.workflow_agent import WorkflowAgentSignature, initialize_workflow_tool_agent

# The budgets derive from the agent model's window; pin it to the reference.
pytestmark = pytest.mark.usefixtures("reference_agent_window")

TODO_WORKFLOW = Path(__file__).parent / "todo_list_workflow"


def compact_completed(trajectory, *, scope, selected_archive, **kwargs):
    """Compact as the agent does: earlier execute steps were archived as they completed."""
    last = max(step_indexes(trajectory), default=0)
    for index in execute_step_indexes(trajectory):
        if index != last and selected_archive.get(scope, f"O{index}") is None:
            archive_step(trajectory, index, scope=scope, selected_archive=selected_archive)
    return compact_trajectory(trajectory, step_index=last, scope=scope,
                              selected_archive=selected_archive, **kwargs)


class CompactTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn",
        )

    def compact(self, trajectory, **kwargs):
        return compact_completed(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            **kwargs,
        )

    def test_offloads_old_large_execute_observation_and_keeps_recent(self) -> None:
        large = ("holder uid label\n" + ("aaaa " * 400)) * 4
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_owners_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + (large if index == 0 else f"small-{index}")
        decisions = self.compact(
            trajectory,
            packed_target_tokens=10,
            recent_observations_protected=5,
        )
        self.assertEqual(decisions[0]["action"], "offloaded")
        self.assertEqual(decisions[0]["alias"], "O0")
        self.assertIn("Offloaded observation O", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_6"], alias_line("O6") + "small-6")

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_search_memory_returns_handle_page_not_full_dump(self) -> None:
        large = "477 holder(s).\nAaron Garrison\n" + ("row\n" * 4000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": alias_line("O0") + large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        self.compact(trajectory, packed_target_tokens=10)
        answer = search_memory(
            "Is Aaron Garrison on the offloaded holders?",
            alias="O0",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertIn("Aaron Garrison", answer)
        self.assertLess(len(answer), len(large))
        self.assertIn("Observation O1", answer)

    def test_default_packed_target_is_28000_utf8_bytes(self) -> None:
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": alias_line("O0") + large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        decisions = self.compact(trajectory)
        self.assertEqual(decisions[0]["action"], "offloaded")
        self.assertEqual(trajectory["observation_5"], alias_line("O5") + "small-5")

    def test_default_recent_observations_protected_is_5(self) -> None:
        self.assertEqual(RECENT_OBSERVATIONS_PROTECTED, 5)
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + large
        decisions = self.compact(trajectory)
        recency = {item["alias"]: item["recency_protected"] for item in decisions}
        self.assertEqual(
            recency,
            {
                "O0": False,
                "O1": False,
                "O2": True,
                "O3": True,
                "O4": True,
                "O5": True,
                "O6": True,
            },
        )
        self.assertIn("Offloaded observation O", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_6"], alias_line("O6") + large)

    def test_default_target_measures_multibyte_utf8_not_characters(self) -> None:
        large = "é" * 15_000
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": alias_line("O0") + large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        decisions = self.compact(trajectory)
        self.assertEqual(len(large.encode("utf-8")), 30_000)
        self.assertEqual(decisions[0]["action"], "offloaded")

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_search_reads_the_stored_row_after_offload(self) -> None:
        large = "restart answer\n" + ("row\n" * 1_500)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": alias_line("O0") + large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        self.compact(trajectory, packed_target_tokens=10)
        answer = search_memory(
            "What exact heading appears at the start of this observation?",
            alias="O0",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertIn("restart answer", answer)

    def test_broken_persistence_retains_original_without_label(self) -> None:
        class BrokenArchive:
            def persist(self, *args, **kwargs):
                raise OSError("disk unavailable")

            def get(self, *args, **kwargs):
                return None

        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": alias_line("O0") + large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        compact_completed(
            trajectory,
            scope=self.scope,
            selected_archive=BrokenArchive(),  # type: ignore[arg-type]
            packed_target_tokens=10,
        )
        # The handle is still printed; only the offload was refused.
        self.assertEqual(trajectory["observation_0"], alias_line("O0") + large)

    def test_scope_prevents_alias_collision(self) -> None:
        other = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="other-turn",
        )
        self.archive.persist(
            self.scope,
            alias="O0",
            command_name="show",
            step_index=0,
            text="first-turn secret",
            text_sha256=hashlib.sha256(b"first-turn secret").hexdigest(),
        )
        answer = search_memory(
            "first-turn secret",
            alias="O0",
            scope=other,
            selected_archive=self.archive,
        )
        self.assertIn("no matching offloaded handle", answer)
        with self.assertRaises(PersistenceError):
            self.archive.persist(
                self.scope,
                alias="O0",
                command_name="show",
                step_index=0,
                text="different text",
                text_sha256=hashlib.sha256(b"different text").hexdigest(),
            )


class fastWorkflowReActBehavior(unittest.TestCase):
    """Max-iters exhaustion and alias = step index."""

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn",
        )

    def test_exhaustion_at_max_iters_extracts_with_exhausted_true(self) -> None:
        class ExhaustAgent(fastWorkflowReAct):
            def _run_loop(self, trajectory, idx, input_args, max_iters, exception_count):
                self.iteration_counter += int(max_iters)
                self._exhausted_last_run = True
                return None

        agent = ExhaustAgent(self.Signature, tools=[], max_iters=25)
        agent.extract = lambda trajectory, **kwargs: {"answer": "done"}
        result = agent.forward(user_query="task")
        self.assertTrue(result.exhausted)

    def test_search_memory_accepts_o0(self) -> None:
        with self.assertRaises(ValueError):
            search_memory("q", alias="O", scope=self.scope, selected_archive=self.archive)
        # O0 is syntactically valid even when the handle is missing.
        search_memory("q", alias="O0", scope=self.scope, selected_archive=self.archive)

    def test_aliases_unchanged_after_truncating_earliest_steps(self) -> None:
        trajectory = {}
        for index in range(3):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"c{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"body-{index}"
        compact_completed(trajectory, scope=self.scope, selected_archive=self.archive)
        self.assertEqual(printed_alias(trajectory["observation_1"]), "O1")
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            trajectory.pop(f"{prefix}_0", None)
        compact_completed(trajectory, scope=self.scope, selected_archive=self.archive)
        self.assertEqual(printed_alias(trajectory["observation_1"]), "O1")
        self.assertEqual(printed_alias(trajectory["observation_2"]), "O2")


class TrajectoryManifest(unittest.TestCase):
    def setUp(self) -> None:
        install_span_policy()
        self.addCleanup(uninstall_span_policy)

    def test_manifest_classifies_resident_labelled_absent(self) -> None:
        resident = "holder uid Jane Roe\n" + ("row\n" * 40)
        raw_offloaded = "permission portrait\n" + ("field\n" * 80)
        label = offload_label(
            alias="O4",
            command_name="Permission/show_owners",
            response=raw_offloaded,
        )
        user = (
            "[[ ## thought_0 ## ]]\nlook\n"
            f"[[ ## observation_0 ## ]]\n{resident}\n"
            f"[[ ## observation_1 ## ]]\n{label}\n"
            "[[ ## next_thought ## ]]\nanswer"
        )
        system = "Available execute_workflow_query tool commands:\n" + ("- cmd\n" * 4000)
        payload = json.dumps(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ensure_ascii=False,
        )
        capped = tracing._capped({"messages": payload, "module": "react"})
        manifest = capped["trajectory_manifest"]
        stored = capped["messages"]
        self.assertTrue(stored["truncated"])
        self.assertEqual(stored["value"], "")
        classified = classify_against_steps(
            manifest,
            {
                0: hashlib.sha256(resident.encode("utf-8")).hexdigest(),
                1: hashlib.sha256(raw_offloaded.encode("utf-8")).hexdigest(),
                2: hashlib.sha256(b"later").hexdigest(),
            },
        )
        self.assertEqual(classified["resident"], [0])
        self.assertEqual(classified["labelled"], [1])
        self.assertEqual(classified["absent"], [2])


class _ManifestTraceSink:
    """Keeps every span it is given, so a test can read the step's own record."""

    def __init__(self) -> None:
        self.spans: list = []

    def emit_span(self, span) -> None:
        self.spans.append(span)


class ManifestAgainstRealSteps(unittest.TestCase):
    """The manifest's alias and residency evidence on the normal path.

    No model is called. The tool is a local function and only the reasoning
    call is scripted; the loop, the ``fw.agent.step`` span, the compaction
    hook, the annotation, the archive and the manifest span policy are all
    production code -- which is the point, because the defect was the ORDER of
    two of them. ``_run_loop`` closes the step span on the raw tool return and
    the completion hook prints the handle line afterwards, so the prompt slot
    and the step's record were never going to be the same bytes, and the
    manifest compared them as if they were.
    """

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn",
        )
        install_span_policy()
        self.addCleanup(uninstall_span_policy)

    def _one_execute_step(self, response: str) -> tuple[str, dict]:
        """Run one real execute step; return its recorded digest and trajectory."""

        def execute_workflow_query(command: str) -> str:
            """Run one command against the workflow."""
            return alias_line("O0") + response

        sink = _ManifestTraceSink()
        host = SimpleNamespace(
            trace_sink=sink,
            current_turn_key="fixture-turn",
            observability_channel_id="fixture-channel",
            observability_experiment_claim={},
            trace_span_stack=[],
        )
        agent = fastWorkflowReAct(
            self.Signature,
            tools=[execute_workflow_query],
            max_iters=3,
            # The production hook, with a caller hook that ends the loop after the
            # first completed step so no second reasoning call is needed.
            on_step_complete=build_compacting_step(
                host,
                selected_archive=self.archive,
                on_step_complete=lambda idx, trajectory: False,
            ),
        )
        agent.react = lambda **kwargs: SimpleNamespace(
            next_thought="read the holders",
            next_tool_name="execute_workflow_query",
            next_tool_args={"command": "Permission/show_owners"},
        )
        trajectory: dict = {}
        with tracing.host_scope(host):
            agent._run_loop(trajectory, 0, {"user_query": "who holds it"}, 3, 0)
        steps = [span for span in sink.spans if span.name == tracing.SPAN_AGENT_STEP]
        self.assertEqual(len(steps), 1)
        recorded = steps[0].attributes["observation"]
        digest = (
            recorded["sha256"] if isinstance(recorded, dict)
            else hashlib.sha256(recorded.encode("utf-8")).hexdigest()
        )
        # What the step evidence IS: the tool return, handle line included.
        self.assertEqual(digest, hashlib.sha256((alias_line("O0") + response).encode("utf-8")).hexdigest())
        return digest, trajectory

    @staticmethod
    def _manifest(slots: dict[int, str]) -> dict:
        """The manifest the span policy builds from a prompt carrying *slots*."""
        body = "[[ ## thought_0 ## ]]\nlook\n"
        for index in sorted(slots):
            body += f"[[ ## observation_{index} ## ]]\n{slots[index]}\n"
        body += "[[ ## next_thought ## ]]\nanswer"
        payload = json.dumps(
            [
                {"role": "system", "content": "Available commands:\n" + ("- cmd\n" * 20)},
                {"role": "user", "content": body},
            ],
            ensure_ascii=False,
        )
        return tracing._capped({"messages": payload, "module": "react"})["trajectory_manifest"]

    def test_an_inline_execute_step_is_aliased_and_resident(self) -> None:
        """The whole finding, on the path that is now the normal one.

        The observation stays inline (one execute step is recency-protected),
        so the manifest sees the annotated slot while the step recorded the raw
        return. Without the normalisation this row reports ``alias=null`` and
        the step's own unchanged evidence comes back ``mismatched``.
        """
        response = "holder uid Jane Roe\n" + ("permission row\n" * 12)
        digest, trajectory = self._one_execute_step(response)
        slot = trajectory["observation_0"]
        self.assertTrue(slot.startswith(alias_line("O0")))

        manifest = self._manifest({0: slot})
        row = manifest["observations"][0]
        self.assertEqual(row["alias"], "O0")
        self.assertEqual(row["alias_source"], "header")
        self.assertEqual(row["kind"], "text")
        # The slot is the step's own bytes: the handle line was in the tool return.
        self.assertEqual(row["sha256"], digest)
        # The response inside it is the same bytes without the handle line.
        self.assertEqual(
            row["response_sha256"], hashlib.sha256(response.encode("utf-8")).hexdigest()
        )
        self.assertEqual(
            classify_against_steps(manifest, {0: digest}),
            {"resident": [0], "labelled": [], "absent": [], "mismatched": []},
        )

    def test_an_offload_label_is_aliased_as_a_pointer_and_never_resident(self) -> None:
        """A label names its alias too, and is still a pointer, not evidence."""
        response = "portrait\n" + ("field\n" * 80)
        digest, _trajectory = self._one_execute_step(response)
        label = offload_label(
            alias="O0", command_name="Permission/show_owners", response=response
        )
        manifest = self._manifest({0: label})
        row = manifest["observations"][0]
        self.assertEqual(row["alias"], "O0")
        self.assertEqual(row["alias_source"], "label")
        self.assertEqual(row["kind"], "label")
        self.assertIsNone(row["response_sha256"])
        self.assertEqual(
            classify_against_steps(manifest, {0: digest}),
            {"resident": [], "labelled": [0], "absent": [], "mismatched": []},
        )

    def test_a_rehydrated_label_is_the_step_evidence_put_back(self) -> None:
        """Rehydration is an intentional transformation of the SLOT, not of the evidence.

        The archived response comes back under a re-printed handle line, so the
        slot digest moves and the response digest does not. The row has to say
        both: the transformation is visible, and the evidence is still the step's.
        """
        response = "holder uid Jane Roe\n" + ("permission row\n" * 12)
        digest, _trajectory = self._one_execute_step(response)
        restored = rehydrated_label("O0", scope=self.scope, archive=self.archive)
        self.assertIsNotNone(restored)

        manifest = self._manifest({0: restored})
        row = manifest["observations"][0]
        self.assertEqual(row["alias"], "O0")
        self.assertEqual(row["alias_source"], "header")
        self.assertEqual(row["sha256"], digest)
        self.assertEqual(classify_against_steps(manifest, {0: digest})["resident"], [0])

    def test_a_rehydrated_listing_carrying_its_stored_rows_is_not_claimed_resident(self) -> None:
        """Evidence that really did change is still mismatched.

        A slot whose text has grown rows beyond what the step returned is more
        than the step's own bytes, so ``mismatched`` is the honest answer. The
        normalisation only covers the case where nothing but our header had
        changed.
        """
        response = "page 1 of holders\n" + ("row\n" * 5)
        digest, _trajectory = self._one_execute_step(response)
        restored = rehydrated_label("O0", scope=self.scope, archive=self.archive)
        with_rows = restored + "row 6\nrow 7\n"

        manifest = self._manifest({0: with_rows})
        row = manifest["observations"][0]
        self.assertEqual(row["alias"], "O0")
        self.assertNotEqual(row["response_sha256"], digest)
        self.assertEqual(
            classify_against_steps(manifest, {0: digest}),
            {"resident": [], "labelled": [], "absent": [], "mismatched": [0]},
        )

    def test_one_normalisation_decides_the_alias_and_the_kind(self) -> None:
        """Pre-existing: the alias was read off a stripped copy and the kind was not.

        A slot with leading whitespace therefore reported an offload label's
        alias while calling itself resident text -- a row that claimed to be
        evidence for a handle whose bytes were in the archive.
        """
        label = offload_label(alias="O3", command_name="show", response="x" * 500)
        row = observation_row("observation_4", "\n  " + label)
        self.assertEqual(row["alias"], "O3")
        self.assertEqual(row["alias_source"], "label")
        self.assertEqual(row["kind"], "label")
        self.assertIsNone(row["response_sha256"])

    def test_an_unannotated_observation_is_still_resident_by_its_slot(self) -> None:
        """A slot this package never touched: one digest, and it is the response.

        Non-execute tools are never annotated, and recordings predating the
        handle line are not either. ``strip_alias_line`` returns such a slot
        unchanged, so the two digests agree and residency is decided from the
        slot's own bytes.
        """
        text = "what_can_i_do listed 4 commands"
        manifest = self._manifest({0: text})
        row = manifest["observations"][0]
        self.assertIsNone(row["alias"])
        self.assertIsNone(row["alias_source"])
        self.assertEqual(row["sha256"], row["response_sha256"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(classify_against_steps(manifest, {0: digest})["resident"], [0])


class TruncationTolerance(unittest.TestCase):
    """execute_step_indexes and aliases after context-window truncation."""

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn",
        )

    @staticmethod
    def _trajectory(count: int) -> dict:
        trajectory = {}
        for index in range(count):
            trajectory[f"thought_{index}"] = f"think-{index}"
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        return trajectory

    def test_execute_indexes_survive_a_missing_leading_step(self) -> None:
        trajectory = self._trajectory(4)
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_0"]
        self.assertEqual(execute_step_indexes(trajectory), [1, 2, 3])

    def test_execute_indexes_skip_non_execute_steps_and_interior_gaps(self) -> None:
        trajectory = self._trajectory(5)
        trajectory["tool_name_1"] = "what_can_i_do"
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_2"]
        self.assertEqual(execute_step_indexes(trajectory), [0, 3, 4])

    def test_compaction_continues_after_truncation_without_alias_collision(self) -> None:
        large_first = "first dump\n" + ("row\n" * 3_000)
        large_second = "second dump\n" + ("col\n" * 3_000)
        large_third = "third dump\n" + ("val\n" * 3_000)
        # Eight executes: the newest five are recency-protected, so O1..O3 are
        # the offloadable slots before truncation and O2..O3 after it. The third
        # dump is present from the start: every execute observation is archived
        # when its step completes, so rewriting a completed step's text under a
        # live alias is a collision, not a fixture shortcut.
        trajectory = self._trajectory(8)
        trajectory["observation_0"] = large_first
        trajectory["observation_1"] = large_second
        trajectory["observation_2"] = large_third
        # A byte target that two offloads satisfy, leaving O3 inline and large.
        first = compact_completed(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            packed_target_bytes=20_000,
            recent_observations_protected=5,
        )
        offloaded = [item["alias"] for item in first if item["action"] == "offloaded"]
        self.assertEqual(offloaded, ["O0", "O1"])

        # The base ReAct context-window fallback drops step 0 entirely; aliases stay O1/O2.
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_0"]
        second = compact_completed(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            packed_target_tokens=10,
            recent_observations_protected=5,
        )
        by_alias = {item["alias"]: item for item in second}
        self.assertEqual(by_alias["O1"]["reason"], "already_label")
        self.assertEqual(by_alias["O2"]["action"], "offloaded")
        stored = {row["alias"]: row["text"] for row in self.archive.list(self.scope)}
        self.assertEqual(stored["O0"], large_first)
        self.assertEqual(stored["O1"], large_second)
        self.assertTrue(stored["O2"].startswith("third dump"))


class PerTurnScope(unittest.TestCase):
    """Each turn's observations are filed under that turn's scope."""

    @staticmethod
    def _scope(turn_key: str) -> RuntimeHandleScope:
        return RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key=turn_key,
        )

    def setUp(self) -> None:
        reset_observation_state()

    def test_two_turns_offload_their_own_O0_into_one_archive(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        archive = RuntimeHandleArchive(str(Path(tempdir.name) / "handles.sqlite3"))
        turn = {"key": "turn-1"}
        step = build_compacting_step(object(), selected_archive=archive)
        texts = {}
        with patch("fastworkflow.observation_offloading.agent.scope_for_host",
                   side_effect=lambda host: self._scope(turn["key"])):
            for turn["key"] in ("turn-1", "turn-2"):
                text = f"{turn['key']} dump\n" + ("row\n" * 3_000)
                texts[turn["key"]] = text
                trajectory = {}
                for index in range(7):
                    trajectory[f"tool_name_{index}"] = "execute_workflow_query"
                    trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
                    trajectory[f"observation_{index}"] = text if index == 0 else "small"
                trajectory["observation_1"] = "z" * 30_000
                for index in range(7):
                    self.assertTrue(step(index, trajectory))
                self.assertIn("Offloaded observation O", trajectory["observation_0"])
        for turn_key, text in texts.items():
            handle = archive.get(self._scope(turn_key), "O0")
            self.assertEqual(handle["text"], text)


class HookIsolation(unittest.TestCase):
    """Review finding: a failure inside the step hook aborted the whole turn."""

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="fixture-turn",
        )

    def _set_env(self, name: str, value: str) -> None:
        previous = os.environ.get(name)

        def restore() -> None:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addCleanup(restore)
        os.environ[name] = value

    def test_compaction_failure_is_recorded_and_the_step_continues(self) -> None:
        seen = []
        step = build_compacting_step(
            object(),
            selected_archive=self.archive,
            on_step_complete=lambda idx, trajectory: seen.append(idx) or True,
        )
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show"},
            "observation_0": "x" * 30_000,
        }
        with patch(
            "fastworkflow.observation_offloading.agent.scope_for_host",
            return_value=self.scope,
        ), patch(
            "fastworkflow.observation_offloading.agent.compact_trajectory",
            side_effect=ValueError("broken compaction"),
        ):
            self.assertTrue(step(0, trajectory))
        self.assertEqual(seen, [0])
        self.assertEqual(trajectory["observation_0"], "x" * 30_000)
        failures = [e for e in snapshot_events() if e["kind"] == "compaction_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error"], "ValueError")

    def test_prompt_overhead_leaves_the_trajectory_less_room(self) -> None:
        """The budget bounds the whole executor prompt, not only the trajectory."""
        large = alias_line("O0") + ("aaaa " * 2_000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        archive_step(trajectory, 0, scope=self.scope, selected_archive=self.archive)

        def offloaded_with_overhead(overhead: int) -> bool:
            agent = SimpleNamespace(inputs={}, prompt_overhead_bytes=lambda inputs: overhead)
            step = build_compacting_step(SimpleNamespace(workflow_tool_agent=agent),
                                         selected_archive=self.archive)
            view = dict(trajectory)
            with patch("fastworkflow.observation_offloading.agent.scope_for_host",
                       return_value=self.scope), \
                 patch("fastworkflow.context_budget.trajectory_max_bytes", return_value=40_000):
                step(5, view)
            return view["observation_0"] != large

        self.assertFalse(offloaded_with_overhead(0))
        self.assertTrue(offloaded_with_overhead(35_000))

    def test_malformed_numeric_override_falls_back_to_the_derived_budget(self) -> None:
        self._set_env("FW_TRAJECTORY_MAX_BYTES", "28k")
        self.assertEqual(packed_target_bytes_from_env(), PACKED_TARGET_BYTES)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show"},
            "observation_0": "holder\n" + "x" * 30_000,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = alias_line(f"O{index}") + f"small-{index}"
        decisions = compact_completed(
            trajectory, scope=self.scope, selected_archive=self.archive
        )
        self.assertEqual(decisions[0]["action"], "offloaded")

    def test_unwritable_event_store_does_not_raise(self) -> None:
        # A real SQLite failure: the event database path names a directory.
        self.archive.db_path = self.tempdir.name
        record_event({"kind": "probe"},
                     scope=self.scope, store=self.archive)
        record_event({"kind": "probe-again"},
                     scope=self.scope, store=self.archive)
        self.assertEqual(
            [e["kind"] for e in snapshot_events()], ["probe", "probe-again"]
        )


class AgentConstruction(unittest.TestCase):
    """Review nit: the wrapper discarded a fully built ReAct and rebuilt it."""

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        # The agent's archive now lands beside the workflow's observability DB;
        # the state root is what moves it into this test's temp dir.
        self._set_env("FASTWORKFLOW_STATE_ROOT", self.tempdir.name)

    def _set_env(self, name: str, value: str) -> None:
        previous = os.environ.get(name)

        def restore() -> None:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addCleanup(restore)
        os.environ[name] = value

    @staticmethod
    def noop_tool(command: str) -> str:
        """Return the command unchanged."""
        return command

    def test_the_agent_is_a_continuation_agent(self) -> None:
        """No setting reaches this; it is what build_tool_agent does."""
        agent = build_tool_agent(
            SimpleNamespace(), self.Signature, [self.noop_tool], max_iters=3
        )
        self.assertIsInstance(agent, fastWorkflowReAct)
        self.assertEqual(set(agent.tools), {"noop_tool", "finish"})
        self.assertEqual(agent.max_iters, 3)
        installed = [e for e in snapshot_events() if e["kind"] == "agent_installed"]
        self.assertEqual(len(installed), 1)
        self.assertNotIn(
            "evaluation_controls", [event["kind"] for event in snapshot_events()]
        )

    def test_the_workflow_agent_does_not_offer_search_memory(self) -> None:
        # Disabled 2026-10-09; the function is kept for later.
        agent = initialize_workflow_tool_agent(SimpleNamespace())
        self.assertNotIn("search_memory", agent.tools)

    def test_multiple_agents_share_one_idempotent_manifest_enricher(self) -> None:
        uninstall_span_policy()
        self.addCleanup(uninstall_span_policy)
        first = build_tool_agent(
            SimpleNamespace(), self.Signature, [self.noop_tool], max_iters=3)
        second = build_tool_agent(
            SimpleNamespace(), self.Signature, [self.noop_tool], max_iters=3)
        self.assertIsInstance(first, fastWorkflowReAct)
        self.assertIsInstance(second, fastWorkflowReAct)
        payload = json.dumps([{
            "role": "user",
            "content": "[[ ## observation_0 ## ]]\none\n[[ ## answer ## ]]\ndone",
        }])
        capped = tracing._capped({"messages": payload})
        self.assertEqual(capped["trajectory_manifest"]["observation_count"], 1)

class PrintedObservationHandles(unittest.TestCase):
    """The canonical O alias is printed on every execute result.

    When an alias was only ever visible on an offload label, agents asked
    search_memory for ReAct step numbers instead of aliases and got the wrong
    observation back. These checks pin the printed identifier to exactly what
    the step index assigns, and keep archived text free of it.
    """

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.addCleanup(reset_observation_state)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self._turns = 0
        self.scope = self.new_scope()

    def new_scope(self) -> RuntimeHandleScope:
        """A fresh turn. One text per alias per scope, so each case needs its own."""
        self._turns += 1
        return RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key=f"fixture-turn-{self._turns}",
        )

    def describe(self, command: str, response: str) -> str:
        return self.DESCRIPTION if command.startswith("list_permissions") else ""

    def compact(self, trajectory, **kwargs):
        return compact_completed(
            trajectory, scope=self.scope, selected_archive=self.archive, **kwargs
        )

    @staticmethod
    def _step(trajectory, index, tool, observation, command=None):
        trajectory[f"thought_{index}"] = f"think-{index}"
        trajectory[f"tool_name_{index}"] = tool
        if command is not None:
            trajectory[f"tool_args_{index}"] = {"command": command}
        if tool == "execute_workflow_query":
            observation = alias_line(f"O{index}") + observation
        trajectory[f"observation_{index}"] = observation

    def test_interleaved_tools_number_executes_only(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "holders page", command="show_owners")
        self._step(trajectory, 1, "search_memory", "Observation O1:\nan answer")
        self._step(trajectory, 2, "ask_user", "the user replied")
        self._step(trajectory, 3, "what_can_i_do", "available command metadata")
        self._step(trajectory, 4, "execute_workflow_query", "rights page", command="show_rights")
        self.compact(trajectory)
        self.assertEqual(trajectory["observation_0"], alias_line("O0") + "holders page")
        self.assertEqual(trajectory["observation_4"], alias_line("O4") + "rights page")
        # A non-execute tool output never acquires an O alias; the second execute
        # at step index 4 is O4, not a recount as O1.
        for index in (1, 2, 3):
            self.assertIsNone(printed_alias(trajectory[f"observation_{index}"]))
        self.assertNotIn("O1", trajectory["observation_4"])

    def test_inline_alias_equals_the_label_and_archive_alias_after_offload(self) -> None:
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", large, command="show_owners")
        for index in range(1, 6):
            self._step(trajectory, index, "execute_workflow_query", f"small-{index}",
                       command=f"find_{index}")
        # Step 0 is recency-protected here, so the agent reads it inline first.
        self.compact(trajectory, recent_observations_protected=6)
        seen_inline = printed_alias(trajectory["observation_0"])
        self.assertEqual(seen_inline, "O0")

        decisions = self.compact(trajectory, recent_observations_protected=5)
        self.assertEqual(decisions[0]["action"], "offloaded")
        self.assertEqual(label_alias(trajectory["observation_0"]), seen_inline)
        row = self.archive.get(self.scope, seen_inline)
        self.assertIsNotNone(row)
        # Archived text is the original response: no alias line, so its digest
        # is comparable with observations recorded before aliases were printed.
        self.assertEqual(row["text"], large)
        self.assertEqual(
            row["text_sha256"], hashlib.sha256(large.encode("utf-8")).hexdigest()
        )
        self.assertIsNone(printed_alias(row["text"]))

    def test_printed_alias_is_what_search_memory_resolves(self) -> None:
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", large, command="show_owners")
        self._step(trajectory, 1, "what_can_i_do", "metadata")
        for index in range(2, 7):
            self._step(trajectory, index, "execute_workflow_query", f"small-{index}",
                       command=f"find_{index}")
        self.compact(trajectory)
        alias = label_alias(trajectory["observation_0"])
        self.assertEqual(alias, "O0")
        # search_memory accepts the printed alias (validated before any model
        # call) and resolves it to the exact original text.
        with self.assertRaises(ValueError):
            search_memory("", alias, scope=self.scope, selected_archive=self.archive)
        self.assertEqual(self.archive.get(self.scope, alias)["text"], large)
        # The step number of the newest execute is 6; its canonical alias is O6.
        self.assertEqual(printed_alias(trajectory["observation_6"]), "O6")
        # A step number that is not a live alias stays an explicit miss: no
        # nearest-handle fallback, no model call.
        miss = search_memory(
            "Which control names the identity?", "O7",
            scope=self.scope, selected_archive=self.archive,
        )
        self.assertIn("no matching offloaded handle O7", miss)
        self.assertEqual(
            [e["status"] for e in snapshot_events() if e["kind"] == "search_memory"],
            ["missing"],
        )

    def test_label_still_saves_space_against_the_printed_observation(self) -> None:
        body = "holder uid label\n" + ("x" * 30_000)
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", body, command="show_owners")
        for index in range(1, 6):
            self._step(trajectory, index, "execute_workflow_query", f"small-{index}",
                       command=f"find_{index}")
        self.compact(trajectory)
        label = trajectory["observation_0"]
        printed = alias_line("O0") + body
        self.assertTrue(is_offload_label(label))
        self.assertLess(len(label), len(printed))
        self.assertLess(len(label.encode("utf-8")), len(printed.encode("utf-8")))
        # And against the response alone, so the added line can never be what
        # makes an offload look profitable.
        self.assertLess(len(label.encode("utf-8")), len(body.encode("utf-8")))

    def test_decision_sizes_and_eligibility_ignore_the_printed_line(self) -> None:
        body = "y" * 4_000
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", body, command="show_owners")
        decisions = self.compact(trajectory, recent_observations_protected=0,
                                 packed_target_tokens=1)
        self.assertEqual(decisions[0]["response_size"]["characters"], len(body))
        self.assertEqual(decisions[0]["response_size"]["utf8_bytes"], len(body.encode("utf-8")))
        # The saving is measured against the response too, so the ~34 B alias
        # line can neither create eligibility nor inflate the reported saving.
        label = trajectory["observation_0"]
        self.assertTrue(is_offload_label(label))
        self.assertEqual(
            decisions[0]["offload_saving_bytes"],
            len(body.encode("utf-8")) - len(label.encode("utf-8")),
        )
        self.assertEqual(decisions[0]["label_size"]["utf8_bytes"],
                         len(label.encode("utf-8")))
        self.assertEqual(self.archive.get(self.scope, "O0")["text"], body)

    def test_error_and_empty_execute_observations_are_still_addressable(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query",
                   "Execution error in execute_workflow_query: boom", command="show_owners")
        self._step(trajectory, 1, "execute_workflow_query", "", command="show_rights")
        self.compact(trajectory)
        self.assertEqual(printed_alias(trajectory["observation_0"]), "O0")
        self.assertEqual(trajectory["observation_1"], alias_line("O1"))


class EagerObservationArchive(unittest.TestCase):
    """Every execute observation is durable when its step ends.

    Before this, only an observation compaction chose to replace was persisted,
    so an alias the run had just printed resolved to "no matching offloaded
    handle" while its text was sitting in the prompt. Offloading is now purely a
    residency decision; availability is unconditional.
    """

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.addCleanup(reset_observation_state)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self._turns = 0
        self.scope = self.new_scope()

    def new_scope(self) -> RuntimeHandleScope:
        """A fresh turn. One text per alias per scope, so each case needs its own."""
        self._turns += 1
        return RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key=f"fixture-turn-{self._turns}",
        )

    def describe(self, command: str, response: str) -> str:
        return self.DESCRIPTION if command.startswith("list_permissions") else ""

    def compact(self, trajectory, **kwargs):
        return compact_completed(
            trajectory, scope=self.scope, selected_archive=self.archive, **kwargs
        )

    @staticmethod
    def _step(trajectory, index, tool, observation, command=None):
        trajectory[f"thought_{index}"] = f"think-{index}"
        trajectory[f"tool_name_{index}"] = tool
        if command is not None:
            trajectory[f"tool_args_{index}"] = {"command": command}
        if tool == "execute_workflow_query":
            observation = alias_line(f"O{index}") + observation
        trajectory[f"observation_{index}"] = observation

    def _search(self, question, alias, *, scope=None, **kwargs):
        """search_memory with a deterministic stand-in for the search model.

        Returns the observation text the model was actually handed, so an
        inline answer and an offloaded answer can be compared byte for byte
        without a provider call.
        """
        seen: dict = {"observation": None}
        lm = SimpleNamespace(
            history=[{"usage": {"completion_tokens": 7}, "cost": 0.0}], model="fixture-lm"
        )

        def predict(_signature):
            def call(question, subject, observation, **_):
                seen["observation"] = observation
                seen["subject"] = subject
                return SimpleNamespace(answer=f"observed {len(observation.encode('utf-8'))} bytes")
            return call

        with patch("fastworkflow.observation_offloading.search.get_lm", return_value=lm) as get_lm, \
                patch("fastworkflow.observation_offloading.search.dspy") as fake_dspy:
            fake_dspy.Predict.side_effect = predict
            seen["answer"] = search_memory(
                question, alias, scope=scope or self.scope,
                selected_archive=self.archive, **kwargs,
            )
            seen["model_calls"] = get_lm.call_count
        seen["event"] = [e for e in snapshot_events() if e["kind"] == "search_memory"][-1]
        return seen

    def test_every_execute_observation_is_archived_whether_or_not_it_is_offloaded(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "holder rows", command="show_owners")
        self._step(trajectory, 1, "what_can_i_do", "command metadata")
        self._step(trajectory, 2, "execute_workflow_query", "rights rows", command="show_rights")
        decisions = self.compact(trajectory)
        # Nothing is big enough to offload; everything is still searchable.
        self.assertEqual([item["action"] for item in decisions], ["kept", "kept"])
        self.assertEqual(printed_alias(trajectory["observation_0"]), "O0")
        self.assertEqual(printed_alias(trajectory["observation_2"]), "O2")
        stored = {row["alias"]: row["text"] for row in self.archive.list(self.scope)}
        self.assertEqual(stored, {"O0": "holder rows", "O2": "rights rows"})
        # The non-execute observation has no handle and is not archived.
        self.assertNotIn("command metadata", stored.values())
        for alias, text in stored.items():
            self.assertEqual(
                self.archive.get(self.scope, alias)["text_sha256"],
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            self.assertIsNone(printed_alias(text))

    def test_search_answers_from_the_same_text_inline_or_offloaded(self) -> None:
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", large, command="show_owners")
        for index in range(1, 6):
            self._step(trajectory, index, "execute_workflow_query", f"small-{index}",
                       command=f"find_{index}")
        # Recency-protected: the agent can read O1 in its prompt right now.
        self.compact(trajectory, recent_observations_protected=6)
        self.assertEqual(printed_alias(trajectory["observation_0"]), "O0")
        inline_row = self.archive.get(self.scope, "O0")
        self.assertEqual(inline_row["text"], large)

        inline = self._search("Which holder?", "O0")
        # F12: what the search model is given is the evidence cut to its own
        # budget, not the whole 30 KB. The point of this test is that the cut is
        # the same read inline and offloaded, which is asserted below.
        self.assertEqual(inline["observation"], large)
        self.assertEqual(inline["event"]["text_sha256"], inline_row["text_sha256"])

        # Now let compaction offload the same observation.
        decisions = self.compact(trajectory, recent_observations_protected=5)
        self.assertEqual(decisions[0]["action"], "offloaded")
        offloaded_row = self.archive.get(self.scope, "O0")
        self.assertEqual(offloaded_row["text"], inline_row["text"])
        self.assertEqual(offloaded_row["text_sha256"], inline_row["text_sha256"])
        self.assertEqual(len(self.archive.list(self.scope, "O0")), 1)

        offloaded = self._search("Which holder?", "O0")
        self.assertEqual(offloaded["observation"], inline["observation"])
        self.assertEqual(offloaded["answer"], inline["answer"])
        self.assertEqual(offloaded["event"]["text_sha256"], inline["event"]["text_sha256"])

    def test_repeated_persistence_keeps_one_row_with_the_same_digest(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "holder rows", command="show_owners")
        for _ in range(4):
            self.compact(trajectory)
            self._step(trajectory, len(trajectory) // 4, "execute_workflow_query",
                       "more rows", command="show_more")
        rows = self.archive.list(self.scope, "O0")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "holder rows")
        self.assertEqual(
            rows[0]["text_sha256"],
            hashlib.sha256(b"holder rows").hexdigest(),
        )
        # One archive event per alias: revisiting a step writes nothing.
        archived = [e for e in snapshot_events() if e["kind"] == "observation_archived"]
        self.assertEqual(len([e for e in archived if e["alias"] == "O0"]), 1)

    def test_another_turns_alias_is_not_visible(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "first-turn holders",
                   command="show_owners")
        self.compact(trajectory)
        other = RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key="another-turn",
        )
        self.assertIsNone(self.archive.get(other, "O0"))
        miss = self._search("Which holders?", "O0", scope=other)
        self.assertIn("no matching offloaded handle O0", miss["answer"])
        self.assertEqual(miss["model_calls"], 0)

    def test_persistence_failure_keeps_inline_evidence_and_records_the_event(self) -> None:
        class BrokenArchive:
            def persist(self, *args, **kwargs):
                raise OSError("disk unavailable")

            def get(self, *args, **kwargs):
                return None

        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "holder rows", command="show_owners")
        decisions = compact_completed(
            trajectory,
            scope=self.scope,
            selected_archive=BrokenArchive(),  # type: ignore[arg-type]
        )
        # The turn continues: the observation keeps its handle and its text.
        self.assertEqual(trajectory["observation_0"], alias_line("O0") + "holder rows")
        self.assertEqual(decisions[0]["action"], "kept")
        refused = [e for e in snapshot_events() if e["kind"] == "archive_refused"]
        self.assertEqual(
            [(e["alias"], e["reason"], e["error"]) for e in refused],
            [("O0", "persistence_failed_original_retained", "OSError")],
        )
        self.assertIsNone(self.archive.get(self.scope, "O0"))

    def test_rewritten_text_under_a_live_alias_is_refused_not_overwritten(self) -> None:
        trajectory: dict = {}
        self._step(trajectory, 0, "execute_workflow_query", "holder rows", command="show_owners")
        self.compact(trajectory)
        # An observation is immutable once its step completed; a different text
        # under the same alias must never replace the stored evidence.
        trajectory["observation_0"] = alias_line("O0") + "rewritten rows"
        self.compact(trajectory)
        self.assertEqual(self.archive.get(self.scope, "O0")["text"], "holder rows")
        refused = [e for e in snapshot_events() if e["kind"] == "archive_refused"]
        self.assertEqual([e["error"] for e in refused], ["PersistenceError"])
        self.assertEqual(trajectory["observation_0"], alias_line("O0") + "rewritten rows")

    def test_wrong_handle_stays_an_explicit_miss_with_no_model_call(self) -> None:
        trajectory: dict = {}
        for index in range(3):
            self._step(trajectory, index, "execute_workflow_query", f"body-{index}",
                       command=f"c{index}")
        self.compact(trajectory)
        # O4 was never printed: no step-number fallback, no nearest handle.
        miss = self._search("Which control?", "O3")
        self.assertIn("no matching offloaded handle O3", miss["answer"])
        self.assertEqual(miss["model_calls"], 0)
        self.assertEqual(miss["event"]["status"], "missing")
        # Every printed handle, by contrast, resolves.
        for alias in ("O0", "O1", "O2"):
            self.assertIsNotNone(self.archive.get(self.scope, alias))

class MinimumOffloadSaving(unittest.TestCase):
    """Eligibility is what the swap saves, not how big the text is.

    The old floor asked whether an observation was over 1,000 estimated tokens
    (~4 KB of ASCII). It therefore kept every 1.3-4 KB listing page resident for
    a whole turn although its label costs a few hundred bytes, and it could in
    principle have offered to replace a 300 B fact with a 400 B pointer to it.
    The rule here is the one thing offloading actually buys:

        utf8(response, alias line stripped) - utf8(that step's real label) >= 1024
    """

    #: A fixed first line keeps output_description's heading -- and so the
    #: label -- the same length however long the body is, which is what makes
    #: an exact byte saving constructible.
    HEAD = "holder uid label\n"
    #: A realistic authored Output description. The measured corpus
    #: (evaluation/artifacts/result-search/c1-prep) shows real labels running
    #: 160-755 B, mostly because of these, so a fixture with no description at
    #: all would make every page look cheaper to offload than it is.
    DESCRIPTION = (
        "permission_uids: The permission_uid of every permission listed, in the "
        "order shown. Pass one to a command that asks for a permission_uid; "
        "labels: Human-readable name of each permission, aligned by index with "
        "permission_uids; total: How many permissions the backend reported."
    )

    def setUp(self) -> None:
        reset_observation_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.addCleanup(reset_observation_state)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self._turns = 0
        self.scope = self.new_scope()

    def new_scope(self) -> RuntimeHandleScope:
        """A fresh turn. One text per alias per scope, so each case needs its own."""
        self._turns += 1
        return RuntimeHandleScope(
            channel_id="fixture-channel",
            turn_key=f"fixture-turn-{self._turns}",
        )

    def describe(self, command: str, response: str) -> str:
        return self.DESCRIPTION if command.startswith("list_permissions") else ""

    def compact(self, trajectory, **kwargs):
        return compact_completed(
            trajectory, scope=self.scope, selected_archive=self.archive, **kwargs
        )

    def label_for(self, body, *, alias="O0", command="show_owners", description=""):
        return offload_label(alias=alias, command_name=command, response=body,
                             description=description)

    def body_saving(self, saving, *, alias="O0", command="show_owners", description=""):
        """A response whose real label frees exactly ``saving`` UTF-8 bytes."""
        probe = self.HEAD + "x" * 4_000
        label = self.label_for(probe, alias=alias, command=command, description=description)
        body = self.HEAD + "x" * (len(label.encode("utf-8")) + saving
                                  - len(self.HEAD.encode("utf-8")))
        self.assertEqual(
            offload_saving_bytes(body, self.label_for(body, alias=alias, command=command,
                                                      description=description)),
            saving,
        )
        return body

    def one_step(self, body, *, command="show_owners"):
        return {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": command},
            "observation_0": body,
        }

    def decide(self, body, **kwargs):
        """One unprotected execute observation, with the target already exceeded."""
        self.scope = self.new_scope()
        trajectory = self.one_step(body)
        decisions = self.compact(trajectory, recent_observations_protected=0,
                                 packed_target_tokens=1, **kwargs)
        return trajectory, decisions[0]

    # -- the boundary ------------------------------------------------------

    def test_default_minimum_saving_is_1024_utf8_bytes(self) -> None:
        self.assertEqual(MIN_OFFLOAD_SAVING_BYTES, 1_024)
        self.assertEqual(min_offload_saving_bytes_from_env(), 1_024)

    def test_1023_bytes_of_saving_is_kept_and_1024_is_offloaded(self) -> None:
        trajectory, decision = self.decide(self.body_saving(1_023))
        self.assertEqual(decision["action"], "kept")
        self.assertEqual(decision["reason"], "below_min_saving")
        self.assertEqual(decision["offload_saving_bytes"], 1_023)
        self.assertEqual(decision["min_offload_saving_bytes"], 1_024)
        self.assertFalse(is_offload_label(trajectory["observation_0"]))

        trajectory, decision = self.decide(self.body_saving(1_024))
        self.assertEqual(decision["action"], "offloaded")
        self.assertEqual(decision["reason"], "oldest_eligible_until_target")
        self.assertEqual(decision["offload_saving_bytes"], 1_024)
        self.assertTrue(is_offload_label(trajectory["observation_0"]))

    def test_1025_bytes_of_saving_is_offloaded(self) -> None:
        trajectory, decision = self.decide(self.body_saving(1_025))
        self.assertEqual(decision["action"], "offloaded")
        self.assertEqual(decision["offload_saving_bytes"], 1_025)
        self.assertTrue(is_offload_label(trajectory["observation_0"]))

    def test_the_old_token_floor_no_longer_decides_anything(self) -> None:
        """4,000 characters was exactly at the old floor and never offloaded."""
        body = self.HEAD + "x" * (4_000 - len(self.HEAD))
        self.assertEqual(estimated_tokens(body), 1_000)  # not > 1000: old = kept
        trajectory, decision = self.decide(body)
        self.assertEqual(decision["action"], "offloaded")
        self.assertGreaterEqual(decision["offload_saving_bytes"], 1_024)
        self.assertEqual(decision["response_size"]["estimated_tokens"], 1_000)

    def test_the_size_record_still_reports_estimated_tokens(self) -> None:
        body = self.body_saving(2_000)
        _trajectory, decision = self.decide(body)
        self.assertEqual(decision["response_size"],
                         {"characters": len(body),
                          "utf8_bytes": len(body.encode("utf-8")),
                          "estimated_tokens": estimated_tokens(body)})

    # -- the label is this step's real label -------------------------------

    def test_the_saving_uses_this_steps_label_not_a_constant(self) -> None:
        """Same bytes of output, two labels: only the cheaper label offloads."""
        body = self.body_saving(1_024)
        long_command = "who_has_access_to <type>permission</type> " + "a" * 200
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_owners"},
            "observation_0": body,
            "tool_name_1": "execute_workflow_query",
            "tool_args_1": {"command": long_command},
            "observation_1": body,
        }
        decisions = self.compact(trajectory, recent_observations_protected=0,
                                 packed_target_tokens=1)
        by_alias = {d["alias"]: d for d in decisions}
        self.assertEqual(by_alias["O0"]["action"], "offloaded")
        self.assertEqual(by_alias["O0"]["offload_saving_bytes"], 1_024)
        # The longer command travels in the label, so the same text saves less.
        self.assertEqual(by_alias["O1"]["action"], "kept")
        self.assertEqual(by_alias["O1"]["reason"], "below_min_saving")
        self.assertEqual(
            by_alias["O1"]["offload_saving_bytes"],
            1_024 - (len(long_command.encode("utf-8")) - len("show_owners")),
        )
        self.assertEqual(by_alias["O1"]["label_size"]["utf8_bytes"],
                         len(self.label_for(body, alias="O1",
                                            command=long_command).encode("utf-8")))

    def test_an_authored_description_lengthens_the_label_and_the_decision(self) -> None:
        """describe_output feeds the same label the offload would write."""
        description = "identity_uids: every holder listed; labels: their names"
        body = self.body_saving(1_024, description=description)
        trajectory, decision = self.decide(
            body, describe_output=lambda command, response: description
        )
        self.assertEqual(decision["action"], "offloaded")
        self.assertEqual(decision["offload_saving_bytes"], 1_024)
        self.assertIn(description, trajectory["observation_0"])
        # Without the description the label is shorter, so the same body saves more.
        _trajectory, plain = self.decide(body)
        self.assertGreater(plain["offload_saving_bytes"], 1_024)

    # -- pages, facts, recency ---------------------------------------------

    def _page_trajectory(self, oldest: str) -> dict:
        """The oldest execute observation, then five newer ones; over target."""
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "list_permissions"},
            "observation_0": oldest,
        }
        for index in range(1, 5):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"open_portrait_{index}"}
            trajectory[f"observation_{index}"] = f"Entered context {index}."
        # A fresh 30 KB result is what pushes the packed trajectory over target.
        trajectory["tool_name_5"] = "execute_workflow_query"
        trajectory["tool_args_5"] = {"command": "show_owners"}
        trajectory["observation_5"] = "holder uid label\n" + "z" * 30_000
        return trajectory

    def test_a_3kb_page_older_than_the_protected_five_is_offloaded(self) -> None:
        page = "permission_uid  label\n" + ("row of a listing page\n" * 140)
        self.assertGreater(len(page.encode("utf-8")), 3_000)
        self.assertLess(len(page.encode("utf-8")), 4_000)
        self.assertLessEqual(estimated_tokens(page), 1_000)  # the old rule kept it
        trajectory = self._page_trajectory(page)
        decisions = self.compact(trajectory, describe_output=self.describe)
        by_alias = {d["alias"]: d for d in decisions}
        self.assertEqual(by_alias["O0"]["action"], "offloaded")
        self.assertEqual(label_alias(trajectory["observation_0"]), "O0")
        # And the newest five are still there, 30 KB result included.
        self.assertEqual(strip_alias_line(trajectory["observation_5"]),
                         "holder uid label\n" + "z" * 30_000)
        for index in range(1, 5):
            self.assertTrue(by_alias[f"O{index + 1}"]["recency_protected"])

    def test_a_12kb_observation_is_not_offloaded(self) -> None:
        small = "permission_uid  label\n" + ("row of a listing page\n" * 54)
        self.assertGreater(len(small.encode("utf-8")), 1_150)
        self.assertLess(len(small.encode("utf-8")), 1_300)
        trajectory = self._page_trajectory(small)
        decisions = self.compact(trajectory, describe_output=self.describe)
        decision = {d["alias"]: d for d in decisions}["O0"]
        self.assertEqual(decision["action"], "kept")
        self.assertEqual(decision["reason"], "below_min_saving")
        self.assertLess(decision["offload_saving_bytes"], 1_024)
        self.assertEqual(strip_alias_line(trajectory["observation_0"]), small)

    def test_short_facts_are_untouched_even_over_target(self) -> None:
        """Facts older than the protected five are kept: their label is bigger."""
        facts = ["Entered Permission context.", "Context is now '*'", "",
                 "1 permission(s). Each line below is `permission_uid  label`."]
        page = "permission_uid  label\n" + ("row of a listing page\n" * 140)
        trajectory = self._page_trajectory(page)
        for offset, fact in enumerate(facts):
            index = 6 + offset
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"reset_context_{offset}"}
            trajectory[f"observation_{index}"] = fact
        decisions = self.compact(trajectory, describe_output=self.describe)
        by_alias = {d["alias"]: d for d in decisions}
        # The oldest page goes; every short fact stays, protected or not.
        self.assertEqual(by_alias["O0"]["action"], "offloaded")
        unprotected_facts = [a for a in ("O1", "O2", "O3", "O4")
                             if not by_alias[a]["recency_protected"]]
        self.assertTrue(unprotected_facts)
        for alias in unprotected_facts:
            self.assertEqual(by_alias[alias]["reason"], "below_min_saving")
            self.assertLess(by_alias[alias]["offload_saving_bytes"], 0)
        for offset, fact in enumerate(facts):
            self.assertEqual(strip_alias_line(trajectory[f"observation_{6 + offset}"]), fact)

    def test_the_five_most_recent_are_protected_before_the_saving_is_asked(self) -> None:
        page = "permission_uid  label\n" + ("row of a listing page\n" * 140)
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"list_permissions_{index}"}
            trajectory[f"observation_{index}"] = page
        decisions = self.compact(trajectory, packed_target_bytes=10_000,
                                 describe_output=self.describe)
        by_alias = {d["alias"]: d for d in decisions}
        self.assertEqual({a: d["recency_protected"] for a, d in by_alias.items()},
                         {"O0": False, "O1": False, "O2": True, "O3": True,
                          "O4": True, "O5": True, "O6": True})
        for alias in ("O2", "O3", "O4", "O5", "O6"):
            self.assertEqual(by_alias[alias]["reason"], "recent_observation_protected")
            # A protected observation is never priced: no label is built for it.
            self.assertNotIn("offload_saving_bytes", by_alias[alias])
        self.assertEqual([by_alias[a]["action"] for a in ("O0", "O1")],
                         ["offloaded", "offloaded"])

    # -- interaction with aliases, eager archiving and bounded search ------

    def test_the_printed_alias_line_is_excluded_from_the_measured_saving(self) -> None:
        """The printed alias line is ~34 B: counting it would flip this observation."""
        body = self.body_saving(1_023)
        trajectory, decision = self.decide(alias_line("O0") + body)
        self.assertEqual(printed_alias(trajectory["observation_0"]), "O0")
        self.assertEqual(decision["offload_saving_bytes"], 1_023)
        self.assertEqual(decision["action"], "kept")

    def test_the_eager_archive_is_unaffected_by_the_rule(self) -> None:
        """Availability is unconditional; the rule only moves residency."""
        kept = self.body_saving(1_023)
        trajectory = self.one_step(kept)
        trajectory["tool_name_1"] = "execute_workflow_query"
        trajectory["tool_args_1"] = {"command": "list_controls"}
        trajectory["observation_1"] = self.body_saving(4_096, alias="O1",
                                                       command="list_controls")
        self.compact(trajectory, recent_observations_protected=0,
                     packed_target_tokens=1)
        self.assertFalse(is_offload_label(trajectory["observation_0"]))
        self.assertTrue(is_offload_label(trajectory["observation_1"]))
        rows = {row["alias"]: row for row in self.archive.list(self.scope)}
        self.assertEqual(set(rows), {"O0", "O1"})
        self.assertEqual(rows["O0"]["text"], kept)

    def test_search_memory_observations_are_still_never_offloaded(self) -> None:
        """Search output is bounded separately; compaction only ever touches executes."""
        answer = "answer line\n" + "a" * 5_000
        trajectory = self._page_trajectory(self.body_saving(4_096))
        trajectory["tool_name_6"] = "search_memory"
        trajectory["tool_args_6"] = {"alias": "O0", "question": "who?"}
        trajectory["observation_6"] = answer
        decisions = self.compact(trajectory)
        self.assertEqual(trajectory["observation_6"], answer)
        self.assertNotIn("S6", {d["alias"] for d in decisions})
        self.assertIsNone(printed_alias(trajectory["observation_6"]))

    # -- the knob ----------------------------------------------------------

    def test_env_override_raises_and_lowers_the_minimum(self) -> None:
        body = self.body_saving(1_024)
        with patch.dict(os.environ, {MIN_OFFLOAD_SAVING_BYTES_ENV: "2048"}):
            # Each self.decide() runs in its own scope, so re-offering the same
            # alias with different text is never the archive refusing a rewrite.
            self.assertEqual(min_offload_saving_bytes_from_env(), 2_048)
            _trajectory, decision = self.decide(body)
            self.assertEqual(decision["reason"], "below_min_saving")
            self.assertEqual(decision["min_offload_saving_bytes"], 2_048)
        with patch.dict(os.environ, {MIN_OFFLOAD_SAVING_BYTES_ENV: "0"}):
            self.assertEqual(min_offload_saving_bytes_from_env(), 0)
            _trajectory, decision = self.decide(self.body_saving(1))
            self.assertEqual(decision["action"], "offloaded")

    def test_a_bad_or_negative_override_falls_back_to_the_default(self) -> None:
        for raw in ("", "   ", "lots", "1_024 bytes", "-1"):
            with patch.dict(os.environ, {MIN_OFFLOAD_SAVING_BYTES_ENV: raw}):
                self.assertEqual(min_offload_saving_bytes_from_env(),
                                 MIN_OFFLOAD_SAVING_BYTES, raw)

    # -- the replan skeleton decides eligibility the same way ---------------

class RestoreWording(unittest.TestCase):
    """The restore is budget-bound, and every place that promises it says so."""

    def test_a_label_says_normally_restored_and_both_earlier_marks_still_parse(self) -> None:
        label = offload_label(alias="O3", command_name="show_owners", response="payload",
                              description="holder rows")
        self.assertTrue(label.endswith("Normally restored for the final answer."))
        self.assertEqual(LABEL_RESTORE_MARK, "Normally restored for the final answer.")
        earlier = ("Offloaded observation O3 returned by show_owners. It contains holder rows. "
                   "Restored in full for the final answer.")
        self.assertTrue(is_offload_label(earlier))
        self.assertEqual(label_alias(earlier), "O3")

    def test_the_agent_signature_names_the_evidence_limit(self) -> None:
        doc = " ".join((WorkflowAgentSignature.__doc__ or "").split())
        self.assertIn("is normally restored in full when the final answer is written", doc)
        self.assertIn("If the answer's evidence limit is reached, the oldest observations are not "
                      "restored and the answer names them.", doc)
        self.assertNotIn("is restored in full", doc)
        self.assertIn("If you need a value from an offloaded observation to choose your next step, "
                      "run its command again", doc)


