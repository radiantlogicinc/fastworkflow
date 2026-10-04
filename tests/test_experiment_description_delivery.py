"""The optional experiment description, as delivered (`fix-1ay`).

The bead's original text asked for a label column and a `DROP COLUMN`; by the
time it was re-read the description-based storage, the HTTP routes and the UI
were already in place, so this file is the verification half rather than new
behaviour. It pins the four things the acceptance names, and it pins them
where each one actually lives:

* one-click creation with an empty description -- `tests/test_benchmark_setup.py`
  covers the setup module and the POST route, so what is added here is the
  end-to-end consequence: an experiment created with no description still
  appears in navigation, carrying an empty label rather than a generated
  sentence nobody wrote;
* the generated navigation label -- the page derives it from the experiment id,
  which `tests/chatbot_hierarchy_dom.cjs` and
  `tests/chatbot_experiment_archive_dom.cjs` already drive in a real DOM. What
  is pinned here is the rule the two of them rely on: the label is generated,
  and the author's description is shown as a description, not as a heading;
* the description edit rules -- editable until a runner claims the
  registration, read-only afterwards, in the UI as well as at the route;
* the ABSENCE of the obsolete surfaces. Absence is the part no behavioural
  test covers by accident: a reinstated hypothesis field or a revived
  per-experiment analysis endpoint would break nothing that exists, so the
  only thing that catches it is looking for it. `analysis.json` on a
  BENCHMARK is a different, still-supported feature and is checked to be
  still there, so this file cannot pass by deleting both.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests import test_chatbot_benchmarks as benchmark_fixtures
from tests.test_chatbot_benchmarks import _request

workflow_dir = benchmark_fixtures.workflow_dir
live_server = benchmark_fixtures.live_server


@pytest.fixture
def recorded_server(workflow_dir, tmp_path, monkeypatch):
    """A workflow whose store holds one experiment with no description."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db_path = state_paths.observability_db(str(workflow_dir))
    store = obs.ObservabilityStore(db_path)
    store.create_experiment(
        "exp-no-description",
        "",
        declared_tasks=1,
        declared_attempts=1,
    )
    server = run_chatbot_server.ChatbotServer(
        db_path=db_path,
        workflow_path=str(workflow_dir),
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, store
    server.shutdown()
    thread.join(timeout=5)


def _experiment_nodes(root):
    found = []

    def walk(node):
        if node.get("kind") == "experiment":
            found.append(node)
        for child in node.get("children") or []:
            walk(child)

    walk(root)
    return found


class TestCreationStaysOneClick:
    def test_an_experiment_created_without_a_description_reaches_navigation(
        self, recorded_server
    ):
        """Empty is a real, complete state -- not a form left half-filled."""
        server, _store = recorded_server

        status, data = _request(server, "/api/navigation")

        assert status == 200
        nodes = _experiment_nodes(data["root"])
        assert [node["experiment_id"] for node in nodes] == ["exp-no-description"]
        # The node's own label is the author's description, and it is empty.
        # The page generates what it displays from the id; nothing invents a
        # sentence and stores it as if the author had written one.
        assert nodes[0]["label"] == ""
        assert nodes[0]["info"]["description"] == ""

    def test_a_description_written_later_becomes_the_nodes_label(
        self, live_server, workflow_dir
    ):
        created = benchmark_setup.create_experiment(workflow_dir, "smoke", "v1")
        experiment_id = created["experiment_id"]
        assert created["description"] == ""

        status, patched = _request(
            live_server,
            f"/api/benchmark-experiments/{experiment_id}",
            "PATCH",
            {"description": "Does the refiner drop the trailing clause?"},
        )

        assert status == 200
        assert patched["experiment"]["description"] == (
            "Does the refiner drop the trailing clause?"
        )
        nodes = _experiment_nodes(_request(live_server, "/api/navigation")[1]["root"])
        labels = {node["experiment_id"]: node["label"] for node in nodes}
        assert labels[experiment_id] == "Does the refiner drop the trailing clause?"


class TestThePageGeneratesTheNameAndShowsTheDescription:
    def test_the_heading_is_derived_from_the_id_not_from_free_text(self):
        """A heading that is a paragraph, or blank, is not a heading.

        The DOM harnesses drive this; the rule itself is pinned here because
        a change that made the heading the description would still render.
        """
        page = run_chatbot_server.load_index_html()

        assert b'"Experiment \xc2\xb7 " + experimentId.slice(-8)' in page
        assert b'? ("Description: " + exp.description)' in page

    def test_the_description_is_editable_only_before_a_runner_claims_it(self):
        page = run_chatbot_server.load_index_html()

        assert b'benchmarkText(note, "Description (optional)"' in page
        assert b"description.readOnly = true;" in page
        assert (
            b"This experiment has been handed to a runner; its description is "
            b"part of the recorded run." in page
        )
        assert b'{description: description.value}' in page

    def test_the_route_enforces_the_same_rule_the_page_shows(
        self, live_server, workflow_dir
    ):
        """Non-vacuous companion to the source pins above.

        The page hides the control; the route is what actually refuses, and a
        client that kept the old page must not be able to rewrite a recorded
        run's declaration.
        """
        created = benchmark_setup.create_experiment(workflow_dir, "smoke", "v1")
        experiment_id = created["experiment_id"]
        path = f"/api/benchmark-experiments/{experiment_id}"
        assert _request(live_server, path, "PATCH", {"description": "before"})[0] == 200

        benchmark_setup.bind_experiment(
            workflow_dir,
            experiment_id,
            str(Path(workflow_dir) / "evidence.sqlite3"),
            "store-1",
        )

        status, data = _request(
            live_server, path, "PATCH", {"description": "after the handoff"}
        )

        assert status == 409
        assert "runner" in json.dumps(data)
        assert benchmark_setup.load_experiment(workflow_dir, experiment_id)[
            "description"
        ] == "before"


class TestTheObsoleteSurfacesAreGone:
    def test_no_hypothesis_field_is_required_or_offered_anywhere(
        self, live_server, workflow_dir
    ):
        """Creation takes a description and a repeat count, and nothing else.

        The owner's intent is that an experiment costs one click; a required
        hypothesis is exactly the ceremony that would undo that.
        """
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/experiments",
            "POST",
            {"version": "v1"},
        )

        assert status == 201
        assert "hypothesis" not in json.dumps(data).lower()
        page = run_chatbot_server.load_index_html().decode("utf-8")
        # No field, no key, no control: the word may only appear as prose. A
        # quoted "hypothesis", a `hypothesis:` property or an element named
        # for it are the three shapes an actual field would take.
        assert '"hypothesis' not in page and "'hypothesis" not in page
        assert "hypothesis:" not in page.lower()
        assert "hypothesis=" not in page.lower()
        # ...and the one surviving mention says none is required, which is the
        # opposite of offering one.
        surviving = [
            line.strip()
            for line in page.splitlines()
            if "hypothesis" in line.lower()
        ]
        assert surviving == [
            "target, rubric, hypothesis or review gate is required to ask for "
            "n attempts"
        ], surviving

    def test_the_per_experiment_analysis_surface_is_gone(self, recorded_server):
        """Removed with `analysis_json`; a client calling it gets a refusal.

        Re-pointing it at something else would be worse than a 404: a silently
        different payload is indistinguishable from the feature still working.
        """
        server, store = recorded_server

        assert "analysis_json" not in store.get_experiment("exp-no-description")
        assert not hasattr(store, "update_experiment_analysis")
        status, _data = _request(
            server, "/api/experiment/exp-no-description/analysis"
        )
        assert status in (404, 405)
        status, _data = _request(
            server,
            "/api/experiment/exp-no-description/analysis",
            "PUT",
            {"analysis": "reinstated"},
        )
        assert status in (400, 404, 405)

    def test_benchmark_analysis_is_a_different_feature_and_still_works(
        self, live_server
    ):
        """The guard that stops this file passing by deleting everything.

        `analysis.json` on a BENCHMARK was never in scope for the removal.
        """
        status, data = _request(
            live_server,
            "/api/benchmarks/smoke/analysis",
            "PUT",
            {"analysis": "what this benchmark is for"},
        )

        assert status in (200, 201), data
        assert (
            _request(live_server, "/api/benchmarks/smoke/analysis")[1]["analysis"]
            == "what this benchmark is for"
        )
        assert b'"Analysis (optional)"' in run_chatbot_server.load_index_html()
