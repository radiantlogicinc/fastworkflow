"""An author must be able to answer a duplicate finding instead of living with it.

bd fix-k0i.20 (the accept-list) and bd fix-k0i.21 (running the model-based detector for
real). Both are about the same gap: R9b says "merge, alias, OR ACCEPT", and until now only
the first two were reachable, while the instrument that catches the pairs the lexical scan
provably cannot see had no caller outside its own tests.

No mocks (`.cursor/rules/testing_rules.mdc`). The workflows are the real ones:

* ``tests/duplicate_capability_workflow`` — the F14 positive control, whose
  ``list_findings`` / ``search_control_findings`` pair the lexical scan reports.
* ``fastworkflow/examples/retail_workflow`` — the false-positive stress test, and the
  source of the one committed trained model in the repo, which is what lets the
  post-training router scan be exercised without a training run.

Accept-list files are written as real JSON at the real path the shipped loader reads.
"""

import json
import os
import shutil
import warnings
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import duplicate_detection as dd
from fastworkflow.train.__main__ import (
    _report_router_confusion,
    _validate_command_inputs,
)

CONTROL_PATH = os.path.join("tests", "duplicate_capability_workflow")
RETAIL_PATH = os.path.join("fastworkflow", "examples", "retail_workflow")
CONTROL_DUPLICATE_PAIR = ("list_findings", "search_control_findings")


def _resolve_env_vars() -> dict:
    """Resolve settings the same way the shipped training entry point does.

    Nothing here calls an LLM; the routing definition simply reads settings during load.
    """
    env_vars = {
        **dotenv_values(os.path.join("fastworkflow", "examples", "fastworkflow.env")),
        **dotenv_values(
            os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
        ),
    }
    for override in (os.path.join("env", ".env"), os.path.join("passwords", ".env")):
        if os.path.exists(override):
            env_vars.update(dotenv_values(override))
    return env_vars


@pytest.fixture(scope="module", autouse=True)
def _initialised_fastworkflow():
    fastworkflow.init(env_vars=_resolve_env_vars())


def _copy_workflow(source: str, destination: Path) -> str:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"
        ),
    )
    return str(destination)


@pytest.fixture
def control_workflow(tmp_path: Path) -> str:
    return _copy_workflow(CONTROL_PATH, tmp_path / "duplicate_control")


def _write_accept_list(workflow_path: str, payload) -> Path:
    path = Path(dd.accept_list_path(workflow_path))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _accepting_the_control_pair(reason: str = "deliberate alias kept for old clients"):
    return {
        "schema_version": dd.ACCEPT_LIST_SCHEMA_VERSION,
        "accepted": [{"commands": list(CONTROL_DUPLICATE_PAIR), "reason": reason}],
    }


# ---------------------------------------------------------------------------
# Where the file lives, and what an absent one means
# ---------------------------------------------------------------------------


def test_the_accept_list_lives_at_the_workflow_root_not_in_generated_output(
    control_workflow: str,
):
    """It is hand-authored input, so it must not sit where the trainer prunes and rebuilds.

    ``___command_info`` holds generated artifacts and is the directory developers are told
    to delete to force a rebuild. An acceptance stored there would be silently discarded
    and the answered warning would come back.
    """
    expected = Path(control_workflow) / dd.ACCEPT_LIST_FILENAME
    assert Path(dd.accept_list_path(control_workflow)) == expected
    assert dd.COMMAND_INFO_DIRNAME not in dd.accept_list_path(control_workflow)


def test_no_accept_list_is_the_normal_case_and_reports_no_problem(control_workflow: str):
    """A workflow that has never needed to accept a pair must not be nagged about a file."""
    accept_list = dd.load_accept_list(control_workflow)
    assert accept_list.exists is False
    assert accept_list.pairs == []
    assert accept_list.problems == []


# ---------------------------------------------------------------------------
# fix-k0i.20: fire, and then do not fire once accepted
# ---------------------------------------------------------------------------


def test_the_control_duplicate_fires_through_the_shipped_train_preflight(
    control_workflow: str, capsys
):
    """Baseline for the pair below: without an accept-list the pair must be reported.

    Also the fire half of the finding's "test both fire and accepted-no-fire paths through
    the shipped train preflight".
    """
    report = _validate_command_inputs(control_workflow)
    rendered = capsys.readouterr().out

    assert {
        tuple(sorted((f.command_a, f.command_b))) for f in report.duplicates
    } == {CONTROL_DUPLICATE_PAIR}
    assert report.accepted == []
    assert "DUPLICATE CAPABILITIES" in rendered
    assert dd.ACCEPT_LIST_FILENAME in rendered, (
        "a reported pair must come with the way to accept it, or 'accept' is not an option"
    )


def test_an_accepted_pair_stops_being_a_warning_but_stays_visible(
    control_workflow: str, capsys
):
    """The whole point: the author's decision is recorded once, not re-litigated per run.

    Suppressed from the warning band, still listed as accepted with the stated reason, so
    the tool never hides that it found the pair.
    """
    _write_accept_list(control_workflow, _accepting_the_control_pair())

    report = _validate_command_inputs(control_workflow)
    rendered = capsys.readouterr().out

    assert report.duplicates == []
    assert [
        tuple(sorted((f.command_a, f.command_b))) for f in report.accepted
    ] == [CONTROL_DUPLICATE_PAIR]
    assert "DUPLICATE CAPABILITIES" not in rendered
    assert "ACCEPTED (1)" in rendered
    assert "deliberate alias kept for old clients" in rendered
    assert report.unmatched_accepted_pairs == []


def test_the_written_json_report_records_the_acceptance(control_workflow: str):
    """A CI job reading the report file must see the same verdict the console showed."""
    _write_accept_list(control_workflow, _accepting_the_control_pair())
    _validate_command_inputs(control_workflow)

    payload = json.loads(
        Path(
            control_workflow, dd.COMMAND_INFO_DIRNAME, dd.REPORT_FILENAME
        ).read_text(encoding="utf-8")
    )
    report = payload["report"]
    assert report["duplicates"] == []
    assert [
        tuple(sorted((f["command_a"], f["command_b"]))) for f in report["accepted"]
    ] == [CONTROL_DUPLICATE_PAIR]
    assert report["accept_list_path"].endswith(dd.ACCEPT_LIST_FILENAME)
    assert report["accepted_pairs"][0]["reason"]


def test_pair_order_in_the_file_does_not_matter(control_workflow: str):
    """An author copying two names out of the report should not have to guess the order."""
    _write_accept_list(
        control_workflow,
        {"accepted": [{"commands": list(reversed(CONTROL_DUPLICATE_PAIR))}]},
    )
    report = _validate_command_inputs(control_workflow)
    assert report.duplicates == []
    assert len(report.accepted) == 1


def test_a_bare_pair_list_is_accepted_as_a_shorthand(control_workflow: str):
    """Both hand-written forms work; only the object form has room for a reason."""
    _write_accept_list(control_workflow, [list(CONTROL_DUPLICATE_PAIR)])
    report = _validate_command_inputs(control_workflow)
    assert report.duplicates == []
    assert report.accepted_pairs[0].reason is None


def test_an_accept_list_entry_must_name_the_exact_commands(control_workflow: str):
    """A near-miss must not silently suppress anything.

    Failing toward reporting matters more than convenience here: an entry that quietly
    matched the wrong pair would hide a real duplicate, which is worse than a warning the
    author has to answer twice.
    """
    _write_accept_list(
        control_workflow,
        {"accepted": [{"commands": ["list_findings", "search_findings"]}]},
    )
    report = _validate_command_inputs(control_workflow)
    assert {
        tuple(sorted((f.command_a, f.command_b))) for f in report.duplicates
    } == {CONTROL_DUPLICATE_PAIR}
    assert report.accepted == []
    assert len(report.unmatched_accepted_pairs) == 1


def test_a_stale_entry_is_named_so_it_can_be_deleted(control_workflow: str, capsys):
    """An entry for a pair that no longer fires is dead weight; say so."""
    _write_accept_list(
        control_workflow,
        {
            "accepted": [
                {"commands": list(CONTROL_DUPLICATE_PAIR)},
                {"commands": ["create_user", "acknowledge_finding"]},
            ]
        },
    )
    report = _validate_command_inputs(control_workflow)
    rendered = capsys.readouterr().out

    assert [
        tuple(sorted(entry.pair)) for entry in report.unmatched_accepted_pairs
    ] == [("acknowledge_finding", "create_user")]
    assert "STALE ACCEPT-LIST ENTRIES (1)" in rendered


@pytest.mark.parametrize(
    "payload,expected_fragment",
    [
        ({"accepted": [{"commands": ["only_one"]}]}, "exactly two commands"),
        ({"accepted": ["not-a-pair"]}, "not a pair"),
        ({"accepted": "not a list"}, "must be a list"),
        ({"nothing": "here"}, 'no "accepted" list'),
        (
            {"accepted": [{"commands": ["same", "same"]}]},
            "two different commands",
        ),
    ],
)
def test_a_malformed_entry_is_described_and_suppresses_nothing(
    control_workflow: str, payload, expected_fragment: str
):
    """A typo in this file must never be what stops an expensive training run.

    But it must also never look like a working suppression: the author has to learn that
    the pair they believe is accepted is not.
    """
    _write_accept_list(control_workflow, payload)
    report = _validate_command_inputs(control_workflow)

    assert {
        tuple(sorted((f.command_a, f.command_b))) for f in report.duplicates
    } == {CONTROL_DUPLICATE_PAIR}
    assert any(
        expected_fragment in problem for problem in report.accept_list_problems
    ), report.accept_list_problems


def test_an_unparseable_accept_list_is_loud_rather_than_fatal(
    control_workflow: str, capsys
):
    Path(dd.accept_list_path(control_workflow)).write_text(
        "{not json", encoding="utf-8"
    )

    report = _validate_command_inputs(control_workflow)
    rendered = capsys.readouterr().out

    assert report.accept_list_problems
    assert "ACCEPT-LIST PROBLEMS" in rendered
    assert report.duplicates, "an unreadable accept-list must not suppress a real finding"


def test_accept_list_problems_are_not_filed_as_routine_scan_notes(control_workflow: str):
    """`notes` carries observations a clean workflow has too, so this needs its own field.

    Folded into `notes`, an unreadable accept-list could not raise the report's visibility
    without also printing the report on every clean run.
    """
    _write_accept_list(control_workflow, {"accepted": [{"commands": ["one"]}]})
    report = _validate_command_inputs(control_workflow)
    assert report.accept_list_problems
    assert not any(
        dd.ACCEPT_LIST_FILENAME in note for note in report.notes
    )


def test_applying_the_accept_list_twice_changes_nothing(control_workflow: str):
    """The post-training router scan re-applies it to attach its own findings."""
    _write_accept_list(control_workflow, _accepting_the_control_pair())
    report = dd.scan_workflow(control_workflow)
    first = report.model_dump()

    dd.apply_accept_list(report, dd.load_accept_list(control_workflow))
    assert report.model_dump() == first


def test_an_accepted_pair_is_suppressed_on_the_model_axis_too(control_workflow: str):
    """One decision, both instruments.

    The router used here is a real nearest-centroid classifier over the control workflow's
    own seeds — the same construction `test_duplicate_detection` uses for its model-axis
    control — built so the scored utterances are genuinely unseen by their own centroid.
    """
    seeds = dd.utterances_from_workflow(control_workflow)
    idf = dd.inverse_document_frequencies(
        dd.document_frequencies(seeds), len(seeds)
    )
    centroids = {
        command: dd.tfidf_vector(utterances[::2], idf)
        for command, utterances in seeds.items()
    }

    def predict(utterance: str) -> list[str]:
        vector = dd.tfidf_vector(utterance, idf)
        return sorted(
            centroids,
            key=lambda command: (-dd.cosine(vector, centroids[command]), command),
        )[:1]

    confusions = dd.find_confusable_commands(seeds, predict, confusion_threshold=0.0)
    assert confusions, "precondition: the control router must confuse the pair"

    report = dd.DuplicateReport(workflow_folderpath=control_workflow)
    report.confusable = confusions
    _write_accept_list(control_workflow, _accepting_the_control_pair())
    dd.apply_accept_list(report, dd.load_accept_list(control_workflow))

    assert [
        tuple(sorted((f.command_a, f.command_b))) for f in report.accepted_confusable
    ] == [CONTROL_DUPLICATE_PAIR]
    assert all(
        tuple(sorted((f.command_a, f.command_b))) != CONTROL_DUPLICATE_PAIR
        for f in report.confusable
    )


# ---------------------------------------------------------------------------
# fix-k0i.21: the model-based detector now has a production caller
# ---------------------------------------------------------------------------


def _retail_model_dir() -> Path:
    return Path(RETAIL_PATH, "___command_info", "global")


@pytest.fixture
def retail_workflow_with_version(tmp_path: Path) -> tuple[str, str]:
    """A retail copy whose committed trained model sits inside an artifact version.

    `_report_router_confusion` reads the routers out of the version being published, which
    is the only place a just-trained model exists before publication. Staging the one
    committed model in that layout exercises the real `CommandRouter` against real weights
    without a training run.
    """
    workflow_path = _copy_workflow(RETAIL_PATH, tmp_path / "retail")
    version_id = av.new_version_id()
    shutil.copytree(
        _retail_model_dir(),
        av.context_artifact_dir(workflow_path, version_id, "*"),
        dirs_exist_ok=True,
    )
    av.write_manifest(workflow_path, version_id)
    return workflow_path, version_id


@pytest.mark.skipif(
    not (_retail_model_dir() / "threshold.json").is_file(),
    reason="retail_workflow has no committed trained model in ___command_info/global",
)
def test_the_post_training_scan_runs_the_real_router_and_reports_nothing_on_retail(
    retail_workflow_with_version: tuple[str, str],
):
    """The wiring itself: a real router, real seeds, and the model-side stress test.

    Retail's nineteen commands include three families of near-siblings. A detector wired
    into every training run that flagged them would be a wall of noise, so "the shipped
    workflow produces no findings" is the load-bearing assertion here — the same role the
    lexical stress test plays for the pre-flight scan.
    """
    workflow_path, version_id = retail_workflow_with_version
    report = dd.scan_workflow(workflow_path)

    # Preconditions, so that an empty result cannot be a vacuous pass. "No findings" is
    # only meaningful if the scan really loaded a router and routed something through it.
    staged = av.version_dir(workflow_path, version_id) / av.context_folder_name("*")
    assert (staged / "threshold.json").is_file()
    assert len(dd.utterances_from_workflow(workflow_path)) > 2
    report_path = Path(workflow_path, dd.COMMAND_INFO_DIRNAME, dd.REPORT_FILENAME)
    assert not report_path.exists(), "precondition: nothing has written the report yet"

    with warnings.catch_warnings():
        # The committed label encoder was pickled by an older scikit-learn.
        warnings.simplefilter("ignore")
        probe = CommandRouter(str(staged)).predict("cancel my pending order")
        assert [str(label) for label in probe], (
            "the staged router returned nothing; the scan below would be vacuous"
        )
        # Emptied so that finding this path in the cache afterwards is evidence that the
        # scan itself built a router there, not that the probe above left one behind.
        CommandRouter._instances_cache.clear()
        findings = _report_router_confusion(workflow_path, version_id, report)

    assert str(staged) in CommandRouter._instances_cache, (
        "the scan never loaded a router; it cannot have measured anything"
    )
    assert findings == [], "\n".join(
        dd.format_confusion_line(finding) for finding in findings
    )
    assert report_path.is_file(), (
        "the model axis must reach the written report, not only the console"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report"]["confusable"] == []


def test_the_post_training_scan_survives_a_version_with_no_routers(
    control_workflow: str,
):
    """A diagnostic that runs after the money is spent must never fail a trained run.

    A selective run's version, a context whose model failed to save, an unreadable
    checkpoint: all of them reach this code, and none of them is a reason to discard hours
    of training.
    """
    version_id = av.new_version_id()
    av.context_artifact_dir(control_workflow, version_id, "*")
    report = dd.scan_workflow(control_workflow)

    assert _report_router_confusion(control_workflow, version_id, report) == []
    assert report.confusable == []
