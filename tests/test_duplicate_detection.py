"""Integration tests for near-duplicate capability detection (spec R9b / finding F14).

Everything here runs against real workflows loaded through the real `CommandDirectory`:

* ``fastworkflow/examples/retail_workflow`` — the false-positive stress test. Nineteen
  commands including ``modify_pending_order_address`` / ``_items`` / ``_payment``,
  ``return_delivered_order_items`` / ``exchange_delivered_order_items`` and
  ``find_user_id_by_email`` / ``find_user_id_by_name_zip``. A detector that flags related
  commands would produce a wall of warnings here.
* ``tests/duplicate_capability_workflow`` — the positive control, reproducing the F14 pair.
  Without it, every workflow in the repo is a negative and passing tests would only show
  the detector never fires.
* ``fastworkflow/examples/retail_workflow/___command_info/global`` — the committed trained
  model, used to exercise the model-based detector against a real ``CommandRouter``.

No mocks and no fabricated utterances: the seed lists come from the shipped command files.
No API key is required — the lexical detector reads seeds, never generated utterances.
"""

import json
import os
import shutil
import warnings

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.duplicate_detection import (
    DEFAULT_DUPLICATE_SEPARABILITY,
    NON_CAPABILITY_LABELS,
    contexts_from_workflow,
    cosine,
    document_frequencies,
    find_confusable_commands,
    find_duplicate_capabilities,
    format_report,
    inverse_document_frequencies,
    pairwise_separability,
    scan_workflow,
    tfidf_vector,
    tokenize,
    utterances_from_workflow,
    write_report,
)

RETAIL_PATH = os.path.join("fastworkflow", "examples", "retail_workflow")
CONTROL_PATH = os.path.join("tests", "duplicate_capability_workflow")
HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")

CONTROL_DUPLICATE_PAIR = ("list_findings", "search_control_findings")


def _copy_workflow(source: str, destination) -> str:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    return str(destination)


@pytest.fixture
def control_workflow(tmp_path) -> str:
    return _copy_workflow(CONTROL_PATH, tmp_path / "duplicate_control")


@pytest.fixture
def retail_workflow(tmp_path) -> str:
    return _copy_workflow(RETAIL_PATH, tmp_path / "retail")


def _resolve_env_vars() -> dict:
    """Build the env the same way `test_train_modern_stack._resolve_env_vars` does.

    Only the non-secret settings matter here (nothing in this file calls an LLM), but the
    resolution order is copied so a workflow that reads a setting during load behaves the
    same as it does under training.
    """
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}

    local_env = os.path.join("env", ".env")
    local_pwd = os.path.join("passwords", ".env")
    if os.path.exists(local_env):
        env_vars.update(dotenv_values(local_env))
    if os.path.exists(local_pwd):
        env_vars.update(dotenv_values(local_pwd))
    return env_vars


@pytest.fixture(scope="module", autouse=True)
def _initialised_fastworkflow():
    fastworkflow.init(env_vars=_resolve_env_vars())


@pytest.fixture(scope="module")
def retail_seeds() -> dict:
    return utterances_from_workflow(RETAIL_PATH)


@pytest.fixture(scope="module")
def control_seeds() -> dict:
    return utterances_from_workflow(CONTROL_PATH)


@pytest.fixture(scope="module")
def retail_report():
    return scan_workflow(RETAIL_PATH)


@pytest.fixture(scope="module")
def control_report():
    return scan_workflow(CONTROL_PATH)


# ---------------------------------------------------------------------------
# Vector space primitives
# ---------------------------------------------------------------------------


def test_tokenize_drops_single_characters_and_lowercases():
    assert tokenize("Cancel my order, I don't need it") == [
        "cancel", "my", "order", "don't", "need", "it"
    ]


def test_idf_discounts_terms_every_command_uses(retail_seeds):
    """The mechanism that stops the retail `modify_pending_order_*` family firing.

    "order" is used by most retail commands and "gift" by one, so IDF must rank them in
    that order. Without this the shared boilerplate would dominate every vector and the
    family would look identical.
    """
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    assert idf["order"] < idf["gift"]


def test_tfidf_vector_is_l2_normalised(retail_seeds):
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    vector = tfidf_vector(retail_seeds["cancel_pending_order"], idf)
    norm = sum(weight * weight for weight in vector.values()) ** 0.5
    assert norm == pytest.approx(1.0)


def test_tfidf_vector_of_untokenizable_text_is_empty(retail_seeds):
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    assert tfidf_vector("!!! ...", idf) == {}
    assert cosine({}, tfidf_vector("cancel my order", idf)) == 0.0


def test_pairwise_separability_needs_two_utterances_per_side(retail_seeds):
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    assert pairwise_separability(["only one"], ["a", "b"], idf) is None
    assert pairwise_separability(["a", "b"], ["only one"], idf) is None


def test_pairwise_separability_is_symmetric(retail_seeds):
    """Swapping the arguments must swap the recalls and leave the balanced score alone."""
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    a = retail_seeds["cancel_pending_order"]
    b = retail_seeds["get_user_details"]
    forward = pairwise_separability(a, b, idf)
    backward = pairwise_separability(b, a, idf)
    assert forward.balanced_accuracy == pytest.approx(backward.balanced_accuracy)
    assert forward.recall_a == pytest.approx(backward.recall_b)
    assert forward.recall_b == pytest.approx(backward.recall_a)


# ---------------------------------------------------------------------------
# Positive control: the detector must actually fire
# ---------------------------------------------------------------------------


def test_control_workflow_duplicate_pair_is_reported(control_report):
    """The deliberate duplicate must be reported, and it must be the only one."""
    reported = {
        tuple(sorted((f.command_a, f.command_b))) for f in control_report.duplicates
    }
    assert reported == {CONTROL_DUPLICATE_PAIR}


def test_control_duplicate_is_far_below_the_threshold(control_report):
    """Not a near miss: the pair must be comfortably inside the duplicate band.

    A control that only just clears the threshold would make the test a thermometer for
    threshold drift rather than for the detector working.
    """
    finding = control_report.duplicates[0]
    assert finding.separability <= DEFAULT_DUPLICATE_SEPARABILITY / 2


def test_control_duplicate_names_the_pair_and_explains_itself(control_report):
    """F14's whole value is naming the pair, so the finding must carry enough to act on."""
    finding = control_report.duplicates[0]
    assert finding.command_a == CONTROL_DUPLICATE_PAIR[0]
    assert finding.command_b == CONTROL_DUPLICATE_PAIR[1]
    assert "findings" in finding.shared_terms
    assert finding.utterances_a >= 3 and finding.utterances_b >= 3
    # Both commands live in the global context, so this is a live classifier conflict
    # rather than a cross-context design ambiguity.
    assert finding.shared_contexts


def test_control_hard_negative_is_not_reported(control_report):
    """`acknowledge_finding` shares the subject matter but is a different capability."""
    flagged = {
        command
        for finding in control_report.duplicates + control_report.overlapping
        for command in (finding.command_a, finding.command_b)
    }
    assert "acknowledge_finding" not in flagged
    assert "create_user" not in flagged


def test_control_report_renders_the_pair(control_report):
    rendered = format_report(control_report)
    assert "DUPLICATE CAPABILITIES (1)" in rendered
    assert "list_findings" in rendered
    assert "search_control_findings" in rendered


def test_report_keeps_duplicate_and_overlap_guidance_separate(control_report):
    duplicate_text = format_report(control_report)
    assert "knowingly accept them as duplicates" in duplicate_text
    assert "legitimate neighbours or opposites" not in duplicate_text

    overlapping_report = control_report.model_copy(
        update={
            "duplicates": [],
            "overlapping": list(control_report.duplicates),
        }
    )
    overlapping_text = format_report(overlapping_report)
    assert "OVERLAPPING (1)" in overlapping_text
    assert "legitimate neighbours or opposites" in overlapping_text
    assert "distinctive seed utterances" in overlapping_text


# ---------------------------------------------------------------------------
# False-positive stress test: the retail workflow
# ---------------------------------------------------------------------------


def test_retail_workflow_reports_no_duplicates(retail_report):
    """The stress test. Nineteen commands, several families of near-siblings, zero warnings.

    If this ever fails, read the finding before changing the threshold: the detector may be
    right and a genuine duplicate may have been added.
    """
    assert retail_report.duplicates == [], format_report(retail_report)


def test_retail_modify_pending_order_family_is_not_flagged_as_duplicate(retail_report):
    """The specific family named as the realistic false-positive risk."""
    family = {
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
    }
    for finding in retail_report.duplicates:
        assert not ({finding.command_a, finding.command_b} <= family), (
            f"{finding.command_a} and {finding.command_b} were reported as duplicate "
            f"capabilities; they modify different parts of an order."
        )


def test_retail_related_command_families_are_not_flagged_as_duplicates(retail_report):
    """The other two families of legitimately related retail commands."""
    related_families = [
        {"find_user_id_by_email", "find_user_id_by_name_zip"},
        {"return_delivered_order_items", "exchange_delivered_order_items"},
        {"modify_pending_order_address", "modify_user_address"},
    ]
    flagged = {
        tuple(sorted((f.command_a, f.command_b))) for f in retail_report.duplicates
    }
    for family in related_families:
        assert tuple(sorted(family)) not in flagged


def test_retail_overlapping_findings_stay_a_handful(retail_report):
    """The lower-severity band must not become the wall of noise it exists to avoid.

    Retail has 171 command pairs. A double-digit overlapping list would mean the band is
    reporting the workflow's shared vocabulary rather than anything actionable.
    """
    assert len(retail_report.overlapping) <= 3, format_report(retail_report)


def test_retail_worst_pair_has_headroom_above_the_duplicate_threshold(retail_seeds):
    """Measure the margin rather than asserting only that nothing fired.

    A test that passes because every pair scores 0.51 is one seed edit away from failing.
    This records how much room there actually is.
    """
    idf = inverse_document_frequencies(
        document_frequencies(retail_seeds), len(retail_seeds)
    )
    names = sorted(retail_seeds)
    scores = []
    for index, command_a in enumerate(names):
        for command_b in names[index + 1:]:
            separability = pairwise_separability(
                retail_seeds[command_a], retail_seeds[command_b], idf
            )
            if separability is not None:
                scores.append((separability.balanced_accuracy, command_a, command_b))
    worst = min(scores)
    assert worst[0] > DEFAULT_DUPLICATE_SEPARABILITY, (
        f"worst retail pair {worst[1]} / {worst[2]} scored {worst[0]:.3f}"
    )


def test_hello_world_reports_nothing():
    report = scan_workflow(HELLO_WORLD_PATH)
    assert report.duplicates == []
    assert report.overlapping == []


# ---------------------------------------------------------------------------
# Workflow input
# ---------------------------------------------------------------------------


def test_utterances_from_workflow_excludes_non_capability_labels(retail_seeds):
    """`wildcard` and `parameter_value` are not capabilities and must never be compared."""
    for command in retail_seeds:
        assert command.split("/")[-1] not in NON_CAPABILITY_LABELS


def test_utterances_from_workflow_returns_the_shipped_seeds(retail_seeds):
    """Pin the input to the real command file, not to whatever the loader felt like."""
    assert "cancel_pending_order" in retail_seeds
    assert (
        "Can you cancel my pending order?"
        in retail_seeds["cancel_pending_order"]
    )


def test_contexts_from_workflow_includes_the_global_context():
    contexts = contexts_from_workflow(RETAIL_PATH)
    assert contexts
    assert any("cancel_pending_order" in commands for commands in contexts.values())


def test_shared_contexts_reported_for_a_co_occurring_pair(control_report):
    finding = control_report.duplicates[0]
    contexts = contexts_from_workflow(CONTROL_PATH)
    for context_name in finding.shared_contexts:
        assert finding.command_a in contexts[context_name]
        assert finding.command_b in contexts[context_name]


# ---------------------------------------------------------------------------
# Guards and edge cases
# ---------------------------------------------------------------------------


def test_too_few_commands_is_reported_not_crashed():
    report = find_duplicate_capabilities({"only/one": ["a", "b", "c"]})
    assert report.duplicates == []
    assert report.notes


def test_pairs_with_too_few_utterances_are_skipped_and_counted(control_seeds):
    """A two-sentence command must be skipped, not scored on two sentences."""
    corpus = dict(control_seeds)
    corpus["thin_command"] = list(control_seeds[CONTROL_DUPLICATE_PAIR[0]])[:2]
    report = find_duplicate_capabilities(corpus, min_utterances=3)
    assert report.pairs_skipped_too_few_utterances >= 1
    flagged = {
        command
        for finding in report.duplicates + report.overlapping
        for command in (finding.command_a, finding.command_b)
    }
    assert "thin_command" not in flagged


def test_scan_is_deterministic(control_seeds):
    """Same input, same findings. Nothing here may depend on set iteration order."""
    first = find_duplicate_capabilities(control_seeds)
    second = find_duplicate_capabilities(control_seeds)
    assert first.model_dump() == second.model_dump()


def test_command_order_does_not_change_the_findings(control_seeds):
    reversed_corpus = dict(reversed(list(control_seeds.items())))
    forward = find_duplicate_capabilities(control_seeds)
    backward = find_duplicate_capabilities(reversed_corpus)
    assert [f.pair for f in forward.duplicates] == [f.pair for f in backward.duplicates]


def test_report_round_trips_to_disk(tmp_path, control_report):
    path = write_report(str(tmp_path), control_report)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["schema_version"] == 1
    assert payload["report"]["duplicates"][0]["command_a"] == CONTROL_DUPLICATE_PAIR[0]
    assert "definition" in payload


def test_empty_report_renders_without_findings():
    rendered = format_report(find_duplicate_capabilities({}))
    assert "No near-duplicate capabilities detected." in rendered


# ---------------------------------------------------------------------------
# Model-based detection against the committed retail model
# ---------------------------------------------------------------------------


def _retail_model_dir() -> str:
    return os.path.join(RETAIL_PATH, "___command_info", "global")


def _retail_model_is_trained() -> bool:
    return os.path.exists(os.path.join(_retail_model_dir(), "threshold.json"))


@pytest.mark.skipif(
    not _retail_model_is_trained(),
    reason="retail_workflow has no committed trained model in ___command_info/global",
)
def test_confusable_commands_on_the_real_retail_router(retail_seeds):
    """Run every retail seed utterance through the committed router.

    These utterances are in the model's training data, so a high confusion rate would mean
    the model cannot separate two commands even with the answer in front of it. Nothing
    reaches the default threshold, which is the model-side counterpart of the lexical
    stress test above.
    """
    from fastworkflow.model_pipeline_training import CommandRouter

    with warnings.catch_warnings():
        # The committed label encoder was pickled by an older scikit-learn.
        warnings.simplefilter("ignore")
        router = CommandRouter(_retail_model_dir())
        findings = find_confusable_commands(retail_seeds, router.predict)

    assert findings == [], "\n".join(
        f"{f.command_a} <-> {f.command_b}: {f.symmetric_confusion:.0%}" for f in findings
    )


def test_confusable_commands_reports_a_router_that_cannot_separate_a_pair(control_seeds):
    """Positive control for the model-based axis, using a real deterministic router.

    The "router" here is not a mock of anything: it is a genuine nearest-centroid
    classifier over the control workflow's own seed utterances, of the same kind the
    lexical detector uses. It is the smallest real model that reproduces the behaviour
    F14 describes, and it lets the confusion path be exercised without a trained
    checkpoint.

    Centroids are built from every other utterance and scored against all of them, so the
    odd-indexed ones are genuinely unseen. Without that the router memorises its own
    training rows and separates even the duplicate pair — which is finding F1 in
    miniature, and the reason the real detector is meant to be pointed at a real model.
    """
    idf = inverse_document_frequencies(
        document_frequencies(control_seeds), len(control_seeds)
    )
    centroids = {
        command: tfidf_vector(utterances[::2], idf)
        for command, utterances in control_seeds.items()
    }

    def predict(utterance: str) -> list[str]:
        vector = tfidf_vector(utterance, idf)
        ranked = sorted(
            centroids, key=lambda command: (-cosine(vector, centroids[command]), command)
        )
        return ranked[:1]

    findings = find_confusable_commands(control_seeds, predict, confusion_threshold=0.0)
    assert findings, "the router confused nothing at all; the control is not exercising"
    top = findings[0]
    assert tuple(sorted((top.command_a, top.command_b))) == CONTROL_DUPLICATE_PAIR, (
        "the deliberate duplicate must be the most-confused pair, not merely present"
    )
    others = [f.symmetric_confusion for f in findings[1:]]
    assert not others or top.symmetric_confusion > max(others)


def test_confusion_ignores_non_capability_predictions(control_seeds):
    """A router answering `wildcard` everywhere must produce no duplicate findings."""
    findings = find_confusable_commands(
        control_seeds, lambda _utterance: ["wildcard"], confusion_threshold=0.0
    )
    assert findings == []


def test_confusion_tolerates_a_router_returning_nothing(control_seeds):
    findings = find_confusable_commands(
        control_seeds, lambda _utterance: None, confusion_threshold=0.0
    )
    assert findings == []


# ---------------------------------------------------------------------------
# Training preflight wiring
#
# R9b shipped as a module with no call site: `duplicate_detection.py` was complete and
# tested, and nothing in the package ever called it, so a developer had no way to run it.
# These tests exist to keep that from silently recurring. Duplicate validation now belongs
# to `fastworkflow train` rather than a second command the user has to discover and run.
#
# Wiring it also surfaced two defects that no unit test could have caught: the scan needs
# `fastworkflow.init()` before the routing definition resolves (it failed with
# "'NoneType' object has no attribute 'get_definition'"), and `find_confusable_commands`
# takes a trained model's predict_fn, so it cannot run in a pre-training scan at all.
# ---------------------------------------------------------------------------

import subprocess
import sys as _sys

from fastworkflow.train.__main__ import _validate_command_inputs


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_sys.executable, "-m", "fastworkflow.cli", *args],
        capture_output=True, text=True, timeout=300,
    )


def test_duplicate_validation_is_integrated_in_train_not_a_second_cli_command():
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr
    assert "duplicates" not in result.stdout
    assert "report" not in result.stdout
    assert "versions" not in result.stdout


def test_training_preflight_finds_the_control_workflow_duplicate_pair(
    control_workflow, capsys
):
    """The positive control's deliberate duplicate must survive the whole path: env
    resolution, init, seed collection, the scan, and rendering."""
    report = _validate_command_inputs(control_workflow)
    rendered = capsys.readouterr().out
    reported_pairs = {
        tuple(sorted((finding.command_a, finding.command_b)))
        for finding in report.duplicates
    }
    assert reported_pairs == {CONTROL_DUPLICATE_PAIR}
    for command in CONTROL_DUPLICATE_PAIR:
        assert command in rendered, (
            f"{command} missing from the report; the control pair is what this scan exists "
            f"to catch"
        )
    assert rendered.count("DUPLICATE CAPABILITIES") == 1
    assert "knowingly accept them as duplicates" in rendered
    assert "legitimate neighbours or opposites" not in rendered
    assert "No amount of utterance engineering" not in rendered


def test_training_preflight_writes_the_duplicate_report(control_workflow):
    _validate_command_inputs(control_workflow)
    report_path = os.path.join(
        control_workflow, "___command_info", "duplicate_capabilities.json"
    )
    assert os.path.isfile(report_path)

    with open(report_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["schema_version"] == 1
    duplicate_pairs = {
        tuple(sorted((finding["command_a"], finding["command_b"])))
        for finding in payload["report"]["duplicates"]
    }
    assert duplicate_pairs == {CONTROL_DUPLICATE_PAIR}


def test_training_preflight_is_advisory_for_duplicate_and_clean_workflows(
    control_workflow, retail_workflow
):
    """The scan must preserve both its diagnostic and its reports-only contract."""
    _validate_command_inputs(control_workflow)
    _validate_command_inputs(retail_workflow)


def test_a_non_workflow_path_fails_cleanly_rather_than_crashing():
    result = _run_cli("train", os.path.join("tests", "does_not_exist_at_all"))
    assert result.returncode != 0
    assert "Traceback" not in result.stdout
