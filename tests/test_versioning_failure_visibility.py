"""A suppressed failure that changes what ships must say so (bd fix-k0i.33, .41, .32).

No mocks (repo rule `.cursor/rules/testing_rules.mdc`): every failure below is provoked
with the real filesystem — a directory replaced by a file so a real `mkdir` raises, a
manifest overwritten with real invalid JSON, a real second OS process holding a real
`flock`. The functions under test are the shipped ones, called with real arguments.

Three findings meet here because they share a shape: something that could go wrong
silently and change the artifacts a training run publishes.

* fix-k0i.33 — a `contextlib.suppress(OSError)` with no log around the held-out report
  write and the provenance save, and `read_manifest` returning `{}` for a corrupt file so
  that retention read damage as "there is no previous version" and pruned the recovery
  point.
* fix-k0i.41 — nothing serialised publish-and-prune across processes.
* fix-k0i.32 — the versions table pointed developers at a `versions prune` command that
  the 2026-08-03 UX decision removed.
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import determinism, heldout_evaluation
from fastworkflow.utils.logging import logger as fastworkflow_logger
from fastworkflow.train.__main__ import (
    _repair_noop_publication,
    _save_run_provenance,
)
from fastworkflow.train.selective_training import TrainingPlan
from fastworkflow.model_pipeline_training import (
    TrainingDataError,
    _write_heldout_report,
)


def _write_context_artifacts(context_dir: Path, confidence: float) -> None:
    """Create the artifact set `is_workflow_trained` and `CommandRouter` look for."""
    context_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "threshold.json",
        "tiny_ambiguous_threshold.json",
        "large_ambiguous_threshold.json",
    ):
        (context_dir / name).write_text(
            json.dumps({"confidence_threshold": confidence}), encoding="utf-8"
        )
    (context_dir / "label_encoder.pkl").write_bytes(b"not-a-real-pickle")


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    wf = tmp_path / "visibility_workflow"
    (wf / "_commands").mkdir(parents=True)
    return wf


@pytest.fixture
def fastworkflow_logs(caplog):
    """Capture the real `fastWorkflow` logger, which does not propagate to root.

    `caplog` alone sees nothing here: `utils/logging.py` sets `propagate = False` so its
    records never reach the root handler pytest installs. Attaching pytest's own capture
    handler to the real logger keeps these tests asserting on the shipped logging path
    rather than on a substitute for it — the whole point of the finding is that these
    failures must be *visible*, so the test has to look where a developer would.
    """
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(logging.DEBUG)
    fastworkflow_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        fastworkflow_logger.removeHandler(caplog.handler)
        fastworkflow_logger.setLevel(previous_level)


def _make_version(workflow: Path, contexts: dict[str, float], **manifest) -> str:
    version_id = av.new_version_id()
    for context_name, confidence in contexts.items():
        _write_context_artifacts(
            av.context_artifact_dir(str(workflow), version_id, context_name),
            confidence,
        )
    av.write_manifest(str(workflow), version_id, **manifest)
    return version_id


# ---------------------------------------------------------------------
# fix-k0i.33 (1): the held-out report write
# ---------------------------------------------------------------------


def test_failed_heldout_report_write_is_reported_not_suppressed(
    tmp_path: Path, fastworkflow_logs, capsys
):
    """A real unwritable destination must produce an ERROR naming the consequence.

    Provoked for real: `___command_info` is a FILE, so `write_report`'s `mkdir` raises
    `FileExistsError`. Before this, the write was wrapped in `contextlib.suppress(OSError)`
    with no log at all — the run printed its scores, claimed success, and left
    `heldout_evaluation.json` stale, which is the input the next selective run merges
    against.
    """
    workflow = tmp_path / "unwritable"
    workflow.mkdir()
    (workflow / "___command_info").write_text("not a directory", encoding="utf-8")

    reports = [heldout_evaluation.HeldoutReport(context="*", in_distribution_f1=0.5)]
    assert _write_heldout_report(str(workflow), reports) is None

    logged = fastworkflow_logs.text
    assert "held-out evaluation report" in logged
    assert "heldout_evaluation.json is stale or absent" in logged
    assert "wrong baseline" in logged, (
        "the log must name the downstream consequence, not just the errno"
    )
    assert "WARNING" in capsys.readouterr().out, (
        "a developer watching the run must see it without reading the log file"
    )


def test_successful_heldout_report_write_still_returns_the_path(tmp_path: Path):
    """The happy path must be untouched: same file, same announcement."""
    workflow = tmp_path / "writable"
    workflow.mkdir()
    reports = [heldout_evaluation.HeldoutReport(context="*", in_distribution_f1=0.5)]

    written = _write_heldout_report(str(workflow), reports)
    assert written is not None
    assert Path(written).name == heldout_evaluation.REPORT_FILENAME
    assert json.loads(Path(written).read_text(encoding="utf-8"))["contexts"]


# ---------------------------------------------------------------------
# fix-k0i.33 (2): the provenance save
# ---------------------------------------------------------------------


def test_failed_provenance_save_refuses_to_reach_the_publish_gate(
    tmp_path: Path, fastworkflow_logs
):
    """A provenance save that cannot happen must stop the run, not be suppressed.

    `_require_publishable_training_report` reads `training_provenance.json`. With the save
    suppressed, that file still held the PREVIOUS run's records, so the gate passed or
    failed on data describing different artifacts and then published this run's models
    anyway. Refusing leaves the previous version current and complete.
    """
    workflow = tmp_path / "no_provenance"
    workflow.mkdir()
    (workflow / "___command_info").write_text("not a directory", encoding="utf-8")

    recorder = determinism.ProvenanceRecorder(str(workflow))
    with pytest.raises(TrainingDataError) as excinfo:
        _save_run_provenance(
            str(workflow),
            "20260101T000000Z-aaaaaa",
            recorder,
            TrainingPlan(),
            None,
        )

    assert "provenance" in str(excinfo.value).lower()
    assert "previous run" in str(excinfo.value)
    assert "Could not save training provenance" in fastworkflow_logs.text


def test_provenance_copy_into_the_version_is_logged_but_not_fatal(
    workflow: Path, fastworkflow_logs
):
    """The version's self-describing copy is worth an ERROR, not a discarded run.

    The top-level provenance for THIS run is already correct here, so the gate and the
    report are sound and hours of trained models are worth publishing. What is lost is the
    version's ability to say which seed produced it once the next run overwrites the
    top-level file, so that is what the log has to name.
    """
    # A real cause, deterministic on any POSIX filesystem: the version "directory" is a
    # regular file, so writing a path underneath it raises NotADirectoryError.
    version_id = av.new_version_id()
    av.ensure_versions_root(str(workflow))
    av.version_dir(str(workflow), version_id).write_text("not a directory", encoding="utf-8")

    recorder = determinism.ProvenanceRecorder(str(workflow))
    provenance_path = _save_run_provenance(
        str(workflow), version_id, recorder, TrainingPlan(), None
    )

    assert os.path.isfile(provenance_path), "the top-level provenance must still be there"
    logged = fastworkflow_logs.text
    assert "Could not copy training provenance into artifact version" in logged
    assert "seed and personas" in logged


# ---------------------------------------------------------------------
# fix-k0i.33 (3): a damaged manifest must not be read as "no previous version"
# ---------------------------------------------------------------------


def test_manifest_is_damaged_distinguishes_corrupt_from_absent(workflow: Path):
    """Absence is a normal intermediate state; damage is not. They must not be one case."""
    version_id = av.new_version_id()
    av.context_artifact_dir(str(workflow), version_id, "*")
    assert av.manifest_is_damaged(str(workflow), version_id) is False, (
        "a version whose manifest has not been written yet is not damaged"
    )

    av.write_manifest(str(workflow), version_id, seed=7)
    assert av.manifest_is_damaged(str(workflow), version_id) is False

    (av.version_dir(str(workflow), version_id) / av.MANIFEST_FILENAME).write_text(
        "{not json at all", encoding="utf-8"
    )
    assert av.manifest_is_damaged(str(workflow), version_id) is True

    (av.version_dir(str(workflow), version_id) / av.MANIFEST_FILENAME).write_text(
        '["a", "list", "not", "an", "object"]', encoding="utf-8"
    )
    assert av.manifest_is_damaged(str(workflow), version_id) is True


def test_a_damaged_manifest_does_not_prune_the_recovery_version(
    workflow: Path, fastworkflow_logs
):
    """The R4 violation this finding names: damage triggering implicit destruction.

    `previous_version` is read out of the current version's manifest. An unreadable
    manifest yields None, which used to be indistinguishable from "this is the first
    version" — and retention then deleted the one version a developer could roll back to,
    in response to a corrupt JSON file.
    """
    previous = _make_version(workflow, {"*": 0.11, "TodoItem": 0.11})
    current = _make_version(
        workflow, {"*": 0.22, "TodoItem": 0.22}, previous_version=previous
    )
    av.publish_version(str(workflow), current)

    (av.version_dir(str(workflow), current) / av.MANIFEST_FILENAME).write_text(
        "{corrupt", encoding="utf-8"
    )

    removed = av.retain_current_and_previous(str(workflow), None)

    assert removed == []
    assert av.version_dir(str(workflow), previous).is_dir(), (
        "the recovery point was destroyed because a manifest could not be parsed"
    )
    assert "unreadable" in fastworkflow_logs.text
    assert "recovery point" in fastworkflow_logs.text


def test_the_noop_training_path_keeps_the_recovery_version_too(workflow: Path):
    """The exact route the finding cites: `_repair_noop_publication` on an up-to-date run.

    It derives `previous_version` from the current manifest and hands it straight to
    retention, so a damaged manifest there is what actually reached the prune.
    """
    previous = _make_version(workflow, {"*": 0.11})
    current = _make_version(workflow, {"*": 0.22}, previous_version=previous)
    av.publish_version(str(workflow), current)
    (av.version_dir(str(workflow), current) / av.MANIFEST_FILENAME).write_text(
        "}{", encoding="utf-8"
    )

    _repair_noop_publication(str(workflow), current)

    surviving = {info.version_id for info in av.list_versions(str(workflow))}
    assert surviving == {previous, current}
    assert av.resolve_current_version(str(workflow)) == current


def test_retention_still_prunes_when_the_manifest_is_readable(workflow: Path):
    """The guard must not become a licence to never prune anything.

    Without this, "keep everything" would pass the test above and quietly reintroduce the
    unbounded artifact growth versioning exists to bound.
    """
    oldest = _make_version(workflow, {"*": 0.1})
    previous = _make_version(workflow, {"*": 0.2})
    current = _make_version(workflow, {"*": 0.3}, previous_version=previous)
    av.publish_version(str(workflow), current)

    assert av.retain_current_and_previous(str(workflow), previous) == [oldest]
    surviving = {info.version_id for info in av.list_versions(str(workflow))}
    assert surviving == {previous, current}


# ---------------------------------------------------------------------
# fix-k0i.41: the cross-process publication lock
# ---------------------------------------------------------------------

_HOLD_LOCK_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from fastworkflow.train import artifact_versioning as av

    workflow = sys.argv[1]
    with av.publication_lock(workflow):
        print("held", flush=True)
        time.sleep(float(sys.argv[2]))
    """
)


def test_a_second_process_cannot_publish_while_the_first_holds_the_lock(
    workflow: Path, tmp_path: Path
):
    """The finding's actual scenario, with a real second OS process.

    Two `fastworkflow train` runs interleaving publish and prune is a cross-process race,
    so nothing short of a second process tests it. The child holds the lock; this process
    must refuse rather than proceed into the critical section.
    """
    av.command_info_root(str(workflow)).mkdir(parents=True, exist_ok=True)
    script = tmp_path / "hold_lock.py"
    script.write_text(_HOLD_LOCK_SCRIPT, encoding="utf-8")

    child = subprocess.Popen(
        [sys.executable, str(script), str(workflow), "30"],
        stdout=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    try:
        assert child.stdout.readline().strip() == "held", "child never took the lock"
        with pytest.raises(av.PublicationLockTimeout) as excinfo:
            with av.publication_lock(str(workflow), timeout=0.5):
                pytest.fail("acquired a lock another process was holding")
        assert "interleave pointer writes" in str(excinfo.value)
    finally:
        child.kill()
        child.wait(timeout=30)

    # And the lock is available again the moment the holder is gone: `flock` is owned by
    # the open file description, so a killed process cannot strand it.
    with av.publication_lock(str(workflow), timeout=5):
        pass


def test_the_lock_is_reentrant_so_nesting_cannot_self_deadlock(workflow: Path):
    """`retain_current_and_previous` takes the lock and runs inside a caller holding it."""
    with av.publication_lock(str(workflow), timeout=0.5):
        with av.publication_lock(str(workflow), timeout=0.5):
            version_id = _make_version(workflow, {"*": 0.1})
            av.publish_version(str(workflow), version_id)
            assert av.retain_current_and_previous(str(workflow), None) == []


def test_the_lock_is_released_when_the_critical_section_raises(workflow: Path):
    """A failed publish must not leave the workflow unpublishable until a restart."""
    with pytest.raises(RuntimeError):
        with av.publication_lock(str(workflow), timeout=0.5):
            raise RuntimeError("publish blew up")

    with av.publication_lock(str(workflow), timeout=0):
        pass


def test_the_lock_file_is_not_mistaken_for_an_artifact(workflow: Path):
    """It lives in `___command_info`, so every entry sweep there must skip it."""
    version_id = _make_version(workflow, {"*": 0.1, "TodoItem": 0.1})
    with av.publication_lock(str(workflow)):
        av.publish_version(str(workflow), version_id)

    lock_file = av.command_info_root(str(workflow)) / av.PUBLICATION_LOCK_FILENAME
    assert lock_file.is_file(), "precondition: the lock file exists after an acquisition"
    assert av.legacy_layout_in_use(str(workflow)) is False
    assert av.resolve_current_version(str(workflow)) == version_id
    assert av.PUBLICATION_LOCK_FILENAME not in av.version_context_names(
        str(workflow), version_id
    )
    # A second publish must not trip over it while sweeping stale compatibility entries.
    av.publish_version(str(workflow), version_id)
    assert lock_file.is_file()


# fix-k0i.32's remaining assertion lived here: it pinned the corrected retention
# footer inside `format_versions_table`. fix-k0i.50 deleted that function as
# test-only code, which supersedes the fix -- a footer cannot cite a cut command
# if it does not exist. The retention policy it described is asserted for real by
# the pruning tests above.
