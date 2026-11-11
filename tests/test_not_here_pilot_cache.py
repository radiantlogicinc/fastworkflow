"""The pilot driver's cache handling: copy only the two caches, copy (never edit) the env,
and count expected cache hits by slug. Tmp fixtures only; nothing is trained."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastworkflow.train import not_here_pilot as pilot
from fastworkflow.train.utterance_cache import slugify


@pytest.fixture
def owner(tmp_path) -> Path:
    """A stand-in for the owner's checkout: caches, trained heads, env and passwords."""
    root = tmp_path / "owner"
    info = root / "___command_info"
    (info / "utterance_cache").mkdir(parents=True)
    (info / "param_example_cache").mkdir(parents=True)
    (info / "Account" / "trained").mkdir(parents=True)  # a trained head: must NOT be copied
    (info / "utterance_cache" / "Account_list.v1.json").write_text("{}")
    (info / "utterance_cache" / "Account_list.v2.json").write_text("{}")
    (info / "utterance_cache" / "Stale_gone.v1.json").write_text("{}")
    (info / "utterance_cache" / f"{slugify('IntentDetection/go_up')}.v1.json").write_text("{}")
    (info / "param_example_cache" / "p.json").write_text("{}")
    (root / "fastworkflow.env").write_text("LLM_SYNDATA_GEN=bedrock/x\n")
    (root / "fastworkflow.passwords.env").write_text("LITELLM_API_KEY_SYNDATA_GEN=secret\n")
    return root


@pytest.fixture
def worktree(tmp_path) -> Path:
    root = tmp_path / "worktree"
    (root / "_commands" / "Account").mkdir(parents=True)
    (root / "_commands" / "Account" / "list.py").write_text("class Signature:\n    plain_utterances = []\n")
    (root / "_commands" / "root_plan.py").write_text("class PlanGenerator:\n    pass\n")  # planning: skipped
    return root


def test_only_the_two_caches_are_copied(owner, tmp_path):
    destination = tmp_path / "scratch_copy"
    (destination / "___command_info").mkdir(parents=True)
    copied = pilot.copy_caches(str(owner), str(destination))
    assert copied == ["utterance_cache", "param_example_cache"]
    target = destination / "___command_info"
    assert (target / "utterance_cache" / "Account_list.v1.json").is_file()
    assert (target / "param_example_cache" / "p.json").is_file()
    assert not (target / "Account").exists()  # trained heads are never copied


def test_the_default_env_is_a_copy_and_the_owner_file_is_untouched(owner, tmp_path):
    before = (owner / "fastworkflow.env").read_text()
    scratch = tmp_path / "scratch_env"
    copied = pilot.copy_env(str(owner), str(scratch))
    assert Path(copied).read_text() == before
    Path(copied).write_text("changed in the copy\n")
    assert (owner / "fastworkflow.env").read_text() == before
    assert pilot.default_passwords(str(owner)) == str(owner / "fastworkflow.passwords.env")


def test_expected_hits_count_worktree_and_core_commands_and_name_the_stale_files(owner, worktree):
    report = pilot.expected_cache_hits(str(worktree), str(owner))
    assert report["worktree_execution_commands"] == 1          # Account/list; the planning file is skipped
    assert report["core_commands_in_every_head"] == 4           # the four IntentDetection core commands
    assert report["files_matched_to_worktree_commands"] == 2    # both Account_list variants
    assert report["core_command_files_matched"] == 1
    assert report["files_not_matching_any_worktree_command"] == 1  # Stale_gone
    assert report["distinct_commands_with_a_matching_file"] == 2
    assert report["param_example_cache_files"] == 1


def test_the_dry_run_touches_nothing_in_the_owner_or_worktree(owner, worktree, tmp_path):
    import subprocess
    import sys

    scratch = tmp_path / "dry"
    before = sorted(str(p) for p in owner.rglob("*"))
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.train.not_here_pilot", "--arm", "sparse_1",
         "--scratch", str(scratch), "--source", str(worktree), "--cache-from", str(owner), "--dry-run"],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Expected cache hits" in result.stdout
    assert not scratch.exists()
    assert sorted(str(p) for p in owner.rglob("*")) == before


def test_copy_workflow_drops_trained_artifacts_and_keeps_the_caches(tmp_path):
    """A source whose ___command_info links into versions/ must copy without those links.

    The copy must not contain versions/, current, global, the per-context links, current.json
    or the publication lock; the caches are copied separately by copy_caches. No training.
    """
    source = tmp_path / "source"
    info = source / "___command_info"
    version = info / "versions" / "20260101T000000Z-abc"
    (version / "Book").mkdir(parents=True)
    (version / "Book" / "model.bin").write_text("trained")
    (version / "global").mkdir(parents=True)
    (info / "current").symlink_to(os.path.join("versions", version.name))
    (info / "global").symlink_to(os.path.join("versions", version.name, "global"))
    (info / "Book").symlink_to(os.path.join("versions", version.name, "Book"))
    (info / "current.json").write_text("{}")
    (info / ".publication.lock").write_text("")
    (info / "utterance_cache").mkdir()
    (info / "utterance_cache" / "u.json").write_text("{}")
    (source / "_commands").mkdir()
    (source / "_commands" / "list.py").write_text("class Signature:\n    pass\n")

    destination = tmp_path / "copy"
    pilot.copy_workflow(str(source), str(destination))
    copied_info = destination / "___command_info"
    assert not (copied_info / "versions").exists()
    assert not (copied_info / "current").exists()
    assert not (copied_info / "global").exists()
    assert not (copied_info / "Book").exists()
    assert not (copied_info / "current.json").exists()
    assert not (copied_info / ".publication.lock").exists()
    assert (copied_info / "utterance_cache").is_dir()
    assert (destination / "_commands" / "list.py").is_file()

    # The caches still come from --cache-from, and only the two cache directories.
    owner = tmp_path / "owner"
    (owner / "___command_info" / "utterance_cache").mkdir(parents=True)
    (owner / "___command_info" / "utterance_cache" / "Account_list.v1.json").write_text("{}")
    (owner / "___command_info" / "param_example_cache").mkdir()
    assert pilot.copy_caches(str(owner), str(destination)) == ["utterance_cache", "param_example_cache"]
    assert (copied_info / "utterance_cache" / "Account_list.v1.json").is_file()
    assert (copied_info / "param_example_cache").is_dir()
    assert not (copied_info / "versions").exists()
