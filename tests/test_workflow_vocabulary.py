"""The workflow vocabulary: built by train, read at runtime, derived when the stored file is absent.

Every test works on a temporary copy of the tutorial workflow's source (its ``___command_info``
is left out of the copy), so the real ``___command_info`` is never read for writing and never
written. No training job runs: the builder is called directly, as train calls it.
"""
import json
import shutil
from pathlib import Path

import pytest

from fastworkflow.command_directory import compute_commands_source_fingerprint
from fastworkflow.train import artifact_versioning
from fastworkflow.train.__main__ import _prune_stale_artifacts
from fastworkflow.utils import workflow_vocabulary as wv

TUTORIAL = Path(__file__).resolve().parent / "skill_tutorial_workflow"


@pytest.fixture
def tutorial_copy(tmp_path):
    """A copy of the tutorial workflow's source, without any ``___command_info``."""
    copy = tmp_path / "tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    return str(copy)


def _stored(folder: str) -> dict:
    return json.loads(wv.vocabulary_path(folder).read_text(encoding="utf-8"))


def _store(folder: str, payload: dict) -> None:
    wv.vocabulary_path(folder).parent.mkdir(parents=True, exist_ok=True)
    wv.vocabulary_path(folder).write_text(json.dumps(payload), encoding="utf-8")


def test_the_builder_stores_the_vocabulary_at_the_top_of_command_info(tutorial_copy):
    path = wv.write_workflow_vocabulary(tutorial_copy)

    assert path == Path(tutorial_copy) / "___command_info" / "workflow_vocabulary.json"
    payload = _stored(tutorial_copy)
    assert payload["format"] == wv.FORMAT_VERSION
    assert payload["source_fingerprint"] == compute_commands_source_fingerprint(tutorial_copy)
    assert payload["pronouns"] == sorted(wv.PRONOUNS)
    words = set(payload["words"])
    assert words == {word.lower() for word in words}
    # The command names and their parts, the field names, the field stems and their record forms.
    assert {"find_member", "member", "author", "title", "book_uid", "book uid", "book_name"} <= words
    # Declared choices of another input (find_record's kind Literal) are words too.
    assert {"book", "member"} <= words
    # Nothing versioned: the stored file is the only thing the builder writes.
    assert not (Path(tutorial_copy) / "___command_info" / "versions").exists()


def test_the_loader_reads_the_stored_file_when_it_is_fresh(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    payload = _stored(tutorial_copy)
    payload["words"] = sorted([*payload["words"], "sentinel_word"])
    _store(tutorial_copy, payload)  # the same fingerprint: the file is what the loader must read

    vocabulary = wv.load_workflow_vocabulary(tutorial_copy)

    assert vocabulary.origin == "file"
    assert vocabulary.has_word("Sentinel_Word")
    assert vocabulary.source_fingerprint == payload["source_fingerprint"]


def test_the_loader_derives_in_memory_when_the_file_is_absent_and_writes_nothing(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    expected = frozenset(_stored(tutorial_copy)["words"])
    wv.vocabulary_path(tutorial_copy).unlink()

    vocabulary = wv.load_workflow_vocabulary(tutorial_copy)

    assert vocabulary.origin == "derived"
    assert vocabulary.words == expected  # the in-memory derivation agrees with the builder
    assert vocabulary.pronouns == wv.PRONOUNS
    assert not wv.vocabulary_path(tutorial_copy).exists()


def test_the_loader_derives_in_memory_when_the_stored_file_is_stale_and_leaves_it_alone(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    payload = _stored(tutorial_copy)
    payload["words"] = sorted([*payload["words"], "sentinel_word"])
    _store(tutorial_copy, payload)
    stored_bytes = wv.vocabulary_path(tutorial_copy).read_bytes()
    skill = Path(tutorial_copy) / "_skills" / "author_books.py"
    skill.write_text(skill.read_text(encoding="utf-8") + "\n# an edit after the last train\n", encoding="utf-8")

    vocabulary = wv.load_workflow_vocabulary(tutorial_copy)

    assert vocabulary.origin == "derived"
    assert not vocabulary.has_word("sentinel_word")
    assert vocabulary.source_fingerprint == compute_commands_source_fingerprint(tutorial_copy)
    assert wv.vocabulary_path(tutorial_copy).read_bytes() == stored_bytes


def test_the_loader_derives_in_memory_when_the_stored_file_has_another_format(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    payload = _stored(tutorial_copy)
    payload["format"] = wv.FORMAT_VERSION + 1
    payload["words"] = ["sentinel_word"]
    _store(tutorial_copy, payload)

    vocabulary = wv.load_workflow_vocabulary(tutorial_copy)

    assert vocabulary.origin == "derived"
    assert not vocabulary.has_word("sentinel_word")


def test_the_loader_returns_the_same_result_for_a_workflow_path_while_its_source_is_unchanged(tutorial_copy):
    first = wv.load_workflow_vocabulary(tutorial_copy)

    assert wv.load_workflow_vocabulary(tutorial_copy) is first


def test_pruning_removes_orphans_and_leaves_the_vocabulary_file_alone(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    stored_bytes = wv.vocabulary_path(tutorial_copy).read_bytes()
    orphan = Path(tutorial_copy) / "___command_info" / "ghost_command_param_labeled.json"
    orphan.write_text("{}", encoding="utf-8")

    _prune_stale_artifacts(tutorial_copy)

    assert not orphan.exists()  # the prune ran and removed the orphan it is meant to remove
    assert wv.vocabulary_path(tutorial_copy).read_bytes() == stored_bytes


def test_migrating_legacy_artifacts_leaves_the_vocabulary_file_at_the_top(tutorial_copy):
    wv.write_workflow_vocabulary(tutorial_copy)
    stored_bytes = wv.vocabulary_path(tutorial_copy).read_bytes()
    legacy = Path(tutorial_copy) / "___command_info" / "global"
    legacy.mkdir(parents=True)
    (legacy / "threshold.json").write_text("{}", encoding="utf-8")

    version_id = artifact_versioning.migrate_legacy_to_version(tutorial_copy)

    assert version_id is not None
    assert not legacy.exists() or legacy.is_symlink()  # the legacy tree moved into the version
    assert wv.vocabulary_path(tutorial_copy).read_bytes() == stored_bytes
    assert not (Path(tutorial_copy) / "___command_info" / "versions" / version_id / "workflow_vocabulary.json").exists()
