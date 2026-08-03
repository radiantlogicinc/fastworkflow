"""Integration tests for versioned training artifacts (spec R4 / bd fix-551.7).

No mocks (repo rule `.cursor/rules/testing_rules.mdc`): every test builds a real
`___command_info` tree of real files under `tmp_path` and exercises the real
filesystem primitives — symlink swaps, hardlinks, `shutil.move`. The artifacts are
small stand-ins for the 276 MB-per-context real thing; the *layout* is faithful.

The single most important test here is
`test_old_literal_reader_path_still_opens_after_publish`: every existing reader in
the package builds `<workflow>/___command_info/<Context>/threshold.json` by hand
(`intent_detection.py:39` does it as an f-string), and a layout change that breaks
that path produces the repo's best-known failure signature — FileNotFoundError on
threshold.json.
"""

import json
import os
from pathlib import Path

import pytest

from fastworkflow.train import artifact_versioning as av

# Mirrors the real per-context artifact set (see model_pipeline_training.train:
# tinymodel.pth / largemodel.pth are save_pretrained *directories*).
_THRESHOLD_FILES = (
    "threshold.json",
    "tiny_ambiguous_threshold.json",
    "large_ambiguous_threshold.json",
)


def _write_context_artifacts(context_dir: Path, confidence: float) -> None:
    """Create a realistic (small) trained-context artifact set."""
    context_dir.mkdir(parents=True, exist_ok=True)
    for name in _THRESHOLD_FILES:
        (context_dir / name).write_text(
            json.dumps({"confidence_threshold": confidence}), encoding="utf-8"
        )
    (context_dir / "label_encoder.pkl").write_bytes(b"not-a-real-pickle")
    for model_dir_name in ("tinymodel.pth", "largemodel.pth"):
        model_dir = context_dir / model_dir_name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps({"threshold_marker": confidence}), encoding="utf-8"
        )
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 512)


def _make_version(workflow: Path, contexts: dict[str, float]) -> str:
    """Create a version holding *contexts* (name -> confidence marker) and return its id."""
    version_id = av.new_version_id()
    for context_name, confidence in contexts.items():
        _write_context_artifacts(
            av.context_artifact_dir(str(workflow), version_id, context_name),
            confidence,
        )
    av.write_manifest(str(workflow), version_id, seed=42)
    return version_id


def _legacy_threshold_path(workflow: Path, context_folder: str) -> str:
    """The literal path form every pre-versioning reader builds."""
    return f"{workflow}/___command_info/{context_folder}/threshold.json"


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    wf = tmp_path / "my_workflow"
    (wf / "_commands").mkdir(parents=True)
    return wf


# ---------------------------------------------------------------------
# Constants and ids
# ---------------------------------------------------------------------


def test_global_context_folder_matches_model_pipeline_training():
    """The '*' -> folder mapping is duplicated, not imported; prove it agrees."""
    from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER

    assert av.GLOBAL_CONTEXT_FOLDER == GLOBAL_CONTEXT_FOLDER
    assert av.context_folder_name("*") == GLOBAL_CONTEXT_FOLDER
    assert av.context_folder_name("TodoItem") == "TodoItem"


def test_version_ids_are_unique_and_shaped_for_sorting():
    """Ids look like `20260802T144233Z-a1b2c3`: fixed width, timestamp-prefixed.

    Ordering *within* one second is decided by the random suffix, which is why
    `list_versions` sorts on the manifest's microsecond `created_at` instead.
    """
    ids = [av.new_version_id() for _ in range(5)]
    assert len(set(ids)) == 5
    assert len({len(i) for i in ids}) == 1
    assert all(i[8] == "T" and i[15] == "Z" and i[16] == "-" for i in ids)


@pytest.mark.parametrize(
    "bad", ["../escape", "a/b", "", ".hidden", "versions", "current"]
)
def test_invalid_version_ids_are_rejected(workflow: Path, bad: str):
    with pytest.raises(ValueError):
        av.version_dir(str(workflow), bad)


# ---------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------


def test_publish_routes_per_context_compat_path_to_the_version(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.11, "TodoItem": 0.22})
    av.publish_version(str(workflow), version_id)

    assert av.resolve_current_version(str(workflow)) == version_id

    for context_folder, expected in (("global", 0.11), ("TodoItem", 0.22)):
        compat = av.command_info_root(str(workflow)) / context_folder
        assert compat.is_dir(), f"{compat} must look like a directory to old readers"
        payload = json.loads((compat / "threshold.json").read_text(encoding="utf-8"))
        assert payload["confidence_threshold"] == expected
        # It really resolves into the version, not a copy left at the top level.
        assert Path(os.path.realpath(compat)).parent.name == version_id


def test_old_literal_reader_path_still_opens_after_publish(workflow: Path):
    """The regression that matters most: `f"{wf}/___command_info/{ctx}/threshold.json"`.

    intent_detection.py:39 builds exactly this string. If versioning breaks it, the
    runtime fails with FileNotFoundError on threshold.json.
    """
    version_id = _make_version(workflow, {"*": 0.31, "TodoItem": 0.32})
    av.publish_version(str(workflow), version_id)

    for context_folder, expected in (("global", 0.31), ("TodoItem", 0.32)):
        with open(_legacy_threshold_path(workflow, context_folder)) as f:
            assert json.load(f)["confidence_threshold"] == expected

    # CommandRouter builds five sibling paths off the same directory string; all of
    # them must resolve, including the save_pretrained model *directories*.
    model_dir = f"{workflow}/___command_info/global/tinymodel.pth"
    assert os.path.isdir(model_dir)
    assert os.path.isfile(os.path.join(model_dir, "config.json"))
    assert os.path.isfile(f"{workflow}/___command_info/global/label_encoder.pkl")


def test_chat_session_style_warmup_iteration_still_sees_contexts(workflow: Path):
    """chat_session.py:213 iterates `d for d in root.iterdir() if d.is_dir()`."""
    version_id = _make_version(workflow, {"*": 0.4, "TodoItem": 0.4})
    av.publish_version(str(workflow), version_id)

    root = av.command_info_root(str(workflow))
    subdir_names = {d.name for d in root.iterdir() if d.is_dir()}
    assert {"global", "TodoItem"} <= subdir_names


def test_os_walk_still_finds_model_dirs(workflow: Path):
    """test_train_modern_stack._find_model_dirs walks with followlinks=False.

    It therefore ignores the compatibility symlinks and finds the real directories
    inside versions/, which is why that test keeps passing unchanged.
    """
    version_id = _make_version(workflow, {"*": 0.5})
    av.publish_version(str(workflow), version_id)

    found = [
        root
        for root, dirs, files in os.walk(av.command_info_root(str(workflow)))
        if "tinymodel.pth" in dirs or "tinymodel.pth" in files
    ]
    assert found, "no trained model directory discoverable by os.walk"
    assert all(version_id in path for path in found)


def test_publishing_second_version_reroutes_and_leaves_first_intact(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.11, "TodoItem": 0.12})
    av.publish_version(str(workflow), v1)
    v2 = _make_version(workflow, {"*": 0.91, "TodoItem": 0.92})
    av.publish_version(str(workflow), v2)

    assert av.resolve_current_version(str(workflow)) == v2

    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.91

    # v1's bytes are untouched on disk — this is the rollback / paired-evaluation
    # property the whole issue exists for.
    v1_threshold = av.version_dir(str(workflow), v1) / "global" / "threshold.json"
    assert json.loads(v1_threshold.read_text())["confidence_threshold"] == 0.11
    assert {info.version_id for info in av.list_versions(str(workflow))} == {v1, v2}


def test_publish_is_idempotent(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.6})
    av.publish_version(str(workflow), version_id)
    av.publish_version(str(workflow), version_id)
    av.publish_version(str(workflow), version_id)

    assert av.resolve_current_version(str(workflow)) == version_id
    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.6


def test_current_pointer_is_the_final_publish_commit_point(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
):
    """A pre-commit failure keeps current.json old while an idempotent retry heals."""
    v1 = _make_version(workflow, {"*": 0.1, "Retired": 0.1})
    av.publish_version(str(workflow), v1)
    v2 = _make_version(workflow, {"*": 0.9, "TodoItem": 0.9})
    info = av.command_info_root(str(workflow))
    original_write_pointer = getattr(av, "_write_pointer")

    def fail_at_pointer_commit(
        workflow_folderpath: str, version_id: str, layout: str
    ) -> None:
        # Every reader path, including stale-entry removal, must be prepared before
        # current.json is touched. Raising here simulates the last pre-commit fault.
        assert version_id == v2
        assert layout == "symlink"
        assert (
            json.loads(av.pointer_path(workflow_folderpath).read_text())["version_id"]
            == v1
        )
        assert Path(os.path.realpath(info / "global")).parent.name == v2
        assert Path(os.path.realpath(info / "TodoItem")).parent.name == v2
        assert (
            Path(os.path.realpath(av.current_link_path(workflow_folderpath))).name == v2
        )
        assert not (info / "Retired").exists()
        raise RuntimeError("injected failure before current.json commit")

    monkeypatch.setattr(av, "_write_pointer", fail_at_pointer_commit)
    with pytest.raises(RuntimeError, match="injected failure"):
        av.publish_version(str(workflow), v2)

    # The pointer remains authoritative and old even though preparation can leave
    # reader paths mixed. Pruning must protect both the committed and routed versions.
    assert json.loads(av.pointer_path(str(workflow)).read_text())["version_id"] == v1
    assert av.resolve_current_version(str(workflow)) == v1
    with pytest.raises(ValueError, match="currently referenced"):
        av.prune_versions(str(workflow), version_ids=[v2], dry_run=False)
    assert av.version_dir(str(workflow), v1).is_dir()
    assert av.version_dir(str(workflow), v2).is_dir()

    # Retrying the same publication is the bounded recovery mechanism in this scope.
    monkeypatch.setattr(av, "_write_pointer", original_write_pointer)
    av.publish_version(str(workflow), v2)
    assert av.resolve_current_version(str(workflow)) == v2
    with open(_legacy_threshold_path(workflow, "TodoItem")) as f:
        assert json.load(f)["confidence_threshold"] == 0.9


def test_publish_removes_compat_entry_for_a_context_the_new_version_lacks(
    workflow: Path,
):
    v1 = _make_version(workflow, {"*": 0.1, "Retired": 0.1})
    av.publish_version(str(workflow), v1)
    assert (av.command_info_root(str(workflow)) / "Retired").exists()

    v2 = _make_version(workflow, {"*": 0.2})
    av.publish_version(str(workflow), v2)

    retired = av.command_info_root(str(workflow)) / "Retired"
    assert not retired.exists() and not retired.is_symlink()
    # The bytes still exist in v1: the entry was unrouted, not destroyed.
    assert (av.version_dir(str(workflow), v1) / "Retired" / "threshold.json").is_file()


def test_publish_refuses_when_an_unversioned_directory_blocks_the_name(workflow: Path):
    info = av.command_info_root(str(workflow))
    _write_context_artifacts(info / "TodoItem", 0.7)
    version_id = _make_version(workflow, {"TodoItem": 0.8})

    with pytest.raises(av.LegacyArtifactsPresentError) as excinfo:
        av.publish_version(str(workflow), version_id)
    assert "migrate_legacy_to_version" in str(excinfo.value)
    # Nothing was destroyed.
    assert (
        json.loads((info / "TodoItem" / "threshold.json").read_text())[
            "confidence_threshold"
        ]
        == 0.7
    )


def test_publish_missing_version_raises(workflow: Path):
    with pytest.raises(FileNotFoundError):
        av.publish_version(str(workflow), av.new_version_id())


def test_publish_falls_back_to_a_real_directory_when_symlinks_are_unavailable(
    workflow: Path,
):
    """The Windows / restricted-filesystem path, which Linux never takes naturally.

    Forced by pre-seeding the platform probe cache with False — that is a
    configuration value, not a mock: every filesystem operation below is real.
    """
    info = av.command_info_root(str(workflow))
    info.mkdir(parents=True, exist_ok=True)
    av._symlink_support[str(info)] = False
    try:
        v1 = _make_version(workflow, {"*": 0.81, "TodoItem": 0.82})
        av.publish_version(str(workflow), v1)

        compat = info / "global"
        assert compat.is_dir() and not compat.is_symlink()
        assert (compat / av.COMPAT_MARKER_FILENAME).is_file()
        # No `current` symlink is created in this mode; the pointer file answers.
        assert not av.current_link_path(str(workflow)).exists()
        assert av.resolve_current_version(str(workflow)) == v1
        with open(_legacy_threshold_path(workflow, "global")) as f:
            assert json.load(f)["confidence_threshold"] == 0.81

        # A materialised entry must still be replaceable by the next publish.
        v2 = _make_version(workflow, {"*": 0.95, "TodoItem": 0.96})
        av.publish_version(str(workflow), v2)
        assert av.resolve_current_version(str(workflow)) == v2
        with open(_legacy_threshold_path(workflow, "TodoItem")) as f:
            assert json.load(f)["confidence_threshold"] == 0.96
        # v1 survives, and the materialised entry was not mistaken for legacy.
        assert av.legacy_layout_in_use(str(workflow)) is False
        assert (
            json.loads(
                (
                    av.version_dir(str(workflow), v1) / "global" / "threshold.json"
                ).read_text()
            )["confidence_threshold"]
            == 0.81
        )
    finally:
        av._symlink_support.pop(str(info), None)


# ---------------------------------------------------------------------
# Pointer resolution
# ---------------------------------------------------------------------


def test_resolve_current_version_works_from_pointer_when_symlink_is_missing(
    workflow: Path,
):
    version_id = _make_version(workflow, {"*": 0.15})
    av.publish_version(str(workflow), version_id)

    link = av.current_link_path(str(workflow))
    assert link.is_symlink(), "expected a symlink on this platform"
    os.unlink(link)
    assert not link.exists()

    assert av.resolve_current_version(str(workflow)) == version_id
    # Per-context entries point straight at the version, so losing `current` does
    # not break any reader either.
    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.15


def test_resolve_current_version_falls_back_to_symlink_when_pointer_is_gone(
    workflow: Path,
):
    version_id = _make_version(workflow, {"*": 0.16})
    av.publish_version(str(workflow), version_id)

    os.unlink(av.pointer_path(str(workflow)))
    assert av.resolve_current_version(str(workflow)) == version_id


def test_resolve_current_version_ignores_a_pointer_to_a_vanished_version(
    workflow: Path,
):
    av.ensure_versions_root(str(workflow))
    av.pointer_path(str(workflow)).write_text(
        json.dumps({"version_id": "20200101T000000Z-deadbe"}), encoding="utf-8"
    )
    assert av.resolve_current_version(str(workflow)) is None


def test_resolve_current_version_is_none_for_an_unversioned_workflow(workflow: Path):
    assert av.resolve_current_version(str(workflow)) is None
    assert av.list_versions(str(workflow)) == []


# ---------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------


def test_write_manifest_merges_and_read_manifest_round_trips(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.5, "TodoItem": 0.5})

    av.write_manifest(str(workflow), version_id, seed=7, notes="first pass")
    av.write_manifest(str(workflow), version_id, train_duration_seconds=12875.0)

    manifest = av.read_manifest(str(workflow), version_id)
    assert manifest["seed"] == 7
    assert manifest["notes"] == "first pass"
    assert manifest["train_duration_seconds"] == 12875.0
    assert manifest["version_id"] == version_id
    assert sorted(manifest["contexts"]) == ["TodoItem", "global"]
    assert manifest["created_at"]


def test_read_manifest_of_unknown_version_is_empty(workflow: Path):
    assert av.read_manifest(str(workflow), av.new_version_id()) == {}


def test_list_versions_reports_current_size_and_manifest_fields(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    av.publish_version(str(workflow), v1)
    v2 = _make_version(workflow, {"*": 0.2, "TodoItem": 0.2})
    av.write_manifest(
        str(workflow), v2, seed=99, notes="capped wildcard", train_duration_seconds=7200
    )
    av.publish_version(str(workflow), v2)

    infos = av.list_versions(str(workflow))
    assert [info.version_id for info in infos] == [v2, v1], "newest first"
    latest = infos[0]
    assert latest.is_current is True
    assert infos[1].is_current is False
    assert sorted(latest.contexts) == ["TodoItem", "global"]
    assert latest.size_bytes > 0
    assert latest.seed == 99
    assert latest.notes == "capped wildcard"
    assert latest.train_duration_seconds == 7200


# ---------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------


def test_prune_versions_requires_an_explicit_request(workflow: Path):
    _make_version(workflow, {"*": 0.1})
    with pytest.raises(ValueError):
        av.prune_versions(str(workflow))
    with pytest.raises(ValueError):
        av.prune_versions(str(workflow), keep=1, version_ids=["x"])


def test_prune_versions_defaults_to_dry_run(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    v2 = _make_version(workflow, {"*": 0.2})
    av.publish_version(str(workflow), v2)

    planned = av.prune_versions(str(workflow), version_ids=[v1])
    assert planned == [v1]
    assert av.version_dir(str(workflow), v1).is_dir(), "dry run must not delete"
    assert {i.version_id for i in av.list_versions(str(workflow))} == {v1, v2}


def test_prune_versions_removes_only_what_was_asked(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    v2 = _make_version(workflow, {"*": 0.2})
    v3 = _make_version(workflow, {"*": 0.3})
    av.publish_version(str(workflow), v3)

    removed = av.prune_versions(str(workflow), version_ids=[v1], dry_run=False)
    assert removed == [v1]
    assert not av.version_dir(str(workflow), v1).exists()
    assert av.version_dir(str(workflow), v2).is_dir()
    assert av.version_dir(str(workflow), v3).is_dir()


def test_prune_versions_refuses_to_remove_the_current_version(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    av.publish_version(str(workflow), v1)

    with pytest.raises(ValueError) as excinfo:
        av.prune_versions(str(workflow), version_ids=[v1], dry_run=False)
    assert v1 in str(excinfo.value)
    assert av.version_dir(str(workflow), v1).is_dir()
    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.1


def test_prune_versions_keep_window_never_drops_the_current_version(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    v2 = _make_version(workflow, {"*": 0.2})
    v3 = _make_version(workflow, {"*": 0.3})
    # Publish the OLDEST so it sits outside a keep=1 window.
    av.publish_version(str(workflow), v1)

    planned = av.prune_versions(str(workflow), keep=1)
    assert v1 not in planned
    assert set(planned) == {v2}

    removed = av.prune_versions(str(workflow), keep=1, dry_run=False)
    assert removed == [v2]
    assert av.version_dir(str(workflow), v1).is_dir()
    assert av.version_dir(str(workflow), v3).is_dir()


def test_prune_versions_rejects_unknown_ids_and_bad_keep(workflow: Path):
    _make_version(workflow, {"*": 0.1})
    with pytest.raises(ValueError):
        av.prune_versions(str(workflow), version_ids=["20200101T000000Z-abcdef"])
    with pytest.raises(ValueError):
        av.prune_versions(str(workflow), keep=0)


def test_automatic_retention_keeps_only_current_and_previous_success(workflow: Path):
    old = _make_version(workflow, {"*": 0.1})
    previous = _make_version(workflow, {"*": 0.2})
    incomplete = av.new_version_id()
    av.write_manifest(str(workflow), incomplete, seed=42)
    current = _make_version(workflow, {"*": 0.3})
    av.publish_version(str(workflow), current)

    removed = av.retain_current_and_previous(str(workflow), previous)

    assert set(removed) == {old, incomplete}
    assert {info.version_id for info in av.list_versions(str(workflow))} == {
        previous,
        current,
    }
    assert av.resolve_current_version(str(workflow)) == current


# ---------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------


def _build_legacy_tree(workflow: Path) -> Path:
    """A realistic pre-versioning ___command_info: two context dirs plus the JSONs."""
    info = av.command_info_root(str(workflow))
    info.mkdir(parents=True, exist_ok=True)
    _write_context_artifacts(info / "global", 0.61)
    _write_context_artifacts(info / "TodoItem", 0.62)
    (info / "command_directory.json").write_text(
        json.dumps({"map_command_2_metadata": {}, "source_fingerprint": "abc"}),
        encoding="utf-8",
    )
    (info / "routing_definition.json").write_text(
        json.dumps({"contexts": {"*": ["add"], "TodoItem": ["complete"]}}),
        encoding="utf-8",
    )
    (info / "add_param_labeled.json").write_text(
        json.dumps({"command_name": "add", "valid_examples": []}), encoding="utf-8"
    )
    return info


def test_legacy_layout_detection(workflow: Path):
    assert av.legacy_layout_in_use(str(workflow)) is False
    _build_legacy_tree(workflow)
    assert av.legacy_layout_in_use(str(workflow)) is True


def test_migrate_legacy_moves_everything_and_publishes(workflow: Path):
    info = _build_legacy_tree(workflow)

    version_id = av.migrate_legacy_to_version(str(workflow))
    assert version_id is not None
    assert av.resolve_current_version(str(workflow)) == version_id

    # No artifact was lost: every file that was in the legacy context dirs is in the
    # version, with identical content.
    vdir = av.version_dir(str(workflow), version_id)
    for context_folder, expected in (("global", 0.61), ("TodoItem", 0.62)):
        assert (
            json.loads((vdir / context_folder / "threshold.json").read_text())[
                "confidence_threshold"
            ]
            == expected
        )
        assert (vdir / context_folder / "label_encoder.pkl").read_bytes() == (
            b"not-a-real-pickle"
        )
        assert (
            vdir / context_folder / "tinymodel.pth" / "model.safetensors"
        ).stat().st_size == 512

    # Old readers are unaffected.
    with open(_legacy_threshold_path(workflow, "TodoItem")) as f:
        assert json.load(f)["confidence_threshold"] == 0.62

    # Workflow-scoped JSONs stayed at the top level as real files.
    for name in (
        "command_directory.json",
        "routing_definition.json",
        "add_param_labeled.json",
    ):
        top = info / name
        assert top.is_file() and not top.is_symlink()

    manifest = av.read_manifest(str(workflow), version_id)
    assert sorted(manifest["contexts"]) == ["TodoItem", "global"]
    assert "Migrated" in manifest["notes"]


def test_migrate_legacy_is_idempotent(workflow: Path):
    _build_legacy_tree(workflow)
    first = av.migrate_legacy_to_version(str(workflow))

    assert av.migrate_legacy_to_version(str(workflow)) is None
    assert av.migrate_legacy_to_version(str(workflow)) is None

    assert av.resolve_current_version(str(workflow)) == first
    assert [i.version_id for i in av.list_versions(str(workflow))] == [first]
    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.61


def test_migrate_legacy_on_a_fresh_workflow_is_a_no_op(workflow: Path):
    assert av.migrate_legacy_to_version(str(workflow)) is None
    assert not av.versions_root(str(workflow)).exists()


def test_migrate_then_publish_a_new_version_keeps_the_migrated_one(workflow: Path):
    _build_legacy_tree(workflow)
    migrated = av.migrate_legacy_to_version(str(workflow))

    retrained = _make_version(workflow, {"*": 0.99, "TodoItem": 0.98})
    av.publish_version(str(workflow), retrained)

    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.99
    assert (
        json.loads(
            (
                av.version_dir(str(workflow), migrated) / "global" / "threshold.json"
            ).read_text()
        )["confidence_threshold"]
        == 0.61
    )


# ---------------------------------------------------------------------
# Carry-forward
# ---------------------------------------------------------------------


def test_carry_forward_context_produces_identical_readable_content(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.21, "TodoItem": 0.22})
    v2 = av.new_version_id()
    _write_context_artifacts(av.context_artifact_dir(str(workflow), v2, "*"), 0.51)

    assert av.carry_forward_context(str(workflow), v1, v2, "TodoItem") is True
    av.write_manifest(str(workflow), v2)
    av.publish_version(str(workflow), v2)

    source = av.version_dir(str(workflow), v1) / "TodoItem"
    carried = av.version_dir(str(workflow), v2) / "TodoItem"
    for relative in (
        "threshold.json",
        "label_encoder.pkl",
        "tinymodel.pth/model.safetensors",
        "largemodel.pth/config.json",
    ):
        assert (carried / relative).read_bytes() == (source / relative).read_bytes()

    # And the carried context is routable through the old reader path.
    with open(_legacy_threshold_path(workflow, "TodoItem")) as f:
        assert json.load(f)["confidence_threshold"] == 0.22
    with open(_legacy_threshold_path(workflow, "global")) as f:
        assert json.load(f)["confidence_threshold"] == 0.51


def test_carry_forward_context_maps_wildcard_to_global(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.31})
    v2 = av.new_version_id()
    assert av.carry_forward_context(str(workflow), v1, v2, "*") is True
    assert (av.version_dir(str(workflow), v2) / "global" / "threshold.json").is_file()


def test_carry_forward_context_returns_false_for_a_missing_source(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    v2 = av.new_version_id()
    assert av.carry_forward_context(str(workflow), v1, v2, "Nonexistent") is False


def test_carry_forward_context_is_idempotent(workflow: Path):
    v1 = _make_version(workflow, {"TodoItem": 0.41})
    v2 = av.new_version_id()
    assert av.carry_forward_context(str(workflow), v1, v2, "TodoItem") is True
    assert av.carry_forward_context(str(workflow), v1, v2, "TodoItem") is True
    assert (
        json.loads(
            (
                av.version_dir(str(workflow), v2) / "TodoItem" / "threshold.json"
            ).read_text()
        )["confidence_threshold"]
        == 0.41
    )


def test_carry_forward_context_with_hardlinks_disabled_copies(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
):
    """Flipping the module's hardlink flag must still produce identical content."""
    monkeypatch.setattr(av, "USE_HARDLINKS_FOR_CARRY_FORWARD", False)
    v1 = _make_version(workflow, {"TodoItem": 0.61})
    v2 = av.new_version_id()
    assert av.carry_forward_context(str(workflow), v1, v2, "TodoItem") is True

    source = av.version_dir(str(workflow), v1) / "TodoItem" / "threshold.json"
    copied = av.version_dir(str(workflow), v2) / "TodoItem" / "threshold.json"
    assert copied.read_bytes() == source.read_bytes()
    assert copied.stat().st_ino != source.stat().st_ino, "expected a real copy"


def test_pruning_a_carried_forward_source_leaves_the_new_version_readable(
    workflow: Path,
):
    """Hardlinked carry-forward must survive its donor being pruned."""
    v1 = _make_version(workflow, {"TodoItem": 0.71})
    v2 = av.new_version_id()
    av.carry_forward_context(str(workflow), v1, v2, "TodoItem")
    av.write_manifest(str(workflow), v2)
    av.publish_version(str(workflow), v2)

    assert av.prune_versions(str(workflow), version_ids=[v1], dry_run=False) == [v1]
    with open(_legacy_threshold_path(workflow, "TodoItem")) as f:
        assert json.load(f)["confidence_threshold"] == 0.71


# ---------------------------------------------------------------------
# Expensive-artifact ergonomics (R4)
# ---------------------------------------------------------------------


def test_versions_readme_warns_that_artifacts_are_expensive(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.1})
    av.publish_version(str(workflow), version_id)

    readme = av.versions_root(str(workflow)) / av.VERSIONS_README_FILENAME
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "EXPENSIVE TO REGENERATE" in text
    assert "Do not delete" in text


def test_current_pointer_carries_the_warning_and_the_version(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.1})
    av.publish_version(str(workflow), version_id)

    payload = json.loads(av.pointer_path(str(workflow)).read_text(encoding="utf-8"))
    assert payload["version_id"] == version_id
    assert payload["layout"] in {"symlink", "hardlink"}
    assert "rebuild" in payload["warning"]


def test_format_versions_table_shows_cost_and_current_marker(workflow: Path):
    v1 = _make_version(workflow, {"*": 0.1})
    v2 = _make_version(workflow, {"*": 0.2, "TodoItem": 0.2})
    av.write_manifest(str(workflow), v2, train_duration_seconds=12900, notes="v4 cap")
    av.publish_version(str(workflow), v2)

    table = av.format_versions_table(str(workflow))
    assert v1 in table and v2 in table
    assert f"* {v2}" in table, "the current version must be marked"
    assert "3h35m" in table, "build time must be shown when the manifest records it"
    assert "v4 cap" in table
    assert "KB" in table or "MB" in table
    assert "prune" in table


def test_format_versions_table_flags_an_unmigrated_legacy_layout(workflow: Path):
    _build_legacy_tree(workflow)
    table = av.format_versions_table(str(workflow))
    assert "unversioned" in table


def test_format_versions_table_on_a_fresh_workflow(workflow: Path):
    assert "No trained artifact versions" in av.format_versions_table(str(workflow))


def test_describe_version_reports_manifest_and_contexts(workflow: Path):
    version_id = _make_version(workflow, {"*": 0.1, "TodoItem": 0.1})
    av.write_manifest(str(workflow), version_id, seed=13)
    av.publish_version(str(workflow), version_id)

    described = av.describe_version(str(workflow), version_id)
    assert version_id in described
    assert "(current)" in described
    assert "seed" in described
    assert "TodoItem" in described


def test_human_helpers():
    assert av.human_size(0) == "0 B"
    assert av.human_size(1536) == "1.5 KB"
    assert av.human_size(276 * 1024 * 1024) == "276.0 MB"
    assert av.human_duration(None) == "-"
    assert av.human_duration(12900) == "3h35m"
    assert av.human_duration(7200) == "2h00m"
    assert av.human_duration(45) == "45s"
    assert av.human_age("not-a-date") == "unknown"


# ---------------------------------------------------------------------
# The hard constraint, end to end against the real reader
# ---------------------------------------------------------------------


def test_is_workflow_trained_accepts_a_migrated_versioned_workflow(workflow: Path):
    """`is_workflow_trained` reads routing_definition.json then per-context
    threshold.json off the top level. Both must keep working after migration —
    otherwise `fastworkflow run` and the MCP server refuse to start.
    """
    from fastworkflow.model_pipeline_training import is_workflow_trained

    _build_legacy_tree(workflow)
    assert is_workflow_trained(str(workflow))[0] is True, "precondition"

    version_id = av.migrate_legacy_to_version(str(workflow))
    trained, missing = is_workflow_trained(str(workflow))
    assert trained is True, f"missing after migration: {missing}"

    # And after publishing a retrained version on top.
    v2 = _make_version(workflow, {"*": 0.9, "TodoItem": 0.9})
    av.publish_version(str(workflow), v2)
    trained, missing = is_workflow_trained(str(workflow))
    assert trained is True, f"missing after publish: {missing}"
    assert version_id != v2


def test_unroute_context_removes_the_pointer_but_not_the_version_bytes(workflow: Path):
    # `_prune_stale_artifacts` must be able to drop an orphaned context without
    # destroying the trained artifacts, which cost hours of LLM + GPU time to rebuild.
    wf = str(workflow)
    version_id = _make_version(workflow, {"Alpha": 0.6, "Beta": 0.7})
    av.publish_version(wf, version_id)

    compat = av.command_info_root(wf) / "Beta"
    real = av.version_dir(wf, version_id) / "Beta"
    assert compat.exists() and real.is_dir()

    assert av.unroute_context(wf, "Beta") is True

    assert not compat.exists(), "the compatibility entry should be gone"
    assert (real / "threshold.json").is_file(), "the version's bytes must survive"
    # Alpha is untouched and still reachable through the legacy literal path.
    assert os.path.isfile(_legacy_threshold_path(workflow, "Alpha"))

    # Idempotent, and it refuses to touch the reserved top-level entries.
    assert av.unroute_context(wf, "Beta") is False
    assert av.unroute_context(wf, av.VERSIONS_DIRNAME) is False
    assert av.versions_root(wf).is_dir()

    # Republishing restores the pointer, so unrouting is recoverable.
    av.publish_version(wf, version_id)
    assert os.path.isfile(_legacy_threshold_path(workflow, "Beta"))
