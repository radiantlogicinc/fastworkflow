"""The credential-free runtime snapshot on 3.3 (fix-qe2, trimmed port).

The ido source tests pinned planner arms, packing and catalogue fields; none
of those exist here and none are ported. What is pinned instead is the
3.3-shaped snapshot: manifest identity and feature vector, the trained model
version, the observability regime, the served command count, the pid -- and,
above all, that nothing resembling a credential can reach it.
"""

from __future__ import annotations

import json
import os

import pytest

from fastworkflow.observability import store as obs
from fastworkflow.runtime_manifest import (
    RuntimeManifest,
    clear_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.experiment.readiness import (
    capture_regime,
    runtime_readiness_snapshot,
    snapshot_env_names,
    workflow_model_legacy_layout,
    workflow_model_version,
)


KEPT_FIELDS = {
    "runtime_metadata_registered",
    "effective_features",
    "has_workflow_manifest",
    "workflow_fingerprint",
    "workflow_scope_rule_version",
    "command_surface_count",
    "workflow_model_version",
    "workflow_model_legacy_layout",
    "observability_enabled",
    "pruning_suppressed",
    "capture_profile",
    "capture_profile_valid",
    "capture_policy_version",
    "pid",
    "configuration_valid",
}

# Present in the ido source; deliberately absent here (see the module
# docstring for the reason each one is missing on 3.3).
NOT_PORTED_FIELDS = {
    "plan_decomposition",
    "plan_execution_arm",
    "execution_path",
    "packing_configuration",
    "stress_mode",
    "safety_wall_seconds",
    "turn_deadline_seconds",
    "catalog_loaded",
    "catalog_mode",
    "catalog_fingerprint",
    "catalog_skill_count",
    "task_card_count",
    "selector_surface_task_only",
    "presented_result_max_bytes",
    "extraction_constants",
}


def _metadata(**manifest_fields):
    manifest = (
        RuntimeManifest(schema_version=1, manifest_version="1.0.0", **manifest_fields)
        if manifest_fields
        else None
    )
    return merge_and_gate(manifest, deployment_features={})


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_runtime_metadata()
    yield
    clear_runtime_metadata()


# ----------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------


def test_the_snapshot_has_exactly_the_kept_fields_and_none_of_the_unported_ones(
    tmp_path,
):
    snapshot = runtime_readiness_snapshot(
        str(tmp_path), metadata=_metadata(workflow_fingerprint="sha256:abc")
    )

    assert set(snapshot) == KEPT_FIELDS
    assert not (set(snapshot) & NOT_PORTED_FIELDS)
    # Everything is JSON-serialisable as-is: it is stored verbatim on the
    # attempt row and answered verbatim by the probe.
    json.dumps(snapshot)


def test_manifest_identity_features_and_command_count_come_from_the_metadata(
    tmp_path,
):
    metadata = _metadata(workflow_fingerprint="sha256:abc")

    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=metadata)

    assert snapshot["runtime_metadata_registered"] is True
    assert snapshot["has_workflow_manifest"] is True
    assert snapshot["workflow_fingerprint"] == "sha256:abc"
    assert snapshot["workflow_scope_rule_version"] == metadata.workflow_scope_rule_version
    assert snapshot["effective_features"] == dict(sorted(metadata.feature_modes.items()))
    assert list(snapshot["effective_features"]) == sorted(snapshot["effective_features"])
    assert snapshot["command_surface_count"] == len(metadata.commands)
    assert snapshot["command_surface_count"] > 0
    assert snapshot["pid"] == os.getpid()
    assert snapshot["configuration_valid"] is True


def test_a_workflow_without_a_manifest_still_reports_the_core_surface(tmp_path):
    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())

    assert snapshot["runtime_metadata_registered"] is True
    assert snapshot["has_workflow_manifest"] is False
    assert snapshot["workflow_fingerprint"] is None
    assert snapshot["command_surface_count"] > 0
    assert snapshot["configuration_valid"] is True


def test_the_snapshot_defaults_to_the_metadata_registered_at_startup(tmp_path):
    """The probe and the bind both call with only the workflow path; what
    they describe is what `register_runtime_metadata` retained."""
    register_runtime_metadata(str(tmp_path), _metadata(workflow_fingerprint="sha256:reg"))

    snapshot = runtime_readiness_snapshot(str(tmp_path))

    assert snapshot["runtime_metadata_registered"] is True
    assert snapshot["workflow_fingerprint"] == "sha256:reg"


def test_an_unregistered_runtime_is_described_but_not_valid(tmp_path):
    """None is a real answer (an embedder that ran no entry point), reported
    rather than raised -- but a server that cannot state its feature vector
    cannot be certified as running any configuration."""
    snapshot = runtime_readiness_snapshot(str(tmp_path))

    assert snapshot["runtime_metadata_registered"] is False
    assert snapshot["effective_features"] == {}
    assert snapshot["workflow_fingerprint"] is None
    assert snapshot["command_surface_count"] == 0
    assert snapshot["configuration_valid"] is False


# ----------------------------------------------------------------------
# Observability regime
# ----------------------------------------------------------------------


def test_the_regime_is_read_the_way_the_sink_reads_it(tmp_path, monkeypatch):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    monkeypatch.setenv(obs.SUPPRESS_PRUNE_VAR, "1")

    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())

    assert snapshot["capture_profile"] == "evidence"
    assert snapshot["capture_profile_valid"] is True
    assert snapshot["capture_policy_version"] == obs.CAPTURE_POLICY_VERSION
    assert snapshot["pruning_suppressed"] is True
    assert snapshot["configuration_valid"] is True


def test_in_process_pruning_suppression_is_visible(tmp_path, monkeypatch):
    monkeypatch.delenv(obs.SUPPRESS_PRUNE_VAR, raising=False)
    assert runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())[
        "pruning_suppressed"
    ] is False
    with obs.suppress_pruning():
        assert runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())[
            "pruning_suppressed"
        ] is True


def test_an_unknown_capture_profile_is_reported_invalid_not_raised(
    tmp_path, monkeypatch
):
    """The sink refuses to start under a profile it does not know. The probe
    must say so with the reason rather than fail to answer."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidnce")

    regime = capture_regime()
    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())

    assert regime == {
        "capture_profile": "evidnce",
        "capture_profile_valid": False,
        "capture_policy_version": obs.CAPTURE_POLICY_VERSION,
    }
    assert snapshot["capture_profile_valid"] is False
    assert snapshot["configuration_valid"] is False


# ----------------------------------------------------------------------
# Trained model version
# ----------------------------------------------------------------------


def test_the_snapshot_pins_the_trained_model_version_and_never_invents_one(
    tmp_path,
):
    """`___command_info` is gitignored and under no hashed root, so a retrain
    moves the system under measurement while the tree hash, the revision and
    the dirty flag all stay identical. The published version id is the one
    field that moves -- and a workflow with no published set reports None
    rather than a placeholder, because a placeholder pin is worse than an
    absent one. Built through the versioning module's own path helpers so
    the test cannot drift from the layout it resolves."""
    from fastworkflow.train.artifact_versioning import pointer_path, version_dir

    assert workflow_model_version(str(tmp_path)) is None

    version_dir(str(tmp_path), "20260905T042633Z-46ef78").mkdir(parents=True)
    pointer = pointer_path(str(tmp_path))
    pointer.write_text(
        '{"version_id": "20260905T042633Z-46ef78"}', encoding="utf-8"
    )
    assert workflow_model_version(str(tmp_path)) == "20260905T042633Z-46ef78"

    pointer.write_text("not json", encoding="utf-8")
    assert workflow_model_version(str(tmp_path)) is None

    # A pointer naming a version that is not on disk is not trusted.
    pointer.write_text('{"version_id": "20990101T000000Z-000000"}', encoding="utf-8")
    assert workflow_model_version(str(tmp_path)) is None

    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())
    assert "workflow_model_version" in snapshot


def test_a_legacy_layout_is_distinguished_from_an_untrained_workflow(tmp_path):
    """None from the resolver does not mean untrained: the pre-versioning
    layout has no pointer while being fully trained. The flag carries that."""
    from fastworkflow.train.artifact_versioning import MODEL_ARTIFACT_MARKERS

    assert workflow_model_legacy_layout(str(tmp_path)) is False

    legacy = tmp_path / "___command_info" / "SomeContext"
    legacy.mkdir(parents=True)
    (legacy / next(iter(MODEL_ARTIFACT_MARKERS))).write_bytes(b"\x00")

    snapshot = runtime_readiness_snapshot(str(tmp_path), metadata=_metadata())
    assert snapshot["workflow_model_version"] is None
    assert snapshot["workflow_model_legacy_layout"] is True


# ----------------------------------------------------------------------
# Credential-free
# ----------------------------------------------------------------------


def test_the_snapshot_carries_no_value_from_any_secret_looking_env_var(
    tmp_path, monkeypatch
):
    """Nothing that looks like a credential may reach the snapshot: it is
    answered by an unauthenticated probe and stored on every attempt row.
    Sentinels are planted under every name shape the rule names, plus the
    workflow path and the env-file layer the sink also reads."""
    sentinels = {
        "FW_LLM_API_KEY": "sk-plant-key-1",
        "OPENAI_API_KEY": "sk-plant-key-2",
        "FW_CAPTURE_HMAC_KEY": "plant-hmac-key",
        "SOME_SECRET": "plant-secret",
        "AUTH_TOKEN": "plant-token",
        "JWT_TOKEN_SECRET": "plant-jwt",
        "DB_PASSWORD": "plant-password",
        "MY_PASSWORD": "plant-pass-2",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)
    # The env-file layer the sink's `_env` consults after the process env.
    import fastworkflow

    monkeypatch.setattr(
        fastworkflow,
        "_env_vars",
        {**sentinels, "FILE_ONLY_SECRET_KEY": "plant-file-secret"},
        raising=False,
    )
    register_runtime_metadata(str(tmp_path), _metadata(workflow_fingerprint="sha256:x"))

    snapshot = runtime_readiness_snapshot(str(tmp_path))
    rendered = json.dumps(snapshot)

    for value in list(sentinels.values()) + ["plant-file-secret"]:
        assert value not in rendered
    assert str(tmp_path) not in rendered
    for key in snapshot:
        upper = key.upper()
        assert not any(word in upper for word in ("KEY", "SECRET", "TOKEN", "PASSWORD")), key
    # Only scalars and the (string -> string) feature vector; no nested
    # structures that could smuggle a config dump.
    for key, value in snapshot.items():
        if key == "effective_features":
            assert all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            )
        else:
            assert value is None or isinstance(value, (bool, int, str)), key


def test_the_env_names_the_snapshot_consults_are_pinned():
    """Two profile/pruning names and nothing else. A new env read must be
    added here deliberately, with its credential-freeness argued. The
    retired recording switch is not among them: recording is always on."""
    assert snapshot_env_names() == (
        obs.CAPTURE_PROFILE_VAR,
        obs.SUPPRESS_PRUNE_VAR,
    )
    assert all(
        not any(word in name for word in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
        for name in snapshot_env_names()
    )


def test_observability_enabled_reports_whether_a_writer_is_live(tmp_path, monkeypatch):
    """The key stays in the stored snapshot; its value is now a fact, not a switch.

    Recording is always on, so the only thing that can differ is whether this
    process holds a live writer for the workflow's store.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
    workflow = tmp_path / "wf"
    workflow.mkdir()
    assert runtime_readiness_snapshot(str(workflow))["observability_enabled"] is False
    sink = obs.get_observability_sink(str(workflow))
    try:
        assert sink is not None
        assert runtime_readiness_snapshot(str(workflow))["observability_enabled"] is True
    finally:
        obs.close_all_sinks()
