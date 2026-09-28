"""Credential-free runtime introspection for deployment readiness probes.

Ported (trimmed) from the ido branch's `experiment/readiness.py` (fix-qe2). The
source module reports the planner execution path -- plan decomposition mode,
execution arm, packing, task-card catalogue, stress-mode deadlines. None of
that exists on this branch, and none of it is ported: a snapshot field whose
value is a constant here would be noise for the cross-server comparison it is
meant to serve. What is kept is everything the 3.3 runtime actually has and
that changes what a server does per turn:

* the effective feature vector and manifest identity from the runtime
  manifest registered at startup (`runtime_manifest`);
* the trained model version, because `___command_info/` is under no hashed
  root, so a retrain changes the system under measurement while every
  source fingerprint stays byte-identical;
* the observability regime in effect in THIS process -- capture profile,
  policy version, whether retention pruning is suppressed;
* the served command count and the process id.

Omitted, with the reason, so a reader of the ido snapshot knows why the
fields are missing rather than assuming the port forgot them:

* catalogue fingerprint / skill count: no `skill_catalog` module on 3.3.
* presented-result cap: no `result_handles` module on 3.3.
* extraction constants: `utils/react.py` on 3.3 has no `EXTRACT_*`
  constants and no env resolution for them; reporting values it does not
  apply would certify the wrong runtime.
* safety wall / turn deadline seconds: nothing on 3.3 reads
  `FW_PLAN_WALL_TIME_LIMIT_SECONDS` or `FW_TURN_DEADLINE_SECONDS`. A snapshot
  that reported them would describe a setting with no effect.

**Credential-free.** The only environment names consulted are the three
FW_OBS_* switches named below, every one a profile name or a flag. No path,
no key, no token, no env-file value reaches the snapshot; the fingerprint is
a content hash and the model version is a generated id.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastworkflow.observability import store as observability_store
from fastworkflow.observability.capture_policy import (
    CAPTURE_POLICY_VERSION,
    CaptureProfileError,
    policy_for_profile,
)
from fastworkflow.runtime_manifest import RuntimeMetadata, get_runtime_metadata


def workflow_model_version(workflow_path: str) -> Optional[str]:
    """The trained model set this workflow is currently serving, or None.

    `___command_info/` is gitignored and is under none of the roots any source
    fingerprint hashes, so a retrain changes the system under measurement while
    the tree hash, the git revision and the dirty flag all stay byte-identical.
    The published version id is the one cheap field that moves, and a readiness
    snapshot that does not carry it cannot tell two runs apart across a retrain.

    Resolved through `train.artifact_versioning.resolve_current_version`, the
    same reader training and provenance use, so this never disagrees with them
    about which version is live. Read, never cached: only a successful train
    publishes, so the value is a fact about the moment the probe asked. Absent
    or unreadable answers None rather than raising -- a workflow with no trained
    set is a real deployment state, and a readiness probe reports it rather
    than refusing to describe the runtime at all.

    None does NOT mean untrained: a workflow on the pre-versioning layout has
    no pointer file while being fully trained (both bundled `hello_world`
    workflows are in that state). `workflow_model_legacy_layout` in the
    snapshot carries that difference, as `provenance` does.
    """
    try:
        from fastworkflow.train.artifact_versioning import resolve_current_version

        version = resolve_current_version(workflow_path)
    except Exception:
        return None
    return str(version) if version else None


def workflow_model_legacy_layout(workflow_path: str) -> Optional[bool]:
    """True when unversioned artifacts sit directly in `___command_info`.

    None means "could not tell", never False: the field exists so a None
    `workflow_model_version` is not misread as an untrained workflow.
    """
    try:
        from fastworkflow.train.artifact_versioning import legacy_layout_in_use

        return bool(legacy_layout_in_use(workflow_path))
    except Exception:
        return None


def capture_regime() -> dict[str, Any]:
    """The capture profile and policy version this process records under.

    Resolved the way the sink resolves it (`FW_OBS_CAPTURE_PROFILE`, process
    env first, then the workflow env file, then the default) and validated
    through `capture_policy.policy_for_profile`, which refuses an unknown name
    rather than falling back. A profile the sink would refuse to start under
    is reported as invalid here rather than raising, so the probe can say
    "not ready" with the reason instead of failing to answer.
    """
    name = observability_store._env(
        observability_store.CAPTURE_PROFILE_VAR,
        observability_store._DEFAULT_CAPTURE_PROFILE,
    )
    try:
        policy_for_profile(name)
        valid = True
    except CaptureProfileError:
        valid = False
    return {
        "capture_profile": name,
        "capture_profile_valid": valid,
        "capture_policy_version": CAPTURE_POLICY_VERSION,
    }


def runtime_readiness_snapshot(
    workflow_path: str,
    *,
    metadata: Optional[RuntimeMetadata] = None,
) -> dict[str, Any]:
    """Report this process's effective runtime without exposing deployment secrets.

    `metadata` defaults to what `register_runtime_metadata` retained at startup
    for `workflow_path`. None is a real answer (an embedder that never ran a
    fastWorkflow entry point) and is reported as `runtime_metadata_registered:
    False` with an empty feature vector, which makes `configuration_valid`
    False: a server that cannot describe its feature vector cannot be
    certified as running any particular configuration.
    """
    effective_metadata = (
        get_runtime_metadata(workflow_path) if metadata is None else metadata
    )
    registered = effective_metadata is not None
    regime = capture_regime()

    snapshot: dict[str, Any] = {
        "runtime_metadata_registered": registered,
        "effective_features": (
            dict(sorted(effective_metadata.feature_modes.items()))
            if registered
            else {}
        ),
        "has_workflow_manifest": (
            bool(effective_metadata.has_workflow_manifest) if registered else False
        ),
        "workflow_fingerprint": (
            effective_metadata.workflow_fingerprint if registered else None
        ),
        "workflow_scope_rule_version": (
            effective_metadata.workflow_scope_rule_version if registered else None
        ),
        "command_surface_count": (
            len(effective_metadata.commands) if registered else 0
        ),
        "workflow_model_version": workflow_model_version(workflow_path),
        "workflow_model_legacy_layout": workflow_model_legacy_layout(workflow_path),
        # Recording has no switch. What can still differ between processes is
        # whether this one holds a live writer for the workflow's store -- a
        # store that could not be opened degrades to none -- so that is what
        # this key, kept for the stored snapshot's shape, reports.
        "observability_enabled": (
            observability_store.existing_observability_sink(workflow_path)
            is not None
        ),
        "pruning_suppressed": observability_store.pruning_suppressed(),
        "pid": os.getpid(),
    }
    snapshot.update(regime)
    snapshot["configuration_valid"] = bool(
        registered and regime["capture_profile_valid"]
    )
    return snapshot


def snapshot_env_names() -> tuple[str, ...]:
    """Every environment name the snapshot consults. Pinned by test."""
    return (
        observability_store.CAPTURE_PROFILE_VAR,
        observability_store.SUPPRESS_PRUNE_VAR,
    )


__all__ = [
    "capture_regime",
    "runtime_readiness_snapshot",
    "snapshot_env_names",
    "workflow_model_legacy_layout",
    "workflow_model_version",
]

