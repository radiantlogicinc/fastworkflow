"""Coverage for the fastWorkflow-origin half of evaluation provenance (arch §6.3).

The module exists because these values are individually easy to get wrong in
ways nothing detects, so most of what is tested here is the *wrongness* it is
supposed to prevent: a version that is silently five releases stale, a revision
recorded without noting the tree was dirty, a trained workflow reported as
untrained, a benchmark run that never called the model it names.

Also asserts the arch §22 leaf constraint structurally, by reading the import
statements, rather than by hoping nobody adds a heavy import later.
"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from fastworkflow.observability.provenance import (
    LLM_ROLE_VARS,
    LOCK_FILENAMES,
    ORPHAN_ROLE_VARS,
    WORKFLOW_SCOPE_RULE_VERSION,
    EngineProvenance,
    ModelProvenance,
    RuntimeProvenance,
    WorkflowProvenance,
    capture_engine_provenance,
    capture_model_provenance,
    capture_runtime_provenance,
    capture_workflow_provenance,
    workflow_identity,
    git_is_dirty,
    git_revision,
    installed_version,
    lock_hashes,
    secret_fingerprint,
    source_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_workflow(root, files: dict[str, str]) -> str:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(root)


# ======================================================================
# Leaf constraint (arch §22)
# ======================================================================


@pytest.mark.parametrize("module", ["observability/provenance.py"])
def test_leaf_modules_import_only_stdlib_pydantic_and_other_leaves(module):
    """Arch §22, checked structurally so it cannot rot.

    These modules must not deepen the package-root/WEC/turn import cycle. The
    rule is about what they import, so read the imports rather than trusting a
    comment: any `fastworkflow.*` import must name another declared leaf.
    """
    leaves = {"fastworkflow.observability.provenance"}
    tree = ast.parse((REPO_ROOT / "fastworkflow" / module).read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = [
        name
        for name in imported
        if name == "fastworkflow" or (name.startswith("fastworkflow.") and name not in leaves)
    ]
    assert offenders == [], f"{module} imports non-leaf fastWorkflow modules: {offenders}"


# ======================================================================
# Engine identity
# ======================================================================


def test_both_version_answers_are_recorded_because_they_disagree():
    """An editable install's dist-info freezes at install time.

    Measured here as 2.30.1 versus a pyproject saying 3.2.0. Recording one
    number would report a version five minor releases stale, so both are kept
    and the disagreement is queryable.
    """
    engine = capture_engine_provenance()
    assert engine.version_source == _pyproject_version()
    assert engine.version == (engine.version_source or engine.version_installed)
    if engine.version_installed and engine.version_source:
        assert engine.version_metadata_disagrees == (
            engine.version_installed != engine.version_source
        )


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["poetry"]["version"]


def test_source_version_reads_the_running_checkout():
    assert source_version() == _pyproject_version()


def test_installed_version_of_an_absent_distribution_is_none_not_an_exception():
    assert installed_version("definitely-not-a-real-distribution-xyz") is None


def test_git_revision_matches_the_git_binary():
    """The pure-stdlib resolver must agree with git itself, including the case
    this checkout actually has: a loose ref that shadows a stale packed-refs
    entry for the same branch."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    assert git_revision(str(REPO_ROOT)) == result.stdout.strip()


def test_git_revision_resolves_from_a_subdirectory():
    assert git_revision(str(REPO_ROOT / "fastworkflow" / "utils")) == git_revision(
        str(REPO_ROOT)
    )


@pytest.mark.parametrize("path", ["/", "/tmp", "/nonexistent-path-xyz"])
def test_git_revision_outside_a_repo_is_none_not_an_exception(path):
    assert git_revision(path) is None


def test_git_revision_reads_a_detached_head(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    assert git_revision(str(tmp_path)) == "a" * 40


def test_a_loose_ref_shadows_packed_refs(tmp_path):
    """Git's own precedence. Reading packed-refs first would report a revision
    the tree has not been at for some time — and in this repo the two really
    do disagree."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled\n" + "b" * 40 + " refs/heads/main\n", encoding="utf-8"
    )
    assert git_revision(str(tmp_path)) == "b" * 40

    loose = git_dir / "refs" / "heads"
    loose.mkdir(parents=True)
    (loose / "main").write_text("c" * 40 + "\n", encoding="utf-8")
    assert git_revision(str(tmp_path)) == "c" * 40


@pytest.mark.parametrize(
    "head",
    [
        "ref: /etc/hostname\n",          # absolute ref discards the git dir entirely
        "ref: ../../../etc/hostname\n",  # traversal out of the git dir
        "ref: refs/../../../etc/hostname\n",
        "not-a-sha-and-not-a-ref\n",
        "\xff\xfe\x00\x01binary\n",
    ],
)
def test_a_hostile_head_cannot_turn_provenance_into_a_file_read(tmp_path, head):
    """`git_dir / ref` with an absolute ref drops the git dir (pathlib joins that
    way), and `..` traverses out of it. Whatever came back would be published as
    `source_revision`, so the ref is validated before the join and the result is
    validated to be an object id."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(head, encoding="utf-8", errors="replace")
    assert git_revision(str(tmp_path)) is None


def test_a_broken_gitdir_pointer_does_not_climb_to_the_enclosing_repo(tmp_path):
    """A `.git` file is an explicit repository boundary. Climbing past a broken
    one returns the outer checkout's revision — a confident wrong answer, which
    in a provenance record is worse than None."""
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    (outer / ".git" / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    assert git_revision(str(outer)) == "a" * 40

    inner = outer / "submodule"
    inner.mkdir()
    (inner / ".git").write_text("gitdir: /nonexistent/path/xyz\n", encoding="utf-8")
    assert git_revision(str(inner)) is None


def test_an_empty_loose_ref_falls_back_to_packed_refs(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text("b" * 40 + " refs/heads/main\n", encoding="utf-8")
    loose = git_dir / "refs" / "heads"
    loose.mkdir(parents=True)
    (loose / "main").write_text("", encoding="utf-8")
    assert git_revision(str(tmp_path)) == "b" * 40


def test_source_version_survives_a_malformed_pyproject(tmp_path, monkeypatch):
    """`tool = "oops"` breaks the .get() chain with AttributeError, which is not
    in the obvious except tuple."""
    import fastworkflow.observability.provenance as provenance_module

    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text('tool = "oops"\n', encoding="utf-8")
    monkeypatch.setattr(provenance_module, "__file__", str(package_dir / "observability/provenance.py"))
    assert source_version() is None


def test_git_dirtiness_is_tri_state():
    """None means "could not tell" and is not False. A caller that conflated
    them would report an unknown tree as clean, which is the direction that
    overstates reproducibility."""
    assert git_is_dirty("/nonexistent-path-xyz") is None
    assert git_is_dirty(str(REPO_ROOT)) in (True, False)


def test_a_dirty_tree_is_not_reproducible():
    engine = capture_engine_provenance()
    if engine.source_dirty:
        assert engine.reproducible is False
    assert EngineProvenance(
        version="3.2.0",
        source_hash="sha256:x",
        imported_module_count=1,
        python_version="3.13.14",
        python_build="3.13.14",
        source_revision="abc",
        source_dirty=False,
    ).reproducible is True


def test_unknown_dirtiness_does_not_count_as_reproducible():
    assert EngineProvenance(
        version="3.2.0",
        source_hash="sha256:x",
        imported_module_count=1,
        python_version="3.13.14",
        python_build="3.13.14",
        source_revision="abc",
        source_dirty=None,
    ).reproducible is False


def test_lock_hashes_cover_the_files_that_exist():
    hashes = lock_hashes(str(REPO_ROOT))
    assert "poetry.lock" in hashes
    assert hashes["poetry.lock"].startswith("sha256:")
    # Enumerated, not globbed: a lock file that does not exist is simply absent
    # rather than silently folded into one combined value. Checked against the
    # checkout rather than a hardcoded "uv.lock is absent", so a stray local
    # lock file is reported as present instead of failing the test. At least one
    # enumerated name must be missing, or the absent half checks nothing.
    for name in LOCK_FILENAMES:
        assert (name in hashes) == (REPO_ROOT / name).is_file(), name
    assert any(not (REPO_ROOT / name).exists() for name in LOCK_FILENAMES)


def test_lock_hashes_of_a_directory_without_locks_is_empty(tmp_path):
    assert lock_hashes(str(tmp_path)) == {}


# ======================================================================
# Workflow identity
# ======================================================================


def test_workflow_identity_is_the_computed_content_hash(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_workflow_provenance(root)
    assert provenance.workflow_source_hash == workflow_identity(root)
    assert provenance.workflow_source_hash == workflow_identity(root)


def test_the_scope_rule_that_produced_the_hash_is_recorded(tmp_path):
    """fix-ijf: a hash is comparable only against one computed the same way.

    §6.3 makes bundles immutable, so a recorded hash outlives every rule that
    could recompute it. Without this field, changing the file selection later
    makes every earlier bundle look like a workflow that drifted; with it, the
    comparison reports that it cannot be made, which is the true answer.

    Separate from `RuntimeProvenance.schema_version` because they move
    independently — this record can grow a field without changing what gets
    hashed, and the selection can change without changing the record's shape.
    """
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_workflow_provenance(root)
    assert provenance.workflow_scope_rule_version == WORKFLOW_SCOPE_RULE_VERSION

    # A bundle recorded under an older rule keeps saying so, rather than being
    # silently reinterpreted under whatever rule the reader happens to run.
    archived = provenance.model_copy(update={"workflow_scope_rule_version": 0})
    assert archived.workflow_scope_rule_version != WORKFLOW_SCOPE_RULE_VERSION
    assert archived.workflow_source_hash == provenance.workflow_source_hash


def test_records_written_under_the_retired_manifest_fields_still_load(tmp_path):
    """Rows stored before the manifest's fingerprint and feature modes were
    retired carry extra keys. They load, and the retired values are dropped."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    current = capture_workflow_provenance(root)
    legacy = current.model_dump() | {
        "declared_workflow_fingerprint": "sha256:old",
        "declared_scope_rule_version": 1,
        "fingerprint_verified": False,
        "runtime_manifest_fingerprint": "sha256:manifest",
        "runtime_feature_snapshot": {"turn_budget_v1": "off"},
    }
    loaded = WorkflowProvenance.model_validate(legacy)
    assert loaded == current
    assert not hasattr(loaded, "fingerprint_verified")


def test_an_unknown_field_is_still_refused(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    with pytest.raises(ValidationError):
        WorkflowProvenance.model_validate(
            capture_workflow_provenance(root).model_dump() | {"invented_field": 1}
        )


def test_a_legacy_layout_workflow_is_not_reported_as_untrained():
    """`resolve_current_version` returns None for a trained workflow on the
    pre-versioning layout — both bundled hello_world workflows are in that
    state — so the id alone cannot answer the question."""
    assert WorkflowProvenance(
        workflow_source_hash="sha256:x",
        trained_artifact_id=None,
        trained_artifact_legacy_layout=True,
    ).trained is True
    assert WorkflowProvenance(
        workflow_source_hash="sha256:x", trained_artifact_id="20260803T214845Z-9f005a"
    ).trained is True
    assert WorkflowProvenance(workflow_source_hash="sha256:x").trained is None


def test_the_caller_supplied_values_land_where_they_are_labelled(tmp_path):
    """command_surface_fingerprint is the legacy mtime/path hash. It is recorded
    because §6.3 asks for it, under a name that says which of the two hashes it
    is — the misnaming elsewhere in the repo is what this avoids."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_workflow_provenance(
        root,
        command_surface_fingerprint="legacy-abc",
        trained_artifact_id="v1",
        trained_artifact_legacy_layout=False,
    )
    assert provenance.command_surface_fingerprint == "legacy-abc"
    assert provenance.workflow_source_hash != "legacy-abc"
    assert provenance.trained_artifact_id == "v1"


# ======================================================================
# Model identity
# ======================================================================


def test_every_role_in_the_registry_is_snapshotted():
    env = {model_var: f"vendor/{model_var.lower()}" for model_var, _ in LLM_ROLE_VARS}
    provenance = capture_model_provenance(env)
    assert set(provenance.roles) == {model_var for model_var, _ in LLM_ROLE_VARS}
    assert len(provenance.model_identifiers) == len(LLM_ROLE_VARS)


def test_the_command_metadata_key_pairing_is_the_irregular_one():
    """LLM_COMMAND_METADATA_GEN pairs with LITELLM_API_KEY_COMMANDMETADATA_GEN —
    no underscore. Deriving the key name from the role name silently reads an
    unset variable for this one role, which is why the pairing is a table."""
    pairs = dict(LLM_ROLE_VARS)
    assert pairs["LLM_COMMAND_METADATA_GEN"] == "LITELLM_API_KEY_COMMANDMETADATA_GEN"

    derived_would_be = "LITELLM_API_KEY_" + "LLM_COMMAND_METADATA_GEN".removeprefix("LLM_")
    assert derived_would_be != pairs["LLM_COMMAND_METADATA_GEN"]

    provenance = capture_model_provenance(
        {
            "LLM_COMMAND_METADATA_GEN": "vendor/m",
            "LITELLM_API_KEY_COMMANDMETADATA_GEN": "secret",
        }
    )
    assert provenance.roles["LLM_COMMAND_METADATA_GEN"].api_key_fingerprint is not None


def test_the_orphan_role_var_is_snapshotted_but_not_an_active_role():
    """LLM_RESPONSE_GEN is in every env file and AGENTS.md, and no Python reads
    it. A setting an operator believes is in effect belongs in the record; it
    does not belong in the active-role set."""
    provenance = capture_model_provenance(
        {"LLM_RESPONSE_GEN": "vendor/unused", "LLM_AGENT": "vendor/real"}
    )
    assert provenance.orphan_role_vars == {"LLM_RESPONSE_GEN": "vendor/unused"}
    assert "LLM_RESPONSE_GEN" not in provenance.roles
    assert set(ORPHAN_ROLE_VARS) == {"LLM_RESPONSE_GEN"}


def test_api_keys_are_fingerprinted_never_recorded():
    secret = "sk-super-secret-value"
    provenance = capture_model_provenance(
        {"LLM_AGENT": "vendor/m", "LITELLM_API_KEY_AGENT": secret}
    )
    assert secret not in provenance.model_dump_json()
    assert provenance.roles["LLM_AGENT"].api_key_fingerprint == secret_fingerprint(secret)


def test_secret_fingerprint_distinguishes_keys_and_handles_absence():
    assert secret_fingerprint(None) is None
    assert secret_fingerprint("") is None
    assert secret_fingerprint("a") != secret_fingerprint("b")


def test_proxy_basic_auth_credentials_never_reach_the_record():
    """LITELLM_PROXY_API_BASE comes from the same credential-bearing env as the
    API keys, and `https://user:pass@host` is a normal way to write it."""
    provenance = capture_model_provenance(
        {
            "LLM_AGENT": "litellm_proxy/m",
            "LITELLM_PROXY_API_BASE": "https://bob:hunter2@proxy.internal:4000/v1",
        }
    )
    serialized = provenance.model_dump_json()
    assert "hunter2" not in serialized
    assert "bob" not in serialized
    role = provenance.roles["LLM_AGENT"]
    assert role.proxy_base == "https://proxy.internal:4000/v1"
    assert role.proxy_credentials_fingerprint is not None


def test_a_proxy_url_without_credentials_is_recorded_unchanged():
    provenance = capture_model_provenance(
        {"LLM_AGENT": "litellm_proxy/m", "LITELLM_PROXY_API_BASE": "https://proxy.internal"}
    )
    assert provenance.roles["LLM_AGENT"].proxy_base == "https://proxy.internal"
    assert provenance.roles["LLM_AGENT"].proxy_credentials_fingerprint is None


def test_config_hash_is_self_delimiting():
    """A newline inside a value must not be confusable with an extra key.

    The sibling canonical_content_hash exists precisely to prevent this class of
    collision, so config_hash is built through it rather than by joining
    strings.
    """
    collide_a = capture_model_provenance({}, extra_config={"a": "1\nb=2"})
    collide_b = capture_model_provenance({}, extra_config={"a": "1", "b": "2"})
    assert collide_a.config_hash != collide_b.config_hash


def test_a_proxy_model_records_where_it_was_actually_served_from():
    """A `litellm_proxy/` prefix is routing, not authentication: the model
    identifier alone no longer says which endpoint answered."""
    provenance = capture_model_provenance(
        {
            "LLM_AGENT": "litellm_proxy/some-model",
            "LLM_PLANNER": "mistral/mistral-small-latest",
            "LITELLM_PROXY_API_BASE": "https://proxy.internal",
        }
    )
    assert provenance.roles["LLM_AGENT"].via_proxy is True
    assert provenance.roles["LLM_AGENT"].proxy_base == "https://proxy.internal"
    assert provenance.roles["LLM_PLANNER"].via_proxy is False
    assert provenance.roles["LLM_PLANNER"].proxy_base is None


def test_config_hash_tracks_models_and_ignores_credentials():
    base = {"LLM_AGENT": "vendor/a", "LITELLM_API_KEY_AGENT": "key-one"}
    rotated = {"LLM_AGENT": "vendor/a", "LITELLM_API_KEY_AGENT": "key-two"}
    changed = {"LLM_AGENT": "vendor/b", "LITELLM_API_KEY_AGENT": "key-one"}

    assert (
        capture_model_provenance(base).config_hash
        == capture_model_provenance(rotated).config_hash
    )
    assert (
        capture_model_provenance(base).config_hash
        != capture_model_provenance(changed).config_hash
    )


def test_config_hash_covers_caller_supplied_call_settings():
    a = capture_model_provenance({"LLM_AGENT": "vendor/a"}, extra_config={"max_tokens": "2000"})
    b = capture_model_provenance({"LLM_AGENT": "vendor/a"}, extra_config={"max_tokens": "4000"})
    assert a.config_hash != b.config_hash


def test_a_cached_run_is_not_evidence_about_the_model():
    """DSPy's cache is process-global. A cache hit returns a stored completion
    and calls nothing, so the run says nothing about the models it names."""
    assert capture_model_provenance({}, cache_enabled=True).evidence_about_the_model is False
    assert capture_model_provenance({}, cache_enabled=False).evidence_about_the_model is True
    assert capture_model_provenance({}).evidence_about_the_model is None


def test_an_empty_environment_records_no_models_rather_than_failing():
    provenance = capture_model_provenance({})
    assert provenance.model_identifiers == {}
    assert all(role.model is None for role in provenance.roles.values())


# ======================================================================
# The whole record
# ======================================================================


def test_capture_runtime_provenance_assembles_all_three_sections(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_runtime_provenance(
        root,
        env={"LLM_AGENT": "vendor/a"},
        cache_enabled=False,
    )
    assert provenance.schema_version == 1
    assert provenance.engine.python_version
    assert provenance.workflow.workflow_source_hash.startswith("sha256:")
    assert provenance.models.model_identifiers == {"LLM_AGENT": "vendor/a"}


def _reproducible_record(tmp_path, **overrides):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_runtime_provenance(
        root, env={}, cache_enabled=False
    )
    return provenance.model_copy(
        update={
            "engine": provenance.engine.model_copy(
                update={"source_revision": "a" * 40, "source_dirty": False}
            ),
            **overrides,
        }
    )


def test_reproducibility_requires_clean_source_and_no_cache(tmp_path):
    """Two independent ways a run can fail to be reproducible, each of which
    a harness would otherwise have to remember to check separately."""
    assert _reproducible_record(tmp_path).reproducible is True

    record = _reproducible_record(tmp_path)
    dirty = record.model_copy(
        update={"engine": record.engine.model_copy(update={"source_dirty": True})}
    )
    assert dirty.reproducible is False

    cached = record.model_copy(
        update={"models": record.models.model_copy(update={"cache_enabled": True})}
    )
    assert cached.reproducible is False


def test_an_unknown_cache_state_is_not_reproducible(tmp_path):
    """The regression this property most needs.

    `cache_enabled` defaults to None on every capture path, so accepting
    "not False" would report an ordinary clean-tree run as reproducible while
    nothing is known about whether a cached completion answered instead of the
    model. Every condition must be positively established, matching how the
    engine half already refuses `source_dirty=None`.
    """
    record = _reproducible_record(tmp_path)
    unknown = record.model_copy(
        update={"models": record.models.model_copy(update={"cache_enabled": None})}
    )
    assert unknown.models.evidence_about_the_model is None
    assert unknown.reproducible is False


def test_a_serialized_record_round_trips(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    provenance = capture_runtime_provenance(
        root, env={"LLM_AGENT": "vendor/a"}
    )
    assert RuntimeProvenance.model_validate_json(provenance.model_dump_json()) == provenance


def test_the_record_is_strict_about_unknown_fields():
    with pytest.raises(ValidationError):
        ModelProvenance(config_hash="sha256:x", invented_field=1)
