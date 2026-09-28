"""Conformance coverage for the workflow runtime manifest (arch §18.1).

Slice 0 changes no behavior, so there is nothing here about what the runtime
*does* with a manifest — only that it reads one correctly, merges it safely,
gates it inertly, and fingerprints it reproducibly.

Four properties carry the slice, and each has a section below:

* **fingerprint stability** — the canonical hash depends on content and nothing
  else, and is bit-identical to the IDO generator's independent implementation
  of the same recipe (verification-register item O3);
* **occupancy** — what a manifest says about which contexts can be entered, and
  what "it did not say" means as distinct from "it said no";
* **merge precedence** — a workflow manifest may add and may tighten, and may
  never loosen a framework declaration (§7.3);
* **dual gating** — a feature runs only when workflow and deployment both
  enable it, at the more restrictive of the two, and a deployment asking for
  more than the workflow declared is a startup failure rather than a clamp.

Plus the compatibility case that gates the whole slice: a workflow with no
manifest must behave exactly as it does today.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import shutil
import sys
import time
import types

import pytest
from pydantic import ValidationError

from fastworkflow.command_directory import compute_commands_source_fingerprint
from fastworkflow.runtime_manifest import (
    CORE_MANIFEST,
    DEPLOYMENT_FEATURES_ENV_VAR,
    FEATURES_BY_ID,
    FRAMEWORK_ARTIFACT_PREFIX,
    WORKFLOW_SCOPE_RULE_VERSION,
    CommandDeclaration,
    ContextDeclaration,
    EffectContract,
    FingerprintVerification,
    ManifestConformanceError,
    NavigationEffect,
    RuntimeManifest,
    canonical_content_hash,
    check_startup_conformance,
    deployment_env,
    deployment_features_from_env,
    imported_package_entries,
    imported_package_hash,
    load_manifest,
    merge_and_gate,
    occupancy_completeness_problems,
    parse_deployment_features,
    verify_workflow_fingerprint,
    workflow_content_entries,
    workflow_content_hash,
)


def _manifest(**overrides) -> RuntimeManifest:
    """A minimal valid manifest; keyword args override any field."""
    fields = {"schema_version": 1, "manifest_version": "1.0.0"}
    fields.update(overrides)
    return RuntimeManifest(**fields)


def _write_workflow(root, files: dict[str, str]) -> str:
    """Materialize ``{relative path: text}`` under ``root`` and return the path."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(root)


# ======================================================================
# Fingerprint stability (fix-i6h.2; register item O3)
# ======================================================================


def _ido_reference_fingerprint(entries):
    """The IDO generator's recipe, transcribed from ``canonical_fingerprint``.

    Deliberately a copy rather than an import: the two repositories cannot
    import each other, so the recipe *is* the contract between them and this is
    the only place a drift in it can be caught. If this ever diverges, one of
    the two implementations has silently changed what a workflow's identity
    means. The ``%``-formatting below is theirs; keep it, so a future diff
    against their source is character-for-character.
    """
    digest = hashlib.sha256()
    for path, content in sorted(entries):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return "sha256:%s" % digest.hexdigest()


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [("a.py", b"x")],
        [("b/c.py", b"hello"), ("a.py", b""), ("z/y/x.json", b"{}")],
        [("f", b"b\0c\0d")],
        [("f", b"b"), ("g", b"d")],
        [("unicode/\u00e9.py", "na\u00efve".encode("utf-8"))],
    ],
)
def test_recipe_matches_the_ido_generator_implementation(entries):
    assert canonical_content_hash(entries) == _ido_reference_fingerprint(entries)


def test_length_prefix_separates_trees_that_nul_alone_would_merge():
    """The specific ambiguity the length field exists to prevent.

    One file holding ``b"b\\0c\\0d"`` and two files holding ``b"b"`` and ``b"d"``
    serialize identically under NUL separators alone. They are different trees
    and must hash differently.
    """
    one_file = canonical_content_hash([("f", b"b\0c\0d")])
    two_files = canonical_content_hash([("f", b"b"), ("d", b"")])
    assert one_file != two_files


def test_input_order_does_not_change_the_hash():
    forward = canonical_content_hash([("a.py", b"1"), ("b.py", b"2"), ("c.py", b"3")])
    reverse = canonical_content_hash([("c.py", b"3"), ("b.py", b"2"), ("a.py", b"1")])
    assert forward == reverse


def test_content_change_changes_the_hash():
    before = canonical_content_hash([("a.py", b"1")])
    after = canonical_content_hash([("a.py", b"2")])
    assert before != after


def test_moving_content_between_files_changes_the_hash():
    """Paths are hashed, not just bytes, so a rename is not a no-op."""
    assert canonical_content_hash([("a.py", b"x")]) != canonical_content_hash(
        [("b.py", b"x")]
    )


def test_windows_separators_normalize_to_the_posix_form():
    assert canonical_content_hash([("a\\b.py", b"x")]) == canonical_content_hash(
        [("a/b.py", b"x")]
    )


def test_absolute_paths_are_rejected():
    """An absolute path is the exact defect this hash exists to not have, and it
    is invisible in the output, so it must fail loudly instead."""
    with pytest.raises(ValueError, match="tree-relative"):
        canonical_content_hash([("/tmp/workflow/a.py", b"x")])


@pytest.mark.parametrize(
    "path", ["/tmp/wf/a.py", "C:/wf/a.py", "C:\\wf\\a.py", "../outside.py", "a/../../b.py"]
)
def test_paths_that_escape_the_tree_root_are_rejected(path):
    """A traversal is as location-dependent as an absolute path, and just as
    invisible in the output."""
    with pytest.raises(ValueError, match="tree-relative"):
        canonical_content_hash([(path, b"x")])


def test_a_colon_in_a_filename_is_not_mistaken_for_a_drive_letter():
    assert canonical_content_hash([("odd:name.py", b"x")]).startswith("sha256:")


def test_duplicate_paths_are_rejected():
    with pytest.raises(ValueError, match="duplicate path"):
        canonical_content_hash([("a.py", b"x"), ("a.py", b"y")])


def test_empty_tree_hashes_to_the_empty_sha256():
    assert canonical_content_hash([]) == f"sha256:{hashlib.sha256().hexdigest()}"


def test_fingerprint_survives_copying_the_tree_elsewhere(tmp_path):
    """Criterion 2: independent of copied-deployment location.

    Contrasted against the legacy source fingerprint, which is *supposed* to
    move here — it keys a cache on absolute path plus ``(size, mtime_ns)``. The
    contrast is the point: the two values coexist and must not be substituted
    for one another.
    """
    original = _write_workflow(
        tmp_path / "here", {"_commands/a.py": "x = 1\n", "context_hierarchy_model.json": "{}"}
    )
    copied = str(tmp_path / "there")
    shutil.copytree(original, copied)

    assert workflow_content_hash(original) == workflow_content_hash(copied)
    assert compute_commands_source_fingerprint(
        original
    ) != compute_commands_source_fingerprint(copied)


def test_fingerprint_survives_touching_files_without_editing_them(tmp_path):
    """Regeneration rewrites byte-identical files and gives them new mtimes; a
    workflow that did not change must not appear to have changed."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_content_hash(root)
    legacy_before = compute_commands_source_fingerprint(root)

    future = time.time() + 3600
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            os.utime(os.path.join(dirpath, filename), (future, future))

    assert workflow_content_hash(root) == before
    assert compute_commands_source_fingerprint(root) != legacy_before


def test_derived_and_secret_trees_are_excluded_from_the_fingerprint(tmp_path):
    """Trained artifacts, distillation output, and local env files are not
    identity: two deployments of the same workflow differ in all three."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_content_hash(root)

    _write_workflow(
        tmp_path / "wf",
        {
            "___command_info/model.json": '{"weights": 1}',
            "Insights/note.md": "learned something",
            "fastworkflow.env": "LLM_AGENT=whatever\n",
            "workflow_runtime.json": '{"schema_version": 1}',
        },
    )
    assert workflow_content_hash(root) == before


def test_generator_bookkeeping_files_are_out_of_scope(tmp_path):
    """A dot-prefixed file is tool bookkeeping, not source.

    A generated manifest carrying a `generated_at` timestamp makes this hash
    time-dependent — the exact property §7.1 says it must not have. Excluding
    the class is what stops the next bookkeeping file from doing it again.
    """
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_content_hash(root)

    _write_workflow(
        tmp_path / "wf",
        {
            ".generated_manifest.json": '{"generated_at": "2026-08-27T12:00:00Z"}',
            ".hidden/notes.md": "scratch",
        },
    )
    assert workflow_content_hash(root) == before


def test_a_missing_workflow_folder_raises_rather_than_hashing_nothing(tmp_path):
    """os.walk yields nothing for a missing directory, so without a guard a
    typo'd path produces the hash of the empty tree: a confident, well-formed
    sha256 that every wrong path shares, recorded as a workflow's identity."""
    with pytest.raises(ValueError, match="does not exist"):
        workflow_content_hash(str(tmp_path / "no-such-workflow"))


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("{not json,}", "could not be read as a manifest"),
        ('{"schema_version": 1, "manifest_version": "1.0.0", "typo": 1}', "manifest"),
        ('{"schema_version": 99, "manifest_version": "1.0.0"}', "manifest"),
    ],
)
def test_an_unreadable_manifest_arrives_as_a_conformance_error(tmp_path, body, fragment):
    """The CLI and FastAPI startup paths catch ManifestConformanceError, so a
    trailing comma must not reach the operator as a raw JSONDecodeError."""
    (tmp_path / "workflow_runtime.json").write_text(body, encoding="utf-8")
    with pytest.raises(ManifestConformanceError, match=fragment):
        load_manifest(str(tmp_path))
    with pytest.raises(ManifestConformanceError):
        check_startup_conformance(str(tmp_path), env={})


def test_declared_fingerprint_verifies_against_the_tree(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    computed = workflow_content_hash(root)
    manifest = _manifest(workflow_fingerprint=computed)

    verification = verify_workflow_fingerprint(root, manifest)
    assert verification.matches
    assert verification.problem() is None


def test_a_stale_declared_fingerprint_is_reported(tmp_path):
    """A mismatch means somebody changed a command and did not regenerate."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    manifest = _manifest(workflow_fingerprint="sha256:deadbeef")

    verification = verify_workflow_fingerprint(root, manifest)
    assert not verification.matches
    assert "stale with respect to the workflow" in verification.problem()


def test_an_omitted_fingerprint_is_not_a_mismatch(tmp_path):
    """§7.2 makes the field optional; a generator that cannot compute it omits
    it rather than writing a guess. That is not a failure."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    verification = verify_workflow_fingerprint(root, _manifest())
    assert verification.not_declared
    assert not verification.matches
    assert verification.problem() is None


def test_fingerprint_verification_is_opt_in_at_startup(tmp_path):
    """Slice 0 changes no behavior, and verification costs a full tree read, so
    a stale manifest starts by default and only fails under the flag."""
    root = _write_workflow(
        tmp_path / "wf",
        {
            "_commands/a.py": "x = 1\n",
            "workflow_runtime.json": _manifest(
                workflow_fingerprint="sha256:deadbeef"
            ).model_dump_json(exclude_none=True),
        },
    )
    assert check_startup_conformance(root, env={}).workflow_fingerprint == "sha256:deadbeef"
    with pytest.raises(ManifestConformanceError, match="stale with respect to"):
        check_startup_conformance(root, env={}, verify_fingerprint=True)


def test_a_manifest_may_declare_the_scope_rule_that_produced_its_fingerprint():
    """The field has to be modelled before a generator can emit it.

    `_Strict` forbids unknown keys, so a generator declaring a field this class
    does not carry has its manifest refused by name and every startup
    conformance check on that workflow fails. Without the field modelled here a
    generator can only hold the value internally.
    """
    manifest = _manifest(workflow_fingerprint="sha256:abc", workflow_scope_rule_version=1)
    assert manifest.workflow_scope_rule_version == 1
    assert merge_and_gate(manifest, env={}).workflow_scope_rule_version == 1

    # Still optional: manifests written before the rule was versioned parse.
    assert _manifest().workflow_scope_rule_version is None


def test_a_fingerprint_from_a_different_scope_rule_is_not_reported_as_drift(tmp_path):
    """The distinction the version exists to make reachable.

    Unequal digests have two causes that look identical: the tree changed, or
    the two sides selected different files. Both values are well-formed sha256
    digests of real trees and neither says which question it answered. Calling
    the second one staleness sends someone to regenerate a workflow that never
    drifted.
    """
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    verification = verify_workflow_fingerprint(
        root,
        _manifest(
            workflow_fingerprint="sha256:computed-under-an-older-selection",
            workflow_scope_rule_version=WORKFLOW_SCOPE_RULE_VERSION + 1,
        ),
    )
    assert verification.incomparable
    assert not verification.matches
    problem = verification.problem()
    assert "different questions" in problem
    assert "stale" not in problem


def test_an_unversioned_manifest_is_still_held_to_the_current_rule(tmp_path):
    """Absent means "generated before the rule was versioned", and v1 is the
    only rule that has ever existed, so an unversioned generator used it.
    Reading absence as "cannot compare" would retire every manifest in
    existence, including the live IDO one."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    verification = verify_workflow_fingerprint(root, _manifest(workflow_fingerprint="sha256:old"))
    assert not verification.incomparable
    assert "stale with respect to the workflow" in verification.problem()


def test_matching_digests_are_a_pass_whatever_the_rule_versions_say(tmp_path):
    """Equal digests mean the two selections produced identical bytes on this
    tree, which is a genuine pass. Failing it on a version mismatch would be a
    false alarm on a healthy workflow."""
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    verification = verify_workflow_fingerprint(
        root,
        _manifest(
            workflow_fingerprint=workflow_content_hash(root),
            workflow_scope_rule_version=WORKFLOW_SCOPE_RULE_VERSION + 1,
        ),
    )
    assert verification.matches
    assert not verification.incomparable
    assert verification.problem() is None


def test_framework_artifact_directories_are_excluded_by_prefix(tmp_path):
    """The artifact directories are a class, not three names.

    Naming ``___command_info``, ``___workflow_contexts`` and ``___convo_info``
    closes the instances that exist. A framework that adds a fourth would fold
    runtime state into workflow identity until somebody extended the list — and
    every generator transcribing that list would diverge until it did too.
    """
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_content_hash(root)

    _write_workflow(
        tmp_path / "wf",
        {"___future_info/state.json": "{}", "_commands/___nested_info/x.py": "y = 2\n"},
    )
    assert workflow_content_hash(root) == before


# ======================================================================
# Cross-repo scope agreement (fix-ijf; IDO counterpart ido-o1o)
# ======================================================================
#
# The recipe has a contract — `_ido_reference_fingerprint` above — and the scope
# did not, which is what fix-ijf was filed about. This is the same move for file
# selection, in three layers, because a single transcription turned out not to
# be enough.
#
# `_ido_fingerprint_included` is the agreed rule, written out so these tests run
# without the sibling repo. A pure copy is *not* a drift detector, though, and
# briefly pretending otherwise cost something real: the first version of this
# section marked three classes `xfail(strict=True)` on the reasoning that they
# would flip red when IDO adopted the matching rules. They would not have. The
# copy diverges from IDO by construction the moment IDO changes, so the xfails
# would have gone on passing-as-expected-failures while describing a divergence
# that no longer existed — the "test of a filter nobody runs" the marker's own
# reason string was trying to prevent. IDO adopted the rules the same day and
# the markers noticed nothing, which is how this was found.
#
# So the copy is checked against the real thing:
# `test_the_transcribed_ido_filter_still_matches_ido` imports
# `gen_ido_scaffold` and compares them case by case, skipping when the sibling
# repo is absent. That is the layer that actually detects drift here. IDO runs
# the mirror of it against the real `workflow_content_entries`, so neither side
# depends on the other noticing first.
#
# `_SCOPE_CASES` is the shared contract: one table of paths with the answer
# both sides owe, agreed with the IDO side and mirrored there as
# `ScopeContract.CLASSES`.
#
# All three layers pin IDO's FILTER and not its SELECTOR, and the difference is
# not a technicality. `gen_ido_scaffold.fingerprint_entries` never walks the
# workflow tree the way `workflow_content_entries` does: it takes the
# generator's pending emission unioned with the declared-preserved paths, and
# only then applies the filter. The two coincide because `replace_tree` makes
# disk equal pending-union-preserved immediately after a run — the load-bearing
# invariant behind IDO declaring a fingerprint for a tree it has not written
# yet. Neither `pending` nor the preservation manifest exists outside a
# generator run, so nothing here can reach that invariant, and reading these
# tests as coverage of it would be reading coverage we do not have. It belongs
# where it holds, as a post-`replace_tree` assertion on the IDO side.

_IDO_EXCLUDED_TREES = (
    "___command_info",
    "Insights",
    "___workflow_contexts",
    "___convo_info",
)
_IDO_EXCLUDED_FILES = (
    "workflow_runtime.json",
    ".generated_manifest.json",
    "fastworkflow.env",
    "fastworkflow.passwords.env",
    "fastworkflow.env.example",
    "fastworkflow.passwords.env.example",
)
_IDO_EXCLUDED_ROOT_TREES = ("benchmarks",)
_IDO_RUNTIME_STORE_PREFIX = "observability.sqlite3"
_IDO_SUFFIXES = (".py", ".json", ".md")
_IDO_CRUFT_DIRNAMES = frozenset({"__pycache__", ".pytest_cache"})
_IDO_FRAMEWORK_ARTIFACT_PREFIX = "___"

#: Root of a sibling IDO checkout, named by ``FW_TEST_IDO_ROOT``. The cases
#: below compare fastWorkflow's file-selection rule against IDO's own, so they
#: need IDO's source on disk; they skip when the variable is unset.
IDO_ROOT = os.environ.get("FW_TEST_IDO_ROOT", "")
IDO_GENERATOR = os.path.join(IDO_ROOT, "gen_ido_scaffold.py") if IDO_ROOT else ""


@functools.lru_cache(maxsize=1)
def _import_ido_generator():
    """Load ``gen_ido_scaffold`` from the sibling repo, or skip.

    Loaded by path rather than installed, because the dependency is one-way by
    design: fastWorkflow must not require IDO to exist. Its own directory goes
    on ``sys.path`` for the duration because the module imports siblings of its
    own; it defines only functions, classes and constants at module level, so
    importing it runs no generation. Cached because the filter is checked once
    per contract case.
    """
    if not IDO_GENERATOR:
        pytest.skip("set FW_TEST_IDO_ROOT to a sibling ido checkout to run this")
    directory = os.path.dirname(IDO_GENERATOR)
    spec = importlib.util.spec_from_file_location("gen_ido_scaffold", IDO_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.skip(f"sibling ido generator is not importable: {e}")
    finally:
        if inserted and directory in sys.path:
            sys.path.remove(directory)
    return module


def _ido_fingerprint_included(relpath):
    """Transcribed from ``gen_ido_scaffold._fingerprint_included``.

    Their ``os.sep`` split and their loop shape are kept, so a future diff
    against their source stays readable. The rule: a ``benchmarks`` directory
    directly under the root and a root file named after the runtime store are
    excluded (scope rule v2), excluded tree names and cruft names match at any
    depth, a dot prefix excludes a file or a directory, ``___`` excludes a
    directory segment only, the excluded-file list matches the whole
    root-relative path, and what survives must carry a content suffix.
    """
    segments = relpath.split(os.sep)
    if len(segments) > 1 and segments[0] in _IDO_EXCLUDED_ROOT_TREES:
        return False
    if len(segments) == 1 and segments[0].startswith(_IDO_RUNTIME_STORE_PREFIX):
        return False
    for position, segment in enumerate(segments):
        if segment in _IDO_EXCLUDED_TREES or segment in _IDO_CRUFT_DIRNAMES:
            return False
        if segment.startswith("."):
            return False
        # Directory segments only; the last segment is the filename.
        if position < len(segments) - 1 and segment.startswith(
            _IDO_FRAMEWORK_ARTIFACT_PREFIX
        ):
            return False
    if relpath in _IDO_EXCLUDED_FILES:
        return False
    return relpath.endswith(_IDO_SUFFIXES)


def _ido_selection(root: str) -> set[str]:
    """The paths IDO's filter admits from a tree on disk.

    No directory pruning, deliberately, though IDO's own walk prunes cruft.
    Pruning first would mean the filter is never asked about a path under an
    excluded directory, so a check deleted from the filter would stay invisible
    here — mutation testing on the IDO side found exactly that. It matters
    because the *writer* has no pruning to hide behind: `fingerprint_entries`
    applies the filter to a flat dict of pending paths, so the filter alone
    decides whether a generated ``___`` path reaches the fingerprint.
    """
    selected = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            relative = os.path.relpath(os.path.join(dirpath, filename), root)
            if _ido_fingerprint_included(relative):
                selected.add(relative.replace(os.sep, "/"))
    return selected


def _fw_selection(root: str) -> set[str]:
    return {path for path, _ in workflow_content_entries(root)}


# The shared contract: every path class the two scopes must answer identically,
# with the answer they owe. Mirrored on the IDO side as `ScopeContract.CLASSES`
# in their `tests/test_workflow_fingerprint.py`.
#
# `___notes.py` is in scope and is the one case worth pausing on, because it
# looks like an oversight and is not. `FRAMEWORK_ARTIFACT_PREFIX` applies to
# directory segments only, while the dot rule applies to files too, and the
# asymmetry is deliberate: the framework creates ``___`` *directories* and has
# never created a ``___`` file, so extending the rule to filenames would close
# an empty class while opening a way to silently drop a real command from the
# workflow's identity. Between the two directions of error that is the worse
# one — an over-included artifact file shows up as a hash that moves when it
# should not, which is visible, and an under-included source file shows up as a
# hash that stays still when behavior changed, which is not.
_SCOPE_CASES: dict[str, bool] = {
    # Ordinary content, at the root and nested.
    "README.md": True,
    "_commands/ok.py": True,
    "_commands/Identity/nested/deep.json": True,
    # Framework artifact directories, by name and by prefix, at any depth.
    "___command_info/m.json": False,
    "_commands/___command_info/m.json": False,
    "___convo_info/state.json": False,
    "___workflow_contexts/ctx.json": False,
    "___brand_new_artifact/x.json": False,
    "_commands/___brand_new_artifact/x.json": False,
    # Directory prefix only: these are files, not directories.
    "___notes.py": True,
    "_commands/___notes.py": True,
    # Distillation output, at any depth.
    "Insights/top.md": False,
    "_commands/Insights/nested.md": False,
    # Dot prefix, file or directory, at any depth.
    ".generated_manifest.json": False,
    "_commands/.hidden.md": False,
    "_commands/.tooling/x.py": False,
    # Cruft.
    "_commands/__pycache__/y.py": False,
    # Whole-path excluded files, and a suffix outside the content set.
    "workflow_runtime.json": False,
    # Carried because IDO carries it, not because it discriminates: it has no
    # content suffix, so both sides would drop it whatever the excluded-file
    # lists said. A case that passes for a reason other than the one it is named
    # for is worth marking as such rather than counting as coverage.
    "fastworkflow.env": False,
    "_commands/notes.txt": False,
    # Scope rule v2. The benchmark corpus directly under the root is what a
    # workflow is measured with rather than what it is, and on the tree that
    # forced this decision most of it was git-ignored - so a fingerprint over it
    # could not be reproduced from a clean clone.
    "benchmarks/g2e-tuning/v1.json": False,
    "benchmarks/ad-admin-selfcompact-pilot/analysis.json": False,
    # Root-relative, unlike every other exclusion here. A command package called
    # ``benchmarks`` is source like any other, and dropping it would be the
    # invisible direction of error: a hash that holds still while behaviour
    # moves.
    "_commands/benchmarks/case.py": True,
    # The runtime observability store. Neither of these has a content suffix, so
    # both sides drop them whatever the prefix rule says - marked as such rather
    # than counted as coverage of it.
    "observability.sqlite3": False,
    "observability.sqlite3-wal": False,
    # The case the prefix rule is actually for: a sidecar written beside the
    # store in a format the suffix rule would otherwise admit.
    "observability.sqlite3.offload-handles.json": False,
    # Near misses, so the rule cannot grow into "anything called observability"
    # or follow the store's name into a subdirectory.
    "observability.md": True,
    "_commands/observability.sqlite3.notes.md": True,
}


def _scope_case_tree(tmp_path) -> str:
    return _write_workflow(
        tmp_path / "wf", {relative: f"content of {relative}\n" for relative in _SCOPE_CASES}
    )


def test_the_shared_scope_contract_holds_for_this_side(tmp_path):
    """Every case in the agreed table, answered by ``workflow_content_entries``.

    Path by path rather than as a set difference, so a failure names the class
    that moved instead of dumping two sets.
    """
    selected = _fw_selection(_scope_case_tree(tmp_path))
    actual = {relative: relative in selected for relative in _SCOPE_CASES}
    assert actual == _SCOPE_CASES


def test_the_transcribed_ido_filter_agrees_on_the_shared_contract(tmp_path):
    selected = _ido_selection(_scope_case_tree(tmp_path))
    actual = {relative: relative in selected for relative in _SCOPE_CASES}
    assert actual == _SCOPE_CASES


@pytest.mark.parametrize("relative", sorted(_SCOPE_CASES))
def test_the_transcribed_ido_filter_is_checked_against_ido_itself(relative):
    """The layer that actually detects IDO-side drift.

    The transcription above is a copy, and a copy cannot notice that the thing
    it copied has changed — which is how the previous version of this section
    ended up asserting a divergence IDO had already closed. Importing their
    generator is what closes that gap from this side; IDO runs the mirror of it
    against the real ``workflow_content_entries``, so neither side is relying on
    the other to notice first.

    Path-level rather than tree-level on purpose: their filter is a pure
    function of one relative path, and calling it directly means no directory
    pruning stands between a deleted check and this assertion.
    """
    if not os.path.isfile(IDO_GENERATOR):
        pytest.skip("sibling ido repo not present")
    generator = _import_ido_generator()
    assert _ido_fingerprint_included(relative) == generator._fingerprint_included(
        relative
    ), f"transcription and gen_ido_scaffold disagree about {relative!r}"


def test_the_scope_rule_version_agrees_with_ido():
    """Both sides name the rule they implement, and must name the same one.

    Until IDO's manifest carries the field this is the only thing keeping the
    two constants together, and after that it is what keeps a bump on one side
    from going unnoticed on the other.
    """
    if not os.path.isfile(IDO_GENERATOR):
        pytest.skip("sibling ido repo not present")
    generator = _import_ido_generator()
    assert generator.WORKFLOW_SCOPE_RULE_VERSION == WORKFLOW_SCOPE_RULE_VERSION
    assert generator.FRAMEWORK_ARTIFACT_PREFIX == FRAMEWORK_ARTIFACT_PREFIX


def test_a_benchmarks_tree_is_excluded_at_the_root_and_kept_below_it(tmp_path):
    """Scope rule v2, stated on its own rather than only inside the table.

    Two assertions in one test because the pair is the rule: excluding at the
    root is the decision, and keeping it below the root is what stops the
    decision from eating source.
    """
    root = _write_workflow(
        tmp_path / "wf",
        {
            "benchmarks/corpus/case.json": "measured with\n",
            "_commands/benchmarks/case.py": "measured by\n",
        },
    )
    assert _fw_selection(root) == {"_commands/benchmarks/case.py"}


def test_the_runtime_store_and_its_sidecars_are_excluded_at_the_root(tmp_path):
    """The prefix rule, on the names the store actually takes.

    ``.json`` because that is the only one of these the suffix rule would admit
    on its own; the others are here to say that one store under four names is
    one exclusion.
    """
    root = _write_workflow(
        tmp_path / "wf",
        {
            "observability.sqlite3": "store\n",
            "observability.sqlite3-wal": "sidecar\n",
            "observability.sqlite3-shm": "sidecar\n",
            "observability.sqlite3.offload-handles.json": "sidecar\n",
            "observability.md": "documentation\n",
            "README.md": "source\n",
        },
    )
    assert _fw_selection(root) == {"observability.md", "README.md"}


def test_a_fingerprint_declared_under_the_previous_rule_is_incomparable():
    """What the bump to v2 buys the records already written.

    Every provenance record written before the benchmark corpus left the scope
    declares a v1 digest over a larger file set. Under v2 those digests differ,
    and the difference is not evidence that the workflow drifted - so the reader
    declines the comparison instead of reporting a stale manifest, which is the
    whole reason the selection carries a version.
    """
    assert WORKFLOW_SCOPE_RULE_VERSION > 1
    verification = FingerprintVerification(
        declared="sha256:" + "a" * 64,
        computed="sha256:" + "b" * 64,
        declared_scope_rule_version=1,
    )
    assert not verification.matches
    assert verification.incomparable
    assert "different questions" in verification.problem()


def test_scope_rules_select_the_same_files_on_the_live_ido_tree():
    """The agreement ``verify_workflow_fingerprint`` rests on, as a set.

    The same file set on both sides. The hash equality next door already
    implies this; stating it as a set means a future divergence arrives naming
    the files that moved instead of as two unequal digests.

    Necessary and nowhere near sufficient: the live tree contains none of the
    divergent classes, which is precisely why the two scopes agreed on it for
    as long as they did while differing in seven ways. The contract table is
    what covers those.
    """
    if not IDO_ROOT:
        pytest.skip("set FW_TEST_IDO_ROOT to a sibling ido checkout to run this")
    ido = os.path.join(IDO_ROOT, "ido_workflow")
    if not os.path.isdir(ido):
        pytest.skip(f"no ido_workflow under FW_TEST_IDO_ROOT ({IDO_ROOT})")
    assert _fw_selection(ido) == _ido_selection(ido)


def test_imported_package_hash_covers_only_files_under_the_package(tmp_path):
    """§6.3: the *imported module tree*, not the checkout path.

    A module whose file lives outside the package directory has no stable
    relative form, so it is skipped rather than reached for — including it would
    reintroduce exactly the location dependence being removed.
    """
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "inside.py").write_text("a = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("b = 2\n", encoding="utf-8")

    def module(name, filename):
        mod = types.ModuleType(name)
        mod.__file__ = str(filename)
        return mod

    table = {
        "pkg": module("pkg", package_dir / "__init__.py"),
        "pkg.inside": module("pkg.inside", package_dir / "inside.py"),
        "pkg.elsewhere": module("pkg.elsewhere", outside),
        "pkg.builtinish": types.ModuleType("pkg.builtinish"),  # no __file__
        "unrelated": module("unrelated", outside),
    }
    entries = imported_package_entries("pkg", modules=table)
    assert [path for path, _ in entries] == ["pkg/__init__.py", "pkg/inside.py"]
    assert imported_package_hash("pkg", modules=table) == canonical_content_hash(entries)


def test_imported_package_hash_tracks_what_was_actually_imported(tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "extra.py").write_text("a = 1\n", encoding="utf-8")

    def module(name, filename):
        mod = types.ModuleType(name)
        mod.__file__ = str(filename)
        return mod

    minimal = {"pkg": module("pkg", package_dir / "__init__.py")}
    fuller = dict(minimal, **{"pkg.extra": module("pkg.extra", package_dir / "extra.py")})
    assert imported_package_hash("pkg", modules=minimal) != imported_package_hash(
        "pkg", modules=fuller
    )


def test_imported_package_hash_of_an_unimported_package_is_the_empty_hash():
    assert imported_package_hash("pkg", modules={}) == canonical_content_hash([])


def test_the_real_package_hashes_without_error():
    """Smoke: the live ``sys.modules`` table must not contain a shape that
    trips the collector (namespace packages, extension modules, editable shims).
    """
    value = imported_package_hash("fastworkflow")
    assert value.startswith("sha256:")
    entries = imported_package_entries("fastworkflow")
    assert ("fastworkflow/runtime_manifest.py", ) == tuple(
        path for path, _ in entries if path.endswith("runtime_manifest.py")
    )
    assert all(not os.path.isabs(path) for path, _ in entries)


# ======================================================================
# Schema (fix-i6h.1)
# ======================================================================


def test_unknown_top_level_key_is_rejected():
    """A typo'd declaration that parses is a declaration nobody notices is
    missing; §17.4 also forbids silently accepting newer state."""
    with pytest.raises(ValidationError):
        RuntimeManifest(schema_version=1, manifest_version="1.0.0", contexts_typo={})


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(ValidationError, match="schema_version 2 is not supported"):
        RuntimeManifest(schema_version=2, manifest_version="1.0.0")


def test_descend_requires_a_target_context():
    with pytest.raises(ValidationError, match="requires a target context"):
        NavigationEffect(kind="descend")


def test_the_bounded_dynamic_target_set_is_accepted():
    effect = NavigationEffect(kind="temporary", target_contexts=("A", "B"))
    assert effect.declared_targets() == ("A", "B")


def test_temporary_may_omit_its_target_set():
    """§7.4 makes the bounded set optional. A composite whose delegates happen
    not to navigate has a real temporary transition and no target list, and the
    IDO generator emits exactly that shape — rejecting it here would break a
    conformant manifest from the other side of the contract.
    """
    assert NavigationEffect(kind="temporary").declared_targets() == ()
    assert NavigationEffect(kind="temporary", target_contexts=()).declared_targets() == ()


def test_non_navigating_kinds_cannot_declare_a_target():
    with pytest.raises(ValidationError, match="cannot declare a target context"):
        NavigationEffect(kind="none", target_context="Permission")


def test_conditional_parameter_contradicts_kind_none():
    with pytest.raises(ValidationError, match="contradicts navigation kind 'none'"):
        NavigationEffect(kind="none", when_parameter_present="uid")


def test_absent_effect_contract_reads_as_unknown_never_read_only():
    """§7.3, and the safe direction: unknown is treated as write-capable."""
    assert CommandDeclaration().effect_kind() == "unknown"
    metadata = merge_and_gate(
        _manifest(commands={"App/mystery": CommandDeclaration()}), deployment_features={}
    )
    assert metadata.effect_kind("App/mystery") == "unknown"
    assert metadata.effect_kind("App/never_heard_of_it") == "unknown"


def test_invalid_capture_classification_is_rejected():
    with pytest.raises(ValidationError):
        CommandDeclaration(capture={"uid": "secret-sauce"})


def test_manifest_absent_from_disk_loads_as_none(tmp_path):
    assert load_manifest(str(tmp_path)) is None


def test_manifest_on_disk_round_trips(tmp_path):
    source = _manifest(
        workflow_fingerprint="sha256:abc",
        features={"turn_budget_v1": "shadow"},
        contexts={"Account": ContextDeclaration(occupiable=True)},
        commands={
            "Account/list": CommandDeclaration(
                navigation_effect=NavigationEffect(
                    kind="descend", target_context="Permission", remains_active=True
                ),
                effect=EffectContract(kind="read_only"),
                capture={"permission_uid": "identifier"},
            )
        },
    )
    (tmp_path / "workflow_runtime.json").write_text(
        source.model_dump_json(exclude_none=True), encoding="utf-8"
    )
    assert load_manifest(str(tmp_path)) == source


# ======================================================================
# Occupancy (fix-i6h.1)
# ======================================================================


def test_occupiable_contexts_are_listed_sorted():
    metadata = merge_and_gate(
        _manifest(
            contexts={
                "Permission": ContextDeclaration(occupiable=True),
                "Account": ContextDeclaration(occupiable=True),
                "Resource": ContextDeclaration(occupiable=False),
            }
        ),
        deployment_features={},
    )
    assert metadata.occupiable_contexts() == ("Account", "Permission")


def test_undeclared_occupancy_is_none_not_false():
    """"Nobody said" and "said no" are different answers.

    Collapsing them would make every context of a manifest-less workflow
    unenterable, which is the compatibility break §7.1 forbids.
    """
    metadata = merge_and_gate(
        _manifest(contexts={"Resource": ContextDeclaration(occupiable=False)}),
        deployment_features={},
    )
    assert metadata.is_occupiable("Resource") is False
    assert metadata.is_occupiable("NeverDeclared") is None


def test_core_contexts_are_non_occupiable_without_the_workflow_saying_so():
    metadata = merge_and_gate(None, deployment_features={})
    assert metadata.is_occupiable("IntentDetection") is False
    assert metadata.is_occupiable("ErrorCorrection") is False
    assert metadata.occupiable_contexts() == ()


def test_occupancy_completeness_excludes_internal_contexts():
    """§7.3: a workflow author is not responsible for declaring
    ``IntentDetection``."""
    metadata = merge_and_gate(
        _manifest(contexts={"Account": ContextDeclaration(occupiable=True)}),
        deployment_features={},
    )
    problems = occupancy_completeness_problems(
        metadata, ["Account", "IntentDetection", "ErrorCorrection", "Permission"]
    )
    assert problems == [
        "context 'Permission' has no occupancy declaration in the workflow manifest"
    ]


def test_occupancy_completeness_is_silent_without_a_workflow_manifest():
    metadata = merge_and_gate(None, deployment_features={})
    assert occupancy_completeness_problems(metadata, ["Account", "Permission"]) == []


# ======================================================================
# Merge precedence (fix-i6h.1, arch §7.3)
# ======================================================================


def test_workflow_declarations_are_added_alongside_core_ones():
    metadata = merge_and_gate(
        _manifest(
            contexts={"Account": ContextDeclaration(occupiable=True)},
            commands={
                "Account/close": CommandDeclaration(effect=EffectContract(kind="write"))
            },
        ),
        deployment_features={},
    )
    assert metadata.is_occupiable("Account") is True
    assert metadata.effect_kind("Account/close") == "write"
    assert metadata.navigation_effect("IntentDetection/go_up").kind == "ascend"


def test_workflow_cannot_make_an_internal_context_occupiable():
    with pytest.raises(ManifestConformanceError) as excinfo:
        merge_and_gate(
            _manifest(contexts={"IntentDetection": ContextDeclaration(occupiable=True)}),
            deployment_features={},
        )
    assert "occupiable" in str(excinfo.value)
    assert "never weaker" in str(excinfo.value)


def test_workflow_may_make_a_core_context_stricter():
    """Tightening is the allowed direction, and the merge keeps the tighter
    value rather than the workflow's."""
    core = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        contexts={"Internal": ContextDeclaration(occupiable=True)},
    )
    metadata = merge_and_gate(
        _manifest(contexts={"Internal": ContextDeclaration(occupiable=False)}),
        deployment_features={},
        core_manifest=core,
    )
    assert metadata.is_occupiable("Internal") is False


def test_workflow_cannot_downgrade_a_core_effect_contract():
    core = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={"core/danger": CommandDeclaration(effect=EffectContract(kind="write"))},
    )
    with pytest.raises(ManifestConformanceError, match="downgrades the effect contract"):
        merge_and_gate(
            _manifest(
                commands={
                    "core/danger": CommandDeclaration(effect=EffectContract(kind="read_only"))
                }
            ),
            deployment_features={},
            core_manifest=core,
        )


def test_workflow_may_raise_a_core_effect_contract_and_the_raise_sticks():
    metadata = merge_and_gate(
        _manifest(
            commands={
                "IntentDetection/go_up": CommandDeclaration(
                    effect=EffectContract(kind="write")
                )
            }
        ),
        deployment_features={},
    )
    assert metadata.effect_kind("IntentDetection/go_up") == "write"


def test_unknown_is_stricter_than_read_only_and_may_be_declared():
    """§6.6.1 makes an unknown contract write-capable for the strict write gate,
    so unknown sits above read_only rather than below it."""
    metadata = merge_and_gate(
        _manifest(
            commands={
                "IntentDetection/go_up": CommandDeclaration(
                    effect=EffectContract(kind="unknown")
                )
            }
        ),
        deployment_features={},
    )
    assert metadata.effect_kind("IntentDetection/go_up") == "unknown"


def test_workflow_cannot_redeclare_a_core_navigation_effect():
    with pytest.raises(ManifestConformanceError, match="redeclares the navigation effect"):
        merge_and_gate(
            _manifest(
                commands={
                    "IntentDetection/go_up": CommandDeclaration(
                        navigation_effect=NavigationEffect(kind="none")
                    )
                }
            ),
            deployment_features={},
        )


def test_restating_a_core_navigation_effect_identically_is_allowed():
    """A generator that emits its whole command surface should not have to
    special-case the core commands it happens to describe correctly."""
    metadata = merge_and_gate(
        _manifest(
            commands={
                "IntentDetection/go_up": CommandDeclaration(
                    navigation_effect=NavigationEffect(kind="ascend")
                )
            }
        ),
        deployment_features={},
    )
    assert metadata.navigation_effect("IntentDetection/go_up").kind == "ascend"


def test_workflow_may_supply_a_declaration_the_core_manifest_omits():
    """Adding where core was silent is an addition, not an override, and must
    not be dropped by the merge."""
    core = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={"core/quiet": CommandDeclaration(effect=EffectContract(kind="read_only"))},
    )
    metadata = merge_and_gate(
        _manifest(
            commands={
                "core/quiet": CommandDeclaration(
                    navigation_effect=NavigationEffect(kind="ascend"),
                    capture={"uid": "identifier"},
                )
            }
        ),
        deployment_features={},
        core_manifest=core,
    )
    assert metadata.navigation_effect("core/quiet").kind == "ascend"
    assert metadata.capture_classification("core/quiet", "uid") == "identifier"


def test_workflow_cannot_reclassify_a_core_capture_field():
    core = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={"core/lookup": CommandDeclaration(capture={"uid": "identifier"})},
    )
    with pytest.raises(ManifestConformanceError, match="reclassifies 'uid'"):
        merge_and_gate(
            _manifest(commands={"core/lookup": CommandDeclaration(capture={"uid": "user-text"})}),
            deployment_features={},
            core_manifest=core,
        )


def test_every_problem_is_reported_at_once():
    """An operator fixing a manifest one restart at a time is the failure mode
    the accumulating error avoids."""
    with pytest.raises(ManifestConformanceError) as excinfo:
        merge_and_gate(
            _manifest(
                features={"nonsense_v1": "shadow", "structured_outcomes_v9": "shadow"},
                contexts={"IntentDetection": ContextDeclaration(occupiable=True)},
                commands={
                    "IntentDetection/go_up": CommandDeclaration(
                        navigation_effect=NavigationEffect(kind="reset")
                    )
                },
            ),
            deployment_features={},
        )
    reported = "\n".join(excinfo.value.problems)
    assert "unknown feature 'nonsense_v1'" in reported
    assert "version 9 is not supported" in reported
    assert "occupiable" in reported
    assert "redeclares the navigation effect" in reported


# ======================================================================
# Dual gating (fix-i6h.3; arch §7.1, §17.1; FW-REQ-019C clause 2)
# ======================================================================


def test_manifest_presence_alone_enables_nothing():
    """FW-REQ-019C clause 2, stated as a test: a manifest declaring every
    feature in shadow, with no deployment enablement, runs nothing."""
    metadata = merge_and_gate(
        _manifest(features={feature: "shadow" for feature in FEATURES_BY_ID}),
        deployment_features={},
    )
    assert set(metadata.feature_modes.values()) == {"off"}
    assert not any(metadata.is_enabled(feature) for feature in FEATURES_BY_ID)


def test_both_sides_enabling_yields_the_feature():
    metadata = merge_and_gate(
        _manifest(features={"turn_budget_v1": "shadow"}),
        deployment_features={"turn_budget_v1": "shadow"},
    )
    assert metadata.feature_mode("turn_budget_v1") == "shadow"
    assert metadata.is_enabled("turn_budget_v1")


def test_deployment_may_be_more_restrictive_than_the_workflow():
    metadata = merge_and_gate(
        _manifest(features={"turn_budget_v1": "enforce"}),
        deployment_features={"turn_budget_v1": "shadow"},
    )
    assert metadata.feature_mode("turn_budget_v1") == "shadow"


def test_deployment_may_switch_a_declared_feature_off_entirely():
    metadata = merge_and_gate(
        _manifest(features={"turn_budget_v1": "enforce"}),
        deployment_features={"turn_budget_v1": "off"},
    )
    assert metadata.feature_mode("turn_budget_v1") == "off"


def test_deployment_asking_for_more_than_the_workflow_fails_startup():
    """Not clamped — failed. The disagreement means one of the two is wrong, and
    guessing which is how a feature reaches production unnoticed."""
    with pytest.raises(ManifestConformanceError, match="equally or more restrictive"):
        merge_and_gate(
            _manifest(features={"turn_budget_v1": "shadow"}),
            deployment_features={"turn_budget_v1": "enforce"},
        )


def test_deployment_enabling_a_feature_the_workflow_never_declared_fails():
    with pytest.raises(ManifestConformanceError, match="equally or more restrictive"):
        merge_and_gate(_manifest(), deployment_features={"turn_budget_v1": "shadow"})


def test_unknown_feature_id_in_the_manifest_fails_startup():
    with pytest.raises(ManifestConformanceError, match="unknown feature 'telepathy_v1'"):
        merge_and_gate(_manifest(features={"telepathy_v1": "shadow"}), deployment_features={})


def test_unsupported_version_of_a_known_feature_reports_the_version():
    """A version skew reads differently from an invented name — the first says
    the engine is older than the workflow — so it gets its own message."""
    with pytest.raises(ManifestConformanceError) as excinfo:
        merge_and_gate(
            _manifest(features={"turn_budget_v7": "shadow"}), deployment_features={}
        )
    message = str(excinfo.value)
    assert "version 7 is not supported" in message
    assert "supported: v1" in message


def test_unknown_feature_id_from_the_deployment_fails_startup():
    with pytest.raises(ManifestConformanceError, match=DEPLOYMENT_FEATURES_ENV_VAR):
        merge_and_gate(_manifest(), deployment_features={"telepathy_v1": "shadow"})


def test_decision_signals_has_no_enforce_mode_in_p0():
    """§17.1: enforcement is the P1 decision table and depends on a calibration
    that does not exist until G2A, so the mode is undefined rather than merely
    unbuilt."""
    with pytest.raises(ManifestConformanceError, match="does not support"):
        merge_and_gate(
            _manifest(features={"decision_signals_v1": "enforce"}),
            deployment_features={"decision_signals_v1": "enforce"},
        )
    metadata = merge_and_gate(
        _manifest(features={"decision_signals_v1": "shadow"}),
        deployment_features={"decision_signals_v1": "shadow"},
    )
    assert metadata.feature_mode("decision_signals_v1") == "shadow"


def test_unknown_feature_id_queried_at_runtime_reads_as_off():
    metadata = merge_and_gate(None, deployment_features={})
    assert metadata.feature_mode("telepathy_v1") == "off"
    assert metadata.is_enabled("telepathy_v1") is False


# ---------------------------------------------------------------- env parsing


def test_deployment_features_parse_from_the_environment_mapping():
    modes, problems = deployment_features_from_env(
        {DEPLOYMENT_FEATURES_ENV_VAR: "turn_budget_v1=shadow, structured_outcomes_v1=enforce"}
    )
    assert problems == []
    assert modes == {"turn_budget_v1": "shadow", "structured_outcomes_v1": "enforce"}


def test_env_file_values_win_over_the_os_environment(monkeypatch):
    """The precedence the rest of fastWorkflow uses, reproduced without the leaf
    importing the package to reach ``fastworkflow._env_vars``."""
    monkeypatch.setenv(DEPLOYMENT_FEATURES_ENV_VAR, "turn_budget_v1=enforce")
    assert deployment_env({})[DEPLOYMENT_FEATURES_ENV_VAR] == "turn_budget_v1=enforce"
    merged = deployment_env({DEPLOYMENT_FEATURES_ENV_VAR: "turn_budget_v1=shadow"})
    assert merged[DEPLOYMENT_FEATURES_ENV_VAR] == "turn_budget_v1=shadow"
    assert deployment_features_from_env(merged)[0] == {"turn_budget_v1": "shadow"}


def test_deployment_env_ignores_none_valued_env_file_entries(monkeypatch):
    """``dotenv_values`` yields None for a bare key; it must not shadow the OS
    environment with nothing."""
    monkeypatch.setenv(DEPLOYMENT_FEATURES_ENV_VAR, "turn_budget_v1=shadow")
    merged = deployment_env({DEPLOYMENT_FEATURES_ENV_VAR: None})
    assert merged[DEPLOYMENT_FEATURES_ENV_VAR] == "turn_budget_v1=shadow"


def test_absent_or_blank_deployment_declaration_enables_nothing():
    assert deployment_features_from_env({}) == ({}, [])
    assert deployment_features_from_env({DEPLOYMENT_FEATURES_ENV_VAR: "   "}) == ({}, [])


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ("turn_budget_v1", "is not 'feature_id=mode'"),
        ("turn_budget_v1=", "is not 'feature_id=mode'"),
        ("turn_budget_v1=sometimes", "unknown mode"),
        ("turn_budget_v1=off,turn_budget_v1=shadow", "twice"),
    ],
)
def test_malformed_deployment_declarations_are_reported_not_ignored(raw, fragment):
    _, problems = parse_deployment_features(raw)
    assert any(fragment in problem for problem in problems)


def test_malformed_deployment_declaration_fails_startup(tmp_path):
    with pytest.raises(ManifestConformanceError, match="is not 'feature_id=mode'"):
        check_startup_conformance(
            str(tmp_path), env={DEPLOYMENT_FEATURES_ENV_VAR: "turn_budget_v1"}
        )


# ======================================================================
# No-manifest compatibility: absent manifest preserves current behavior
# ======================================================================


def test_a_workflow_with_no_manifest_starts_clean(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    metadata = check_startup_conformance(root, env={})

    assert metadata.has_workflow_manifest is False
    assert metadata.workflow_fingerprint is None
    assert set(metadata.feature_modes.values()) == {"off"}
    # Only the framework's own declarations are in play, and none of them says
    # anything about a workflow context.
    assert dict(metadata.contexts) == dict(CORE_MANIFEST.contexts)
    assert metadata.is_occupiable("Account") is None
    assert metadata.occupiable_contexts() == ()


def test_no_manifest_plus_a_deployment_declaration_still_fails_closed(tmp_path):
    """A deployment cannot enable a feature for a workflow that never declared
    one — including by having no manifest at all."""
    with pytest.raises(ManifestConformanceError, match="equally or more restrictive"):
        check_startup_conformance(
            str(tmp_path), env={DEPLOYMENT_FEATURES_ENV_VAR: "turn_budget_v1=shadow"}
        )


@pytest.mark.parametrize(
    "workflow", ["example_workflow", "hello_world_workflow", "todo_list_workflow"]
)
def test_the_real_test_workflows_have_no_manifest_and_start_unchanged(workflow):
    """Golden parity on the actual workflows the suite runs.

    None of them ships a manifest, so startup conformance must be a no-op for
    all three — this is the concrete form of "absent manifest preserves current
    behavior exactly", checked against real trees rather than a fixture.
    """
    root = os.path.join(os.path.dirname(__file__), workflow)
    assert load_manifest(root) is None
    metadata = check_startup_conformance(root, env={})
    assert metadata.has_workflow_manifest is False
    assert set(metadata.feature_modes.values()) == {"off"}


def test_dropping_a_manifest_into_a_real_workflow_activates_nothing(tmp_path):
    """FW-REQ-019C clause 2 against a real tree: copy a trained test workflow,
    drop in a manifest declaring every feature, and nothing turns on."""
    source = os.path.join(os.path.dirname(__file__), "hello_world_workflow")
    root = tmp_path / "hello_world_workflow"
    shutil.copytree(source, root)
    (root / "workflow_runtime.json").write_text(
        _manifest(
            features={feature: "shadow" for feature in FEATURES_BY_ID},
            contexts={"*": ContextDeclaration(occupiable=True)},
        ).model_dump_json(exclude_none=True),
        encoding="utf-8",
    )

    metadata = check_startup_conformance(str(root), env={})
    assert metadata.has_workflow_manifest is True
    assert set(metadata.feature_modes.values()) == {"off"}
    assert metadata.occupiable_contexts() == ("*",)


def test_the_core_manifest_covers_every_context_mutating_internal_command():
    """§11.3: every core context-mutating command must be covered by the core
    manifest. ``go_up`` and ``reset_context`` are the two that move context, and
    the diagnostics for undeclared context changes rest on them being here."""
    assert CORE_MANIFEST.commands["IntentDetection/go_up"].navigation_effect.kind == "ascend"
    assert (
        CORE_MANIFEST.commands["IntentDetection/reset_context"].navigation_effect.kind
        == "reset"
    )
    assert all(
        declaration.navigation_effect is not None
        for declaration in CORE_MANIFEST.commands.values()
    )
    assert all(
        declaration.effect_kind() == "read_only"
        for declaration in CORE_MANIFEST.commands.values()
    )
