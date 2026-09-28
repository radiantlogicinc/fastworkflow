"""Workflow runtime manifest: schema, canonical hashing, and dual feature gates.

Slice 0 of the complex-task architecture (arch §7, §16 Slice 0). A workflow may
ship an optional ``workflow_runtime.json`` at its root declaring what its
contexts and commands *are*: which contexts can be occupied, where a command
leaves the current context, whether it writes, and what class of data each of
its parameters holds. IDO emits it from ``gen_ido_scaffold.py``; nothing
hand-edits it.

**Nothing in this module changes runtime behavior.** It parses, validates,
merges, hashes and gates. No caller acts on the result yet — later slices do.
That is the point of Slice 0: the declarations and the gates land first, with
every feature ``off``, so that when a feature is built there is already a
governed way to turn it on.

Three properties are load-bearing, and each exists because its absence caused a
specific defect.

**File presence does nothing** (FW-REQ-019C clause 2). Dropping a manifest into
a workflow root activates nothing at all. A feature runs only when the workflow
manifest declares it *and* the deployment enables that exact feature id, and
then only at the more restrictive of the two modes. A deployment that asks for
more than the workflow declared is not quietly clamped — it fails startup
conformance, because the disagreement means one of the two is wrong and
guessing which would be how a feature reaches production unnoticed.

**Canonical hashes are content, not location.** ``canonical_content_hash`` is
reproducible from the bytes of a tree and nothing else — no absolute path, no
mtime, no deployment directory. This closes verification-register item O3: a
manifest hash existed in the provenance chain whose recipe was written down
nowhere, so the value could not be recomputed and therefore attested to
nothing. The recipe is now stated in full at ``canonical_content_hash`` and
implemented identically on the IDO side (``gen_ido_scaffold.canonical_fingerprint``).
Both implementations exist deliberately — a generator that has not written its
tree to disk yet cannot call a reader — so the recipe is the contract between
them and any drift in it is a silent disagreement about workflow identity.

Note that fastWorkflow already has a *different* thing called a fingerprint,
``command_directory.compute_commands_source_fingerprint``, which hashes absolute
paths plus ``(size, mtime_ns)``. It is correct for its job (invalidating
``command_directory.json`` when sources change) and wrong for this one: it moves
when a file is touched without being edited and when a tree is copied
elsewhere. The two coexist and must not be substituted for each other.

**Core declarations cannot be weakened** (arch §7.3). fastWorkflow ships its own
manifest for the internal contexts — ``IntentDetection``, ``ErrorCorrection``,
``wildcard`` — and a workflow manifest merges *over* it only in the safe
direction: it may add domain contexts and commands, and it may make a core
declaration stricter, but a workflow cannot declare an internal context
occupiable or downgrade a core command's effect contract to ``read_only``.

Leaf-module constraint (arch §22): standard library and Pydantic only. This must
not import ``fastworkflow`` (the package root), the WEC, ``turn.py``, or command
routing, at import time or otherwise, so that it does not deepen the existing
package-root/WEC/turn import cycle. Deployment configuration therefore arrives
as an explicit ``Mapping`` defaulting to ``os.environ``; a caller that wants
env-file precedence passes ``fastworkflow._env_vars`` itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_FILENAME = "workflow_runtime.json"

# Only schema 1 exists. A manifest declaring anything else is rejected rather
# than best-effort parsed: §17.4 requires that older engines never silently
# accept newer state.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

FeatureMode = Literal["off", "shadow", "enforce"]

# How much a mode permits. Used for the restrictive-only comparison between the
# deployment and the workflow declaration; not a schema value.
_MODE_PERMISSIVENESS: dict[str, int] = {"off": 0, "shadow": 1, "enforce": 2}

EffectKind = Literal["read_only", "write", "unknown"]

# How much caution an effect contract demands. `unknown` sits above `read_only`
# because §6.6.1 makes unknown write-capable for StrictWriteGate and high
# consequence for reporting — an absent contract is a reason for more care, not
# less. A workflow may raise a core command along this order, never lower it.
_EFFECT_SEVERITY: dict[str, int] = {"read_only": 0, "unknown": 1, "write": 2}

NavigationKind = Literal["none", "descend", "ascend", "reset", "temporary"]

# What a parameter *is*. The workflow's to say; what to do about it (omit,
# digest, bounded-text, opaque-ref — arch §6.6) is deployment capture policy and
# is deliberately not declarable here, so that a workflow cannot override a
# deployment's evidence profile.
DataClassification = Literal[
    "identifier",
    "opaque-payload",
    "controlled-vocabulary",
    "user-text",
]


class ManifestConformanceError(Exception):
    """A manifest, or a deployment's gating of one, failed startup conformance.

    Carries *every* problem found rather than the first. An operator fixing a
    manifest one restart at a time is the failure mode this avoids; the message
    is the numbered list.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        detail = "\n".join(
            f"  {i}. {problem}" for i, problem in enumerate(self.problems, start=1)
        )
        super().__init__(
            f"workflow runtime manifest failed startup conformance:\n{detail}"
        )


# ----------------------------------------------------------------------
# Canonical versioned feature ids (arch §17.1)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDefinition:
    """One P0 capability, at one version, with the modes it actually supports."""

    feature_id: str
    name: str
    version: int
    supported_modes: tuple[FeatureMode, ...]


def _feature(name: str, version: int, *, modes: tuple[FeatureMode, ...]) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}_v{version}", name=name, version=version, supported_modes=modes
    )


_ALL_MODES: tuple[FeatureMode, ...] = ("off", "shadow", "enforce")
_SHADOW_ONLY: tuple[FeatureMode, ...] = ("off", "shadow")

# THE canonical set. A workflow manifest naming anything not in here fails
# startup conformance; that is how a generator authored against the design doc
# rather than against this module finds out it disagreed (which is the intended
# way to find out — see the IDO generator's MANIFEST_FEATURES comment).
FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    _feature("structured_outcomes", 1, modes=_ALL_MODES),
    # No `enforce`: enforcement is the P1 decision table (arch §19.6) and depends
    # on a calibration that does not exist until G2A, so the mode is not merely
    # unimplemented, it is undefined (§17.1).
    _feature("decision_signals", 1, modes=_SHADOW_ONLY),
    _feature("turn_budget", 1, modes=_ALL_MODES),
    _feature("command_identity", 1, modes=_ALL_MODES),
    _feature("context_capabilities", 1, modes=_ALL_MODES),
    _feature("external_operations", 1, modes=_ALL_MODES),
    _feature("side_effect_safety", 1, modes=_ALL_MODES),
)

FEATURES_BY_ID: dict[str, FeatureDefinition] = {
    definition.feature_id: definition for definition in FEATURE_DEFINITIONS
}

_SUPPORTED_VERSIONS_BY_NAME: dict[str, set[int]] = {}
for _definition in FEATURE_DEFINITIONS:
    _SUPPORTED_VERSIONS_BY_NAME.setdefault(_definition.name, set()).add(_definition.version)


def _split_feature_id(feature_id: str) -> tuple[str, Optional[int]]:
    """``"turn_budget_v1"`` -> ``("turn_budget", 1)``; unparseable -> ``(id, None)``."""
    name, separator, version = feature_id.rpartition("_v")
    if not separator or not version.isdigit():
        return feature_id, None
    return name, int(version)


def _feature_id_problem(feature_id: str, source: str) -> Optional[str]:
    """Why this feature id is not usable, or None if it is.

    An unsupported *version* of a known feature reads differently from an
    outright unknown id — the first says "this engine is older than the
    workflow", the second says "somebody invented a name" — so they get separate
    messages even though both fail.
    """
    if feature_id in FEATURES_BY_ID:
        return None
    name, version = _split_feature_id(feature_id)
    if version is not None and name in _SUPPORTED_VERSIONS_BY_NAME:
        supported = sorted(_SUPPORTED_VERSIONS_BY_NAME[name])
        return (
            f"{source} declares feature '{feature_id}': feature '{name}' exists but "
            f"version {version} is not supported by this engine "
            f"(supported: {', '.join(f'v{v}' for v in supported)})"
        )
    known = ", ".join(sorted(FEATURES_BY_ID))
    return f"{source} declares unknown feature '{feature_id}' (known features: {known})"


# ----------------------------------------------------------------------
# Canonical content hashing (arch §7.1, §6.3; closes register item O3)
# ----------------------------------------------------------------------


def canonical_content_hash(entries: Iterable[tuple[str, bytes]]) -> str:
    """``sha256:<hex>`` over ``(relative path, exact bytes)`` pairs.

    THE RECIPE, written out because being independently recomputable is the
    entire purpose of the value. ``gen_ido_scaffold.canonical_fingerprint``
    implements the same steps; any extra element on either side is a place the
    two can silently disagree about workflow identity.

    1. Collect ``(relative path, exact bytes)`` for every file that determines
       what the tree does. Directories, permissions, ownership and timestamps
       are not inputs.
    2. Normalize each path to POSIX separators, relative to the tree root. No
       absolute path, no deployment location.
    3. Sort by normalized path, so collection order cannot change the value.
    4. Per entry feed the hash: path bytes, NUL, decimal content length in
       ASCII, NUL, content bytes. Nothing else — no terminator.
    5. SHA-256, rendered ``sha256:<hex>``.

    Step 4 is self-delimiting, and the length is load-bearing rather than
    decoration. With NUL separators alone, one file holding ``b"b\\0c\\0d"`` and
    two files holding ``b"b"`` and ``b"d"`` serialize identically, so two
    different trees would share a hash.

    Absolute paths and duplicates raise rather than hash. Both are precisely the
    defect this function exists to close — an absolute path makes the value
    depend on where the tree was checked out, and a duplicated path means the
    caller's collection lost a file — and neither is detectable in the output.
    """
    normalized: dict[str, bytes] = {}
    for path, content in entries:
        posix_path = str(path).replace(os.sep, "/").replace("\\", "/")
        segments = posix_path.split("/")
        drive_letter = len(segments[0]) == 2 and segments[0][1] == ":"
        if not posix_path or posix_path.startswith("/") or drive_letter or ".." in segments:
            raise ValueError(
                f"canonical_content_hash requires tree-relative paths; got {path!r}. "
                "An absolute path (or one escaping the root) makes the hash depend "
                "on where the tree lives, which is the property this hash exists "
                "to not have."
            )
        if posix_path in normalized:
            raise ValueError(
                f"canonical_content_hash received duplicate path {posix_path!r}; "
                "the caller's collection has lost a file."
            )
        normalized[posix_path] = bytes(content)

    digest = hashlib.sha256()
    for path in sorted(normalized):
        content = normalized[path]
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


# Which revision of the file-selection rule below produced a given hash.
#
# The recipe and the scope are versioned separately because they move for
# different reasons: ``canonical_content_hash`` changes how bytes become a
# digest, everything from here down changes which bytes are offered to it. Both
# change the value, and a consumer holding a hash needs to know which one moved.
#
# Provenance records this beside the hash (``provenance.WorkflowProvenance``)
# because §6.3 makes result bundles immutable, so a recorded hash outlives every
# rule that could recompute it. Without the version, a bundle written under one
# scope and checked under another reports a workflow that drifted; with it, the
# same comparison reports that it cannot be made — which is the true answer.
#
# Bumping this means saying what changed about the selection, since declining to
# compare is the only thing a consumer can do with a version it does not know.
#
# v1: the rules below including ``FRAMEWORK_ARTIFACT_PREFIX`` and the dot rule.
# v2 (16 September 2026, bead ido-gxc): two classes leave the selection - a
#     ``benchmarks`` tree directly under the workflow root, and the runtime
#     observability store at the root. Both are things a workflow accumulates
#     rather than things that determine what it does, and on the IDO tree that
#     forced the decision five of the six benchmark files were git-ignored, so
#     the declared identity could not be reproduced from a clean clone - the one
#     property this hash exists to have. A value declared under v1 therefore
#     reads as ``incomparable`` here, not as stale: the tree it described did not
#     drift, it was a different question.
WORKFLOW_SCOPE_RULE_VERSION = 2

# What determines what a workflow does. Deliberately narrow: command sources,
# the context models and the manifest-adjacent JSON, and the guidance markdown
# that distillation acts on.
DEFAULT_CONTENT_SUFFIXES: tuple[str, ...] = (".py", ".json", ".md")

# Under a workflow root but not part of its identity. Every name here is a
# fastWorkflow convention that fastWorkflow itself writes — the three artifact
# directories, ``Insights`` from ``distillation.py``, and Python's own — so core
# is naming its own output, not learning any particular workflow's filenames.
DEFAULT_EXCLUDED_TREES: frozenset[str] = frozenset({
    "___command_info",        # trained artifacts: derived, gigabytes, not source
    "___workflow_contexts",   # runtime state
    "___convo_info",          # runtime state
    "Insights",               # distillation output: documentation, not behavior
    "__pycache__",
})

# The three artifact directories above share this prefix, and listing them by
# name closes instances where the class is what recurs — the same gap the dot
# rule was opened to close (fix-oxr). A framework that adds a fourth would put
# runtime state into workflow identity until somebody extended the list, and a
# generator transcribing that list would diverge until it did too (fix-ijf).
# Structural, so not parameterized: a caller narrowing ``excluded_trees`` is
# choosing which conventions to skip, not asking for framework state to count.
FRAMEWORK_ARTIFACT_PREFIX = "___"
DEFAULT_EXCLUDED_FILES: frozenset[str] = frozenset({
    MANIFEST_FILENAME,                 # carries the value: circular
    "fastworkflow.env",                # local and secret; identical builds differ
    "fastworkflow.passwords.env",
    "fastworkflow.env.example",
    "fastworkflow.passwords.env.example",
})

# Excluded directly under the workflow root only, unlike DEFAULT_EXCLUDED_TREES
# above, which matches at any depth. ``benchmarks`` is an ordinary word and a
# workflow may legitimately have a command package called that, so excluding it
# at every depth would quietly drop source from the workflow's identity. At the
# root it is the corpus a workflow is *measured with*: it changes when the
# measurements change and not when the workflow does, and it is routinely
# git-ignored, which made the declared identity unreproducible from a clean
# clone (bead ido-gxc, scope rule v2).
DEFAULT_EXCLUDED_ROOT_TREES: frozenset[str] = frozenset({
    "benchmarks",
})

# The runtime observability store ``fastworkflow.observability.store`` writes
# under the workflow root, named here rather than left to the suffix rule.
# Nothing it writes today carries a content suffix, so this exclusion changes no
# selection on any tree that exists - and that is the argument for stating it:
# ``observability.sqlite3``, its ``-wal`` and ``-shm`` sidecars and the legacy
# ``observability.sqlite3.offload-handles.sqlite3`` older builds wrote (removed
# when a store next opens) are one store under several names, and an exclusion that holds only "in practice" is the same latent
# divergence in a smaller box, waiting for the first sidecar somebody writes as
# JSON. Matched as a name prefix, and at the root only, for that reason.
RUNTIME_STORE_PREFIX = "observability.sqlite3"


def workflow_content_entries(
    workflow_folderpath: str,
    *,
    suffixes: Iterable[str] = DEFAULT_CONTENT_SUFFIXES,
    excluded_trees: Iterable[str] = DEFAULT_EXCLUDED_TREES,
    excluded_files: Iterable[str] = DEFAULT_EXCLUDED_FILES,
    excluded_root_trees: Iterable[str] = DEFAULT_EXCLUDED_ROOT_TREES,
    runtime_store_prefix: str = RUNTIME_STORE_PREFIX,
) -> list[tuple[str, bytes]]:
    """``(relative posix path, bytes)`` for the files that define this workflow.

    The reader half of the recipe. A generator that has not written its tree
    yet builds the same entry list from its pending emission instead; both feed
    ``canonical_content_hash``.

    **Dot-prefixed files and directories are excluded.** A leading dot
    is the usual signal for "tool bookkeeping, not source", and a generator's
    bookkeeping file is the one class of content most likely to be unstable: a
    generated manifest carrying a ``generated_at`` timestamp makes this hash
    *time-dependent*, which was measured as two different hashes for the same
    tree across a simulated regeneration — precisely the property §7.1 says the
    hash must not have. Excluding the class is what stops the next bookkeeping
    file from doing it again, and it does so without core needing to know any
    particular workflow's filenames.

    **Two classes are excluded at the root and only at the root** (scope rule
    v2): a ``benchmarks`` directory, and any file whose name
    begins with ``RUNTIME_STORE_PREFIX``. Root-relative because both are
    conventions about what sits *beside* a workflow rather than names that mean
    the same thing wherever they appear: ``_commands/benchmarks/`` is as much
    source as its neighbours, and a nested file named after the store is not the
    store. Excluding them at any depth would drop real source, which is the
    error that does not announce itself — a hash that stays still while
    behaviour moves.

    With that rule, this function selects exactly the files IDO's generator
    selects on the current tree, so the hash it produces equals the
    ``workflow_fingerprint`` IDO declares — which is what makes
    ``verify_workflow_fingerprint`` usable at all.

    **The two scope rules agree on that tree, not by construction.** They are
    structurally different: IDO filters by a name list applied to the first path
    segment only, this filters by dot-prefix and ``FRAMEWORK_ARTIFACT_PREFIX``
    at any depth. Measured on a synthetic tree, IDO selects and this
    excludes three classes: ``_commands/Insights/x.md`` and
    ``_commands/___command_info/x.json``, where the same names nested deeper
    escape a first-segment test; any dot-prefixed file or directory, for which
    IDO has no rule at all; and non-``.sqlite3`` payloads under
    ``___convo_info`` or ``___workflow_contexts``, which IDO's list omits.
    ``__pycache__`` at any depth agrees, because IDO drops its cruft names at
    every level of its walk. ``tests/test_context_runtime_manifest.py`` pins
    each class against a transcription of IDO's filter.

    **What no test here can check.** IDO's ``fingerprint_entries`` does not walk
    a tree the way this does. It selects the generator's *pending* emission
    unioned with the declared-preserved paths, and only then applies its filter.
    The two coincide because ``replace_tree`` makes disk equal
    pending-union-preserved immediately afterwards — an unwritten third
    invariant, and the load-bearing one, since it is what makes a fingerprint
    IDO declares before writing describe the tree that ends up on disk. Neither
    ``pending`` nor the preservation manifest exists outside a generator run, so
    the transcription pins IDO's filter and not its selector; the invariant can
    only be asserted where it holds, on the IDO side.
    """
    root = Path(workflow_folderpath)
    if not root.is_dir():
        # os.walk yields nothing for a missing directory, so without this a
        # typo'd path produces the hash of the empty tree: a confident,
        # well-formed sha256 that every wrong path shares. It would be recorded
        # as the workflow's identity, and would make a manifest look stale when
        # the real problem is the path.
        raise ValueError(f"workflow folder does not exist: {workflow_folderpath}")

    suffix_tuple = tuple(suffixes)
    excluded_tree_set = frozenset(excluded_trees)
    excluded_file_set = frozenset(excluded_files)
    excluded_root_tree_set = frozenset(excluded_root_trees)

    entries: list[tuple[str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        at_root = os.path.relpath(dirpath, root) == os.curdir
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in excluded_tree_set
            and not (at_root and name in excluded_root_tree_set)
            and not name.startswith(".")
            and not name.startswith(FRAMEWORK_ARTIFACT_PREFIX)
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            if (at_root and runtime_store_prefix
                    and filename.startswith(runtime_store_prefix)):
                continue
            full = Path(dirpath) / filename
            relative = full.relative_to(root).as_posix()
            if relative in excluded_file_set:
                continue
            if not relative.endswith(suffix_tuple):
                continue
            entries.append((relative, full.read_bytes()))
    return entries


def workflow_content_hash(workflow_folderpath: str, **kwargs) -> str:
    """Canonical content hash of a workflow tree on disk."""
    return canonical_content_hash(workflow_content_entries(workflow_folderpath, **kwargs))


@dataclass(frozen=True)
class FingerprintVerification:
    """Whether a manifest's declared ``workflow_fingerprint`` matches the tree.

    Four outcomes, deliberately not two. ``not_declared`` is not a failure: a
    generator that cannot compute the hash omits the field rather than writing a
    guess, and §7.2 makes it optional. Collapsing it into "mismatch" would
    report a conformant manifest as broken.

    ``incomparable`` is the fourth, and it exists because the digests being
    unequal has two possible causes that look identical in the output. Either
    the tree changed since generation — the answer this check is for — or the
    two sides selected different files, because the generator ran under an
    older ``WORKFLOW_SCOPE_RULE_VERSION``. Reporting the second as the first
    sends someone to regenerate a workflow that never drifted, and the whole
    reason the scope rule is versioned is that this distinction is otherwise
    unreachable: both values are well-formed sha256 digests of real trees, and
    nothing in either one says which question it answered.

    Note that unequal digests are the *only* case where the rule version is
    consulted. Equal digests mean the generator's selection and this engine's
    selection produced identical bytes on this tree, which is a genuine pass
    whatever versions the two sides wrote down — and treating it as a failure
    would be a false alarm on a healthy workflow.
    """

    declared: Optional[str]
    computed: str
    # None means the generator predates the versioned scope rule. That is read
    # as agreement rather than as a reason to give up: v1 is the only rule that
    # has ever existed, so an unversioned generator necessarily used it.
    declared_scope_rule_version: Optional[int] = None
    computed_scope_rule_version: int = WORKFLOW_SCOPE_RULE_VERSION

    @property
    def not_declared(self) -> bool:
        return self.declared is None

    @property
    def matches(self) -> bool:
        return self.declared is not None and self.declared == self.computed

    @property
    def incomparable(self) -> bool:
        """Digests differ, and were produced by different file selections."""
        if self.not_declared or self.matches:
            return False
        return (
            self.declared_scope_rule_version is not None
            and self.declared_scope_rule_version != self.computed_scope_rule_version
        )

    def problem(self) -> Optional[str]:
        """A conformance message, or None when there is nothing to report."""
        if self.not_declared or self.matches:
            return None
        if self.incomparable:
            return (
                f"workflow manifest declares workflow_fingerprint {self.declared} "
                f"under scope rule v{self.declared_scope_rule_version}, but this "
                f"engine selects files under v{self.computed_scope_rule_version} "
                f"and hashes the tree to {self.computed}; the two describe "
                "different questions, so this is not evidence that the workflow "
                "changed (regenerate under this engine to get a comparable value)"
            )
        return (
            f"workflow manifest declares workflow_fingerprint {self.declared} but "
            f"the tree hashes to {self.computed}; the manifest is stale with "
            "respect to the workflow it describes (regenerate it)"
        )


def verify_workflow_fingerprint(
    workflow_folderpath: str,
    manifest: Optional[RuntimeManifest],
    **kwargs,
) -> FingerprintVerification:
    """Compare a manifest's declared fingerprint against the tree on disk.

    A mismatch usually means the manifest is stale: someone changed a command
    and did not regenerate. That matters most to evaluation provenance, which
    records the declared value — recording a fingerprint that does not describe
    the tree it was captured from is worse than recording nothing. When the
    manifest declares a different scope rule the mismatch means something else
    entirely; see ``FingerprintVerification.incomparable``.

    This is offered as a check rather than performed inside ``load_manifest``
    because it costs a full tree read, and because Slice 0 changes no behavior:
    see ``check_startup_conformance``'s ``verify_fingerprint`` flag for opting
    into enforcement.
    """
    return FingerprintVerification(
        declared=manifest.workflow_fingerprint if manifest else None,
        computed=workflow_content_hash(workflow_folderpath, **kwargs),
        declared_scope_rule_version=(
            manifest.workflow_scope_rule_version if manifest else None
        ),
    )


def imported_package_entries(
    package: str = "fastworkflow",
    *,
    modules: Optional[Mapping[str, ModuleType]] = None,
) -> list[tuple[str, bytes]]:
    """``(path relative to the package's parent, bytes)`` for the imported tree.

    Arch §6.3 requires the source hash to cover "the actually imported
    fastWorkflow module tree ... not merely the checkout path". The distinction
    matters because the checkout can contain modules a run never imports (the
    training stack, the FastAPI server) whose contents say nothing about what
    that run did.

    Only files under the package directory are included. A module whose
    ``__file__`` lives elsewhere — an editable-install shim, a namespace
    package, a C extension — is skipped rather than reached for, because a path
    outside the package has no stable relative form and would reintroduce the
    location dependence the hash exists to avoid.
    """
    module_table = sys.modules if modules is None else modules
    root_module = module_table.get(package)
    root_file = getattr(root_module, "__file__", None) if root_module else None
    if not root_file:
        return []
    package_dir = Path(root_file).resolve().parent
    parent_dir = package_dir.parent

    entries: dict[str, bytes] = {}
    for name, module in list(module_table.items()):
        if name != package and not name.startswith(f"{package}."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_file() or package_dir not in path.parents:
            continue
        entries[path.relative_to(parent_dir).as_posix()] = path.read_bytes()
    return sorted(entries.items())


def imported_package_hash(
    package: str = "fastworkflow",
    *,
    modules: Optional[Mapping[str, ModuleType]] = None,
) -> str:
    """Canonical content hash of the package modules imported *right now*.

    The value is a function of the import set at the moment of the call, which
    is what makes it evidence about a run rather than about a checkout — and
    also what makes it meaningless if captured at an arbitrary point. Capture it
    where provenance is recorded, and use ``imported_package_entries`` to see
    which modules produced a given value.
    """
    return canonical_content_hash(imported_package_entries(package, modules=modules))


# ----------------------------------------------------------------------
# Schema (arch §7.2)
# ----------------------------------------------------------------------


class _Strict(BaseModel):
    """Reject unknown keys.

    A typo'd declaration that parses is a declaration nobody notices is missing,
    and §17.4 requires that an older engine not silently accept newer state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class NavigationEffect(_Strict):
    """Where a command leaves the current context (arch §7.4).

    P0 does not auto-navigate on these. They drive metadata validation,
    context-before/after checks, menu filtering, and undeclared-context-change
    diagnostics.
    """

    kind: NavigationKind
    target_context: Optional[str] = None
    # The bounded dynamic set of §7.4, for a command that can land in one of
    # several contexts (or, for `temporary`, passes through them).
    target_contexts: Optional[tuple[str, ...]] = None
    # Conditional navigation: the command lists without this parameter and
    # descends with it. §11.3 requires the named field to exist in the command's
    # input schema; that check needs the command surface and so lives with the
    # build-time validator, not here.
    when_parameter_present: Optional[str] = None
    remains_active: Optional[bool] = None

    @model_validator(mode="after")
    def _targets_match_kind(self) -> "NavigationEffect":
        targets = self.declared_targets()
        # `descend` is the only kind meaningless without a target: you descend
        # *into* something. `temporary` may omit one, because §7.4 makes the
        # bounded dynamic set something an effect "may also declare" — a
        # composite whose delegates happen not to navigate has a real temporary
        # transition and no target list, and rejecting that would be stricter
        # than the contract the IDO generator was written against.
        if self.kind == "descend":
            if not targets:
                raise ValueError("navigation kind 'descend' requires a target context")
        elif self.kind != "temporary" and targets:
            raise ValueError(
                f"navigation kind '{self.kind}' cannot declare a target context; "
                f"got {sorted(targets)}"
            )
        if self.when_parameter_present and self.kind == "none":
            raise ValueError(
                "'when_parameter_present' declares a conditional transition, which "
                "contradicts navigation kind 'none'"
            )
        return self

    def declared_targets(self) -> tuple[str, ...]:
        """Every context this effect can reach, conditional or not."""
        targets: list[str] = []
        if self.target_context:
            targets.append(self.target_context)
        targets.extend(self.target_contexts or ())
        return tuple(sorted(set(targets)))


class EffectContract(_Strict):
    """Whether a command writes. Absent or invalid reads as ``unknown`` (§7.3)."""

    kind: EffectKind


class ContextDeclaration(_Strict):
    """What a context is: occupiable, and how its handle is projected."""

    occupiable: bool
    # A projector name is a promise that a registered projector exists. None
    # means type-only context evidence, which is all P0 has until the
    # ContextHandle work lands.
    handle_projector: Optional[str] = None


class CommandDeclaration(_Strict):
    """What a command does to context, to the world, and with its parameters."""

    navigation_effect: Optional[NavigationEffect] = None
    effect: Optional[EffectContract] = None
    capture: Optional[dict[str, DataClassification]] = None

    def effect_kind(self) -> EffectKind:
        """``unknown`` when no contract is declared — never ``read_only`` (§7.3)."""
        return self.effect.kind if self.effect else "unknown"


class RuntimeManifest(_Strict):
    """A parsed ``workflow_runtime.json`` (or the framework's core manifest)."""

    schema_version: int
    manifest_version: str
    # Absent is legitimate: a generator that cannot compute the canonical hash
    # omits it rather than writing a guess. Presence is not verified here —
    # verifying it means rehashing the tree, which is the caller's decision.
    workflow_fingerprint: Optional[str] = None
    # Which file selection produced `workflow_fingerprint`. The generator's
    # value, not this engine's: reading the local constant at verification time
    # would record the reader's rule and conceal the very drift the version
    # exists to surface. Absent means a generator predating the versioned rule,
    # which is read as v1 — the only rule that has ever existed.
    #
    # Modelled here rather than left to `extra="forbid"` because a generator
    # cannot emit a key this class does not carry: the manifest is rejected by
    # name and every startup conformance check on that workflow fails.
    workflow_scope_rule_version: Optional[int] = None
    features: dict[str, FeatureMode] = Field(default_factory=dict)
    contexts: dict[str, ContextDeclaration] = Field(default_factory=dict)
    commands: dict[str, CommandDeclaration] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(
                f"schema_version {value} is not supported by this engine "
                f"(supported: {supported})"
            )
        return value


def load_manifest(workflow_folderpath: str) -> Optional[RuntimeManifest]:
    """Parse ``<workflow>/workflow_runtime.json``, or None when it is absent.

    Absent is the overwhelmingly common case and is not an error: a workflow
    without a manifest keeps current behavior exactly, with every feature off
    and only the core declarations in play.
    """
    path = Path(workflow_folderpath) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return RuntimeManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        # A malformed or schema-invalid manifest is a conformance failure like
        # any other, and must arrive as one: the CLI and FastAPI startup paths
        # catch ManifestConformanceError, so a raw JSONDecodeError from a
        # trailing comma would reach the operator as a traceback instead of a
        # message naming the file.
        raise ManifestConformanceError([f"{path} could not be read as a manifest: {e}"]) from e


# ----------------------------------------------------------------------
# The framework-owned core manifest (arch §7.3)
# ----------------------------------------------------------------------

# fastWorkflow's own internal contexts and the commands that mutate context
# inside them. §11.3 requires every core context-mutating command to be covered
# here, and §7.3 requires these declarations to be unweakenable: a workflow that
# could declare `IntentDetection` occupiable, or `go_up` non-navigating, would
# defeat the undeclared-context-change diagnostics that the whole occupancy
# story rests on.
#
# These are the commands of the internal `command_metadata_extraction` workflow.
# `wildcard` is unqualified because it lives at that workflow's `_commands/`
# root; the rest are qualified by their context folder.
CORE_MANIFEST = RuntimeManifest(
    schema_version=1,
    manifest_version="1.0.0",
    features={},
    contexts={
        "IntentDetection": ContextDeclaration(occupiable=False),
        "ErrorCorrection": ContextDeclaration(occupiable=False),
    },
    commands={
        "IntentDetection/go_up": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="ascend"),
            effect=EffectContract(kind="read_only"),
        ),
        "IntentDetection/reset_context": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="reset"),
            effect=EffectContract(kind="read_only"),
        ),
        "IntentDetection/what_can_i_do": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="none"),
            effect=EffectContract(kind="read_only"),
        ),
        "IntentDetection/what_is_current_context": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="none"),
            effect=EffectContract(kind="read_only"),
        ),
        "ErrorCorrection/abort": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="none"),
            effect=EffectContract(kind="read_only"),
        ),
        "ErrorCorrection/you_misunderstood": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="none"),
            effect=EffectContract(kind="read_only"),
        ),
        # Resolves to command_name=None and walks the parent chain rather than
        # setting a context itself, so it declares no transition of its own.
        "wildcard": CommandDeclaration(
            navigation_effect=NavigationEffect(kind="none"),
            effect=EffectContract(kind="read_only"),
        ),
    },
)

CORE_CONTEXT_NAMES: frozenset[str] = frozenset(CORE_MANIFEST.contexts)
CORE_COMMAND_NAMES: frozenset[str] = frozenset(CORE_MANIFEST.commands)


# ----------------------------------------------------------------------
# Merge (arch §7.3) and the resolved view
# ----------------------------------------------------------------------


def _context_merge_problems(name: str, core: ContextDeclaration, workflow: ContextDeclaration) -> list[str]:
    problems: list[str] = []
    if workflow.occupiable and not core.occupiable:
        problems.append(
            f"workflow manifest declares core context '{name}' occupiable, but the "
            "core manifest declares it non-occupiable; core declarations may be "
            "made stricter, never weaker (arch §7.3)"
        )
    if (
        core.handle_projector is not None
        and workflow.handle_projector is not None
        and workflow.handle_projector != core.handle_projector
    ):
        problems.append(
            f"workflow manifest overrides the handle projector for core context "
            f"'{name}' ('{core.handle_projector}' -> '{workflow.handle_projector}')"
        )
    return problems


def _command_merge_problems(name: str, core: CommandDeclaration, workflow: CommandDeclaration) -> list[str]:
    problems: list[str] = []

    if workflow.effect is not None:
        core_severity = _EFFECT_SEVERITY[core.effect_kind()]
        workflow_severity = _EFFECT_SEVERITY[workflow.effect.kind]
        if workflow_severity < core_severity:
            problems.append(
                f"workflow manifest downgrades the effect contract of core command "
                f"'{name}' from '{core.effect_kind()}' to '{workflow.effect.kind}'; a "
                "core safety declaration may only be made stricter (arch §7.3)"
            )

    # Navigation has no ordering — a transition is a fact about the core command's
    # body, not a policy dial — so any disagreement is rejected rather than
    # ranked. Restating the identical declaration is harmless and allowed, since
    # a generator that emits the full surface should not have to special-case
    # core commands.
    if (
        workflow.navigation_effect is not None
        and core.navigation_effect is not None
        and workflow.navigation_effect != core.navigation_effect
    ):
        problems.append(
            f"workflow manifest redeclares the navigation effect of core command "
            f"'{name}' ('{core.navigation_effect.kind}' -> "
            f"'{workflow.navigation_effect.kind}'); core transitions are facts "
            "about framework code and are not overridable"
        )

    if workflow.capture is not None and core.capture is not None:
        for field_name, classification in workflow.capture.items():
            core_classification = core.capture.get(field_name)
            if core_classification is not None and core_classification != classification:
                problems.append(
                    f"workflow manifest reclassifies '{field_name}' of core command "
                    f"'{name}' ('{core_classification}' -> '{classification}')"
                )
    return problems


@dataclass(frozen=True)
class RuntimeMetadata:
    """The validated merge of the core and workflow manifests, plus the gates.

    This is what the rest of the runtime will read once features exist. It is
    deliberately a read-only view with accessors rather than a bag of dicts, so
    that the "absent means unknown, never read_only" rule of §7.3 is enforced by
    the only path callers have.
    """

    contexts: Mapping[str, ContextDeclaration]
    commands: Mapping[str, CommandDeclaration]
    feature_modes: Mapping[str, FeatureMode]
    workflow_fingerprint: Optional[str]
    has_workflow_manifest: bool
    # Carried alongside the fingerprint because the two are only meaningful
    # together. A scope rule this engine does not know is deliberately not a
    # startup failure, unlike an unknown `schema_version`: it does not make the
    # manifest unparseable or any of its declarations wrong, it narrows what one
    # optional field can be compared against, and that is reported where the
    # comparison happens (`FingerprintVerification.incomparable`) rather than by
    # refusing to run a workflow whose contexts and commands are all valid.
    workflow_scope_rule_version: Optional[int] = None

    def is_occupiable(self, context_name: str) -> Optional[bool]:
        """True/False when declared, None when the manifest is silent.

        None is not False. A caller that cannot distinguish "declared
        non-occupiable" from "not declared" would treat every undeclared context
        in a manifest-less workflow as unenterable, which is the compatibility
        break §7.1 forbids.
        """
        declaration = self.contexts.get(context_name)
        return None if declaration is None else declaration.occupiable

    def occupiable_contexts(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, ctx in self.contexts.items() if ctx.occupiable)
        )

    def effect_kind(self, command_name: str) -> EffectKind:
        """``unknown`` for an undeclared command — never ``read_only`` (§7.3)."""
        declaration = self.commands.get(command_name)
        return declaration.effect_kind() if declaration else "unknown"

    def navigation_effect(self, command_name: str) -> Optional[NavigationEffect]:
        """The declared transition, or None when undeclared.

        None means "nobody said", which §11.3 turns into a build-time failure for
        a command that actually moves context. It does not mean ``kind="none"``.
        """
        declaration = self.commands.get(command_name)
        return declaration.navigation_effect if declaration else None

    def capture_classification(self, command_name: str, field_name: str) -> Optional[DataClassification]:
        declaration = self.commands.get(command_name)
        if declaration is None or declaration.capture is None:
            return None
        return declaration.capture.get(field_name)

    def feature_mode(self, feature_id: str) -> FeatureMode:
        """The effective mode after dual gating. Unknown ids are ``off``.

        Unknown ids never reach here in a conformant startup — ``merge_and_gate``
        rejects them — so this is the answer for a caller asking about a feature
        this engine has never heard of, which is exactly "it is not running".
        """
        return self.feature_modes.get(feature_id, "off")

    def is_enabled(self, feature_id: str) -> bool:
        """Anything other than ``off``. Shadow counts: it runs, it just cannot act."""
        return self.feature_mode(feature_id) != "off"


# ----------------------------------------------------------------------
# Dual gating (arch §7.1, §17.1; FW-REQ-019C clause 2)
# ----------------------------------------------------------------------

# Comma-separated `feature_id=mode`, e.g.
# "structured_outcomes_v1=shadow,decision_signals_v1=shadow".
DEPLOYMENT_FEATURES_ENV_VAR = "FASTWORKFLOW_RUNTIME_FEATURES"


def parse_deployment_features(
    raw: Optional[str],
) -> tuple[dict[str, FeatureMode], list[str]]:
    """Parse the deployment feature string into ``(modes, problems)``.

    Returns problems rather than raising so that a bad deployment string and a
    bad manifest are reported together in one conformance failure.
    """
    modes: dict[str, FeatureMode] = {}
    problems: list[str] = []
    if not raw or not raw.strip():
        return modes, problems

    for clause in raw.split(","):
        clause = clause.strip()
        if not clause:
            continue
        feature_id, separator, mode = clause.partition("=")
        feature_id = feature_id.strip()
        mode = mode.strip()
        if not separator or not feature_id or not mode:
            problems.append(
                f"{DEPLOYMENT_FEATURES_ENV_VAR} clause {clause!r} is not "
                "'feature_id=mode'"
            )
            continue
        if mode not in _MODE_PERMISSIVENESS:
            problems.append(
                f"{DEPLOYMENT_FEATURES_ENV_VAR} sets '{feature_id}' to unknown mode "
                f"'{mode}' (expected off, shadow, or enforce)"
            )
            continue
        if feature_id in modes and modes[feature_id] != mode:
            problems.append(
                f"{DEPLOYMENT_FEATURES_ENV_VAR} sets '{feature_id}' twice, to "
                f"'{modes[feature_id]}' and '{mode}'"
            )
            continue
        modes[feature_id] = mode  # type: ignore[assignment]
    return modes, problems


def deployment_env(env_file_vars: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """``os.environ`` with workflow env-file values layered over it.

    Reproduces the precedence the rest of fastWorkflow uses (env file first,
    then the OS environment) without importing the package to reach
    ``fastworkflow._env_vars``, which a leaf module may not do (arch §22). The
    caller supplies that dict; this only merges.
    """
    merged = dict(os.environ)
    for key, value in (env_file_vars or {}).items():
        if value is not None:
            merged[key] = str(value)
    return merged


def deployment_features_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> tuple[dict[str, FeatureMode], list[str]]:
    """Read the deployment declaration from a mapping (default ``os.environ``).

    Takes a mapping rather than reaching for ``fastworkflow.get_env_var`` because
    this module is a leaf (arch §22). A caller wanting env-file-then-OS-env
    precedence passes ``deployment_env(fastworkflow._env_vars)``.
    """
    source = os.environ if env is None else env
    return parse_deployment_features(source.get(DEPLOYMENT_FEATURES_ENV_VAR))


def _gating_problems(
    manifest_features: Mapping[str, str],
    deployment_features: Mapping[str, str],
) -> tuple[dict[str, FeatureMode], list[str]]:
    """Resolve effective modes and collect every dual-gating violation."""
    problems: list[str] = []

    for feature_id in sorted(manifest_features):
        if problem := _feature_id_problem(feature_id, "workflow manifest"):
            problems.append(problem)
    for feature_id in sorted(deployment_features):
        if problem := _feature_id_problem(feature_id, DEPLOYMENT_FEATURES_ENV_VAR):
            problems.append(problem)

    for source, declared in (
        ("workflow manifest", manifest_features),
        (DEPLOYMENT_FEATURES_ENV_VAR, deployment_features),
    ):
        for feature_id, mode in sorted(declared.items()):
            definition = FEATURES_BY_ID.get(feature_id)
            if definition and mode not in definition.supported_modes:
                supported = ", ".join(definition.supported_modes)
                problems.append(
                    f"{source} sets '{feature_id}' to mode '{mode}', which the "
                    f"feature does not support (supported: {supported})"
                )

    effective: dict[str, FeatureMode] = {}
    for feature_id in sorted(FEATURES_BY_ID):
        # A feature the manifest does not mention is off, and the workflow has
        # therefore not enabled it. That is what makes file presence, and indeed
        # partial presence, inert.
        manifest_mode = manifest_features.get(feature_id, "off")
        deployment_mode = deployment_features.get(feature_id, "off")
        if manifest_mode not in _MODE_PERMISSIVENESS or deployment_mode not in _MODE_PERMISSIVENESS:
            continue  # already reported above; do not compound it

        if _MODE_PERMISSIVENESS[deployment_mode] > _MODE_PERMISSIVENESS[manifest_mode]:
            problems.append(
                f"deployment sets '{feature_id}' to '{deployment_mode}' but the "
                f"workflow manifest declares '{manifest_mode}'; the deployment may "
                "only be equally or more restrictive than the workflow (arch §7.1)"
            )
            continue
        effective[feature_id] = deployment_mode  # type: ignore[assignment]

    return effective, problems


def merge_and_gate(
    workflow_manifest: Optional[RuntimeManifest],
    *,
    deployment_features: Optional[Mapping[str, str]] = None,
    env: Optional[Mapping[str, str]] = None,
    core_manifest: RuntimeManifest = CORE_MANIFEST,
) -> RuntimeMetadata:
    """Validate, merge, and gate — the whole startup conformance check.

    Raises ``ManifestConformanceError`` listing every problem. A workflow with
    no manifest yields core declarations, every feature off, and behavior
    identical to today.
    """
    problems: list[str] = []

    if deployment_features is None:
        deployment_features, parse_problems = deployment_features_from_env(env)
        problems.extend(parse_problems)

    manifest_features = workflow_manifest.features if workflow_manifest else {}
    effective_modes, gating_problems = _gating_problems(manifest_features, deployment_features)
    problems.extend(gating_problems)

    contexts: dict[str, ContextDeclaration] = dict(core_manifest.contexts)
    commands: dict[str, CommandDeclaration] = dict(core_manifest.commands)

    if workflow_manifest is not None:
        for name, declaration in workflow_manifest.contexts.items():
            if core := core_manifest.contexts.get(name):
                problems.extend(_context_merge_problems(name, core, declaration))
                # Keep the stricter of the two rather than the workflow's, so a
                # rejected weakening cannot also silently take effect in a
                # deployment that catches the error and continues.
                contexts[name] = ContextDeclaration(
                    occupiable=core.occupiable and declaration.occupiable,
                    handle_projector=core.handle_projector or declaration.handle_projector,
                )
            else:
                contexts[name] = declaration

        for name, declaration in workflow_manifest.commands.items():
            if core := core_manifest.commands.get(name):
                problems.extend(_command_merge_problems(name, core, declaration))
                commands[name] = _stricter_command(core, declaration)
            else:
                commands[name] = declaration

    if problems:
        raise ManifestConformanceError(problems)

    return RuntimeMetadata(
        contexts=contexts,
        commands=commands,
        feature_modes=effective_modes,
        workflow_fingerprint=workflow_manifest.workflow_fingerprint if workflow_manifest else None,
        has_workflow_manifest=workflow_manifest is not None,
        workflow_scope_rule_version=(
            workflow_manifest.workflow_scope_rule_version if workflow_manifest else None
        ),
    )


def _stricter_command(core: CommandDeclaration, workflow: CommandDeclaration) -> CommandDeclaration:
    """Core command plus whatever the workflow added that raises strictness."""
    effect = core.effect
    if workflow.effect is not None:
        core_severity = _EFFECT_SEVERITY[core.effect_kind()]
        if _EFFECT_SEVERITY[workflow.effect.kind] > core_severity:
            effect = workflow.effect

    capture = dict(core.capture or {})
    for field_name, classification in (workflow.capture or {}).items():
        capture.setdefault(field_name, classification)

    return CommandDeclaration(
        # Core wins where it spoke; where it did not, the workflow is adding a
        # declaration rather than overriding one, and dropping it would lose
        # information the merge was given.
        navigation_effect=core.navigation_effect or workflow.navigation_effect,
        effect=effect,
        capture=capture or None,
    )


def check_startup_conformance(
    workflow_folderpath: str,
    *,
    deployment_features: Optional[Mapping[str, str]] = None,
    env: Optional[Mapping[str, str]] = None,
    verify_fingerprint: bool = False,
) -> RuntimeMetadata:
    """Load, validate, merge and gate a workflow's manifest at startup.

    The entry point for the CLI and FastAPI startup paths. A workflow with no
    manifest is not an error and not a warning: it is the normal case.

    ``verify_fingerprint`` additionally rehashes the tree and fails a manifest
    whose declared fingerprint does not describe it. It is off by default for
    two reasons: Slice 0 changes no behavior, and it costs a full tree read on
    every startup. An evaluation harness that is about to record the declared
    value in a provenance artifact is exactly the caller that should turn it on.
    """
    manifest = load_manifest(workflow_folderpath)
    metadata = merge_and_gate(
        manifest, deployment_features=deployment_features, env=env
    )
    if verify_fingerprint and manifest is not None:
        # Skipped when nothing is declared: the tree read is the expensive part
        # and its answer could not change the outcome.
        try:
            problem = verify_workflow_fingerprint(workflow_folderpath, manifest).problem()
        except (OSError, ValueError) as e:
            raise ManifestConformanceError(
                [f"could not hash {workflow_folderpath} to verify the manifest: {e}"]
            ) from e
        if problem:
            raise ManifestConformanceError([problem])
    return metadata


# ----------------------------------------------------------------------
# Startup metadata retention (arch §12.0 delta 4)
# ----------------------------------------------------------------------
#
# ``check_startup_conformance`` returns a ``RuntimeMetadata`` and both entry
# points — ``fastworkflow/run/__main__.py`` and
# ``fastworkflow/run_fastapi_mcp/__main__.py`` — discarded it. The declarations
# were therefore validated at startup and then thrown away: nothing at runtime
# could reach ``effect_kind()`` or ``capture_classification()``, so a workflow
# that declared its effect contracts got the same ``unknown`` as one that
# declared nothing. This table is the smallest thing that keeps them reachable;
# it is modelled on ``observability_store._sinks``, process-wide and keyed by
# workflow path.
#
# Registration is explicit rather than a side effect of the conformance check.
# A build step or a test validating a manifest for some other tree must not
# change what a running workflow reports about itself, and a check that silently
# published its result would do exactly that.
#
# An unregistered path answers None. Every reader of that None owes §7.3's rule:
# fall back to ``unknown``, never to ``read_only``.

_runtime_metadata_lock = threading.Lock()
_runtime_metadata: dict[str, RuntimeMetadata] = {}


@lru_cache(maxsize=256)
def _metadata_key(workflow_folderpath: str) -> str:
    """Resolved absolute path, matching what ``Workflow`` stores.

    ``Workflow.__init__`` resolves the folderpath it is given, so a table keyed
    on the raw CLI argument would miss for any relative or symlinked
    invocation — and would miss *silently*, reporting ``unknown`` for a workflow
    that had declared its effects. Resolution failures fall back to the string
    rather than raising: a lookup key is not worth failing a turn over.

    Cached because the lookup runs once per executed command and
    ``Path.resolve`` is a realpath syscall — measured at 14 µs, four fifths of
    the whole consequence assessment, for an answer that cannot change within a
    process. Capture overhead is bounded by FW-NFR-005, so the cheap fix is
    worth taking here rather than defending later.
    """
    try:
        return str(Path(workflow_folderpath).resolve())
    except Exception:
        return str(workflow_folderpath)


def register_runtime_metadata(
    workflow_folderpath: str, metadata: RuntimeMetadata
) -> None:
    """Retain a workflow's startup-conformance result for the process lifetime."""
    with _runtime_metadata_lock:
        _runtime_metadata[_metadata_key(workflow_folderpath)] = metadata


def get_runtime_metadata(workflow_folderpath: str) -> Optional[RuntimeMetadata]:
    """The retained metadata for a workflow, or None when nothing registered it.

    None is a real answer and not an error: an embedder that never ran a
    fastWorkflow entry point has no startup conformance step to register one.
    """
    return _runtime_metadata.get(_metadata_key(workflow_folderpath))


def clear_runtime_metadata() -> None:
    """Drop every retained registration (test isolation)."""
    with _runtime_metadata_lock:
        _runtime_metadata.clear()


def occupancy_completeness_problems(
    metadata: RuntimeMetadata,
    known_contexts: Iterable[str],
) -> list[str]:
    """Contexts the workflow left without an explicit occupancy declaration.

    §11.3 requires every context to declare occupancy, but §7.3 excludes the
    framework's internal contexts from the workflow's completeness obligation —
    a workflow author is not responsible for declaring ``IntentDetection``.

    Returns problems rather than raising: this is a build-time check whose
    caller decides whether an incomplete manifest is fatal, and it is not part
    of startup conformance (a manifest-less workflow must start).
    """
    if not metadata.has_workflow_manifest:
        return []
    return [
        f"context '{name}' has no occupancy declaration in the workflow manifest"
        for name in sorted(set(known_contexts) - CORE_CONTEXT_NAMES)
        if name not in metadata.contexts
    ]
