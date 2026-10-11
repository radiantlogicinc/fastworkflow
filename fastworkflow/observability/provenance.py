"""What produced this run: engine, workflow, and model identity.

The fastWorkflow-origin half of the evaluation provenance contract in
architecture §6.3. An evaluation harness embeds this in its own
`EvaluationProvenance` alongside the axes it owns — corpus, task contracts,
evaluators, tenant/principal/dataset/timeslot fingerprints, and backend
attestations. Those are deliberately absent here: fastWorkflow cannot attest to
them, and a provenance writer that invents a value from local configuration is
the failure §6.3 spends a paragraph forbidding.

The point of the module is that these values are *hard to get right one at a
time*, and a harness assembling them ad hoc gets them wrong in ways nothing
detects. Four traps, each measured on this codebase rather than imagined:

**The installed version is stale.** `importlib.metadata.version("fastworkflow")`
answers 2.30.1 while `pyproject.toml` says 3.2.0, because an editable install's
`dist-info` is written once and never tracks a version bump. Neither number is
"the" version, so both are recorded and their disagreement is itself evidence.

**A dirty tree makes the revision a lie.** A git revision identifies the code
only if nothing is modified on top of it. Dirtiness is therefore recorded beside
it, and `None` means "could not tell" — never `False`.

**`None` from the artifact resolver does not mean untrained.** A workflow on the
pre-versioning layout has no `current.json` and resolves to `None` while being
fully trained; both bundled `hello_world` workflows are in exactly that state.
`legacy_layout` carries the difference, and without it the record reports a
trained workflow as untrained.

**A cache hit means no model was called.** DSPy's cache is process-global, so a
run with caching on is not evidence about the model that appears in
`model_identifiers`. The cache state is a peer of the model string here, not a
footnote.

Naming, because two things in this repo are called a fingerprint and only one of
them is identity: `workflow_source_hash` is the canonical content hash of §7.1 —
reproducible from bytes, independent of path and mtime.
`command_surface_fingerprint` is `compute_commands_source_fingerprint`, which
hashes absolute paths plus `(size, mtime_ns)`. That one is correct for
invalidating a cache and wrong for identifying a workflow; it is recorded
because §6.3 asks for it, under a name that says which one it is.

Leaf module (arch §22): standard library and Pydantic only.
Values that live behind a non-leaf module — the trained-artifact id and layout
flag in `train.artifact_versioning`, the legacy fingerprint in
`command_directory`, DSPy's process-global cache state — are caller-supplied
arguments rather than something this module reaches for. That is about the
dependency graph, not about speed: importing anything under `fastworkflow.`
already executes the package root, so no arrangement of this module avoids that
cost. What it avoids is this module depending on the training stack and command
routing, which is what §22 is protecting.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Lock files worth hashing, enumerated rather than globbed. Recording *which*
# were found is itself provenance: a `uv.lock` appearing later should show up as
# a new key, not silently change what a single hash covers. Only `poetry.lock`
# exists in this repo today.
LOCK_FILENAMES: tuple[str, ...] = (
    "poetry.lock",
    "uv.lock",
    "requirements.txt",
    "Pipfile.lock",
)

# Every per-role model env var read anywhere in the codebase, paired with the
# key var that authenticates it. This is the first such registry here — until
# now every name was a string literal at its call site — so two things are worth
# stating rather than leaving to be rediscovered:
#
#   * `LLM_COMMAND_METADATA_GEN` pairs with `LITELLM_API_KEY_COMMANDMETADATA_GEN`.
#     There is no underscore between COMMAND and METADATA on the key side, so
#     code deriving the key name from the role name silently reads an unset
#     variable for this one role. That is why the pairing is a table and not a
#     format string.
#   * `LLM_RESPONSE_GEN` is an orphan. It appears in every env file and in
#     AGENTS.md, and no Python reads it. It is snapshotted below under
#     `ORPHAN_ROLE_VARS` because a value an operator believes is in effect is
#     worth having in the record, but it is not an active role.
LLM_ROLE_VARS: tuple[tuple[str, str], ...] = (
    ("LLM_SYNDATA_GEN", "LITELLM_API_KEY_SYNDATA_GEN"),
    ("LLM_PARAM_EXTRACTION", "LITELLM_API_KEY_PARAM_EXTRACTION"),
    ("LLM_PLANNER", "LITELLM_API_KEY_PLANNER"),
    ("LLM_AGENT", "LITELLM_API_KEY_AGENT"),
    ("LLM_CONVERSATION_STORE", "LITELLM_API_KEY_CONVERSATION_STORE"),
    ("LLM_COMMAND_METADATA_GEN", "LITELLM_API_KEY_COMMANDMETADATA_GEN"),
    ("LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION"),
    ("LLM_TEACHER_AGENT", "LITELLM_API_KEY_TEACHER_AGENT"),
    ("LLM_TEACHER_PLANNER", "LITELLM_API_KEY_TEACHER_PLANNER"),
    ("LLM_STUDENT_AGENT", "LITELLM_API_KEY_STUDENT_AGENT"),
    ("LLM_STUDENT_PLANNER", "LITELLM_API_KEY_STUDENT_PLANNER"),
)

ORPHAN_ROLE_VARS: tuple[str, ...] = ("LLM_RESPONSE_GEN",)

# Routing, not authentication: a model string prefixed `litellm_proxy/` is
# served from somewhere else entirely, which changes what the model
# identifier means.
PROXY_BASE_VAR = "LITELLM_PROXY_API_BASE"


def _digest(value: str, length: int = 12) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def secret_fingerprint(value: Optional[str]) -> Optional[str]:
    """Enough to tell two credentials apart, and nothing more.

    Never the value. A provenance artifact is written to disk and shipped
    around; the only question it needs to answer about a key is "was this the
    same key as last run", and a short digest answers it.
    """
    return None if not value else _digest(value)


def strip_url_credentials(url: Optional[str]) -> Optional[str]:
    """A URL with any ``user:password@`` removed.

    ``LITELLM_PROXY_API_BASE`` is read from the same credential-bearing env as
    the API keys, and embedding basic-auth in a proxy URL is common enough that
    storing it verbatim would put a password in an artifact this module's own
    docstring describes as written to disk and shipped around.
    """
    if not url:
        return None
    parsed = urlsplit(url)
    if "@" not in parsed.netloc:
        return url
    return urlunsplit(parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[-1]))


# ----------------------------------------------------------------------
# Engine identity
# ----------------------------------------------------------------------


def installed_version(distribution: str = "fastworkflow") -> Optional[str]:
    """Version recorded in installed distribution metadata, or None.

    For an editable install this is frozen at install time and drifts behind
    `pyproject.toml` with every bump — 2.30.1 versus 3.2.0 as measured here — so
    it is one of two answers rather than the answer.
    """
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


def source_version() -> Optional[str]:
    """Version from a `pyproject.toml` beside the package, or None.

    Present in a source or editable checkout, absent from an installed wheel.
    This is the one that matches the code actually running when the two differ.
    """
    try:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if not pyproject.is_file():
            return None
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        poetry = data.get("tool", {}).get("poetry", {}).get("version")
        return poetry or data.get("project", {}).get("version")
    except (OSError, ValueError, IndexError, AttributeError):
        # AttributeError covers a pyproject whose `tool` or `tool.poetry` is not
        # a table; the .get() chain assumes both are.
        return None


def _git_dir(start: Path) -> Optional[Path]:
    """Nearest `.git` walking upward, resolving the worktree/submodule pointer.

    A `.git` **file** ends the search whether or not its pointer resolves. It is
    an explicit repository boundary, and climbing past a broken one lands in the
    enclosing checkout and returns *that* repository's revision — a confident
    wrong answer, which is worse in a provenance record than no answer.
    """
    for directory in (start, *start.parents):
        entry = directory / ".git"
        if entry.is_dir():
            return entry
        if entry.is_file():
            text = entry.read_text(encoding="utf-8", errors="replace").strip()
            if not text.startswith("gitdir:"):
                return None
            candidate = Path(text.partition(":")[2].strip())
            if not candidate.is_absolute():
                candidate = directory / candidate
            return candidate.resolve() if candidate.is_dir() else None
    return None


def _read_ref(git_dir: Path, ref: str) -> Optional[str]:
    """Resolve a ref, loose file first.

    Loose refs shadow `packed-refs`, and in this very checkout the two disagree
    — loose `main` is one commit, packed `main` is an older one. A resolver that
    consulted `packed-refs` first would confidently report a revision the tree
    has not been at for some time.
    """
    # A linked worktree's git directory (`.git/worktrees/<name>`) holds only
    # the per-worktree refs; branch heads and `packed-refs` live in the common
    # directory its `commondir` file points at. Search the worktree directory
    # first (it shadows), then the common one. (3.3 port adaptation: the
    # observability branch is developed in a linked worktree, where the
    # single-directory lookup returned None.)
    for directory in _ref_dirs(git_dir):
        loose = directory / ref
        if loose.is_file():
            # An empty loose ref is corruption, not an answer: fall through to
            # packed-refs rather than reporting "no revision".
            if value := loose.read_text(encoding="utf-8", errors="replace").strip():
                return value
    for directory in _ref_dirs(git_dir):
        packed = directory / "packed-refs"
        if not packed.is_file():
            continue
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "^")):
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha or None
    return None


def _ref_dirs(git_dir: Path) -> tuple[Path, ...]:
    """`git_dir`, plus the common directory when `git_dir` is a linked worktree's."""
    commondir = git_dir / "commondir"
    if not commondir.is_file():
        return (git_dir,)
    text = commondir.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return (git_dir,)
    common = Path(text)
    if not common.is_absolute():
        common = git_dir / common
    common = common.resolve()
    return (git_dir, common) if common.is_dir() and common != git_dir else (git_dir,)


def _is_object_id(value: Optional[str]) -> bool:
    """A 40-hex (SHA-1) or 64-hex (SHA-256) object id and nothing else."""
    if not value or len(value) not in (40, 64):
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def git_revision(start_path: str) -> Optional[str]:
    """HEAD sha for the checkout containing `start_path`, or None.

    Pure standard library: no git binary, no subprocess, and no exception. A
    provenance capture that can fail is a provenance capture that gets wrapped
    in a bare `except` by its caller.

    The ref name is validated before it is joined onto the git directory, and
    the result is validated to be an object id. Without the first check an
    absolute `ref:` discards the git directory entirely (pathlib joins that
    way) and a `../` traverses out of it, so a corrupt or hostile `.git/HEAD`
    would turn provenance capture into an arbitrary file read whose contents get
    published as `source_revision`. Real git rejects both.
    """
    try:
        git_dir = _git_dir(Path(start_path).resolve())
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head.startswith("ref:"):
            return head if _is_object_id(head) else None
        ref = head.partition(":")[2].strip()
        if not ref.startswith("refs/") or ".." in ref.split("/"):
            return None
        revision = _read_ref(git_dir, ref)
        return revision if _is_object_id(revision) else None
    except (OSError, ValueError):
        return None


def git_is_dirty(repo_path: str, timeout_seconds: float = 10.0) -> Optional[bool]:
    """True, False, or None for "could not tell" — which is not False.

    Unlike the revision, this cannot be computed from `.git` without
    reimplementing git's index comparison, so it costs a subprocess. Worth it:
    a modified tree means the revision alone does not identify the code, and an
    evaluation harness recording a revision without that caveat is publishing a
    number that does not reproduce.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def lock_hashes(repo_root: str) -> dict[str, str]:
    """`sha256:<hex>` per lock file present. Absent files are simply omitted."""
    hashes: dict[str, str] = {}
    for name in LOCK_FILENAMES:
        path = Path(repo_root) / name
        try:
            if path.is_file():
                hashes[name] = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        except OSError:
            continue
    return hashes


class EngineProvenance(BaseModel):
    """Which fastWorkflow, built from what, running on which interpreter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    version_installed: Optional[str] = None
    version_source: Optional[str] = None
    source_revision: Optional[str] = None
    source_dirty: Optional[bool] = None
    source_hash: str
    imported_module_count: int
    dependency_lock_hashes: dict[str, str] = Field(default_factory=dict)
    python_version: str
    python_build: str

    @property
    def version_metadata_disagrees(self) -> bool:
        """Installed metadata and source disagree — the editable-install case."""
        return (
            self.version_installed is not None
            and self.version_source is not None
            and self.version_installed != self.version_source
        )

    @property
    def reproducible(self) -> bool:
        """Whether the recorded revision actually identifies the running code.

        False for a dirty tree and for an unknown revision. A run that is not
        reproducible may still be useful; it may not be reported as if it were.
        """
        return self.source_revision is not None and self.source_dirty is False


def capture_engine_provenance() -> EngineProvenance:
    """Identity of the fastWorkflow that is running right now.

    `source_hash` covers the *actually imported* module tree per §6.3, so this
    is meaningful only at a defined point in a run — capture it where the
    harness records provenance, not at import time.

    Everything here describes the running package, deliberately with no override
    parameter: revision, dirtiness and lock hashes are read from the checkout
    that `source_hash` was computed from, so they cannot come to describe
    different codebases.
    """
    root = Path(__file__).resolve().parent.parent
    repo_root = root.parent
    entries = imported_package_entries()
    installed = installed_version()
    source = source_version()
    return EngineProvenance(
        version=source or installed or "unknown",
        version_installed=installed,
        version_source=source,
        source_revision=git_revision(str(root)),
        source_dirty=git_is_dirty(str(repo_root)),
        source_hash=canonical_content_hash(entries),
        imported_module_count=len(entries),
        dependency_lock_hashes=lock_hashes(str(repo_root)),
        python_version=platform.python_version(),
        python_build=sys.version,
    )


# ----------------------------------------------------------------------
# Content hashing
# ----------------------------------------------------------------------


def canonical_content_hash(entries: Iterable[tuple[str, bytes]]) -> str:
    """``sha256:<hex>`` over ``(relative path, exact bytes)`` pairs.

    The recipe, written out because being independently recomputable is the
    point: sort entries by POSIX path; per entry feed the hash the path bytes, a
    NUL, the decimal content length, a NUL, then the content bytes. The length
    makes the encoding self-delimiting: with NUL separators alone, one file
    holding ``b"b\\0c\\0d"`` and two files holding ``b"b"`` and ``b"d"`` would
    hash identically.

    Absolute paths, paths escaping the root and duplicate paths raise rather than
    hash. An absolute path makes the value depend on where the tree lives, and a
    duplicate means the caller's collection lost a file.
    """
    normalized: dict[str, bytes] = {}
    for path, content in entries:
        posix_path = str(path).replace(os.sep, "/").replace("\\", "/")
        segments = posix_path.split("/")
        drive_letter = len(segments[0]) == 2 and segments[0][1] == ":"
        if not posix_path or posix_path.startswith("/") or drive_letter or ".." in segments:
            raise ValueError(
                f"canonical_content_hash requires tree-relative paths; got {path!r}."
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


# Which revision of the file-selection rule below produced a hash. Provenance
# records it beside `workflow_source_hash`: bundles are immutable, so a hash
# recorded under one rule must not be compared with one from another. Bump it
# when the selection changes.
#
# v1: the rules below, including FRAMEWORK_ARTIFACT_PREFIX and the dot rule.
# v2 (16 September 2026): a root-level `benchmarks` tree and the runtime store
#     at the root left the selection. Both accumulate beside a workflow rather
#     than determining what it does, and benchmarks were routinely git-ignored,
#     so the identity could not be reproduced from a clean clone.
WORKFLOW_SCOPE_RULE_VERSION = 2

# What determines what a workflow does: command sources, context models and the
# guidance markdown that distillation acts on.
_CONTENT_SUFFIXES: tuple[str, ...] = (".py", ".json", ".md")

# Under a workflow root but not part of its identity. Matched at any depth.
# Names starting with FRAMEWORK_ARTIFACT_PREFIX (trained artifacts, runtime state)
# are excluded structurally rather than listed.
_EXCLUDED_TREES: frozenset[str] = frozenset({"Insights", "__pycache__"})
_FRAMEWORK_ARTIFACT_PREFIX = "___"
# Matched directly under the workflow root only: `benchmarks` is an ordinary word,
# and a command package with that name is source.
_EXCLUDED_ROOT_TREES: frozenset[str] = frozenset({"benchmarks"})
# The runtime observability store and its sidecars, at the root only.
_RUNTIME_STORE_PREFIX = "observability.sqlite3"
_EXCLUDED_FILES: frozenset[str] = frozenset({
    "fastworkflow.env",  # local and secret; identical builds differ
    "fastworkflow.passwords.env",
    "fastworkflow.env.example",
    "fastworkflow.passwords.env.example",
})


def workflow_content_entries(workflow_folderpath: str) -> list[tuple[str, bytes]]:
    """``(relative posix path, bytes)`` for the files that define a workflow.

    Dot-prefixed files and directories are tool bookkeeping, not source, and are
    skipped; the rest is selected by suffix minus the exclusions above.
    """
    root = Path(workflow_folderpath)
    if not root.is_dir():
        # os.walk yields nothing for a missing path, so a typo would hash the empty
        # tree and record a well-formed identity for it.
        raise ValueError(f"workflow folder does not exist: {workflow_folderpath}")

    entries: list[tuple[str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        at_root = Path(dirpath) == root
        dirnames[:] = sorted(
            name for name in dirnames
            if not name.startswith((".", _FRAMEWORK_ARTIFACT_PREFIX))
            and name not in _EXCLUDED_TREES
            and not (at_root and name in _EXCLUDED_ROOT_TREES)
        )
        for filename in sorted(filenames):
            if filename.startswith(".") or (at_root and filename.startswith(_RUNTIME_STORE_PREFIX)):
                continue
            full = Path(dirpath) / filename
            relative = full.relative_to(root).as_posix()
            if relative in _EXCLUDED_FILES or not relative.endswith(_CONTENT_SUFFIXES):
                continue
            entries.append((relative, full.read_bytes()))
    return entries


def imported_package_entries() -> list[tuple[str, bytes]]:
    """``(path relative to the package's parent, bytes)`` for the imported fastworkflow modules.

    Arch §6.3 asks for the modules a run actually imported, not the checkout: a
    checkout holds modules a run never loads. A module whose file lies outside the
    package directory (an editable-install shim, a C extension) is skipped, because
    it has no stable relative path.
    """
    root_module = sys.modules.get("fastworkflow")
    root_file = getattr(root_module, "__file__", None) if root_module else None
    if not root_file:
        return []
    package_dir = Path(root_file).resolve().parent
    parent_dir = package_dir.parent

    entries: dict[str, bytes] = {}
    for name, module in list(sys.modules.items()):
        if name != "fastworkflow" and not name.startswith("fastworkflow."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_file() or package_dir not in path.parents:
            continue
        entries[path.relative_to(parent_dir).as_posix()] = path.read_bytes()
    return sorted(entries.items())


# ----------------------------------------------------------------------
# Workflow identity
# ----------------------------------------------------------------------


# Fields this record used to carry, from when the workflow's runtime manifest
# declared a fingerprint and feature modes. Rows written under that schema still
# load; the values are dropped rather than migrated.
_RETIRED_WORKFLOW_FIELDS = frozenset({
    "declared_workflow_fingerprint",
    "declared_scope_rule_version",
    "fingerprint_verified",
    "runtime_manifest_fingerprint",
    "runtime_feature_snapshot",
})


class WorkflowProvenance(BaseModel):
    """Which workflow, at which content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_source_hash: str
    # Which file-selection rule produced `workflow_source_hash`. Not a peer of
    # `RuntimeProvenance.schema_version`, which describes the shape of this
    # record: the two move independently, since a field can be added without
    # changing what gets hashed and the selection can change without changing
    # the shape. Two hashes are comparable only when this agrees; §6.3 makes
    # bundles immutable, so an old bundle cannot be re-derived under a new rule
    # and a reader that ignores this reports rule changes as workflow drift.
    workflow_scope_rule_version: int = WORKFLOW_SCOPE_RULE_VERSION
    command_surface_fingerprint: Optional[str] = None
    trained_artifact_id: Optional[str] = None
    trained_artifact_legacy_layout: bool = False

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            return {k: v for k, v in data.items() if k not in _RETIRED_WORKFLOW_FIELDS}
        return data

    @property
    def trained(self) -> Optional[bool]:
        """True when these fields establish training, None when they cannot.

        Never False. Neither input can distinguish "never trained" from "trained
        by some path that records nothing here", so claiming absence would be
        claiming more than the inputs support: a legacy-layout workflow is fully
        trained and still resolves to no version id, which is why the flag
        exists at all. A caller needing a definite negative has to look at the
        artifacts on disk.
        """
        if self.trained_artifact_id is not None:
            return True
        return True if self.trained_artifact_legacy_layout else None


def workflow_identity(workflow_folderpath: str) -> str:
    """The workflow's identity: the canonical content hash of its tree on disk.

    Shared by provenance and the readiness probe so both name a workflow the
    same way.
    """
    return canonical_content_hash(workflow_content_entries(workflow_folderpath))


def capture_workflow_provenance(
    workflow_folderpath: str,
    *,
    command_surface_fingerprint: Optional[str] = None,
    trained_artifact_id: Optional[str] = None,
    trained_artifact_legacy_layout: bool = False,
) -> WorkflowProvenance:
    """Identity of the workflow this run executed.

    The three keyword arguments are caller-supplied because reaching for them
    would make this module depend on command routing and the training stack,
    which §22 forbids: `command_surface_fingerprint` lives in
    `command_directory` and both artifact values in
    `train.artifact_versioning`. Callers get them from::

        from fastworkflow.command_directory import compute_commands_source_fingerprint
        from fastworkflow.train.artifact_versioning import (
            legacy_layout_in_use, resolve_current_version)

    Omitting `trained_artifact_legacy_layout` when the workflow is on the old
    layout records a trained workflow as untrained; see `WorkflowProvenance.trained`.
    """
    return WorkflowProvenance(
        workflow_source_hash=workflow_identity(workflow_folderpath),
        command_surface_fingerprint=command_surface_fingerprint,
        trained_artifact_id=trained_artifact_id,
        trained_artifact_legacy_layout=trained_artifact_legacy_layout,
    )


# ----------------------------------------------------------------------
# Model identity
# ----------------------------------------------------------------------


class RoleProvenance(BaseModel):
    """One LLM role: which model, reached how, authenticated with which key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Optional[str] = None
    via_proxy: bool = False
    proxy_base: Optional[str] = None
    proxy_credentials_fingerprint: Optional[str] = None
    api_key_fingerprint: Optional[str] = None


class ModelProvenance(BaseModel):
    """Which models a run was configured to use, and whether it called them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    roles: dict[str, RoleProvenance] = Field(default_factory=dict)
    orphan_role_vars: dict[str, str] = Field(default_factory=dict)
    config_hash: str
    cache_enabled: Optional[bool] = None

    @property
    def model_identifiers(self) -> dict[str, str]:
        """`role -> model string` for the roles that have one, per §6.3."""
        return {
            role: value.model
            for role, value in sorted(self.roles.items())
            if value.model
        }

    @property
    def evidence_about_the_model(self) -> Optional[bool]:
        """Whether this run's outputs say anything about the models named here.

        False when caching is on: a cache hit returns a stored completion and
        calls nothing, so the run is evidence about the cache. None when the
        caller did not say.
        """
        return None if self.cache_enabled is None else not self.cache_enabled


def capture_model_provenance(
    env: Optional[Mapping[str, str]] = None,
    *,
    cache_enabled: Optional[bool] = None,
    extra_config: Optional[Mapping[str, str]] = None,
) -> ModelProvenance:
    """Snapshot the per-role model configuration.

    `env` takes a mapping rather than reading `os.environ`, because none of
    these variables are in the OS environment in a normal deployment — they are
    loaded from the workflow's env files into `fastworkflow._env_vars`. A leaf
    may not reach for that, so callers pass the env-file values merged with
    `os.environ`; reading `os.environ` alone would capture nothing and record it
    as "no models configured".

    `cache_enabled` is caller-supplied because DSPy's cache is process-global
    state (`dspy.settings`), not configuration this module can see. Pass it: a
    run with the cache on is not evidence about the models named here.

    `extra_config` folds any additional behavioral settings the caller knows
    about — per-call `max_tokens`, `timeout`, `num_retries` literals — into
    `config_hash`, which otherwise covers only what is visible from the
    environment.
    """
    source = os.environ if env is None else env

    roles: dict[str, RoleProvenance] = {}
    proxy_base = strip_url_credentials(source.get(PROXY_BASE_VAR))
    proxy_credentials = _proxy_credentials(source.get(PROXY_BASE_VAR))
    for model_var, key_var in LLM_ROLE_VARS:
        model = source.get(model_var) or None
        via_proxy = bool(model and model.startswith("litellm_proxy/"))
        roles[model_var] = RoleProvenance(
            model=model,
            via_proxy=via_proxy,
            proxy_base=proxy_base if via_proxy else None,
            proxy_credentials_fingerprint=proxy_credentials if via_proxy else None,
            api_key_fingerprint=secret_fingerprint(source.get(key_var)),
        )

    orphans = {
        name: source[name] for name in ORPHAN_ROLE_VARS if source.get(name)
    }

    # Hash the behavior-determining settings, not the credentials: two runs
    # differing only in which key was used are the same configuration, and the
    # key fingerprints above already record that separately.
    #
    # Routed through canonical_content_hash rather than a joined string because
    # that function is self-delimiting: with a plain "k=v\n" join, a value
    # containing a newline is indistinguishable from an extra key, so two
    # genuinely different configurations can share a hash. That is the one thing
    # this value exists not to do, and the fix was already imported.
    material: list[tuple[str, bytes]] = []
    for role, value in sorted(roles.items()):
        material.append((f"role/{role}/model", (value.model or "").encode("utf-8")))
        material.append((f"role/{role}/proxy", (value.proxy_base or "").encode("utf-8")))
    for key, value in sorted((extra_config or {}).items()):
        material.append((f"extra/{key}", str(value).encode("utf-8")))
    material.append(("cache_enabled", str(cache_enabled).encode("utf-8")))

    return ModelProvenance(
        roles=roles,
        orphan_role_vars=orphans,
        config_hash=canonical_content_hash(material),
        cache_enabled=cache_enabled,
    )


def _proxy_credentials(url: Optional[str]) -> Optional[str]:
    """Fingerprint of any ``user:password`` in a proxy URL, or None."""
    if not url:
        return None
    userinfo = urlsplit(url).netloc.rpartition("@")[0]
    return secret_fingerprint(userinfo) if userinfo else None


# ----------------------------------------------------------------------
# Observability configuration (arch §12.0 delta 6, requirements §12.4)
# ----------------------------------------------------------------------


class ObservabilityProvenance(BaseModel):
    """Under what observability configuration a run's trace evidence was recorded.

    §12.4 requires an evidence run to record "observability configuration
    (`FW_OBS_*` values in effect) and the span-contract version", because trace
    evidence read without them is uninterpretable: two runs whose span-contract
    versions differ cannot be compared attribute by attribute at all.

    Values arrive as arguments rather than being read here, per the §22 leaf
    constraint — `observability_store` and `tracing` are not leaves, and this
    module deliberately cannot reach them.

    `dspy_history_enabled` is included because §12.4 makes it a precondition of
    evidence-grade capture: with DSPy history off there is no token or cost
    evidence at all, and its absence is indistinguishable from a run that cost
    nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    enabled: bool
    span_contract_version: int
    # Per-emitter attribute-contract versions (arch §12.0 delta 5). Additive and
    # defaulted: this model is frozen with `extra="forbid"`, so a required field
    # would reject every already-written record and every caller that predates it.
    #
    # It does not replace `span_contract_version` above, which stays the answer to
    # "did anything change between these two runs". This is the answer to "which
    # emitter", which the aggregate cannot give — and an empty map means the run
    # predates per-emitter versioning, not that no emitter declared one.
    span_contract_versions: dict[str, int] = Field(default_factory=dict)
    db_schema_version: int
    config: dict[str, str] = Field(default_factory=dict)
    dspy_history_enabled: Optional[bool] = None
    # Set only by an evidence-grade run; None means normal best-effort operation,
    # where drops are acceptable and nobody asserted otherwise.
    evidence_grade: Optional[bool] = None

    @property
    def evidence_interpretable(self) -> bool:
        """Whether this record is sufficient to read the run's traces.

        Every condition demands a positive answer. `dspy_history_enabled` defaults
        to None, and accepting "not False" here would report a run with no token
        or cost evidence as fully interpretable — the one direction this must not
        fail in, exactly as with `cache_enabled` on `ModelProvenance`.
        """
        return self.enabled and self.dspy_history_enabled is True


# ----------------------------------------------------------------------
# The whole record
# ----------------------------------------------------------------------


class RuntimeProvenance(BaseModel):
    """Everything about a run that originates in fastWorkflow.

    A harness embeds this in the §6.3 `EvaluationProvenance` next to the axes it
    owns. It is deliberately not that model: corpus, contract and evaluator
    bundles, tenant/principal/dataset/timeslot fingerprints, backend
    attestations and seed records are the harness's to attest, and §6.3 is
    explicit that a writer must record `unknown` rather than invent one from
    local configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    engine: EngineProvenance
    workflow: WorkflowProvenance
    models: ModelProvenance
    # Additive and optional: a harness that consumes no trace evidence owes no
    # observability configuration, and requiring one would break every existing caller to
    # record a fact that does not apply to it.
    observability: Optional[ObservabilityProvenance] = None

    @property
    def reproducible(self) -> bool:
        """Whether the *code and models* this run used are pinned by this record.

        Requires a known-clean revision and a cache known to be off. The
        workflow is pinned by `workflow_source_hash`.

        Every condition demands a positive answer, never merely the absence of a
        negative. `cache_enabled` defaults to None, so accepting "not False"
        here would report an ordinary clean-tree run as reproducible while
        nothing at all is known about whether a cached completion answered
        instead of the model — the one direction this property must not fail in.

        It says nothing about the dependency closure: `dependency_lock_hashes`
        records what was resolvable, and an empty one is not consulted here.
        """
        return (
            self.engine.reproducible
            and self.models.evidence_about_the_model is True
        )


def capture_runtime_provenance(
    workflow_folderpath: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    command_surface_fingerprint: Optional[str] = None,
    trained_artifact_id: Optional[str] = None,
    trained_artifact_legacy_layout: bool = False,
    cache_enabled: Optional[bool] = None,
    extra_config: Optional[Mapping[str, str]] = None,
    observability: Optional[ObservabilityProvenance] = None,
) -> RuntimeProvenance:
    """Capture the whole fastWorkflow-origin provenance record in one call.

    Intended to run once, at the point a harness begins a measured run, because
    `engine.source_hash` describes the module tree imported at that moment.

    `observability` is caller-supplied for the §22 reason given on that model: it
    is built by `evidence_run.capture_observability_provenance()`, which imports
    the store and therefore cannot be reached from here.
    """
    return RuntimeProvenance(
        engine=capture_engine_provenance(),
        workflow=capture_workflow_provenance(
            workflow_folderpath,
            command_surface_fingerprint=command_surface_fingerprint,
            trained_artifact_id=trained_artifact_id,
            trained_artifact_legacy_layout=trained_artifact_legacy_layout,
        ),
        models=capture_model_provenance(
            env, cache_enabled=cache_enabled, extra_config=extra_config
        ),
        observability=observability,
    )
