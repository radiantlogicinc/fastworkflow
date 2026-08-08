"""Versioned training artifacts with a `current` pointer (spec R4, decision D8).

Before this module, `fastworkflow train` wrote per-context model artifacts directly
into `<workflow>/___command_info/<Context>/`. There was no version, no rollback, and
nothing marking those directories as expensive: comparing two training runs meant
moving 8.6 GB by hand, and an unrelated scaffold-regeneration step once destroyed a
complete trained set. Versioning is also the precondition for paired evaluation —
scoring two artifact sets on identical cases — which is the acceptance mechanism for
the wildcard work (R7) and the substrate for the convergence loop (R8).

Layout
------
    <workflow>/___command_info/
        command_directory.json          <- workflow-scoped, NOT versioned
        routing_definition.json         <- workflow-scoped, NOT versioned
        <command>_param_labeled.json    <- workflow-scoped, NOT versioned
        current.json                    <- authoritative pointer (this module)
        versions/
            README.md                   <- "these cost hours to rebuild" warning
            <version_id>/
                manifest.json
                global/{tinymodel.pth, largemodel.pth, threshold.json, ...}
                <Context>/...
        current    -> versions/<version_id>              (symlink, best effort)
        <Context>  -> versions/<version_id>/<Context>    (compatibility entry)
        global     -> versions/<version_id>/global       (compatibility entry)

Only the **per-context model directories** belong to a version. The two JSON
snapshots are build artifacts guarded by `source_fingerprint` and are rewritten
whenever a command source's mtime changes — i.e. by merely importing a workflow, not
by training. Versioning something that a read can rewrite would manufacture versions
on import. `is_workflow_trained` also reads `routing_definition.json` to *enumerate*
the contexts it then checks for, so it must be resolvable before any version is.
`<command>_param_labeled.json` files are kilobytes and are read at runtime from the
top level by `utils/signatures.py`; they stay put.

Why per-context compatibility entries
-------------------------------------
Every existing reader builds `<workflow>/___command_info/<Context>/...` — including
`intent_detection.py:39`, which builds it as a literal f-string. The compatibility
entries mean those readers keep working byte-for-byte unchanged while the real bytes
live under a version. They point **directly** at `versions/<id>/<Context>` rather
than hopping through `current`, so deleting or losing the `current` symlink cannot
break every context at once, and `os.path.realpath` on any context entry names the
version in one hop.

The pointer file `current.json` is the authoritative record of which version is
current, so the current version is discoverable even where symlinks are unavailable
or were clobbered. `publish_version` writes it *last*, after preparing every reader
path, so it is the publication commit point: any earlier failure leaves the
authoritative pointer on the old version. During that preparation window,
`prune_versions` also protects versions referenced by compatibility entries or the
convenience link.

Promotion is also gated on the version being self-consistent:
`verify_version_consistency` re-derives the retraining closure recorded in
`manifest.json` from that manifest alone and refuses to advance `current` when it
contradicts itself or the version whose models were carried forward into it (AR3).
That check reads only files, deliberately — see its docstring for why a check sharing
inputs with the planner would not be a check.

R4 also asked for a human-facing display surface, and this module used to carry one:
`format_versions_table`, `describe_version` and the `human_size`/`human_duration`/
`human_age` formatters, written for a `versions` CLI that was cut before it shipped.
They were removed (bd fix-k0i.50) once it was clear nothing but their own tests had
ever called them; the stale comments pointing at that CLI went with them. If a
version listing is wanted again, `list_versions` returns everything it needs, and
`VersionInfo.size_bytes` is now walked on access (bd fix-44d) rather than eagerly
populated: as a field it made *every* `list_versions` call walk *every* version's
tree, including the call `retain_current_and_previous` makes on every train and now
makes while holding `publication_lock` — real I/O, under a lock, for a figure
nothing in the training path reads.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel

from fastworkflow.nlu_labels import WILDCARD_LABEL
from fastworkflow.utils.logging import logger

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

COMMAND_INFO_FOLDERNAME: str = "___command_info"
VERSIONS_DIRNAME: str = "versions"
CURRENT_LINK_NAME: str = "current"
CURRENT_POINTER_FILENAME: str = "current.json"
MANIFEST_FILENAME: str = "manifest.json"
VERSIONS_README_FILENAME: str = "README.md"

# Dropped inside a compatibility entry that had to be materialised as a real
# directory (hardlink farm or copy) because symlinks were unavailable. It is the only
# way to tell such an entry apart from a genuine legacy artifact directory, and
# therefore the only thing that makes it safe to replace on the next publish.
COMPAT_MARKER_FILENAME: str = ".fastworkflow_compat"

# Must match fastworkflow.model_pipeline_training.GLOBAL_CONTEXT_FOLDER exactly. It is
# duplicated rather than imported because that module pulls in torch/transformers and
# this one must stay cheap; `test_artifact_versioning.py` asserts the two agree.
GLOBAL_CONTEXT_FOLDER: str = "global"

# Must match fastworkflow.train.utterance_cache.CACHE_DIRNAME exactly. Duplicated for
# the same reason as GLOBAL_CONTEXT_FOLDER above — importing that module would pull
# `fastworkflow` and pydantic into this one, which stays cheap on purpose — and
# `test_utterance_cache.py` asserts the two agree.
UTTERANCE_CACHE_DIRNAME: str = "utterance_cache"

# Must match fastworkflow.train.param_example_cache.CACHE_DIRNAME exactly, and is
# duplicated for the same reason as UTTERANCE_CACHE_DIRNAME above;
# `test_param_example_cache.py` asserts the two agree. It holds the DSPy
# parameter-example draws (fix-czb), the second of the two LLM paths that a
# `TRAINING_SEED` cannot reach.
PARAM_EXAMPLE_CACHE_DIRNAME: str = "param_example_cache"

# Top-level names inside ___command_info that are never a context. Anything listed
# here is skipped by `publish_version`'s stale-entry sweep, by
# `migrate_legacy_to_version`, and by `_prune_stale_artifacts` in `train/__main__.py`
# — which is what exempts the generated-utterance cache from being pruned. It is not
# a context, it is not an artifact version, and it is the only thing that makes two
# training runs at the same seed train on the same data (R6).
RESERVED_TOPLEVEL_NAMES: frozenset[str] = frozenset(
    {
        VERSIONS_DIRNAME,
        CURRENT_LINK_NAME,
        UTTERANCE_CACHE_DIRNAME,
        PARAM_EXAMPLE_CACHE_DIRNAME,
        "__pycache__",
    }
)

# Files whose presence marks a directory as a (possibly partial) trained context.
# `threshold.json` alone is the marker used by `is_workflow_trained` and by
# `_prune_stale_artifacts`; migration deliberately uses a wider net so a
# half-written 276 MB context is moved rather than left behind to be mistaken for a
# legacy layout forever.
MODEL_ARTIFACT_MARKERS: frozenset[str] = frozenset(
    {
        "threshold.json",
        "tinymodel.pth",
        "largemodel.pth",
        "label_encoder.pkl",
        "tiny_ambiguous_threshold.json",
        "large_ambiguous_threshold.json",
    }
)

# Hardlinks make `carry_forward_context` free instead of copying 276 MB per context.
# The trade-off: a hardlinked file edited *in place* (`open(path, "w")` truncates the
# shared inode) would mutate every version that shares it. Training never does that —
# it writes into a fresh version directory via `save_pretrained` — but a human poking
# at `threshold.json` under a compatibility entry would. Flip this to False to force
# copies if that ever becomes a real workflow.
USE_HARDLINKS_FOR_CARRY_FORWARD: bool = True

_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_EXPENSIVE_ARTIFACT_WARNING = """\
# Trained intent-detection artifacts — EXPENSIVE TO REGENERATE

**Do not delete anything in this directory to "clean up".**

Each subdirectory here is one *version* of a workflow's trained intent-detection
models. Rebuilding a single version costs hours of LLM calls (synthetic utterance
generation) plus GPU/CPU fine-tuning time, and the utterances are not reproducible
byte-for-byte across runs. On a large workflow one version is roughly 276 MB per
context — several gigabytes in total.

A previous incident destroyed a complete trained set because nothing said so.

* `current.json` (one level up) records which version is live. It is authoritative.
* The `<Context>` entries one level up are compatibility links into the current
  version; every reader in the package resolves models through them.
* fastWorkflow retains the current version and one previous successful version.
  Older and incomplete versions are removed automatically after publication.

The previous version is an internal recovery point, not a user-managed history.
"""


class LegacyArtifactsPresentError(RuntimeError):
    """Raised when a real (unversioned) context directory blocks a publish.

    Removing it would destroy artifacts, which R4 forbids doing implicitly, so the
    caller is told to run `migrate_legacy_to_version` first.
    """


class ArtifactConsistencyError(RuntimeError):
    """Raised when a version's own records contradict each other at promotion time.

    Publication is refused rather than reported, because the alternative is shipping a
    version whose carried-forward models were trained against different data than the
    version claims — the failure R5's closure exists to prevent and that nothing at
    runtime would detect (spec F1/F10).
    """


class VersionInfo(BaseModel):
    """Summary of one artifact version, as surfaced to a developer.

    Every field is a manifest read, a shallow directory listing or the caller's own
    argument, so building one is cheap enough for the publication path to build one
    per version. `size_bytes` is the exception, and is a property for that reason.
    """

    version_id: str
    #: Carried so `size_bytes` can find the version's tree at the moment it is
    #: asked for it, rather than being handed a size nobody wanted.
    workflow_folderpath: str
    created_at: str
    is_current: bool
    contexts: list[str]
    seed: Optional[int] = None
    notes: Optional[str] = None
    # Recorded by the trainer when it knows. It drove the "this took 3h35m to build"
    # line in `format_versions_table` -- deleted as test-only code (bd fix-k0i.50) --
    # and is kept because it is the cheapest way to make the cost of these directories
    # obvious at the moment someone is about to delete one, whatever displays it next.
    train_duration_seconds: Optional[float] = None

    @property
    def size_bytes(self) -> int:
        """Apparent bytes under this version, walked on access (bd fix-44d).

        This was an eagerly populated field, and populating it meant `list_versions`
        walked *every* version's entire tree on every call. `retain_current_and_previous`
        calls `list_versions` on every train and now does so inside `publication_lock`,
        so a multi-gigabyte workflow spent real I/O — holding a lock another training
        process waits on — computing a figure nothing in the training path reads.

        Deliberately not cached. The callers are display surfaces, a version's tree
        does change under an interested reader (publishing hardlinks a carried-forward
        context into it), and a stale size is worse than a slow one; a caller that
        wants the figure twice should hold on to it. Returns 0 for a version that is
        no longer on disk, which is what walking an absent tree reports.

        Being a property rather than a field also keeps it out of `model_dump()`,
        which costs nothing today because nothing serialises a `VersionInfo`. A future
        surface that needs it serialised should reach for `computed_field` knowing it
        reintroduces the walk on every dump.
        """
        return _dir_size_bytes(version_dir(self.workflow_folderpath, self.version_id))


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


def command_info_root(workflow_folderpath: str) -> Path:
    """Return `<workflow>/___command_info` without creating it.

    Deliberately does not `mkdir`, unlike
    `CommandDirectory.get_commandinfo_folderpath`, so read-only inspection of an
    unbuilt workflow stays read-only.
    """
    return Path(workflow_folderpath) / COMMAND_INFO_FOLDERNAME


def versions_root(workflow_folderpath: str) -> Path:
    """Return `<workflow>/___command_info/versions` without creating it."""
    return command_info_root(workflow_folderpath) / VERSIONS_DIRNAME


def version_dir(workflow_folderpath: str, version_id: str) -> Path:
    """Return the directory holding *version_id*'s artifacts. Does not create it."""
    _validate_version_id(version_id)
    return versions_root(workflow_folderpath) / version_id


def context_folder_name(context_name: str) -> str:
    """Map a routing context name to its artifact folder name.

    The wildcard context `"*"` is not a legal directory name, so it maps to
    `GLOBAL_CONTEXT_FOLDER`, matching `model_pipeline_training.get_artifact_path`.
    """
    return GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name


def context_artifact_dir(
    workflow_folderpath: str, version_id: str, context_name: str
) -> Path:
    """Return (and create) the artifact directory for *context_name* in *version_id*.

    Creates the directory because this is the call `get_artifact_path` will make on
    the write path, and `get_artifact_path` has always created its directory.
    """
    target = version_dir(workflow_folderpath, version_id) / context_folder_name(
        context_name
    )
    target.mkdir(parents=True, exist_ok=True)
    return target


def pointer_path(workflow_folderpath: str) -> Path:
    """Return the path of the authoritative current-version pointer file."""
    return command_info_root(workflow_folderpath) / CURRENT_POINTER_FILENAME


def current_link_path(workflow_folderpath: str) -> Path:
    """Return the path of the convenience `current` symlink."""
    return command_info_root(workflow_folderpath) / CURRENT_LINK_NAME


def new_version_id() -> str:
    """Return a sortable, human-readable version id, e.g. `20260802T144233Z-a1b2c3`.

    Lexicographic order equals chronological order, which is what lets
    `prune_versions(keep=N)` and `list_versions` order by a plain string sort. The
    random suffix keeps two runs started in the same second distinct.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _validate_version_id(version_id: str) -> None:
    """Reject anything that could escape `versions/` or confuse the layout."""
    if not isinstance(version_id, str) or not _VERSION_ID_RE.match(version_id):
        raise ValueError(
            f"Invalid version id {version_id!r}: must match {_VERSION_ID_RE.pattern} "
            f"(no path separators, no leading dot)"
        )
    if version_id in RESERVED_TOPLEVEL_NAMES or ".." in version_id:
        raise ValueError(f"Reserved or unsafe version id: {version_id!r}")


# ---------------------------------------------------------------------
# Filesystem primitives (symlink with graceful degradation)
# ---------------------------------------------------------------------

_symlink_support: dict[str, bool] = {}


def _symlinks_supported(directory: Path) -> bool:
    """Probe once per directory whether we may create directory symlinks there.

    Cached because publishing touches this per context and the answer is a property
    of the filesystem, not of the call.
    """
    key = str(directory)
    if key in _symlink_support:
        return _symlink_support[key]

    probe = directory / f".symlink-probe-{uuid.uuid4().hex[:8]}"
    supported = False
    try:
        os.symlink(".", os.fspath(probe), target_is_directory=True)
        supported = True
    except (OSError, NotImplementedError, AttributeError):
        supported = False
    finally:
        _unlink_any(probe)

    _symlink_support[key] = supported
    if not supported:
        logger.warning(
            f"Symlinks unavailable under {directory}; artifact version compatibility "
            f"entries will be materialised as hardlink farms (or copies)."
        )
    return supported


def _unlink_any(path: Path) -> None:
    """Remove *path* whether it is a file, a symlink, or a directory symlink."""
    with contextlib.suppress(FileNotFoundError, OSError):
        os.unlink(os.fspath(path))
        return
    # Windows represents directory symlinks as directories for removal purposes.
    with contextlib.suppress(FileNotFoundError, OSError):
        os.rmdir(os.fspath(path))


def _atomic_replace_symlink(dest: Path, target: Path) -> None:
    """Point *dest* at *target* atomically, replacing an existing symlink.

    Uses a relative link target so the whole workflow directory stays relocatable
    (a `copytree` or a Docker `COPY` of the workflow must not leave dangling links
    into the build machine's filesystem).
    """
    relative = os.path.relpath(os.fspath(target), os.fspath(dest.parent))
    tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    _unlink_any(tmp)
    os.symlink(relative, os.fspath(tmp), target_is_directory=True)
    try:
        # os.replace over an existing *symlink* is atomic; over an existing real
        # directory it raises, which is why callers pre-check for that case.
        os.replace(os.fspath(tmp), os.fspath(dest))
    except OSError:
        _unlink_any(tmp)
        raise


def _link_tree(source: Path, dest: Path) -> None:
    """Recreate *source*'s tree at *dest*, hardlinking files where possible."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        target = dest / entry.name
        if entry.is_dir() and not entry.is_symlink():
            _link_tree(entry, target)
            continue
        try:
            os.link(os.fspath(entry), os.fspath(target))
        except OSError:
            shutil.copy2(entry, target, follow_symlinks=False)


def _materialize_compat_dir(dest: Path, source: Path) -> None:
    """Fallback for `dest -> source` where symlinks are unavailable.

    Builds a hardlink farm (no bytes copied on the same filesystem, falling back to
    a copy per file) in a temporary sibling and swaps it in. Not atomic, but the
    previous entry is only removed once the replacement is complete.
    """
    staging = dest.parent / f".{dest.name}.staging-{uuid.uuid4().hex[:8]}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    _link_tree(source, staging)
    (staging / COMPAT_MARKER_FILENAME).write_text(
        json.dumps({"source": str(source), "created_at": _utc_now()}, indent=2),
        encoding="utf-8",
    )

    retired = dest.parent / f".{dest.name}.retired-{uuid.uuid4().hex[:8]}"
    had_previous = dest.exists() or dest.is_symlink()
    if had_previous:
        os.replace(os.fspath(dest), os.fspath(retired))
    try:
        os.replace(os.fspath(staging), os.fspath(dest))
    except OSError:
        if had_previous:
            os.replace(os.fspath(retired), os.fspath(dest))
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if had_previous:
        shutil.rmtree(retired, ignore_errors=True)


def _is_compat_entry(path: Path) -> bool:
    """True if *path* is a compatibility entry this module created (and may replace)."""
    if path.is_symlink():
        return True
    return path.is_dir() and (path / COMPAT_MARKER_FILENAME).is_file()


def _point_compat_entry(dest: Path, target: Path) -> None:
    """Route *dest* at *target*, preferring a symlink and degrading to a link farm."""
    if _symlinks_supported(dest.parent):
        _atomic_replace_symlink(dest, target)
        return
    _materialize_compat_dir(dest, target)


def _remove_compat_entry(path: Path) -> bool:
    """Remove a compatibility entry. Never removes a real artifact directory."""
    if path.is_symlink():
        _unlink_any(path)
        return True
    if path.is_dir() and (path / COMPAT_MARKER_FILENAME).is_file():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def _utc_now() -> str:
    """UTC timestamp with microseconds.

    Microseconds, not seconds: `created_at` is the sort key `list_versions` (and
    therefore `prune_versions(keep=N)`) orders by, and two versions produced inside
    the same second must still order correctly — a version id only carries
    second resolution.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _coerce_int(value: object) -> Optional[int]:
    """Return *value* as an int when it plausibly is one, else None.

    Manifests are hand-editable JSON, so a seed can arrive as `7`, `"7"` or garbage.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


# ---------------------------------------------------------------------
# versions/ bookkeeping
# ---------------------------------------------------------------------


def ensure_versions_root(workflow_folderpath: str) -> Path:
    """Create `versions/` and its warning README, and return the path.

    The README is the R4 requirement that "the output must make it obvious these
    directories are expensive to regenerate" — expressed where a person cleaning up
    a disk will actually read it, not only in CLI output they may never run.
    """
    root = versions_root(workflow_folderpath)
    root.mkdir(parents=True, exist_ok=True)
    readme = root / VERSIONS_README_FILENAME
    if not readme.is_file():
        readme.write_text(_EXPENSIVE_ARTIFACT_WARNING, encoding="utf-8")
    return root


def version_context_names(workflow_folderpath: str, version_id: str) -> list[str]:
    """Return the context folder names present in *version_id*, sorted."""
    vdir = version_dir(workflow_folderpath, version_id)
    if not vdir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in vdir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _dir_size_bytes(path: Path) -> int:
    """Apparent size of *path*.

    Hardlinked carry-forwards are counted in every version that shares them, so the
    sum over versions overstates real disk usage. That bias is the safe direction:
    it never makes a version look cheaper than it is.

    Reached only through `VersionInfo.size_bytes`, i.e. only when something actually
    wants a size; that is what keeps this walk off the publication path (bd fix-44d).
    """
    total = 0
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(root, d))]
        for name in filenames:
            full = os.path.join(root, name)
            with contextlib.suppress(OSError):
                total += os.stat(full, follow_symlinks=False).st_size
    return total


# ---------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------


def write_manifest(workflow_folderpath: str, version_id: str, **fields) -> str:
    """Merge *fields* into *version_id*'s `manifest.json` and return its path.

    Merging (rather than overwriting) lets the trainer stamp what it knows as it
    goes: seed and notes up front, contexts and duration at the end. `version_id`
    and `created_at` are always present; `contexts` defaults to what is on disk.
    """
    _validate_version_id(version_id)
    vdir = version_dir(workflow_folderpath, version_id)
    vdir.mkdir(parents=True, exist_ok=True)
    ensure_versions_root(workflow_folderpath)

    manifest = read_manifest(workflow_folderpath, version_id)
    manifest.update(fields)
    manifest["version_id"] = version_id
    manifest.setdefault("created_at", _utc_now())
    manifest["updated_at"] = _utc_now()
    if not manifest.get("contexts"):
        manifest["contexts"] = version_context_names(workflow_folderpath, version_id)

    path = vdir / MANIFEST_FILENAME
    tmp = vdir / f".{MANIFEST_FILENAME}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    os.replace(os.fspath(tmp), os.fspath(path))
    return str(path)


def read_manifest(workflow_folderpath: str, version_id: str) -> dict:
    """Return *version_id*'s manifest, or `{}` when there is none / it is unreadable.

    An empty result therefore means "nothing recorded" OR "damaged", which is safe for
    the display and merge paths but NOT for anything that would destroy artifacts on the
    strength of a field being absent. `manifest_is_damaged` distinguishes the two, and
    `retain_current_and_previous` uses it before pruning.
    """
    _validate_version_id(version_id)
    path = version_dir(workflow_folderpath, version_id) / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            f"Unreadable manifest {path}: {exc}. Everything it recorded — seed, notes, "
            f"previous_version, build duration — is unavailable to every reader."
        )
        return {}
    if isinstance(data, dict):
        return data
    logger.error(
        f"Manifest {path} is valid JSON but not an object; treating it as unreadable."
    )
    return {}


def manifest_is_damaged(workflow_folderpath: str, version_id: str) -> bool:
    """True when *version_id* has a `manifest.json` that cannot be read as an object.

    Absence is deliberately NOT damage: a version whose manifest has not been written
    yet is a normal intermediate state (`carry_forward_context` creates the directory
    before the trainer stamps it). Only a manifest that exists and cannot be parsed
    counts, because that is the case where reading a missing field as "not set" turns
    damage into a decision.
    """
    _validate_version_id(version_id)
    path = version_dir(workflow_folderpath, version_id) / MANIFEST_FILENAME
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return not isinstance(data, dict)


# ---------------------------------------------------------------------
# Promotion-time consistency (spec AR3)
# ---------------------------------------------------------------------

#: Manifest key holding, per context, everything needed to re-derive that context's
#: share of R5's closure from the version alone: the ancestor set, the label space, the
#: wildcard-class sources, and the utterance-set fingerprints those two command lists
#: hash to. AR3 asks for the fingerprints specifically — recording wildcard sources as
#: command NAMES cannot see an ancestor command whose *content* changed, which is the
#: only thing that makes an ancestor-driven staleness visible at all.
CONTEXT_CONTRIBUTIONS_KEY: str = "context_contributions"

#: Manifest key mapping every command name to a digest of its training inputs. It is
#: the raw material `utterance_set_fingerprint` hashes over, and keeping it beside the
#: per-context digests is what lets the check recompute rather than take on trust.
COMMAND_UTTERANCE_FINGERPRINTS_KEY: str = "command_utterance_fingerprints"

#: Manifest key naming the version whose artifacts were carried forward into this one.
CARRY_FORWARD_FROM_KEY: str = "carry_forward_from"

CONTRIBUTION_FORMAT_KEY: str = "contribution_format_version"

#: Bump when the shape of the two keys above changes. A version recorded under an
#: unrecognised value is reported as unverifiable rather than checked against rules
#: that no longer describe it.
CONTRIBUTION_FORMAT_VERSION: int = 1

#: Recorded for a command whose training inputs could not be fingerprinted at all. It
#: is deliberately a fixed string rather than a fresh uuid: the digests below must be
#: reproducible from the manifest, and the "this can never prove cleanliness" property
#: is enforced by refusing to carry such a command's context forward, not by making the
#: value differ from itself.
UNRESOLVED_FINGERPRINT: str = "unresolved"

#: Substituted when a command named by a context is absent from the fingerprint map.
#: The absence is reported as a problem in its own right; this only keeps the digest
#: well defined so the report says which context, not merely that something is wrong.
MISSING_FINGERPRINT: str = "missing"

_CONTRIBUTION_LIST_FIELDS: tuple[str, ...] = (
    "ancestors",
    "label_space",
    "wildcard_sources",
)


class VersionConsistency(BaseModel):
    """The outcome of re-deriving one version's recorded closure from its own records.

    `problems` is promotion-blocking. `unverifiable_reasons` is not: it names what the
    version does not record well enough to check, which is a different claim and must
    never be silently rounded to "consistent".
    """

    version_id: str
    verified: bool = False
    unverifiable_reasons: list[str] = []
    problems: list[str] = []


def utterance_set_fingerprint(
    command_fingerprints: Mapping[str, str], command_names: Iterable[str]
) -> str:
    """Digest the training inputs *command_names* contribute, order-independently.

    Sorted and de-duplicated because a context's contribution is a set: the trainer
    pools these commands' utterances, so a reordering is not a change and must not read
    as one.
    """
    parts: list[str] = []
    for name in sorted(set(command_names)):
        value = command_fingerprints.get(name)
        if not isinstance(value, str) or not value:
            value = MISSING_FINGERPRINT
        parts.append(f"{name}\x00{value}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _contribution_block(manifest: dict) -> tuple[dict, dict]:
    """Return (`context_contributions`, `command_utterance_fingerprints`) as dicts."""
    contributions = manifest.get(CONTEXT_CONTRIBUTIONS_KEY)
    fingerprints = manifest.get(COMMAND_UTTERANCE_FINGERPRINTS_KEY)
    return (
        contributions if isinstance(contributions, dict) else {},
        fingerprints if isinstance(fingerprints, dict) else {},
    )


def _malformed_contribution_record(record: object) -> Optional[str]:
    """Describe why *record* cannot be read as a context contribution, or None."""
    if not isinstance(record, dict):
        return "is not a JSON object"
    for field in _CONTRIBUTION_LIST_FIELDS:
        value = record.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            return f"field {field!r} is not a list of command/context names"
    for field in ("own_utterances_sha256", "ancestor_utterances_sha256"):
        if not isinstance(record.get(field), str) or not record.get(field):
            return f"field {field!r} is not a recorded digest"
    return None


def _describe_fingerprint_drift(
    context_name: str,
    commands: Iterable[str],
    current: Mapping[str, str],
    previous: Mapping[str, str],
) -> str:
    """Name the commands whose recorded training inputs moved between two versions."""
    changed = sorted(
        name
        for name in set(commands)
        if current.get(name, MISSING_FINGERPRINT)
        != previous.get(name, MISSING_FINGERPRINT)
    )
    if not changed:
        return (
            f"the ancestor contribution digest of context {context_name!r} differs "
            f"from the version it was carried from, but no individual command's "
            f"fingerprint does — the recorded ancestor set itself moved"
        )
    shown = ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else "")
    return (
        f"context {context_name!r} was carried forward, but the training inputs of "
        f"{len(changed)} command(s) feeding its wildcard class changed since the "
        f"version it was carried from: {shown}"
    )


def verify_version_consistency(
    workflow_folderpath: str, version_id: str
) -> VersionConsistency:
    """Re-derive *version_id*'s recorded closure from *version_id*'s own records.

    This is AR3's promotion-time postcondition, and its whole value is that it does not
    share inputs with the thing it checks. The plan-time signature diff cannot serve as
    the check, because both sides of that diff come out of the same
    `build_context_maps` call as the closure itself: a deterministic bug in
    `commands()` or `get_ancestor_contexts` reproduces identically in the baseline and
    in the current signature, the diff comes out clean, and the bad version publishes.
    A check that shares its inputs with the thing it checks is not a check.

    So nothing below calls the planner, imports `selective_training`, or touches the
    routing registry. Everything it compares is already on disk, and it compares it
    three ways, each of which fails for a different reason:

    * against **laws that hold of any correct hierarchy** — ancestry is transitively
      closed, no context is its own ancestor, every named ancestor is a context this
      version describes, and a context's wildcard sources are exactly the union of its
      ancestors' label spaces minus the reserved escalation label. A closure that
      truncated an ancestor chain for one context but not for its parent contradicts
      the first of those, and re-running `get_ancestor_contexts` would not notice
      because it would return the same truncated chain;
    * against **the fingerprints actually present in this version** — each recorded
      per-context digest is recomputed from this version's own command fingerprints,
      so a manifest whose per-context digests and per-command digests disagree cannot
      be promoted;
    * against **a different run's records** — for every context carried forward, the
      digests recorded here must equal the digests recorded by the version the
      artifacts came from. Those bytes were trained under that version's ancestor
      contribution; if this version's differs, the closure failed to notice and the
      models about to ship are stale. This is the comparison the plan-time diff cannot
      make, because `wildcard_sources` holds command *names* and an ancestor command
      whose content changed keeps its name.

    Compatibility. A version published before this check existed records no
    contribution block, and refusing to promote it would strand every workflow trained
    by an earlier build — including the repair path, which republishes the version that
    is already current. Absence is therefore reported as *unverifiable* and named in
    `unverifiable_reasons` for the caller to log. It is never rounded up to "consistent".
    Damage is different and is a problem: a manifest that exists and cannot be parsed
    must not be read as "nothing was recorded", the same rule `retain_current_and_previous`
    applies before it prunes a recovery point.
    """
    _validate_version_id(version_id)
    result = VersionConsistency(version_id=version_id)

    if manifest_is_damaged(workflow_folderpath, version_id):
        result.problems.append(
            f"the manifest of version {version_id} exists but cannot be read as an "
            f"object, so nothing it recorded about the closure can be checked; a "
            f"damaged manifest must not be read as 'nothing was recorded'"
        )
        return result

    manifest = read_manifest(workflow_folderpath, version_id)
    if CONTEXT_CONTRIBUTIONS_KEY not in manifest:
        result.unverifiable_reasons.append(
            f"version {version_id} records no per-context contribution fingerprints "
            f"(it predates the promotion-time consistency check), so its closure "
            f"cannot be re-derived"
        )
        return result

    recorded_format = manifest.get(CONTRIBUTION_FORMAT_KEY)
    if recorded_format != CONTRIBUTION_FORMAT_VERSION:
        result.unverifiable_reasons.append(
            f"version {version_id} records contribution format {recorded_format!r}, "
            f"which this build does not understand (it writes "
            f"{CONTRIBUTION_FORMAT_VERSION})"
        )
        return result

    contributions, fingerprints = _contribution_block(manifest)
    if not contributions:
        result.problems.append(
            f"version {version_id} records a {CONTEXT_CONTRIBUTIONS_KEY} block that is "
            f"empty or not an object, so it claims to be checkable and is not"
        )
        return result

    carried = {
        name
        for name, record in contributions.items()
        if isinstance(record, dict) and record.get("carried_forward")
    }
    source_contributions, source_fingerprints = _resolve_carry_forward_records(
        workflow_folderpath, manifest, carried, result
    )

    for context_name in sorted(contributions):
        record = contributions[context_name]
        if malformed := _malformed_contribution_record(record):
            result.problems.append(
                f"the recorded contribution of context {context_name!r} in version "
                f"{version_id} {malformed}"
            )
            continue
        _verify_one_context(
            workflow_folderpath,
            version_id,
            context_name,
            record,
            contributions,
            fingerprints,
            source_contributions,
            source_fingerprints,
            result,
        )

    result.verified = not result.problems
    return result


def _resolve_carry_forward_records(
    workflow_folderpath: str,
    manifest: dict,
    carried: set[str],
    result: VersionConsistency,
) -> tuple[dict, dict]:
    """Load the records of the version this one carried artifacts forward from.

    Returns empty maps when there is nothing to compare against, having recorded why.
    A source version that is no longer on disk is *unverifiable* rather than a problem:
    the documented rollback procedure republishes an old version by id, and its source
    was very likely pruned years of runs ago. Refusing there would make rollback — the
    thing versioning exists for — impossible after two more training runs.
    """
    if not carried:
        return {}, {}

    source_id = manifest.get(CARRY_FORWARD_FROM_KEY)
    if not isinstance(source_id, str) or not source_id:
        result.problems.append(
            f"version {result.version_id} carries {len(carried)} context(s) forward "
            f"but names no source version, so what those models were trained against "
            f"cannot be established"
        )
        return {}, {}

    try:
        _validate_version_id(source_id)
    except ValueError as exc:
        result.problems.append(
            f"version {result.version_id} names an unusable carry-forward source "
            f"{source_id!r}: {exc}"
        )
        return {}, {}

    if not version_dir(workflow_folderpath, source_id).is_dir():
        result.unverifiable_reasons.append(
            f"the version {source_id} that {result.version_id} carried "
            f"{len(carried)} context(s) forward from is no longer on disk, so their "
            f"recorded contributions cannot be compared against it"
        )
        return {}, {}

    if manifest_is_damaged(workflow_folderpath, source_id):
        result.problems.append(
            f"the manifest of {source_id}, which {result.version_id} carried "
            f"{len(carried)} context(s) forward from, cannot be read as an object, so "
            f"what those carried models were trained against cannot be established"
        )
        return {}, {}

    source_manifest = read_manifest(workflow_folderpath, source_id)
    if source_manifest.get(CONTRIBUTION_FORMAT_KEY) != CONTRIBUTION_FORMAT_VERSION:
        result.unverifiable_reasons.append(
            f"version {source_id}, which {result.version_id} carried {len(carried)} "
            f"context(s) forward from, records no usable contribution block, so those "
            f"contexts cannot be checked against what their models were trained on"
        )
        return {}, {}
    return _contribution_block(source_manifest)


def _verify_one_context(
    workflow_folderpath: str,
    version_id: str,
    context_name: str,
    record: dict,
    contributions: dict,
    fingerprints: dict,
    source_contributions: dict,
    source_fingerprints: dict,
    result: VersionConsistency,
) -> None:
    """Apply every promotion-blocking rule to one recorded context contribution."""
    ancestors: list[str] = record["ancestors"]
    label_space: list[str] = record["label_space"]
    wildcard_sources: list[str] = record["wildcard_sources"]

    if context_name in ancestors:
        result.problems.append(
            f"context {context_name!r} is recorded as its own ancestor in version "
            f"{version_id}"
        )

    expected_sources: set[str] = set()
    for ancestor in ancestors:
        ancestor_record = contributions.get(ancestor)
        if _malformed_contribution_record(ancestor_record) is not None:
            result.problems.append(
                f"context {context_name!r} names ancestor {ancestor!r}, which version "
                f"{version_id} does not describe — that ancestor's commands would have "
                f"contributed nothing to {context_name!r}'s wildcard class and nothing "
                f"would have said so"
            )
            continue
        if missing := sorted(set(ancestor_record["ancestors"]) - set(ancestors)):
            result.problems.append(
                f"the recorded ancestry of context {context_name!r} is not transitively "
                f"closed: its ancestor {ancestor!r} lists {', '.join(missing)}, which "
                f"{context_name!r} does not"
            )
        expected_sources |= {
            command
            for command in ancestor_record["label_space"]
            if command.split("/")[-1] != WILDCARD_LABEL
        }

    if expected_sources != set(wildcard_sources):
        absent = sorted(expected_sources - set(wildcard_sources))
        extra = sorted(set(wildcard_sources) - expected_sources)
        result.problems.append(
            f"the recorded wildcard-class sources of context {context_name!r} are not "
            f"the union of its ancestors' label spaces"
            + (f"; missing {', '.join(absent)}" if absent else "")
            + (f"; unexpected {', '.join(extra)}" if extra else "")
        )

    expects = record.get("expects_wildcard_label")
    if isinstance(expects, bool) and expects != bool(
        set(wildcard_sources) - set(label_space)
    ):
        result.problems.append(
            f"context {context_name!r} records expects_wildcard_label={expects}, which "
            f"contradicts its own wildcard sources and label space"
        )

    named = set(label_space) | set(wildcard_sources)
    if orphans := sorted(named - set(fingerprints)):
        result.problems.append(
            f"context {context_name!r} names {len(orphans)} command(s) that version "
            f"{version_id} records no training-input fingerprint for: "
            f"{', '.join(orphans[:5])}{' ...' if len(orphans) > 5 else ''}"
        )

    for field, commands in (
        ("own_utterances_sha256", label_space),
        ("ancestor_utterances_sha256", wildcard_sources),
    ):
        recomputed = utterance_set_fingerprint(fingerprints, commands)
        if record[field] != recomputed:
            result.problems.append(
                f"the recorded {field} of context {context_name!r} disagrees with the "
                f"command fingerprints present in version {version_id}"
            )

    if not record.get("carried_forward"):
        return

    # Only from here down: rules that apply because these artifacts were NOT produced
    # by this run and therefore encode an earlier run's training data.
    if unresolved := sorted(
        command
        for command in named
        if fingerprints.get(command) == UNRESOLVED_FINGERPRINT
    ):
        result.problems.append(
            f"context {context_name!r} was carried forward, but the training inputs of "
            f"{len(unresolved)} command(s) it depends on could not be fingerprinted, so "
            f"nothing establishes that its models are still current: "
            f"{', '.join(unresolved[:5])}{' ...' if len(unresolved) > 5 else ''}"
        )

    folder = version_dir(workflow_folderpath, version_id) / context_folder_name(
        context_name
    )
    if not (folder / "threshold.json").exists():
        result.problems.append(
            f"context {context_name!r} is recorded as carried forward into version "
            f"{version_id} but has no model artifacts there; publishing would remove "
            f"its compatibility entry and un-train that part of the workflow"
        )

    source_record = source_contributions.get(context_name)
    if _malformed_contribution_record(source_record) is not None:
        if source_contributions:
            result.problems.append(
                f"context {context_name!r} was carried forward, but the version it came "
                f"from records no usable contribution for it, so what its models were "
                f"trained against cannot be established"
            )
        return

    if source_record["ancestor_utterances_sha256"] != record[
        "ancestor_utterances_sha256"
    ]:
        result.problems.append(
            _describe_fingerprint_drift(
                context_name,
                set(wildcard_sources) | set(source_record["wildcard_sources"]),
                fingerprints,
                source_fingerprints,
            )
        )
    if source_record["own_utterances_sha256"] != record["own_utterances_sha256"]:
        result.problems.append(
            f"context {context_name!r} was carried forward, but its own label space's "
            f"training inputs differ from the version it was carried from"
        )


# ---------------------------------------------------------------------
# Cross-process publication lock
# ---------------------------------------------------------------------

#: Lives inside `___command_info` rather than `versions/` because it guards the
#: workflow-scoped pointer and compatibility entries, not one version's bytes. The
#: leading dot keeps it out of every entry sweep in this module and in
#: `train.__main__._prune_stale_artifacts`, all of which skip dotted names.
PUBLICATION_LOCK_FILENAME: str = ".publication.lock"

#: Publish plus prune is seconds of work on directory entries. A wait this long means
#: another process is wedged rather than busy, and failing loudly beats interleaving
#: pointer writes with someone else's prune.
PUBLICATION_LOCK_TIMEOUT_SECONDS: float = 300.0

_PUBLICATION_LOCK_POLL_SECONDS: float = 0.1

# resolved ___command_info path -> [file descriptor, reentrancy depth]
_publication_lock_state: dict[str, list] = {}
_publication_lock_bookkeeping = threading.RLock()
_fcntl_unavailable_warned = False


class PublicationLockTimeout(RuntimeError):
    """Raised when another process held the publication lock past the timeout."""


@contextlib.contextmanager
def publication_lock(
    workflow_folderpath: str,
    timeout: float = PUBLICATION_LOCK_TIMEOUT_SECONDS,
):
    """Serialise publish-and-prune against other processes on the same workflow.

    Two concurrent `fastworkflow train` runs on one workflow interleave compatibility
    entry swaps, `current.json` writes and `retain_current_and_previous` prunes. The
    damaging interleaving is one process pruning the version another is mid-`_link_tree`
    carrying forward from: the reader path survives, the recovery point does not.
    Publication is presented as transactional, so the transaction needs a boundary wider
    than one process.

    Why this cannot deadlock:

    * There is exactly one lock, so no lock-ordering cycle exists to enter.
    * It is reentrant within a process, so nesting (`retain_current_and_previous` called
      inside an orchestrator that already holds it) is a counter bump, not a wait.
    * `flock` is owned by the open file description, so the kernel releases it when the
      descriptor closes or the process dies. A crash cannot strand it, which is why this
      needs no stale-lock timeout of its own.
    * Acquisition is bounded by *timeout*; the worst case is a named failure, not a hang.

    Scope is deliberately cross-process only. Two threads in one process publishing the
    same workflow concurrently were never serialised and are not serialised here — the
    nested acquisition succeeds for either of them. Where `fcntl` is unavailable
    (Windows) this degrades to no locking with one warning, because refusing to train at
    all on that platform would be a worse trade than the single-developer race it guards.
    """
    info = command_info_root(workflow_folderpath)
    info.mkdir(parents=True, exist_ok=True)
    key = str(info.resolve())

    with _publication_lock_bookkeeping:
        nested = key in _publication_lock_state
        if nested:
            _publication_lock_state[key][1] += 1
    if nested:
        try:
            yield
        finally:
            with _publication_lock_bookkeeping:
                if state := _publication_lock_state.get(key):
                    state[1] -= 1
        return

    if fcntl is None:
        global _fcntl_unavailable_warned
        if not _fcntl_unavailable_warned:
            _fcntl_unavailable_warned = True
            logger.warning(
                "fcntl is unavailable on this platform; artifact publication is not "
                "guarded against a concurrent `fastworkflow train` on the same workflow. "
                "Run one training process per workflow at a time."
            )
        yield
        return

    lock_path = info / PUBLICATION_LOCK_FILENAME
    fd = os.open(os.fspath(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + max(timeout, 0.0)
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise PublicationLockTimeout(
                        f"Another process has held the artifact publication lock "
                        f"{lock_path} for more than {timeout:g}s. Concurrent training "
                        f"runs on one workflow would interleave pointer writes and prune "
                        f"each other's recovery point, so this run is stopping instead. "
                        f"Its version directory is intact under "
                        f"{versions_root(workflow_folderpath)}."
                    )
                time.sleep(_PUBLICATION_LOCK_POLL_SECONDS)
        with _publication_lock_bookkeeping:
            _publication_lock_state[key] = [fd, 1]
        yield
    finally:
        with _publication_lock_bookkeeping:
            _publication_lock_state.pop(key, None)
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------
# current pointer
# ---------------------------------------------------------------------


def _write_pointer(workflow_folderpath: str, version_id: str, layout: str) -> None:
    path = pointer_path(workflow_folderpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version_id": version_id,
        "updated_at": _utc_now(),
        "layout": layout,
        "warning": (
            "Trained artifacts under versions/ cost hours of LLM and training time "
            "to rebuild. Do not delete them to reclaim disk space."
        ),
    }
    tmp = path.parent / f".{CURRENT_POINTER_FILENAME}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(os.fspath(tmp), os.fspath(path))


def resolve_current_version(workflow_folderpath: str) -> Optional[str]:
    """Return the current version id, or None when the workflow has no version.

    Consults the pointer file first (it is the authoritative record and survives a
    filesystem that lost or never supported symlinks), then the `current` symlink,
    then gives up. A pointer naming a version that is no longer on disk is treated
    as absent rather than trusted.
    """
    pointer = pointer_path(workflow_folderpath)
    if pointer.is_file():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            candidate = data.get("version_id")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Unreadable current pointer {pointer}: {exc}")
            candidate = None
        if isinstance(candidate, str) and candidate:
            with contextlib.suppress(ValueError):
                if version_dir(workflow_folderpath, candidate).is_dir():
                    return candidate
                logger.warning(
                    f"current.json names version {candidate!r} which is not on disk "
                    f"under {versions_root(workflow_folderpath)}"
                )

    link = current_link_path(workflow_folderpath)
    if link.is_symlink() or link.exists():
        with contextlib.suppress(OSError):
            resolved = Path(os.path.realpath(os.fspath(link)))
            candidate = resolved.name
            if resolved.is_dir() and resolved.parent.name == VERSIONS_DIRNAME:
                with contextlib.suppress(ValueError):
                    if version_dir(workflow_folderpath, candidate).is_dir():
                        return candidate
    return None


def _routed_version_ids(workflow_folderpath: str) -> set[str]:
    """Return versions referenced by compatibility entries or the convenience link.

    Publication prepares those reader paths before committing `current.json`. This
    scan preserves prune safety during that mixed-state window without turning the
    compatibility entries into another authoritative current-version record.
    """
    info = command_info_root(workflow_folderpath)
    root = versions_root(workflow_folderpath)
    if not info.is_dir() or not root.is_dir():
        return set()

    resolved_root = root.resolve()
    routed: set[str] = set()
    for entry in info.iterdir():
        target: Optional[Path] = None
        if entry.is_symlink():
            with contextlib.suppress(OSError):
                target = Path(os.path.realpath(os.fspath(entry)))
        elif entry.is_dir() and (entry / COMPAT_MARKER_FILENAME).is_file():
            marker = entry / COMPAT_MARKER_FILENAME
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
                source = marker_data.get("source")
                if isinstance(source, str) and source:
                    target = Path(source).resolve()
            except (OSError, json.JSONDecodeError):
                target = None

        if target is None:
            continue
        with contextlib.suppress(ValueError):
            relative = target.resolve().relative_to(resolved_root)
            if not relative.parts:
                continue
            candidate = relative.parts[0]
            _validate_version_id(candidate)
            if version_dir(workflow_folderpath, candidate).is_dir():
                routed.add(candidate)
    return routed


def _require_consistent_version(workflow_folderpath: str, version_id: str) -> None:
    """Refuse to promote *version_id* when its own records contradict each other."""
    consistency = verify_version_consistency(workflow_folderpath, version_id)
    if consistency.problems:
        raise ArtifactConsistencyError(
            f"Refusing to make artifact version {version_id} current: its own records "
            f"contradict each other, so at least one carried-forward context's models "
            f"were not trained on the data this version describes.\n  - "
            + "\n  - ".join(consistency.problems)
            + f"\nThe previously current version is untouched and still complete. "
            f"Re-run training with --regenerate-utterances, or publish an earlier "
            f"version from {versions_root(workflow_folderpath)}."
        )
    for reason in consistency.unverifiable_reasons:
        logger.warning(
            f"Publishing artifact version {version_id} without the promotion-time "
            f"closure check: {reason}."
        )


def publish_version(workflow_folderpath: str, version_id: str) -> None:
    """Make *version_id* the current version, rewiring every reader path to it.

    Order matters. The per-context compatibility entries are prepared first (each
    one an atomic symlink swap), then the convenience `current` symlink, then
    compatibility entries for contexts this version no longer has are removed.
    The pointer file is written *last* as the publication commit point, so any
    earlier failure leaves `current.json` naming the old version. `prune_versions`
    protects the prepared reader targets during this pre-commit window.

    Idempotent: safe to call repeatedly with the same version id.

    Raises `LegacyArtifactsPresentError` if a real, unversioned context directory
    still occupies a name we must route. Deleting it would destroy artifacts, which
    R4 forbids doing implicitly — run `migrate_legacy_to_version` first.

    Raises `ArtifactConsistencyError` when `verify_version_consistency` finds that the
    version's own records contradict each other (AR3). That check gates *advancing*
    `current` only. Re-publishing the version that is already current repairs reader
    paths for artifacts that are live either way, and `_repair_noop_publication` does
    exactly that on a workflow whose manifest may be damaged; refusing there would turn
    a lost manifest into an unroutable workflow instead of a blocked promotion.

    Not itself locked: publication and the retention prune that follows it are one
    critical section, so the boundary belongs to the caller holding `publication_lock`
    across both. `train.__main__` does that on every path that publishes.
    """
    _validate_version_id(version_id)
    vdir = version_dir(workflow_folderpath, version_id)
    if not vdir.is_dir():
        raise FileNotFoundError(f"Artifact version directory does not exist: {vdir}")

    if resolve_current_version(workflow_folderpath) != version_id:
        _require_consistent_version(workflow_folderpath, version_id)

    info = command_info_root(workflow_folderpath)
    info.mkdir(parents=True, exist_ok=True)
    ensure_versions_root(workflow_folderpath)

    contexts = version_context_names(workflow_folderpath, version_id)

    blocked = [
        name
        for name in contexts
        if (info / name).is_dir() and not _is_compat_entry(info / name)
    ]
    if blocked:
        raise LegacyArtifactsPresentError(
            f"Unversioned artifact directories block publishing {version_id}: "
            f"{', '.join(blocked)} under {info}. Run "
            f"migrate_legacy_to_version() first — they are not deleted implicitly."
        )

    layout = "symlink" if _symlinks_supported(info) else "hardlink"

    for name in contexts:
        _point_compat_entry(info / name, vdir / name)

    # The `current` entry is a convenience for humans, `du`, and diagnostics. Where
    # symlinks are unavailable we skip it rather than duplicate an entire version's
    # tree for a shortcut; current.json already answers the question authoritatively.
    if _symlinks_supported(info):
        _atomic_replace_symlink(current_link_path(workflow_folderpath), vdir)

    known = set(contexts)
    for entry in sorted(info.iterdir()):
        if entry.name in RESERVED_TOPLEVEL_NAMES or entry.name in known:
            continue
        if entry.name.startswith("."):
            continue
        if _is_compat_entry(entry) and _remove_compat_entry(entry):
            logger.info(
                f"Removed stale artifact compatibility entry {entry.name} "
                f"(absent from version {version_id})"
            )

    _write_pointer(workflow_folderpath, version_id, layout)


# ---------------------------------------------------------------------
# Listing and pruning
# ---------------------------------------------------------------------


def list_versions(workflow_folderpath: str) -> list[VersionInfo]:
    """Return every version, newest first.

    Ordered by the manifest's `created_at` (microsecond resolution), with the
    version id as a tiebreaker. Sorting on the id alone would be wrong for two
    versions produced inside the same second, because the id's random suffix would
    decide the order — and `prune_versions(keep=N)` depends on this order to pick
    which versions are the old ones.

    Costs one manifest read and one shallow listing per version, and never walks a
    version's tree: `prune_versions` and `retain_current_and_previous` both call this
    on the publication path, and neither of them wants a size (bd fix-44d).
    """
    root = versions_root(workflow_folderpath)
    if not root.is_dir():
        return []

    current = resolve_current_version(workflow_folderpath)
    infos: list[VersionInfo] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.is_symlink() or entry.name.startswith("."):
            continue
        try:
            _validate_version_id(entry.name)
        except ValueError:
            continue
        manifest = read_manifest(workflow_folderpath, entry.name)
        contexts = manifest.get("contexts") or version_context_names(
            workflow_folderpath, entry.name
        )
        created_at = manifest.get("created_at")
        if not created_at:
            created_at = datetime.fromtimestamp(
                entry.stat().st_mtime, timezone.utc
            ).isoformat(timespec="microseconds")
        duration = manifest.get("train_duration_seconds")
        infos.append(
            VersionInfo(
                version_id=entry.name,
                workflow_folderpath=workflow_folderpath,
                created_at=str(created_at),
                is_current=(entry.name == current),
                contexts=[str(c) for c in contexts],
                seed=_coerce_int(manifest.get("seed")),
                notes=manifest.get("notes"),
                train_duration_seconds=(
                    float(duration) if isinstance(duration, (int, float)) else None
                ),
            )
        )
    infos.sort(key=lambda info: (info.created_at, info.version_id), reverse=True)
    return infos


def prune_versions(
    workflow_folderpath: str,
    keep: Optional[int] = None,
    version_ids: Optional[list[str]] = None,
    dry_run: bool = True,
) -> list[str]:
    """Remove artifact versions, but only when explicitly asked.

    Exactly one of *keep* (retain the N newest) or *version_ids* (remove exactly
    these) must be given; calling with neither raises rather than guessing, because
    R4's rule is that a previous version is never destroyed implicitly. `dry_run`
    defaults to True and returns what *would* be removed.

    The current version is never removed: it is filtered out of a *keep* window and
    an explicit request to delete it raises `ValueError`. Versions referenced by
    prepared compatibility entries or the convenience link are protected as well,
    preserving safety while publication has not yet flipped `current.json`.
    """
    if (keep is None) == (version_ids is None):
        raise ValueError(
            "prune_versions requires exactly one of keep= or version_ids=; refusing "
            "to guess what should be deleted"
        )

    existing = [info.version_id for info in list_versions(workflow_folderpath)]
    current = resolve_current_version(workflow_folderpath)
    protected = _routed_version_ids(workflow_folderpath)
    if current is not None:
        protected.add(current)

    if version_ids is not None:
        requested = list(dict.fromkeys(version_ids))
        for vid in requested:
            _validate_version_id(vid)
        if routed := [vid for vid in requested if vid in protected]:
            raise ValueError(
                "Refusing to prune artifact version(s) currently referenced by "
                f"current.json or reader paths: {', '.join(sorted(routed))}. "
                "Publish another version first."
            )
        if unknown := [vid for vid in requested if vid not in existing]:
            raise ValueError(
                f"Unknown artifact version(s): {', '.join(sorted(unknown))}"
            )
        doomed = requested
    else:
        if keep is None or keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep!r}")
        # `existing` is newest-first, so everything past the window is older.
        doomed = [vid for vid in existing[keep:] if vid not in protected]

    if dry_run:
        return doomed

    removed: list[str] = []
    for vid in doomed:
        target = version_dir(workflow_folderpath, vid)
        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.error(f"Failed to prune artifact version {vid}: {exc}")
            continue
        removed.append(vid)
        logger.info(f"Pruned artifact version {vid} ({target})")
    return removed


def retain_current_and_previous(
    workflow_folderpath: str,
    previous_version: Optional[str],
) -> list[str]:
    """Keep only the current version and its previous successful version.

    Training uses immutable version directories so publication can be atomic and a
    failed run cannot overwrite the working model. That safety property needs a staging
    directory and one recovery point, not an ever-growing user-managed history.

    Refuses to prune anything when the current version's manifest is damaged. Callers
    derive *previous_version* from that manifest, and a damaged manifest yields None —
    indistinguishable from "there is no previous version", which would delete the only
    recovery point in response to a corrupt JSON file. R4 forbids destroying a version
    implicitly, and acting on a field that could not be read is exactly that.

    Holds `publication_lock` for the same reason: this is the destructive half of
    publication, and it must not run against a version another process is publishing or
    carrying forward from. The acquisition is free when the caller already holds it.
    """
    with publication_lock(workflow_folderpath):
        current = resolve_current_version(workflow_folderpath)
        if current is not None and manifest_is_damaged(workflow_folderpath, current):
            logger.error(
                f"Skipping artifact retention for {workflow_folderpath}: the manifest of "
                f"current version {current} is unreadable, so its previous_version cannot "
                f"be distinguished from absent and pruning could destroy the only "
                f"recovery point. Repair or remove "
                f"{version_dir(workflow_folderpath, current) / MANIFEST_FILENAME} to "
                f"re-enable retention; older versions are retained until then."
            )
            return []
        retained = {
            version_id for version_id in (current, previous_version) if version_id
        }
        doomed = [
            info.version_id
            for info in list_versions(workflow_folderpath)
            if info.version_id not in retained
        ]
        if not doomed:
            return []
        return prune_versions(
            workflow_folderpath,
            version_ids=doomed,
            dry_run=False,
        )


# ---------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------


def _legacy_context_dirs(workflow_folderpath: str) -> list[Path]:
    """Real (non-link) top-level directories holding model artifacts."""
    info = command_info_root(workflow_folderpath)
    if not info.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(info.iterdir()):
        if entry.name in RESERVED_TOPLEVEL_NAMES or entry.name.startswith("."):
            continue
        if not entry.is_dir() or entry.is_symlink():
            continue
        if (entry / COMPAT_MARKER_FILENAME).is_file():
            continue
        if any((entry / marker).exists() for marker in MODEL_ARTIFACT_MARKERS):
            found.append(entry)
    return found


def legacy_layout_in_use(workflow_folderpath: str) -> bool:
    """True when unversioned per-context artifacts sit directly in `___command_info`.

    A compatibility symlink or hardlink farm does not count: those *are* the
    versioned layout as seen by an old reader.
    """
    return bool(_legacy_context_dirs(workflow_folderpath))


def migrate_legacy_to_version(
    workflow_folderpath: str, version_id: Optional[str] = None
) -> Optional[str]:
    """Move an unversioned artifact tree into `versions/<id>/` and publish it.

    Returns the version id it created, or None when there was nothing to migrate —
    which makes it idempotent and therefore safe to call unconditionally at the
    start of a train run or of any read-only inspection. Artifacts are `shutil.move`d,
    never copied and never deleted, so a migration cannot lose a 276 MB context.
    """
    legacy = _legacy_context_dirs(workflow_folderpath)
    if not legacy:
        return None

    version_id = version_id or new_version_id()
    _validate_version_id(version_id)
    ensure_versions_root(workflow_folderpath)
    vdir = version_dir(workflow_folderpath, version_id)
    vdir.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for source in legacy:
        destination = vdir / source.name
        if destination.exists():
            # A partially completed earlier migration into this same id. Leave the
            # already-migrated copy alone rather than overwrite it.
            logger.warning(
                f"Artifact version {version_id} already contains {source.name}; "
                f"leaving {source} in place for manual review"
            )
            continue
        shutil.move(os.fspath(source), os.fspath(destination))
        moved.append(source.name)

    write_manifest(
        workflow_folderpath,
        version_id,
        migrated_from="unversioned ___command_info layout",
        notes="Migrated from the pre-versioning layout; provenance unknown.",
        contexts=sorted(moved)
        or version_context_names(workflow_folderpath, version_id),
    )
    publish_version(workflow_folderpath, version_id)
    logger.info(
        f"Migrated unversioned artifacts ({', '.join(moved)}) into version {version_id}"
    )
    return version_id


# ---------------------------------------------------------------------
# Unrouting (used by the trainer's stale-artifact prune)
# ---------------------------------------------------------------------


def unroute_context(workflow_folderpath: str, context_folder: str) -> bool:
    """Remove the compatibility entry for *context_folder*, leaving version bytes intact.

    Returns True when an entry was removed, False when the path is absent or is a real
    artifact directory rather than a compatibility entry.

    This is what `_prune_stale_artifacts` should call instead of `shutil.rmtree` once
    versioning is on. `rmtree` on a symlink raises rather than deleting through it, so the
    prune would silently stop cleaning orphans and leave dangling entries pointing at
    contexts that no longer exist. Unrouting removes the pointer only: the version's
    artifacts stay recoverable by republishing it, which is the entire point of R4.
    """
    entry = command_info_root(workflow_folderpath) / context_folder
    if entry.name in RESERVED_TOPLEVEL_NAMES:
        return False
    return _remove_compat_entry(entry) if _is_compat_entry(entry) else False


# ---------------------------------------------------------------------
# Carry-forward (makes selective training affordable)
# ---------------------------------------------------------------------


def carry_forward_context(
    workflow_folderpath: str,
    from_version: str,
    to_version: str,
    context_name: str,
) -> bool:
    """Reuse an untouched context's artifacts from *from_version* in *to_version*.

    Hardlinks each file where the filesystem allows it, so carrying a context
    forward costs inodes rather than the 276 MB a copy would, and falls back to a
    per-file copy otherwise. Returns False when the source context is absent.

    Idempotent: a destination that already has content is left untouched and True is
    returned, so a resumed training run does not re-link work it already did.
    """
    _validate_version_id(from_version)
    _validate_version_id(to_version)
    folder = context_folder_name(context_name)

    source = version_dir(workflow_folderpath, from_version) / folder
    if not source.is_dir():
        logger.warning(
            f"Cannot carry forward context {context_name!r}: {source} does not exist"
        )
        return False

    destination = version_dir(workflow_folderpath, to_version) / folder
    if destination.is_dir() and any(destination.iterdir()):
        logger.info(
            f"Context {context_name!r} already present in version {to_version}; "
            f"carry-forward is a no-op"
        )
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    if USE_HARDLINKS_FOR_CARRY_FORWARD:
        _link_tree(source, destination)
    else:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return True
