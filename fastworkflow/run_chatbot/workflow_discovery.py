"""Workflow discovery for the chatbot picker.

Moved verbatim from ``run_chatbot.server``. The candidate-directory walk
cache lives only in this module.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from fastworkflow.run_chatbot import launcher


def _looks_like_workflow(path: str) -> bool:
    """A fastWorkflow workflow dir: authored commands or trained artifacts."""
    return os.path.isdir(os.path.join(path, "_commands")) or os.path.isdir(
        os.path.join(path, "___command_info")
    )


# Mirrors model_pipeline_training.GLOBAL_CONTEXT_FOLDER without importing
# that module — it pulls in torch/transformers, which the chatbot must not.
_GLOBAL_CONTEXT_FOLDER = "global"
_CME_CONTEXT_NAMES: Optional[set[str]] = None


def _cme_context_names() -> set[str]:
    """Internal command_metadata_extraction context names (cached).

    App workflow ``routing_definition.json`` lists these too; they are trained
    in the CME workflow, not per app, so they must not count as missing.
    """
    global _CME_CONTEXT_NAMES
    if _CME_CONTEXT_NAMES is not None:
        return _CME_CONTEXT_NAMES
    names: set[str] = set()
    try:
        import fastworkflow

        internal = fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        )
    except Exception:
        _CME_CONTEXT_NAMES = names
        return names
    json_path = os.path.join(internal, "command_context_model.json")
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            names.update(str(key) for key in data)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    commands = os.path.join(internal, "_commands")
    try:
        for entry in os.listdir(commands):
            full = os.path.join(commands, entry)
            if (
                os.path.isdir(full)
                and not entry.startswith(".")
                and entry != "__pycache__"
            ):
                names.add(entry)
    except OSError:
        pass
    _CME_CONTEXT_NAMES = names
    return names


def _workflow_is_trained(path: str) -> bool:
    """Filesystem check matching ``is_workflow_trained`` without importing torch.

    ``___command_info`` appearing is not enough: train writes that directory
    immediately, before any ``threshold.json`` exists.
    """
    command_info_root = os.path.join(path, "___command_info")
    routing_def_path = os.path.join(command_info_root, "routing_definition.json")
    if not os.path.isfile(routing_def_path):
        return False
    try:
        with open(routing_def_path, encoding="utf-8") as handle:
            routing_definition = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    contexts = routing_definition.get("contexts") or {}
    if not isinstance(contexts, dict) or not contexts:
        return False
    contexts_to_check = (set(contexts) - _cme_context_names()) | {"*"}
    for context_name in contexts_to_check:
        folder = _GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        threshold_path = os.path.join(command_info_root, folder, "threshold.json")
        if not os.path.isfile(threshold_path):
            return False
    return True


def _rel_under(path: str, root: str) -> str:
    """Path relative to ``root``, posix slashes, or ``""`` if outside ``root``."""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        return ""
    if rel == ".":
        return ""
    if rel == ".." or rel.startswith(".." + os.sep):
        return ""
    return rel.replace("\\", "/")


def _workflow_entry(path: str, source: str, rel: str = "") -> dict[str, Any]:
    path = os.path.abspath(path)
    if launcher.is_bundled_example_path(path):
        source = "examples"
    trained = _workflow_is_trained(path)
    training = launcher.is_train_running(path)
    bundled = source == "examples" or launcher.is_bundled_example_path(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "rel": rel or os.path.basename(path),
        "trained": trained,
        "training": training,
        "source": source,
        "trainable": (not bundled) and (not trained) and (not training),
    }


_SKIP_DIR_NAMES = {
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "venv",
    "_commands",
    "___command_info",
    "___workflow_contexts",
    "___convo_info",
}

# Nested project layouts (apps/team/workflow) sit deeper than the old
# two-level scan; five is enough to find them without walking the world.
_MAX_WF_SCAN_DEPTH = 5
_MAX_WF_CANDIDATES = 100
# Directory walk is the expensive part; trained/training flags are cheap and
# must stay fresh on every poll while a train is running.
_WF_WALK_TTL_S = 30.0
_wf_walk_cache: dict[str, tuple[float, list[tuple[str, str, str]]]] = {}


def invalidate_workflow_candidate_walk_cache() -> None:
    """Drop the cached candidate-directory walk (tests / browse after mkdir)."""
    _wf_walk_cache.clear()


def _walk_workflow_candidate_paths(cwd: str) -> list[tuple[str, str, str]]:
    """Return ``(abspath, source, rel)`` rows for workflow dirs under ``cwd``.

    Does not probe trained/training state — callers refresh those per request.
    """
    seen: dict[str, tuple[str, str]] = {}

    def add(path: str, source: str, rel: str) -> None:
        path = os.path.abspath(path)
        if path not in seen and _looks_like_workflow(path):
            seen[path] = (source, rel)

    add(cwd, "local", _rel_under(cwd, cwd))

    def walk(current: str, depth: int) -> None:
        if depth > _MAX_WF_SCAN_DEPTH or len(seen) >= _MAX_WF_CANDIDATES:
            return
        try:
            names = sorted(os.listdir(current))
        except OSError:
            return
        for name in names:
            if len(seen) >= _MAX_WF_CANDIDATES:
                return
            if name.startswith(".") or name in _SKIP_DIR_NAMES:
                continue
            full = os.path.join(current, name)
            if not os.path.isdir(full):
                continue
            rel = _rel_under(full, cwd)
            if _looks_like_workflow(full):
                add(full, "local", rel)
            walk(full, depth + 1)

    walk(cwd, 1)
    try:
        import fastworkflow

        examples = os.path.join(
            os.path.dirname(os.path.abspath(fastworkflow.__file__)), "examples"
        )
        for entry in sorted(os.listdir(examples)):
            full = os.path.join(examples, entry)
            rel = _rel_under(full, cwd)
            if not rel:
                rel = "Bundled examples/" + entry
            add(full, "examples", rel)
    except Exception:
        pass
    # A directory that both looks like a workflow and contains other workflows
    # (the library package has _commands/ plus examples/) is a folder, not a
    # leaf the developer would pick.
    for path in list(seen):
        if any(other != path and other.startswith(path + os.sep) for other in seen):
            seen.pop(path, None)
    return [(path, source, rel) for path, (source, rel) in seen.items()]


def list_workflow_candidates() -> list[dict[str, Any]]:
    """Workflow dirs the developer most likely wants: the bundled examples,
    plus a bounded nested scan below the launch directory.

    Each entry carries ``rel`` (path relative to the launch directory, or a
    ``Bundled examples/`` prefix when the workflow lives outside it) so the
    picker can group them under folders instead of a flat list.

    The directory walk is cached briefly per cwd; trained/training flags are
    recomputed on every call so a poll during training stays accurate.
    """
    cwd = os.getcwd()
    now = time.monotonic()
    cached = _wf_walk_cache.get(cwd)
    if cached is not None and cached[0] > now:
        paths = cached[1]
    else:
        paths = _walk_workflow_candidate_paths(cwd)
        _wf_walk_cache[cwd] = (now + _WF_WALK_TTL_S, paths)
    candidates = [
        _workflow_entry(path, source, rel) for path, source, rel in paths
    ]
    candidates.sort(
        key=lambda w: (w["source"] != "local", not w["trained"], w["name"].lower())
    )
    return candidates[:_MAX_WF_CANDIDATES]


def browse_directories(dir_path: str) -> dict[str, Any]:
    """One level of the local filesystem for the workflow picker: directories
    only, never file contents; each entry flagged when it is a workflow.
    """
    base = os.path.abspath(dir_path or os.getcwd())
    if not os.path.isdir(base):
        return {"error": f"not a directory: {base}"}
    entries = []
    try:
        names = sorted(os.listdir(base))
    except OSError as exc:
        return {"error": f"cannot list {base}: {exc}"}
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        is_workflow = _looks_like_workflow(full)
        entry = {
            "name": name,
            "path": full,
            "is_workflow": is_workflow,
            "trained": _workflow_is_trained(full) if is_workflow else False,
        }
        if is_workflow:
            entry["training"] = launcher.is_train_running(full)
        entries.append(entry)
        if len(entries) >= 300:
            break
    parent = os.path.dirname(base)
    return {
        "dir": base,
        "parent": parent if parent != base else None,
        "entries": entries,
    }
