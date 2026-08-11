"""Single source of truth for every persistent-state path fastWorkflow writes.

Historically each store composed its own path off the ``SPEEDDICT_FOLDERNAME``
env var, which was interpreted inconsistently: ``@enablecache`` rooted it under
the workflow folder, while the FastAPI conversation/session-state/checkpoint
trees rooted it under the process CWD. Relative values therefore meant two
different things, and durable data moved when a server was launched from a
different directory. Conversations and pending sessions were keyed by
``channel_id`` alone, so two workflows sharing a root could collide.

This module fixes both problems:

* **One absolute root.** ``FASTWORKFLOW_STATE_ROOT`` (default
  ``~/.local/state/fastworkflow``) is expanded and absolutised once, so state
  lives in a stable location independent of the code's on-disk location or the
  launch directory.
* **Per-workflow namespace.** Every subtree lives under
  ``<root>/workflows/<workflow-id>/``. The id defaults to the workflow folder's
  basename but is overridable via ``FASTWORKFLOW_WORKFLOW_ID`` so a rename or a
  redeploy from a different path keeps addressing the same state. It is encoded
  with the same injective encoder the stores use for channel ids.

Layout::

    <state-root>/workflows/<workflow-id>/
        conversations/<channel_id>.sqlite3
        session_state/<channel_id>.pending.json
        checkpoints/channels/<deployment>/<fingerprint>/<channel_key>/...
        function_cache/<fingerprint>/<function_name>/cache.sqlite3

The fingerprint subdivisions are deliberate: checkpoints and function caches
depend on the workflow's command source, so a code change must not read a stale
entry written by an incompatible version (checkpoints already partitioned this
way; function caches now do too).
"""

from __future__ import annotations

import os

import fastworkflow
from fastworkflow.storage_keys import encode_path_component

# Kept relative-looking on purpose; ``state_root`` expands the ``~`` and
# absolutises it. XDG's conventional location for persistent app state.
_DEFAULT_STATE_ROOT = os.path.join("~", ".local", "state", "fastworkflow")

_WORKFLOWS_DIRNAME = "workflows"
_CONVERSATIONS_DIRNAME = "conversations"
_SESSION_STATE_DIRNAME = "session_state"
_CHECKPOINTS_DIRNAME = "checkpoints"
_FUNCTION_CACHE_DIRNAME = "function_cache"


def _optional_env(name: str) -> str | None:
    """Read an optional env var (env file first, then OS env) without warning.

    ``get_env_var`` logs a warning when a variable is absent and no default is
    given; both variables here are legitimately optional, so this mirrors its
    file-then-environment precedence quietly.
    """
    value = fastworkflow._env_vars.get(name)
    if value is None:
        value = os.getenv(name)
    return value


def state_root() -> str:
    """Absolute root for all persistent state.

    Reads ``FASTWORKFLOW_STATE_ROOT`` (env file OR OS environment), falling back
    to ``~/.local/state/fastworkflow``. Always returned expanded and absolute so
    callers never re-resolve it against a shifting CWD.
    """
    raw = _optional_env("FASTWORKFLOW_STATE_ROOT")
    if not raw or not str(raw).strip():
        raw = _DEFAULT_STATE_ROOT
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def workflow_id(workflow_path: str) -> str:
    """Stable, filesystem-safe namespace for one workflow's state.

    Defaults to the workflow folder's basename so most deployments need no
    configuration, but ``FASTWORKFLOW_WORKFLOW_ID`` overrides it for the cases
    the basename cannot serve: a workflow renamed on disk, two unrelated
    workflows that happen to share a basename, or a redeploy from a different
    path that must keep its existing state.
    """
    override = _optional_env("FASTWORKFLOW_WORKFLOW_ID")
    if override and str(override).strip():
        raw = str(override).strip()
    else:
        resolved = os.path.abspath(os.path.expanduser(str(workflow_path)))
        raw = os.path.basename(os.path.normpath(resolved))
    if not raw:
        # A path like "/" has no basename; fall back to a fixed bucket rather
        # than crash the encoder on an empty component.
        raw = "workflow"
    return encode_path_component(raw)


def workflow_state_dir(workflow_path: str) -> str:
    """``<state-root>/workflows/<workflow-id>`` (created)."""
    path = os.path.join(state_root(), _WORKFLOWS_DIRNAME, workflow_id(workflow_path))
    os.makedirs(path, exist_ok=True)
    return path


def conversations_dir(workflow_path: str) -> str:
    """Per-channel conversation SQLite DBs for this workflow (created)."""
    path = os.path.join(workflow_state_dir(workflow_path), _CONVERSATIONS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def session_state_dir(workflow_path: str) -> str:
    """Suspended (awaiting_user) Topology-B blobs for this workflow (created)."""
    path = os.path.join(workflow_state_dir(workflow_path), _SESSION_STATE_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def checkpoints_dir(workflow_path: str) -> str:
    """Base folder for this workflow's channel checkpoints (created).

    ``ChannelCheckpointStore`` adds ``channels/<deployment>/<fingerprint>/
    <channel_key>/`` beneath this.
    """
    path = os.path.join(workflow_state_dir(workflow_path), _CHECKPOINTS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def function_cache_dir(workflow_path: str, function_name: str) -> str:
    """``@enablecache`` folder for one function, partitioned by command source.

    The fingerprint segment means a workflow whose commands changed does not
    read cache entries written by the prior version.
    """
    path = os.path.join(
        workflow_state_dir(workflow_path),
        _FUNCTION_CACHE_DIRNAME,
        _commands_fingerprint(workflow_path),
        function_name,
    )
    os.makedirs(path, exist_ok=True)
    return path


def _commands_fingerprint(workflow_path: str) -> str:
    """Hash of the workflow's command sources; ``unfingerprinted`` on failure.

    Imported lazily to avoid a module-import cycle through command_directory.
    """
    from fastworkflow.command_directory import compute_commands_source_fingerprint

    try:
        return compute_commands_source_fingerprint(workflow_path)
    except Exception:
        return "unfingerprinted"
