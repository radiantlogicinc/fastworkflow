"""Standing auto-approvals for approval-annotated commands (ADR-EXP-025, owner amendment 2026-11-02).

A user can record a standing approval for one command: "always" on the bound confirmation, or
nothing else. The grant is per user, per workflow and per command; the user can list it and revoke
it, and every recording, use and revocation writes one audit span (``fw.approval``).

Storage is user-scoped and lives under the fastWorkflow state root, which is per OS user:

    <state-root>/approvals/auto_approvals.json

Each entry names its identity: ``end_user:<uid>`` for a verified FastAPI/MCP user (the JWT ``uid``),
or ``channel:<os user>`` for the CLI and embedding hosts that run as one local user. A host request
whose identity is not verified has no identity, so it can neither record nor use a grant.

The grant is an input to the policy decision point (``approval_binding.decide``), not an override:
a recorded grant yields ``run`` without a prompt. It is also the first concrete mechanism of
fix-47a9's delegation for ``delegatable`` commands.

A corrupt or unreadable file is treated as having no grants, so the gate asks again (fail closed).
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

from fastworkflow import state_paths, tracing

logger = logging.getLogger(__name__)

GRANTS_DIRNAME = "approvals"
GRANTS_FILENAME = "auto_approvals.json"
SCHEMA_VERSION = 1

_lock = threading.Lock()


def grants_path() -> str:
    return os.path.join(state_paths.state_root(), GRANTS_DIRNAME, GRANTS_FILENAME)


def local_user() -> str:
    try:
        return getpass.getuser() or "local"
    except Exception:  # noqa: BLE001 - no login name (containers) falls back to one bucket
        return "local"


def identity_key(surface: Any, channel: str) -> Optional[str]:
    """The user a grant belongs to, or None when the surface cannot name one (fail closed)."""
    if getattr(surface, "kind", "channel") == "host":
        if surface.satisfies_end_user():
            return f"end_user:{surface.uid}"
        return None
    return f"channel:{local_user()}"


def _read() -> dict:
    path = grants_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("grants"), list):
            raise ValueError("unexpected shape")
        return data
    except FileNotFoundError:
        return {"version": SCHEMA_VERSION, "grants": []}
    except Exception as exc:  # noqa: BLE001 - a corrupt file grants nothing
        logger.warning(f"auto-approval file unreadable; no grants are honoured ({type(exc).__name__})")
        return {"version": SCHEMA_VERSION, "grants": [], "_unreadable": True}


def _write(data: dict) -> None:
    path = grants_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".auto_approvals.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _active(data: dict, identity: str, workflow_id: str, command: str) -> Optional[dict]:
    for grant in data["grants"]:
        if (grant.get("identity") == identity and grant.get("workflow") == workflow_id
                and grant.get("command") == command and grant.get("revoked_at") is None):
            return grant
    return None


def find(identity: Optional[str], workflow_id: str, command: str) -> Optional[dict]:
    """The active grant for this user, workflow and command; None when there is none."""
    if identity is None:
        return None
    with _lock:
        data = _read()
    if data.get("_unreadable"):
        return None
    return _active(data, identity, workflow_id, command)


def record(identity: str, workflow_id: str, command: str, annotation: str) -> dict:
    """Record a standing approval. Recording the same command twice keeps one active grant."""
    with _lock:
        data = _read()
        if data.get("_unreadable"):
            raise RuntimeError("the auto-approval file is unreadable; fix or remove it first")
        existing = _active(data, identity, workflow_id, command)
        if existing is not None:
            return existing
        grant = {"id": uuid.uuid4().hex, "identity": identity, "workflow": workflow_id,
                 "command": command, "annotation": annotation, "recorded_at": time.time(),
                 "revoked_at": None}
        data["grants"].append(grant)
        _write({"version": SCHEMA_VERSION, "grants": data["grants"]})
        return grant


def revoke(identity: str, workflow_id: str, command: str) -> Optional[dict]:
    """Revoke the active grant (kept in the file as history); None when there was none."""
    with _lock:
        data = _read()
        if data.get("_unreadable"):
            raise RuntimeError("the auto-approval file is unreadable; fix or remove it first")
        grant = _active(data, identity, workflow_id, command)
        if grant is None:
            return None
        grant["revoked_at"] = time.time()
        _write({"version": SCHEMA_VERSION, "grants": data["grants"]})
        return grant


def list_active(identity: str, workflow_id: str) -> list[dict]:
    with _lock:
        data = _read()
    return [g for g in data["grants"]
            if g.get("identity") == identity and g.get("workflow") == workflow_id
            and g.get("revoked_at") is None]


# ----------------------------------------------------------------------
# Audit (ADR section 12): one span per recording, use and revocation
# ----------------------------------------------------------------------

STATE_NONE = "none"
STATE_GRANTED = "auto_approved"


def audit(host: Any, grant_id: str, *, command: str, annotation: str, before: str, after: str,
          reason: str, identity: str, channel: str, surface_kind: str, outcome: str) -> None:
    """One ``fw.approval`` span for a grant transition. Never carries a token or raw parameters."""
    try:
        attributes = {
            "approval_record_id": grant_id,
            "approval_command": command,
            "approval_annotation": annotation,
            "approval_state_before": before,
            "approval_state_after": after,
            "approval_reason": reason,
            "approval_outcome": outcome,
            "approval_fingerprint": "",
            "approval_rendering_hash": "",
            "approval_channel": channel,
            "approval_identity_class": "end_user" if identity.startswith("end_user:") else "channel",
            "approval_identity_ref": identity.split(":", 1)[1] if identity.startswith("end_user:") else "none",
            "approval_surface": surface_kind,
            "approval_origin": "user",
            "approval_mode": "enforce",
            "approval_shadow": False,
        }
        span = tracing.start_span(host, tracing.SPAN_APPROVAL, kind=tracing.KIND_INTERNAL,
                                  attributes=attributes, command_name=command)
        tracing.end_span(host, span, status=tracing.STATUS_OK)
    except Exception:  # noqa: BLE001 - auditing must never fail the decision it records
        logger.debug("auto-approval audit skipped", exc_info=True)
