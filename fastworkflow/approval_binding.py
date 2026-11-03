"""Approval binding for approval-annotated commands (ADR-EXP-025, sections 6-12 and 15).

A thin ``action``-kind wrapper over the review engine (``fastworkflow.review_engine``). The engine owns
the record, the state machine, the cascade, the audit and the redaction rule. This module owns what is
specific to a gated write:

* the binding and its fingerprint (sha256 over canonical JSON; section 6.2);
* the policy decision point for the gate, which returns a graded outcome: run / ask / queue / refuse
  (ADR section 19.0 condition of approval; deterministic, no model call);
* the gate itself (``gate``), called from the single execution choke point (``CommandExecutor``);
* the resolution of a reply: exact "yes", "no" or "always", token, identity class, fingerprint, TTL,
  execute-once (sections 8, 10, 11); a reply on a proposal record goes to the review engine;
* the user's management messages (standing approvals) and the origin contextvars.

Switch: ``FW_APPROVAL_BINDING`` = ``off`` | ``shadow`` | ``enforce`` (default ``enforce``), read by the
engine. ``off`` restores the pre-change path; ``shadow`` records and audits every decision but executes as
before.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import string  # noqa: F401 - kept importable for callers that formatted templates here
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import fastworkflow
from fastworkflow import approval_grants, review_engine, state_paths
from fastworkflow.approval_annotations import APPROVAL_VALUES, command_approval
from fastworkflow.context_identity import context_identity
# The engine's public names, re-exported: the approval surface the gate, the hosts and the tests use.
from fastworkflow.review_engine import (  # noqa: F401
    ACTIVE_STATES,
    AWAITING,
    CHANNEL_SURFACE,
    CONFIRMED,
    DECLINED,
    DEFAULT_MODE,
    ENV_SWITCH,
    EXECUTED,
    EXPIRED,
    IDENTITY_CHANNEL,
    IDENTITY_END_USER,
    INVALIDATED,
    KIND_ACTION,
    MAX_HISTORY,
    MODE_ENFORCE,
    MODE_OFF,
    MODE_SHADOW,
    MODES,
    TTL_SECONDS,
    TTL_TURNS,
    ReviewRecord as PendingAction,
    ReviewStore as ApprovalStore,
    Surface,
    active_record,
    audit as _audit,
    canonical_json,
    clear_inbound,
    default_rendering,
    expire_if_due,
    fingerprint,
    new_token,
    propagate,
    reply_to_proposal,
    render,
    rendering_hash,
    store_for,
    switch_mode,
    take_delivery,
    token_hash,
    token_matches,
    transition as _transition,
)
from fastworkflow.review_engine import ActionReplyPolicy, Verdict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Exact user-facing texts (ADR section 10). Framework-rendered only.
# ----------------------------------------------------------------------

MSG_ASK = ("The agent wants to run {command} with {rendering}. This cannot be undone. "
           "Reply yes to approve, no to decline, or always to approve this command from now on.")
MSG_ASK_USER = ("You asked to run {command} with {rendering}. This cannot be undone. "
                "Reply yes to approve, no to decline, or always to approve this command from now on.")
MSG_ASK_DELEGATABLE = ("The agent wants to run {command} with {rendering}. "
                       "Reply yes to approve, no to decline, or always to approve this command from now on.")
MSG_UNATTENDED = ("Approval required for {command}; no user is available. The action was not run "
                  "and is queued until it expires in {ttl}.")
MSG_BEHIND = ("Approval required for {command}; another approval is awaiting the user. The action "
              "was not run and is queued until it expires in {ttl}.")
MSG_REPLY = "Reply yes to approve, no to decline, or always to approve this command from now on. The action has not run."
MSG_RECORD_FAILED = "The standing approval for {command} could not be recorded, so the action was not run."
MSG_NO_GRANT = "No standing approval for {command} is recorded for you."
MSG_GRANT_REVOKED = "The standing approval for {command} is revoked. It will ask you again."
MSG_DECLINED = "Declined. {command} was not run."
MSG_INVALIDATED = ("The parameters for {command} changed after the approval was requested. The action "
                   "was not run. Review the new request.")
MSG_EXPIRED = "The approval for {command} expired and the action was not run."
MSG_CANCELLED = "The approval for {command} was cancelled and the action was not run."
MSG_NO_LONGER_VALID = "That approval is no longer valid."
MSG_IDENTITY = ("This action requires a verified user identity, which this server is not configured "
                "to check.")
MSG_FAILED = "{command} failed after approval; it was not retried: {error}"

#: The reply that approves, and the one that declines. Exact match only (ADR section 10).
REPLY_YES = "yes"
REPLY_NO = "no"
#: Approves this one binding and records a standing auto-approval for the command (owner 2026-11-02).
REPLY_ALWAYS = "always"
#: The user's management messages (listing and revoking standing approvals); never the agent's.
MANAGE_LIST = "/approvals"
MANAGE_REVOKE = "/revoke_approval"
CANCEL_COMMAND = "/cancel_pending"
#: The framework's review command (a core execution command in the command metadata extraction workflow).
REVIEW_COMMAND = "review_proposal"


# ----------------------------------------------------------------------
# Origin and approved-execution context (ADR section 5)
# ----------------------------------------------------------------------


class Origin(str, Enum):
    """Who asked for the call. Only agent-originated calls are gated."""

    USER = "user"                  # the user's own instruction: assistant mode, '/', direct actions
    AGENT = "agent"                # the agent's tool loop, including planning-command steps: suspendable
    AGENT_DIRECT = "agent_direct"  # agent-originated MCP tool calls (owner answer 3): not suspendable


_origin_var: contextvars.ContextVar[Origin] = contextvars.ContextVar(
    "fw_approval_origin", default=Origin.USER)
_approved_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fw_approval_approved_record", default=None)


def current_origin() -> Origin:
    return _origin_var.get()


@contextlib.contextmanager
def origin_scope(value: Origin):
    token = _origin_var.set(value)
    try:
        yield
    finally:
        _origin_var.reset(token)


@contextlib.contextmanager
def approved_execution(record_id: str):
    """Mark the execution of one confirmed record: the gate lets exactly that binding run."""
    token = _approved_var.set(record_id)
    try:
        yield
    finally:
        _approved_var.reset(token)


# ----------------------------------------------------------------------
# Binding (ADR sections 6.1, 6.2)
# ----------------------------------------------------------------------


def make_binding(*, command: str, params: dict, targets: list, principal: dict,
                 task: str, channel: str) -> dict:
    return {"command": command, "params": params, "targets": targets,
            "principal": principal, "task": task, "channel": channel}


# ----------------------------------------------------------------------
# Policy decision point (ADR section 19.0 condition of approval)
# ----------------------------------------------------------------------


class Outcome(str, Enum):
    RUN = "run"
    ASK = "ask"
    QUEUE = "queue"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str


def decide(annotation: Optional[str], *, origin: Origin, surface: Surface, unattended: bool,
           awaiting_on_channel: bool, auto_granted: bool = False, consequence: Any = None,
           uncertainty: Any = None) -> Decision:
    """The gate's policy: a pure, deterministic function from its inputs to a graded outcome.

    ``auto_granted`` is the user's standing approval for this command (approval_grants): an input,
    so a recorded grant yields ``run`` without a prompt, on any surface. ``consequence`` and
    ``uncertainty`` are reserved inputs (fix-ajv.11, fix-47a9 delegation): accepted so the decision
    can grow without a new signature, and they never turn an outcome into a boolean.

    A user's own ``delegatable`` instruction runs (owner 2026-11-02 leaves it unchanged). A user's
    own ``mandatory`` instruction asks, the same as an agent's call (owner amendment 2026-11-02).
    """
    del consequence, uncertainty  # reserved; see docstring
    if annotation not in APPROVAL_VALUES:
        return Decision(Outcome.RUN, "unannotated")
    if origin is Origin.USER and annotation == "delegatable":
        return Decision(Outcome.RUN, "delegatable, and the user's own instruction")
    if surface.required_identity == IDENTITY_END_USER and not surface.satisfies_end_user():
        return Decision(Outcome.REFUSE, "end_user identity required and not verifiable")
    if auto_granted:
        return Decision(Outcome.RUN, "standing auto-approval recorded by the user")
    if unattended:
        return Decision(Outcome.QUEUE, "no human channel")
    if awaiting_on_channel:
        return Decision(Outcome.QUEUE, "behind an awaiting approval")
    if annotation == "delegatable":
        return Decision(Outcome.ASK, "delegatable asks until delegation exists")
    return Decision(Outcome.ASK, "mandatory")


# ----------------------------------------------------------------------
# Turns and queued records
# ----------------------------------------------------------------------


def on_new_turn(host: Any, workflow: Any, *, unattended: bool) -> None:
    """A new user turn starts: advance the turn counter, note whether a human can answer this
    turn (ask_user is off for an unattended turn), and expire queued records by turns or time."""
    store = store_for(workflow)
    if store is None:
        return
    store.turn_seq += 1
    store.unattended = bool(unattended)
    now = time.time()
    for record in list(store.records):
        if record.state in ACTIVE_STATES and (record.time_expired(now) or record.turns_expired(turn_seq=store.turn_seq)):
            if record.queued:
                _transition(host, store, record, EXPIRED, reason="queued approval expired",
                            surface=store.surface)


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------


def _qualified(workflow, command_name: str) -> Optional[str]:
    """The qualified name of *command_name* in the workflow's routing, or None."""
    try:
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    except Exception:  # noqa: BLE001 - no routing means no annotation to read
        return None
    try:
        directory = routing.command_directory.map_command_2_metadata
    except AttributeError:  # not a routing definition: no annotation can be read (ADR section 2.2)
        return None
    if command_name in directory:
        return command_name
    leaf = [name for name in directory if name.split("/")[-1] == command_name]
    return leaf[0] if len(leaf) == 1 else None


def _params_of(input_obj: Any) -> dict:
    if input_obj is None:
        return {}
    if hasattr(input_obj, "model_dump"):
        return input_obj.model_dump(mode="json")
    return dict(input_obj) if isinstance(input_obj, dict) else {}


def build_binding(workflow, command: str, params: dict, surface: Surface, store: ApprovalStore) -> dict:
    """The binding that would execute now: the target is the instance in the current context."""
    principal = ({"class": IDENTITY_END_USER, "ref": surface.uid or "none"}
                 if surface.required_identity == IDENTITY_END_USER
                 else {"class": IDENTITY_CHANNEL, "ref": store.channel})
    return make_binding(command=command, params=params, targets=list(context_identity(workflow)),
                        principal=principal, task=store.task, channel=store.channel)


def awaiting_output(workflow, text: str):
    """An awaiting CommandOutput carrying the framework text, and any token just delivered."""
    command_response = fastworkflow.CommandResponse(response=text)
    command_response.artifacts["awaiting_user"] = True
    output = fastworkflow.CommandOutput(command_response=command_response)
    if workflow is not None:
        output.workflow_name = workflow.folderpath.split("/")[-1]
    return output


def _refusal_output(workflow, command: str, text: str):
    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response=text, success=False),
        command_name=command)
    if workflow is not None:
        output.workflow_name = workflow.folderpath.split("/")[-1]
    return output


def gate(chat_session: Any, workflow: Any, command_name: str, input_obj: Any, raw_command: str):
    """Decide whether a command call may run now.

    Returns None to run; otherwise a CommandOutput the caller returns as the command's result
    (refuse, queue, or an awaiting output for a non-suspendable caller). For an agent-origin ask
    inside the agent's tool loop, raises ``AskUserSuspend``: the ask_user suspension.
    """
    store = store_for(workflow)
    if store is None:
        return None
    approved = _approved_var.get()
    if approved is not None:
        # The one execution a confirmed record authorised: the stored binding, unchanged.
        record = store.get(approved)
        if record is not None and record.state == CONFIRMED:
            return None
    mode = switch_mode()
    if mode == MODE_OFF:
        return None
    origin = current_origin()
    qualified = _qualified(workflow, command_name)
    if qualified is None:
        return None
    annotation = command_approval(workflow.folderpath, qualified)
    if annotation is None:
        return None

    params = _params_of(input_obj)
    binding = build_binding(workflow, qualified, params, store.surface, store)
    fp = fingerprint(binding)
    rendering = render(_approval_template(workflow, qualified), params)
    identity = approval_grants.identity_key(store.surface, store.channel)
    grant = approval_grants.find(identity, state_paths.workflow_id(workflow.folderpath), qualified)
    decision = decide(annotation, origin=origin, surface=store.surface, unattended=store.unattended,
                      awaiting_on_channel=store.active() is not None, auto_granted=grant is not None)
    if grant is not None and decision.outcome is Outcome.RUN and mode == MODE_ENFORCE:
        approval_grants.audit(chat_session, grant["id"], command=qualified, annotation=annotation,
                              before=approval_grants.STATE_GRANTED, after="used", reason=decision.reason,
                              identity=identity, channel=store.channel, surface_kind=store.surface.kind,
                              outcome="used")

    def new_record(*, state: str, queued: bool, suspended: bool, text: str) -> PendingAction:
        record = PendingAction(
            id=uuid.uuid4().hex, token_hash="", command=qualified, annotation=annotation,
            raw_command=raw_command, binding=binding, fingerprint=fp, rendering=text,
            rendering_hash=rendering_hash(text), required_identity=store.surface.required_identity,
            created_turn=store.turn_seq, turns_seen=0, ttl_turns=TTL_TURNS,
            ttl_seconds=TTL_SECONDS, created_at=time.time(), state=state, origin=origin.value,
            suspended_agent=suspended, queued=queued, shadow=(mode == MODE_SHADOW))
        store.add(record)
        return record

    if decision.outcome is Outcome.REFUSE:
        # A refusal creates no record (ADR section 7 defines none for it); it is audited on a
        # transient record that is never stored.
        transient = PendingAction(
            id=uuid.uuid4().hex, token_hash="", command=qualified, annotation=annotation,
            raw_command=raw_command, binding=binding, fingerprint=fp, rendering="",
            rendering_hash=rendering_hash(""), required_identity=store.surface.required_identity,
            created_turn=store.turn_seq, turns_seen=0, ttl_turns=TTL_TURNS, ttl_seconds=TTL_SECONDS,
            created_at=time.time(), state="refused", origin=origin.value, shadow=(mode == MODE_SHADOW))
        _audit(chat_session, transient, before="", after="refused", reason=decision.reason,
               surface=store.surface, mode=mode, outcome="refused")
        if mode == MODE_SHADOW:
            return None
        return _refusal_output(workflow, qualified, MSG_IDENTITY)

    if mode == MODE_SHADOW:
        # Shadow records the decision and runs as before: the record is born executed, and its
        # one audit row says what enforcement would have done.
        record = new_record(state=EXECUTED, queued=False, suspended=False, text="")
        record.outcome = "shadow"
        _audit(chat_session, record, before=EXECUTED, after=EXECUTED,
               reason=f"shadow: would {decision.outcome.value} ({decision.reason}); ran",
               surface=store.surface, mode=mode, outcome="shadow")
        return None

    if decision.outcome is Outcome.RUN:
        return None

    ttl = f"{TTL_TURNS} turns or {TTL_SECONDS // 60} minutes"
    if decision.outcome is Outcome.QUEUE:
        template = MSG_UNATTENDED if store.unattended else MSG_BEHIND
        text = template.format(command=qualified, ttl=ttl)
        record = new_record(state=AWAITING, queued=True, suspended=False, text=text)
        _audit(chat_session, record, before="", after=AWAITING, reason=decision.reason,
               surface=store.surface, mode=mode, outcome="queued")
        return _refusal_output(workflow, qualified, text)

    # ASK
    if origin is Origin.USER:
        template = MSG_ASK_USER
    else:
        template = MSG_ASK if annotation == "mandatory" else MSG_ASK_DELEGATABLE
    text = template.format(command=qualified, rendering=rendering)
    suspendable = origin is Origin.AGENT
    record = new_record(state=AWAITING, queued=False, suspended=suspendable, text=text)
    token = new_token()
    record.token_hash = token_hash(token)
    if store.surface.kind == "host":
        store.delivery = {"record_id": record.id, "token": token}
    _audit(chat_session, record, before="", after=AWAITING, reason=decision.reason,
           surface=store.surface, mode=mode, outcome="asked")
    if suspendable:
        from fastworkflow.utils.react import AskUserSuspend

        raise AskUserSuspend(text)
    return awaiting_output(workflow, text)


def _approval_template(workflow, qualified: str) -> Optional[str]:
    try:
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        cls = routing.get_command_class(qualified, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        value = getattr(cls, "approval_template", None)
        return value if isinstance(value, str) else None
    except Exception:  # noqa: BLE001 - a missing template falls back to the default rendering
        return None


# ----------------------------------------------------------------------
# Resolution of a reply (ADR sections 8, 10, 11)
# ----------------------------------------------------------------------


@dataclass
class Resolution:
    """``done`` False: the record is still open; ``text`` is what the user is told. ``done``
    True: the record left the open states; ``text`` is the observation for the agent, ``output`` is the
    command output to return when no agent is waiting, and ``value`` is what a review accepted (None for
    an action)."""

    done: bool
    text: str
    output: Any = None
    value: Any = None


def _ttl_text(record: PendingAction) -> str:
    return f"{record.ttl_turns} turns or {record.ttl_seconds // 60} minutes"


def resolve_reply(chat_session: Any, workflow: Any, message: str) -> Optional[Resolution]:
    """Resolve the next message on a session that has an open record; None when there is none."""
    store = store_for(workflow)
    record = store.active() if store is not None else None
    if record is None:
        return None
    surface = store.surface
    cmd = record.command

    if message == CANCEL_COMMAND:
        _transition(chat_session, store, record, INVALIDATED, reason="cancelled",
                    surface=surface, outcome="cancelled")
        propagate(chat_session, store, record, surface=surface)
        return Resolution(True, MSG_CANCELLED.format(command=cmd))

    if record.time_expired(time.time()) or record.turns_expired(turn_seq=store.turn_seq) \
            or record.turns_seen >= record.ttl_turns:
        _transition(chat_session, store, record, EXPIRED, reason="ttl", surface=surface, outcome="expired")
        propagate(chat_session, store, record, surface=surface)
        return Resolution(True, MSG_EXPIRED.format(command=cmd))

    if record.required_identity == IDENTITY_END_USER:
        if not token_matches(store.inbound_token, record.token_hash):
            return Resolution(False, MSG_NO_LONGER_VALID)
        if not surface.satisfies_end_user():
            return Resolution(False, MSG_IDENTITY)

    record.turns_seen += 1
    if record.kind != KIND_ACTION:
        # A proposal (sub_tasks, plan): typed edits, accept or decline, through the review engine.
        review = reply_to_proposal(chat_session, store, record, message)
        if review.done and review.proposal is not None:
            from fastworkflow.review_proposals import PROPOSAL_KINDS

            if any(cls.spec == record.command for cls in PROPOSAL_KINDS.values()):
                # A framework round accepted: the review command runs once more under this record, so its typed
                # output is the accepted proposal (the step the planning command is waiting on).
                output = _execute_review(workflow, record)
                return Resolution(True, output.command_response.response or review.text, output, review.proposal)
            # Any other proposal (a caller's own spec): the accepted proposal is the result.
            return Resolution(True, review.text, None, review.proposal)
        return Resolution(review.done, review.text, None, review.proposal)

    verdict = ActionReplyPolicy().classify(message)
    if verdict is Verdict.REPROMPT:
        return Resolution(False, MSG_REPLY)

    if verdict is Verdict.DECLINE:
        _transition(chat_session, store, record, DECLINED, reason="declined", surface=surface,
                    outcome="declined",
                    resolution=_resolution(record, surface, "declined"))
        return Resolution(True, MSG_DECLINED.format(command=cmd))

    # always: the standing approval needs an identity the grant can belong to.
    identity = approval_grants.identity_key(surface, store.channel)
    if verdict is Verdict.ALWAYS and identity is None:
        return Resolution(False, MSG_IDENTITY)

    # yes / always: the binding that would execute now must still be the one the user saw.
    params = dict(record.binding.get("params") or {})
    current = build_binding(workflow, cmd, params, surface, store)
    if fingerprint(current) != record.fingerprint:
        _transition(chat_session, store, record, INVALIDATED, reason="binding changed",
                    surface=surface, outcome="invalidated")
        return Resolution(True, MSG_INVALIDATED.format(command=cmd))

    if verdict is Verdict.ALWAYS:
        # Recorded only now, once the binding is confirmed: a rejected reply records nothing.
        try:
            grant = approval_grants.record(identity, state_paths.workflow_id(workflow.folderpath), cmd,
                                           record.annotation)
        except (RuntimeError, OSError) as exc:
            logger.warning(f"standing approval not recorded ({type(exc).__name__})")
            return Resolution(False, MSG_RECORD_FAILED.format(command=cmd))
        approval_grants.audit(chat_session, grant["id"], command=cmd, annotation=record.annotation,
                              before=approval_grants.STATE_NONE, after=approval_grants.STATE_GRANTED,
                              reason="recorded by the user (always)", identity=identity,
                              channel=store.channel, surface_kind=surface.kind, outcome="recorded")

    _transition(chat_session, store, record, CONFIRMED, reason="confirmed", surface=surface,
                resolution=_resolution(record, surface, "confirmed"))
    try:
        with approved_execution(record.id):
            output = _execute_stored(workflow, record)
    except Exception as exc:  # noqa: BLE001 - executed once; the failure is the outcome, never retried
        _transition(chat_session, store, record, EXECUTED, reason="execution failed", surface=surface,
                    outcome="failed")
        text = MSG_FAILED.format(command=cmd, error=f"{type(exc).__name__}: {exc}")
        return Resolution(True, text, _refusal_output(workflow, cmd, text))
    _transition(chat_session, store, record, EXECUTED, reason="executed", surface=surface,
                outcome="success" if output.success else "failed")
    text = output.command_response.response or ""
    return Resolution(True, text, output)


def _resolution(record: PendingAction, surface: Surface, verdict: str) -> dict:
    return {"verdict": verdict, "identity_class": record.required_identity,
            "identity_ref": surface.uid if surface.uid else "none"}


def _execute_review(workflow, record: PendingAction):
    """Run the review command once for an accepted proposal record (the approved execution)."""
    from fastworkflow.command_executor import CommandExecutor

    from fastworkflow.review_proposals import PROPOSAL_KINDS

    number = next(key for key, cls in PROPOSAL_KINDS.items() if cls.spec == record.command)
    action = fastworkflow.Action(command_name=REVIEW_COMMAND, command=REVIEW_COMMAND,
                                 parameters={"round": number, "items": record.binding.get("items_text", "")})
    with review_engine.approved_review(record.id):
        return CommandExecutor.perform_action(workflow, action)


def _execute_stored(workflow, record: PendingAction):
    """Run the stored binding once, through the same executor the agent's call uses."""
    from fastworkflow.command_executor import CommandExecutor

    action = fastworkflow.Action(command_name=record.command, command=record.raw_command,
                                 parameters=dict(record.binding.get("params") or {}))
    return CommandExecutor.perform_action(workflow, action)


def cancel_all(chat_session: Any, workflow: Any, *, reason: str = "session ended") -> int:
    """Invalidate every open or queued record: session end, eviction, /cancel_pending (ADR section 11).

    Returns how many records were cancelled. Nothing executes, and a later reply finds no record.
    """
    store = store_for(workflow)
    if store is None:
        return 0
    cancelled = 0
    for record in list(store.records):
        if record.state in ACTIVE_STATES:
            _transition(chat_session, store, record, INVALIDATED, reason=reason,
                        surface=store.surface, outcome="cancelled")
            propagate(chat_session, store, record, surface=store.surface)
            cancelled += 1
    store.delivery = None
    store.inbound_token = None
    return cancelled


def take_host_delivery(workflow: Any) -> dict:
    """The live-response fields for a host: ``{"approval": {record_id, token}}``, or {} when no token is
    pending. Taken once; the caller attaches it to the response of the request that produced it."""
    delivery = take_delivery(workflow)
    return {"approval": delivery} if delivery is not None else {}


# ----------------------------------------------------------------------
# The user's management messages: list and revoke standing approvals (owner 2026-11-02)
# ----------------------------------------------------------------------


def management_reply(chat_session: Any, workflow: Any, message: str) -> Optional[str]:
    """The reply to ``/approvals`` or ``/revoke_approval <command>``; None for any other message.

    These are the user's own messages on the session, handled before routing, so the agent has no
    tool for them. Recording a standing approval is only ever the ``always`` reply to a prompt.
    """
    text = message.strip()
    if not text.startswith((MANAGE_LIST, MANAGE_REVOKE)):
        return None
    store = store_for(workflow)
    if store is None:
        return None
    identity = approval_grants.identity_key(store.surface, store.channel)
    if identity is None:
        return MSG_IDENTITY
    workflow_id = state_paths.workflow_id(workflow.folderpath)
    if text == MANAGE_LIST:
        grants = approval_grants.list_active(identity, workflow_id)
        if not grants:
            return "No standing approvals are recorded for you."
        lines = [f"- {grant['command']}" for grant in grants]
        return "Standing approvals (revoke one with /revoke_approval <command>):\n" + "\n".join(lines)
    if not text.startswith(MANAGE_REVOKE + " ") or not text[len(MANAGE_REVOKE):].strip():
        return f"Use {MANAGE_REVOKE} <command>, or {MANAGE_LIST} to see the standing approvals."
    name = text[len(MANAGE_REVOKE):].strip()
    matches = [g for g in approval_grants.list_active(identity, workflow_id)
               if g["command"] == name or g["command"].split("/")[-1] == name]
    if not matches:
        return MSG_NO_GRANT.format(command=name)
    if len(matches) > 1:
        return f"{name} matches more than one standing approval; name it fully (see {MANAGE_LIST})."
    grant = approval_grants.revoke(identity, workflow_id, matches[0]["command"])
    if grant is None:
        return MSG_NO_GRANT.format(command=name)
    approval_grants.audit(chat_session, grant["id"], command=grant["command"], annotation=grant["annotation"],
                          before=approval_grants.STATE_GRANTED, after="auto_revoked",
                          reason="revoked by the user", identity=identity, channel=store.channel,
                          surface_kind=store.surface.kind, outcome="revoked")
    return MSG_GRANT_REVOKED.format(command=grant["command"])
