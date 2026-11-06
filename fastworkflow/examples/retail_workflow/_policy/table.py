"""Worked example: a policy verifier table for the retail workflow.

The rules here are RETAIL CONTENT - an example of what a table compiled from
distillation looks like, not part of the framework. The engine
(``fastworkflow.policy_decision``) knows no workflow; this file is what a
workflow supplies. It is on because it is here, at ``_policy/table.py`` in the
workflow folder; there is no setting.

How it was made: failing trials of a small model (gpt-oss-20b) on six retail
tasks were distilled, over four rounds, against passing trials - of the same
model where one passed, else of a teacher model
(``fastworkflow distill --self`` / ``--pool-students ... --pool-teachers ...``).
The cumulative insights are in ``corpus/``; ``taxonomy.yaml`` clusters them
into failure modes and says which became the rows below and why the rest did
not. This is the table the 15-task evaluation runs used (retail-demo/0.4.1).
See the ``distill-policy-verifiers`` and ``insights-to-failure-modes`` coding
skills for the process.

Rows:
  S1-S8  from self-distillation (a failing trial against a passing trial of
         the same model): refund total, refund method, cancel reason, double
         cancel, return from a delivered order only, no claimed cancellation
         that did not run, every pending order when all were asked, read every
         order before writing.
  C2 C4 C5 C6 C8  from pool distillation (passing reference of the same model,
         else a teacher): status fits the command, explicit yes before a write,
         no write that changes nothing, every request carried out, no transfer
         while the request is doable.
  D1     from a later round: the order-locking write last.

Rows read only declared workflow facts - which command performs each operation,
which order status each command requires, the accepted argument values - and the
conversation so far. No row names a task's situation, record or amount.
"""

from __future__ import annotations

import json
import re

from fastworkflow.policy_decision import (
    USER_MESSAGE,
    USER_UTTERANCES,
    ContractFacts,
    DecisionPhase,
    DecisionTable,
    PolicyOutcome,
    PolicyRow,
)

#: Every decision records this, so a run can be attributed to a rule set.
#: The version the evaluation runs recorded; this file is that table with its
#: helpers inlined (and JSON read whether or not a response prefixes it).
TABLE_VERSION = "retail-demo/0.4.1"

_BEFORE_WRITE = frozenset({DecisionPhase.BEFORE_SIDE_EFFECT})
_BEFORE_FINISH = frozenset({DecisionPhase.BEFORE_FINISH})

# ---------------------------------------------------------------------------
# declared workflow facts
# ---------------------------------------------------------------------------

#: Commands that change the store. Declared, not inferred.
WRITE_COMMANDS = frozenset({
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
})

ORDER_ID = r"#W\d+"
NUMERIC_ID = r"\d{6,}"  # item, product and payment ids

#: Arguments whose domain is closed.
ENUM_ARGUMENTS = {
    "cancel_pending_order.reason": frozenset({"no longer needed", "ordered by mistake"}),
}

#: What a customer can ask this workflow to do. The intent reader answers only
#: in these names, so everything it produces maps onto commands (SATISFIED_BY).
INTENT_OPERATIONS = {
    "cancel_order": (
        'Cancel an ENTIRE pending order - every item in it. NOT "cancel the X '
        'and replace it with Y" (that is swap_item), and NOT cancelling one '
        "item out of an order (that is remove_item)."
    ),
    "return_item": "Return item(s) from an order that has already been delivered.",
    "swap_item": (
        "Replace an item with a different variant of the same product - another "
        "size, colour, model, capacity - in a pending or a delivered order."
    ),
    "remove_item": (
        "Take an item out of a pending order without replacing it. (This "
        "workflow cannot do that; record the request anyway.)"
    ),
    "change_order_address": "Change the shipping address of one or more pending orders.",
    "change_profile_address": "Change the default address saved on the customer's account.",
    "change_order_payment": (
        "Switch the payment method an existing pending order is charged to. "
        "NOT saying which card or account should cover a price difference, "
        "refund, or new charge arising from another request - that is detail "
        "of the other request, not a separate operation."
    ),
    "report_refund_total": (
        "The customer asks to be told how much money they will get back "
        "(a refund amount or total). Only a request for the figure - not a "
        "question about timing or the payment method."
    ),
}

#: Which commands carry out each operation. An operation with no command is
#: recorded for context but never demands one.
SATISFIED_BY = {
    "cancel_order": {"cancel_pending_order"},
    "return_item": {"return_delivered_order_items"},
    "swap_item": {"modify_pending_order_items", "exchange_delivered_order_items"},
    "remove_item": set(),
    "change_order_address": {"modify_pending_order_address"},
    "change_profile_address": {"modify_user_address"},
    # A card named for a price difference travels with the swap command itself,
    # so a swap also meets a payment request: a misread "pay the difference with
    # my card" must never demand a payment write nobody asked for.
    "change_order_payment": {
        "modify_pending_order_payment",
        "modify_pending_order_items",
        "exchange_delivered_order_items",
    },
    "report_refund_total": set(),
}

#: The status an order must have for each command to apply.
REQUIRES_STATUS = {
    "cancel_pending_order": "pending",
    "modify_pending_order_address": "pending",
    "modify_pending_order_items": "pending",
    "modify_pending_order_payment": "pending",
    "return_delivered_order_items": "delivered",
    "exchange_delivered_order_items": "delivered",
}

#: What a successful command leaves the order's status as. Address and payment
#: changes keep an order "pending"; an item change makes it "pending (items
#: modified)", which no further modify accepts.
STATUS_AFTER = {
    "cancel_pending_order": "cancelled",
    "modify_pending_order_items": "pending (items modified)",
    "return_delivered_order_items": "return requested",
    "exchange_delivered_order_items": "exchange requested",
}

#: What the customer must be shown before saying yes to each write (the
#: domain policy requires explicit confirmation of the action details).
CONFIRM_DETAILS = {
    "cancel_pending_order": (
        "the order id, EVERY item in that order (all of them are cancelled, not "
        "just one), and the reason"
    ),
    "return_delivered_order_items": (
        "the order id, each item being returned by name, options and item id, "
        "and the payment method the refund goes to"
    ),
    "exchange_delivered_order_items": (
        "the order id, each item and the variant it becomes (name, options, item "
        "id), the price difference, and the payment method"
    ),
    "modify_pending_order_items": (
        "the order id, each item and the variant it becomes (name, options, item "
        "id), the price difference, and the payment method"
    ),
    "modify_pending_order_address": "the order id and the full new address",
    "modify_pending_order_payment": "the order id and the new payment method",
    "modify_user_address": "the full new default address",
}

FACTS = ContractFacts(
    write_commands=WRITE_COMMANDS,
    id_patterns=(ORDER_ID, NUMERIC_ID),
    enum_arguments=ENUM_ARGUMENTS,
    intent_operations=INTENT_OPERATIONS,
    # The retail tools report errors as text ("Error: order not found") in a
    # response that still counts as successful.
    failure_pattern=r"(?i)\b(error|not found|invalid|cannot|failed)\b",
)

# ---------------------------------------------------------------------------
# reading the conversation
# ---------------------------------------------------------------------------

_ORDER_RE = re.compile(ORDER_ID)
_ORDER_FIELD_RE = re.compile(r'"order_id"\s*:\s*"(#W\d+)"')
_ADDRESS_FIELDS = ("address1", "address2", "city", "state", "zip")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_MONEY_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?")
_CLOSING_QUESTION = re.compile(
    r"anything else|further (help|assistance)|help you with|other questions", re.I)
#: Stands in for "no user message yet", which is never consent.
_NO_UTTERANCE = type("NoUtterance", (), {"args": {}})()


def _json_in(text) -> dict | None:
    """The JSON object in a response ("Order details: {...}" or bare JSON)."""
    text = str(text or "")
    start = text.find("{")
    if start < 0:
        return None
    try:
        found = json.loads(text[start:])
    except (TypeError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _order_in(args, response: str = "") -> str:
    """The order a command acted on: its order_id argument, else the order_id
    its response names."""
    if order_id := str((args or {}).get("order_id", "") or ""):
        return order_id
    if found := _ORDER_FIELD_RE.search(str(response or "")[:300]):
        return found.group(1)
    return ""


def _target(inputs) -> str:
    """The order the pending command is aimed at, or ""."""
    return _order_in(inputs.command_args)


def _uninspected(inputs) -> list[str]:
    """Orders the account lists that no get_order_details has read."""
    known: set[str] = set()
    for record in inputs.ran("get_user_details"):
        known.update(_ORDER_RE.findall(str(record.response)))
    read = {_order_in(r.args, r.response) for r in inputs.ran("get_order_details")}
    return sorted(known - read)


def _awaits_the_user(answer: str) -> bool:
    """Whether the answer ends the turn by asking the customer something real.

    A turn that stops to ask ("shall I go ahead with the swap?") is waiting for
    consent, not leaving work undone. "Anything else?" is a closing.
    """
    return any(
        "?" in sentence and not _CLOSING_QUESTION.search(sentence)
        for sentence in re.split(r"(?<=[.?!])\s+", answer or "")
    )


def _order_json(inputs, order_id: str) -> dict | None:
    """The last successful read of an order, parsed."""
    found = None
    for record in inputs.ran("get_order_details"):
        if _order_in(record.args, record.response) == order_id:
            found = _json_in(record.response) or found
    return found


def _status(inputs) -> dict[str, str]:
    """Each order's status: last read, then updated by successful writes."""
    status: dict[str, str] = {}
    for record in inputs.commands_run:
        if not record.success:
            continue
        order = _order_in(record.args, record.response)
        if not order:
            continue
        if record.name == "get_order_details":
            found = (_json_in(record.response) or {}).get("status")
            if found:
                status[order] = str(found).strip().lower()
        elif record.name in STATUS_AFTER:
            status[order] = STATUS_AFTER[record.name]
    return status


def _ever_status(inputs) -> dict[str, set[str]]:
    """Every status each order was ever read in (for "all" requests)."""
    seen: dict[str, set[str]] = {}
    for record in inputs.ran("get_order_details"):
        order = _order_in(record.args, record.response)
        found = (_json_in(record.response) or {}).get("status")
        if order and found:
            seen.setdefault(order, set()).add(str(found).strip().lower())
    return seen


def _address_key(fields) -> tuple | None:
    if not isinstance(fields, dict):
        return None
    key = tuple(str(fields.get(k, "") or "").strip().lower() for k in _ADDRESS_FIELDS)
    return key if all(key[i] for i in (0, 2, 4)) else None


def _current_address(inputs, target: str) -> tuple | None:
    """The stored address of an order ("#W..") or of the profile ("@profile"),
    as last read or last successfully written."""
    current = None
    for record in inputs.commands_run:
        if not record.success:
            continue
        if target == "@profile":
            if record.name == "get_user_details":
                current = _address_key((_json_in(record.response) or {}).get("address"))
            elif record.name == "modify_user_address":
                current = _address_key(record.args) or current
        elif _order_in(record.args, record.response) == target:
            if record.name == "get_order_details":
                current = _address_key((_json_in(record.response) or {}).get("address"))
            elif record.name == "modify_pending_order_address":
                current = _address_key(record.args) or current
    return current


def _succeeded_on(inputs, commands: set[str]) -> set[str]:
    """Orders a successful command from ``commands`` targeted ("" if none)."""
    return {
        _order_in(r.args, r.response)
        for r in inputs.commands_run
        if r.success and r.name in commands
    }


_STATUS_RE = re.compile(r'"status"\s*:\s*"([^"]+)"')


def _orders_known(inputs) -> set[str]:
    """Order ids the account is known to have, from get_user_details output."""
    known: set[str] = set()
    for record in inputs.ran("get_user_details"):
        known.update(_ORDER_RE.findall(str(record.response)))
    return known


def _order_status(inputs) -> dict[str, str]:
    """Each order's status as last read, then "cancelled" after a successful
    cancel (the S rows' view; C rows use ``_status``, which also follows the
    other writes)."""
    status: dict[str, str] = {}
    for record in inputs.commands_run:
        order_id = _order_in(record.args, record.response)
        if not order_id or not record.success:
            continue
        if record.name == "get_order_details":
            found = (_json_in(record.response) or {}).get("status")
            if not found:
                match = _STATUS_RE.search(str(record.response))
                found = match.group(1) if match else None
            if found:
                status[order_id] = str(found).lower()
        elif record.name == "cancel_pending_order":
            status[order_id] = "cancelled"
    return status


# ---------------------------------------------------------------------------
# S2 - a refund goes to the original payment method or a gift card
# ---------------------------------------------------------------------------


def _refund_method_invalid(inputs) -> bool:
    if inputs.command_name != "return_delivered_order_items":
        return False
    method = str(inputs.command_args.get("payment_method_id", "")).strip()
    order = _order_json(inputs, _target(inputs))
    if not method or not order:
        return False  # nothing to compare against - unknown, not a violation
    original = {
        str(p.get("payment_method_id"))
        for p in order.get("payment_history", [])
        if p.get("transaction_type") == "payment"
    }
    return method not in original and not method.startswith("gift_card_")


# ---------------------------------------------------------------------------
# S5 - every pending order cancelled when all were asked for
# ---------------------------------------------------------------------------


def _all_pending_requested(inputs) -> bool:
    return any(r.op == "cancel_order" and r.scope == "all" and not r.conditional
               for r in inputs.intent)


def _pending_left(inputs) -> list[str]:
    """Orders not yet shown to be cancelled or non-pending: unread ones, and
    ones read as pending with no successful cancel since."""
    if not _orders_known(inputs):
        return []
    status = _order_status(inputs)
    cancelled = {_order_in(r.args, r.response) for r in inputs.ran("cancel_pending_order")}
    # An order already cancelled needs no reading - there is nothing left to do.
    unread = [o for o in _uninspected(inputs) if o not in cancelled]
    pending = sorted(o for o, s in status.items() if s == "pending")
    return [f"{o} (not read yet)" for o in unread] + [f"{o} (pending)" for o in pending]


def _pending_missed(inputs) -> bool:
    return (_all_pending_requested(inputs) and not _awaits_the_user(inputs.answer)
            and bool(_pending_left(inputs)))


# ---------------------------------------------------------------------------
# S6 - returns only from delivered orders
# ---------------------------------------------------------------------------


def _return_not_delivered(inputs) -> bool:
    if inputs.command_name != "return_delivered_order_items":
        return False
    status = _order_status(inputs).get(_target(inputs))
    return status is not None and status != "delivered"


# ---------------------------------------------------------------------------
# S7 - no reported cancellation that did not run
# ---------------------------------------------------------------------------

_CANCELLED_RE = re.compile(r"\bcancell?ed\b", re.I)
#: A line that reports a cancellation as NOT done is not a claim of one.
_NOT_DONE_RE = re.compile(
    r"\bnot\b|n['\u2019]t\b|\bnothing\b|\bnever\b|\bunable\b|\bfail|\bpending\b"
    r"|\bwill\b|\bcan\b",
    re.I,
)


def _claimed_cancellations(inputs) -> list[str]:
    """Order ids the answer reports as cancelled with no successful cancel.

    An order the tools already show as cancelled is not a false claim: the
    answer may be quoting its status (an order cancelled before this
    conversation).
    """
    done = {_order_in(r.args, r.response) for r in inputs.ran("cancel_pending_order")} | {
        o for o, s in _order_status(inputs).items() if s == "cancelled"}
    claimed: list[str] = []
    for line in re.split(r"[\n.;]+", inputs.answer or ""):
        if not _CANCELLED_RE.search(line) or _NOT_DONE_RE.search(line):
            continue
        for order_id in _ORDER_RE.findall(line):
            if order_id not in done and order_id not in claimed:
                claimed.append(order_id)
    return claimed


# ---------------------------------------------------------------------------
# C5 - a write that changes nothing
# ---------------------------------------------------------------------------

#: Writes that swap items for new ones; swapping an item for itself is a no-op.
_ITEM_SWAPS = ("modify_pending_order_items", "exchange_delivered_order_items")


def _writes_nothing(inputs) -> bool:
    name, args = inputs.command_name, inputs.command_args or {}
    if name == "modify_pending_order_address":
        new, old = _address_key(args), _current_address(inputs, _target(inputs))
        return new is not None and new == old
    if name == "modify_user_address":
        new, old = _address_key(args), _current_address(inputs, "@profile")
        return new is not None and new == old
    if name in _ITEM_SWAPS:
        old_ids = sorted(map(str, args.get("item_ids") or []))
        new_ids = sorted(map(str, args.get("new_item_ids") or []))
        return bool(new_ids) and old_ids == new_ids
    return False


# ---------------------------------------------------------------------------
# C6 - requested work left undone: does the request demand a command, and has
# that command run on every record the request covers?
# ---------------------------------------------------------------------------


def _requested_work(inputs) -> dict[str, dict]:
    """The firm requests that demand a command, merged per operation.

    Records are never parsed out of the reader's free-text ``detail``: it says
    where things come from as often as what to change.
    """
    work: dict[str, dict] = {}
    for request in inputs.intent:
        if not SATISFIED_BY.get(request.op) or request.conditional:
            continue
        entry = work.setdefault(request.op, {"all": False, "details": []})
        entry["all"] |= request.scope == "all"
        if request.detail:
            entry["details"].append(request.detail)
    return work


def _written_addresses(inputs) -> set[tuple]:
    return {
        key
        for r in inputs.commands_run
        if r.success and r.name in ("modify_pending_order_address", "modify_user_address")
        if (key := _address_key(r.args)) is not None
    }


def _already_holds(inputs, op: str, target: str) -> bool:
    """The requested address change is already in place on ``target``."""
    if op not in ("change_order_address", "change_profile_address"):
        return False
    current = _current_address(inputs, target)
    return current is not None and current in _written_addresses(inputs)


def _unfinished(inputs) -> list[str]:
    """For each demanding request, what no successful command has handled.

    Each entry names the request, what is left, and the command that does it.
    """
    left: list[str] = []
    status, ever = _status(inputs), _ever_status(inputs)
    for op, entry in _requested_work(inputs).items():
        commands = SATISFIED_BY[op]
        done = _succeeded_on(inputs, commands)
        on_orders = sorted(c for c in commands if c in REQUIRES_STATUS)
        if not on_orders and _already_holds(inputs, op, "@profile"):
            continue
        label = (f"'{entry['details'][-1]}'" if entry["details"]
                 else "the " + op.replace("_", " ") + " request")
        how = (", or ".join(f"{c} on a '{REQUIRES_STATUS[c]}' order" for c in on_orders)
               if on_orders else ", ".join(sorted(commands)))
        if not done:
            left.append(f"{label}: never carried out (the command is {how})")
            continue
        if not (entry["all"] and on_orders):
            continue
        if unread := _uninspected(inputs):
            left.append(f"{label}: orders {', '.join(unread)} not read, so which ones "
                        "it covers is unknown - read them with get_order_details")
            continue
        eligible = {REQUIRES_STATUS[c] for c in on_orders}
        covered = {o for o, seen in ever.items()
                   if seen & eligible and not _already_holds(inputs, op, o)}
        missing = sorted(covered - done)
        if not missing:
            continue
        text = f"{label}: not done for {', '.join(missing)}"
        possible = [o for o in missing if status.get(o) in eligible]
        closed = [f"{o} (now '{status.get(o)}')" for o in missing
                  if status.get(o) not in eligible]
        if possible:
            text += f" - still possible: {', '.join(possible)}"
        if closed:
            text += f" - no longer possible: {', '.join(closed)}, tell the customer"
        left.append(f"{text} (the command is {how})")
    return left


def _work_left_at_finish(inputs) -> bool:
    return bool(_unfinished(inputs)) and not _awaits_the_user(inputs.answer)


# ---------------------------------------------------------------------------
# S1 - the refund total the customer asked for
# ---------------------------------------------------------------------------


def _refund_total(inputs):
    """(total, breakdown) the customer gets back, from tool output alone; None
    when no refunding write succeeded or one hit an order never read."""
    total, parts = 0.0, []
    for record in inputs.commands_run:
        if not record.success or record.name not in (
                "cancel_pending_order", "return_delivered_order_items"):
            continue
        order_id = _order_in(record.args, record.response)
        order = _order_json(inputs, order_id) if order_id else None
        if not order:
            return None
        if record.name == "cancel_pending_order":
            paid = sum(float(p.get("amount", 0)) for p in order.get("payment_history", [])
                       if p.get("transaction_type") == "payment")
            total += paid
            parts.append(f"{order_id} cancelled: ${paid:,.2f}")
        else:
            wanted = {str(i) for i in record.args.get("item_ids") or []}
            prices = [float(i.get("price", 0)) for i in order.get("items", [])
                      if str(i.get("item_id")) in wanted]
            if not prices:
                return None
            total += sum(prices)
            parts.append(f"{order_id} returned items: ${sum(prices):,.2f}")
    return (round(total, 2), parts) if parts else None


def _states_amount(answer: str, amount: float) -> bool:
    for whole, cents in _MONEY_RE.findall(answer or ""):
        try:
            if abs(float(whole.replace(",", "") + (cents or "")) - amount) < 0.005:
                return True
        except ValueError:
            continue
    return False


def _refunded_this_turn(inputs) -> bool:
    """A cancel or return succeeded since the customer's last message: a later
    turn's closing does not have to repeat a total already given."""
    for record in reversed(tuple(inputs.commands_run)):
        if record.name == USER_MESSAGE:
            return False
        if record.success and record.name in (
                "cancel_pending_order", "return_delivered_order_items"):
            return True
    return False


def _total_missing(inputs) -> bool:
    if not inputs.requested("report_refund_total") or not _refunded_this_turn(inputs):
        return False
    if _awaits_the_user(inputs.answer):
        return False
    computed = _refund_total(inputs)
    return computed is not None and not _states_amount(inputs.answer, computed[0])


# ---------------------------------------------------------------------------
# D1 - the order-locking write last
# ---------------------------------------------------------------------------

#: The write that leaves its order ineligible for any further write (an item
#: change: the order cannot be modified or cancelled afterwards).
_LOCKING_WRITE = "modify_pending_order_items"


def _open_on_target(inputs) -> list[str]:
    """Firm requests still open on the locking write's target order.

    A request covers the target only when the customer said "all" (records are
    never read out of the intent's free-text detail). The request must be one
    a command needing the order still "pending" performs - other than the
    locking write itself.
    """
    if inputs.command_name != _LOCKING_WRITE:
        return []
    target = _target(inputs)
    if not target:
        return []
    open_ops = []
    for op, entry in _requested_work(inputs).items():
        commands = {c for c in SATISFIED_BY[op]
                    if c != _LOCKING_WRITE and REQUIRES_STATUS.get(c) == "pending"}
        if not commands or not entry["all"]:
            continue
        if target in _succeeded_on(inputs, commands) or _already_holds(inputs, op, target):
            continue
        open_ops.append(op)
    return open_ops


# ---------------------------------------------------------------------------
# C8 - transfer while the request is doable
# ---------------------------------------------------------------------------


def _identifiable(inputs) -> bool:
    """The account was found, or the customer gave an email not yet tried (an
    email whose lookup failed does not count)."""
    if inputs.ran("get_user_details") or any(
            r.name.startswith("find_user_id") and r.success for r in inputs.commands_run):
        return True
    given = {
        e.lower()
        for r in inputs.commands_run if r.name in USER_UTTERANCES
        for e in (m.rstrip(".") for m in _EMAIL_RE.findall(str(r.response)))
    }
    tried = {
        str(v).lower()
        for r in inputs.commands_run if r.name == "find_user_id_by_email"
        for v in (r.args or {}).values()
    }
    return bool(given - tried)


def _doable(inputs) -> list[str]:
    """Firm requests a command can still carry out: a profile change not yet in
    place, or an order change while an order it could act on remains (orders
    not listed or not all read yet, or one last seen in the status it needs)."""
    left: list[str] = []
    status = _status(inputs)
    listed = bool(inputs.ran("get_user_details"))
    unread = _uninspected(inputs)
    for op, entry in _requested_work(inputs).items():
        commands = SATISFIED_BY[op]
        done = _succeeded_on(inputs, commands)
        on_orders = [c for c in commands if c in REQUIRES_STATUS]
        if not on_orders:
            if not done and not _already_holds(inputs, op, "@profile"):
                left.append(op)
            continue
        if done and not entry["all"]:
            continue
        eligible = {REQUIRES_STATUS[c] for c in on_orders}
        open_orders = [o for o, seen in status.items()
                       if seen in eligible and o not in done
                       and not _already_holds(inputs, op, o)]
        if not listed or unread or open_orders:
            left.append(op)
    return left


def _transfer_while_doable(inputs) -> bool:
    return (inputs.command_name == "transfer_to_human_agents"
            and _identifiable(inputs) and bool(_doable(inputs)))


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

TABLE = DecisionTable(
    TABLE_VERSION,
    [
        # The self-distilled rows first, in their own order (writes: S3 S2 S6 S4
        # S8; finish: S7 S5 S1), then the pool rows: impossible (C2), pointless
        # (C5), premature - D1 before C4, so the agent is told what to do first
        # rather than asked for a yes to a write it should not make yet - then
        # C6 and the transfer guard C8.
        PolicyRow(
            row_id="S3-cancel-reason-must-be-an-accepted-value",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="cancel_pending_order rejects any reason outside its accepted values",
            source_insights=("DE1",),
            predicate=lambda i: (
                i.command_name == "cancel_pending_order"
                and "reason" in i.command_args
                and str(i.command_args["reason"]).strip().lower()
                not in ENUM_ARGUMENTS["cancel_pending_order.reason"]
            ),
            rewrite=(
                "cancel_pending_order accepts only 'no longer needed' or 'ordered "
                "by mistake' as its reason; anything else fails with 'Invalid "
                "reason'. Re-issue this same cancel_pending_order with the reason "
                "the customer gave. If they have not given one, ask them which of "
                "the two applies, and as soon as they answer, re-issue this cancel "
                "with it before doing anything else."
            ),
        ),
        PolicyRow(
            row_id="S2-refund-to-the-original-payment-method",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason=("a return's refund was aimed at a payment method that neither paid "
                    "for the order nor is a gift card"),
            source_insights=("DE4",),
            predicate=_refund_method_invalid,
            rewrite=lambda i: (
                "A return can only be refunded to the payment method the order was "
                "paid with, or to a gift card. "
                f"{i.command_args.get('payment_method_id')} is neither for order "
                f"{_target(i)}: take the method from its payment_history in "
                "get_order_details, or a gift card the customer chooses."
            ),
        ),
        PolicyRow(
            row_id="S6-return-only-from-a-delivered-order",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="a return was aimed at an order whose status is not delivered",
            source_insights=("DE11", "DP9"),
            predicate=_return_not_delivered,
            rewrite=lambda i: (
                f"Order {_target(i)} is {_order_status(i).get(_target(i))}, not "
                "delivered - it cannot be returned. The item the customer received is "
                "in a DELIVERED order: look through the items of each delivered "
                "order's get_order_details and match their description by meaning, "
                "not by exact name, then return it from that order."
            ),
        ),
        PolicyRow(
            row_id="S4-do-not-cancel-an-order-twice",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="this order was already cancelled successfully",
            source_insights=("DP2",),
            predicate=lambda i: (
                i.command_name == "cancel_pending_order"
                and bool(_target(i))
                and any(_order_in(r.args, r.response) == _target(i)
                        for r in i.ran("cancel_pending_order"))
            ),
            rewrite=lambda i: (
                f"{_target(i)} is already cancelled - the earlier cancel_pending_order "
                "succeeded. Do not cancel it again; move on to the rest of the request."
            ),
        ),
        PolicyRow(
            row_id="S8-read-every-order-before-writing-to-one",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="a write targets an order while other orders on the account are unread",
            source_insights=("DE54",),
            predicate=lambda i: (
                i.facts.is_write(i.command_name) and bool(_target(i))
                and bool(_uninspected(i))
            ),
            rewrite=lambda i: (
                "Before writing to an order, read every order on the account: "
                f"{', '.join(_uninspected(i))} not read yet. Call get_order_details on "
                "each, then choose the target from what they show - only a 'pending' "
                "order can be cancelled, only a 'delivered' one returned."
            ),
        ),
        PolicyRow(
            row_id="S7-do-not-report-a-cancellation-that-did-not-run",
            phases=_BEFORE_FINISH,
            outcome=PolicyOutcome.REWRITE,
            reason="the answer reports an order cancelled with no successful cancel",
            source_insights=("DE10",),
            predicate=lambda i: bool(_claimed_cancellations(i)),
            rewrite=lambda i: (
                "Your answer says " + ", ".join(_claimed_cancellations(i))
                + " was cancelled, but no cancel_pending_order for it succeeded in this "
                "conversation. If the customer wants it cancelled, run the cancel now "
                "and check that it succeeds; otherwise correct the answer."
            ),
        ),
        PolicyRow(
            row_id="S5-cancel-every-pending-order-when-all-were-asked",
            phases=_BEFORE_FINISH,
            outcome=PolicyOutcome.REWRITE,
            reason=("the customer asked to cancel all pending orders, and some order is "
                    "unread or still pending"),
            source_insights=("DE7", "DE8", "DP6"),
            predicate=_pending_missed,
            rewrite=lambda i: (
                "The customer asked to cancel ALL pending orders, and not all are done: "
                + ", ".join(_pending_left(i))
                + ". Read each unread order with get_order_details, and cancel every "
                "order that is pending (with the reason the customer gave) before you "
                "finish."
            ),
        ),
        PolicyRow(
            row_id="S1-report-the-refund-total-they-asked-for",
            phases=_BEFORE_FINISH,
            outcome=PolicyOutcome.REWRITE,
            reason=("the customer asked how much they will get back, the refunds are "
                    "done, and the answer does not state the total"),
            source_insights=("DE2", "DE3", "DP1", "DP3"),
            predicate=_total_missing,
            rewrite=lambda i: (
                "The customer asked how much they will get back in total, and your "
                "answer does not state it. From the tool results: "
                + "; ".join(_refund_total(i)[1])
                + f" - ${_refund_total(i)[0]:,.2f} in all. Confirm it with calculate, "
                "then tell the customer the total and its breakdown."
            ),
        ),
        PolicyRow(
            row_id="C2-write-must-match-the-order-status",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="the target order's current status does not allow this command",
            source_insights=("DE65", "DP46", "DP48"),
            predicate=lambda i: (
                i.command_name in REQUIRES_STATUS
                and _status(i).get(_target(i)) not in (None, REQUIRES_STATUS[i.command_name])
            ),
            rewrite=lambda i: (
                f"Order {_target(i)} is '{_status(i).get(_target(i))}'; "
                f"{i.command_name} only applies to a '{REQUIRES_STATUS[i.command_name]}' "
                "order and will be rejected. Pending orders take cancel_pending_order "
                "and modify_pending_order_*; delivered orders take "
                "return_delivered_order_items and exchange_delivered_order_items. An "
                "order whose items were already modified cannot be modified again. "
                "Choose the command that fits this order's status, or tell the "
                "customer it cannot be done."
            ),
        ),
        PolicyRow(
            row_id="C5-do-not-write-what-is-already-stored",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="the write's new values equal the target's current values",
            source_insights=("DE36", "DP27"),
            predicate=_writes_nothing,
            rewrite=lambda i: (
                f"This {i.command_name} swaps each item for itself - it changes "
                "nothing, and an item can only be exchanged or modified to a different "
                "option of the same product. Tell the customer the identical item "
                "cannot be swapped for itself, and ask which other option they want."
                if i.command_name in _ITEM_SWAPS else
                f"This {i.command_name} would write the values already stored - it "
                "changes nothing. Get the new values from the customer (or read back "
                "an on-file option and get an explicit yes), and make sure they differ "
                "from what is stored before writing."
            ),
        ),
        PolicyRow(
            row_id="D1-irreversible-write-last",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason=("an item change would lock the order while another requested change "
                    "on that order is still open"),
            source_insights=("DE14", "DE19", "DE28", "DE32", "DE37"),
            predicate=lambda i: bool(_open_on_target(i)),
            rewrite=lambda i: (
                f"Do not modify the items of {_target(i)} yet. Once its items are "
                "modified the order cannot be modified again, and the customer also "
                f"asked for: {', '.join(_open_on_target(i))} - not yet done on "
                f"{_target(i)}. Settle and carry that out first (confirm the details "
                "with the customer if you have not), then make this item change."
            ),
        ),
        PolicyRow(
            row_id="C4-explicit-yes-to-the-details-before-a-write",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason="a write was about to run without the customer's explicit yes",
            # Carried over from the earlier pool round as the rule several modes are
            # covered by (taxonomy.yaml: covered-by C4); no corpus item of its own.
            source_insights=(),
            predicate=lambda i: (
                i.facts.is_write(i.command_name)
                and not (i.last_user_utterance() or _NO_UTTERANCE).args.get("consents")
            ),
            rewrite=lambda i: (
                f"Do not run {i.command_name} yet - the customer has not agreed to it. "
                "First tell them exactly what will happen: "
                f"{CONFIRM_DETAILS.get(i.command_name, 'the full action details')}. "
                "Ask them to confirm, and wait for an explicit yes before calling it."
            ),
        ),
        PolicyRow(
            row_id="C6-carry-out-every-request-before-finishing",
            phases=_BEFORE_FINISH,
            outcome=PolicyOutcome.REWRITE,
            reason="the turn is closing with requested work not carried out",
            source_insights=("DE16", "DE20", "DE23", "DE25", "DE27", "DE29", "DE31", "DE33",
                             "DE42", "DE43", "DE49", "DP12", "DP15", "DP19", "DP20",
                             "DP22", "DP23", "DP25"),
            predicate=_work_left_at_finish,
            rewrite=lambda i: (
                "Before you finish, this is still not done: "
                + "; ".join(_unfinished(i))
                + ". Carry out what is still possible now (with the customer's yes "
                "where a write needs one). For anything no longer possible, tell the "
                "customer plainly what was not done and why. Do not close as if "
                "everything were done."
            ),
        ),
        PolicyRow(
            row_id="C8-no-transfer-while-the-request-is-doable",
            phases=_BEFORE_WRITE,
            outcome=PolicyOutcome.REWRITE,
            reason=("transfer to a human while the customer is identifiable and a "
                    "requested change the workflow can perform is still open"),
            source_insights=("DE47", "DP30", "DP43"),
            predicate=_transfer_while_doable,
            rewrite=lambda i: (
                "Do not transfer: this request can be completed with the workflow's "
                "commands, and the customer can be identified. Still open: "
                + "; ".join(_doable(i))
                + ". Look the account up if you have not, and carry the request out "
                "in this conversation - details the customer types here are what the "
                "commands use; there is no portal or link."
            ),
        ),
    ],
)
