"""The retail worked example's policy table (examples/retail_workflow/policy/table.py).

Each row is checked on the conversations that motivated it - where it must fire
and where a passing trial shows it must stay silent. Order reads are written as
the example workflow's get_order_details returns them ("Order details: {...}").

Two fixture sets: the S rows' (two pending orders cancelled, a watch returned,
a customer asking how much they get back) and the C/D rows' (three orders, an
address change).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fastworkflow.policy_decision import (
    USER_MESSAGE,
    BeforeFinishInput,
    BeforeSideEffectInput,
    CommandRecord,
    RequestedOperation,
)

_PATH = (
    Path(__file__).resolve().parents[1]
    / "fastworkflow" / "examples" / "retail_workflow" / "policy" / "table.py"
)
_spec = importlib.util.spec_from_file_location("retail_example_table", _PATH)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)
TABLE, FACTS = example.TABLE, example.FACTS


def test_the_example_loads_through_the_policy_loader():
    from fastworkflow.policy_decision import PolicyMode
    from fastworkflow.policy_loader import load_policy

    point, facts = load_policy(
        str(_PATH.parents[1]), mode=PolicyMode.ENFORCE, reference="policy/table.py")
    assert point.table.version == example.TABLE_VERSION == "retail-demo/0.4.1"
    assert [r.row_id.split("-")[0] for r in point.table.rows] == [
        "S3", "S2", "S6", "S4", "S8", "S7", "S5", "S1",
        "C2", "C5", "D1", "C4", "C6", "C8"]
    assert facts.is_write("cancel_pending_order")
    assert facts.reports_failure("User ID: Error: User not found")


# ===========================================================================
# S rows
# ===========================================================================

def _order(order_id, status, paid, items=()):
    body = {
        "order_id": order_id,
        "status": status,
        "items": [{"name": n, "item_id": i, "price": p} for n, i, p in items],
        "payment_history": [
            {
                "transaction_type": "payment",
                "amount": paid,
                "payment_method_id": "paypal_1",
            }
        ],
    }
    return CommandRecord(
        "get_order_details", {"order_id": order_id}, "Order details: " + json.dumps(body)
    )


S_USER = CommandRecord(
    USER_MESSAGE,
    {"consents": False},
    "Cancel my pending orders and return the watch. How much will I get back in total?",
)
S_READ = (
    S_USER,
    CommandRecord(
        "get_user_details", {"user_id": "u"}, '{"orders": ["#W1", "#W2", "#W3"]}'
    ),
    _order("#W1", "pending", 3131.10),
    _order("#W2", "pending", 4777.75),
    _order(
        "#W3",
        "delivered",
        918.95,
        [("Smart Watch", "2554056026", 367.38), ("Boots", "4127323219", 251.82)],
    ),
)
S_DONE = S_READ + (
    CommandRecord(
        "cancel_pending_order",
        {"order_id": "#W1", "reason": "no longer needed"},
        "Order #W1 cancelled successfully",
    ),
    CommandRecord(
        "cancel_pending_order",
        {"order_id": "#W2", "reason": "no longer needed"},
        "Order #W2 cancelled successfully",
    ),
    CommandRecord(
        "return_delivered_order_items",
        {
            "order_id": "#W3",
            "item_ids": ["2554056026"],
            "payment_method_id": "paypal_1",
        },
        "Response: return status is: ok",
    ),
)
S_INTENT = (
    RequestedOperation("cancel_order", scope="all"),
    RequestedOperation("return_item", detail="watch"),
    RequestedOperation("report_refund_total"),
)


def _s_fires(inputs):
    decision = TABLE.evaluate(inputs)
    return decision.source_policy if decision else None


def s_finish(answer, history=S_DONE, intent=S_INTENT):
    return BeforeFinishInput(
        facts=FACTS,
        commands_run=tuple(history),
        observations=tuple(r.response for r in history),
        intent=intent,
        answer=answer,
    )


#: The customer's explicit yes just before a write: these fixtures test the S
#: rows, so the confirmation rule (C4) is satisfied.
S_YES = CommandRecord("ask_user", {"clarification_request": "Go ahead?", "consents": True},
                      "Yes.")


def s_write(command, history=S_READ, **args):
    history = tuple(history) + (S_YES,)
    return BeforeSideEffectInput(
        facts=FACTS,
        commands_run=history,
        observations=tuple(r.response for r in history),
        intent=S_INTENT,
        command_name=command,
        command_args=args,
    )


def test_the_intent_vocabulary_declares_the_refund_total():
    assert "report_refund_total" in FACTS.intent_operations


class TestS1RefundTotal:
    def test_fires_when_only_the_returned_item_is_reported(self):
        """Only the returned item's price given as the refund."""
        decision = TABLE.evaluate(s_finish("Your refund of $367.38 is on its way."))
        assert decision.source_policy == "S1-report-the-refund-total-they-asked-for"
        assert "$8,276.23" in decision.rewrite

    def test_silent_when_the_total_is_stated(self):
        """The total is in the answer, in either format."""
        assert _s_fires(s_finish("You will get back $8,276.23 in total.")) is None
        assert _s_fires(s_finish("Total refund: 8276.23 USD.")) is None

    def test_silent_in_a_later_turn_that_did_no_refunding(self):
        later = S_DONE + (CommandRecord(USER_MESSAGE, {}, "Thanks, all good!"),)
        assert _s_fires(s_finish("You're welcome - have a great day!", later)) is None

    def test_silent_when_the_total_was_not_asked_for(self):
        intent = S_INTENT[:2]
        assert _s_fires(s_finish("All done.", intent=intent)) is None

    def test_silent_when_an_order_was_never_read(self):
        """No guessing: an unread order makes the total unknown. (The request
        to cancel all pending orders is not settled while one is unread, so C6
        speaks instead.)"""
        history = tuple(
            r
            for r in S_DONE
            if r.args.get("order_id") != "#W2" or r.name != "get_order_details"
        )
        assert _s_fires(s_finish("All done.", history)) == (
            "C6-carry-out-every-request-before-finishing")


class TestS2RefundMethod:
    def test_fires_on_a_method_that_did_not_pay_for_the_order(self):
        """A card that did not pay for the order."""
        assert _s_fires(
            s_write(
                "return_delivered_order_items",
                order_id="#W3",
                item_ids=["2554056026"],
                payment_method_id="credit_card_0000000",
            )
        ) == ("S2-refund-to-the-original-payment-method")

    def test_silent_on_the_original_method_or_a_gift_card(self):
        for method in ("paypal_1", "gift_card_9"):
            assert (
                _s_fires(
                    s_write(
                        "return_delivered_order_items",
                        order_id="#W3",
                        item_ids=["2554056026"],
                        payment_method_id=method,
                    )
                )
                is None
            )

    def test_silent_when_the_order_was_not_read(self):
        assert (
            _s_fires(
                s_write(
                    "return_delivered_order_items",
                    history=(S_USER,),
                    order_id="#W3",
                    payment_method_id="credit_card_0",
                )
            )
            is None
        )


class TestS3CancelReason:
    def test_fires_on_an_improvised_reason(self):
        """An improvised reason."""
        assert _s_fires(
            s_write("cancel_pending_order", order_id="#W1", reason="User request")
        ) == ("S3-cancel-reason-must-be-an-accepted-value")

    def test_silent_on_accepted_reasons_and_when_none_is_given(self):
        for args in (
            {"order_id": "#W1", "reason": "no longer needed"},
            {"order_id": "#W1", "reason": "Ordered by mistake"},
            {"_untagged": "#W1 no longer needed"},
        ):
            assert _s_fires(s_write("cancel_pending_order", **args)) is None


class TestS4CancelTwice:
    def test_fires_on_a_second_cancel_of_the_same_order(self):
        assert (
            _s_fires(
                s_write(
                    "cancel_pending_order",
                    history=S_DONE,
                    order_id="#W1",
                    reason="no longer needed",
                )
            )
            == "S4-do-not-cancel-an-order-twice"
        )

    def test_silent_for_another_order_or_after_a_failed_cancel(self):
        failed = S_READ + (
            CommandRecord(
                "cancel_pending_order",
                {"order_id": "#W1"},
                "Error: Invalid reason",
                success=False,
            ),
        )
        assert (
            _s_fires(
                s_write(
                    "cancel_pending_order",
                    history=failed,
                    order_id="#W1",
                    reason="no longer needed",
                )
            )
            is None
        )
        assert (
            _s_fires(
                s_write(
                    "cancel_pending_order",
                    history=S_DONE[:-2],
                    order_id="#W2",
                    reason="no longer needed",
                )
            )
            is None
        )


class TestS5EveryPendingOrder:
    def test_fires_when_a_pending_order_was_never_cancelled(self):
        """Cancelled #W1, read #W2 as pending, never cancelled it."""
        partial = S_READ + (S_DONE[5], S_DONE[7])  # cancel #W1, return
        decision = TABLE.evaluate(s_finish("All done. Total $3,498.48.", partial))
        assert (
            decision.source_policy
            == "S5-cancel-every-pending-order-when-all-were-asked"
        )
        assert "#W2 (pending)" in decision.rewrite

    def test_fires_on_an_unread_order(self):
        unread = tuple(
            r
            for r in S_DONE
            if not (
                r.name in ("get_order_details", "cancel_pending_order")
                and r.args.get("order_id") == "#W2"
            )
        )
        decision = TABLE.evaluate(s_finish("Done.", unread))
        assert "#W2 (not read yet)" in decision.rewrite

    def test_silent_when_every_pending_order_is_cancelled(self):
        assert _s_fires(s_finish("You get back $8,276.23 in total.")) is None

    def test_silent_when_all_was_not_asked(self):
        intent = (RequestedOperation("cancel_order", detail="#W1"),)
        partial = S_READ + (S_DONE[5],)
        assert _s_fires(s_finish("Done.", partial, intent)) is None


class TestS6ReturnFromDelivered:
    def test_fires_on_a_return_from_a_pending_order(self):
        """A 'wristwatch' returned from pending #W1."""
        assert _s_fires(
            s_write(
                "return_delivered_order_items",
                order_id="#W1",
                item_ids=["1994478369"],
                payment_method_id="paypal_1",
            )
        ) == ("S6-return-only-from-a-delivered-order")

    def test_silent_on_a_delivered_order_or_an_unread_one(self):
        assert (
            _s_fires(
                s_write(
                    "return_delivered_order_items",
                    order_id="#W3",
                    item_ids=["2554056026"],
                    payment_method_id="paypal_1",
                )
            )
            is None
        )
        assert (
            _s_fires(
                s_write(
                    "return_delivered_order_items",
                    history=(S_USER,),
                    order_id="#W9",
                    payment_method_id="paypal_1",
                )
            )
            is None
        )


class TestS7ClaimedCancellation:
    def test_fires_on_a_cancellation_reported_but_never_run(self):
        """'#W1 | Cancelled' while its cancel never ran."""
        history = S_READ + (S_DONE[6],)  # only #W2 cancelled
        answer = "| #W1 | Cancelled | $3,131.10 |\n| #W2 | Cancelled | $4,777.75 |"
        decision = TABLE.evaluate(s_finish(answer, history, S_INTENT[:2]))
        assert (
            decision.source_policy == "S7-do-not-report-a-cancellation-that-did-not-run"
        )
        assert "#W1" in decision.rewrite and "#W2" not in decision.rewrite

    def test_silent_when_the_cancellations_ran_or_are_reported_as_not_done(self):
        assert _s_fires(s_finish("#W1 and #W2 are cancelled; total $8,276.23.")) is None
        history = S_READ + (S_DONE[6],)
        answer = "#W2 is cancelled. #W1 could not be cancelled yet."
        assert _s_fires(s_finish(answer, history, ())) is None

    def test_silent_on_nothing_cancelled_or_a_curly_apostrophe(self):
        """'I couldn\u2019t find a charger in #W1, so nothing was cancelled there.'"""
        history = S_READ + (S_DONE[6],)
        for answer in (
            "I couldn\u2019t find a charger in order #W1, so nothing was cancelled there.",
            "Order #W1 was never cancelled.",
        ):
            assert _s_fires(s_finish(answer, history, ())) is None, answer

    def test_silent_on_an_order_the_tools_show_as_cancelled(self):
        """Quoting the tracking number of an order cancelled before the conversation."""
        history = S_READ + (S_DONE[6], _order("#W9", "cancelled", 120.0))
        answer = "Tracking number for the order you cancelled (#W9): 286422338955."
        assert _s_fires(s_finish(answer, history, ())) is None

    def test_still_fires_when_the_order_was_read_as_pending(self):
        history = S_READ + (S_DONE[6],)  # #W1 read as pending, never cancelled
        answer = "Order #W1 has been cancelled."
        assert _s_fires(s_finish(answer, history, ())) == (
            "S7-do-not-report-a-cancellation-that-did-not-run"
        )


# ===========================================================================
# C and D rows
# ===========================================================================

OLD = {
    "address1": "1 Old St",
    "address2": "",
    "city": "A",
    "state": "TX",
    "zip": "11111",
}
NEW = {
    "address1": "2 New St",
    "address2": "",
    "city": "B",
    "state": "DC",
    "zip": "20001",
}


def order(oid, status, address=OLD, paid=100.0):
    body = {
        "order_id": oid,
        "status": status,
        "address": address,
        "items": [{"item_id": "1000001", "price": 10.0}],
        "payment_history": [
            {
                "transaction_type": "payment",
                "amount": paid,
                "payment_method_id": "paypal_1",
            }
        ],
    }
    return CommandRecord(
        "get_order_details", {"order_id": oid}, "Order details: " + json.dumps(body)
    )


USER = CommandRecord(
    "get_user_details",
    {"user_id": "u_1"},
    json.dumps({"address": OLD, "orders": ["#W1", "#W2", "#W3"]}),
)
YES = CommandRecord(
    "ask_user",
    {
        "clarification_request": (
            f"Go ahead and change #W1 to {NEW['address1']} {NEW['zip']}?"
        ),
        "consents": True,
    },
    "Yes.",
)
READ = (
    USER,
    order("#W1", "pending"),
    order("#W2", "pending"),
    order("#W3", "delivered"),
)


def ok(name, **args):
    return CommandRecord(name, args, "Response: Modified details: ok")


def ask(op, detail="", scope="unspecified", conditional=False):
    return RequestedOperation(
        op=op, detail=detail, scope=scope, conditional=conditional
    )


def write(command, history=READ + (YES,), intent=(), **args):
    return BeforeSideEffectInput(
        facts=FACTS,
        commands_run=tuple(history),
        observations=tuple(r.response for r in history),
        intent=tuple(intent),
        command_name=command,
        command_args=args,
    )


def finish(answer, history, intent):
    return BeforeFinishInput(
        facts=FACTS,
        commands_run=tuple(history),
        observations=tuple(r.response for r in history),
        intent=tuple(intent),
        answer=answer,
    )


def fires(inputs):
    decision = TABLE.evaluate(inputs)
    return decision.source_policy.split("-")[0] if decision else None


class TestC6TwoSteps:
    def test_step1_a_request_with_no_command_demands_nothing(self):
        intent = [ask("remove_item", "the jacket"), ask("report_refund_total")]
        assert fires(finish("Done.", READ, intent)) is None

    def test_step1_a_conditional_request_demands_nothing(self):
        intent = [ask("change_order_address", "if it is wrong", conditional=True)]
        assert fires(finish("Done.", READ, intent)) is None

    def test_step2_a_firm_request_never_carried_out(self):
        intent = [ask("change_profile_address")]
        decision = TABLE.evaluate(finish("All set.", READ, intent))
        assert decision.source_policy.startswith("C6")
        assert "never carried out" in decision.rewrite

    def test_step2_all_covers_every_eligible_record(self):
        """One of the pending orders skipped."""
        history = READ + (
            YES,
            ok("modify_pending_order_address", order_id="#W1", **NEW),
        )
        intent = [ask("change_order_address", "all pending orders", scope="all")]
        decision = TABLE.evaluate(finish("All updated.", history, intent))
        assert "#W2" in decision.rewrite and "#W3" not in decision.rewrite  # delivered

    def test_step2_all_with_unread_records_is_not_done(self):
        history = (
            USER,
            order("#W1", "pending"),
            YES,
            ok("modify_pending_order_address", order_id="#W1", **NEW),
        )
        intent = [ask("change_order_address", scope="all")]
        assert "not read" in TABLE.evaluate(finish("Done.", history, intent)).rewrite

    def test_records_are_not_read_from_the_readers_free_text(self):
        """'the address on file for order #W2' says where the address comes
        from, not an order to change (a passing trial)."""
        history = READ + (YES, ok("modify_user_address", user_id="u_1", **NEW))
        intent = [ask("change_profile_address", "the address on file for order #W2")]
        assert fires(finish("All done.", history, intent)) is None

    def test_a_change_already_in_place_demands_nothing(self):
        """The profile already held the new address (a passing trial)."""
        user_new = CommandRecord(
            "get_user_details",
            {"user_id": "u_1"},
            json.dumps({"address": NEW, "orders": ["#W1", "#W2", "#W3"]}),
        )
        history = (
            user_new,
            order("#W1", "pending"),
            order("#W2", "pending", NEW),
            order("#W3", "delivered"),
            YES,
            ok("modify_pending_order_address", order_id="#W1", **NEW),
        )
        intent = [
            ask("change_order_address", "the order at the old address"),
            ask("change_profile_address"),
        ]
        assert fires(finish("All done.", history, intent)) is None

    def test_all_skips_an_order_already_at_the_new_address(self):
        history = (
            USER,
            order("#W1", "pending"),
            order("#W2", "pending", NEW),
            order("#W3", "delivered"),
            YES,
            ok("modify_pending_order_address", order_id="#W1", **NEW),
        )
        intent = [ask("change_order_address", scope="all")]
        assert fires(finish("All updated.", history, intent)) is None

    def test_silent_when_everything_was_done_or_the_answer_waits(self):
        history = (
            READ
            + (YES,)
            + tuple(
                ok("modify_pending_order_address", order_id=o, **NEW)
                for o in ("#W1", "#W2")
            )
        )
        intent = [ask("change_order_address", scope="all")]
        assert fires(finish("All updated.", history, intent)) is None
        assert (
            fires(
                finish(
                    "Shall I also update your profile?",
                    READ,
                    [ask("change_profile_address")],
                )
            )
            is None
        )

    def test_a_cancelled_order_counts_as_handled_for_all(self):
        history = (
            READ
            + (YES,)
            + tuple(
                ok("cancel_pending_order", order_id=o, reason="no longer needed")
                for o in ("#W1", "#W2")
            )
        )
        assert (
            fires(
                finish("Both cancelled.", history, [ask("cancel_order", scope="all")])
            )
            is None
        )


class TestC2Status:
    def test_return_from_a_pending_order(self):
        # S6 states the same fact first (returns only from delivered orders).
        assert (
            fires(
                write(
                    "return_delivered_order_items",
                    order_id="#W1",
                    payment_method_id="paypal_1",
                )
            )
            == "S6"
        )

    def test_a_second_item_modification(self):
        history = READ + (
            YES,
            ok(
                "modify_pending_order_items",
                order_id="#W1",
                item_ids=["1000001"],
                new_item_ids=["2000002"],
            ),
        )
        assert (
            fires(
                write(
                    "modify_pending_order_items",
                    history,
                    order_id="#W1",
                    item_ids=["1000001"],
                    new_item_ids=["3000003"],
                )
            )
            == "C2"
        )

    def test_an_address_change_keeps_the_order_modifiable(self):
        history = READ + (
            YES,
            ok("modify_pending_order_address", order_id="#W1", **NEW),
        )
        assert (
            fires(
                write(
                    "modify_pending_order_payment",
                    history,
                    order_id="#W1",
                    payment_method_id="credit_card_2",
                )
            )
            != "C2"
        )

    def test_silent_on_an_unread_order(self):
        assert (
            fires(
                write(
                    "cancel_pending_order",
                    (USER, YES),
                    order_id="#W9",
                    reason="no longer needed",
                )
            )
            != "C2"
        )


class TestC5NoOp:
    def test_writing_the_stored_order_address(self):
        assert (
            fires(write("modify_pending_order_address", order_id="#W1", **OLD)) == "C5"
        )

    def test_writing_the_stored_profile_address(self):
        assert fires(write("modify_user_address", user_id="u_1", **OLD)) == "C5"

    def test_swapping_an_item_for_itself(self):
        assert (
            fires(
                write(
                    "modify_pending_order_items",
                    order_id="#W1",
                    item_ids=["1000001"],
                    new_item_ids=["1000001"],
                )
            )
            == "C5"
        )

    def test_exchanging_an_item_for_itself(self):
        """An item exchanged for the identical item."""
        decision = TABLE.evaluate(
            write(
                "exchange_delivered_order_items",
                order_id="#W3",  # delivered in READ
                item_ids=["1000001"],
                new_item_ids=["1000001"],
            )
        )
        assert decision.source_policy.startswith("C5")
        assert "different option" in decision.rewrite

    def test_exchanging_for_another_option_is_not_a_no_op(self):
        assert (
            fires(
                write(
                    "exchange_delivered_order_items",
                    order_id="#W3",
                    item_ids=["1000001"],
                    new_item_ids=["1000002"],
                )
            )
            is None
        )

    def test_a_real_change_is_not_a_no_op(self):
        assert (
            fires(write("modify_pending_order_address", order_id="#W1", **NEW)) is None
        )


class TestS8S3C4:
    # C1 and C3 are not in this table: S8 and S3 are the same checks.
    def test_s8_write_before_reading_every_order(self):
        history = (USER, order("#W1", "pending"), YES)
        assert (
            fires(
                write(
                    "cancel_pending_order",
                    history,
                    order_id="#W1",
                    reason="no longer needed",
                )
            )
            == "S8"
        )

    def test_s8_profile_write_is_not_an_order_write(self):
        history = (USER, YES)
        assert (
            fires(write("modify_user_address", history, user_id="u_1", **NEW)) is None
        )

    def test_c4_an_explicit_yes_is_enough(self):
        # The yes is not checked against the details the agent proposed.
        bare = CommandRecord(
            "ask_user", {"clarification_request": "Go ahead?", "consents": True}, "Yes."
        )
        assert (
            fires(write("modify_user_address", (USER, bare), user_id="u_1", **NEW))
            is None
        )

    def test_s3_invalid_reason(self):
        assert (
            fires(write("cancel_pending_order", order_id="#W1", reason="user request"))
            == "S3"
        )

    def test_c4_needs_a_yes(self):
        history = READ + (
            CommandRecord(USER_MESSAGE, {"consents": False}, "Cancel it"),
        )
        decision = TABLE.evaluate(
            write(
                "cancel_pending_order",
                history,
                order_id="#W1",
                reason="no longer needed",
            )
        )
        assert (
            decision.source_policy.startswith("C4") and "EVERY item" in decision.rewrite
        )


class TestC8Transfer:
    INTENT = [ask("change_order_address", scope="all"), ask("change_profile_address")]

    def test_blocks_a_transfer_when_identifiable_and_work_is_open(self):
        """Email given, nothing looked up, transferred for a 'secure portal'."""
        history = (CommandRecord(USER_MESSAGE, {}, "my email is a.b@example.com"),)
        decision = TABLE.evaluate(
            write(
                "transfer_to_human_agents",
                history,
                self.INTENT,
                summary="wants a secure link",
            )
        )
        assert decision.source_policy.startswith("C8")

    def test_allows_a_transfer_for_a_customer_who_cannot_be_identified(self):
        history = (CommandRecord(USER_MESSAGE, {}, "I won't share my name or email"),)
        assert (
            fires(
                write(
                    "transfer_to_human_agents",
                    history,
                    self.INTENT,
                    summary="refuses to identify",
                )
            )
            is None
        )

    def test_allows_a_transfer_after_the_only_email_failed_lookup(self):
        """Email not found, no ZIP - not identifiable."""
        history = (
            # trailing full stop, as the customer actually wrote it
            CommandRecord(
                USER_MESSAGE, {}, "I gave you my email—x.y@example.com. Fine."
            ),
            CommandRecord(
                "find_user_id_by_email",
                {"email": "x.y@example.com"},
                "User ID: Error: User not found",
                success=False,
            ),
        )
        assert (
            fires(
                write(
                    "transfer_to_human_agents",
                    history,
                    self.INTENT,
                    summary="cannot find account",
                )
            )
            is None
        )

    def test_allows_a_transfer_when_the_requested_order_is_closed(self):
        """The only open request is a cancel, and the account's only order is
        already 'pending (items modified)' - no command can cancel it now."""
        user = CommandRecord(
            "get_user_details",
            {"user_id": "u_1"},
            json.dumps({"address": OLD, "orders": ["#W1"]}),
        )
        history = (user, order("#W1", "pending (items modified)"))
        assert (
            fires(
                write(
                    "transfer_to_human_agents",
                    history,
                    [ask("cancel_order")],
                    summary="wants the modified order cancelled",
                )
            )
            is None
        )

    def test_blocks_a_transfer_when_the_requested_order_is_still_pending(self):
        user = CommandRecord(
            "get_user_details",
            {"user_id": "u_1"},
            json.dumps({"address": OLD, "orders": ["#W1"]}),
        )
        history = (user, order("#W1", "pending"))
        decision = TABLE.evaluate(
            write(
                "transfer_to_human_agents",
                history,
                [ask("cancel_order")],
                summary="wants a cancel",
            )
        )
        assert decision.source_policy.startswith("C8")

    def test_allows_a_transfer_once_nothing_is_open(self):
        history = (
            READ
            + (YES, ok("modify_user_address", user_id="u_1", **NEW))
            + tuple(
                ok("modify_pending_order_address", order_id=o, **NEW)
                for o in ("#W1", "#W2")
            )
        )
        assert (
            fires(
                write(
                    "transfer_to_human_agents",
                    history,
                    self.INTENT,
                    summary="other question",
                )
            )
            is None
        )


class TestD1LockingWriteLast:
    INTENT = [ask("change_order_address", "all my pending orders", scope="all"),
              ask("swap_item", "the jacket")]

    def test_an_item_change_waits_for_an_open_change_on_the_same_order(self):
        decision = TABLE.evaluate(write(
            "modify_pending_order_items", intent=self.INTENT, order_id="#W1",
            item_ids=["1000001"], new_item_ids=["2000002"], payment_method_id="paypal_1"))
        assert decision.source_policy.startswith("D1")
        assert "change_order_address" in decision.rewrite and "#W1" in decision.rewrite

    def test_silent_once_the_other_change_is_done_on_that_order(self):
        history = READ + (YES, ok("modify_pending_order_address", order_id="#W1", **NEW))
        assert fires(write(
            "modify_pending_order_items", history, self.INTENT, order_id="#W1",
            item_ids=["1000001"], new_item_ids=["2000002"],
            payment_method_id="paypal_1")) is None

    def test_silent_when_the_other_request_does_not_cover_every_order(self):
        intent = [ask("change_order_address", "the order to my office"),
                  ask("swap_item", "the jacket")]
        assert fires(write(
            "modify_pending_order_items", intent=intent, order_id="#W1",
            item_ids=["1000001"], new_item_ids=["2000002"],
            payment_method_id="paypal_1")) is None
