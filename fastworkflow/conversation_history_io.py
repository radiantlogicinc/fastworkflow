"""Shared helpers for dspy.History <-> JSON-serializable turns."""

from __future__ import annotations

from typing import Any

import dspy


# Two keys, not three. The third was `feedback`, carried here so that whatever
# a caller had posted to the old `/post_feedback` was replayed into the agent's
# `dspy.History` on its next turn -- `_refine_user_query` renders every key of
# every remembered turn into the refiner's prompt. fix-9eg.16 removed the table
# that fed it and the route that wrote it, so this key could only ever be None,
# and a `feedback: None` line in a prompt is noise that reads like a fact.
#
# Recorded review notes are a different thing and stay a different thing: they
# are append-only, categorized, anchored to evidence, and no part of this build
# feeds them back into a model.
def extract_turns_from_history(
    conversation_history: dspy.History,
) -> list[dict[str, Any]]:
    return [
        {
            "conversation summary": msg_dict.get("conversation summary"),
            "conversation_traces": msg_dict.get("conversation_traces"),
        }
        for msg_dict in conversation_history.messages
    ]


def restore_history_from_turns(turns: list[dict[str, Any]]) -> dspy.History:
    messages = [
        {
            "conversation summary": turn.get("conversation summary"),
            "conversation_traces": turn.get("conversation_traces"),
        }
        for turn in turns
    ]
    return dspy.History(messages=messages)
