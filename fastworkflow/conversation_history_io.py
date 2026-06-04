"""Shared helpers for dspy.History <-> JSON-serializable turns."""

from __future__ import annotations

from typing import Any

import dspy


def extract_turns_from_history(
    conversation_history: dspy.History,
) -> list[dict[str, Any]]:
    return [
        {
            "conversation summary": msg_dict.get("conversation summary"),
            "conversation_traces": msg_dict.get("conversation_traces"),
            "feedback": msg_dict.get("feedback"),
        }
        for msg_dict in conversation_history.messages
    ]


def restore_history_from_turns(turns: list[dict[str, Any]]) -> dspy.History:
    messages = [
        {
            "conversation summary": turn.get("conversation summary"),
            "conversation_traces": turn.get("conversation_traces"),
            "feedback": turn.get("feedback"),
        }
        for turn in turns
    ]
    return dspy.History(messages=messages)
