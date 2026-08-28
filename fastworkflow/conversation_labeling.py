"""Conversation topic/summary generation (Phase 7 §2.5).

Moved out of the since-deleted ``run_fastapi_mcp/conversation_store.py``
unchanged when the observability DB became the single source of truth for
conversations. Only the LLM call and its timeout plumbing live here; where the
label is written, and which topic wins a collision, are the store's business
(``ObservabilityStore.record_conversation_label``).
"""

import json
import os
from typing import Any

import dspy

import fastworkflow
from fastworkflow.utils.dspy_utils import get_lm

# Topic generation is the only LLM call this module makes, and async callers run
# it in an executor so it does not block the event loop. That offload is only
# safe if the call is bounded at the *client*. asyncio.wait_for around
# loop.run_in_executor cancels the await, not the thread: the litellm request
# keeps running, and because this repo uses the default executor everywhere, the
# orphan is a default-executor worker. Python 3.13 closes a loop with
# shutdown_default_executor(asyncio.constants.THREAD_JOIN_TIMEOUT), measured at
# 300 s here, so an unbounded request can hold process exit for five minutes -
# worse than the in-loop call it replaced. Passing timeout= to dspy.LM puts the
# deadline on the request itself (dspy merges it into the litellm kwargs), so the
# thread ends on its own whether or not anyone is still waiting for the result.
TOPIC_GENERATION_TIMEOUT_ENV_VAR = "LLM_CONVERSATION_STORE_TIMEOUT_SECONDS"

# Per attempt, in seconds. Two things set the value. Upper bound: litellm retries
# a timed-out request, so the wall-clock worst case is this times the attempt
# count below, plus about a second of backoff - 25 s, which still fits inside the
# 30 s shutdown drain in __main__.lifespan, so a rotate in flight when the
# process is asked to stop cannot outlive the drain. Lower bound: a topic plus a
# summary over a handful of turn summaries typically answers in a few seconds, so
# 12 s leaves several times the usual latency before ordinary provider slowness
# turns into a failed rotate. gh-65 suggested 2-5 s; that is inside the normal
# latency range for this call and would fail rotates that were merely slow.
DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS = 12.0

# dspy.LM defaults to num_retries=3. Four attempts of the timeout above plus
# backoff is roughly 55 s, which is outside the drain, so the retry count is part
# of the bound and is pinned here. One retry still absorbs a transient rate limit
# or reset connection on what is a user-facing request.
TOPIC_GENERATION_MAX_RETRIES = 1


def resolve_topic_generation_timeout() -> tuple[float, str]:
    """Resolve the per-attempt deadline with the process environment taking precedence.

    OS first, for the same reason resolve_max_live_sessions does it (utils.py):
    ``get_env_var`` returns a supplied default *before* consulting ``os.environ``,
    so passing a default straight to it would make the container variable
    unreachable. This is an operator control on a server — the person who needs
    to raise it against a slow provider, or lower it under a tighter termination
    grace period, sets it on the deployment, not in the workflow's env file.

    Returns the value and where it came from, because an operator has to be able
    to see whether their override actually took effect.
    """
    raw = os.environ.get(TOPIC_GENERATION_TIMEOUT_ENV_VAR)
    source = "process environment"
    if raw is None or raw == "":
        raw = fastworkflow.get_env_var(TOPIC_GENERATION_TIMEOUT_ENV_VAR, default=None)
        source = "workflow env file"
    if raw is None or raw == "":
        return DEFAULT_TOPIC_GENERATION_TIMEOUT_SECONDS, "default"

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{TOPIC_GENERATION_TIMEOUT_ENV_VAR}={raw!r} (from {source}) is not a number"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"{TOPIC_GENERATION_TIMEOUT_ENV_VAR}={value} (from {source}) must be "
            "greater than zero"
        )
    return value, source


def topic_generation_timeout_seconds() -> float:
    """Per-attempt client-side deadline for the topic/summary LLM call."""
    return resolve_topic_generation_timeout()[0]


def generate_topic_and_summary(turns: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Generate topic and summary for a conversation using DSPy.

    Only passes conversation summaries (not verbose traces) to the AI model
    for better quality topic/summary generation.

    Blocking and synchronous, so an async caller must run it in an executor. It
    is bounded at the client rather than by the caller's await (see
    TOPIC_GENERATION_TIMEOUT_ENV_VAR above), which is what makes that offload
    safe: the thread finishes on its own even if nobody is waiting any more.

    Raises on timeout or transport failure; it never substitutes an empty topic
    for a failed generation. Whether a missing label is fatal is the caller's
    decision - /new_conversation treats it as fatal, the store treats a blank
    topic as "not generated yet" - and neither can tell the difference if this
    function swallows the error.
    """
    class TopicSummarySignature(dspy.Signature):
        """Generate a concise topic and summary for a conversation"""
        conversation_turns: str = dspy.InputField(desc="JSON representation of conversation turns")
        topic: str = dspy.OutputField(desc="Short topic (3-6 words)")
        summary: str = dspy.OutputField(desc="Brief summary paragraph")

    # Extract only summaries for topic/summary generation (not verbose traces)
    summaries_only = [
        {"conversation summary": turn.get("conversation summary", "")}
        for turn in turns
    ]
    turns_str = json.dumps(summaries_only, indent=2)

    # Configure DSPy with the conversation store LM using context manager
    lm = get_lm(
        "LLM_CONVERSATION_STORE",
        "LITELLM_API_KEY_CONVERSATION_STORE",
        timeout=topic_generation_timeout_seconds(),
        num_retries=TOPIC_GENERATION_MAX_RETRIES,
    )
    with dspy.context(lm=lm):
        generator = dspy.ChainOfThought(TopicSummarySignature)
        result = generator(conversation_turns=turns_str)
        return result.topic, result.summary
