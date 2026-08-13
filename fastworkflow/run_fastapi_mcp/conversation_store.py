"""
Conversation persistence layer for FastWorkflow
Provides SQLite-backed storage for multi-turn conversations with AI-generated topics/summaries
"""

import json
import os
import time
from typing import Any, Optional

import dspy
from pydantic import BaseModel

import fastworkflow
from fastworkflow.kvstore import KVStore
from fastworkflow.utils.logging import logger
from fastworkflow.utils.dspy_utils import get_lm


from fastworkflow.conversation_history_io import (
    extract_turns_from_history,
    restore_history_from_turns,
)


class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: int
    topic: str
    summary: str
    created_at: int
    updated_at: int


class ConversationStore:
    """SQLite-backed conversation persistence per user.

    Turns are stored one key per turn (``conv:{id}:turn:{index}``) so an
    incremental save writes only the new turns instead of rewriting the whole
    conversation. The ``conv:{id}`` record keeps the metadata plus
    ``appended_turn_count``; reads rehydrate ``turns`` so callers see the same
    record shape as before. Pre-migration ``.rdb`` (RocksDB) files are abandoned;
    this store uses ``{channel_id}.sqlite3`` only.
    """
    
    def __init__(self, channel_id: str, base_folder: str):
        self.channel_id = channel_id
        self.db_path = os.path.join(base_folder, f"{channel_id}.sqlite3")
        os.makedirs(base_folder, exist_ok=True)
    
    def _get_db(self) -> KVStore:
        """Get KVStore instance"""
        return KVStore(self.db_path)

    @staticmethod
    def _turn_key(conversation_id: int, index: int) -> str:
        return f"conv:{conversation_id}:turn:{index}"

    def _iter_turn_records(
        self, db: KVStore, conversation_id: int, conv: dict[str, Any]
    ):
        """Yield a conversation's turns in order, one at a time."""
        for index in range(int(conv.get("appended_turn_count") or 0)):
            turn_key = self._turn_key(conversation_id, index)
            if turn_key in db:
                yield db[turn_key]

    def _read_turns(
        self, db: KVStore, conversation_id: int, conv: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Rehydrate a conversation's turns."""
        return list(self._iter_turn_records(db, conversation_id, conv))

    def _read_turn_window(
        self,
        db: KVStore,
        conversation_id: int,
        conv: dict[str, Any],
        max_turns: int,
    ) -> list[dict[str, Any]]:
        """The newest ``max_turns`` turns, reading only those.

        Turns are keyed by index and the count is on the record, so the newest
        window is a direct read of the last few keys rather than a scan. Nothing
        older than the window is ever deserialized, which is the whole point: a
        turn record carries its full payload, so materializing turns that are
        about to be discarded costs the size of the whole conversation.
        """
        count = int(conv.get("appended_turn_count") or 0)
        return [
            db[turn_key]
            for index in range(max(0, count - max_turns), count)
            if (turn_key := self._turn_key(conversation_id, index)) in db
        ]

    def _hydrated(
        self, db: KVStore, conversation_id: int, conv: dict[str, Any]
    ) -> dict[str, Any]:
        """The stored record as callers expect it: turns rehydrated, bookkeeping hidden."""
        hydrated = {k: v for k, v in conv.items() if k != "appended_turn_count"}
        hydrated["turns"] = self._read_turns(db, conversation_id, conv)
        return hydrated

    def _replace_turns(
        self,
        db: KVStore,
        conversation_id: int,
        conv: dict[str, Any],
        turns: list[dict[str, Any]],
    ) -> None:
        """Overwrite a conversation's turns, updating ``conv`` in place (caller stores it).

        Writes the replacements before deleting the tail they displace, so a
        crash part-way leaves a readable conversation rather than holes where the
        count still points at deleted entries.
        """
        previous_count = int(conv.get("appended_turn_count") or 0)
        for index, turn in enumerate(turns):
            db[self._turn_key(conversation_id, index)] = turn
        for index in range(len(turns), previous_count):
            turn_key = self._turn_key(conversation_id, index)
            if turn_key in db:
                del db[turn_key]
        conv["turns"] = []
        conv["appended_turn_count"] = len(turns)
    
    def get_last_conversation_id(self) -> Optional[int]:
        """Get the last conversation ID for this user"""
        with self._get_db() as db:
            meta = db.get("meta", {})
            return meta.get("last_conversation_id")
    
    def _increment_conversation_id(self, db: KVStore) -> int:
        """Increment and return new conversation ID"""
        meta = db.get("meta", {"last_conversation_id": 0})
        new_id = meta["last_conversation_id"] + 1
        meta["last_conversation_id"] = new_id
        db["meta"] = meta
        return new_id
    
    def reserve_next_conversation_id(self) -> int:
        """Reserve the next conversation ID by incrementing the counter without creating a conversation"""
        with self._get_db() as db:
            return self._increment_conversation_id(db)
    
    def _ensure_unique_topic(
        self,
        db: KVStore,
        candidate_topic: str,
        exclude_conversation_id: Optional[int] = None,
    ) -> str:
        """Ensure topic is unique per user with case/whitespace insensitive comparison

        ``exclude_conversation_id`` is the conversation whose topic is being
        written. Its own stored topic must stay out of the collision set,
        otherwise rewriting a conversation's existing topic collides with itself
        and gets a numeric suffix - and the next rewrite takes it off again.

        A blank candidate - empty or whitespace-only - comes back as "" and is
        never suffixed. Blank is the placeholder every unlabeled conversation
        carries, so uniquifying it collides with all of them and yields ' 1',
        which is neither a real title nor the blank "no title yet" sentinel the
        lazy-fill triggers watch for (fix-dzs.3). Blank is therefore exempt from
        uniqueness, and the exemption is decided before the scan below so the
        blank case costs no reads at all.

        Costs one indexed read per conversation on the channel, so a channel that
        accumulates thousands of conversations pays that on every topic write.
        """
        # Normalize for comparison
        normalized_candidate = candidate_topic.lower().strip()
        
        if not normalized_candidate:
            return ""
        
        # Get all existing topics
        existing_topics = []
        meta = db.get("meta", {"last_conversation_id": 0})
        for i in range(1, meta.get("last_conversation_id", 0) + 1):
            if i == exclude_conversation_id:
                continue
            conv_key = f"conv:{i}"
            if conv_key in db:
                conv = db[conv_key]
                existing_topics.append(conv.get("topic", ""))
        
        # Check for collision
        collision_count = 0
        final_topic = candidate_topic
        while any(final_topic.lower().strip() == t.lower().strip() for t in existing_topics):
            collision_count += 1
            final_topic = f"{candidate_topic} {collision_count}"
        
        return final_topic
    
    def save_conversation(
        self,
        topic: str,
        summary: str,
        turns: list[dict[str, Any]],
        conversation_id: Optional[int] = None
    ) -> int:
        """
        Save a conversation and return its ID.
        
        Args:
            topic: Conversation topic
            summary: Conversation summary
            turns: List of conversation turns
            conversation_id: Optional specific ID to use. If None, increments to get next ID.
        
        Returns:
            The conversation ID used
        """
        with self._get_db() as db:
            if conversation_id is not None:
                # Use the specified ID (assumes it's valid and reserved)
                conv_id = conversation_id
            else:
                # Increment to get next ID
                conv_id = self._increment_conversation_id(db)
            
            # conv_id is the record being written, so it cannot collide with
            # itself. A freshly incremented id has no record yet, so excluding it
            # is a no-op there; it matters when the caller named an existing one.
            unique_topic = self._ensure_unique_topic(db, topic, exclude_conversation_id=conv_id)
            
            conversation = {
                "topic": unique_topic,
                "summary": summary,
                "created_at": int(time.time() * 1000),
                "updated_at": int(time.time() * 1000),
                "turns": []
            }
            self._replace_turns(db, conv_id, conversation, turns)
            db[f"conv:{conv_id}"] = conversation
            return conv_id
    
    def get_conversation(self, conv_id: int) -> Optional[dict[str, Any]]:
        """Get a conversation by ID, with every turn.

        Callers that only want the newest window want get_conversation_window()
        instead; see its docstring for why the difference is not cosmetic.
        """
        with self._get_db() as db:
            conv = db.get(f"conv:{conv_id}")
            return None if conv is None else self._hydrated(db, conv_id, conv)

    def get_conversation_window(
        self, conv_id: int, max_turns: int
    ) -> Optional[dict[str, Any]]:
        """A conversation by ID carrying only its newest ``max_turns`` turns.

        Same record shape as get_conversation(), so callers are interchangeable
        apart from how much of the conversation comes back.

        This exists because the restore paths keep a bounded window of history in
        memory and used to obtain it by hydrating the whole conversation and
        slicing the rest away. MAX_CONVERSATION_TURNS_IN_MEMORY bounded what they
        KEPT, not what they read, so at 450 KB payloads restoring a long
        conversation was tens of megabytes resident to end up with twenty turns.

        Returns None when there is no record at all, which is what lets a caller
        tell "no such conversation" from "a conversation with no turns yet" - the
        reserved-but-empty state a rotate leaves behind, and the case the cold
        restore path falls back on.
        """
        with self._get_db() as db:
            conv = db.get(f"conv:{conv_id}")
            if conv is None:
                return None
            windowed = {k: v for k, v in conv.items() if k != "appended_turn_count"}
            windowed["turns"] = self._read_turn_window(
                db, conv_id, conv, max_turns
            )
            return windowed
    
    def get_conversation_by_topic(self, topic: str) -> Optional[tuple[int, dict[str, Any]]]:
        """Get conversation ID and data by topic (case/whitespace insensitive)"""
        with self._get_db() as db:
            meta = db.get("meta", {"last_conversation_id": 0})
            normalized_topic = topic.lower().strip()
            
            for i in range(1, meta.get("last_conversation_id", 0) + 1):
                conv_key = f"conv:{i}"
                if conv_key in db:
                    conv = db[conv_key]
                    if conv.get("topic", "").lower().strip() == normalized_topic:
                        return i, self._hydrated(db, i, conv)
            return None
    
    def list_conversations(self, limit: int) -> list[ConversationSummary]:
        """List conversations ordered by updated_at desc, up to limit"""
        with self._get_db() as db:
            meta = db.get("meta", {"last_conversation_id": 0})
            conversations = []
            
            for i in range(1, meta.get("last_conversation_id", 0) + 1):
                conv_key = f"conv:{i}"
                if conv_key in db:
                    conv = db[conv_key]
                    conversations.append(
                        ConversationSummary(
                            conversation_id=i,
                            topic=conv.get("topic", ""),
                            summary=conv.get("summary", ""),
                            created_at=conv.get("created_at", 0),
                            updated_at=conv.get("updated_at", 0)
                        )
                    )
            
            # Sort by updated_at desc and limit
            conversations.sort(key=lambda c: c.updated_at, reverse=True)
            return conversations[:limit]
    
    def update_conversation(
        self,
        conv_id: int,
        topic: str,
        summary: str,
        turns: list[dict[str, Any]]
    ) -> None:
        """Update an existing conversation with new topic, summary, and turns"""
        with self._get_db() as db:
            conv_key = f"conv:{conv_id}"
            if conv_key not in db:
                raise ValueError(f"Conversation {conv_id} not found")
            
            conv = db[conv_key]
            # exclude_conversation_id for the same reason as its two siblings: the
            # record being written must stay out of its own collision set, or
            # rewriting a conversation's existing topic suffixes it and the next
            # rewrite takes the suffix off again.
            unique_topic = self._ensure_unique_topic(
                db, topic, exclude_conversation_id=conv_id
            )
            
            # Preserve created_at, update other fields
            conv["topic"] = unique_topic
            conv["summary"] = summary
            conv["updated_at"] = int(time.time() * 1000)
            self._replace_turns(db, conv_id, conv, turns)
            
            db[conv_key] = conv
    
    def update_conversation_topic_summary(
        self,
        conv_id: int,
        topic: str,
        summary: str
    ) -> None:
        """
        Update only the topic and summary of an existing conversation.
        Used when finalizing a conversation (turns already saved incrementally).

        A blank topic - empty or whitespace-only - means generation failed: a
        partial JSON parse, a truncated completion, or a model that emitted
        nothing for that field. It is not written as a title. The stored topic is
        left as it stands, which for an unlabeled conversation is exactly "", the
        sentinel that makes the next eligible trigger regenerate it; a
        conversation that already carries a good title keeps it rather than losing
        it to one failed refresh.

        The summary and updated_at are still written in that case. A summary is
        useful without a title, storing it does not disturb the blank topic that
        preserves the retry, and updated_at has to move because the record did
        change - it is what orders list_conversations().

        LIMITATION (accepted here, fix-dzs.3): "" means both "never generated" and
        "generated and failed", so there is no way to stop retrying a conversation
        the model cannot title, and no way for /admin/dump_all_conversations to
        tell the two apart. Tolerable while every trigger is user-initiated and
        therefore self-limiting; separating them would need a new field in the
        stored record.
        """
        with self._get_db() as db:
            conv_key = f"conv:{conv_id}"
            if conv_key not in db:
                raise ValueError(f"Conversation {conv_id} not found")
            
            conv = db[conv_key]
            
            # Only update topic, summary, and timestamp - preserve turns
            if topic.strip():
                conv["topic"] = self._ensure_unique_topic(
                    db, topic, exclude_conversation_id=conv_id
                )
            else:
                logger.warning(
                    f"Blank topic generated for conversation {conv_id} on channel "
                    f"{self.channel_id}; leaving its topic blank so it is retried"
                )
            conv["summary"] = summary
            conv["updated_at"] = int(time.time() * 1000)
            
            db[conv_key] = conv
    
    def save_conversation_turns(
        self,
        conversation_id: int,
        turns: list[dict[str, Any]]
    ) -> int:
        """
        Create a new conversation with placeholder topic/summary, or replace its turns.

        This rewrites the whole turn list. Incremental saves use
        append_conversation_turns() instead, so that writing turn n does not
        rewrite turns 1..n-1.
        
        Args:
            conversation_id: The conversation ID to use
            turns: List of conversation turns
        
        Returns:
            The conversation ID used
        """
        with self._get_db() as db:
            conv_key = f"conv:{conversation_id}"
            
            if conv_key in db:
                # Conversation exists, just update turns
                conv = db[conv_key]
                conv["updated_at"] = int(time.time() * 1000)
            else:
                # Create new conversation with placeholder topic/summary
                conv = {
                    "topic": "",  # Will be generated later
                    "summary": "",  # Will be generated later
                    "created_at": int(time.time() * 1000),
                    "updated_at": int(time.time() * 1000),
                    "turns": []
                }
            self._replace_turns(db, conversation_id, conv, turns)
            db[conv_key] = conv
            
            return conversation_id

    def append_conversation_turns(
        self,
        conversation_id: int,
        new_turns: list[dict[str, Any]]
    ) -> int:
        """
        Append turns to a conversation without rewriting the ones already stored.

        This is the incremental-save path: the bytes written for a turn are the
        bytes of that turn, so a conversation of N turns costs O(N) writes in
        total rather than O(N^2). It is also what lets the in-memory history be
        windowed safely - the durable record only ever grows.

        Args:
            conversation_id: The conversation ID to append to (created if absent)
            new_turns: Turns to append, in order

        Returns:
            The conversation ID used
        """
        if not new_turns:
            return conversation_id

        with self._get_db() as db:
            conv_key = f"conv:{conversation_id}"
            now = int(time.time() * 1000)

            if conv_key in db:
                conv = db[conv_key]
            else:
                conv = {
                    "topic": "",  # Will be generated later
                    "summary": "",  # Will be generated later
                    "created_at": now,
                    "turns": [],
                    "appended_turn_count": 0,
                }

            next_index = int(conv.get("appended_turn_count") or 0)

            for offset, turn in enumerate(new_turns):
                db[self._turn_key(conversation_id, next_index + offset)] = turn

            conv["appended_turn_count"] = next_index + len(new_turns)
            conv["updated_at"] = now
            db[conv_key] = conv

            return conversation_id

    def count_conversation_turns(self, conversation_id: int) -> int:
        """Number of turns durably recorded for a conversation (0 if absent)."""
        with self._get_db() as db:
            conv = db.get(f"conv:{conversation_id}")
            if conv is None:
                return 0
            return int(conv.get("appended_turn_count") or 0)

    def get_conversation_label_state(self, conversation_id: int) -> tuple[str, int]:
        """The stored topic and the durable turn count, in one record read.

        Exactly what a lazy topic/summary trigger needs in order to decide
        whether to spend an LLM call: the topic, where blank means "no
        successful title yet" (see update_conversation_topic_summary), and how
        many turns the conversation now holds. Both live on the ``conv:{id}``
        record, so asking for them together costs one indexed read and no turn
        reads at all - ``get_conversation`` would rehydrate every turn to answer
        the same question, which on a long conversation is the memory spike this
        store exists to keep out of the process.

        Returns ("", 0) for a conversation with no record yet, which is the same
        answer as an unlabeled empty one: nothing to label.
        """
        with self._get_db() as db:
            conv = db.get(f"conv:{conversation_id}")
            if conv is None:
                return "", 0
            return (
                conv.get("topic") or "",
                int(conv.get("appended_turn_count") or 0),
            )

    def get_conversation_summaries(self, conversation_id: int) -> list[dict[str, Any]]:
        """Each turn's summary, without holding whole turns in memory.

        Topic and summary generation reads only this field. Loading every turn of
        a long conversation to throw its payload away would spike memory by the
        size of the whole conversation, which is the growth this store exists to
        keep out of the process.
        """
        with self._get_db() as db:
            conv = db.get(f"conv:{conversation_id}")
            if conv is None:
                return []
            return [
                {"conversation summary": turn.get("conversation summary")}
                for turn in self._iter_turn_records(db, conversation_id, conv)
            ]
    
    # NOTE: update_turn_feedback() removed - feedback is saved from the incremental
    # save flow after modifying conversation_history in memory, via the append path
    # when the turn is not durable yet and via update_last_conversation_turn() when
    # it already is.

    def update_last_conversation_turn(
        self,
        conversation_id: int,
        turn: dict[str, Any]
    ) -> bool:
        """
        Rewrite the newest durable turn of a conversation in place.

        Used when a turn that is already recorded is edited (feedback), which the
        append path cannot express. Returns False if there is no turn to rewrite.
        """
        with self._get_db() as db:
            conv_key = f"conv:{conversation_id}"
            if conv_key not in db:
                return False

            conv = db[conv_key]
            appended = int(conv.get("appended_turn_count") or 0)
            if not appended:
                return False

            db[self._turn_key(conversation_id, appended - 1)] = turn
            conv["updated_at"] = int(time.time() * 1000)
            db[conv_key] = conv
            return True
    
    def get_all_conversations_for_dump(self) -> list[dict[str, Any]]:
        """Get all conversations for admin dump"""
        with self._get_db() as db:
            meta = db.get("meta", {"last_conversation_id": 0})
            conversations = []
            
            for i in range(1, meta.get("last_conversation_id", 0) + 1):
                conv_key = f"conv:{i}"
                if conv_key in db:
                    conversations.append({
                        "channel_id": self.channel_id,
                        "conversation_id": i,
                        **self._hydrated(db, i, db[conv_key])
                    })
            
            return conversations


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
