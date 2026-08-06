"""
Conversation persistence layer for FastWorkflow
Provides Rdict-backed storage for multi-turn conversations with AI-generated topics/summaries
"""

import json
import os
from re import I
import time
from typing import Any, Optional

import dspy
from pydantic import BaseModel
from speedict import Rdict

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
    """Rdict-backed conversation persistence per user.

    Turns are stored one Rdict entry per turn (``conv:{id}:turn:{index}``) so an
    incremental save writes only the new turns instead of rewriting the whole
    conversation. The ``conv:{id}`` record keeps the metadata plus
    ``appended_turn_count``; reads rehydrate ``turns`` so callers see the same
    record shape as before. Records written by earlier versions keep their turns
    inline under ``turns`` and are migrated to per-turn entries on first append.
    """
    
    def __init__(self, channel_id: str, base_folder: str):
        self.channel_id = channel_id
        self.db_path = os.path.join(base_folder, f"{channel_id}.rdb")
        os.makedirs(base_folder, exist_ok=True)
    
    def _get_db(self) -> Rdict:
        """Get Rdict instance"""
        return Rdict(self.db_path)

    @staticmethod
    def _turn_key(conversation_id: int, index: int) -> str:
        return f"conv:{conversation_id}:turn:{index}"

    def _iter_turn_records(
        self, db: Rdict, conversation_id: int, conv: dict[str, Any]
    ):
        """Yield a conversation's turns in order, one at a time.

        An inline ``turns`` list wins outright. Only a writer that rewrites the
        whole list produces one, and every writer here empties it, so a record
        that still has one was written by an older version of this store — which
        makes it the authoritative list and any leftover per-turn entries stale.
        Concatenating the two instead would duplicate and reorder turns after a
        version rollback.
        """
        if inline_turns := conv.get("turns") or []:
            yield from inline_turns
            return
        for index in range(int(conv.get("appended_turn_count") or 0)):
            turn_key = self._turn_key(conversation_id, index)
            if turn_key in db:
                yield db[turn_key]

    def _read_turns(
        self, db: Rdict, conversation_id: int, conv: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Rehydrate a conversation's turns."""
        return list(self._iter_turn_records(db, conversation_id, conv))

    def _hydrated(
        self, db: Rdict, conversation_id: int, conv: dict[str, Any]
    ) -> dict[str, Any]:
        """The stored record as callers expect it: turns rehydrated, bookkeeping hidden."""
        hydrated = {k: v for k, v in conv.items() if k != "appended_turn_count"}
        hydrated["turns"] = self._read_turns(db, conversation_id, conv)
        return hydrated

    def _replace_turns(
        self,
        db: Rdict,
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
        try:
            db = self._get_db()
            meta = db.get("meta", {})
            return meta.get("last_conversation_id")
        finally:
            db.close()
    
    def _increment_conversation_id(self, db: Rdict) -> int:
        """Increment and return new conversation ID"""
        meta = db.get("meta", {"last_conversation_id": 0})
        new_id = meta["last_conversation_id"] + 1
        meta["last_conversation_id"] = new_id
        db["meta"] = meta
        return new_id
    
    def reserve_next_conversation_id(self) -> int:
        """Reserve the next conversation ID by incrementing the counter without creating a conversation"""
        db = self._get_db()
        try:
            return self._increment_conversation_id(db)
        finally:
            db.close()
    
    def _ensure_unique_topic(self, db: Rdict, candidate_topic: str) -> str:
        """Ensure topic is unique per user with case/whitespace insensitive comparison"""
        # Normalize for comparison
        normalized_candidate = candidate_topic.lower().strip()
        
        # Get all existing topics
        existing_topics = []
        meta = db.get("meta", {"last_conversation_id": 0})
        for i in range(1, meta.get("last_conversation_id", 0) + 1):
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
        db = self._get_db()
        try:
            if conversation_id is not None:
                # Use the specified ID (assumes it's valid and reserved)
                conv_id = conversation_id
            else:
                # Increment to get next ID
                conv_id = self._increment_conversation_id(db)
            
            unique_topic = self._ensure_unique_topic(db, topic)
            
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
        finally:
            db.close()
    
    def get_conversation(self, conv_id: int) -> Optional[dict[str, Any]]:
        """Get a conversation by ID"""
        db = self._get_db()
        try:
            conv = db.get(f"conv:{conv_id}")
            return None if conv is None else self._hydrated(db, conv_id, conv)
        finally:
            db.close()
    
    def get_conversation_by_topic(self, topic: str) -> Optional[tuple[int, dict[str, Any]]]:
        """Get conversation ID and data by topic (case/whitespace insensitive)"""
        db = self._get_db()
        try:
            meta = db.get("meta", {"last_conversation_id": 0})
            normalized_topic = topic.lower().strip()
            
            for i in range(1, meta.get("last_conversation_id", 0) + 1):
                conv_key = f"conv:{i}"
                if conv_key in db:
                    conv = db[conv_key]
                    if conv.get("topic", "").lower().strip() == normalized_topic:
                        return i, self._hydrated(db, i, conv)
            return None
        finally:
            db.close()
    
    def list_conversations(self, limit: int) -> list[ConversationSummary]:
        """List conversations ordered by updated_at desc, up to limit"""
        db = self._get_db()
        try:
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
        finally:
            db.close()
    
    def update_conversation(
        self,
        conv_id: int,
        topic: str,
        summary: str,
        turns: list[dict[str, Any]]
    ) -> None:
        """Update an existing conversation with new topic, summary, and turns"""
        db = self._get_db()
        try:
            conv_key = f"conv:{conv_id}"
            if conv_key not in db:
                raise ValueError(f"Conversation {conv_id} not found")
            
            conv = db[conv_key]
            unique_topic = self._ensure_unique_topic(db, topic)
            
            # Preserve created_at, update other fields
            conv["topic"] = unique_topic
            conv["summary"] = summary
            conv["updated_at"] = int(time.time() * 1000)
            self._replace_turns(db, conv_id, conv, turns)
            
            db[conv_key] = conv
        finally:
            db.close()
    
    def update_conversation_topic_summary(
        self,
        conv_id: int,
        topic: str,
        summary: str
    ) -> None:
        """
        Update only the topic and summary of an existing conversation.
        Used when finalizing a conversation (turns already saved incrementally).
        """
        db = self._get_db()
        try:
            conv_key = f"conv:{conv_id}"
            if conv_key not in db:
                raise ValueError(f"Conversation {conv_id} not found")
            
            conv = db[conv_key]
            unique_topic = self._ensure_unique_topic(db, topic)
            
            # Only update topic, summary, and timestamp - preserve turns
            conv["topic"] = unique_topic
            conv["summary"] = summary
            conv["updated_at"] = int(time.time() * 1000)
            
            db[conv_key] = conv
        finally:
            db.close()
    
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
        db = self._get_db()
        try:
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
        finally:
            db.close()

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

        db = self._get_db()
        try:
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
            # A record written by an earlier version keeps its turns inline.
            # Move them out once, so this and every later append stays O(1) in
            # bytes written instead of rewriting the inline list every turn.
            # Inline is the authoritative list (see _iter_turn_records), so any
            # per-turn entries it coexists with are stale and get replaced.
            if inline_turns := list(conv.get("turns") or []):
                self._replace_turns(db, conversation_id, conv, inline_turns)
                next_index = len(inline_turns)

            for offset, turn in enumerate(new_turns):
                db[self._turn_key(conversation_id, next_index + offset)] = turn

            conv["appended_turn_count"] = next_index + len(new_turns)
            conv["updated_at"] = now
            db[conv_key] = conv

            return conversation_id
        finally:
            db.close()

    def count_conversation_turns(self, conversation_id: int) -> int:
        """Number of turns durably recorded for a conversation (0 if absent)."""
        db = self._get_db()
        try:
            conv = db.get(f"conv:{conversation_id}")
            if conv is None:
                return 0
            if inline_turns := conv.get("turns") or []:
                return len(inline_turns)
            return int(conv.get("appended_turn_count") or 0)
        finally:
            db.close()

    def get_conversation_summaries(self, conversation_id: int) -> list[dict[str, Any]]:
        """Each turn's summary, without holding whole turns in memory.

        Topic and summary generation reads only this field. Loading every turn of
        a long conversation to throw its payload away would spike memory by the
        size of the whole conversation, which is the growth this store exists to
        keep out of the process.
        """
        db = self._get_db()
        try:
            conv = db.get(f"conv:{conversation_id}")
            if conv is None:
                return []
            return [
                {"conversation summary": turn.get("conversation summary")}
                for turn in self._iter_turn_records(db, conversation_id, conv)
            ]
        finally:
            db.close()
    
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
        db = self._get_db()
        try:
            conv_key = f"conv:{conversation_id}"
            if conv_key not in db:
                return False

            conv = db[conv_key]
            appended = int(conv.get("appended_turn_count") or 0)
            if appended:
                db[self._turn_key(conversation_id, appended - 1)] = turn
            elif inline_turns := list(conv.get("turns") or []):
                inline_turns[-1] = turn
                conv["turns"] = inline_turns
            else:
                return False

            conv["updated_at"] = int(time.time() * 1000)
            db[conv_key] = conv
            return True
        finally:
            db.close()
    
    def get_all_conversations_for_dump(self) -> list[dict[str, Any]]:
        """Get all conversations for admin dump"""
        db = self._get_db()
        try:
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
        finally:
            db.close()


def generate_topic_and_summary(turns: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Generate topic and summary for a conversation using DSPy.
    
    Only passes conversation summaries (not verbose traces) to the AI model
    for better quality topic/summary generation.
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
    lm = get_lm("LLM_CONVERSATION_STORE", "LITELLM_API_KEY_CONVERSATION_STORE")
    with dspy.context(lm=lm):
        generator = dspy.ChainOfThought(TopicSummarySignature)
        result = generator(conversation_turns=turns_str)
        return result.topic, result.summary

