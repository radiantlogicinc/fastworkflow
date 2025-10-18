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


def extract_turns_from_history(conversation_history: 'dspy.History') -> list[dict[str, Any]]:
    # sourcery skip: remove-unused-enumerate
    """
    Extract turns from dspy.History format to Rdict format.
    
    dspy.History.messages format:
    [
        {
            "conversation summary": "summary text1", 
            "conversation_traces": "conversation_traces1",
            "feedback": {...} or None
        },
        ...
    ]
    
    Rdict turn format:
    [
        {
            "conversation summary": "...",
            "conversation_traces": "...",
            "feedback": {...} or None
        },
        ...
    ]
    
    Note: dspy.History stores conversation summaries, detailed traces, and optional feedback.
    All fields are extracted and preserved for complete conversation persistence.
    """
    turns = []

    turns.extend(
        {
            "conversation summary": msg_dict.get("conversation summary"),
            "conversation_traces": msg_dict.get("conversation_traces"),
            "feedback": msg_dict.get("feedback"),  # Preserve existing feedback
        }
        for msg_dict in conversation_history.messages
    )
    return turns


def restore_history_from_turns(turns: list[dict[str, Any]]) -> 'dspy.History':
    """
    Restore dspy.History from Rdict turns.
    
    Converts back from Rdict format to dspy.History format.
    Restores conversation summary, conversation_traces, and feedback for each turn.
    """
    messages = []

    messages.extend(
        {
            "conversation summary": turn.get("conversation summary"),
            "conversation_traces": turn.get("conversation_traces"),
            "feedback": turn.get("feedback"),  # Restore feedback if present
        }
        for turn in turns
    )
    return dspy.History(messages=messages)


class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: int
    topic: str
    summary: str
    created_at: int
    updated_at: int


class ConversationStore:
    """Rdict-backed conversation persistence per user"""
    
    def __init__(self, user_id: str, base_folder: str):
        self.user_id = user_id
        self.db_path = os.path.join(base_folder, f"{user_id}.rdb")
        os.makedirs(base_folder, exist_ok=True)
    
    def _get_db(self) -> Rdict:
        """Get Rdict instance"""
        return Rdict(self.db_path)
    
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
                "turns": turns
            }
            db[f"conv:{conv_id}"] = conversation
            return conv_id
        finally:
            db.close()
    
    def get_conversation(self, conv_id: int) -> Optional[dict[str, Any]]:
        """Get a conversation by ID"""
        db = self._get_db()
        try:
            return db.get(f"conv:{conv_id}")
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
                        return i, conv
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
            conv["turns"] = turns
            
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
        Create a new conversation with placeholder topic/summary, or update existing turns.
        Used for incremental saves without generating topic/summary.
        
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
                conv["turns"] = turns
                db[conv_key] = conv
            else:
                # Create new conversation with placeholder topic/summary
                conversation = {
                    "topic": "",  # Will be generated later
                    "summary": "",  # Will be generated later
                    "created_at": int(time.time() * 1000),
                    "updated_at": int(time.time() * 1000),
                    "turns": turns
                }
                db[conv_key] = conversation
            
            return conversation_id
        finally:
            db.close()
    
    # NOTE: update_turn_feedback() removed - feedback is now saved via save_conversation_turns()
    # in the incremental save flow after modifying conversation_history in memory
    
    def get_all_conversations_for_dump(self) -> list[dict[str, Any]]:
        """Get all conversations for admin dump"""
        db = self._get_db()
        try:
            meta = db.get("meta", {"last_conversation_id": 0})
            conversations = []
            
            for i in range(1, meta.get("last_conversation_id", 0) + 1):
                conv_key = f"conv:{i}"
                if conv_key in db:
                    conv = db[conv_key]
                    conversations.append({
                        "user_id": self.user_id,
                        "conversation_id": i,
                        **conv
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

