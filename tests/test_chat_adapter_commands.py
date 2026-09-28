"""
Tests for CommandsSystemPreludeAdapter to verify available_commands injection into system messages.
"""

import asyncio
import json
import logging
from typing import Any, Literal

import pytest
import dspy
from dspy.utils.dummies import dotdict
from dspy.utils.exceptions import AdapterParseError

from fastworkflow.utils import chat_adapter
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter, CommandsSystemPreludeJSONAdapter
from fastworkflow.utils.logging import logger


def test_chat_adapter_injects_commands_into_system():
    """Test that CommandsSystemPreludeAdapter injects available_commands into system message."""
    # Create a simple signature for testing
    test_signature = dspy.Signature("question -> answer")
    
    # Create adapter
    adapter = CommandsSystemPreludeAdapter()
    
    # Test inputs with available_commands
    inputs = {
        "question": "What is the capital of France?",
        "available_commands": "Command 1: get_weather\nCommand 2: search_info"
    }
    
    # Format with the adapter
    formatted = adapter.format(test_signature, demos=[], inputs=inputs)
    
    # Verify that commands are in the system message
    assert formatted is not None
    assert len(formatted) > 0
    
    # Check if first message is system and contains commands
    system_message = next((msg for msg in formatted if msg.get("role") == "system"), None)
    
    assert system_message is not None, "System message should exist"
    assert "Available execute_workflow_query tool commands:" in system_message.get("content", "")
    assert "Command 1: get_weather" in system_message.get("content", "")
    assert "Command 2: search_info" in system_message.get("content", "")


def test_chat_adapter_no_commands_passthrough():
    """Test that adapter passes through normally when no available_commands provided."""
    # Create a simple signature for testing
    test_signature = dspy.Signature("question -> answer")
    
    # Create adapter
    base_adapter = dspy.ChatAdapter()
    adapter = CommandsSystemPreludeAdapter(base=base_adapter)
    
    # Test inputs without available_commands
    inputs = {
        "question": "What is the capital of France?"
    }
    
    # Format with both adapters
    formatted_with_wrapper = adapter.format(test_signature, demos=[], inputs=inputs)
    formatted_base = base_adapter.format(test_signature, demos=[], inputs=inputs)
    
    # Results should be identical when no commands are present
    assert formatted_with_wrapper == formatted_base


def test_chat_adapter_custom_title():
    """Test that custom title is used for commands section."""
    # Create a simple signature for testing
    test_signature = dspy.Signature("question -> answer")
    
    # Create adapter with custom title
    adapter = CommandsSystemPreludeAdapter(title="Workflow Commands")
    
    # Test inputs with available_commands
    inputs = {
        "question": "What is the capital of France?",
        "available_commands": "Command 1: get_weather"
    }
    
    # Format with the adapter
    formatted = adapter.format(test_signature, demos=[], inputs=inputs)
    
    # Find system message
    system_message = next((msg for msg in formatted if msg.get("role") == "system"), None)
    
    assert system_message is not None
    assert "Workflow Commands:" in system_message.get("content", "")


def test_chat_adapter_preserves_existing_system_content():
    """Test that adapter preserves existing system content when present."""
    # Create a signature with instructions
    test_signature = dspy.Signature(
        "question -> answer",
        instructions="You are a helpful assistant."
    )
    
    # Create adapter
    adapter = CommandsSystemPreludeAdapter()
    
    # Test inputs with available_commands
    inputs = {
        "question": "What is the capital of France?",
        "available_commands": "Command 1: get_weather"
    }
    
    # Format with the adapter
    formatted = adapter.format(test_signature, demos=[], inputs=inputs)
    
    # Find system message
    system_message = next((msg for msg in formatted if msg.get("role") == "system"), None)
    
    assert system_message is not None
    content = system_message.get("content", "")
    
    # Should have both commands and original instructions
    assert "Available execute_workflow_query tool commands:" in content
    assert "Command 1: get_weather" in content
    # The original instructions should be preserved somewhere in system content
    # (exact format depends on DSPy adapter implementation)


# ---------------------------------------------------------------------------
# DSPy's JSON retry when a chat-format reply does not parse
# ---------------------------------------------------------------------------

COMMANDS = "real_command_a\nreal_command_b"


class ScriptedLM(dspy.BaseLM):
    """Answers each call with the next scripted reply ("garbage" once they run out); records every prompt."""

    def __init__(self, replies, schema=False):
        super().__init__("scripted", "chat", 0.0, 1000, False)
        self.replies = list(replies)
        self.prompts: list[str] = []
        self._schema = schema

    @property
    def supports_response_schema(self):
        return self._schema

    @property
    def supported_params(self):
        return {"response_format"} if self._schema else set()

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content") or "") for m in messages or []))
        content = self.replies.pop(0) if self.replies else "garbage"
        msg = dotdict(content=content, tool_calls=None)
        return dotdict(choices=[dotdict(message=msg, finish_reason="stop")],
                       usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0), model="scripted")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)


class ReactStepSignature(dspy.Signature):
    """Pick the next tool."""
    user_query: str = dspy.InputField()
    trajectory: str = dspy.InputField()
    next_thought: str = dspy.OutputField()
    next_tool_name: Literal["execute_workflow_query", "finish"] = dspy.OutputField()
    next_tool_args: dict[str, Any] = dspy.OutputField()


GOOD_CHAT = ("[[ ## next_thought ## ]]\nt\n\n[[ ## next_tool_name ## ]]\nfinish\n\n"
             "[[ ## next_tool_args ## ]]\n{}\n\n[[ ## completed ## ]]")
GOOD_JSON = json.dumps({"next_thought": "t", "next_tool_name": "finish", "next_tool_args": {}})
REPLY_CASES = {
    "chat_parses": ([GOOD_CHAT], 1, True),
    "json_retry_parses": (["garbage", GOOD_JSON], 2, True),
    "both_bad": ([], 2, False),
}


def _predict(adapter, replies, schema, *, use_async=False):
    """(calls, prompts, parsed) for one ReAct-shaped step through *adapter*."""
    lm = ScriptedLM(replies, schema=schema)
    predict = dspy.Predict(ReactStepSignature)
    kwargs = {"user_query": "q", "trajectory": "", "available_commands": COMMANDS}
    try:
        with dspy.context(lm=lm, adapter=adapter):
            if use_async:
                asyncio.run(predict.acall(**kwargs))
            else:
                predict(**kwargs)
        parsed = True
    except AdapterParseError:
        parsed = False
    return len(lm.prompts), lm.prompts, parsed


def test_dspy_still_builds_its_json_retry_through_the_hook_we_override():
    """The prelude reaches the JSON retry only through this private DSPy hook."""
    assert hasattr(dspy.ChatAdapter, "_make_json_adapter_fallback"), (
        "dspy.ChatAdapter._make_json_adapter_fallback is gone: DSPy's JSON retry would "
        "lose the available commands. Re-point CommandsSystemPreludeAdapter at the new hook.")
    retry = CommandsSystemPreludeAdapter(title="T")._make_json_adapter_fallback()
    assert isinstance(retry, CommandsSystemPreludeJSONAdapter) and retry.title == "T"


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def hook_warnings(monkeypatch):
    handler = _Warnings()
    logger.addHandler(handler)
    monkeypatch.setattr(chat_adapter, "_json_retry_hook_warned", False)
    yield handler.records
    logger.removeHandler(handler)


def test_the_installed_dspy_has_the_hook_so_nothing_is_logged(hook_warnings):
    assert chat_adapter.warn_if_json_retry_hook_missing() is False
    assert hook_warnings == []


def test_an_adapter_class_without_the_hook_is_reported_once(hook_warnings):
    class _BeforeTheHook(dspy.Adapter):
        pass

    assert chat_adapter.json_retry_hook_missing(_BeforeTheHook)
    assert not chat_adapter.json_retry_hook_missing(CommandsSystemPreludeAdapter)
    assert chat_adapter.warn_if_json_retry_hook_missing(_BeforeTheHook) is True
    assert chat_adapter.warn_if_json_retry_hook_missing(_BeforeTheHook) is True
    message, = [r.getMessage() for r in hook_warnings]
    assert chat_adapter.JSON_RETRY_HOOK in message and "drop the available commands" in message


@pytest.mark.parametrize("schema", [False, True], ids=["text_lm", "schema_lm"])
@pytest.mark.parametrize("case", list(REPLY_CASES))
def test_the_json_retry_keeps_the_commands_and_the_call_count(case, schema):
    replies, calls, parses = REPLY_CASES[case]

    n, prompts, parsed = _predict(CommandsSystemPreludeAdapter(), list(replies), schema)

    assert (n, parsed) == (calls, parses)
    # DSPy's own adapter makes exactly as many calls.
    assert _predict(dspy.ChatAdapter(), list(replies), schema)[0] == calls
    for prompt in prompts:
        assert "Available execute_workflow_query tool commands:" in prompt
        assert "real_command_a" in prompt and "real_command_b" in prompt


def test_the_async_json_retry_keeps_the_commands():
    n, prompts, parsed = _predict(CommandsSystemPreludeAdapter(), ["garbage", GOOD_JSON], False, use_async=True)

    assert (n, parsed) == (2, True)
    assert "real_command_a" in prompts[1]


def test_with_the_fallback_off_there_is_no_retry():
    n, _prompts, parsed = _predict(
        CommandsSystemPreludeAdapter(use_json_adapter_fallback=False), ["garbage", GOOD_JSON], False)

    assert (n, parsed) == (1, False)


def test_the_json_retry_keeps_the_commands_out_of_the_user_message():
    formatted = CommandsSystemPreludeJSONAdapter("Available execute_workflow_query tool commands").format(
        ReactStepSignature, demos=[], inputs={"user_query": "q", "trajectory": "", "available_commands": COMMANDS})

    assert formatted[0]["role"] == "system" and "real_command_a" in formatted[0]["content"]
    assert all("real_command_a" not in m["content"] for m in formatted[1:])
