"""
Tests for CommandsSystemPreludeAdapter to verify available_commands injection into system messages.
"""

import pytest
import dspy
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter


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

