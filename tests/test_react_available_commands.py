"""
Integration tests for ReAct agent with available_commands injection.
"""

import pytest
import dspy
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter


def test_react_accepts_available_commands_parameter():
    """Test that fastWorkflowReAct accepts available_commands as input."""
    
    # Set up the adapter
    dspy.settings.adapter = CommandsSystemPreludeAdapter()
    
    # Configure a simple LM for testing (using a mock-friendly model)
    lm = dspy.LM(model="openai/gpt-3.5-turbo", api_key="test-key", cache=False)
    
    # Create a simple signature
    class TestSignature(dspy.Signature):
        """Answer questions using available tools."""
        question = dspy.InputField()
        answer = dspy.OutputField()
    
    # Define a simple tool
    def get_info(topic: str) -> str:
        """Get information about a topic."""
        return f"Information about {topic}"
    
    # Create ReAct agent
    agent = fastWorkflowReAct(
        signature=TestSignature,
        tools=[get_info],
        max_iters=3
    )
    
    # Test that available_commands can be passed
    # This will fail to actually run without a real API key, but we can verify
    # the signature accepts the parameter
    try:
        with dspy.context(lm=lm):
            # This should not raise a TypeError about unexpected keyword argument
            result = agent(
                question="What is Python?",
                available_commands="Command 1: get_info - Get information about a topic"
            )
    except Exception as e:
        # We expect API errors, not signature errors
        error_msg = str(e)
        assert "available_commands" not in error_msg, \
            f"Should not fail due to available_commands parameter: {error_msg}"
        # If it's an API error (like auth or connection), that's expected in tests
        assert any(keyword in error_msg.lower() for keyword in [
            "api", "auth", "key", "connection", "network", "openai", "litellm"
        ]), f"Expected API-related error, got: {error_msg}"


def test_react_trajectory_excludes_available_commands():
    """Test that available_commands is not included in trajectory formatting."""
    
    # Create a simple signature
    class TestSignature(dspy.Signature):
        """Answer questions using available tools."""
        question = dspy.InputField()
        answer = dspy.OutputField()
    
    # Define a simple tool
    def simple_tool() -> str:
        """A simple tool."""
        return "tool result"
    
    # Create ReAct agent
    agent = fastWorkflowReAct(
        signature=TestSignature,
        tools=[simple_tool],
        max_iters=1
    )
    
    # Create a test trajectory
    test_trajectory = {
        "thought_0": "I should call the tool",
        "tool_name_0": "simple_tool",
        "tool_args_0": {},
        "observation_0": "tool result"
    }
    
    # Format the trajectory
    formatted = agent._format_trajectory(test_trajectory)
    
    # Verify that formatted trajectory doesn't contain 'available_commands'
    assert isinstance(formatted, str)
    assert "available_commands" not in formatted.lower()
    
    # Should contain trajectory items
    assert "thought_0" in formatted or "I should call the tool" in formatted


@pytest.mark.skip(reason="Requires real API key and makes actual API calls")
def test_react_e2e_with_commands_injection():
    """
    End-to-end test that available_commands appears in system message.
    Skipped by default as it requires a real API key.
    """
    import os
    
    # Only run if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available")
    
    # Set up the adapter
    dspy.settings.adapter = CommandsSystemPreludeAdapter()
    
    # Configure LM with real key
    lm = dspy.LM(model="openai/gpt-3.5-turbo")
    
    # Create a simple signature
    class TestSignature(dspy.Signature):
        """Answer questions using available tools."""
        question = dspy.InputField()
        answer = dspy.OutputField()
    
    # Define tools
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"
    
    # Create ReAct agent
    agent = fastWorkflowReAct(
        signature=TestSignature,
        tools=[get_weather],
        max_iters=5
    )
    
    # Run agent with commands
    with dspy.context(lm=lm):
        result = agent(
            question="What's the weather in Paris?",
            available_commands="Command: get_weather - Get weather for a city (param: city)"
        )
    
    # Verify result has expected structure
    assert hasattr(result, "answer")
    assert "Paris" in result.answer or "weather" in result.answer.lower()

