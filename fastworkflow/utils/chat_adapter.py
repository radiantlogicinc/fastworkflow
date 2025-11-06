"""
ChatAdapter wrapper for injecting context-specific available commands into system messages.

Design Overview:
---------------
This module implements a ChatAdapter wrapper that dynamically injects workflow command information
into the system message at runtime, avoiding the need to rebuild ReAct agent modules per context.

Key Benefits:
- Single shared agent: No per-context module caching required
- Dynamic updates: Commands refresh per call based on current workflow context
- Token efficiency: Commands appear in system (not repeated in trajectory/history)
- Zero rebuild cost: Signature and modules remain stable across context changes

Usage:
------
The adapter is used specifically for workflow agent calls via dspy.context():

    from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
    
    agent_adapter = CommandsSystemPreludeAdapter()
    available_commands = _what_can_i_do(chat_session)
    
    with dspy.context(lm=lm, adapter=agent_adapter):
        agent_result = agent(
            user_query="...",
            available_commands=available_commands
        )

The adapter intercepts the format call and prepends commands to the system message,
keeping them out of the trajectory to prevent token bloat across iterations.
This scoped approach ensures the adapter only affects workflow agent calls, not other
DSPy operations in the system.
"""
import dspy


class CommandsSystemPreludeAdapter(dspy.ChatAdapter):
    """
    Wraps a base DSPy ChatAdapter to inject available commands into the system message.
    
    This adapter intercepts the render process and prepends a "Available commands" section
    to the system message when `available_commands` is present in inputs. This ensures
    commands are visible to the model at each step without being added to the trajectory
    or conversation history.
    
    Args:
        base: The underlying ChatAdapter to wrap. Defaults to dspy.ChatAdapter() if None.
        title: The header text for the commands section. Defaults to "Available commands".
    
    Example:
        >>> import dspy
        >>> from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
        >>> dspy.settings.adapter = CommandsSystemPreludeAdapter()
    """
    
    def __init__(self, base: dspy.ChatAdapter | None = None, title: str = "Available execute_workflow_query tool commands"):
        super().__init__()
        self.base = base or dspy.ChatAdapter()
        self.title = title
    
    def format(self, signature, demos, inputs):
        """
        Format the inputs for the model, injecting available_commands into system message.
        
        This method wraps the base adapter's format method and modifies the result
        to include available commands in the system message if present in inputs.
        
        Args:
            signature: The DSPy signature defining the task
            demos: List of demonstration examples
            inputs: Dictionary of input values, may include 'available_commands'
            
        Returns:
            Formatted messages with commands injected into system message
        """
        # Call the base adapter's format method
        formatted = self.base.format(signature, demos, inputs)
        
        # Check if available_commands is in inputs
        cmds = inputs.get("available_commands")
        if not cmds:
            return formatted
        
        # Inject commands into the system message
        prelude = f"{self.title}:\n{cmds}".strip()
        
        # Formatted output is a list of messages, first may be system
        # Find and modify the system message, or prepend one
        if formatted and formatted[0].get("role") == "system":
            # Prepend to existing system message
            existing_content = formatted[0].get("content", "")
            formatted[0]["content"] = f"{prelude}\n\n{existing_content}".strip()
        else:
            # No system message exists, prepend one
            formatted.insert(0, {"role": "system", "content": prelude})
        
        return formatted

