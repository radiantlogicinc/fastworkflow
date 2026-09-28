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
from dspy.adapters.json_adapter import JSONAdapter

from fastworkflow.utils.logging import logger

#: DSPy's private ChatAdapter hook that builds the JSON retry adapter
#: (checked on DSPy 3.3.0; the pin allows ``^3.0.1``).
JSON_RETRY_HOOK = "_make_json_adapter_fallback"

_json_retry_hook_warned = False


def json_retry_hook_missing(adapter_class: type = dspy.ChatAdapter) -> bool:
    """Whether *adapter_class* lacks the hook ``CommandsSystemPreludeAdapter`` overrides."""
    return not callable(getattr(adapter_class, JSON_RETRY_HOOK, None))


def warn_if_json_retry_hook_missing(adapter_class: type = dspy.ChatAdapter) -> bool:
    """Log once per process if the installed DSPy lacks the hook; True if it does."""
    global _json_retry_hook_warned
    missing = json_retry_hook_missing(adapter_class)
    if missing and not _json_retry_hook_warned:
        _json_retry_hook_warned = True
        logger.warning(
            f"dspy {getattr(dspy, '__version__', '?')} has no ChatAdapter.{JSON_RETRY_HOOK}: "
            "when a workflow agent reply does not parse, DSPy's JSON retry will drop the "
            "available commands list from the prompt"
        )
    return missing


def _split_commands(inputs):
    """``available_commands`` and the inputs without it, which the signature does not declare."""
    return inputs.get("available_commands"), {k: v for k, v in inputs.items() if k != "available_commands"}


def _inject_commands_prelude(formatted, title, cmds):
    """Prepend the commands section to the system message of ``formatted``."""
    if not cmds:
        return formatted
    
    # Inject commands into the system message
    prelude = f"{title}:\n{cmds}".strip()
    
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


class CommandsSystemPreludeJSONAdapter(JSONAdapter):
    """The JSONAdapter DSPy retries with when a chat-format reply does not parse,
    carrying the same available-commands prelude as the chat attempt."""

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.title = title

    def format(self, signature, demos, inputs):
        cmds, inputs_for_base = _split_commands(inputs)
        return _inject_commands_prelude(super().format(signature, demos, inputs_for_base), self.title, cmds)


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
        use_json_adapter_fallback: DSPy's retry with a JSONAdapter when a reply
            does not parse. The retry adapter (``CommandsSystemPreludeJSONAdapter``)
            carries the same available commands. Defaults to DSPy's own default (True).
    
    Example:
        >>> import dspy
        >>> from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
        >>> dspy.settings.adapter = CommandsSystemPreludeAdapter()
    """
    
    def __init__(self, base: dspy.ChatAdapter | None = None, title: str = "Available execute_workflow_query tool commands",
                 use_json_adapter_fallback: bool = True):
        super().__init__(use_json_adapter_fallback=use_json_adapter_fallback)
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
        # Extract available_commands before passing to base adapter, and
        # create a copy of inputs without available_commands to avoid including it in user message
        cmds, inputs_for_base = _split_commands(inputs)

        # Call the base adapter's format method with filtered inputs
        formatted = self.base.format(signature, demos, inputs_for_base)

        return _inject_commands_prelude(formatted, self.title, cmds)

    def _make_json_adapter_fallback(self):
        # DSPy's private hook (ChatAdapter.__call__/acall build the retry
        # adapter through it); tests/test_chat_adapter_commands.py fails if it
        # is renamed. Same arguments as DSPy's own JSONAdapter fallback.
        return CommandsSystemPreludeJSONAdapter(
            self.title,
            use_native_function_calling=self.use_native_function_calling,
            parallel_tool_calls=self.parallel_tool_calls,
        )


warn_if_json_retry_hook_missing()
