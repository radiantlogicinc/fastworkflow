from typing import Optional

import fastworkflow


def initialize_main_agent(chat_session: fastworkflow.ChatSession):
    """
    Initialize and return the DSPy ReAct main agent for agentic mode.

    Uses the existing run_agent agent_module to configure the agent with the
    current chat_session and environment variables.
    """
    from fastworkflow.run_agent.agent_module import initialize_dspy_agent

    LLM_AGENT = fastworkflow.get_env_var("LLM_AGENT")
    LITELLM_API_KEY_AGENT = fastworkflow.get_env_var("LITELLM_API_KEY_AGENT")

    # Will raise if LLM_AGENT is missing; callers should handle failures
    return initialize_dspy_agent(
        chat_session,
        LLM_AGENT,
        LITELLM_API_KEY_AGENT,
        clear_cache=False,
    )