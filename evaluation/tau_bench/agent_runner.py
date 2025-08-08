from typing import Any, Dict, Tuple


class AgentRunner:
    """
    Wrap FastWorkflow agent to generate next message/action given observation/history.
    This is a placeholder; integrate with fastworkflow.run_agent.initialize_dspy_agent lazily.
    """

    def __init__(self, chat_session: Any, model_name: str, api_key: str | None = None):
        import fastworkflow
        from fastworkflow.run_agent.agent_module import initialize_dspy_agent
        self._chat_session = chat_session
        self._agent = initialize_dspy_agent(chat_session, model_name, api_key, clear_cache=True)

    def next_action(self, observation: Dict, history: list[Dict]) -> Tuple[Dict, float]:
        # Implement mapping from observation/history to agent call as needed.
        # For now, assume observation contains a "user_query" and the agent returns an Action-like dict and cost 0.
        _ = history  # unused baseline
        user_query = observation.get("instruction") or observation.get("user_query") or ""
        # In a full implementation, pass query into the agent and parse the tool call into Tau Action format
        action = {"name": "noop", "arguments": {"query": user_query}}
        return action, 0.0