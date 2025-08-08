#!/usr/bin/env python3
import argparse
import os
from typing import Optional

from .task_loader import load_tasks
from .env_adapter import RetailEnvAdapter
from .agent_runner import AgentRunner
from .evaluator import evaluate
from .reporter import print_console, write_json


def _create_env(env_kind: str, data_dir: Optional[str] = None):
    # Placeholder: user plugs in a real Tau Bench Retail env here
    class DummyEnv:
        def __init__(self, tasks):
            self._tasks = tasks
            self._idx = -1
        def reset(self, task_index):
            self._idx = task_index
            task = self._tasks[task_index]
            return {"instruction": task.get("instruction", "")}
        def step(self, action):
            # For now, mark all as done with zero reward
            return ({"reward": 0.0}, 0.0, True, {})
    return DummyEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate FastWorkflow agent on Tau Bench Retail tasks")
    parser.add_argument("tasks_path", help="Path to tasks file or directory containing retail.json")
    parser.add_argument("workflow_path", help="Path to workflow folder for the agent")
    parser.add_argument("env_file_path", help="Path to .env file")
    parser.add_argument("passwords_file_path", help="Path to passwords env file")
    parser.add_argument("--model", default=os.getenv("LLM_AGENT", "gpt-4o"))
    parser.add_argument("--provider_api_key", default=os.getenv("LITELLM_API_KEY_AGENT"))
    parser.add_argument("--task-ids", default="", help="Comma-separated indices to select tasks")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--json-out", default="", help="Optional path to write JSON results")
    args = parser.parse_args()

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()] if args.task_ids else None
    tasks = load_tasks(args.tasks_path, task_ids)

    # Prepare fastworkflow chat session
    import json as _json
    from dotenv import dotenv_values
    import fastworkflow

    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path),
    }
    fastworkflow.init(env_vars=env_vars)

    fastworkflow.chat_session = fastworkflow.ChatSession()
    fastworkflow.chat_session.start_workflow(
        args.workflow_path,
        workflow_context=None,
        startup_command="",
        startup_action=None,
        keep_alive=True,
        project_folderpath=None,
    )

    agent = AgentRunner(fastworkflow.chat_session, args.model, args.provider_api_key)

    # Create env adapter
    EnvClass = _create_env("retail")
    env = RetailEnvAdapter(EnvClass(tasks))

    results, metrics = evaluate(tasks, env, agent, max_steps=args.max_steps)
    print_console(results, metrics)

    if args.json_out:
        write_json(args.json_out, results, metrics)


if __name__ == "__main__":
    main()