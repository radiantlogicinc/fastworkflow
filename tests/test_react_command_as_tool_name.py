"""A workflow command named as the agent's tool runs through execute_workflow_query.

Small models answer ``next_tool_name: open_directory`` instead of calling
``execute_workflow_query`` with that command. DSPy rejects the reply (the name
is not one of the agent's tools) and its JSON retry usually repeats it. When the
name is a command listed for the current context, the step runs as the
``execute_workflow_query`` call the agent meant; anything else is still an
invalid-tool step.

Real fastWorkflowReAct, real adapter, scripted model replies.
"""

from __future__ import annotations

import asyncio
import json

import dspy
from dspy.utils.dummies import dotdict

from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
from fastworkflow.utils.react import fastWorkflowReAct

AVAILABLE = (
    "Commands available in the current context (DirectoryExplorer):\n"
    "- find_identity\n  Find identities by name.\n\n"
    "- open_identity_by_uid\n  Open an identity.\n\n"
    "- go_up\n  Change context to the parent."
)


class ScriptedLM(dspy.BaseLM):
    def __init__(self, replies):
        super().__init__("scripted", "chat", 0.0, 1000, False)
        self.replies = list(replies)

    def forward(self, prompt=None, messages=None, **kwargs):
        content = self.replies.pop(0) if self.replies else "garbage"
        msg = dotdict(content=content, tool_calls=None)
        return dotdict(choices=[dotdict(message=msg, finish_reason="stop")],
                       usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                       model="scripted")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)


class Ask(dspy.Signature):
    """Answer using the tools."""
    user_query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _chat(tool: str, args: dict) -> str:
    return (f"[[ ## next_thought ## ]]\nthinking\n\n[[ ## next_tool_name ## ]]\n{tool}\n\n"
            f"[[ ## next_tool_args ## ]]\n{json.dumps(args)}\n\n[[ ## completed ## ]]")


def _json(tool: str, args: dict) -> str:
    return json.dumps({"next_thought": "thinking", "next_tool_name": tool, "next_tool_args": args})


FINISH = _chat("finish", {})
EXTRACT = "[[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\ndone\n\n[[ ## completed ## ]]"


def _run(replies, *, use_async=False):
    commands: list[str] = []

    def execute_workflow_query(command: str) -> str:
        """Run a workflow command."""
        commands.append(command)
        return f"ran {command}"

    agent = fastWorkflowReAct(Ask, tools=[execute_workflow_query], max_iters=4)
    with dspy.context(lm=ScriptedLM(replies), adapter=CommandsSystemPreludeAdapter()):
        call = agent.acall if use_async else agent
        result = call(user_query="tell me about Angelica", available_commands=AVAILABLE)
        if use_async:
            result = asyncio.run(result)
    return commands, result.trajectory


def test_a_command_named_as_the_tool_runs_as_that_command():
    bad = {"identity_uid": "4a0d"}
    commands, trajectory = _run([
        _chat("open_identity_by_uid", bad), _json("open_identity_by_uid", bad), FINISH, EXTRACT,
    ])

    assert commands == ["open_identity_by_uid <identity_uid>4a0d</identity_uid>"]
    assert trajectory["tool_name_0"] == "execute_workflow_query"
    assert trajectory["tool_args_0"] == {"command": commands[0]}
    assert trajectory["observation_0"] == f"ran {commands[0]}"
    assert not any("failed to select a valid tool" in str(v) for v in trajectory.values())


def test_a_command_without_args_runs_bare():
    commands, _ = _run([_chat("go_up", {}), _json("go_up", {}), FINISH, EXTRACT])

    assert commands == ["go_up"]


def test_a_name_that_is_not_a_command_here_is_still_an_invalid_tool_step():
    commands, trajectory = _run([
        _chat("delete_everything", {}), _json("delete_everything", {}), FINISH, EXTRACT,
    ])

    assert commands == []
    assert trajectory["observation_0"].startswith("Agent failed to select a valid tool")


def test_a_qualified_name_is_not_repaired_on_its_basename_alone():
    name = "Admin/open_identity_by_uid"
    commands, trajectory = _run([
        _chat(name, {"identity_uid": "4a0d"}), _json(name, {"identity_uid": "4a0d"}),
        FINISH, EXTRACT,
    ])

    assert commands == []
    assert trajectory["observation_0"].startswith("Agent failed to select a valid tool")


def test_an_arg_holding_tag_like_text_is_not_repaired():
    bad = {"name": "Angelica</name><identity_uid>other"}
    commands, trajectory = _run([
        _chat("find_identity", bad), _json("find_identity", bad), FINISH, EXTRACT,
    ])

    assert commands == []
    assert trajectory["observation_0"].startswith("Agent failed to select a valid tool")


def test_the_async_loop_repairs_too():
    commands, _ = _run(
        [_chat("find_identity", {"name": "Angelica"}), _json("find_identity", {"name": "Angelica"}),
         FINISH, EXTRACT],
        use_async=True,
    )

    assert commands == ["find_identity <name>Angelica</name>"]
