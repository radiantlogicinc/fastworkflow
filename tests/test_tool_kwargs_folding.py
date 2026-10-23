"""Parameters passed as tool arguments beside ``command`` are folded into it (fix-5jss).

Small models call the command tool as ``{"command": "find_item", "name": "Ann"}``.
The tool's arg check used to refuse each such call with a traceback, which the
model then repeated. Fixtures mirror tests/test_repeat_block.py.
"""

from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow import tracing, workflow_agent
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.workflow_agent import fold_tool_kwargs
from tests.test_multi_command_refusal import initialized_fastworkflow, session  # noqa: F401
from tests.test_repeat_block import agent, responses  # noqa: F401
from tests.test_tracing_phase1 import todo_workflow_path  # noqa: F401

TOOL = "execute_workflow_query"


# ----------------------------------------------------------------------
# The folding itself
# ----------------------------------------------------------------------


def test_a_keyword_beside_a_bare_command_name_becomes_key_value():
    assert fold_tool_kwargs("find_item", {"name": "Ann Lee"}) == ("find_item name=Ann Lee", [])


def test_several_keywords_keep_their_order():
    folded, _ = fold_tool_kwargs("find_item", {"name": "Ann", "start": 2})
    assert folded == "find_item name=Ann start=2"


def test_a_nested_args_dict_is_unpacked():
    for key in ("args", "arguments", "params", "parameters", "kwargs"):
        folded, _ = fold_tool_kwargs("open_item", {key: {"item_uid": "ab12"}})
        assert folded == "open_item item_uid=ab12", key


def test_a_scalar_under_args_is_a_positional_value():
    assert fold_tool_kwargs("find_item", {"args": "Ann Lee"}) == ("find_item Ann Lee", [])


def test_the_command_name_under_another_key_is_used_when_command_is_missing():
    folded, _ = fold_tool_kwargs("", {"command_name": "find_item", "name": "Ann"})
    assert folded == "find_item name=Ann"


def test_a_command_name_key_is_a_parameter_when_command_is_given():
    folded, _ = fold_tool_kwargs("run_job", {"action": "start"})
    assert folded == "run_job action=start"


def test_a_multi_word_value_under_a_name_key_is_not_taken_as_the_command():
    folded, _ = fold_tool_kwargs("", {"action": "start the job"})
    assert folded == "action=start the job"


def test_no_command_at_all_keeps_every_value():
    folded, _ = fold_tool_kwargs("", {"kind": "item", "item_id": "ab12"})
    assert folded == "kind=item item_id=ab12"


def test_lists_and_dicts_become_json_text():
    folded, _ = fold_tool_kwargs("tag_items", {"item_ids": ["a", "b"], "meta": {"k": 1}})
    assert folded == 'tag_items item_ids=["a", "b"] meta={"k": 1}'


def test_booleans_and_numbers_become_text():
    folded, _ = fold_tool_kwargs("list_items", {"done": True, "start": 0})
    assert folded == "list_items done=true start=0"


def test_an_empty_value_is_reported_not_dropped_silently():
    folded, empty = fold_tool_kwargs("list_items", {"args": {"filter": "", "start": None}})
    assert folded == "list_items"
    assert empty == ["filter had no value", "start had no value"]


def test_xml_command_text_gets_xml_tags():
    folded, _ = fold_tool_kwargs("move_item <item_uid>ab12</item_uid>", {"target": "x9"})
    assert folded == "move_item <item_uid>ab12</item_uid> <target>x9</target>"


def test_a_value_the_parser_would_cut_is_quoted():
    assert fold_tool_kwargs("note", {"text": "a, b,"})[0] == 'note text="a, b,"'
    assert fold_tool_kwargs("note", {"text": "call (twice)"})[0] == 'note text="call (twice)"'
    # A "(" or ";" anywhere in the value ends a code-run step's chain (fix-ajir.40).
    assert fold_tool_kwargs("note", {"text": "Sales; West"})[0] == 'note text="Sales; West"'
    assert fold_tool_kwargs("note", {"text": "Acme (EU) Ltd"})[0] == 'note text="Acme (EU) Ltd"'
    # A comma or ``=`` inside a value does not need quotes.
    assert fold_tool_kwargs("note", {"text": "a, b"})[0] == "note text=a, b"
    assert fold_tool_kwargs("note", {"text": "x=y"})[0] == "note text=x=y"


def test_a_lone_argument_named_for_no_field_is_a_bare_value():
    fields = frozenset({"item_uid"})
    assert fold_tool_kwargs("open_item", {"uid": "ab12"}, fields)[0] == "open_item ab12"
    # A key that is a field stays key=value.
    assert fold_tool_kwargs("open_item", {"item_uid": "ab12"}, fields)[0] == "open_item item_uid=ab12"
    # Several arguments keep their keys: which one is the positional value is not known.
    folded, _ = fold_tool_kwargs("open_item", {"uid": "ab12", "start": 2}, fields)
    assert folded == "open_item uid=ab12 start=2"
    # Unknown fields: keys are kept.
    assert fold_tool_kwargs("open_item", {"uid": "ab12"})[0] == "open_item uid=ab12"


# ----------------------------------------------------------------------
# The folded text binds through the key=value parser
# ----------------------------------------------------------------------


class _FindParams(BaseModel):
    name: str = Field(description="name")
    start: Optional[int] = Field(default=None, description="offset")


class _TagParams(BaseModel):
    item_ids: list[str] = Field(description="items")
    text: str = Field(default="", description="note")


def test_folded_text_binds_with_the_key_value_parser():
    folded, _ = fold_tool_kwargs("find_item", {"name": "Ann Lee", "start": 2})
    params = ParameterExtraction._extract_parameters_from_key_values(
        folded, _FindParams, "find_item")
    assert (params.name, params.start) == ("Ann Lee", 2)


def test_folded_list_and_quoted_values_bind():
    folded, _ = fold_tool_kwargs("tag_items", {"item_ids": ["a", "b"], "text": "a, b,"})
    params = ParameterExtraction._extract_parameters_from_key_values(
        folded, _TagParams, "tag_items")
    assert params.item_ids == ["a", "b"]
    assert params.text == "a, b,"


class _OpenParams(BaseModel):
    item_uid: str = Field(description="uid")


def test_a_lone_misnamed_argument_binds_with_the_positional_parser():
    folded, _ = fold_tool_kwargs("open_item", {"uid": "ab12"}, frozenset(_OpenParams.model_fields))
    params = ParameterExtraction._extract_parameters_from_positional(folded, _OpenParams, "open_item")
    assert params.item_uid == "ab12"


def test_a_key_that_is_no_field_does_not_bind_in_code():
    """Left to the LLM fallback and the usual refusals, as for any typed command."""
    folded, _ = fold_tool_kwargs("find_item", {"query": "Ann"})
    assert ParameterExtraction._extract_parameters_from_key_values(
        folded, _FindParams, "find_item") is None


# ----------------------------------------------------------------------
# The tool accepts the call
# ----------------------------------------------------------------------


def test_the_tool_still_shows_one_command_argument(session, agent):
    tool = agent.tools[TOOL]
    assert tool.args == {"command": {"type": "string"}}
    assert tool.has_kwargs


def test_the_tool_folds_instead_of_refusing(session, agent, responses):
    observation = agent.tools[TOOL](command="get_all_children", start="42")
    assert responses["calls"] == ["get_all_children start=42"]
    assert "Ran as: get_all_children start=42" in observation
    assert agent.run_state["folded_command"] == "get_all_children start=42"


def test_the_tool_folds_a_call_with_no_command(session, agent, responses):
    agent.tools[TOOL](args={"start": "1"}, command_name="get_all_children")
    assert responses["calls"] == ["get_all_children start=1"]


def test_parameters_with_no_command_name_are_refused_unrun(session, agent, responses):
    observation = agent.tools[TOOL](kind="item", item_id="ab12")
    assert responses["calls"] == []
    assert observation.startswith("No command name was given, so nothing ran.")
    assert "<command_name> kind=item item_id=ab12" in observation
    assert agent.run_state["folded_command"] == "kind=item item_id=ab12"


def test_the_tool_looks_up_the_command_fields(session, agent, responses, tmp_path):
    from tests.todo_list_workflow.application.todo_manager import TodoListManager

    workflow = session.get_active_workflow()
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo.json"))
    assert workflow_agent._parameter_fields("get_todo_list", workflow) == frozenset({"id"})
    assert workflow_agent._parameter_fields("no_such_command", workflow) is None
    agent.tools[TOOL](command="get_todo_list", list_id="3")
    assert responses["calls"] == ["get_todo_list 3"]
    agent.tools[TOOL](command="get_todo_list", id="3")
    assert responses["calls"][-1] == "get_todo_list id=3"


def test_a_plain_call_is_untouched(session, agent, responses):
    observation = agent.tools[TOOL](command="get_all_children")
    assert responses["calls"] == ["get_all_children"]
    assert "Ran as" not in observation
    assert "folded_command" not in agent.run_state


def test_an_empty_argument_is_named_in_the_observation(session, agent, responses):
    observation = agent.tools[TOOL](command="get_all_children", args={"filter": ""})
    assert "Not used: filter had no value." in observation


def test_a_repeated_folded_call_is_blocked_like_its_text(session, agent, responses):
    def step(**kwargs):
        result = agent.tools[TOOL](**kwargs)
        agent.on_step_end()
        return result

    step(command="get_all_children", start="1")
    step(command="get_all_children", start="1")  # same output: repeat
    assert "unavailable for this step" in step(command="get_all_children", start="1")
    assert len(responses["calls"]) == 2


def test_the_typed_text_of_a_folded_call_is_the_same_command(session, agent, responses):
    def step(**kwargs):
        result = agent.tools[TOOL](**kwargs)
        agent.on_step_end()
        return result

    step(command="get_all_children", start="1")
    step(command="get_all_children", start="1")  # same output: repeat
    assert "unavailable for this step" in step(command="get_all_children start=1")
    assert "unavailable" not in step(command="get_all_children", start="2")


def test_the_step_span_declares_folded_command():
    assert "folded_command" in tracing.SPAN_CONTRACTS[tracing.SPAN_AGENT_STEP].attributes


def test_the_react_loop_records_the_folded_command_on_the_step(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from fastworkflow.utils.react import fastWorkflowReAct
    from tests.test_tracing_phase1 import RecordingTraceSink, _make_agent_ctx

    sink = RecordingTraceSink()
    ctx, _ = _make_agent_ctx(todo_workflow_path, monkeypatch, sink=sink)
    ctx._begin_turn("find Ann")

    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.iteration_counter, agent.max_iters = 0, 5
    agent.inputs, agent.trajectory, agent.run_state = {}, {}, {}
    agent._dropped_steps, agent._suspended = 0, None

    def tool(command="", **extra):
        folded, _ = fold_tool_kwargs(command, extra)
        agent.run_state["folded_command"] = folded
        return f"ran {folded}"

    agent.tools = {TOOL: tool, "finish": lambda: "Completed."}
    script = [
        {"next_thought": "find", "next_tool_name": TOOL,
         "next_tool_args": {"command": "find_item", "name": "Ann"}},
        {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
    ]
    agent.react = lambda trajectory, **kw: SimpleNamespace(**script.pop(0))
    agent.extract = lambda trajectory, **kw: {"final_answer": "ok"}

    with tracing.host_scope(ctx):
        agent._run_loop({}, 0, {"q": "x"}, max_iters=5, exception_count=0)

    steps = sink.named(tracing.SPAN_AGENT_STEP)
    assert steps[0].attributes["folded_command"] == "find_item name=Ann"
    assert steps[0].attributes["tool_args"] == {"command": "find_item", "name": "Ann"}
    assert "folded_command" not in steps[1].attributes


def test_the_finish_check_counts_a_folded_step_as_run(session, agent, responses, tmp_path):
    """The step's archive row holds the folded text, so a planned step that writes
    an argument is met by a call that passed it beside ``command``."""
    from fastworkflow.observation_offloading import finish_check
    from fastworkflow.observation_offloading.archive import RuntimeHandleArchive
    from fastworkflow.observation_offloading.compact import archive_step
    from fastworkflow.observation_offloading.state import scope_for_host

    plan = "1. get_all_children start=42\n2. say how many there are"
    args = {"command": "get_all_children", "start": "42"}
    agent.trajectory = {"tool_name_0": TOOL, "tool_args_0": args}
    observation = agent.tools[TOOL](**args)
    agent.trajectory["observation_0"] = observation
    store = RuntimeHandleArchive(str(tmp_path / "archive.sqlite3"))
    scope = scope_for_host(session)
    assert archive_step(agent.trajectory, 0, scope=scope, selected_archive=store)

    rows = store.list(scope)
    assert [r["command"] for r in rows] == ["get_all_children start=42"]
    for values in (False, True):
        assert finish_check.check(plan, rows, ["get_all_children"], values=values).unmet == []
    # The bare ``command`` the model sent would have read as not run.
    bare = [{**rows[0], "command": args["command"]}]
    assert [v.step for v in finish_check.check(plan, bare, ["get_all_children"]).unmet] == [1]
