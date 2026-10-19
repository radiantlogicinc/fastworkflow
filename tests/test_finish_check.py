"""Finish check (fix-3xts): plan steps judged against what the run archived.

Offline: the model is a stub callable; the fixtures are a neutral library workflow.
"""
import json

import dspy

from fastworkflow.observation_offloading import finish_check as fc
from fastworkflow.utils.react import fastWorkflowReAct

COMMANDS = ["Library/find_book", "Library/open_book_by_uid", "Book/show_details",
            "Book/list_loans", "Book/add_note", "open_library", "wildcard"]
CORE = ["IntentDetection/go_up", "IntentDetection/reset_context", "IntentDetection/what_can_i_do"]

PLAN = """1. open_library
2. find_book query=Dune (take its uid from the match list)
3. open_book_by_uid with the uid of the Dune match
4. show_details (the book's author and its shelf location)
5. list_loans
6. only when the list is cut: next page with start=<n>
7. go_up
8. do not run add_note
9. say who wrote it and where it is shelved"""


def row(index, command, text):
    return {"step_index": index, "command": command, "text": text}


ROWS_DONE = [
    row(0, "open_library", "Entered Library context."),
    row(1, "find_book query=Dune", "1 book(s).\nb1  Dune"),
    row(2, "open_book_by_uid b1", "Entered Book context."),
    row(3, "show_details", "title: Dune\nauthor: Frank Herbert\nshelf: C4"),
    row(4, "list_loans", "2 loan(s); complete=true."),
]


def kinds(steps):
    return {s.number: (s.kind, s.command) for s in steps}


def test_the_plan_is_read_into_classified_steps():
    steps = fc.parse_plan(PLAN, COMMANDS, CORE)
    assert kinds(steps) == {
        1: ("command", "open_library"),
        2: ("command", "find_book"),
        3: ("command", "open_book_by_uid"),
        4: ("command", "show_details"),
        5: ("command", "list_loans"),
        6: ("conditional", ""),
        7: ("skipped", "go_up"),
        8: ("skipped", "add_note"),
        9: ("skipped", ""),
    }


def test_a_bullet_plan_is_numbered_in_order():
    steps = fc.parse_plan("- `open_library`\n- `show_details` for the shelf", COMMANDS)
    assert [(s.number, s.command) for s in steps] == [(1, "open_library"), (2, "show_details")]


def test_a_step_names_its_workflow_command_before_a_navigation_one():
    steps = fc.parse_plan("1. for each book: a. go_up b. open_book_by_uid <uid>", COMMANDS, CORE)
    assert steps[0].command == "open_book_by_uid"


def test_a_command_step_that_never_ran_is_unmet():
    result = fc.check(PLAN, ROWS_DONE[:3], COMMANDS, core_commands=CORE)
    assert [(v.step, v.missing, v.next_command) for v in result.unmet] == [
        (4, fc.NOT_EXECUTED, "show_details"), (5, fc.NOT_EXECUTED, "list_loans")]
    assert "Step 4 (show_details) was not run" in result.note
    assert len(result.note.encode("utf-8")) <= fc.NOTE_MAX_BYTES


def test_a_failed_run_does_not_count_as_a_run():
    rows = ROWS_DONE[:3] + [
        row(3, "show_details", "Execution error in execute_workflow_query: boom"),
        row(4, "list_loans", "Command 'list_loans' is not available in the current context 'Library'"),
    ]
    result = fc.check(PLAN, rows, COMMANDS, core_commands=CORE)
    assert {v.step for v in result.unmet} == {4, 5}


def test_a_command_run_twice_satisfies_two_steps_and_a_missing_third_is_named():
    plan = "1. show_details\n2. go_up\n3. show_details\n4. go_up\n5. show_details"
    rows = [row(0, "show_details", "a"), row(2, "show_details", "b")]
    result = fc.check(plan, rows, COMMANDS, core_commands=CORE)
    assert [v.step for v in result.unmet] == [5]


def test_setup_steps_done_in_an_earlier_run_count():
    earlier = ROWS_DONE[:3]
    rows = [row(5, "show_details", "author: Frank Herbert"), row(6, "list_loans", "0 loans")]
    plan = "1. open_library\n2. open_book_by_uid with the uid\n3. show_details\n4. list_loans"
    assert fc.check(plan, rows, COMMANDS, earlier_rows=earlier).unmet == []
    assert [v.step for v in fc.check(plan, rows, COMMANDS).unmet] == [1, 2]


def test_the_command_is_read_from_qualified_and_odd_forms():
    names = {"show_details", "open_library"}
    assert fc.command_token("Book/show_details start=3", names) == "show_details"
    assert fc.command_token("command open_library", names) == "open_library"
    assert fc.command_token("show_details<Dune>", names) == "show_details"


def stub(reply):
    calls = []

    def ask(system, user):
        calls.append(user)
        return reply if isinstance(reply, str) else json.dumps(reply)

    ask.calls = calls
    return ask


def test_the_model_is_asked_only_about_steps_that_ask_for_something():
    ask = stub({"steps": []})
    result = fc.check(PLAN, ROWS_DONE, COMMANDS, core_commands=CORE, ask=ask)
    assert result.model_called and result.unmet == []
    judged = [s.number for s in fc.judged_steps(result.steps)]
    assert judged == [4, 6]
    assert "### Step 4" in ask.calls[0] and "### Step 5" not in ask.calls[0]


def test_an_unmet_part_the_output_lacks_is_kept_with_a_real_command():
    rows = ROWS_DONE[:3] + [row(3, "show_details", "title: Dune\nauthor: Frank Herbert"),
                           row(4, "list_loans", "0 loans")]
    ask = stub({"steps": [{"step": 4, "verdict": "unmet", "missing": "its shelf location",
                           "next_command": "list_loans"}]})
    result = fc.check(PLAN, rows, COMMANDS, core_commands=CORE, ask=ask)
    assert [(v.step, v.missing, v.next_command, v.source) for v in result.unmet] == [
        (4, "its shelf location", "list_loans", "model")]
    assert 'asks for "its shelf location" and O3 does not show it' in result.note


def test_invented_unmets_are_refused():
    ask = stub({"steps": [
        {"step": 4, "verdict": "unmet", "missing": "the publisher", "next_command": ""},
        {"step": 4, "verdict": "unmet", "missing": "author", "next_command": ""},
    ]})
    result = fc.check(PLAN, ROWS_DONE, COMMANDS, core_commands=CORE, ask=ask)
    assert result.unmet == []
    assert [v["reason"] for v in result.vetoes] == ["quote not in step", "words in output"]


def test_a_quote_of_the_command_or_the_answer_is_refused():
    plan = "1. show_details (the author and shelf)\n2. show_details and present the author as a list"
    rows = [row(0, "show_details", "title: Dune"), row(1, "show_details", "title: Dune")]
    ask = stub({"steps": [
        {"step": 1, "verdict": "unmet", "missing": "show_details (the author and shelf)"},
        {"step": 2, "verdict": "unmet", "missing": "present the author as a list"},
    ]})
    result = fc.check(plan, rows, COMMANDS, ask=ask)
    assert [v["reason"] for v in result.vetoes] == [
        "quote names the command", "quote is about the answer"]


def test_a_suggestion_that_is_not_a_command_or_already_ran_is_blanked():
    rows = ROWS_DONE[:3] + [row(3, "show_details", "title: Dune")]
    for suggestion in ("fetch_shelf", "show_details"):
        ask = stub({"steps": [{"step": 4, "verdict": "unmet", "missing": "shelf location",
                               "next_command": suggestion}]})
        result = fc.check(PLAN, rows, COMMANDS, core_commands=CORE, ask=ask)
        model = [v for v in result.unmet if v.source == "model"]
        assert model and model[0].next_command == ""


def test_a_model_failure_leaves_the_command_pass_standing():
    def broken(system, user):
        raise TimeoutError("slow")

    result = fc.check(PLAN, ROWS_DONE[:4], COMMANDS, core_commands=CORE, ask=broken)
    assert result.error.startswith("TimeoutError")
    assert [v.step for v in result.unmet] == [5]
    garbled = fc.check(PLAN, ROWS_DONE, COMMANDS, core_commands=CORE, ask=stub("no json"))
    assert garbled.error and garbled.unmet == [] and garbled.note is None


def test_the_reply_may_wrap_its_json_in_reasoning():
    reply = '<think>{"x": 1}</think>Here: {"steps": [{"step": 4, "verdict": "satisfied"}]}'
    assert fc.parse_response(reply) == [{"step": 4, "verdict": "satisfied"}]


def test_a_long_output_is_cut_to_head_tail_and_the_lines_naming_the_step():
    body = "\n".join([f"row {i} filler text" for i in range(400)] + ["shelf: C4"]
                     + [f"row {i} filler" for i in range(400)])
    excerpt = fc.output_excerpt(body, "the shelf location", 2048)
    assert "shelf: C4" in excerpt and len(excerpt.encode("utf-8")) < 2600


def test_the_prompt_stays_near_its_budget():
    rows = [row(3, "show_details", "x" * 50_000)] + ROWS_DONE[:3]
    result = fc.check(PLAN, rows, COMMANDS, core_commands=CORE, ask=stub({"steps": []}),
                      max_bytes=8192)
    assert len(result.prompt.encode("utf-8")) < 8192 + 1024


def test_the_mode_comes_from_the_switch(monkeypatch):
    monkeypatch.delenv(fc.CHECK_ENV, raising=False)
    assert fc.mode() is None
    monkeypatch.setenv(fc.CHECK_ENV, "1")
    assert fc.mode() == fc.MODE_FULL
    monkeypatch.setenv(fc.CHECK_ENV, "commands")
    assert fc.mode() == fc.MODE_COMMANDS
    monkeypatch.setenv(fc.CHECK_ENV, "off")
    assert fc.mode() is None


def test_review_without_room_to_act_says_nothing():
    assert fc.review_finish(plan_text=PLAN, request="", floor=0, iters_left=1,
                            commands=COMMANDS) is None


# ---------------------------------------------------------------------------
# The ReAct hook
# ---------------------------------------------------------------------------

def a_tool(command: str) -> str:
    """Runs a command."""
    return f"ran {command}"


def scripted(agent, monkeypatch, tools):
    replies = iter([dspy.Prediction(next_thought="", next_tool_name=name,
                                    next_tool_args=args) for name, args in tools])
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation",
                        lambda *a, **k: next(replies))


def test_a_note_reopens_the_loop_once_and_the_second_finish_ends_it(monkeypatch):
    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
    seen = []
    agent.finish_reviewer = lambda trajectory, left: seen.append(left) or "Step 2 was not run."
    scripted(agent, monkeypatch, [("a_tool", {"command": "x"}), ("finish", {}),
                                  ("a_tool", {"command": "y"}), ("finish", {})])
    trajectory = {}
    assert agent._run_loop(trajectory, 0, {}, 10, 0) is None
    assert trajectory["observation_1"] == "Step 2 was not run."
    assert trajectory["observation_2"] == "ran y"
    assert trajectory["observation_3"] == "Completed."
    assert seen == [10 - 1 - 1]


def test_no_reviewer_or_no_note_finishes_at_once(monkeypatch):
    for reviewer in (None, lambda trajectory, left: None):
        agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
        agent.finish_reviewer = reviewer
        scripted(agent, monkeypatch, [("finish", {})])
        trajectory = {}
        assert agent._run_loop(trajectory, 0, {}, 10, 0) is None
        assert trajectory["observation_0"] == "Completed."


def test_a_failing_reviewer_never_fails_the_run(monkeypatch):
    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)

    def broken(trajectory, left):
        raise RuntimeError("archive locked")

    agent.finish_reviewer = broken
    scripted(agent, monkeypatch, [("finish", {})])
    trajectory = {}
    assert agent._run_loop(trajectory, 0, {}, 10, 0) is None
    assert trajectory["observation_0"] == "Completed."


# ---------------------------------------------------------------------------
# The execution context attaches the check per run
# ---------------------------------------------------------------------------

def test_the_execution_context_attaches_a_check_only_when_switched_on(monkeypatch):
    from types import SimpleNamespace

    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._workflow_tool_agent = SimpleNamespace(finish_reviewer="stale")
    ctx._app_workflow = SimpleNamespace(folderpath="/nowhere")
    ctx._initial_plan_text = PLAN

    monkeypatch.delenv(fc.CHECK_ENV, raising=False)
    ctx._attach_finish_check("find Dune", 0)
    assert ctx._workflow_tool_agent.finish_reviewer is None

    monkeypatch.setenv(fc.CHECK_ENV, "commands")
    seen = {}
    monkeypatch.setattr(fc, "review_finish", lambda **kwargs: seen.update(kwargs) or "note")
    import fastworkflow
    from fastworkflow import workflow_agent

    monkeypatch.setattr(fastworkflow, "RoutingRegistry", SimpleNamespace(
        get_definition=lambda folder: SimpleNamespace(
            routing_definition_map={c: set() for c in COMMANDS})))
    monkeypatch.setattr(workflow_agent, "core_command_names", lambda: frozenset(CORE))
    ctx._attach_finish_check("find Dune", 3)
    assert ctx._workflow_tool_agent.finish_reviewer({}, 7) == "note"
    assert (seen["plan_text"], seen["floor"], seen["iters_left"], seen["use_model"]) == (PLAN, 3, 7, False)

    ctx._initial_plan_text = ""
    ctx._attach_finish_check("find Dune", 0)
    assert ctx._workflow_tool_agent.finish_reviewer is None
