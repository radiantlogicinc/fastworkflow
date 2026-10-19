"""Finish check (fix-3xts): plan steps checked against what the run archived.

Offline, no model; the fixtures are a neutral library workflow.
"""
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


def test_review_without_room_to_act_says_nothing():
    assert fc.review_finish(plan_text=PLAN, floor=0, iters_left=1, commands=COMMANDS) is None


def test_review_without_an_archive_says_nothing(monkeypatch):
    from fastworkflow.observation_offloading import state

    monkeypatch.setattr(state, "durable_archive", lambda: None)
    monkeypatch.setattr(state, "record_event", lambda *a, **k: None)
    assert fc.review_finish(plan_text=PLAN, floor=0, iters_left=5, commands=COMMANDS) is None


def test_review_reads_the_runs_from_the_floor_on(monkeypatch):
    from types import SimpleNamespace

    from fastworkflow.observation_offloading import state

    events = []
    store = SimpleNamespace(list=lambda scope, timeout: ROWS_DONE[:4])
    monkeypatch.setattr(state, "current_scope", lambda: "scope")
    monkeypatch.setattr(state, "durable_archive", lambda: store)
    monkeypatch.setattr(state, "record_event", lambda event, **k: events.append(event))
    note = fc.review_finish(plan_text=PLAN, floor=1, iters_left=5, commands=COMMANDS,
                            core_commands=CORE)
    assert note.splitlines()[1] == "- Step 5 (list_loans) was not run: run it now."
    assert events[-1]["reason"] == "checked" and events[-1]["unmet_not_executed"] == [5]


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


def test_finish_with_arguments_ignores_them_and_is_reviewed(monkeypatch):
    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
    seen = []
    agent.finish_reviewer = lambda trajectory, left: seen.append(left) or "Step 2 was not run."
    scripted(agent, monkeypatch, [("finish", {"analysis": "done", "command": ""}),
                                  ("finish", {"final_answer": "x", "__args__": {}})])
    trajectory = {}
    assert agent._run_loop(trajectory, 0, {}, 10, 0) is None
    assert seen == [10 - 0 - 1]
    assert trajectory["observation_0"] == "Step 2 was not run."
    assert trajectory["observation_1"] == "Completed."
    assert "observation_2" not in trajectory


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

def test_the_execution_context_attaches_a_check_whenever_the_run_has_a_plan(monkeypatch):
    from types import SimpleNamespace

    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._workflow_tool_agent = SimpleNamespace(finish_reviewer="stale")
    ctx._app_workflow = SimpleNamespace(folderpath="/nowhere")
    ctx._initial_plan_text = PLAN

    seen = {}
    monkeypatch.setattr(fc, "review_finish", lambda **kwargs: seen.update(kwargs) or "note")
    import fastworkflow
    from fastworkflow import workflow_agent

    monkeypatch.setattr(fastworkflow, "RoutingRegistry", SimpleNamespace(
        get_definition=lambda folder: SimpleNamespace(
            routing_definition_map={c: set() for c in COMMANDS})))
    monkeypatch.setattr(workflow_agent, "core_command_names", lambda: frozenset(CORE))
    ctx._attach_finish_check(3)
    assert ctx._workflow_tool_agent.finish_reviewer({}, 7) == "note"
    assert (seen["plan_text"], seen["floor"], seen["iters_left"]) == (PLAN, 3, 7)

    ctx._initial_plan_text = ""
    ctx._attach_finish_check(0)
    assert ctx._workflow_tool_agent.finish_reviewer is None
