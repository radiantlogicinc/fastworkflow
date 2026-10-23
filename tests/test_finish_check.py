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


# A resolved alias, a "(skip when ...)" step, a bare run against a step with an argument

def test_an_alias_counts_for_the_command_it_resolved_to():
    plan = "1. open_book_by_uid with the uid\n2. show_details"
    rows = [row(0, "open_book_by_uid b1", "Entered Book context."),
            row(1, "describe_book", "author: Frank Herbert")]
    assert [v.step for v in fc.check(plan, rows, COMMANDS).unmet] == [2]
    log = [{"command": "open_book_by_uid b1", "command_name": "Library/open_book_by_uid",
            "response": "Entered Book context."},
           {"command": "describe_book", "command_name": "Book/show_details",
            "response": "author: Frank Herbert"}]
    resolved = fc.resolved_command_names(log)
    assert resolved["describe_book"] == "show_details"
    assert fc.check(plan, rows, COMMANDS, resolved=resolved).unmet == []


def test_a_text_resolving_to_two_commands_keeps_its_own_name():
    log = [{"command": "details", "command_name": "Book/show_details"},
           {"command": "details", "command_name": "Library/find_book"},
           {"command": "x", "command_name": None}]
    assert fc.resolved_command_names(log) == {}


def test_review_passes_the_resolved_names_on(monkeypatch):
    from types import SimpleNamespace

    from fastworkflow.observation_offloading import state

    store = SimpleNamespace(list=lambda scope, timeout: [row(0, "describe_book", "author: X")])
    monkeypatch.setattr(state, "current_scope", lambda: "scope")
    monkeypatch.setattr(state, "durable_archive", lambda: store)
    monkeypatch.setattr(state, "record_event", lambda *a, **k: None)
    plan = "1. show_details\n2. list_loans"
    kwargs = dict(plan_text=plan, floor=0, iters_left=5, commands=COMMANDS)
    assert "Step 1" in fc.review_finish(**kwargs)
    note = fc.review_finish(**kwargs, resolved={"describe_book": "show_details"})
    assert "Step 1" not in note and "Step 2 (list_loans)" in note


def test_a_step_with_skip_when_after_its_command_is_conditional():
    plan = ("1. show_details\n"
            "2. list_loans book_uid=<the uid> (skip when it has no loans)\n"
            "3. add_note (skip if the book has a note already)")
    steps = fc.parse_plan(plan, COMMANDS, CORE)
    assert [(s.kind, s.command) for s in steps] == [
        ("command", "show_details"), ("conditional", "list_loans"), ("conditional", "add_note")]
    assert fc.check(plan, [row(0, "show_details", "a")], COMMANDS, core_commands=CORE).unmet == []


def test_a_condition_word_in_the_chain_after_the_command_is_conditional():
    # fix-ajir.32: the condition is read in the chain (after the command, up to the first
    # "(" or ";"), so it never runs as part of the last value; a note is never read.
    plan = ("1. find_book query=Dune only when needed\n"
            "2. open_book_by_uid with the uid of the Dune match if there is one\n"
            "3. only when needed, find_book query=Dune\n"
            "4. find_book query=Ann Lee\n"
            "5. find_book query=Dune (only when overdue)\n"
            "6. list_loans; only the first page")
    steps = fc.parse_plan(plan, COMMANDS, CORE)
    assert [(s.kind, s.command) for s in steps] == [
        ("conditional", "find_book"), ("conditional", "open_book_by_uid"),
        ("conditional", "find_book"), ("command", "find_book"), ("command", "find_book"),
        ("command", "list_loans")]
    assert fc.plain_value(steps[3].arguments["query"]) == "ann lee"


def test_a_step_that_writes_an_argument_reads_it():
    steps = fc.parse_plan("1. list_loans book_uid=<the first book>\n2. list_loans\n"
                          "3. list_loans with the uid\n4. find_book query=Dune", COMMANDS)
    assert [s.wants_arguments for s in steps] == [True, False, False, True]
    assert fc.has_arguments("Book/list_loans", {"list_loans"}) is False
    assert fc.has_arguments("list_loans book_uid=b1", {"list_loans"}) is True
    assert fc.has_arguments("list_loans <book_uid>b1</book_uid>", {"list_loans"}) is True


def test_a_bare_run_does_not_satisfy_a_step_that_names_an_argument():
    plan = "1. list_loans (the roster)\n2. list_loans book_uid=<the first book's uid>\n3. show_details"
    bare = [row(0, "list_loans", "2 loan(s)"), row(1, "show_details", "a"),
            row(2, "list_loans", "2 loan(s)")]
    result = fc.check(plan, bare, COMMANDS)
    assert [(v.step, v.next_command) for v in result.unmet] == [(2, "list_loans")]
    with_args = bare[:2] + [row(2, "list_loans book_uid=b1", "1 loan(s)")]
    assert fc.check(plan, with_args, COMMANDS).unmet == []
    # The run with arguments goes to the step that needs it, whatever the order.
    swapped = [row(0, "list_loans book_uid=b1", "1 loan(s)"), row(1, "list_loans", "2 loan(s)"),
               row(2, "show_details", "a")]
    assert fc.check(plan, swapped, COMMANDS).unmet == []


def test_a_bare_earlier_run_does_not_satisfy_a_setup_step_with_an_argument():
    plan = "1. find_book query=Dune\n2. show_details"
    rows = [row(5, "show_details", "a")]
    assert fc.check(plan, rows, COMMANDS, earlier_rows=[row(0, "find_book", "x")]).unmet[0].step == 1
    assert fc.check(plan, rows, COMMANDS, earlier_rows=[row(0, "find_book Dune", "x")]).unmet == []


def test_a_step_value_must_be_the_one_the_run_carries():
    steps = fc.parse_plan("1. find_book query=Dune Messiah (the sequel)\n"
                          "2. open_book_by_uid book_uid=<the uid>\n3. list_loans start=<n>", COMMANDS)
    assert [s.arguments for s in steps] == [{"query": "dune messiah"}, {}, {}]
    find = steps[0]
    # Archive rows: the value as a word of the command text, compared case-blind.
    assert fc.satisfies(find, row(0, "find_book query='dune  messiah'", "x"), {"find_book"})
    assert not fc.satisfies(find, row(0, "find_book query=Emma", "x"), {"find_book"})
    # Command runs: the named parameter; another page of the same query still does it.
    assert fc.satisfies(find, {"command": "Library/find_book", "input": {"query": "Dune Messiah"}})
    assert fc.satisfies(find, {"command": "find_book", "input": {"query": "Dune Messiah", "start": 20}})
    assert not fc.satisfies(find, {"command": "find_book", "input": {"query": "Emma", "start": None}})
    assert not fc.satisfies(find, {"command": "find_book", "input": {"query": None}})
    # A placeholder names no value: any run with arguments does.
    assert fc.satisfies(steps[1], {"command": "open_book_by_uid", "input": {"book_uid": "b7"}})
    assert not fc.satisfies(steps[1], {"command": "open_book_by_uid", "input": {"book_uid": None}})


def test_in_a_phase_an_off_plan_lookup_does_not_meet_a_step_with_another_value():
    plan = "1. find_book query=Dune\n2. show_details"
    rows = [row(0, "find_book query=Emma", "1 book(s)"), row(1, "show_details", "a")]
    assert [v.step for v in fc.check(plan, rows, COMMANDS, values=True).unmet] == [1]
    rows.append(row(2, "find_book Dune", "1 book(s)"))
    assert fc.check(plan, rows, COMMANDS, values=True).unmet == []


def test_outside_a_phase_the_check_does_not_compare_values():
    # The stage-7 rule: any run with arguments meets a step that writes one.
    plan = "1. find_book query=Jesse Rodgers-ADM\n2. show_details"
    rows = [row(0, "find_book query=Jesse Rodgers", "1 book(s)"), row(1, "show_details", "a")]
    assert fc.check(plan, rows, COMMANDS).unmet == []
    assert [v.step for v in fc.check(plan, rows, COMMANDS, values=True).unmet] == [1]
    # The bare-run rule still holds.
    bare = [row(0, "find_book", "1 book(s)"), row(1, "show_details", "a")]
    assert [v.step for v in fc.check(plan, bare, COMMANDS).unmet] == [1]


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
    ctx._action_log = [{"command": "describe_book", "command_name": "Book/show_details",
                        "response": "a"}]

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
    assert seen["resolved"] == {"describe_book": "show_details"}

    ctx._initial_plan_text = ""
    ctx._attach_finish_check(0)
    assert ctx._workflow_tool_agent.finish_reviewer is None


def test_a_step_value_ends_at_comma_prose_and_an_identifier_at_its_token():
    uid = "902d138bd2eb570305bc1019007f4922"
    args = fc.step_arguments
    # An identifier value is its first token; prose after it is not the value.
    assert args(f" scope=shelf scope_uid={uid}, no filter, no start") == {
        "scope": "shelf", "scope_uid": uid}
    assert args(f" book_uid={uid} with no filter") == {"book_uid": uid}
    # A name of several words is kept whole, up to ", " prose.
    assert args(" query=Ann Lee") == {"query": "ann lee"}
    assert args(" query=Ann Lee, the second match") == {"query": "ann lee"}
    assert args(" query=Shelf_Reading Room 2") == {"query": "shelf_reading room 2"}
    assert args(" title=Dune, author=Herbert") == {"title": "dune", "author": "herbert"}
    # A quoted value is kept whole: its "(", ";" and ", " are the value's (fix-ajir.40).
    assert args(' query="Acme (EU)" (all of them)') == {"query": "acme (eu)"}
    assert args(" query='Sales; West' start=20") == {"query": "sales; west", "start": "20"}
    assert args(' query="Lee, Ann", the second match') == {"query": "lee, ann"}
    # Unquoted, or with an unclosed quote, a value still ends at "(" or ";".
    assert args(" query=Acme (EU)") == {"query": "acme"}
    assert args(' query="Acme (EU)') == {"query": "acme"}
    assert args(" query=O'Brien (only when needed)") == {"query": "o'brien"}
    run = {"command": "list_loans", "success": True,
           "input": {"scope": "shelf", "scope_uid": uid, "filter": None}}
    step = fc.parse_plan(f"1. list_loans scope=shelf scope_uid={uid}, no filter, no start",
                         COMMANDS)[0]
    assert fc.satisfies(step, run)
