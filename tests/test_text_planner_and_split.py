"""Plain-text planner: signature choice and span attributes."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.workflow_agent import (
    build_query_with_next_steps, initialize_workflow_tool_agent, run_progress_check)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

TODO_WORKFLOW = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())
COMMANDS_TITLE = "Available execute_workflow_query tool commands"
STRUCTURED_MARK = "Return the plan as structured steps"
TEXT_MARK = "[[ ## next_steps ## ]]"


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


@pytest.fixture
def workflow_env(tmp_path, monkeypatch):
    """Load settings from a real fastworkflow.env / passwords file pair, process env cleared."""
    RoutingRegistry.clear_registry()

    def load(**settings: str) -> None:
        env_file = tmp_path / "fastworkflow.env"
        passwords_file = tmp_path / "fastworkflow.passwords.env"
        env_file.write_text("".join(f"{name}={value}\n" for name, value in settings.items()))
        passwords_file.write_text("")
        fastworkflow.init(env_vars={**dotenv_values(env_file), **dotenv_values(passwords_file)})

    yield load
    RoutingRegistry.clear_registry()


def _session(sink, folder=TODO_WORKFLOW):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"planner-{uuid.uuid4().hex}")
    ctx.bind_app_workflow(wf)
    ctx._workflow_tool_agent = initialize_workflow_tool_agent(ctx)
    return ctx, wf


def _plan(ctx, wf, lm, **kwargs):
    ctx.push_active_workflow(wf)
    try:
        return build_query_with_next_steps("show my todo items", ctx, planner_lm=lm, **kwargs)
    finally:
        ctx.pop_active_workflow()


def _prompt(lm, index: int) -> str:
    return "\n".join(str(m.get("content", "")) for m in lm.history[index]["messages"])


def _text_answer():
    return {"reasoning": "r", "next_steps": "1. Show all todo items"}


def _replan_answer():
    return {"reasoning": "r", "whats_done": "Listed the todo items", "next_steps": "1. Show all todo items"}


def test_the_text_planner_runs_and_appends_next_steps(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    assert COMMANDS_TITLE in _prompt(lm, 0)
    assert "Show all todo items" in result
    assert "Execute these next steps:" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert plan.attributes["plan_source"] == "text"


def test_a_replan_uses_the_text_planner(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_replan_answer(), _replan_answer()])
    result = _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="ask_user_response")

    assert len(lm.history) == 1
    assert result == (
        "User Query:\nshow my todo items\n\n"
        "What's done:\nListed the todo items\n\n"
        "What remains (execute these next steps):\n1. Show all todo items"
    )
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["plan_source"] == "text"
    assert not sink.named(tracing.SPAN_PLANNER_PLAN)


def test_the_progress_check_tool_replans(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._current_planner_lm = DummyLM([_replan_answer()])
    agent = ctx._workflow_tool_agent
    agent.iteration_counter = 24

    ctx._begin_turn("show my todo items")
    ctx.push_active_workflow(wf)
    try:
        result = run_progress_check(ctx, "progress_check")
    finally:
        ctx.pop_active_workflow()

    assert result.startswith("What's done:\nListed the todo items")
    assert "What remains (execute these next steps):\n1. Show all todo items" in result
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["replan_trigger"] == "progress_check"


def test_a_progress_check_with_nothing_remaining_says_to_finish(workflow_env):
    workflow_env()
    ctx, wf = _session(RecordingTraceSink())

    ctx._begin_turn("show my todo items")
    lm = DummyLM([{"reasoning": "r", "whats_done": "Listed the todo items", "next_steps": ""}])
    result = _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="progress_check")

    assert result.endswith("What's done:\nListed the todo items\n\nNothing left to do. Call finish now.")



def test_a_zero_argument_tool_ignores_a_stray_argument(workflow_env, monkeypatch):
    import fastworkflow.workflow_agent as workflow_agent

    workflow_env()
    ctx, wf = _session(RecordingTraceSink())
    monkeypatch.setattr(workflow_agent, "_what_can_i_do", lambda chat_session_obj: "the commands")
    tool = ctx._workflow_tool_agent.tools["what_can_i_do"]

    assert tool.args == {}
    assert tool(clarification_request="list the commands please") == "the commands"


def _reentry_session(monkeypatch, workflow_env):
    import fastworkflow.workflow_agent as workflow_agent
    from fastworkflow.command_executor import AlreadyInContextError

    workflow_env()
    ctx, wf = _session(RecordingTraceSink())
    calls = []
    state = {"inside": True}

    def fake_execute(command, chat_session_obj):
        calls.append(command)
        if command == "go_up":
            state["inside"] = False
            return "Context is now 'ItemExplorer'"
        if state["inside"]:
            raise AlreadyInContextError("You are already in context 'Account' (a1 Jane Roe)")
        return "Entered Account context."

    monkeypatch.setattr(workflow_agent, "_execute_workflow_query", fake_execute)
    monkeypatch.setattr(workflow_agent, "context_clause_for", lambda wf: "Account a1 Jane Roe")
    return ctx, wf, calls


def test_opening_another_instance_of_the_current_context_goes_up_first(workflow_env, monkeypatch):
    ctx, wf, calls = _reentry_session(monkeypatch, workflow_env)
    execute = ctx._workflow_tool_agent.tools["execute_workflow_query"]

    observation = execute(command="open_account_by_uid b2")

    assert calls == ["open_account_by_uid b2", "go_up", "open_account_by_uid b2"]
    assert "(Left the current Account with go_up first.)\nEntered Account context." in observation


def test_reopening_the_same_instance_is_refused_without_moving(workflow_env, monkeypatch):
    ctx, wf, calls = _reentry_session(monkeypatch, workflow_env)
    execute = ctx._workflow_tool_agent.tools["execute_workflow_query"]

    observation = execute(command="open_account_by_uid a1")

    assert calls == ["open_account_by_uid a1"]
    assert "You are already in context 'Account'" in observation


def _split(answer):
    from fastworkflow.workflow_agent import split_into_subtasks
    return split_into_subtasks("the request", planner_lm=DummyLM([answer]))


def test_a_numbered_answer_becomes_the_subtasks():
    assert _split({"reasoning": "r", "subtasks": "1. Audit right A.\n2) Audit right B.\n  3. Sweep Hubbard."}) == [
        "Audit right A.", "Audit right B.", "Sweep Hubbard."]


def test_an_unusable_split_leaves_the_request_whole():
    from fastworkflow.workflow_agent import MAX_SUBTASKS

    assert _split({"reasoning": "r", "subtasks": "no numbered lines here"}) == ["the request"]
    many = "\n".join(f"{n}. part {n}" for n in range(1, MAX_SUBTASKS + 2))
    # too many sub-tasks is no reason to run the request whole: the list is reduced to the cap (see
    # test_a_split_over_the_cap_is_reduced_never_run_whole)
    assert len(_split({"reasoning": "r", "subtasks": many})) <= MAX_SUBTASKS


def test_a_failing_split_leaves_the_request_whole(monkeypatch):
    import dspy
    from fastworkflow.workflow_agent import split_into_subtasks

    def broken(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(dspy, "ChainOfThought", broken)
    assert split_into_subtasks("the request", planner_lm=DummyLM([{}])) == ["the request"]


def test_a_visibly_broken_split_is_retried_once_then_left_whole():
    from fastworkflow.workflow_agent import split_into_subtasks

    duplicated = {"reasoning": "r", "subtasks": "1. Same thing.\n2. Same thing."}
    fixed = {"reasoning": "r", "subtasks": "1. Part A.\n2. Part B."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([duplicated, fixed])) == [
        "Part A.", "Part B."]
    assert split_into_subtasks("the request", planner_lm=DummyLM([duplicated, duplicated])) == [
        "the request"]


def test_a_split_that_refers_back_to_the_request_is_retried_once():
    from fastworkflow.workflow_agent import split_into_subtasks

    referring = {"reasoning": "r", "subtasks": "1. Audit each of them in Finance.\n2. Sweep Hubbard."}
    named = {"reasoning": "r", "subtasks": "1. Audit Alice and Bob.\n2. Sweep Hubbard."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([referring, named])) == [
        "Audit Alice and Bob.", "Sweep Hubbard."]
    # Not fixed by the retry: a split with a leftover reference beats no split.
    assert split_into_subtasks("the request", planner_lm=DummyLM([referring, referring])) == [
        "Audit each of them in Finance.", "Sweep Hubbard."]


def test_a_single_sub_task_is_the_request_as_the_user_wrote_it():
    # The model's rewrite of a one-job request dropped its leading record phrase: the request comes back whole.
    query = ("For Ada Ng's card, list the fines it carries with the payments that would clear them, and every "
             "renewal option open to it. Renew nothing.")
    answer = {"reasoning": "r", "subtasks": "1. List the fines the card carries with the payments that would clear "
                                            "them, and every renewal option open to it. Renew nothing."}
    assert _split_with_query(query, answer) == [query]


def test_a_numbered_list_written_inline_after_reasoning_gives_one_sub_task_per_item():
    query = "Renew the loan of Ada Ng and check its fines. Also list the holds of Ben Ito."
    answer = {"reasoning": "r", "subtasks": (
        "Two jobs: the renewal, and the holds. 1. Renew the loan of Ada Ng and check its fines. "
        "2. List the holds of Ben Ito.   3. List the holds of Ben Ito.   {")}
    assert _split_with_query(query, answer, answer) == [
        "Renew the loan of Ada Ng and check its fines.", "List the holds of Ben Ito."]


def test_a_clause_the_model_wrote_twice_in_one_sub_task_is_written_once():
    query = ("Ada Ng has asked for the reading room pass: check the pass and what Ada Ng can reach today. "
             "Ben Ito has asked for the same pass: check it too.")
    answer = {"reasoning": "r", "subtasks": (
        "1. Ada Ng has asked for the reading room pass: check the pass and what Ada Ng has asked for the reading "
        "room pass: check the pass and what Ada Ng can reach today.\n"
        "2. Ben Ito has asked for the same pass: check it too.")}
    assert _split_with_query(query, answer) == [
        "Ada Ng has asked for the reading room pass: check the pass and what Ada Ng can reach today.",
        "Ben Ito has asked for the same pass: check it too."]


def _split_with_query(query, *answers):
    from fastworkflow.workflow_agent import split_into_subtasks

    return split_into_subtasks(query, planner_lm=DummyLM(list(answers)))


def test_the_split_goes_through_the_merge():
    from fastworkflow.workflow_agent import split_into_subtasks

    query = "How risky is Zoe Hale, and which findings are driving that?"
    follow_up = {"reasoning": "r", "subtasks": "1. How risky is Zoe Hale?\n2. Which findings are driving that?"}
    assert split_into_subtasks(query, planner_lm=DummyLM([follow_up])) == [query]


def _split_file(folder, text):
    (folder / "_commands").mkdir(parents=True, exist_ok=True)
    (folder / "_commands" / "_split.py").write_text(text)
    return str(folder)


def test_identifier_patterns_come_from_the_workflows_split_file(tmp_path):
    from fastworkflow.workflow_agent import SplitSettings, identifier_patterns, split_settings

    assert identifier_patterns(str(tmp_path / "no_file")) == ()
    assert split_settings(str(tmp_path / "no_file")) == SplitSettings()
    folder = _split_file(tmp_path, "import re\n\n# case numbers\n"
                                   "identifier_patterns = [re.compile(r'\\bcase_[A-Za-z0-9]+')]\n")
    assert [p.pattern for p in identifier_patterns(folder)] == [r"\bcase_[A-Za-z0-9]+"]


@pytest.mark.parametrize("text, match", [
    ("import re\nidentifier_patterns = [re.compile('[unclosed')]\n", "could not import the split settings"),
    ("identifier_patterns = [r'\\bcase_[0-9]+']\n", "write re.compile"),
    ("import re\nidentifier_patterns = re.compile('x')\n", "must be a list of compiled patterns"),
    ("subject_hints = ['not', 'text']\n", "subject_hints must be a str"),
    ("Signature = 'text'\n", "Signature must be a class"),
])
def test_a_malformed_split_file_fails_the_load(tmp_path, text, match):
    from fastworkflow.planning_commands import PlanningCommandFormatError
    from fastworkflow.workflow_agent import load_planning_commands

    folder = _split_file(tmp_path, text)
    with pytest.raises(PlanningCommandFormatError, match=match):
        load_planning_commands(folder)   # the skills carry the identifier patterns, so the split file loads with them


def test_the_merge_gets_the_workflows_identifier_patterns(tmp_path, monkeypatch):
    import fastworkflow.workflow_agent as agent

    _split_file(tmp_path, "import re\nidentifier_patterns = [re.compile(r'\\bcase_[A-Za-z0-9]+')]\n")
    seen = {}

    def merge(user_query, subtasks, identifiers=(), **_):
        seen["identifiers"] = [p.pattern for p in identifiers]
        return subtasks

    monkeypatch.setattr(agent, "merge_subtasks", merge)
    agent.split_into_subtasks("the request", planner_lm=DummyLM([
        {"reasoning": "r", "subtasks": "1. Explain case_X7kQ2Lm9.\n2. Explain case_B4nW8Rt2."}]),
        workflow_folderpath=str(tmp_path))
    assert seen["identifiers"] == [r"\bcase_[A-Za-z0-9]+"]


def test_a_split_sub_task_gets_the_referent_jev_chooses_written_in(tmp_path, monkeypatch):
    import fastworkflow.workflow_agent as agent

    _split_file(tmp_path, 'subject_hints = """Tag names are not records."""\n')
    asked = []

    def choose(user_query, text, candidates, notes, jev_lm):
        asked.append((text, candidates, notes))
        return "Ann Lee" if "on-leave" in text else None

    monkeypatch.setattr(agent, "_jev_lm", lambda: object())
    monkeypatch.setattr(agent, "_choose_referent", choose)
    monkeypatch.setattr(agent, "_choose_description", lambda *args: None)
    q = "Ann Lee leaves on Friday: sweep her reach. Then tag the customer `on-leave` and show its tags."
    assert agent.resolve_referents(q, ["Ann Lee leaves on Friday: sweep her reach.",
                                       "Tag the customer `on-leave` and show its tags."], str(tmp_path)) == [
        "Ann Lee leaves on Friday: sweep her reach.", "Tag the customer `on-leave` and show Ann Lee's tags."]
    # Jev is asked about each sub-task that leaves request subjects unnamed, with the workflow's notes
    assert asked == [("Ann Lee leaves on Friday: sweep her reach.", ["on-leave"], "Tag names are not records."),
                     ("Tag the customer `on-leave` and show its tags.", ["Ann Lee"], "Tag names are not records.")]


def test_a_merged_sub_task_is_never_given_the_record_listed_beside_its_own(monkeypatch):
    import fastworkflow.workflow_agent as agent
    from fastworkflow.split_merge import merge_subtasks

    # a chooser that takes a listed sibling whenever it is offered one (the mistake seen live)
    monkeypatch.setattr(agent, "_jev_lm", lambda: object())
    monkeypatch.setattr(agent, "_choose_referent", lambda q, text, candidates, *a: next(
        (c for c in ("Bob Kerr", "Ann Lee") if c in candidates), None))
    monkeypatch.setattr(agent, "_choose_description", lambda *args: None)
    q = ("Northwind Ltd is closing. Walk its whole roster: every person on it. Ann Lee and Bob Kerr leave at the end "
         "of the month, so each of them down to their first login listed. And Cid Moss has asked for the right "
         "Payroll_Admin: what it exposes and who holds it.")
    merged = merge_subtasks(q, [
        "Walk the whole roster of Northwind Ltd: every person on it.",
        "Ann Lee leaves at the end of the month, so each of them down to their first login listed.",
        "Bob Kerr leaves at the end of the month, so each of them down to their first login listed.",
        "Cid Moss has asked for the right Payroll_Admin: what it exposes and who holds it."])
    resolved = agent.resolve_referents(q, merged, None)
    assert [t for t in resolved if "Ann Lee" in t] == [t for t in merged if "Ann Lee" in t]
    assert not any("Ann Lee" in t and "Bob Kerr" in t for t in resolved)


def test_the_record_is_written_after_the_definite_description_jev_chooses(tmp_path, monkeypatch):
    import fastworkflow.workflow_agent as agent

    asked = []

    def describe(user_query, text, subject, descriptions, notes, jev_lm):
        asked.append((text, subject, descriptions))
        return "the customer"

    monkeypatch.setattr(agent, "_jev_lm", lambda: object())
    monkeypatch.setattr(agent, "_choose_referent", lambda q, text, *a: "Ann Lee" if "on-leave" in text else None)
    monkeypatch.setattr(agent, "_choose_description", describe)
    q = "Ann Lee leaves on Friday: sweep her reach. Then tag the customer `on-leave` and show its tags."
    assert agent.resolve_referents(q, ["Ann Lee leaves on Friday: sweep her reach.",
                                       "Tag the customer `on-leave` and show its tags."], str(tmp_path)) == [
        "Ann Lee leaves on Friday: sweep her reach.",
        "Tag the customer Ann Lee `on-leave` and show Ann Lee's tags."]
    # asked only for the sub-task whose referent was chosen, with its definite descriptions
    assert asked == [("Tag the customer `on-leave` and show its tags.", "Ann Lee", ["the customer"])]


def test_without_jev_the_sub_tasks_are_left_as_they_are(monkeypatch):
    import fastworkflow.workflow_agent as agent

    monkeypatch.setattr(agent, "_jev_lm", lambda: None)
    assert agent.resolve_referents("Ann Lee leaves. Tag the customer `x`.", ["Ann Lee leaves.", "Tag the customer `x`."],
                                   None) == ["Ann Lee leaves.", "Tag the customer `x`."]


def test_a_failing_merge_keeps_the_models_split(monkeypatch):
    import fastworkflow.workflow_agent as agent

    def broken(*args, **kwargs):
        raise RuntimeError("merge bug")

    monkeypatch.setattr(agent, "merge_subtasks", broken)
    assert _split({"reasoning": "r", "subtasks": "1. Audit right A.\n2. Sweep Hubbard."}) == [
        "Audit right A.", "Sweep Hubbard."]


def test_a_clean_split_is_not_retried():
    from fastworkflow.workflow_agent import split_into_subtasks

    clean = {"reasoning": "r", "subtasks": "1. Audit as well as sweep Alice.\n2. Sweep Hubbard."}
    retry_would_differ = {"reasoning": "r", "subtasks": "1. Other part.\n2. Another part."}
    assert split_into_subtasks("the request", planner_lm=DummyLM([clean, retry_would_differ])) == [
        "Audit as well as sweep Alice.", "Sweep Hubbard."]


def test_the_splitter_call_is_capped_at_split_max_tokens(monkeypatch):
    from fastworkflow import workflow_agent as agent

    lm = DummyLM([{"reasoning": "r", "subtasks": "1. Part A.\n2. Part B."}])
    copies = []
    real_copy = lm.copy

    def recording_copy(**kwargs):
        copies.append(kwargs)
        return real_copy(**kwargs)

    monkeypatch.setattr(lm, "copy", recording_copy)
    assert agent.split_into_subtasks("the request", planner_lm=lm) == ["Part A.", "Part B."]
    assert copies == [{"temperature": 0.0, "max_tokens": agent.SPLIT_MAX_TOKENS}]
    assert agent.SPLIT_MAX_TOKENS > 0


class _ScriptedReplies(DummyLM):
    """An LM that returns scripted raw replies ``(text, finish_reason, completion_tokens)``, in order,
    across its copies; ``finishes`` records each call's finish reason."""

    def __init__(self, script):
        super().__init__([])
        self.script = iter(script)
        self.finishes = []
        self._engine_spec = self._engine()

    def _engine(self):
        from dspy.clients.engines.dummy_engine import DummyEngine
        from dspy.lm15 import Message, Response, TextPart, Usage

        owner = self

        class Engine(DummyEngine):
            def _complete_messages(self, messages):
                text, finish, tokens = next(owner.script)
                owner.finishes.append(finish)
                return Response(id=None, model="scripted", message=Message.assistant([TextPart(text)]),
                                finish_reason=finish,
                                usage=Usage(input_tokens=0, output_tokens=tokens, total_tokens=tokens))

        return Engine(self)

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied._engine_spec = self._engine()  # the copy answers from this LM's script
        return copied


_SPLIT = "[[ ## reasoning ## ]]\nTwo jobs.\n\n[[ ## subtasks ## ]]\n1. Audit right A.\n2. Sweep Hubbard.\n\n[[ ## completed ## ]]"
_GOOD = (_SPLIT, "stop", 40)
# Cut off mid-reasoning after echoing the output template: the chat adapter parses it, as "{subtasks}".
_ECHOED_THEN_CUT = ("[[ ## reasoning ## ]]\nThe request names two jobs. Now we must produce the output fields:\n\n"
                    "[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## subtasks ## ]]\n{subtasks}\n\n"
                    "[[ ## completed ## ]]\n\nNow we need to write the reasoning that walks through", "length", 4700)
# Cut off mid-reasoning with no field header: does not parse, so DSPy retries with its JSON adapter.
_CUT = ("[[ ## reasoning ## ]]\nRule 7 says to name it; but rule 9 says not to; so rule 7 again", "length", 4700)
_JSON_SPLIT = ('{"reasoning": "Two jobs.", "subtasks": "1. Audit right A.\\n2. Sweep Hubbard."}', "stop", 40)


def test_a_normal_split_reply_is_one_call():
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ScriptedReplies([_GOOD, _GOOD])
    assert split_into_subtasks("the request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.finishes == ["stop"]


def test_a_cut_off_reply_that_parses_is_not_the_split_and_is_retried():
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ScriptedReplies([_ECHOED_THEN_CUT, _GOOD])
    assert split_into_subtasks("the request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.finishes == ["length", "stop"]


def test_a_cut_off_reply_that_does_not_parse_uses_dspys_json_retry():
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ScriptedReplies([_CUT, _JSON_SPLIT, _GOOD])
    assert split_into_subtasks("the request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.finishes == ["length", "stop"]


def test_a_cut_off_json_retry_still_gets_the_split_retry():
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ScriptedReplies([_CUT, ('{"reasoning": "Rule 7 says', "length", 4700), _GOOD])
    assert split_into_subtasks("the request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.finishes == ["length", "length", "stop"]


def test_a_split_cut_off_twice_leaves_the_request_whole():
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ScriptedReplies([_ECHOED_THEN_CUT, _ECHOED_THEN_CUT, _GOOD])
    assert split_into_subtasks("the request", planner_lm=lm) == ["the request"]
    assert lm.finishes == ["length", "length"]


# A sub-tasks field that is not empty but holds no numbered item: visibly broken, like a cut-off reply.
_NO_NUMBERED_ITEM = "[[ ## reasoning ## ]]\nr\n\n[[ ## subtasks ## ]]\nit\n\n[[ ## completed ## ]]"
_THREE_ITEMS = ("[[ ## reasoning ## ]]\nThree jobs.\n\n[[ ## subtasks ## ]]\n"
                "1. Renew the loan of Ada Ng.\n2. List the holds of Ben Ito.\n3. Check the fines of Cy Ro.\n\n"
                "[[ ## completed ## ]]")


@pytest.mark.skip(reason="fix-kdag disabled after benchmark pass 4")
def test_a_split_with_no_numbered_item_is_retried_and_the_retry_is_used():
    from fastworkflow.workflow_agent import split_into_subtasks

    query = "Ada Ng renews a loan, Ben Ito lists holds and Cy Ro checks fines."
    lm = _ScriptedReplies([(_NO_NUMBERED_ITEM, "stop", 10), (_THREE_ITEMS, "stop", 60)])
    assert split_into_subtasks(query, planner_lm=lm) == [
        "Renew the loan of Ada Ng.", "List the holds of Ben Ito.", "Check the fines of Cy Ro."]
    assert lm.finishes == ["stop", "stop"]


@pytest.mark.skip(reason="fix-kdag disabled after benchmark pass 4")
def test_a_split_with_no_numbered_item_twice_leaves_the_request_whole():
    from fastworkflow.workflow_agent import split_into_subtasks

    query = "Ada Ng renews a loan, Ben Ito lists holds and Cy Ro checks fines."
    lm = _ScriptedReplies([(_NO_NUMBERED_ITEM, "stop", 10), (_NO_NUMBERED_ITEM, "stop", 10)])
    assert split_into_subtasks(query, planner_lm=lm) == [query]
    assert lm.finishes == ["stop", "stop"]


def test_a_genuine_one_item_split_still_returns_the_request_verbatim():
    from fastworkflow.workflow_agent import split_into_subtasks

    query = "Renew the loan of Ada Ng, please."
    one_item = {"reasoning": "r", "subtasks": "1. Renew the loan of Ada Ng."}
    lm = _ScriptedReplies([(_split_reply_for(one_item), "stop", 20)])
    assert split_into_subtasks(query, planner_lm=lm) == [query]
    assert lm.finishes == ["stop"]


def _split_reply_for(answer):
    return ("[[ ## reasoning ## ]]\n" + answer["reasoning"] + "\n\n[[ ## subtasks ## ]]\n"
            + answer["subtasks"] + "\n\n[[ ## completed ## ]]")


class _ProviderLM(DummyLM):
    """A planner LM whose calls are checked as litellm checks them. ``only_temperature_1`` makes it a model
    that takes only ``temperature=1`` (Bedrock Claude Haiku 5.5): any other temperature is refused with
    litellm's ``UnsupportedParamsError``, unless the call has ``drop_params``, which leaves the temperature
    out. ``refused`` and ``sent`` record each call as ``(temperature, drop_params, cache)``; ``fail`` makes
    every sent call raise it, as a provider that is down. Each instance is its own model, so no call is
    answered from the DSPy response cache of another test or an earlier run."""

    def __init__(self, answers, *, only_temperature_1=False, fail=None):
        super().__init__(answers)
        self.model = f"scripted/provider-model-{uuid.uuid4().hex}"
        self.only_temperature_1 = only_temperature_1
        self.fail = fail
        self.refused: list = []
        self.sent: list = []
        self._engine_spec = self._engine()

    def _engine(self):
        import litellm
        from dspy.clients.engines.dummy_engine import DummyEngine

        owner = self

        class Engine(DummyEngine):
            def complete_legacy(self, lm, request, **context):
                temperature, dropping = request.get("temperature"), bool(request.get("drop_params"))
                call = (temperature, dropping, lm.cache)
                if owner.only_temperature_1 and temperature not in (None, 1):
                    if not dropping:
                        owner.refused.append(call)
                        raise litellm.UnsupportedParamsError(
                            message=f"{lm.model} does not support temperature={temperature}. Only temperature=1 "
                                    "is supported. To drop unsupported params, set `litellm.drop_params = True`.",
                            status_code=400)
                owner.sent.append(call)
                if owner.fail is not None:
                    raise owner.fail
                return super().complete_legacy(lm, request, **context)

        return Engine(self)

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        copied._engine_spec = self._engine()  # the copy answers from, and records on, this LM
        return copied


@pytest.fixture
def fresh_process(monkeypatch):
    """No model refused a parameter yet and no split failure logged yet, as in a new process."""
    from fastworkflow import workflow_agent as agent
    from fastworkflow.utils import dspy_utils

    monkeypatch.setattr(dspy_utils, "_PARAMS_REJECTED", set())
    monkeypatch.setattr(agent, "_SPLIT_FAILURES_WARNED", set())


@pytest.fixture
def warnings_log(caplog):
    """Capture fastWorkflow warnings: that logger does not propagate to root, so attach the handler directly."""
    import logging

    log = logging.getLogger("fastWorkflow")
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="fastWorkflow"):
            yield caplog
    finally:
        log.removeHandler(caplog.handler)


_CLEAN = {"reasoning": "r", "subtasks": "1. Audit right A.\n2. Sweep Hubbard."}
_DUPLICATED = {"reasoning": "r", "subtasks": "1. Same thing.\n2. Same thing."}
_FIXED = {"reasoning": "r", "subtasks": "1. Part A.\n2. Part B."}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_planner_that_takes_only_temperature_1_still_splits(fresh_process, warnings_log):
    """fix-z62t: temperature 0.0 is refused, so the call goes again without it, and the split is used."""
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ProviderLM([_CLEAN], only_temperature_1=True)
    assert split_into_subtasks("the request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.refused == [(0.0, False, True)]
    assert lm.sent == [(0.0, True, True)]  # with drop_params, litellm leaves the 0.0 out
    refusals = [r for r in warnings_log.records if "refused a parameter" in r.getMessage()]
    assert len(refusals) == 1 and "Only temperature=1 is supported" in refusals[0].getMessage()
    assert not [r for r in warnings_log.records if "could not split" in r.getMessage()]

    # Later splits on that model leave the temperature out from the start: nothing more is refused or logged.
    lm.answers = iter([_CLEAN])
    assert split_into_subtasks("another request", planner_lm=lm) == ["Audit right A.", "Sweep Hubbard."]
    assert lm.refused == [(0.0, False, True)]
    assert lm.sent == [(0.0, True, True), (0.0, True, True)]
    assert len([r for r in warnings_log.records if "refused a parameter" in r.getMessage()]) == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_retry_on_a_planner_without_its_temperature_skips_the_cache(fresh_process):
    """The retry's 0.7 cannot change that model's answer, so the retry is sent past the response cache."""
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ProviderLM([_DUPLICATED, _FIXED], only_temperature_1=True)
    assert split_into_subtasks("the request", planner_lm=lm) == ["Part A.", "Part B."]
    assert lm.sent == [(0.0, True, True), (0.7, True, False)]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_planner_that_takes_any_temperature_is_called_as_before(fresh_process, warnings_log):
    """No behaviour change for a model that takes the temperatures: 0.0 then 0.7, no drop_params, cached."""
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ProviderLM([_DUPLICATED, _FIXED])
    assert split_into_subtasks("the request", planner_lm=lm) == ["Part A.", "Part B."]
    assert lm.refused == []
    assert lm.sent == [(0.0, False, True), (0.7, False, True)]
    assert not [r for r in warnings_log.records if "refused a parameter" in r.getMessage()]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_split_that_always_fails_is_logged_once_with_its_cause(fresh_process, warnings_log):
    """A misconfigured planner fails every turn the same way: one WARNING naming the cause, not one per turn."""
    from fastworkflow.workflow_agent import split_into_subtasks

    lm = _ProviderLM([], fail=RuntimeError("provider down"))
    for _ in range(3):
        assert split_into_subtasks("the request", planner_lm=lm) == ["the request"]
    assert len(lm.sent) == 3  # each turn tried; only the log is once
    failures = [r for r in warnings_log.records if "could not split" in r.getMessage()]
    assert len(failures) == 1
    assert lm.model in failures[0].getMessage()
    assert "provider down" in failures[0].getMessage()

    # Another cause is new, so it gets its own WARNING.
    lm.fail = ValueError("bad reply")
    assert split_into_subtasks("the request", planner_lm=lm) == ["the request"]
    assert len([r for r in warnings_log.records if "could not split" in r.getMessage()]) == 2


def test_only_litellms_refusal_of_a_parameter_counts_as_one():
    from fastworkflow.utils.dspy_utils import params_rejected
    import litellm

    refusal = litellm.UnsupportedParamsError(message="does not support temperature=0.0", status_code=400)
    try:
        try:
            raise refusal
        except litellm.UnsupportedParamsError as inner:
            raise RuntimeError("wrapped by DSPy") from inner
    except RuntimeError as wrapped:
        assert params_rejected(wrapped)
    assert not params_rejected(RuntimeError("does not support temperature=0.0"))


_TWICE_REQUEST = ("Audit the right Billing_Admin. Ann Lee, Bob Kerr and Cid Moss need the same treatment, so I want "
                  "each of them down to their logins. Dee Ray leaves on Friday, so sweep Dee Ray's reach.")


def _reply(*subtasks):
    numbered = "\n".join(f"{n}. {text}" for n, text in enumerate(subtasks, 1))
    return (f"[[ ## reasoning ## ]]\nJobs.\n\n[[ ## subtasks ## ]]\n{numbered}\n\n[[ ## completed ## ]]", "stop", 40)


# Both replies keep "the same treatment"; the first also invents a job for Ann Lee and adds a summary sub-task
# naming the people the others already hold (stage-17 T5 5.09 R1).
_SUMMARY_AND_INVENTED = _reply(
    "Audit the right Billing_Admin.",
    "Ann Lee leaves on Friday, so sweep Ann Lee's reach.",
    "Bob Kerr needs the same treatment, down to Bob Kerr's logins.",
    "Cid Moss needs the same treatment, down to Cid Moss's logins.",
    "Review the list: Ann Lee, Bob Kerr and Cid Moss, each of them down to their logins.")
_ONE_PER_JOB = _reply(
    "Audit the right Billing_Admin.",
    "Ann Lee needs the same treatment, down to Ann Lee's logins.",
    "Bob Kerr needs the same treatment, down to Bob Kerr's logins.",
    "Cid Moss needs the same treatment, down to Cid Moss's logins.",
    "Dee Ray leaves on Friday, so sweep Dee Ray's reach.")


def test_of_two_broken_splits_the_one_naming_no_record_twice_is_used(monkeypatch):
    import fastworkflow.workflow_agent as agent
    from fastworkflow.split_merge import merge_subtasks

    monkeypatch.setattr(agent, "_jev_lm", lambda: None)
    lm = _ScriptedReplies([_SUMMARY_AND_INVENTED, _ONE_PER_JOB])
    retry = [line.split(". ", 1)[1] for line in _ONE_PER_JOB[0].splitlines() if line[:1].isdigit()]
    assert agent.split_into_subtasks(_TWICE_REQUEST, planner_lm=lm) == merge_subtasks(_TWICE_REQUEST, retry)
    assert lm.finishes == ["stop", "stop"]
    # the order of the attempts does not matter: the split naming no record twice wins either way
    lm = _ScriptedReplies([_ONE_PER_JOB, _SUMMARY_AND_INVENTED])
    assert agent.split_into_subtasks(_TWICE_REQUEST, planner_lm=lm) == merge_subtasks(_TWICE_REQUEST, retry)


def test_two_equally_broken_splits_keep_the_first(monkeypatch):
    import fastworkflow.workflow_agent as agent
    from fastworkflow.split_merge import merge_subtasks

    monkeypatch.setattr(agent, "_jev_lm", lambda: None)
    first = [line.split(". ", 1)[1] for line in _ONE_PER_JOB[0].splitlines() if line[:1].isdigit()]
    reworded = _reply(*[t.replace(", down to", " down to") for t in first])
    lm = _ScriptedReplies([_ONE_PER_JOB, reworded])
    assert agent.split_into_subtasks(_TWICE_REQUEST, planner_lm=lm) == merge_subtasks(_TWICE_REQUEST, first)


def test_breakage_counts_records_named_twice_then_leftover_references():
    from fastworkflow.workflow_agent import _breakage

    summary = [line.split(". ", 1)[1] for line in _SUMMARY_AND_INVENTED[0].splitlines() if line[:1].isdigit()]
    per_job = [line.split(". ", 1)[1] for line in _ONE_PER_JOB[0].splitlines() if line[:1].isdigit()]
    # Ann Lee, Bob Kerr and Cid Moss each appear in a second sub-task; three sub-tasks keep a reference
    assert _breakage(_TWICE_REQUEST, summary, None) == (3, 3)
    assert _breakage(_TWICE_REQUEST, per_job, None) == (0, 3)


def test_a_reply_is_cut_off_by_its_finish_reason_or_by_reaching_max_tokens():
    from types import SimpleNamespace

    from fastworkflow.utils.dspy_logger import reply_cut_off

    lm = SimpleNamespace(kwargs={"max_tokens": 4700})
    call = SimpleNamespace(kwargs={})
    assert reply_cut_off(lm, call, SimpleNamespace(raw={"choices": [{"finish_reason": "length"}]}))
    assert reply_cut_off(lm, call, SimpleNamespace(responses=(SimpleNamespace(finish_reason="length"),)))
    assert reply_cut_off(lm, call, SimpleNamespace(usage={"completion_tokens": 4700}))
    assert not reply_cut_off(lm, call, SimpleNamespace(usage={"completion_tokens": 4699},
                                                       raw={"choices": [{"finish_reason": "stop"}]}))
    assert not reply_cut_off(SimpleNamespace(kwargs={}), call, SimpleNamespace(usage={"completion_tokens": 9}))


def planning_command_file(text: str) -> str:
    """The skill file (``_commands/<name>.py``) of a skill without params: ``text``'s first line is its description,
    every other non-empty line one step."""
    description, *steps = [line.strip() for line in text.splitlines() if line.strip()]
    return (f"class Signature:\n    __doc__ = {description!r}\n\n\nclass PlanGenerator:\n"
            f"    def steps(self):\n        return {steps!r}\n")


def test_planning_commands_without_params_give_their_steps_and_all_are_shown(tmp_path):
    from fastworkflow.workflow_agent import _job_types_for, load_planning_commands

    (tmp_path / "_commands").mkdir()
    for name, text in {"leaver_sweep": "One person is leaving: their reach. One job per person.\nfind them\nopen them",
                       "unit_walk": "Walk a whole unit roster. The unit is one job.\nopen the unit"}.items():
        (tmp_path / "_commands" / f"{name}.py").write_text(planning_command_file(text))

    assert load_planning_commands(str(tmp_path))[0] == (
        "leaver_sweep", "One person is leaving: their reach. One job per person.", ("find them", "open them"))
    assert _job_types_for(str(tmp_path)).splitlines() == [
        "- leaver_sweep: One person is leaving: their reach. One job per person.",
        "- unit_walk: Walk a whole unit roster. The unit is one job.",
    ]
    assert _job_types_for(str(tmp_path / "missing")) == "(none)"


def _write_planning_commands(tmp_path, planning_commands):
    (tmp_path / "_commands").mkdir(parents=True)
    for name, text in planning_commands.items():
        (tmp_path / "_commands" / f"{name}.py").write_text(planning_command_file(text))
    return str(tmp_path)


def test_the_planner_template_gets_the_planning_commands_steps(tmp_path, monkeypatch):
    from fastworkflow import workflow_agent

    folder = _write_planning_commands(tmp_path, {"leaver": "Someone is leaving.\nfind them\nclose it\n"})
    monkeypatch.setattr(workflow_agent, "_planning_command_matcher", lambda f: {"": _fake_matcher("leaver", 0.9)})
    assert workflow_agent.match_planning_command("Pat is leaving", folder) == ("leaver", ("find them", "close it"))
    assert "- find them\n- close it\n" in workflow_agent.plan_template_for("Pat is leaving", folder)


def test_the_split_files_signature_docstring_replaces_the_default_instructions(tmp_path):
    from fastworkflow.workflow_agent import SplitIntoSubtasks, _split_signature, load_planning_commands

    folder = _split_file(tmp_path, 'class Signature:\n    """Split it my way.\n\n    - one rule\n    """\n')
    other = tmp_path / "other"
    (other / "_commands").mkdir(parents=True)

    assert _split_signature(folder).instructions == "Split it my way.\n\n- one rule"
    assert _split_signature(folder).fields["job_types"].json_schema_extra["desc"] == \
        SplitIntoSubtasks.fields["job_types"].json_schema_extra["desc"]
    assert _split_signature(str(other)).instructions == SplitIntoSubtasks.instructions
    assert load_planning_commands(folder) == ()   # _split.py is not a skill


def test_the_split_files_input_and_output_describe_the_splitter_fields(tmp_path):
    from fastworkflow.planning_commands import PlanningCommandFormatError
    from fastworkflow.workflow_agent import SplitIntoSubtasks, _split_signature

    folder = _split_file(tmp_path / "ok", (
        "from pydantic import BaseModel, Field\n\n"
        "class Signature:\n"
        "    class Input(BaseModel):\n"
        "        job_types: str = Field(description='the kinds of job')\n"
        "        user_query: str\n"
        "    class Output(BaseModel):\n"
        "        subtasks: str = Field(description='one sub-task per numbered line')\n"))
    signature = _split_signature(folder)
    assert signature.instructions == SplitIntoSubtasks.instructions   # no docstring: the default
    assert signature.fields["job_types"].json_schema_extra["desc"] == "the kinds of job"
    assert signature.fields["subtasks"].json_schema_extra["desc"] == "one sub-task per numbered line"
    assert signature.fields["user_query"].json_schema_extra["desc"] == \
        SplitIntoSubtasks.fields["user_query"].json_schema_extra["desc"]

    bad = _split_file(tmp_path / "bad", (
        "from pydantic import BaseModel\n\n"
        "class Signature:\n"
        "    class Input(BaseModel):\n"
        "        request: str\n"))
    with pytest.raises(PlanningCommandFormatError, match="Signature.Input.request is not a splitter input"):
        _split_signature(bad)


def _planning_command_folder(tmp_path):
    (tmp_path / "_commands").mkdir(parents=True)
    (tmp_path / "_commands" / "leaver.py").write_text(planning_command_file("Someone is leaving.\nfind them\nopen them\n"))
    return str(tmp_path)


def _fake_matcher(value, probability):
    from types import SimpleNamespace

    def predict(**_inputs):
        choice = SimpleNamespace(value=value, probabilities={value: probability}, confidence=probability)
        return SimpleNamespace(job=choice)

    return predict, DummyLM([])


def test_a_plan_template_is_the_matched_planning_command_steps_in_a_fixed_format(monkeypatch):
    from fastworkflow import workflow_agent

    monkeypatch.setattr(workflow_agent, "match_planning_command", lambda text, folder, **_: ("leaver", ("find them", "open them")))
    assert workflow_agent.plan_template_for("Two people are leaving", "unused") == (
        "A job like this is usually done as:\n- find them\n- open them\n"
        "Adapt this to the request: drop steps it does not need and add steps it asks for.")
    monkeypatch.setattr(workflow_agent, "match_planning_command", lambda text, folder, **_: None)
    assert workflow_agent.plan_template_for("What is the weather in Paris?", "unused") == ""


def test_match_planning_command_only_returns_a_confident_planning_command_pick(tmp_path, monkeypatch):
    from fastworkflow import workflow_agent

    folder = _planning_command_folder(tmp_path)
    match = lambda: workflow_agent.match_planning_command("Two people are leaving", folder)  # noqa: E731

    monkeypatch.setattr(workflow_agent, "_planning_command_matcher", lambda f: {"": _fake_matcher("leaver", 0.9)})
    assert match() == ("leaver", ("find them", "open them"))
    monkeypatch.setattr(workflow_agent, "_planning_command_matcher", lambda f: {"": _fake_matcher("none", 0.9)})
    assert match() is None
    monkeypatch.setattr(workflow_agent, "_planning_command_matcher", lambda f: {"": _fake_matcher("leaver", 0.49)})
    assert match() is None

    def broken(folder):
        raise RuntimeError("provider down")

    monkeypatch.setattr(workflow_agent, "_planning_command_matcher", broken)
    assert match() is None


def test_no_jev_key_means_no_matcher_and_no_template(tmp_path, workflow_env, monkeypatch):
    from fastworkflow import workflow_agent

    monkeypatch.delenv("JEV_API_KEY", raising=False)
    workflow_env()
    folder = _planning_command_folder(tmp_path)
    assert workflow_agent._planning_command_matcher(folder) is None
    assert workflow_agent.plan_template_for("Two people are leaving", folder) == ""


def test_the_plan_template_reaches_the_planner_but_not_the_returned_query(workflow_env):
    workflow_env()
    ctx, wf = _session(RecordingTraceSink())

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm, plan_template="TEMPLATE TEXT")

    assert "TEMPLATE TEXT" in _prompt(lm, 0)
    assert "TEMPLATE TEXT" not in result


# ------------------------------------------------------------------ rule 3's joins are Jev's call (fix-bydz)
TUTORIAL = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())
DUNE = "Who wrote Dune and who has it out?"
DUNE_SPLIT = {"reasoning": "r", "subtasks": "1. Who wrote Dune?\n2. Who has Dune out?"}


def _scripted_jev(monkeypatch, verdict):
    """Jev's SameRequest answer scripted (``verdict``: (value, probability), or an exception to raise); other Jev
    questions (the referent) answer nothing. Returns the inputs each SameRequest call got."""
    import fastworkflow.workflow_agent as agent
    from types import SimpleNamespace

    calls: list[dict] = []
    real_predict = agent.dspy.Predict

    def predict_for(signature):
        if signature is not agent.SameRequest:
            return real_predict(signature)

        def predict(**inputs):
            calls.append(inputs)
            if isinstance(verdict, Exception):
                raise verdict
            value, probability = verdict
            other = "separate" if value == "one" else "one"
            return SimpleNamespace(relation=SimpleNamespace(
                value=value, confidence=probability, probabilities={value: probability, other: 1 - probability}))
        return predict

    monkeypatch.setattr(agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(agent.dspy, "Predict", predict_for)
    monkeypatch.setattr(agent, "_choose_referent", lambda *args: None)
    return calls


def _split_dune():
    from fastworkflow.workflow_agent import split_into_subtasks
    return split_into_subtasks(DUNE, planner_lm=DummyLM([DUNE_SPLIT]), workflow_folderpath=TUTORIAL)


def test_two_questions_jev_calls_separate_stay_two_sub_tasks(monkeypatch):
    calls = _scripted_jev(monkeypatch, ("separate", 0.99))
    assert _split_dune() == ["Who wrote Dune?", "Who has Dune out?"]
    assert calls == [{"request": DUNE, "earlier": "Who wrote Dune?", "later": "Who has Dune out?"}]


def test_steps_jev_calls_one_request_are_joined_into_the_request(monkeypatch):
    from fastworkflow.workflow_agent import SAME_REQUEST_THRESHOLD, split_into_subtasks

    _scripted_jev(monkeypatch, ("one", SAME_REQUEST_THRESHOLD))
    assert _split_dune() == [DUNE]
    query = "Find the member Raj Patel and show his loans."
    steps = {"reasoning": "r", "subtasks": "1. Find the member Raj Patel.\n2. Show the loans of Raj Patel."}
    _scripted_jev(monkeypatch, ("one", 0.95))
    assert split_into_subtasks(query, planner_lm=DummyLM([steps]), workflow_folderpath=TUTORIAL) == [query]


def test_one_request_below_the_threshold_is_not_joined(monkeypatch):
    from fastworkflow.workflow_agent import SAME_REQUEST_THRESHOLD

    _scripted_jev(monkeypatch, ("one", SAME_REQUEST_THRESHOLD - 0.05))
    assert _split_dune() == ["Who wrote Dune?", "Who has Dune out?"]


def test_without_a_jev_key_the_lexical_rule_joins_as_before(monkeypatch):
    calls = _scripted_jev(monkeypatch, ("separate", 0.99))
    import fastworkflow.workflow_agent as agent

    monkeypatch.setattr(agent, "_jev_lm", lambda: None)
    assert _split_dune() == [DUNE]
    assert calls == []


def test_a_failing_jev_call_leaves_the_lexical_rule(monkeypatch):
    calls = _scripted_jev(monkeypatch, RuntimeError("Jev is down"))
    assert _split_dune() == [DUNE]
    assert len(calls) == 1


def test_the_conversation_is_a_separate_splitter_input_only_when_there_is_one(monkeypatch):
    import fastworkflow.workflow_agent as agent

    _scripted_jev(monkeypatch, ("separate", 0.99))
    query = "Who has it out, and how many loans are overdue?"
    context = "conversation summary: The user asked who wrote Dune. Dune is by Frank Herbert."
    answer = {"reasoning": "r", "subtasks": "1. Who has Dune out on loan?\n2. How many loans are overdue?"}
    lm = DummyLM([answer])
    assert agent.split_into_subtasks(query, planner_lm=lm, workflow_folderpath=TUTORIAL, conversation=context) == [
        "Who has Dune out on loan?", "How many loans are overdue?"]
    prompt = _last_prompt()
    assert "[[ ## conversation ## ]]" in prompt and context in prompt
    assert f"[[ ## user_query ## ]]\n{query}" in prompt
    agent.split_into_subtasks(DUNE, planner_lm=DummyLM([DUNE_SPLIT]), workflow_folderpath=TUTORIAL)
    assert "## conversation ##" not in _last_prompt()


def _last_prompt() -> str:
    """The messages of the last model call (the splitter calls a copy of the LM, so its own history is empty)."""
    from dspy.clients.base_lm import GLOBAL_HISTORY

    entry = next(h for h in reversed(GLOBAL_HISTORY) if "subtasks" in str(h.get("messages")))
    return "\n".join(str(m.get("content", "")) for m in entry["messages"])


# The splitter's reply to benchmark pass 9, task 5.10, round 2 (Nemotron Nano), as recorded: the five
# leavers listed once, the unit review and the control, then the five leavers again word for word. 12 lines, 7 jobs.
RECORDED_QUERY_5_10 = "HR has sent me five names for this month's offboarding: Aaron Garrison, Alexander Herrera, David Farrell, Amy Walsh and April Watts. Before I sign the batch off I want the quarterly review of Gardner Robinson and Lawrence LLC 2, the unit picking their work up — the first three people on its roster who have a login, each taken down to logins, rights and collections, and the first of those people's logins opened on its own permissions and custodians. Then each of the five: what they can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied. Work the finding for the control 'Identities who are contractors set as inactive or whose departure date has passed and owning active accounts' through to the first identity it names as well, down to logins, rights and collections."
RECORDED_REPLY_5_10 = "1. Aaron Garrison leaves: what Aaron Garrison can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n2. Alexander Herrera leaves: what Alexander Herrera can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n3. David Farrell leaves: what David Farrell can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n4. Amy Walsh leaves: what Amy Walsh can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n5. April Watts leaves: what April Watts can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n6. Quarterly review of Gardner Robinson and Lawrence LLC 2, the unit picking their work up — the first three people on its roster who have a login, each taken down to logins, rights and collections, and the first of those people's logins opened on its own permissions and custodians.\n7. Work the finding for the control 'Identities who are contractors set as inactive or whose departure date has passed and owning active accounts' through to the first identity it names as well, down to logins, rights and collections.\n8. Aaron Garrison leaves: what Aaron Garrison can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n9. Alexander Herrera leaves: what Alexander Herrera can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n10. David Farrell leaves: what David Farrell can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n11. Amy Walsh leaves: what Amy Walsh can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied.\n12. April Watts leaves: what April Watts can reach, their first login listed walked down to the first entitlement it carries, and the corrections available on them, with none applied."

_LEAVERS = ["Aaron Garrison", "Alexander Herrera", "David Farrell", "Amy Walsh", "April Watts", "Bill Brown",
            "Billy Avila", "Brad Miller", "Barry Jones", "Bethany Oconnor", "Heidi Turner", "Gary Edwards"]


def _numbered(items):
    return "\n".join(f"{n}. {text}" for n, text in enumerate(items, 1))


def _leaves(name):
    return f"{name} leaves: what {name} can reach, their first login listed walked down to its first right."


@pytest.fixture
def no_jev(monkeypatch):
    from fastworkflow import workflow_agent
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert workflow_agent._jev_lm() is None, "this test needs a process without a Jev key"


def _split_with(query, reply):
    from fastworkflow.workflow_agent import split_into_subtasks
    return split_into_subtasks(query, planner_lm=DummyLM([{"reasoning": "r", "subtasks": reply}]))


def test_a_split_over_the_cap_only_because_it_repeats_jobs_is_still_split(no_jev):
    from fastworkflow.workflow_agent import MAX_SUBTASKS, _split_items

    assert len(_split_items(RECORDED_REPLY_5_10)) > MAX_SUBTASKS
    parts = _split_with(RECORDED_QUERY_5_10, RECORDED_REPLY_5_10)
    assert len(parts) == 7
    assert [part.split(" leaves")[0] for part in parts[:5]] == [
        "Aaron Garrison", "Alexander Herrera", "David Farrell", "Amy Walsh", "April Watts"]


def test_a_split_over_the_cap_is_reduced_never_run_whole(no_jev):
    from fastworkflow.workflow_agent import MAX_SUBTASKS

    query = "Offboard " + ", ".join(_LEAVERS) + ". Also audit the right Jira_IT Support Specialist."
    items = [_leaves(name) for name in _LEAVERS] + ["Audit the right Jira_IT Support Specialist."]
    parts = _split_with(query, _numbered(items))
    assert 1 < len(parts) <= MAX_SUBTASKS
    for name in _LEAVERS + ["Jira_IT Support Specialist"]:
        assert any(name in part for part in parts), f"{name} is named by no job"


def test_a_long_split_of_copies_paraphrases_and_jobs_keeps_every_subject(no_jev):
    from fastworkflow.workflow_agent import MAX_SUBTASKS

    names = _LEAVERS[:8]
    rights = ["Jira_IT Support Specialist", "GitLab_Technical Leads", "Active Directory_Cloud Administrator"]
    query = "Offboard " + ", ".join(names) + ". Audit " + ", ".join(rights) + "."
    items = []
    for name in names:
        items += [_leaves(name), f"Please check what access {name} holds and take their first login down to its first right."]
    items += [f"Audit the right {right} across the systems it appears in." for right in rights]
    items += [f"For {right}, tell me who holds it and which system publishes it." for right in rights]
    items += [items[0], items[2]]                       # two exact copies on top
    assert len(items) == 24
    parts = _split_with(query, _numbered(items))
    assert 1 < len(parts) <= MAX_SUBTASKS
    for name in names + rights:
        assert any(name in part for part in parts), f"{name} is named by no job"


def test_the_reduction_asks_jev_no_more_than_its_budget_and_decides_without_it():
    from fastworkflow.split_merge import reduce_to_cap

    asked = []

    def jev_same(earlier, later):
        asked.append(("same", earlier))
        return False

    def jev_label(text):
        asked.append(("label", text))
        return "leave"

    names = [f"{first} {last}" for first in ("Aaron", "Brad", "Cara", "Dev", "Elin") for last in ("Smith", "Jones", "Brown", "Klein")]
    query = "Offboard " + ", ".join(names) + "."
    items = [_leaves(name) for name in names]
    with_jev = reduce_to_cap(query, items, 10, same_job=jev_same, near_copy=jev_same, label=jev_label, max_jev_calls=7)
    assert len(with_jev.items) == 10 and len(asked) <= 7 and with_jev.jev_calls == len(asked)
    without = reduce_to_cap(query, items, 10)
    assert len(without.items) == 10 and without.jev_calls == 0
    for name in names:
        assert any(name in item for item in with_jev.items) and any(name in item for item in without.items)
    # at or under the cap nothing is touched and nothing is asked
    assert reduce_to_cap(query, items[:10], 10, label=jev_label).items == items[:10]


def test_a_dspy_field_marker_in_the_split_reply_is_not_part_of_a_sub_task():
    """A splitter reply once carried "[[ ## completed ## ]]" into the last sub-task's text, which
    the plan review then showed the user. The marker is cut; the item's own words are kept."""
    from fastworkflow.workflow_agent import _split_items

    reply = ("1. Audit the right Library_Archivist across the systems it appears in.\n"
             "2. Dana Reyes has asked for the right Library_Curator 1: what it exposes, who holds it, "
             "and what Dana Reyes can reach today. [[ ## completed ## ]]")
    assert _split_items(reply) == [
        "Audit the right Library_Archivist across the systems it appears in.",
        "Dana Reyes has asked for the right Library_Curator 1: what it exposes, who holds it, "
        "and what Dana Reyes can reach today."]
    assert _split_items("1. Show A [[ ## completed ## ]] 2. Show B [b2]") == ["Show A", "Show B [b2]"]
