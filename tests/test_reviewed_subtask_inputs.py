"""The reviewed sub-task inputs are the inputs that run (fix-48np.1).

With drafting on (``FW_DRAFT_ARGUMENTS``), the sub-task review shows each sub-task's inputs drafted by the shared
extractor (and, for a planning command whose required value is still missing, the typed fill fallback). A sub-task bound
to a planning command then runs with exactly those inputs (``review_proposals.reviewed_arguments``): accepted as drafted,
edited by the user, or ``from sub-task N``. They are presets: never extracted again, still checked
(``check_arguments``, ``validate_extracted_parameters``, the ``Input``'s rules). A refused preset is the sub-task's
answer, shown, never handed to the agent to extract afresh. Only an argument the review left unbound is drafted at run
time. A turn with no one to answer runs the draft unreviewed, with the same values. With drafting off nothing changes.

Integration tests on the annotated library copy with its trained heads (``tests/test_round1_gate.py``'s ``gated``
context): the real agent with scripted steps, the real review, the real phase runner and the real DSPy extractor and
checks. The model boundary is a scripted ``DummyLM`` that records each request. No test calls a model.
"""
from __future__ import annotations

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import review_proposals
from fastworkflow.review_proposals import Means, SubTask, SubTaskInput
from fastworkflow.turn import TurnStatus
from fastworkflow.utils import signatures
from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401 - the fixtures
from tests.test_round1_gate import EMMA, JOKE, _marker, _open_review, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

BOOK_BY_TITLE = "Library/book_by_title"
MEMBER_REPORT = "Library/member_report"
AUTHOR_BOOKS = "Library/author_books"
EMMA_ANSWER = {"reasoning": "the title", "title": "Emma"}
JOKE_ANSWER = {"reasoning": "the member", "member": "Ann", "loans": False, "notes": False}


class _ScriptedLM(DummyLM):
    """Answers its calls in order and records the text of each request. The extractor's answers are JSON; the planner's
    (the typed fill fallback) are in the planner adapter's own format (``json=False``)."""

    def __init__(self, answers, json: bool = True):
        super().__init__(list(answers), **({"adapter": dspy.JSONAdapter()} if json else {}))
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def drafting(monkeypatch):
    """Drafting on (the tests about drafting off turn it off again), Jev's field check off, and the extractor scripted."""
    monkeypatch.setenv(review_proposals.DRAFT_ARGUMENTS_ENV_VAR, "on")
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")


@pytest.fixture
def scripted(monkeypatch):
    """The model boundary of the extractor and the typed fill: *answers* in order; returns the LM."""
    def script(*answers: dict) -> _ScriptedLM:
        lm = _ScriptedLM(answers)
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        return lm
    return script


def _planner(gated, *answers: dict) -> _ScriptedLM:
    """The turn's planner LM (the typed fill fallback runs on it): *answers* in order."""
    lm = _ScriptedLM(answers, json=False)
    gated.ctx._current_planner_lm = lm
    return lm


#: The agent's steps for a turn: the review frame's finish, then a finish for each phase or agent run that asks for one.
#: The phases' own steps run in code (``CodeSteps``); a run is never short of a scripted step.
STEPS = [("finish", {})] * 20


def _calls(out, name: str) -> list:
    """The parameters of every run of command *name* in the turn's outputs."""
    return [output.command_parameters for output in out.command_outputs if output.command_name == name]


def _planning_line(record, number: int) -> str:
    return next(text for text in record.rendering.splitlines() if text.startswith(f"{number}. "))


def _bindings(gated) -> list:
    """The turn span's ``subtask_bindings`` (the trace record of shown == run), from the last turn span."""
    from fastworkflow import tracing

    spans = [span for span in gated.sink.spans if span.name == tracing.SPAN_TURN]
    return list(spans[-1].attributes.get("subtask_bindings") or []) if spans else []


def _review(gated, monkeypatch, subtasks):
    _split_into(monkeypatch, subtasks)
    gated.ctx.process_turn(" and ".join(subtasks))
    return _open_review(gated)


# ----------------------------------------------------------------------
# shown == run
# ----------------------------------------------------------------------


def test_an_accepted_draft_runs_the_values_it_shows_with_no_second_extraction(gated, monkeypatch, scripted):
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    assert "inputs: title=Emma; means: planning:Library/book_by_title" in _planning_line(record, 1)
    assert "inputs: member=Ann; means: planning:Library/member_report" in _planning_line(record, 2)
    drafted = len(lm.prompts)
    assert drafted == 2                                         # one extraction per sub-task, in the draft

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.title for p in _calls(out, BOOK_BY_TITLE)] == ["Emma"]
    assert [p.member for p in _calls(out, MEMBER_REPORT)] == ["Ann"]
    assert len(lm.prompts) == drafted                           # the run extracted nothing again
    assert _marker(gated) == "reviewed"
    assert [(b["shown"], b["match"]) for b in _bindings(gated)] == [({"title": "Emma"}, True), ({"member": "Ann"}, True)]


def test_a_value_the_user_edits_is_the_value_that_runs(gated, monkeypatch, scripted):
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 1: inputs=title=Persuasion")
    assert "inputs: title=Persuasion;" in _planning_line(record, 1)
    drafted = len(lm.prompts)

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.title for p in _calls(out, BOOK_BY_TITLE)] == ["Persuasion"]
    assert [p.member for p in _calls(out, MEMBER_REPORT)] == ["Ann"]
    assert len(lm.prompts) == drafted
    assert [(b["shown"], b["match"]) for b in _bindings(gated)] == [
        ({"title": "Persuasion"}, True), ({"member": "Ann"}, True)]


def test_a_refused_preset_falls_back_to_the_agent_and_is_recorded_as_a_deviation(gated, monkeypatch, scripted):
    """F2: member_report's validate_extracted_parameters refuses a member that names everyone. The edited preset goes
    through it; the sub-task is handed to the agent with the refusal in its text (as with drafting off), nothing of
    member_report runs with the refused value, and the turn span records the deviation."""
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    queries: list = []

    def planner(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, **kwargs):
        queries.append(user_query)
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", planner)
    _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 2: inputs=member=everyone")
    drafted = len(lm.prompts)

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.title for p in _calls(out, BOOK_BY_TITLE)] == ["Emma"]
    assert _calls(out, MEMBER_REPORT) == []
    assert any("The planning command Library/member_report cannot run as planned" in q and "names no one member" in q
               for q in queries)
    first, second = _bindings(gated)
    assert first["match"] is True and first["shown"] == {"title": "Emma"} and first["ran"]["title"] == "Emma"
    assert second["match"] is False and second["ran"] is None
    assert second["deviation"].startswith("refused, the agent ran it instead") and "names no one member" in \
        second["deviation"]
    assert len(lm.prompts) == drafted


def test_a_from_sub_task_input_runs_with_that_sub_tasks_output(gated, monkeypatch, scripted):
    lm = scripted(JOKE_ANSWER, EMMA_ANSWER)
    record = _review(gated, monkeypatch, [JOKE, EMMA])
    gated.ctx.process_turn("change 1: means=from_scratch")
    gated.ctx.process_turn("change 2: inputs=title=from sub-task 1")
    assert "inputs: title=from sub-task 1;" in _planning_line(record, 2)
    drafted = len(lm.prompts)

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    # sub-task 1's answer (the scripted agent's final answer) is sub-task 2's title, as the review said
    assert [p.title for p in _calls(out, BOOK_BY_TITLE)] == ["the answer"]
    assert len(lm.prompts) == drafted
    assert [(b["subtask"], b["shown"], b["match"]) for b in _bindings(gated)] == [(2, {"title": "the answer"}, True)]


@pytest.mark.parametrize("how", ["unattended", "host"])
def test_a_turn_with_no_one_to_answer_runs_the_draft_unreviewed_with_its_values(gated, monkeypatch, scripted, how):
    from fastworkflow.approval_binding import Surface

    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    _split_into(monkeypatch, [EMMA, JOKE])
    kwargs = ({"unattended": True} if how == "unattended"
              else {"approval_surface": Surface(kind="host", verified=True, uid="user-1")})

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", **kwargs)

    assert out.status is TurnStatus.COMPLETED
    assert [r for r in gated.store.records if r.command == "review_subtasks"] == []
    plan = gated.ctx._turn_subtask_plan
    assert plan.items[0].inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]
    assert [p.title for p in _calls(out, BOOK_BY_TITLE)] == ["Emma"]
    assert [p.member for p in _calls(out, MEMBER_REPORT)] == ["Ann"]
    assert len(lm.prompts) == 2
    assert _marker(gated) == "not_reviewed"
    assert [b["match"] for b in _bindings(gated)] == [True, True]


def test_a_means_changed_in_the_review_is_drafted_again_for_its_new_command(gated, monkeypatch, scripted):
    """F3: moving sub-task 1 to count_books drafts its inputs for count_books in the review (no stale title), and the
    run uses them with no extraction at run time."""
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER, {"reasoning": "the title, for count_books", "query": "Emma"})
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 1: means=planning:Library/count_books")
    assert "inputs: query=Emma; means: planning:Library/count_books" in _planning_line(record, 1)
    assert "title=" not in _planning_line(record, 1)
    assert len(lm.prompts) == 3 and EMMA in lm.prompts[2]
    assert "inputs: member=Ann;" in _planning_line(record, 2)  # the sub-task the edit left alone is not drafted again

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.query for p in _calls(out, "Library/count_books")] == ["Emma"]
    assert len(lm.prompts) == 3
    assert [b["match"] for b in _bindings(gated)] == [True, True]


def test_an_inputs_edit_is_not_drafted_again(gated, monkeypatch, scripted):
    """Only a means change re-drafts: an edit of the inputs alone is what runs."""
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 1: inputs=title=Dune")
    assert "inputs: title=Dune;" in _planning_line(record, 1) and len(lm.prompts) == 2


def test_a_sub_task_the_draft_never_drafted_has_only_its_unbound_values_drafted_at_run_time(gated, scripted):
    """A sub-task the draft did not draft for its means (drafted_means differs: one the user added) that leaves a
    required value unbound: that value is drafted at run time by the draft's own function, from the goal, and a bound
    value is kept as it is."""
    scripted({"reasoning": "the member", "member": "Ann", "loans": True, "notes": False})
    item = SubTask(goal=JOKE, means=Means(kind="planning", name=MEMBER_REPORT),
                   inputs=[SubTaskInput(name="notes", value="true")])

    reviewed = review_proposals.reviewed_values(gated.wf, item, review_proposals.resolve_inputs(item, {}),
                                                ["member"], earlier=[], answers={})

    assert reviewed.shown == {"notes": "true"}
    assert reviewed.drafted_at_run == {"member": "Ann", "loans": "true"}
    assert reviewed.values == {"member": "Ann", "loans": "true", "notes": "true"} and reviewed.unbound == ()
    record = review_proposals.binding_record(1, item, reviewed, {"member": "Ann", "loans": True, "notes": True})
    assert record["match"] is False and record["deviation"] == "drafted at run time: loans, member"


def test_a_drafted_sub_task_whose_required_value_is_removed_falls_back_to_the_agent(gated, monkeypatch, scripted):
    """The user clears the drafted title: the run does not extract a value the user never saw; the agent handles the
    sub-task with the refusal (F2), recorded as a deviation."""
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 1: inputs=none")
    assert "inputs: none;" in _planning_line(record, 1)

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert _calls(out, BOOK_BY_TITLE) == []
    assert [p.member for p in _calls(out, MEMBER_REPORT)] == ["Ann"]
    first = _bindings(gated)[0]
    assert first["match"] is False and "missing required argument title" in first["deviation"]
    assert len(lm.prompts) == 2


def test_a_draft_refusal_accepted_as_shown_falls_back_to_the_agent_with_no_new_extraction(gated, monkeypatch,
                                                                                         scripted):
    lm = scripted({"reasoning": "a guess", "title": "Persuasion"}, {"reasoning": "again", "title": "Persuasion"},
                  JOKE_ANSWER)
    planner = _planner(gated, {"reasoning": "typed fill", "title": "Persuasion", "steps_before": [], "steps_after": []})
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    assert "inputs: none;" in _planning_line(record, 1)
    assert "[check: title='Persuasion' is not written in the request]" in _planning_line(record, 1)

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert _calls(out, BOOK_BY_TITLE) == []
    assert _bindings(gated)[0]["match"] is False
    assert len(lm.prompts) == 3 and len(planner.prompts) == 1   # all in the draft: nothing at run time


# ----------------------------------------------------------------------
# F1: a sub-task bound to an execution command runs its reviewed inputs
# ----------------------------------------------------------------------

FIND_ANSWER = {"reasoning": "the name", "query": "Ann"}


def test_an_execution_command_sub_task_runs_its_reviewed_inputs_by_code(gated, monkeypatch, scripted):
    lm = scripted(EMMA_ANSWER, JOKE_ANSWER, FIND_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 2: means=plain:Library/find_member")       # F3 drafts it for find_member
    assert "inputs: query=Ann; means: plain:Library/find_member" in _planning_line(record, 2)
    drafted = len(lm.prompts)

    gated.script.steps = [("finish", {})] * 20                  # the agent only ever finishes: it writes no command
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.query for p in _calls(out, "Library/find_member")] == ["Ann"]
    assert len(lm.prompts) == drafted                           # the CME parsed the values: no extraction call
    second = _bindings(gated)[1]
    assert second == {"subtask": 2, "means": "plain:Library/find_member", "shown": {"query": "Ann"},
                      "ran": second["ran"], "match": True, "deviation": None}
    assert second["ran"]["query"] == "Ann"


def test_an_execution_command_value_the_parameter_checks_refuse_falls_back_to_the_agent(gated, monkeypatch, scripted):
    """F1 + F2: a reviewed value the CME's parameter validation refuses (a placeholder) is not run; the agent takes the
    sub-task over with the refusal in its trajectory, and the deviation is recorded."""
    scripted(EMMA_ANSWER, JOKE_ANSWER, FIND_ANSWER)
    _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 2: means=plain:Library/find_member")
    gated.ctx.process_turn("change 2: inputs=query=<query>")

    gated.script.steps = [("finish", {})] * 20
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p for p in _calls(out, "Library/find_member") if getattr(p, "query", None) == "<query>"] == []
    second = _bindings(gated)[1]
    assert second["match"] is False and second["deviation"].startswith("code did not run it")


def test_an_execution_command_with_an_unbound_required_value_falls_back_to_the_agent(gated, monkeypatch, scripted):
    scripted(EMMA_ANSWER, JOKE_ANSWER, FIND_ANSWER)
    _review(gated, monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn("change 2: means=plain:Library/find_member")
    gated.ctx.process_turn("change 2: inputs=none")

    gated.script.steps = [("finish", {})] * 20
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert _calls(out, "Library/find_member") == []
    second = _bindings(gated)[1]
    assert second["match"] is False and second["deviation"] == "unbound query, the agent ran it instead"


def test_an_execution_command_input_from_an_earlier_sub_task_takes_the_declared_field(gated, monkeypatch, scripted):
    """open_book's book_uid is declared from find_book.book_uids: drafted 'from sub-task 1', it runs with the one uid
    sub-task 1's find_book returned, not with that sub-task's whole output."""
    find_dune = "find the book Persuasion"
    open_it = "open it"
    scripted({"reasoning": "the title", "query": "Persuasion"}, {"reasoning": "no uid", "book_uid": "NOT_FOUND"})
    record = _review(gated, monkeypatch, [find_dune, open_it])
    gated.ctx.process_turn("change 1: means=plain:Library/find_book")
    gated.ctx.process_turn("change 2: means=plain:Library/open_book")
    assert "inputs: book_uid=from sub-task 1; means: plain:Library/open_book" in _planning_line(record, 2)

    gated.script.steps = [("finish", {})] * 20
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    found = _calls(out, "Library/find_book")
    assert [p.query for p in found] == ["Persuasion"]
    assert [p.book_uid for p in _calls(out, "Library/open_book")] == ["b4"]
    assert [b["match"] for b in _bindings(gated)] == [True, True]


# ----------------------------------------------------------------------
# An id input whose declared format a drafted value breaks (the round-2 L17 case)
# ----------------------------------------------------------------------

OPEN_DUNE_MESSIAH = "Find the book Dune Messiah and open it."
#: One answer for any command the draft extracts here: the extractor names a title where open_book wants a uid.
TITLE_AS_UID = {"reasoning": "the book", "book_uid": "Dune Messiah", "title": "Dune Messiah", "query": "Dune Messiah",
                "member": "Ann", "loans": False, "notes": False}


def test_a_drafted_title_is_not_accepted_as_a_uid(gated, scripted):
    """open_book declares its uid's format (``pattern``, the workflow's own ids): the extractor's title for book_uid
    breaks it, so the draft binds nothing and says why, instead of showing a title the run would not use."""
    from fastworkflow.review_proposals import draft_inputs

    scripted(*[TITLE_AS_UID] * 4)
    item = SubTask(goal=OPEN_DUNE_MESSIAH, means=Means(kind="plain", name="Library/open_book"))

    drafted = draft_inputs(gated.wf, item, item.goal, [])

    assert drafted.inputs == ()
    assert "book_uid" in drafted.refusal and "pattern" in drafted.refusal


def test_an_id_the_review_left_unbound_is_found_at_run_time_and_recorded(gated, monkeypatch, scripted):
    """L17: open_book with no find step before it. The review shows book_uid unbound (the title is not a uid), the
    agent finds the book and opens it by its uid (F2), and the turn records that deviation: nothing ran that the review
    showed differently."""
    scripted(*[TITLE_AS_UID] * 12)
    record = _review(gated, monkeypatch, [OPEN_DUNE_MESSIAH, JOKE])
    gated.ctx.process_turn("change 1: means=plain:Library/open_book")
    assert "inputs: none; means: plain:Library/open_book" in _planning_line(record, 1)

    gated.script.steps = ([("finish", {}),
                           ("execute_workflow_query", {"command": "find_book query=Dune Messiah"}),
                           ("execute_workflow_query", {"command": "open_book book_uid=b2"}),
                           ("finish", {})] + [("finish", {})] * 16)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.book_uid for p in _calls(out, "Library/open_book")] == ["b2"]
    first = _bindings(gated)[0]
    assert first["shown"] == {} and first["ran"] is None and first["match"] is False
    assert first["deviation"] == "unbound book_uid, the agent ran it instead"


# ----------------------------------------------------------------------
# F4: the bookkeeping field is not shown
# ----------------------------------------------------------------------


def test_drafted_means_is_neither_shown_nor_editable(gated, monkeypatch, scripted):
    from fastworkflow.review_engine import Change, EditRefused, apply_edits

    scripted(EMMA_ANSWER, JOKE_ANSWER)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    assert "drafted_means" not in record.rendering
    out = gated.ctx.process_turn("change 1: drafted_means=Library/count_books")
    assert "drafted_means" not in (out.answer or "") or "has no field 'drafted_means'" in (out.answer or "")
    proposal = review_proposals.SubTaskProposal(items=[SubTask(goal=EMMA)])
    with pytest.raises(EditRefused) as refused:
        apply_edits(proposal, [Change(index=1, field="nope", value="x")])
    assert "drafted_means" not in str(refused.value) and "goal" in str(refused.value)
    with pytest.raises(EditRefused):
        apply_edits(proposal, [Change(index=1, field="drafted_means", value="x")])


# ----------------------------------------------------------------------
# The draft runs the typed fill fallback, as the run does
# ----------------------------------------------------------------------


def test_the_draft_runs_the_typed_fill_fallback_and_the_run_uses_its_value(gated, monkeypatch, scripted):
    """author_books: the extractor and its retry miss the bare surname; the typed fill fallback (the one
    typed_planning_call runs) gives it, the review shows it, and the run uses it."""
    austen = "show every Austen book in the library"
    lm = scripted({"reasoning": "no author", "author": "NOT_FOUND"},
                  {"reasoning": "still none", "author": "NOT_FOUND"},
                  JOKE_ANSWER)
    planner = _planner(gated, {"reasoning": "the surname is the author", "author": "Austen",
                                "steps_before": [], "steps_after": []})
    record = _review(gated, monkeypatch, [austen, JOKE])
    plan = gated.ctx._turn_subtask_plan
    assert plan.items[0].means == Means(kind="planning", name=AUTHOR_BOOKS)
    assert plan.items[0].inputs == [SubTaskInput(name="author", value="Austen", drafted=True)]
    assert "[check:" not in _planning_line(record, 1)
    drafted = len(lm.prompts)
    assert drafted == 3 and len(planner.prompts) == 1 and austen in planner.prompts[0]

    gated.script.steps = list(STEPS)
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED
    assert [p.author for p in _calls(out, AUTHOR_BOOKS)] == ["Austen"]
    assert len(lm.prompts) == drafted and len(planner.prompts) == 1


def test_a_draft_the_typed_fill_cannot_fix_shows_the_refusal(gated, monkeypatch, scripted):
    from fastworkflow.review_proposals import draft_inputs

    scripted({"reasoning": "no author", "author": "NOT_FOUND"}, {"reasoning": "still none", "author": "NOT_FOUND"})
    planner = _ScriptedLM([{"reasoning": "a guess", "author": "Tolstoy",
                                          "steps_before": [], "steps_after": []}], json=False)
    item = SubTask(goal="show every book by this author", means=Means(kind="planning", name=AUTHOR_BOOKS))

    drafted = draft_inputs(gated.wf, item, item.goal, [], planner_lm=planner)

    # the typed fill ran (one planner call) and its guess is refused by the same checks: the refusal stands
    assert len(planner.prompts) == 1
    assert drafted.inputs == () and drafted.refusal == "missing required argument author"


# ----------------------------------------------------------------------
# Drafting off: as before
# ----------------------------------------------------------------------


def test_with_drafting_off_the_run_is_unchanged(gated, monkeypatch, scripted):
    """No draft, no extraction, and a planning means with no inputs is refused and handed to the agent as before."""
    monkeypatch.setenv(review_proposals.DRAFT_ARGUMENTS_ENV_VAR, "off")
    lm = scripted()
    queries: list = []

    def planner(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, **kwargs):
        queries.append(user_query)
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", planner)
    record = _review(gated, monkeypatch, [EMMA, JOKE])
    assert "inputs: none;" in _planning_line(record, 1) and "[check: missing required argument title]" in \
        _planning_line(record, 1)

    # The review frame used to take its own finish step. Its result-driven last phase now
    # takes none, so only the agent for each sub-task needs one.
    gated.script.steps = [("finish", {})] * 2
    out = gated.ctx.process_turn("accept")

    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []    # exactly as before: the agent runs both
    assert _calls(out, BOOK_BY_TITLE) == []
    assert "The planning command Library/book_by_title cannot run as planned" in queries[0]
    assert lm.prompts == []
    assert _bindings(gated) == []                               # nothing recorded with drafting off
