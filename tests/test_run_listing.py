"""The shortlist of what a sub-task's run is shown (design 3.10, Phase 10 step 3; plan step 6).

Each case runs on a COPY of a real workflow: the library example (the large one the shortlist is proven on),
``tests/todo_list_workflow`` (small: its full listing is given whole) and ``tests/planning_tutorial_workflow``
(planning commands with callees, no descend edges). Nothing is mocked and no model is called: the planning-command
match's distribution is given as data (the probabilities a Jev call would return), and the listing builders run on
the copy's own routing definition, command classes and planning commands.

Left out on purpose: the assistant-mode refusal (10.8, tests/test_assistant_mode_planning.py), the ``enter_command``
load error and the undeclared-move warning (10.4, tests/test_descends_to_navigation.py), and the head labels that
hold planning commands (10.12, tests/test_planning_commands.py). The map of the 10.9 cases to their tests is in the
notes of fix-3rsq.10.9.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import planning_commands, run_listing, workflow_agent
from fastworkflow.planning_commands import PlanningCommandInputError
from fastworkflow.run_listing import (
    COMMAND_SHORTLIST, FULL, FULL_LISTING_MAX_CHARS, ARMS, THRESHOLD, check_planned_arguments,
    listing_arm, planning_candidates, run_listing_text, shortlist_for)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

ROOT = Path(__file__).resolve().parent
LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
TODO = ROOT / "todo_list_workflow"
TUTORIAL = ROOT / "planning_tutorial_workflow"

_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)


@pytest.fixture
def copy_of(tmp_path, initialized_fastworkflow):  # noqa: F811 - fixture; clears the registry per test
    """Open a fresh copy of a real workflow as the agent's app workflow; the context is popped at teardown."""
    pushed: list[WorkflowExecutionContext] = []

    def open_copy(source: Path, name: str, *, root_context=None):
        folder = tmp_path / name
        shutil.copytree(source, folder, ignore=_COPY_IGNORE, symlinks=True)
        app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"listing-{name}")
        ctx = WorkflowExecutionContext(run_as_agent=True)
        ctx.bind_app_workflow(app)
        ctx.push_active_workflow(app)
        pushed.append(ctx)
        _CONTEXT_OF[id(app)] = ctx
        if root_context is not None:
            app.root_command_context = root_context
            app.current_command_context = root_context
        return app

    yield open_copy
    for ctx in pushed:
        ctx.pop_active_workflow()
    _CONTEXT_OF.clear()


def _library_in(open_copy, *, name="library"):
    """The library copy, its command context entered at the Library root (the context the sub-tasks are run in)."""
    from fastworkflow.examples.library.application.library import Library

    return open_copy(LIBRARY, name, root_context=Library())


#: The execution context the copy_of fixture opened for each app (keyed by the app's id).
_CONTEXT_OF: dict[int, WorkflowExecutionContext] = {}


def _ctx_for(app):
    return _CONTEXT_OF[id(app)]


def run_list_bound(workflow, name: str) -> str:
    return run_listing.bound_command_listing(workflow, name)


def _lines_after(text: str, header: str) -> list[str]:
    """The lines of the section that starts with *header* (up to the next section)."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    out = []
    for line in lines[start + 1:]:
        if line.startswith(("Commands available", "Other contexts")):
            break
        out.append(line)
    return out


def _command_names(text: str) -> set[str]:
    """The command names a listing gives a line to (its ``- name`` lines, before any description)."""
    return {line[2:].strip() for line in text.split("\n") if line.startswith("- ")}


# ---------------------------------------------------------------------------
# Production is unchanged; the selector is experiment-only
# ---------------------------------------------------------------------------

def test_production_listing_is_todays_full_listing_under_the_threshold_on_every_workflow(copy_of):
    """The production policy (no experiment arm) gives FULL, which is ``planner_commands_text`` as it is today, on all
    three workflows: each is at most FULL_LISTING_MAX_CHARS."""
    assert run_listing.selected_listing_arm() is None
    for app in (_library_in(copy_of), copy_of(TODO, "todo"), copy_of(TUTORIAL, "tutorial")):
        full = workflow_agent.planner_commands_text(app)
        assert len(full) <= FULL_LISTING_MAX_CHARS
        assert run_listing_text(app, "run list_member_loans for the member") == full


def test_selector_defaults_to_none_restores_the_previous_arm_and_rejects_unknown_arms():
    with listing_arm(COMMAND_SHORTLIST):
        assert run_listing.selected_listing_arm() == COMMAND_SHORTLIST
        with listing_arm(None):
            assert run_listing.selected_listing_arm() is None
        assert run_listing.selected_listing_arm() == COMMAND_SHORTLIST
    assert run_listing.selected_listing_arm() is None
    with pytest.raises(ValueError, match="unknown listing arm"):
        with listing_arm("everything"):
            pass
    with pytest.raises(ValueError, match="unknown listing arm"):
        with listing_arm("by_context"):   # the BY_CONTEXT arm is removed (10.15)
            pass
    assert run_listing.selected_listing_arm() is None
    assert ARMS == (FULL, COMMAND_SHORTLIST, THRESHOLD)
    assert not hasattr(run_listing, "by_context_text") and not hasattr(run_listing, "BY_CONTEXT")


def test_the_production_call_sites_are_byte_identical_with_the_selector_unset(copy_of):
    """With no experiment arm (or FULL), the run's listing is ``_what_can_i_do`` and the planner's is
    ``planner_commands_text``, byte for byte, on all three workflows."""
    from fastworkflow.workflow_agent import _what_can_i_do, planner_listing_text, sub_task_commands_text

    assert run_listing.selected_listing_arm() is None
    for app in (_library_in(copy_of, name="identity_library"), copy_of(TODO, "identity_todo"),
                copy_of(TUTORIAL, "identity_tutorial")):
        ctx = _ctx_for(app)
        expected_run = _what_can_i_do(ctx)
        expected_planner = workflow_agent.planner_commands_text(app)
        assert len(expected_run) <= FULL_LISTING_MAX_CHARS and len(expected_planner) <= FULL_LISTING_MAX_CHARS
        assert sub_task_commands_text(ctx, "run list_member_loans for the member") == expected_run
        assert planner_listing_text(app, "run list_member_loans for the member") == expected_planner
        with listing_arm(FULL):
            assert run_listing.selected_listing_arm() == FULL
            assert sub_task_commands_text(ctx, "run list_member_loans for the member") == expected_run
            assert planner_listing_text(app, "run list_member_loans for the member") == expected_planner


def test_the_selected_shortlist_arm_routes_both_production_call_sites(copy_of):
    from fastworkflow.workflow_agent import _what_can_i_do, planner_listing_text, sub_task_commands_text

    library = _library_in(copy_of, name="routed_library")
    ctx = _ctx_for(library)
    sub_task = "run list_member_loans for the member"
    with listing_arm(COMMAND_SHORTLIST):
        assert run_listing.selected_listing_arm() == COMMAND_SHORTLIST
        assert sub_task_commands_text(ctx, sub_task) == run_listing_text(
            library, sub_task, arm=COMMAND_SHORTLIST, planning_probabilities=None)
        assert planner_listing_text(library, sub_task) == run_listing_text(
            library, sub_task, arm=COMMAND_SHORTLIST, planning_probabilities=None)
        assert sub_task_commands_text(ctx, sub_task) != _what_can_i_do(ctx)
    with listing_arm(FULL):
        assert sub_task_commands_text(ctx, sub_task) == _what_can_i_do(ctx)


def test_the_production_default_is_threshold_and_an_untrained_app_gets_full_above_the_threshold(copy_of, monkeypatch):
    """The production arm is THRESHOLD (fix-3rsq.plan-review.2.10), guarded: a listing above the threshold is still the
    full listing when the workflow has no trained heads (the shortlist would have no trained recall)."""
    from fastworkflow.workflow_agent import _what_can_i_do, planner_listing_text, sub_task_commands_text

    library = _library_in(copy_of, name="production_default")
    ctx = _ctx_for(library)
    sub_task = "run list_member_loans for the member"
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    assert run_listing.PRODUCTION_LISTING_ARM == THRESHOLD
    assert len(workflow_agent.planner_commands_text(library)) > 3000
    assert not run_listing.has_trained_heads(library)
    assert run_listing.selected_listing_arm() is None
    assert run_listing.effective_listing_arm() == THRESHOLD
    assert run_listing.effective_listing_arm(library) == FULL       # the guard: no trained heads
    assert run_listing_text(library, sub_task) == workflow_agent.planner_commands_text(library)
    assert planner_listing_text(library, sub_task) == workflow_agent.planner_commands_text(library)
    assert sub_task_commands_text(ctx, sub_task) == _what_can_i_do(ctx)


def test_the_production_threshold_shortlists_a_workflow_with_trained_heads(copy_of, monkeypatch):
    """With the trained heads and no experiment arm, production shortlists above the threshold at both call sites."""
    from fastworkflow.workflow_agent import _what_can_i_do, planner_listing_text, sub_task_commands_text

    library = _library_in(copy_of, name="production_threshold_heads")
    ctx = _ctx_for(library)
    _with_trained_heads(library)
    sub_task = "who wrote Emma and how many copies do we have"
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    assert run_listing.has_trained_heads(library)
    shortlist = run_listing_text(library, sub_task, arm=COMMAND_SHORTLIST)
    assert run_listing.effective_listing_arm(library) == THRESHOLD
    assert run_listing_text(library, sub_task) == shortlist
    assert planner_listing_text(library, sub_task) == shortlist
    assert sub_task_commands_text(ctx, sub_task) == shortlist
    assert shortlist != workflow_agent.planner_commands_text(library)
    assert "- book_by_title" in shortlist


def test_the_production_guard_does_not_touch_an_experiment_arm(copy_of, monkeypatch):
    """An untrained workflow: production gives FULL above the threshold (the guard), but an experiment's THRESHOLD
    still shortlists, as the 10.7 measurement needs it to."""
    library = _library_in(copy_of, name="guard_experiment_arm")
    sub_task = "run list_member_loans for the member"
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    full = workflow_agent.planner_commands_text(library)
    assert run_listing_text(library, sub_task) == full
    with listing_arm(THRESHOLD):
        assert run_listing_text(library, sub_task) != full
        assert run_listing_text(library, sub_task) == run_listing_text(
            library, sub_task, arm=COMMAND_SHORTLIST)


def test_has_trained_heads_needs_a_complete_head_folder(copy_of):
    """The guard's test: no ___command_info is no head; a context folder missing its model files is no head; the
    library's linked heads are heads."""
    import os

    library = _library_in(copy_of, name="has_trained_heads")
    assert not run_listing.has_trained_heads(library)
    partial = Path(library.folderpath) / "___command_info" / "partial_context"
    partial.mkdir(parents=True)
    (partial / "tinymodel.pth").write_text("x")
    assert not run_listing.has_trained_heads(library)
    root = _with_trained_heads(library)
    assert os.path.isdir(root) and run_listing.has_trained_heads(library)


def test_the_threshold_arm_shortlists_above_the_threshold_at_both_call_sites(copy_of, monkeypatch):
    """The THRESHOLD arm (the rule the 10.7 gate may adopt): above FULL_LISTING_MAX_CHARS the command shortlist at both
    call sites. The library's listings are under the real threshold, so it is lowered here."""
    from fastworkflow.workflow_agent import _what_can_i_do, planner_listing_text, sub_task_commands_text

    library = _library_in(copy_of, name="threshold_cut")
    ctx = _ctx_for(library)
    sub_task = "run list_member_loans for the member"
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    assert len(workflow_agent.planner_commands_text(library)) > 3000
    with listing_arm(THRESHOLD):
        shortlist = run_listing_text(library, sub_task, arm=COMMAND_SHORTLIST, planning_probabilities=None)
        assert planner_listing_text(library, sub_task) == shortlist
        assert sub_task_commands_text(ctx, sub_task) == run_listing_text(
            library, sub_task, arm=COMMAND_SHORTLIST, planning_probabilities=None, full=_what_can_i_do(ctx))
        assert sub_task_commands_text(ctx, sub_task) != _what_can_i_do(ctx)
    # the experiment's FULL still gives today's listing above the threshold
    with listing_arm(FULL):
        assert planner_listing_text(library, sub_task) == workflow_agent.planner_commands_text(library)
        assert sub_task_commands_text(ctx, sub_task) == _what_can_i_do(ctx)


def _with_trained_heads(app) -> str:
    """Give the copy the source library's trained heads, as symlinks (the 5 GB of versions is not copied): the copy's
    ___command_info holds one link per context folder with a trained head. Returns the copy's ___command_info."""
    import os

    source = LIBRARY / "___command_info"
    target = Path(app.folderpath) / "___command_info"
    target.mkdir(exist_ok=True)
    for name in os.listdir(source):
        path = source / name
        if path.is_dir() and (path / "threshold.json").exists():
            os.symlink(path, target / name)
    return str(target)


def test_a_natural_sub_task_gets_its_command_from_the_trained_heads_alone(copy_of, monkeypatch):
    """Production gathering with the trained heads: the sub-task names no command, and the name-only gather misses
    book_by_title, but the library's head answers it (a planning-command label in the head's top-k), so the shortlist
    holds it and the threshold arm lists it."""
    library = _library_in(copy_of, name="heads_natural")
    _with_trained_heads(library)
    sub_task = "who wrote Emma and how many copies do we have"
    assert "Library/book_by_title" not in shortlist_for(library, sub_task)        # names only: no
    labels, planning_scores = run_listing.head_seeds(library, sub_task)
    assert "Library/book_by_title" in planning_scores or "Library/book_by_title" in labels
    assert "Library/book_by_title" in run_listing.shortlist_live(library, sub_task)  # with the heads: yes

    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    with listing_arm(THRESHOLD):
        text = run_listing_text(library, sub_task)
    assert "\n- book_by_title\n" in text or text.startswith("- book_by_title\n")
    assert len(text) < len(workflow_agent.planner_commands_text(library))


def test_head_candidates_keep_the_execution_top_k_as_well_as_the_planning_top_k(copy_of):
    """Round 1's candidates (fix-3rsq.plan-review.2.1): the heads' top-k execution commands are kept, not only the
    planning ones. Real trained heads (the library's, linked in); the split is by kind and never holds a non-command."""
    from fastworkflow.run_listing import NON_COMMAND_LABELS, head_candidates

    library = _library_in(copy_of, name="head_candidates_execution")
    _with_trained_heads(library)
    sub_task = "run list_member_loans for the member"
    candidates = head_candidates(library, sub_task)
    labels, planning = run_listing.head_seeds(library, sub_task)
    # head_seeds is unchanged for its callers: the same labels and the same planning scores
    assert labels == candidates.labels and planning == candidates.planning_scores
    assert candidates.execution_scores, "the heads' execution top-k must be kept"
    assert "Library/list_member_loans" in candidates.execution_scores or "list_member_loans" in {
        label.split("/")[-1] for label in candidates.execution_scores}
    assert not set(candidates.execution_scores) & set(candidates.planning_scores)
    assert not set(candidates.execution_scores) & set(NON_COMMAND_LABELS)


def test_without_trained_heads_the_live_gather_is_the_name_gather(copy_of):
    """A copy with no trained heads (no ___command_info): the live gather reads no head and equals the name gather."""
    library = _library_in(copy_of, name="heads_absent")
    sub_task = "who wrote Emma and how many copies do we have"
    assert run_listing.head_answers(library, sub_task) == {}
    assert run_listing.shortlist_live(library, sub_task) == run_listing.expand(
        library, run_listing.gather(library, sub_task))


def test_the_global_scope_is_one_line_on_a_shortlist_that_does_not_name_it(copy_of):
    """library_hours is callable from every context by name (3.9). A shortlist that does not hold it still tells the
    run it exists, in one line, under both shortlist arms."""
    library = _library_in(copy_of, name="library_global")
    sub_task = "run list_member_loans for the member"
    names = shortlist_for(library, sub_task)
    assert "library_hours" not in names
    for arm in (COMMAND_SHORTLIST,):
        text = run_listing_text(library, sub_task, arm=arm)
        lines = [line for line in text.split("\n") if line.startswith("Global commands (")]
        assert len(lines) == 1 and "library_hours" in lines[0], lines
        assert lines[0].endswith("no navigation): library_hours."), lines[0]   # startup is not callable by a sub-task's run
        assert "startup" not in lines[0]
        assert "Commands available here from the global scope" not in text
    # and when the sub-task names it, the global scope is a section in full, with no duplicate line
    named = run_listing_text(library, "what are library hours", arm=COMMAND_SHORTLIST,
                             planning_probabilities={"library_hours": 0.9})
    assert "Commands available here from the global scope" in named
    assert not [line for line in named.split("\n") if line.startswith("Global commands (")]


def test_the_threshold_is_7000_and_the_three_example_listings_are_under_it(copy_of):
    """FULL_LISTING_MAX_CHARS is 7000 (10.15, from the 10.7 measurement): the todo, library and tutorial full listings
    are each at most it, so production gives each whole. (The real app's listing, 82,055 characters, is above it.)"""
    assert FULL_LISTING_MAX_CHARS == 7000
    todo, library = copy_of(TODO, "todo_size"), _library_in(copy_of, name="library_size")
    tutorial = copy_of(TUTORIAL, "tutorial_size")
    for app in (todo, library, tutorial):
        assert len(workflow_agent.planner_commands_text(app)) <= FULL_LISTING_MAX_CHARS


# ---------------------------------------------------------------------------
# A listing under the threshold is whole; one over it is the command shortlist
# ---------------------------------------------------------------------------

def test_a_small_workflow_gets_the_full_listing_whole_under_the_production_policy(copy_of, monkeypatch):
    todo = copy_of(TODO, "todo_whole")
    full = workflow_agent.planner_commands_text(todo)
    assert len(full) <= FULL_LISTING_MAX_CHARS
    # Nothing is gathered: the text is the full listing, not a cut of it.
    assert run_listing_text(todo, "mark the item completed", planning_probabilities={"": 0.0}) == full
    assert run_listing_text(todo, "mark the item completed", arm=FULL) == full
    # even with a threshold below its size the THRESHOLD arm would shortlist it, so the rule is the size, not the workflow
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 100)
    with listing_arm(THRESHOLD):
        assert run_listing_text(todo, "mark the item completed") != full
    assert run_listing_text(todo, "mark the item completed") == full     # the production default is FULL


def test_a_large_workflow_is_cut_to_the_shortlist_above_the_threshold(copy_of, monkeypatch):
    library = _library_in(copy_of, name="library_cut")
    full = workflow_agent.planner_commands_text(library)
    command = run_listing_text(library, "run list_member_loans for the member", arm=COMMAND_SHORTLIST)
    assert len(command) < len(full)
    # THRESHOLD, lowered threshold (the library is 6,516 characters, under the real 7,000): the shortlist
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    with listing_arm(THRESHOLD):
        assert run_listing_text(library, "run list_member_loans for the member") == command


# ---------------------------------------------------------------------------
# Gathering and expansion
# ---------------------------------------------------------------------------

def test_a_command_in_a_context_not_yet_entered_is_on_the_shortlist_with_its_navigation_steps(copy_of):
    library = _library_in(copy_of, name="library_nav")
    text = run_listing_text(library, "run list_member_loans for the member", arm=COMMAND_SHORTLIST)

    assert "Member/list_member_loans" in shortlist_for(library, "run list_member_loans for the member")
    block = _lines_after(text, "Commands available after entering the Member context")
    assert block[0] == "- list_member_loans"
    assert ("  Before executing list_member_loans, navigate to Member as follows: "
            "open_member <member_uid> (needs member_uid)") in block


def test_producers_navigating_commands_and_callees_are_added(copy_of):
    library = _library_in(copy_of, name="library_expand")
    # a navigating command and its producer: list_member_loans needs Member entered, open_member enters it,
    # and open_member's uid comes from find_member
    navigating = shortlist_for(library, "run list_member_loans for the member")
    assert {"Library/open_member", "Library/find_member"} <= navigating
    # a producer in the active context: open_book's uid comes from find_book
    assert "Library/find_book" in shortlist_for(library, "run open_book")
    # a planning command's callees: member_report calls find_member, open_member, list_member_loans, show_notes
    report = shortlist_for(library, "give me the member report for Jane",
                           planning_probabilities={"Library/member_report": 0.9})
    assert "Library/member_report" in report
    assert {"Library/find_member", "Library/open_member", "Member/list_member_loans",
            "Record/show_notes"} <= report


def test_a_bound_command_listing_shows_the_producer_its_input_comes_from(copy_of):
    """open_book takes its uid from find_book (a producer in the active context): a plain sub-task bound to open_book
    gets find_book in its listing, so the agent can produce the input first (fix-3rsq.plan-review.2.11)."""
    library = _library_in(copy_of, name="library_bound_producer")
    names = _command_names(run_list_bound(library, "Library/open_book"))
    assert {"open_book", "find_book"} <= names


def test_a_bound_command_in_a_context_not_yet_entered_shows_the_entry_command(copy_of):
    """list_member_loans is in the Member context, not yet entered: its listing names the navigation to Member (the enter
    command open_member, with its uid producer find_member), as a bound sub-task needs it to reach the command."""
    library = _library_in(copy_of, name="library_bound_nav")
    text = run_list_bound(library, "Member/list_member_loans")
    assert {"list_member_loans", "open_member", "find_member"} <= _command_names(text)
    assert ("Before executing list_member_loans, navigate to Member as follows: "
            "open_member <member_uid> (needs member_uid)") in text


def test_a_bound_command_listing_leaves_out_commands_it_does_not_depend_on(copy_of):
    library = _library_in(copy_of, name="library_bound_lean")
    names = _command_names(run_list_bound(library, "Library/open_book"))
    assert "loan_report" not in names and "count_books" not in names


def test_core_commands_of_the_active_context_are_always_shortlisted(copy_of):
    library = _library_in(copy_of, name="library_core")
    names = shortlist_for(library, "run list_member_loans for the member")
    assert {"IntentDetection/go_up", "IntentDetection/reset_context"} <= names


def test_every_other_context_appears_as_one_line(copy_of):
    library = _library_in(copy_of, name="library_lines")
    text = run_listing_text(library, "run list_member_loans for the member", arm=COMMAND_SHORTLIST)

    # Book is not on the shortlist: one line, with its route
    assert "\nCommands available after entering the Book context" not in text
    book_lines = [line for line in text.split("\n") if line.startswith("- Book: ")]
    assert len(book_lines) == 1 and book_lines[0].endswith("To enter: open_book <book_uid> (needs book_uid)")
    # Book's commands are not listed in full
    assert "- show_book" not in text and "- loan_report" not in text
    # the global scope is never a line (the executor's list leaves it out), and nothing lists it in full
    assert "- *: " not in text


def test_unsure_candidates_of_the_planning_command_match_are_on_the_shortlist(copy_of):
    library = _library_in(copy_of, name="library_unsure")
    unsure = {"Library/book_by_title": 0.35, "Library/author_books": 0.3, "Library/count_books": 0.25,
              "none": 0.1}
    names = shortlist_for(library, "tell me about Dune", planning_probabilities=unsure)

    assert {"Library/book_by_title", "Library/author_books", "Library/count_books"} <= names
    # a candidate's callees come with it (book_by_title calls find_book and open_book)
    assert {"Library/find_book", "Library/open_book"} <= names


def test_planning_candidates_read_the_confident_and_the_unsure_options():
    confident, unsure = planning_candidates(
        {"Library/book_loans": 0.8, "Library/book_by_title": 0.15, "Library/count_books": 0.05, "none": 0.0})
    assert confident == "Library/book_loans"
    assert unsure == ()

    confident, unsure = planning_candidates(
        {"Library/book_by_title": 0.35, "Library/author_books": 0.3, "Library/count_books": 0.15, "none": 0.2})
    assert confident is None
    assert unsure == ("Library/book_by_title", "Library/author_books")


# ---------------------------------------------------------------------------
# Listing: the command shortlist, the shortlist by context, templates and run-time phases
# ---------------------------------------------------------------------------

def test_a_planning_command_on_a_shortlist_is_listed_with_its_template(copy_of):
    library = _library_in(copy_of, name="library_template")
    probabilities = {"Library/book_loans": 0.9}
    command = run_listing_text(library, "who has Dune on loan", arm=COMMAND_SHORTLIST,
                               planning_probabilities=probabilities)
    template = ("  template:\n  - book_by_title title=<title>\n  - list_loans (full list) (only when book.found)")
    for text in (command, workflow_agent.planner_commands_text(library)):
        assert "- book_loans\n  Say who has a book on loan, given its title.\n" in text
        assert template in text


def test_a_command_in_an_entered_context_with_a_template_is_listed_with_it_on_the_shortlist(copy_of):
    library = _library_in(copy_of, name="library_template_deferred")
    text = run_listing_text(library, "run member_report for Jane", arm=COMMAND_SHORTLIST)
    assert "- member_report" in text
    assert "  - list_member_loans (full list) (only when loans)" in text


# ---------------------------------------------------------------------------
# The BY_CONTEXT arm is removed (10.15): the command shortlist is the only shortlist
# ---------------------------------------------------------------------------

def test_the_by_context_arm_is_gone_and_the_command_shortlist_lists_only_what_is_shortlisted(copy_of):
    """BY_CONTEXT (2.40) listed every context that holds a shortlisted command in full. It is removed: the command
    shortlist lists the shortlisted commands and no other command of a context, and every other context is one line."""
    library = _library_in(copy_of, name="library_by_context")
    probabilities = {"Library/book_loans": 0.9}
    text = run_listing_text(library, "who has Dune on loan", arm=COMMAND_SHORTLIST,
                            planning_probabilities=probabilities)

    # the active context: only the shortlisted commands (and the core ones), not every Library command
    active = _lines_after(text, "Commands available in the current context (Library)")
    assert "- count_overdue" not in active and "- list_branches" not in active
    # Book holds a shortlisted command (list_loans): its section lists the shortlisted ones, not all of Book's
    book = _lines_after(text, "Commands available after entering the Book context")
    assert "- list_loans" in book and "- loan_report" not in book and "- shelf_report" not in book
    # Member holds no shortlisted command: one line
    assert "\nCommands available after entering the Member context" not in text
    assert any(line.startswith("- Member: ") for line in text.split("\n"))
    assert not hasattr(run_listing, "by_context_text")


def test_a_producer_no_from_names_is_left_out_of_the_command_shortlist(copy_of):
    """show_book (Book) is a producer-shaped command that no from_ names: the command shortlist leaves it out (the
    removed BY_CONTEXT arm listed it, as its context holds a shortlisted command)."""
    library = _library_in(copy_of, name="library_unnamed")
    probabilities = {"Library/book_loans": 0.9}
    names = shortlist_for(library, "who has Dune on loan", planning_probabilities=probabilities)
    assert "Book/show_book" not in names
    resolution = planning_commands.resolve_field_sources(library.folderpath)
    named_by_from = {str(ref).rpartition(".")[0]
                     for signature in resolution.signatures.values()
                     for field_info in getattr(getattr(signature, "Input", None), "model_fields", {}).values()
                     for ref in (field_info.json_schema_extra or {}).get("available_from", ())}
    assert "Book/show_book" not in named_by_from

    command = run_listing_text(library, "who has Dune on loan", arm=COMMAND_SHORTLIST,
                               planning_probabilities=probabilities)
    assert "- show_book" not in command


def test_the_tutorial_shortlist_carries_a_planning_commands_callees_and_one_line_per_other_context(copy_of):
    tutorial = copy_of(TUTORIAL, "tutorial_shortlist")
    text = run_listing_text(tutorial, "how many books have Dune in the title", arm=COMMAND_SHORTLIST,
                            planning_probabilities={"count_books": 0.9})
    assert len(text) < len(workflow_agent.planner_commands_text(tutorial))
    assert {"find_book", "count_books"} <= _command_names(text)
    assert "- list_branches" not in _command_names(text)


# ---------------------------------------------------------------------------
# The argument check of a planning command the agent writes as a step (3.10)
# ---------------------------------------------------------------------------

def test_the_argument_check_passes_a_value_an_earlier_step_found(copy_of):
    library = _library_in(copy_of, name="library_args")
    command = planning_commands.planning_command(library.folderpath, "Library/book_by_title")
    assert command is not None

    # the request names the title; a value an earlier step found is passed, as it is in the earlier output
    passed = check_planned_arguments(command, {"title": "Dune Messiah"}, "who wrote Dune Messiah")
    assert passed.passed and passed.args["title"] == "Dune Messiah"

    found = check_planned_arguments(command, {"title": "Foundation"}, "who wrote the book the search found",
                                    earlier_outputs=["find_book: 1 match, the title Foundation, uid b7"])
    assert found.passed and found.args["title"] == "Foundation"

    # a request with one number in it does not turn a str argument into that number (the check's int rule only)
    numbered = check_planned_arguments(command, {"title": "Dune"}, "book 2 is Dune")
    assert numbered.passed and numbered.args["title"] == "Dune"


def test_the_argument_check_refuses_a_value_in_neither_the_sub_task_nor_the_earlier_outputs(copy_of):
    library = _library_in(copy_of, name="library_args_refuse")
    command = planning_commands.planning_command(library.folderpath, "Library/book_by_title")

    refused = check_planned_arguments(command, {"title": "Foundation"}, "who wrote Dune Messiah",
                                      earlier_outputs=["find_book: 1 match, the title Dune Messiah"])
    assert not refused.passed
    assert refused.refusal and "Foundation" in refused.refusal
    # the same refusal as the planning command's own typed call gives
    with pytest.raises(PlanningCommandInputError):
        planning_commands.check_arguments(command, {"title": "Foundation"}, "who wrote Dune Messiah")


# ---------------------------------------------------------------------------
# A phase decided at run time (3.17) is not built yet: see the notes of the report
# ---------------------------------------------------------------------------

def test_the_listing_shows_a_planning_commands_template_steps_verbatim(copy_of):
    """The listing reads the template as the command builds it; the run-time phase marker (3.17, Phase 16) is
    shown when that phase exists, and the template lines are the same in every listing."""
    tutorial = copy_of(TUTORIAL, "tutorial_template")
    full = workflow_agent.planner_commands_text(tutorial)
    shortlist = run_listing_text(tutorial, "who wrote Dune", arm=COMMAND_SHORTLIST,
                                 planning_probabilities={"book_by_title": 0.9})
    template = planning_commands.planning_command(tutorial.folderpath, "book_by_title").template
    for step in template:
        assert f"  - {step}\n" in full + "\n"
        assert f"  - {step}\n" in shortlist + "\n"


# ---------------------------------------------------------------------------
# The turn's planning-match distribution, and the production arm (10.15 step 2 and 3)
# ---------------------------------------------------------------------------

def test_without_a_jev_key_the_match_leaves_the_distribution_empty_and_the_listing_is_unchanged(copy_of, monkeypatch):
    """No JEV_API_KEY: no planning match runs, so the turn's distribution stays empty and every listing is as before."""
    import os

    monkeypatch.delenv("JEV_API_KEY", raising=False)
    if os.environ.get("JEV_API_KEY"):
        pytest.skip("a JEV key is configured in this environment")
    workflow_agent._planning_command_matcher.cache_clear()
    library = _library_in(copy_of, name="no_jev_key")
    distribution: dict[str, float] = {}
    template, command = workflow_agent.planner_planning_command_for(
        "who has Dune on loan", library.folderpath, scopes=("",), distribution=distribution)
    assert distribution == {} and command is None
    sub_task = "who has Dune on loan"
    assert workflow_agent.planner_listing_text(library, sub_task, planning_probabilities=None) == \
        workflow_agent.planner_commands_text(library)
    workflow_agent._planning_command_matcher.cache_clear()


def test_the_planning_distribution_gathers_the_planning_command_into_the_threshold_listing(copy_of, monkeypatch):
    """The turn's distribution (unsure candidates included) reaches the listing: book_loans, named by no word of the
    sub-task, is on the threshold arm's shortlist with the distribution and not without it."""
    from fastworkflow.workflow_agent import _what_can_i_do, sub_task_commands_text

    library = _library_in(copy_of, name="distribution_gather")
    ctx = _ctx_for(library)
    sub_task = "tell me who is holding the copy I want"
    monkeypatch.setattr(run_listing, "FULL_LISTING_MAX_CHARS", 3000)
    distribution = {"Library/book_loans": 0.35, "Library/book_by_title": 0.3, "none": 0.1}
    with listing_arm(THRESHOLD):
        without = sub_task_commands_text(ctx, sub_task)
        with_it = sub_task_commands_text(ctx, sub_task, planning_probabilities=distribution)
    assert without == run_listing_text(library, sub_task, arm=THRESHOLD)
    assert "- book_loans" not in without and "- book_loans" in with_it
    assert "- book_by_title" in with_it               # the unsure candidate (0.3 >= the floor) is shortlisted too
    assert with_it != _what_can_i_do(ctx)


def test_the_production_arm_is_threshold_and_one_constant_switches_it():
    """PRODUCTION_LISTING_ARM is the one line that changes production; it is THRESHOLD (guarded), and the effective
    arm follows it when no experiment selects one."""
    assert run_listing.PRODUCTION_LISTING_ARM == THRESHOLD
    assert run_listing.effective_listing_arm() == THRESHOLD
    with listing_arm(COMMAND_SHORTLIST):
        assert run_listing.effective_listing_arm() == COMMAND_SHORTLIST
    assert run_listing.effective_listing_arm() == THRESHOLD


# ---------------------------------------------------------------------------
# Framework commands are hidden; a listing within the threshold gathers nothing; a run-time phase (Phase 16)
# ---------------------------------------------------------------------------

def test_the_framework_commands_are_hidden_from_every_run_listing(copy_of):
    """The framework's planning commands (review_subtasks, review_plan) are system machinery (fix-3rsq.plan-review.1):
    the loader has them, and no run's listing names them, under any arm, or gathers them for a sub-task that names one."""
    from fastworkflow.command_directory import CommandDirectory

    library = _library_in(copy_of, name="framework_hidden")
    assert "review_plan" in CommandDirectory.load(library.folderpath).map_command_2_metadata   # not vacuous
    sub_task = "run review_plan on the plan"
    assert not {name for name in shortlist_for(library, sub_task) if name.split("/")[-1] in FRAMEWORK_LEAVES}
    texts = [run_listing_text(library, sub_task, arm=FULL),
             run_listing_text(library, sub_task, arm=COMMAND_SHORTLIST),
             run_listing_text(library, sub_task)]
    for text in texts:
        assert not _command_names(text) & FRAMEWORK_LEAVES
        assert not any(leaf in text for leaf in FRAMEWORK_LEAVES)


FRAMEWORK_LEAVES = frozenset({"review_subtasks", "review_plan"})


def test_a_listing_within_the_threshold_is_given_whole_and_gathers_nothing(copy_of, monkeypatch):
    """Under the threshold the listing is the full text and no gathering runs at all (3.10): the shortlist, the name
    gather and the head answers are each made to fail if they are called."""
    todo = copy_of(TODO, "todo_gathers_nothing")
    library = _library_in(copy_of, name="library_gathers_nothing")

    def gathering_is_a_failure(*args, **kwargs):
        raise AssertionError("a listing within FULL_LISTING_MAX_CHARS must not gather a shortlist")

    for name in ("shortlist_live", "gather", "head_seeds", "shortlist_for"):
        monkeypatch.setattr(run_listing, name, gathering_is_a_failure)
    for app in (todo, library):
        full = workflow_agent.planner_commands_text(app)
        assert len(full) <= FULL_LISTING_MAX_CHARS
        with listing_arm(THRESHOLD):
            assert run_listing_text(app, "mark the item completed") == full
        assert run_listing_text(app, "mark the item completed") == full


@pytest.mark.xfail(strict=True, reason="Phase 16 (the run-time phase marker, design 3.17) is not built: no listing "
                                       "marks a phase decided at run time, so this stays an expected failure")
def test_a_phase_decided_at_run_time_is_shown_as_such(copy_of):
    """Library/member_report skips its loan and notes phases when the member is absent, a fact decided at run time.
    Once Phase 16 shows that in the listing, this passes; the strict xfail then fails and must be removed."""
    library = _library_in(copy_of, name="library_run_time_phase")
    text = run_listing_text(library, "give me the member report for Jane", arm=COMMAND_SHORTLIST,
                            planning_probabilities={"Library/member_report": 0.9})
    lines = text.split("\n")
    assert "- member_report" in lines
    block = []
    for line in lines[lines.index("- member_report") + 1:]:
        if line.startswith("- "):
            break
        block.append(line)
    assert any("run time" in line for line in block), block
