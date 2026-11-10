"""The answer hook (design 2.34, 2.45, 3.16; fix-3rsq.p14.4): ``__result__`` writes a planning command's answer.

Runtime cases run through the library copy's real phase runner (``library_env``); the model
is scripted by the fixture, and ``write_answer`` (the one model call that writes an answer
with no tools) is recorded here. Load cases build the planning command the loader builds.
Preview cases render the command's preview from the same ``__result__``. No mocks of the
framework: the only seams are the fixtures' model boundary and ``write_answer``.
"""
from __future__ import annotations

import logging
import types

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import command_preview, planning_commands, planning_phases, tracing, workflow_agent
from fastworkflow.planning import phase, run
from fastworkflow.planning_commands import PlanningCommandFormatError
from fastworkflow.turn import TurnStatus
from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_step_helpers import FoundBooks, OpenedBook, _build_planning_command, library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

from fastworkflow.examples.library._commands.Library.open_book import Signature as OpenBook

SAY_THAT_FILE = "_commands/Library/book_lookup.py"
SAY_THAT_LOAD_ERROR = (
    f"{SAY_THAT_FILE}: phase 'answer' writes the answer as 'say that' lines, which code no longer "
    "copies; return the text from __result__(self, input, ...) instead")


class Declared(BaseModel):
    """The Output a planning command declares when a case needs one."""

    found: bool = Field(description="whether a book matched")


class Other(BaseModel):
    """An Output of another class than the one declared."""

    found: bool = False


@pytest.fixture(autouse=True)
def _fresh_command_caches():
    # library_env copies the library workflow into tmp_path and loads its command modules.
    planning_commands.resolve_field_sources.cache_clear()
    workflow_agent._build_parsed_planning_commands.cache_clear()
    yield
    planning_commands.resolve_field_sources.cache_clear()
    workflow_agent._build_parsed_planning_commands.cache_clear()


@pytest.fixture
def answered(monkeypatch):
    """Record each ``write_answer`` call (the one model call that writes an answer)."""
    queries: list[str] = []

    def write(query):
        queries.append(query)
        return "written"

    monkeypatch.setattr("fastworkflow.planning_phases.write_answer", write)
    return queries


def _run(env, planning_command, args: dict | None = None):
    env.args = dict(args or {"title": "Dune"})
    env.planning_command = planning_command
    return env.ctx.process_turn("look up the book")


def _outputs(env, planning_command):
    return [o for o in env.ctx._turn_outputs if o.command_name == planning_command.name]


def _command_with_declared_output(plan_cls):
    """A planning command declaring ``Signature.Output`` (``Declared``), built by the loader."""
    input_cls = type("Input", (BaseModel,),
                     {"__annotations__": {"title": str},
                      "title": Field(description="the book's title")})
    sig = type("Signature", (), {"__doc__": "Look up a book.", "Input": input_cls, "Output": Declared})
    module = types.ModuleType("book_lookup")
    module.Signature, module.PlanGenerator = sig, plan_cls
    return planning_commands.command_from_module(
        "Library/book_lookup", module, SAY_THAT_FILE).command


# ----------------------------------------------------------------------
# __result__ is the answer, and no phase writes one
# ----------------------------------------------------------------------

def test_a_result_text_is_the_answer_exactly_and_no_phase_writes_one(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return f"{input.title}: {len(find.book_uids)} found"

    command = _build_planning_command(Plan)
    result = _run(library_env, command)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "Dune: 2 found"
    assert answered == []
    assert library_env.model.asked == []
    # The last phase is framed to run its steps and finish, not to answer the request (3.16):
    # code ran its steps, and the agent is asked only to finish; its own final answer is
    # discarded, so no answer call is made on the request either.
    assert "Do only these steps, then finish" in library_env.model.closings[-1]["query"]
    assert all("Answer the request" not in query for query in library_env.model.answers)
    out = _outputs(library_env, command)
    assert len(out) == 1 and out[0].command_response.output is None and out[0].success is True


def test_a_result_that_is_a_command_response_gives_its_text(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return fastworkflow.CommandResponse(response=f"{len(find.book_uids)} copies of {input.title}")

    result = _run(library_env, _build_planning_command(Plan))

    assert result.answer == "2 copies of Dune"
    assert answered == []


def test_result_gets_none_for_a_skipped_phase(library_env, answered):
    seen = []

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find) -> OpenedBook:
            # Filling nothing skips the phase: its output is None for __result__.
            return [run("open_book", named=input.title)] if find.book_uids else []

        def __result__(self, input, find, book):
            seen.append(book)
            return f"{len(find.book_uids)} found; book is {'none' if book is None else book.title}"

    result = _run(library_env, _build_planning_command(Plan), {"title": "NoSuchTitleAtAll"})

    assert result.status == TurnStatus.COMPLETED
    assert seen == [None]
    assert result.answer == "0 found; book is none"


def test_result_gets_none_for_a_phase_not_in_this_calls_phases(library_env, answered):
    seen = []

    class Plan:
        def __phases__(self, input):
            return [self.find] if input.short else [self.find, self.book]

        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def book(self, input, find) -> OpenedBook:
            return [run("open_book", named=input.title)]

        def __result__(self, input, find, book):
            seen.append(book)
            return f"found {len(find.book_uids)}"

    command = _build_planning_command(
        Plan, {"title": (str, Field(description="the book's title")),
               "short": (bool, Field(default=True, description="only find"))})
    result = _run(library_env, command, {"title": "Dune", "short": True})

    assert result.answer == "found 2"
    assert seen == [None]
    assert answered == []


def test_a_capped_fan_out_appends_its_note_after_the_text(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        @phase(over="find.book_uids", max_items=1)
        def opened(self, uid) -> OpenBook.Output:
            return [run("open_book", book_uid=uid)]

        def __result__(self, input, find, opened):
            return "opened " + ", ".join(item.title for item in opened)

    result = _run(library_env, _build_planning_command(Plan))

    note = planning_phases.cap_note("opened", 1, 2, None)
    assert result.status == TurnStatus.COMPLETED
    assert result.answer.startswith("opened ")
    assert result.answer.endswith("\n\n" + note)
    assert answered == []


def test_without_result_the_last_phases_agent_writes_the_answer(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

    result = _run(library_env, _build_planning_command(Plan))

    assert result.status == TurnStatus.COMPLETED
    # The phase ran find_book in code; the agent that ran it wrote the answer, not write_answer.
    assert library_env.runner.sent == ["find_book query=Dune"]
    assert answered == []
    assert library_env.model.answers == ["the answer"] or result.answer == "the answer"


def test_without_result_a_last_phase_naming_no_command_is_one_model_call_with_no_tools(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, input, find):
            return ["list the books that match the title"]

    result = _run(library_env, _build_planning_command(Plan))

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "written"
    assert len(answered) == 1
    assert library_env.runner.sent == ["find_book query=Dune"]
    assert library_env.model.asked == []


def test_the_fallback_model_call_is_given_the_facts_the_phases_filled(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            raise RuntimeError("boom")

    _run(library_env, _build_planning_command(Plan))

    assert len(answered) == 1
    # Facts are shown readably: a uid column is left out where the labels name the rows.
    assert "Dune" in answered[0] and "Dune Messiah" in answered[0]


# ----------------------------------------------------------------------
# A __result__ that fails or returns another form: one warning, the answer model-written
# ----------------------------------------------------------------------

def _raises(input, find):
    raise RuntimeError("boom")


def _empty_text(input, find):
    return ""


def _other_output(input, find):
    return Other(found=True)


def _an_int(input, find):
    return 3


@pytest.mark.parametrize("result_fn", [_raises, _empty_text, _other_output, _an_int],
                         ids=["raises", "empty-str", "other-output-class", "int"])
def test_a_bad_result_logs_one_warning_and_the_model_writes_the_answer(
        library_env, answered, caplog, result_fn):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return result_fn(input, find)

    command = _command_with_declared_output(Plan)
    with caplog.at_level(logging.WARNING):
        result = _run(library_env, command)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "written"
    assert len(answered) == 1
    assert sum(1 for r in caplog.records if "__result__ failed" in r.message) == 1
    out = _outputs(library_env, command)
    assert len(out) == 1
    assert out[0].command_response.output is None
    assert out[0].success is True


# ----------------------------------------------------------------------
# The load: a last phase of 'say that' lines is an error naming __result__ (3.16)
# ----------------------------------------------------------------------

def _say_that_last(answer_lines):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, find):
            return answer_lines(find)

    return Plan


def test_a_last_phase_of_say_that_lines_fails_the_load_naming_result():
    plan = _say_that_last(lambda find: ["say that the books match the title"])

    with pytest.raises(PlanningCommandFormatError) as caught:
        _build_planning_command(plan)
    assert str(caught.value) == SAY_THAT_LOAD_ERROR


def test_a_last_phase_of_say_that_and_a_guard_fails_the_load():
    plan = _say_that_last(lambda find: ["say that the books match the title", "apply no change"])

    with pytest.raises(PlanningCommandFormatError, match="writes the answer as 'say that' lines"):
        _build_planning_command(plan)


def test_a_last_phase_with_a_judgement_step_loads():
    plan = _say_that_last(lambda find: ["say that the books match", "list the titles that match"])

    assert _build_planning_command(plan).plan_fn is not None


def test_a_guard_alone_as_the_last_phase_loads():
    plan = _say_that_last(lambda find: ["apply no change"])

    assert _build_planning_command(plan).plan_fn is not None


def test_say_that_as_a_verbatim_step_in_another_phase_loads():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def note(self, find):
            return ["say that the search ran"]

        def answer(self, find):
            return ["list the titles that match the title"]

    assert _build_planning_command(Plan).plan_fn is not None


# ----------------------------------------------------------------------
# The preview shows the text __result__ writes, with facts as placeholders (3.7), or that it
# is decided at run time (3.17)
# ----------------------------------------------------------------------

def _preview(planning_command) -> str:
    return command_preview.render_preview("look_up", "Look up a book.", (), planning_command, (),
                                          SAY_THAT_FILE)


def _answer_lines(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index("## Answer")
    return [line for line in lines[start:] if line.strip()][:2]


def test_the_preview_shows_the_text_result_writes_with_placeholder_facts():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return f"{input.title}: {len(find.book_uids)} found"

    text = _preview(_build_planning_command(Plan))

    assert _answer_lines(text) == ["## Answer", "Written by `__result__`: \\<title>: 1 found"]


def test_the_preview_says_decided_at_run_time_when_result_raises():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            raise TypeError("no")

    text = _preview(_command_with_declared_output(Plan))

    assert _answer_lines(text)[1] == (
        "Written by `__result__`: decided at run time (TypeError while the template was built, "
        "with every fact unknown).")


def test_the_preview_says_decided_at_run_time_when_result_builds_an_output_from_placeholders():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return Declared(found=find.book_uids)

    text = _preview(_command_with_declared_output(Plan))

    assert "Written by `__result__`: decided at run time (" in _answer_lines(text)[1]


def test_the_preview_says_decided_at_run_time_for_an_output_return():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return Declared(found=True)

    text = _preview(_command_with_declared_output(Plan))

    assert _answer_lines(text)[1] == (
        "Written by `__result__`: decided at run time (it returns an Output, which the agent "
        "words at run time).")


def test_a_command_without_result_has_no_answer_section():
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

    text = _preview(_build_planning_command(Plan))

    assert "## Answer" not in text


# ----------------------------------------------------------------------
# A None from __result__ hands the answer back, as with no __result__ (3.16)
# ----------------------------------------------------------------------

def test_none_from_result_gives_a_flat_call_the_agents_answer(library_env, answered, caplog):
    class Plan:
        # One phase that fills no facts and reads none: the call runs flat (planning.runs_flat).
        def find(self, input):
            return [run("find_book", query=input.title)]

        def __result__(self, input):
            return None

    command = _build_planning_command(Plan)
    with caplog.at_level(logging.WARNING):
        result = _run(library_env, command)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "the answer"
    assert answered == []
    assert not any("__result__" in r.message for r in caplog.records)
    assert _outputs(library_env, command) == []


def test_none_from_result_gives_a_phased_call_one_deferred_answer_from_its_facts(library_env, answered, caplog):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return None

    command = _build_planning_command(Plan)
    with caplog.at_level(logging.WARNING):
        result = _run(library_env, command)

    assert result.status == TurnStatus.COMPLETED
    # The last phase's agent ran its steps only; the one model call that writes the answer
    # is made now, from the facts the phase filled (3.16), and it has its own span.
    assert result.answer == "written"
    assert len(answered) == 1
    # Facts are shown readably: a uid column is left out where the labels name the rows.
    assert "Dune" in answered[0] and "Dune Messiah" in answered[0]
    assert not any("__result__" in r.message for r in caplog.records)
    assert _outputs(library_env, command) == []
    spans = library_env.sink.named(tracing.SPAN_PLANNING_COMMAND_ANSWER)
    assert len(spans) == 1
    assert spans[0].attributes["skill"] == command.name
    assert spans[0].attributes["phase"] == "find"
    assert spans[0].attributes["phase_index"] == 0
    assert spans[0].attributes["phase_count"] == 1


def test_a_text_result_means_the_last_phases_agent_is_not_asked_for_an_answer(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return "the result text"

    result = _run(library_env, _build_planning_command(Plan))

    assert result.answer == "the result text"
    assert answered == []
    # The last phase is framed to run its steps and finish, not to answer the request (3.16).
    last_phase_query = library_env.model.closings[-1]["query"]
    assert "Do only these steps, then finish" in last_phase_query
    assert "Answer the request" not in last_phase_query
    assert library_env.sink.named(tracing.SPAN_PLANNING_COMMAND_ANSWER) == []


def test_none_from_result_lets_the_model_write_a_last_answer_phase(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, input, find):
            return ["list the books that match the title"]

        def __result__(self, input, find, answer):
            return None

    result = _run(library_env, _build_planning_command(Plan))

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "written"
    assert len(answered) == 1


def test_a_result_text_gives_no_model_call_for_a_last_answer_phase(library_env, answered):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, input, find):
            return ["list the books that match the title"]

        def __result__(self, input, find, answer):
            return "the result text"

    result = _run(library_env, _build_planning_command(Plan))

    assert result.answer == "the result text"
    assert answered == []


def test_a_flat_say_that_answer_fails_the_load():
    class Plan:
        def answer(self, input):
            return ["say that the books match the title"]

    with pytest.raises(PlanningCommandFormatError, match="writes the answer as 'say that' lines"):
        _build_planning_command(Plan)


def test_a_flat_say_that_answer_is_refused_at_run_time_when_the_load_check_cannot_read_it(
        library_env, answered):
    # Judgement at the load check's placeholder input, say-that lines for the run's "Dune".
    class Plan:
        def answer(self, input):
            if input.title == "Dune":
                return ["say that the books match the title"]
            return ["list the books that match the title"]

    command = _build_planning_command(Plan)
    result = _run(library_env, command)

    assert "writes the answer as 'say that' lines" in result.answer
    assert answered == []
    assert library_env.runner.sent == []


# ----------------------------------------------------------------------
# (a) A phase before the last made only of 'say that' lines writes no part of the answer (3.16)
# ----------------------------------------------------------------------

def test_a_phase_before_the_last_of_say_that_lines_is_not_in_the_answer(library_env, answered):
    class Plan:
        def note(self, input):
            return ["say that the search was run"]

        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

    result = _run(library_env, _build_planning_command(Plan))

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "the answer"
    assert answered == []


def test_a_phase_before_the_last_of_say_that_lines_reaches_the_answer_only_through_result(
        library_env, answered):
    class Plan:
        def note(self, input):
            return ["say that the search was run"]

        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return f"{len(find.book_uids)} found"

    result = _run(library_env, _build_planning_command(Plan))

    assert result.answer == "2 found"
    assert answered == []


# ----------------------------------------------------------------------
# (b) A last phase of 'say that' lines whose template did not build is refused at run time (3.16)
# ----------------------------------------------------------------------

def test_a_last_say_that_phase_the_load_check_cannot_read_is_refused_at_run_time(library_env, answered):
    # The template reads the title as "<title>", so the load check sees judgement; the run's
    # title gives say-that lines only, which the run refuses as the load check would (3.16).
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, input, find):
            if input.title == "Dune":
                return ["say that the books match the title"]
            return ["list the books that match the title"]

    result = _run(library_env, _build_planning_command(Plan))

    assert SAY_THAT_LOAD_ERROR in result.answer
    assert answered == []
