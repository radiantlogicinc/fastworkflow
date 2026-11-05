"""Phase 3 steps 1 and 1a: ``__result__`` and a skipped phase is ``None``."""
from __future__ import annotations

pytest_plugins = ["tests.test_turn_result_capture"]

import logging
import os
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator

import fastworkflow
from fastworkflow import planning_commands, command_preview, workflow_agent
from fastworkflow.capability_answer import context_commands
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.context_navigation import home_contexts
from fastworkflow.mcp_server import FastWorkflowMCPServer
from fastworkflow.model_pipeline_training import trained_command_labels
from fastworkflow.plan_frame import TURN_MAX_STEPS_ENV
from fastworkflow.planning_needs_agent import planning_command_needs_agent, refusal_message
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.workflow_agent import (
    _explicit_agent_command,
    _parameter_fields,
    _what_can_i_do,
    planner_commands_text,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.examples.library._commands.Book.list_loans import Signature as ListLoans
from fastworkflow.examples.library._commands.Library.book_loans import Signature as BookLoans
from fastworkflow.examples.library._commands.Library.book_by_title import Signature as BookByTitle
from fastworkflow.examples.library._commands.Library.find_book import Signature as FindBook
from fastworkflow.examples.library._commands.Library.find_member import Signature as FindMember
from fastworkflow.examples.library._commands.Library.open_book import Signature as OpenBook
from fastworkflow.planning import full_list, pick, run
from fastworkflow.planning import phase
from fastworkflow.planning_commands import PlanningCommandFormatError
from fastworkflow.turn import TurnStatus, reset_missing_declared_output_warnings
from fastworkflow.utils import python_utils

from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_field_sources import _build_workflow
from tests.test_planning_command_contexts import LIBRARY_PLANNING_COMMAND, ROOT_PLANNING_COMMAND, RULE_CHECK, _with_rule_check
from tests.test_planning_command_code_steps import Model
from tests.test_text_planner_and_split import _fake_matcher
from tests.test_step_helpers import (
    FoundBooks,
    LibraryRunner,
    OpenedBook,
    _build_planning_command,
    _library_workflow,
    _make_library_runner_env,
    _real_planning_command,
    library_env,
    library_proof_env,
    script_command_extraction,
    script_every_agent,
)


class Mixed(BaseModel):
    found: FindBook.Output
    open_args: OpenBook.Input
    member: FindMember.Output | None = None


class Strict(BaseModel):
    found: FindBook.Output

    @model_validator(mode="after")
    def needs_rows(self):
        if not self.found.book_uids:
            raise ValueError("no book matched")
        return self


@pytest.fixture(scope="module")
def retail_workflow_path() -> str:
    return str(
        Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve()
    )


def _planning_command_with_output(build_plan, *, output_fields: dict | None = None):
    output_fields = output_fields or {
        "found": (bool, Field(description="whether a book matched")),
    }
    output_cls = type(
        "Output",
        (BaseModel,),
        {
            "__annotations__": {k: t for k, (t, _) in output_fields.items()},
            **{k: v for k, (_, v) in output_fields.items()},
        },
    )
    plan_cls = build_plan(output_cls)
    input_cls = type(
        "Input",
        (BaseModel,),
        {"title": Field(description="the book's title"), "__annotations__": {"title": str}},
    )
    sig_cls = type("Signature", (), {"__doc__": "Look up a book.", "Input": input_cls, "Output": output_cls})
    module = types.ModuleType("book_lookup")
    module.Signature, module.PlanGenerator = sig_cls, plan_cls
    return planning_commands.command_from_module(
        "Library/book_lookup", module, "_commands/Library/book_lookup.py"
    ).command


def _run_planning_command(env, planning_command, args: dict | None = None):
    env.args = dict(args or {"title": "Dune"})
    env.planning_command = planning_command
    return env.ctx.process_turn("look up the book")


@pytest.fixture(autouse=True)
def _clear_missing_output_warnings():
    reset_missing_declared_output_warnings()
    yield
    reset_missing_declared_output_warnings()


def test_result_str_gives_text_and_no_output(library_env):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return f"matches {len(find.book_uids)}"

    planning_command = _build_planning_command(Plan)
    result = _run_planning_command(library_env, planning_command)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer.startswith("matches ")
    outputs = [o for o in library_env.ctx._turn_outputs if o.command_name == planning_command.name]
    assert len(outputs) == 1
    assert outputs[0].command_response.output is None


def test_missing_output_warning_once_for_planning_command(library_env, caplog):
    def build(output_cls):
        class Plan:
            def find(self, input) -> FoundBooks:
                return [run("find_book", query=input.title)]

            def __result__(self, input, find):
                return f"matches {len(find.book_uids)}"

        return Plan

    planning_command = _planning_command_with_output(build)
    with caplog.at_level(logging.WARNING):
        _run_planning_command(library_env, planning_command)
        _run_planning_command(library_env, planning_command)
    msgs = [r.message for r in caplog.records if "declares an Output but returned none" in r.message]
    assert len(msgs) == 1
    assert planning_command.name in msgs[0]


def test_result_output_json_is_the_response(library_env):
    def build(output_cls):
        class Plan2:
            def find(self, input) -> FoundBooks:
                return [run("find_book", query=input.title)]

            def book(self, input, find) -> OpenedBook:
                return [pick("open_book", named=input.title)]

            def __result__(self, input, find, book):
                return output_cls(found=True, title=book.title)

        return Plan2

    planning_command = _planning_command_with_output(build, output_fields={
        "found": (bool, Field(description="whether a book matched")),
        "title": (str, Field(default="", description="the title")),
    })
    result = _run_planning_command(library_env, planning_command)

    assert result.status == TurnStatus.COMPLETED
    out = next(o for o in library_env.ctx._turn_outputs if o.command_name == planning_command.name)
    assert out.command_response.output.found is True
    assert out.command_response.output.title == "Dune"
    assert '"found":true' in out.command_response.response.replace(" ", "")


def test_with_result_no_phase_agent_writes_the_answer(library_env):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def answer(self, input, find):
            return ["say how many books match"]

        def __result__(self, input, find):
            return f"count {len(find.book_uids)}"

    planning_command = _build_planning_command(Plan)
    _run_planning_command(library_env, planning_command)

    assert library_env.model.asked == []


def test_result_raises_falls_back_to_one_model_call(library_env, caplog):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            raise RuntimeError("boom")

    planning_command = _build_planning_command(Plan)
    with caplog.at_level(logging.WARNING):
        result = _run_planning_command(library_env, planning_command)

    assert result.answer == "written"
    assert library_env.model.asked == []
    assert sum(1 for r in caplog.records if "__result__ failed" in r.message) == 1
    out = next(o for o in library_env.ctx._turn_outputs if o.command_name == planning_command.name)
    assert out.command_response.output is None


def test_skipped_phase_is_none_in_result(library_env):
    def build(output_cls):
        class Plan2:
            def find(self, input) -> FoundBooks:
                return [run("find_book", query=input.title)]

            def __result__(self, input, find):
                return output_cls(found=bool(find.book_uids))

        return Plan2

    planning_command = _planning_command_with_output(build)
    result = _run_planning_command(library_env, planning_command, {"title": "NoSuchTitleAtAll"})

    assert result.status == TurnStatus.COMPLETED
    out = next(o for o in library_env.ctx._turn_outputs if o.command_name == planning_command.name)
    assert out.command_response.output.found is False
    assert out.success is True


def test_output_without_result_fails_load():
    def build(_output_cls):
        class Plan:
            def find(self, input):
                return [run("find_book", query=input.title)]

        return Plan

    with pytest.raises(PlanningCommandFormatError, match="Signature.Output is declared, but nothing builds it"):
        _planning_command_with_output(build)


def test_result_parameter_not_a_phase_fails_load():
    class Plan:
        def find(self, input):
            return [run("find_book", query=input.title)]

        def __result__(self, input, found):
            return "x"

    with pytest.raises(PlanningCommandFormatError, match="__result__ takes 'found', which is not a phase"):
        _build_planning_command(Plan)


def test_plan_generator_subclassing_another_commands_fails_load(tmp_path):
    base_path = tmp_path / "_commands" / "Library" / "base_cmd.py"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("# base", encoding="utf-8")
    base_module = types.ModuleType("library_base_cmd")
    base_module.__file__ = str(base_path)
    base_sig = type("Signature", (), {"__doc__": "base", "Input": type("Input", (BaseModel,), {})})
    base_plan = type("PlanGenerator", (), {"find": lambda self, input: ["find_book query=x"]})
    base_module.Signature = base_sig
    base_module.PlanGenerator = base_plan
    sys.modules[base_module.__name__] = base_module
    base_plan.__module__ = base_module.__name__

    class Child(base_plan):
        def __result__(self, input):
            return "ok"

    child_path = tmp_path / "_commands" / "Library" / "child.py"
    child_path.write_text("# child", encoding="utf-8")
    child_module = types.ModuleType("library_child")
    child_module.__file__ = str(child_path)
    child_module.Signature = type("Signature", (), {"__doc__": "child", "Input": base_sig.Input})
    child_module.PlanGenerator = Child
    Child.__module__ = child_module.__name__

    with pytest.raises(PlanningCommandFormatError, match="inherits from Library/base_cmd's PlanGenerator"):
        planning_commands.command_from_module("Library/child", child_module, str(child_path))


def test_subclassing_a_base_in_underscore_module_loads(tmp_path):
    helper_path = tmp_path / "_commands" / "Library" / "_shelf.py"
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text("# helper", encoding="utf-8")
    helper_module = types.ModuleType("library_shelf")
    helper_module.__file__ = str(helper_path)

    class BookPhases:
        def find(self, input):
            return ["find_book query=x"]

        def __result__(self, input, find):
            return "ok"

    helper_module.PlanGenerator = BookPhases
    sys.modules[helper_module.__name__] = helper_module
    BookPhases.__module__ = helper_module.__name__

    class Child(BookPhases):
        pass

    child_path = tmp_path / "_commands" / "Library" / "child2.py"
    child_path.write_text("# child", encoding="utf-8")
    child_module = types.ModuleType("library_child2")
    child_module.__file__ = str(child_path)
    child_module.Signature = type("Signature", (), {
        "__doc__": "child",
        "Input": type("Input", (BaseModel,), {"__annotations__": {}}),
    })
    child_module.PlanGenerator = Child
    Child.__module__ = child_module.__name__

    parsed = planning_commands.command_from_module("Library/child2", child_module, str(child_path))
    assert parsed.command.uses_result is True


def test_inherited_result_is_checked(tmp_path):
    helper_path = tmp_path / "_commands" / "Library" / "_facts.py"
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text("# facts", encoding="utf-8")
    helper_module = types.ModuleType("library_facts")
    helper_module.__file__ = str(helper_path)

    class Facts:
        def find(self, input):
            return ["find_book query=x"]

    helper_module.PlanGenerator = Facts
    sys.modules[helper_module.__name__] = helper_module
    Facts.__module__ = helper_module.__name__

    class Child(Facts):
        def __result__(self, input, missing):
            return "x"

    child_path = tmp_path / "_commands" / "Library" / "child3.py"
    child_path.write_text("# child", encoding="utf-8")
    child_module = types.ModuleType("library_child3")
    child_module.__file__ = str(child_path)
    child_module.Signature = type("Signature", (), {
        "__doc__": "child",
        "Input": type("Input", (BaseModel,), {"__annotations__": {}}),
    })
    child_module.PlanGenerator = Child
    Child.__module__ = child_module.__name__

    with pytest.raises(PlanningCommandFormatError, match="takes 'missing', which is not a phase"):
        planning_commands.command_from_module("Library/child3", child_module, str(child_path))


_MATCH_RESULT_BASE = '''\
"""Shared find phase and __result__ for title match planning commands."""
from pydantic import BaseModel

from fastworkflow.planning import fact_from, run


class FoundBooks(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids")


class PlanGenerator:
    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title)]

    def __result__(self, input, find):
        return f"matches {len(find.book_uids)}"
'''

_TITLE_MATCHES = '''\
"""How many books match a title (inherits __result__ from _match_result)."""
from pydantic import BaseModel, Field

from ._match_result import PlanGenerator as MatchResultBase


class Signature:
    """Say how many books match the given title."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")

    plain_utterances = ["how many books match that title"]


class PlanGenerator(MatchResultBase):
    pass
'''


def test_inherited_result_from_underscore_module_runs_end_to_end(
    tmp_path, initialized_fastworkflow, monkeypatch,
):
    env = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        args={"title": "Dune"},
    )
    folder = env.wf.folderpath
    base_path = os.path.join(folder, "_commands", "Library", "_match_result.py")
    child_path = os.path.join(folder, "_commands", "Library", "title_matches.py")
    with open(base_path, "w", encoding="utf-8") as handle:
        handle.write(_MATCH_RESULT_BASE)
    with open(child_path, "w", encoding="utf-8") as handle:
        handle.write(_TITLE_MATCHES)

    python_utils.get_module(base_path, folder)
    sf = planning_commands.command_from_module(
        "Library/title_matches", python_utils.get_module(child_path, folder), child_path,
    )
    assert sf.command.uses_result is True

    try:
        env.planning_command = sf.command
        result = env.ctx.process_turn("how many books match Dune")

        assert result.status == TurnStatus.COMPLETED
        assert result.answer.startswith("matches ")
        assert env.runner.sent == ["find_book query=Dune"]
        assert env.model.asked == []
        out = next(o for o in env.ctx._turn_outputs if o.command_name == sf.command.name)
        assert out.command_response.response.startswith("matches ")
        assert out.command_response.output is None
        assert out.success is True
    finally:
        env.ctx.pop_active_workflow()


_TEXT_ONLY_OUTPUT_CMD = '''import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow import CommandOutput, CommandResponse


class Signature:
    """Execution command that declares Output but returns text only."""

    class Input(BaseModel):
        query: str = Field(default="x", description="query")

    class Output(BaseModel):
        value: str = Field(description="a value")


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return CommandResponse(response="text only")
'''


def test_text_only_return_from_execution_command_is_a_call_error(tmp_path, caplog):
    # fix-3rsq.declared-output-required: an execution command that declares an Output must return it.
    root = Path(__file__).resolve().parent.parent
    env_vars = dict(dotenv_values(root / "fastworkflow" / "examples" / "fastworkflow.env"))
    fastworkflow.init(env_vars=env_vars)
    folder = _build_workflow(
        tmp_path,
        "wf_text_only",
        extra_commands={"text_only_output": _TEXT_ONLY_OUTPUT_CMD},
    )
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    workflow = fastworkflow.Workflow.create(
        folder, workflow_id_str=f"text-only-{uuid.uuid4().hex}"
    )
    action = fastworkflow.Action(
        command_name="text_only_output",
        command="run text only",
        parameters={"query": "a"},
    )
    try:
        for _ in range(2):
            with pytest.raises(TypeError) as raised:
                CommandExecutor.perform_action(workflow, action)
            assert str(raised.value).startswith(
                "text_only_output declares an Output but returned CommandResponse without output; return the Output, "
                "or CommandResponse(response=..., output=<Output>)")
        assert not [r for r in caplog.records if "declares an Output but returned none" in r.message]
    finally:
        fastworkflow.RoutingRegistry.clear_registry()


def test_retail_find_user_by_email_does_not_emit_missing_output_warning(
    retail_workflow_path, caplog, tmp_path,
):
    """Retail lookup commands that populate ``CommandResponse.output`` must not warn."""
    root = Path(__file__).resolve().parent.parent
    env_vars = {
        **dotenv_values(root / "fastworkflow" / "examples" / "fastworkflow.env"),
        "FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts"),
    }
    fastworkflow.init(env_vars=env_vars)
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(retail_workflow_path, load_cached=False)
    workflow = fastworkflow.Workflow.create(
        retail_workflow_path,
        workflow_id_str=f"retail-output-{uuid.uuid4().hex}",
    )
    action = fastworkflow.Action(
        command_name="find_user_id_by_email",
        command="Find user id for john.doe@example.com",
        parameters={"email": "john.doe@example.com"},
    )
    try:
        with caplog.at_level(logging.WARNING):
            result = CommandExecutor.perform_action(workflow, action)
        assert result.command_response.output is not None
        assert not any(
            "declares an Output but returned none" in r.message for r in caplog.records
        )
    finally:
        fastworkflow.RoutingRegistry.clear_registry()


def test_phase_annotated_find_book_output_hands_on_run(library_env):
    class Plan:
        def find(self, input) -> FindBook.Output:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return f"total {find.total}"

    library_env.planning_command = _build_planning_command(Plan)
    result = _run_planning_command(library_env, library_env.planning_command)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == "total 2"
    spans = library_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    assert spans[0].attributes["facts"] == {
        "book_uids": ["b1", "b2"],
        "titles": ["Dune", "Dune Messiah"],
        "total": 2,
    }


def test_full_list_step_joins_loan_pages(library_proof_env):
    class Plan:
        def loans(self) -> ListLoans.Output:
            return [full_list("list_loans")]

    library_proof_env.runner(library_proof_env.ctx, "open_book book_uid=b1")
    library_proof_env.planning_command = _build_planning_command(Plan)
    library_proof_env.args = {}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn("list every loan")

    assert result.status == TurnStatus.COMPLETED
    spans = library_proof_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    facts = spans[0].attributes["facts"]
    assert facts["total"] == 2
    assert len(facts["borrowers"]) == 2


def test_fan_out_phase_open_book_output_per_item(library_env):
    class Plan:
        def find(self, input) -> FindBook.Output:
            return [run("find_book", query=input.title)]

        @phase(over="find.book_uids")
        def opened(self, uid) -> OpenBook.Output:
            return [run("open_book", book_uid=uid)]

        def __result__(self, input, find, opened):
            titles = [item.title for item in opened]
            return ", ".join(titles)

    library_env.planning_command = _build_planning_command(Plan)
    result = _run_planning_command(library_env, library_env.planning_command)

    assert result.status == TurnStatus.COMPLETED
    spans = library_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    open_span = next(s for s in spans if s.attributes["phase"] == "opened")
    assert open_span.attributes["item_count"] == 2
    assert open_span.attributes["facts"]["title"] in ("Dune", "Dune Messiah")


def test_phase_output_whose_command_never_runs_is_load_error(tmp_path):
    bad = '''from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_book import Signature as FindBook


class Signature:
    """Bad typed phase."""

    class Input(BaseModel):
        query: str = Field(description="words")


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("show_book")]
'''
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Library", "bad_typed.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(bad)
    planning_commands.resolve_field_sources.cache_clear()
    res = planning_commands.resolve_field_sources(folder)
    assert any(
        "phase 'find' hands on find_book's Output, but none of its steps runs find_book"
        in e.message
        for e in res.errors
    )


def test_union_annotation_hands_on_whichever_member_ran(library_env):
    class Plan:
        def find(self, input) -> FindBook.Output | FindMember.Output:
            if input.kind == "book":
                return [run("find_book", query=input.name)]
            return [run("find_member", query=input.name)]

        def __result__(self, input, find):
            if input.kind == "book":
                return f"books {find.total}"
            return f"members {find.total}"

    input_cls = type(
        "Input",
        (BaseModel,),
        {
            "__annotations__": {"name": str, "kind": str},
            "name": Field(description="the name"),
            "kind": Field(default="book", description="book or member"),
        },
    )
    sig_cls = type("Signature", (), {"__doc__": "Find.", "Input": input_cls})
    module = types.ModuleType("find_union")
    module.Signature, module.PlanGenerator = sig_cls, Plan
    planning_command = planning_commands.command_from_module(
        "Library/find_union", module, "_commands/Library/find_union.py"
    ).command

    library_env.planning_command = planning_command
    result = _run_planning_command(library_env, planning_command, {"name": "Dune", "kind": "book"})
    assert result.answer == "books 2"

    library_env.runner.sent.clear()
    result = _run_planning_command(library_env, planning_command, {"name": "Raj Patel", "kind": "member"})
    assert "members" in result.answer


def test_typed_facts_model_fills_output_input_and_optional_none(library_env):
    class Plan:
        def work(self, input) -> Mixed:
            return [
                run("find_book", query=input.title),
                run("open_book", book_uid="b1"),
            ]

        def __result__(self, input, work):
            return f"{work.found.total}:{work.open_args.book_uid}"

    library_env.planning_command = _build_planning_command(Plan)
    result = _run_planning_command(library_env, library_env.planning_command)

    assert result.status == TurnStatus.COMPLETED
    spans = library_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    facts = spans[0].attributes["facts"]
    assert facts["found"]["total"] == 2
    assert facts["open_args"]["book_uid"] == "b1"
    assert facts["member"] is None
    assert result.answer == "2:b1"


def test_typed_facts_validator_refusal_fails_the_phase(library_env):
    class Plan:
        def find(self, input) -> Strict:
            return [run("find_book", query=input.title)]

    library_env.planning_command = _build_planning_command(Plan, args_fields={
        "title": (str, Field(description="the title")),
    })
    result = _run_planning_command(library_env, library_env.planning_command, {"title": "NoSuchTitleAtAllXYZ"})

    assert result.status == TurnStatus.COMPLETED
    assert "This job stopped at its part 'find', which failed:" in result.answer
    assert "no book matched" in result.answer


def test_list_str_facts_field_without_copy_is_load_error(tmp_path):
    bad = '''from pydantic import BaseModel

from fastworkflow.planning import run

from .show_book import Signature as ShowBook


class Shelf(BaseModel):
    borrowers: list[str]
    book: ShowBook.Output


class Signature:
    """Bad shelf facts."""


class PlanGenerator:
    def shelf(self) -> Shelf:
        return [run("show_book"), run("list_loans")]
'''
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Book", "bad_shelf.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(bad)
    mod = python_utils.get_module(path, folder)
    with pytest.raises(
        PlanningCommandFormatError,
        match=r"Shelf\.borrowers is list\[str\]; a facts field is typed with a command's Output or Input",
    ):
        planning_commands.command_from_module("Book/bad_shelf", mod, path)


def test_library_loan_report_result_uses_list_loans_output(library_proof_env):
    sf = _real_planning_command("Book/loan_report", "_commands/Book/loan_report.py")
    library_proof_env.runner(library_proof_env.ctx, "open_book book_uid=b1")
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn("who has this book")

    assert result.status == TurnStatus.COMPLETED
    spans = library_proof_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    assert spans[0].attributes["facts"]["borrowers"] == ["Ann Lee", "Raj Patel"]
    assert "Ann Lee" in result.answer
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.command_response.output is None
    assert "Raj Patel" in out.command_response.response


def test_library_book_by_title_result_builds_output_from_two_phases(library_proof_env):
    sf = _real_planning_command("Library/book_by_title", "_commands/Library/book_by_title.py")
    title, author, copies_count, book_uid = "Dune", "Frank Herbert", 3, "b1"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"title": title}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"who wrote {title}")

    assert result.status == TurnStatus.COMPLETED
    spans = library_proof_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    assert len(spans) >= 2
    assert [s.attributes["phase"] for s in spans[:2]] == ["find", "book"]
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert title in out.command_response.response
    assert author in out.command_response.response
    output = out.command_response.output
    assert isinstance(output, BookByTitle.Output)
    assert output.found is True
    assert output.title == title
    assert output.author == author
    assert output.copies == copies_count
    assert output.book_uid == book_uid


def test_library_book_by_title_skipped_phase_gives_none_and_found_false(library_proof_env):
    sf = _real_planning_command("Library/book_by_title", "_commands/Library/book_by_title.py")
    title = "Zzyzx Nonexistent Volume"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"title": title}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"look up {title}")

    assert result.status == TurnStatus.COMPLETED
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert out.command_response.output.found is False
    assert out.command_response.response == f"No book matches {title}."
    assert not any(c.startswith("open_book") for c in library_proof_env.runner.sent)


def test_library_author_books_from_real_module_runs_fan_out(library_proof_env):
    sf = _real_planning_command("Library/author_books", "_commands/Library/author_books.py")
    author = "Frank Herbert"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"author": author}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"what do you have by {author}")

    assert result.status == TurnStatus.COMPLETED
    assert "Dune" in result.answer
    assert "Dune Messiah" in result.answer
    assert "3 copies" in result.answer
    assert "1 copy" in result.answer
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert out.command_response.output is None
    assert author in out.command_response.response
    # 3.9: open_book is a Library command, on the path from the Book it opened, so it runs in
    # place (bound to the library) and no go_up is inserted before the second book.
    assert library_proof_env.runner.sent == [
        "find_book query=Frank Herbert",
        "open_book book_uid=b1",
        "open_book book_uid=b2",
    ]
    assert library_proof_env.wf.current_command_context.uid == "b2"
    spans = library_proof_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    assert [s.attributes["phase"] for s in spans[:3]] == ["find", "book", "book"]


def test_library_count_books_from_real_module_uses_find_total(library_proof_env):
    sf = _real_planning_command("Library/count_books", "_commands/Library/count_books.py")
    query = "Jane Austen"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"query": query}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"how many books by {query} do you have")

    assert result.status == TurnStatus.COMPLETED
    expected = f"{query} matches 2 books."
    assert expected in result.answer
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert out.command_response.output is None
    assert out.command_response.response == expected
    assert library_proof_env.runner.sent == [f"find_book query={query}"]


def test_library_member_report_from_real_module_lists_loans_and_notes(library_proof_env):
    sf = _real_planning_command("Library/member_report", "_commands/Library/member_report.py")
    member = "Raj Patel"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"member": member, "loans": True, "notes": True}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(
        f"what has {member} borrowed and any notes on {member}'s record"
    )

    assert result.status == TurnStatus.COMPLETED
    assert "Raj Patel has 2 books (Dune, Emma) on loan." in result.answer
    assert "Raj Patel's record holds 1 note (prefers email reminders)." in result.answer
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert out.command_response.output is None
    assert "Dune" in out.command_response.response
    assert "prefers email reminders" in out.command_response.response
    assert any(c.startswith("find_member") for c in library_proof_env.runner.sent)
    assert "list_member_loans" in library_proof_env.runner.sent
    assert "show_notes" in library_proof_env.runner.sent


def test_library_note_member_from_real_module_writes_note_on_member(library_proof_env):
    sf = _real_planning_command("Library/note_member", "_commands/Library/note_member.py")
    member, text = "Ann Lee", "card renewed"
    lib = library_proof_env.wf.root_command_context
    before = list(lib.members["m1"].notes)
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"member": member, "text": text}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"add a note to {member}: {text}")

    assert result.status == TurnStatus.COMPLETED
    assert lib.members["m1"].notes == before + [text]
    assert library_proof_env.runner.sent == [
        f"find_member query={member}",
        "open_member member_uid=m1",
        f"add_note text={text}",
    ]
    add_out = next(
        o for o in library_proof_env.ctx._turn_outputs if o.command_name == "Record/add_note"
    )
    assert add_out.success is True
    assert add_out.command_response.output.total == len(before) + 1
    assert library_proof_env.model.asked == []


def test_library_find_record_book_kind_from_real_module(library_proof_env):
    sf = _real_planning_command("Library/find_record", "_commands/Library/find_record.py")
    name = "Emma"
    library_proof_env.planning_command = sf.command
    library_proof_env.args = {"name": name, "kind": "book"}
    library_proof_env.runner.sent.clear()
    result = library_proof_env.ctx.process_turn(f"do you have a book called {name}")

    assert result.status == TurnStatus.COMPLETED
    expected = f"{name} matches 1 book (Emma)."
    assert expected in result.answer
    out = next(o for o in library_proof_env.ctx._turn_outputs if o.command_name == sf.command.name)
    assert out.success is True
    assert out.command_response.output is None
    assert out.command_response.response == expected
    assert library_proof_env.runner.sent == [f"find_book query={name}"]
    assert not any(c.startswith("find_member") for c in library_proof_env.runner.sent)
    spans = library_proof_env.sink.named(fastworkflow.tracing.SPAN_PLANNING_COMMAND_PHASE)
    assert "titles" in spans[0].attributes["facts"]


def test_library_find_record_union_return_loads():
    lib_root = str(
        Path(__file__).resolve().parent.parent.joinpath("fastworkflow", "examples", "library")
    )
    path = os.path.join(lib_root, "_commands", "Library", "find_record.py")
    mod = python_utils.get_module(path, lib_root)
    sf = planning_commands.command_from_module("Library/find_record", mod, path)
    assert sf.command.phase_outputs
    assert sf.command.plan_fn is not None


IN_REQUEST_TITLE = '''from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Look up a book by title."""

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input):
        return [run("find_book", query=input.title)]
'''


def test_in_request_on_a_str_field_loads(tmp_path):
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Library", "by_title.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(IN_REQUEST_TITLE)
    mod = python_utils.get_module(path, folder)
    sf = planning_commands.command_from_module("Library/by_title", mod, path)
    assert sf.command.param("title").in_request


def test_in_request_on_a_bool_field_is_a_load_error(tmp_path):
    bad = '''from pydantic import BaseModel, Field

from fastworkflow.planning import run


class Signature:
    """Bad flag."""

    class Input(BaseModel):
        flag: bool = Field(default=False, description="whether",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def find(self, input):
        return [run("find_book")]
'''
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Library", "bad_in_request.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(bad)
    mod = python_utils.get_module(path, folder)
    with pytest.raises(PlanningCommandFormatError, match=r"option in_request does not apply to field '\w+' of type \w+"):
        planning_commands.command_from_module("Library/bad_in_request", mod, path)


def test_in_request_str_must_be_written_in_the_request(tmp_path):
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Library", "by_title.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(IN_REQUEST_TITLE)
    mod = python_utils.get_module(path, folder)
    command = planning_commands.command_from_module("Library/by_title", mod, path).command
    request = "who wrote Dune"
    args, _ = planning_commands.check_arguments(command, {"title": "Dune"}, request)
    assert args["title"] == "Dune"
    with pytest.raises(planning_commands.PlanningCommandInputError, match="title='Emma' is not written in the request"):
        planning_commands.check_arguments(command, {"title": "Emma"}, request)


def test_in_request_str_rule_sentence_is_for_the_extractor_not_the_field_description(tmp_path):
    folder = _library_workflow(tmp_path)
    path = os.path.join(folder, "_commands", "Library", "by_title.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(IN_REQUEST_TITLE)
    mod = python_utils.get_module(path, folder)
    command = planning_commands.command_from_module("Library/by_title", mod, path).command
    # The field's own description says what the value is; the rule sentence is the extractor's (3.4).
    desc = command.input_model.model_fields["title"].description
    assert desc == "the book's title"
    assert planning_commands.IN_REQUEST_STR_SENTENCE not in desc
    assert planning_commands.extraction_sentences(command.param("title")) == (
        planning_commands.IN_REQUEST_STR_SENTENCE,)


PLANNING_CMD = "Library/count_books"
ASSISTANT_PLANNING_REFUSAL = (
    f"{PLANNING_CMD} is a planning command; it runs in agent mode."
)


@pytest.fixture
def planning_dispatch_library(tmp_path, initialized_fastworkflow):
    """Library copy with ``Library/count_books`` and ``Library/note_member``; real ``CommandExecutor`` dispatch."""
    from fastworkflow.command_metadata_api import CommandMetadataAPI
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.mcp_server import FastWorkflowMCPServer
    from fastworkflow.workflow_agent import _what_can_i_do
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    folder = _library_workflow(
        tmp_path, extra_commands=("_commands/Library/count_books.py",
                                  "_commands/Library/note_member.py")
    )
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    wf = fastworkflow.Workflow.create(
        folder, workflow_id_str=f"pd_{uuid.uuid4().hex[:8]}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(wf)
    ctx.push_active_workflow(wf)
    LibraryRunner(wf)(ctx, "startup")
    mcp = FastWorkflowMCPServer(ctx)
    state = SimpleNamespace(
        ctx=ctx,
        wf=wf,
        folder=folder,
        mcp=mcp,
        api=CommandMetadataAPI,
        what_can_i_do=_what_can_i_do,
    )
    yield state
    ctx.pop_active_workflow()


def test_assistant_mode_refuses_planning_command(planning_dispatch_library):
    # Changed (fix-3rsq.10.8): count_books is mechanical, so assistant mode runs it (covered in
    # test_assistant_mode_planning). A needs-agent command, note_member (a prose step and no
    # __result__), is still refused, with its reasons.
    ctx = planning_dispatch_library.ctx
    ctx._cme_workflow.context["is_assistant_mode_command"] = True
    out = CommandExecutor.invoke_command(ctx, "note_member Lee note=hello")
    needs, reasons = planning_command_needs_agent(planning_dispatch_library.folder, "Library/note_member")
    assert needs is True
    assert out.success is False
    assert out.command_response.response == refusal_message("Library/note_member", reasons)


def test_slash_prefixed_refuses_planning_command(planning_dispatch_library):
    # Changed (fix-3rsq.10.8): the slash form of a needs-agent command, note_member (count_books runs).
    ctx = planning_dispatch_library.ctx
    ctx._prepare_message_routing("/note_member Lee note=hello")
    out = CommandExecutor.invoke_command(ctx, "/note_member Lee note=hello")
    needs, reasons = planning_command_needs_agent(planning_dispatch_library.folder, "Library/note_member")
    assert needs is True
    assert out.success is False
    assert out.command_response.response == refusal_message("Library/note_member", reasons)


def test_perform_action_refuses_planning_command(planning_dispatch_library):
    wf = planning_dispatch_library.wf
    out = CommandExecutor.perform_action(
        wf,
        fastworkflow.Action(
            command_name=PLANNING_CMD,
            command="count books",
            parameters={},
        ),
    )
    assert out.success is False
    assert out.command_response.response == ASSISTANT_PLANNING_REFUSAL


def test_perform_mcp_tool_call_refuses_planning_command(planning_dispatch_library):
    wf = planning_dispatch_library.wf
    result = CommandExecutor.perform_mcp_tool_call(
        wf,
        fastworkflow.MCPToolCall(name="count_books", arguments={"command": "how many"}),
    )
    assert result.isError is True
    assert ASSISTANT_PLANNING_REFUSAL in result.content[0].text


def test_agent_step_runs_planning_command(planning_dispatch_library, monkeypatch):
    # Was a refusal ("it cannot be called as a step yet") before run_planning_command: an
    # agent step now extracts the arguments and runs the planning command as a nested frame.
    from fastworkflow.workflow_agent import _execute_workflow_query

    ctx = planning_dispatch_library.ctx
    script_command_extraction(monkeypatch, planning_dispatch_library.wf)
    model = Model(None)
    script_every_agent(monkeypatch, model)
    text = _execute_workflow_query("count_books query=Jane Austen", ctx)
    assert "Jane Austen matches 2 books." in text
    assert ASSISTANT_PLANNING_REFUSAL not in text
    out = next(o for o in ctx._turn_outputs if o.command_name == PLANNING_CMD)
    assert out.success is True
    assert ctx._frame_stack()[1:] == []


def test_planning_command_in_executor_and_planner_catalogs_of_its_context(
    planning_dispatch_library,
):
    ctx = planning_dispatch_library.ctx
    wf = planning_dispatch_library.wf
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    executor_text = planning_dispatch_library.what_can_i_do(ctx)
    assert "- count_books\n" in executor_text
    assert "find_book" in executor_text

    planner_text = planning_dispatch_library.api.get_all_contexts_command_display_text(
        subject_workflow_path=wf.folderpath,
        cme_workflow_path=cme_path,
        active_context_name=wf.current_command_context_name,
    )
    assert "- count_books\n" in planner_text
    assert "find_book" in planner_text


def test_planning_command_absent_from_mcp_list_tools(planning_dispatch_library):
    tools = planning_dispatch_library.mcp.list_tools()
    tool_names = {tool["name"] for tool in tools}
    assert "count_books" not in tool_names
    assert "find_book" in tool_names


# ----------------------------------------------------------------------
# Phase 5 step 3: planning commands in the executor's and planner's catalogs
# (design 3.4 "Catalogs"). The library sample's planning commands, plus a
# test-only planning command in the Record base that Book and Member inherit.
# ----------------------------------------------------------------------

_LIBRARY_PLANNING_FILES = (
    "_commands/Library/book_by_title.py",
    "_commands/Library/book_loans.py",
    "_commands/Library/_wording.py",
    "_commands/Book/loan_report.py",
)

_RECORD_PLANNING_SOURCE = '''"""record_notes: the notes on the open record, as one planned step."""
from fastworkflow.planning import run


class Signature:
    """Show the notes on the open record."""


class PlanGenerator:
    def notes(self):
        return [run("show_notes")]
'''

_BOOK_LOANS_EXECUTOR_ENTRY = (
    "- book_loans\n"
    "  Say who has a book on loan, given its title.\n"
    "  inputs:\n"
    "  - the book's title, type: str\n"
)

_BOOK_LOANS_PLANNER_ENTRY = (
    f"{_BOOK_LOANS_EXECUTOR_ENTRY}"
    "  template:\n"
    "  - book_by_title title=<title>\n"
    "  - list_loans (full list) (only when book.found)\n"
)


@pytest.fixture
def planning_catalog_library(tmp_path, initialized_fastworkflow):
    """Library copy with Library/book_by_title, Library/book_loans, Book/loan_report and
    Record/record_notes; the startup command leaves the workflow in Library."""
    folder = _library_workflow(tmp_path, extra_commands=_LIBRARY_PLANNING_FILES)
    Path(folder, "_commands", "Record", "record_notes.py").write_text(
        _RECORD_PLANNING_SOURCE, encoding="utf-8"
    )
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"pc_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(wf)
    ctx.push_active_workflow(wf)
    LibraryRunner(wf)(ctx, "startup")
    state = SimpleNamespace(
        ctx=ctx,
        wf=wf,
        folder=folder,
        mcp=FastWorkflowMCPServer(ctx),
        cme_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
    )
    yield state
    ctx.pop_active_workflow()


def test_library_planning_commands_listed_in_executor_catalog_with_signature(
    planning_catalog_library,
):
    text = _what_can_i_do(planning_catalog_library.ctx)

    assert _BOOK_LOANS_EXECUTOR_ENTRY in text
    assert (
        "- book_by_title\n"
        "  Show who wrote a book, and how many copies the library owns, given its title.\n"
        "  inputs:\n"
        "  - the book's title, type: str\n"
        "  outputs:\n"
    ) in text
    assert "template:" not in text


def test_planning_commands_of_other_contexts_not_in_executor_catalog(planning_catalog_library):
    folder = planning_catalog_library.folder
    cme_path = planning_catalog_library.cme_path

    book = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=folder, cme_workflow_path=cme_path, active_context_name="Book")
    assert "- loan_report\n" in book
    assert "- record_notes\n" in book
    assert "book_loans" not in book
    assert "book_by_title" not in book

    member = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=folder, cme_workflow_path=cme_path, active_context_name="Member")
    assert "- record_notes\n" in member
    assert "loan_report" not in member
    assert "book_loans" not in member


def test_library_planning_command_in_planner_catalog_of_its_context_with_template(
    planning_catalog_library,
):
    wf = planning_catalog_library.wf
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=wf.folderpath,
        cme_workflow_path=planning_catalog_library.cme_path,
        active_context_name=wf.current_command_context_name,
        navigation_workflow=wf,
    )

    # The global scope is on every lookup path: its commands are callable here (3.9, Catalogs).
    deferred_start = text.index("Commands available here from the global scope")
    assert text.index(_BOOK_LOANS_PLANNER_ENTRY) < deferred_start
    assert "- library_hours\n" in text[deferred_start:]


def test_deferred_planning_commands_in_planner_catalog_show_their_template(
    planning_catalog_library,
):
    wf = planning_catalog_library.wf
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=wf.folderpath,
        cme_workflow_path=planning_catalog_library.cme_path,
        active_context_name=wf.current_command_context_name,
        navigation_workflow=wf,
    )

    book_section = text[text.index("Commands available after entering the Book context"):]
    assert (
        "- loan_report\n"
        "  Say who has this book on loan and when each loan is due back.\n"
    ) in book_section
    assert (
        "  template:\n"
        "  - list_loans (full list)\n"
        "  Before executing loan_report, navigate to Book as follows: "
        "open_book <book_uid> (needs book_uid)\n"
    ) in book_section
    assert (
        "- record_notes\n"
        "  Show the notes on the open record.\n"
        "  template:\n"
        "  - show_notes\n"
    ) in book_section


def test_library_planning_commands_absent_from_mcp_list_tools_and_capability_answer(
    planning_catalog_library,
):
    names = {tool["name"] for tool in planning_catalog_library.mcp.list_tools()}
    assert "find_book" in names
    assert "book_loans" not in names
    assert "book_by_title" not in names

    listed = context_commands(planning_catalog_library.wf, planning_catalog_library.cme_path)
    assert "find_book" in listed
    assert "book_loans" not in listed
    assert "book_by_title" not in listed


def test_routing_contexts_keep_execution_commands_only(planning_catalog_library):
    crd = RoutingRegistry.get_definition(planning_catalog_library.folder)
    assert crd.get_planning_command_names("Library") == [
        "Library/book_by_title",
        "Library/book_loans",
    ]
    assert crd.get_planning_command_names("Book") == ["Book/loan_report", "Record/record_notes"]
    assert crd.get_planning_command_names("Member") == ["Record/record_notes"]
    assert "Library/book_loans" not in crd.contexts["Library"]


def test_library_planning_commands_are_trained_labels_of_their_head(planning_catalog_library):
    """Phase 10 step 1 (production): a planning command labels the heads on its lookup path."""
    from fastworkflow.model_pipeline_training import (
        PLANNING_HEADS_NOT_HERE,
        experiment_planning_heads,
        experiment_reserved_class_arm,
    )

    crd = RoutingRegistry.get_definition(planning_catalog_library.folder, load_cached=False)
    labels = trained_command_labels("Library", crd)
    assert "Library/find_book" in labels
    assert "Library/book_loans" in labels
    assert "Library/book_by_title" in labels
    # the baseline arm keeps the pre-step-1 rule: planning commands are not head labels
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        baseline = trained_command_labels("Library", crd)
    assert "Library/find_book" in baseline
    assert "Library/book_loans" not in baseline
    assert "Library/book_by_title" not in baseline


# ----------------------------------------------------------------------
# Phase 3 step 7: training excludes planning commands (design 3.4 Training)
# ----------------------------------------------------------------------

_PLANNING_COMMAND_FILES = (
    "_commands/Book/loan_report.py",
    "_commands/Library/count_books.py",
)


@pytest.fixture
def library_training_workflow(tmp_path, initialized_fastworkflow):
    """Library copy with planning commands that declare ``plain_utterances``."""
    from fastworkflow.command_routing import RoutingRegistry

    folder = _library_workflow(tmp_path, extra_commands=_PLANNING_COMMAND_FILES)
    RoutingRegistry.clear_registry()
    yield folder
    RoutingRegistry.clear_registry()


def test_training_seed_collection_excludes_planning_commands(library_training_workflow):
    from fastworkflow.train.duplicate_detection import utterances_from_workflow

    seeds = utterances_from_workflow(library_training_workflow)
    assert "Library/find_book" in seeds
    assert "Book/loan_report" not in seeds
    assert "Library/count_books" not in seeds


def test_trained_command_labels_hold_planning_commands_on_their_lookup_path(library_training_workflow):
    """Phase 10 step 1 (production): Library takes its planning command, never the Book one."""
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.model_pipeline_training import (
        PLANNING_HEADS_NOT_HERE,
        experiment_planning_heads,
        experiment_reserved_class_arm,
        trained_command_labels,
    )

    crd = RoutingRegistry.get_definition(library_training_workflow, load_cached=False)
    library_labels = trained_command_labels("Library", crd)
    assert "Library/find_book" in library_labels
    assert "Library/count_books" in library_labels
    assert "Book/loan_report" not in library_labels  # Book's scope is not on Library's lookup path
    # the baseline arm keeps the pre-step-1 rule
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        baseline = trained_command_labels("Library", crd)
    assert "Library/find_book" in baseline
    assert "Library/count_books" not in baseline
    assert "Book/loan_report" not in baseline


def test_duplicate_capability_contexts_exclude_planning_commands(library_training_workflow):
    from fastworkflow.train.duplicate_detection import contexts_from_workflow

    contexts = contexts_from_workflow(library_training_workflow)
    all_commands = {cmd for cmds in contexts.values() for cmd in cmds}
    assert "Library/find_book" in all_commands
    assert "Book/loan_report" not in all_commands
    assert "Library/count_books" not in all_commands


def test_selective_training_fingerprints_exclude_planning_commands(library_training_workflow):
    from fastworkflow.train import selective_training

    fingerprints = selective_training.compute_command_fingerprints(library_training_workflow)
    assert "Library/find_book" in fingerprints
    assert "Book/loan_report" not in fingerprints
    assert "Library/count_books" not in fingerprints


def test_param_example_training_list_excludes_planning_commands(library_training_workflow):
    from fastworkflow.command_directory import CommandDirectory
    from fastworkflow.model_pipeline_training import get_route_layer_filepath_model
    from fastworkflow.train.__main__ import _get_commands_with_parameters

    cmd_dir = CommandDirectory.load(library_training_workflow)
    cmd_dir.save()
    json_path = get_route_layer_filepath_model(library_training_workflow, "command_directory.json")
    commands = _get_commands_with_parameters(json_path)
    assert "find_book" in commands
    assert "count_books" not in commands
    assert "loan_report" not in commands


# ------------------------------------------------------------------
# fix-3rsq.5.7: the call graph of planning commands and its cycles
# ------------------------------------------------------------------

_CYCLE_COPIES = (
    "_commands/Library/_wording.py",
    "_commands/Library/book_loans.py",
    "_commands/Library/book_by_title.py",
    "_commands/Library/count_books.py",
)
_BOOK_BY_TITLE_FIND = 'return [run("find_book", query=input.title)]'
_COUNT_BOOKS_FIND = 'return [run("find_book", query=input.query)]'


def _library_text_with_step(rel: str, old: str, new: str) -> str:
    """The library's *rel* text with one step line swapped for another, the swap asserted."""
    library = Path(fastworkflow.__file__).parent / "examples" / "library"
    text = (library / rel).read_text(encoding="utf-8")
    assert text.count(old) == 1, rel
    return text.replace(old, new)


def test_a_library_copy_whose_commands_call_each_other_names_every_command_on_the_cycle(tmp_path):
    """book_by_title calls book_loans, which calls book_by_title; count_books calls itself.
    Each planning command on a cycle is a load error that names the path; the other commands
    of the library copy load without one."""
    folder = _library_workflow(
        tmp_path,
        extra_commands=_CYCLE_COPIES,
        command_texts={
            "_commands/Library/book_by_title.py": _library_text_with_step(
                "_commands/Library/book_by_title.py", _BOOK_BY_TITLE_FIND,
                'return [run("find_book", query=input.title), '
                'run("book_loans", title=input.title)]'),
            "_commands/Library/count_books.py": _library_text_with_step(
                "_commands/Library/count_books.py", _COUNT_BOOKS_FIND,
                'return [run("find_book", query=input.query), '
                'run("count_books", query=input.query)]'),
        },
    )
    res = planning_commands.resolve_field_sources(folder)

    def cycle_messages(command: str) -> list[str]:
        return [p.message for p in res.errors
                if p.command == command and "cannot call itself" in p.message]

    assert cycle_messages("Library/book_by_title") == [
        "a planning command cannot call itself, even through others: "
        "Library/book_by_title -> Library/book_loans -> Library/book_by_title"]
    assert cycle_messages("Library/book_loans") == [
        "a planning command cannot call itself, even through others: "
        "Library/book_loans -> Library/book_by_title -> Library/book_loans"]
    assert cycle_messages("Library/count_books") == [
        "a planning command cannot call itself: Library/count_books -> Library/count_books"]
    assert {p.command for p in res.errors if "cannot call itself" in p.message} == {
        "Library/book_by_title", "Library/book_loans", "Library/count_books"}


def _refusal(workflow, command: str) -> str:
    """The refusal the agent tool gives a command name, as its message."""
    with pytest.raises(CommandNotFoundError) as refused:
        _explicit_agent_command(command, workflow)
    return str(refused.value)


def test_a_planning_command_is_available_only_where_it_is_runnable(planning_catalog_library):
    """Phase 5 step 1: a planning command follows its context as an execution command does. Startup
    leaves the workflow in Library, where loan_report (Book) and record_notes (Record, a base of
    Book and Member) are out of context. Opening a book makes it a Book, where both are in context.
    book_by_title (Library, the enclosing record of a Book) is on the lookup path from there, so it is
    called by its name, bound to the library (3.9, "Calls by exact name")."""
    wf = planning_catalog_library.wf
    assert home_contexts(wf, "loan_report") == ["Book"]
    assert home_contexts(wf, "record_notes") == ["Book", "Member"]
    assert "It is available in: 'Book'." in _refusal(wf, "loan_report")
    assert "It is available in: 'Book', 'Member'." in _refusal(wf, "record_notes")

    LibraryRunner(wf)(planning_catalog_library.ctx, "open_book book_uid=b1")
    assert wf.current_command_context_name == "Book"
    assert _explicit_agent_command("loan_report", wf) == "loan_report"
    assert _explicit_agent_command("record_notes", wf) == "record_notes"
    assert _explicit_agent_command("book_by_title", wf) == "book_by_title"


def test_an_out_of_context_planning_command_gets_the_execution_refusal_text(planning_catalog_library):
    """The refusal of a planning command out of its context is the refusal an execution command
    of another context gets: the same text with the name changed."""
    wf = planning_catalog_library.wf
    execution = _refusal(wf, "list_loans")
    planning = _refusal(wf, "loan_report")

    assert execution.startswith(
        "Command 'list_loans' is not available in the current context 'Library'. "
        "It is available in: 'Book'.")
    assert planning == execution.replace("'list_loans'", "'loan_report'")


def test_planning_command_parameter_fields_and_utterances_follow_the_planner_hint(
        planning_catalog_library):
    wf = planning_catalog_library.wf
    folder = planning_catalog_library.folder
    assert _parameter_fields("book_loans title=Dune", wf) == frozenset(BookLoans.Input.model_fields)
    # loan_report is out of context in Library: no fields are read for it there.
    assert _parameter_fields("loan_report", wf) is None

    # A planning command's utterances come from its Signature, as an execution command's do, and
    # they reach the words the planner's catalog text is matched on.
    utterances = planning_commands.command_utterances(folder)
    assert utterances["book_loans"] == ["who has borrowed Dune", "is anyone reading Emma right now", "who is reading Persuasion at the moment", "which patrons have Dune Messiah out", "is anyone holding a copy of Emma", "names of everyone who borrowed Persuasion", "has anybody got Dune out, and until when", "who currently holds the title Dune Messiah"]
    assert utterances["loan_report"] == ["who has this book", "when is it due back", "is this book out, and until when", "who is holding this one right now and what's the return date", "tell me who borrowed this book and when it comes back", "is this one on loan to anyone", "when will this copy be returned, and to whom", "who's got this book at the moment"]
    catalog = _what_can_i_do(planning_catalog_library.ctx)
    texts = planning_commands.command_texts(catalog, folder)
    desc_only = planning_commands.command_texts(catalog)
    # The hint is the union: the description's words (what it had before its utterances were read)
    # plus the words that recur in its texts. Utterances add words; they never take one away.
    assert desc_only["book_loans"] == "book given loan say title"
    assert texts["book_loans"] == ("anyone book borrowed dune emma given holding holds loan "
                                   "messiah out persuasion reading say title")
    assert set(desc_only["book_loans"].split()) <= set(texts["book_loans"].split())
    # book_by_title's two utterances both say "Dune", so that word joins its hint.
    assert set(desc_only["book_by_title"].split()) <= set(texts["book_by_title"].split())
    assert "dune" in texts["book_by_title"].split()
    assert "dune" not in desc_only["book_by_title"].split()


# ---------------------------------------------------------------------------
# The rule sentences and the planner's listing of a recipe's steps (design 3.4, 3.6)
# ---------------------------------------------------------------------------

ALL_RULE_SENTENCES = (
    planning_commands.IN_REQUEST_STR_SENTENCE,
    planning_commands.IN_REQUEST_INT_SENTENCE,
    planning_commands.SUBJECT_SENTENCE,
    planning_commands.VERIFY_BOOL_SENTENCE,
)
TUTORIAL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planning_tutorial_workflow")


def test_each_rule_sentence_reaches_only_the_extractor_of_the_field_whose_check_states_it(
        tmp_path, initialized_fastworkflow):
    folder = _with_rule_check(tmp_path)

    # The sentences by field: each option's own, on the fields whose checks state them.
    assert planning_commands.extraction_sentences_for(folder, RULE_CHECK) == {
        "title": (planning_commands.SUBJECT_SENTENCE,),
        "copies": (planning_commands.IN_REQUEST_INT_SENTENCE,),
        "loaned": (planning_commands.VERIFY_BOOL_SENTENCE,),
    }
    assert planning_commands.extraction_sentences_for(folder, "Library/book_by_title") == {
        "title": (planning_commands.IN_REQUEST_STR_SENTENCE,)}
    # An execution command's fields get none (until Phase 11).
    assert planning_commands.extraction_sentences_for(folder, "Library/find_book") == {}

    # The extractor's descriptions: the field's own, then its sentences (built as extract_parameters builds them).
    command = planning_commands.planning_command(folder, RULE_CHECK)
    signature = InputForParamExtraction.create_signature_from_pydantic_model(
        command.input_model, planning_commands.extraction_sentences_for(folder, RULE_CHECK))
    described = {name: field.json_schema_extra["desc"]
                 for name, field in signature.output_fields.items()}
    assert described["title"].startswith(
        "the book's title. Copy it exactly as the request writes it. "
        "It must name a record the request names. This field is Required.")
    assert described["copies"].startswith(
        "how many copies the request asks about. A number the request writes. This field is Required.")
    assert described["loaned"].startswith(
        "whether the request asks about loans. True only when the request asks for it. "
        "This field is Required.")
    # The field's own description, which the listings read, carries none of them.
    assert command.input_model.model_fields["title"].description == "the book's title"

    # The preview of the pinned tutorial workflow shows none of them.
    rendered = command_preview.render_previews(TUTORIAL_FOLDER)
    assert "the book's title" in "\n".join(map(str, rendered.values()))
    assert not any(s in str(text) for text in rendered.values() for s in ALL_RULE_SENTENCES)

    # The run's listings show none of them: the executor's (each command's inputs, in the context it
    # is in, here Library) and the planner's (the templates of every context).
    workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"rc_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    try:
        LibraryRunner(workflow)(ctx, "startup")
        executor_listing = _what_can_i_do(ctx)
    finally:
        ctx.pop_active_workflow()
    planner_listing = planner_commands_text(workflow)
    assert "- rule_check\n" in executor_listing
    assert "how many copies the request asks about, type: int" in executor_listing
    for listing in (executor_listing, planner_listing):
        assert not any(s in listing for s in ALL_RULE_SENTENCES)


HOURS_RECIPE = "hours_overview"
DUNE_RECIPE = "Library/dune_search"
#: The library's book_loans planning command and its helpers (the files of
#: ``test_nested_planning_commands.NESTED_COMMANDS``, repeated here because that module imports this one).
RECIPE_COMMANDS = (
    "_commands/Library/book_by_title.py",
    "_commands/Library/book_loans.py",
    "_commands/Library/_wording.py",
)


def _start(tmp_path, initialized, monkeypatch, *, extra_planning_commands=None):
    """A library copy with the real dispatch, the turn planner unscripted, and no planner model."""
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    return _make_library_runner_env(
        tmp_path,
        initialized,
        monkeypatch,
        extra_planning_commands=extra_planning_commands,
        extra_commands=RECIPE_COMMANDS,
        real_dispatch=True,
    )


#: The real planner_planning_command_for, captured before any test stubs it (_make_library_runner_env does).
REAL_PLANNER_PLANNING_COMMAND_FOR = workflow_agent.planner_planning_command_for


def use_the_real_planner_planning_command_for(monkeypatch) -> None:
    """Undo the stub the library harness puts on ``planner_planning_command_for`` for its whole test."""
    monkeypatch.setattr(workflow_agent, "planner_planning_command_for", REAL_PLANNER_PLANNING_COMMAND_FOR)


@pytest.fixture
def recipe_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """A library copy with a root skill (hours_overview) and a Library skill (dune_search)."""
    state = _start(tmp_path, initialized_fastworkflow, monkeypatch,
                   extra_planning_commands={HOURS_RECIPE: ROOT_PLANNING_COMMAND, DUNE_RECIPE: LIBRARY_PLANNING_COMMAND})
    use_the_real_planner_planning_command_for(monkeypatch)
    yield state
    state.ctx.pop_active_workflow()


def _match_only(monkeypatch, scope: str, value: str) -> None:
    """Jev's match, stubbed: the request asks for skill *value*, confidently, along *scope*."""
    monkeypatch.setattr(workflow_agent, "_planning_command_matcher",
                        lambda folder: {scope: _fake_matcher(value, 0.9)})


def test_the_matched_planning_commands_are_on_the_planner_listing(recipe_env):
    # The one-step call of a recipe resolves only if the planner lists it: its name and template.
    listing = planner_commands_text(recipe_env.ctx.get_active_workflow())
    assert "- hours_overview\n" in listing
    assert "  template:\n  - library_hours\n  - say when the branches are open\n" in listing
    assert "- dune_search\n" in listing
    assert "  template:\n  - find_book query=Dune\n  - say how many books match\n" in listing
