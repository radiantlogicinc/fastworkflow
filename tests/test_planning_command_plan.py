"""The skill-file form: class PlanGenerator (four rules), __phases__, @phase, pick, the filters,
how the planner's template renders unknown facts (and the tests made on them), the load-time
step check, fastworkflow.Refused and the default generate_utterances. Offline,
workflow-neutral."""
import inspect
import shutil
import types
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import command_directory, planning_commands, workflow_agent
from fastworkflow.command_executor import call_response_generator
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.planning import (StepFormError, count, fact_from, listed, num, phase, pick,
                                picks_noted, planned, planning_command_parts)
from fastworkflow.planning_phases import (Placeholder, check_bindings, concatenated_steps,
                                       phase_steps, runs_now, stops_after)
from tests.test_planning_commands import HOURS_RECIPE, _match_only, recipe_env  # noqa: F401 - recipe_env is a fixture
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture


class Found(BaseModel):
    uids: list[str] = fact_from("find_item.item_uids")
    labels: list[str] = fact_from("find_item.labels")


class Signature:
    """Find a named item and walk its parts."""

    class Input(BaseModel):
        name: str = Field(description="the item's name")
        more: bool = Field(default=False, description="also read the notes")
        stop_late: bool = Field(default=False, description="stop at the first late item")


def _late(item):
    return any(i.late for i in item)


class PlanGenerator:
    def __phases__(self, input):
        notes = [self.notes] if input.more else []
        return [self.find, self.item, *notes, self.answer]

    def find(self, input) -> Found:
        return ["open_catalog", f"find_item query={input.name}"]

    class Item(BaseModel):
        uid: str = fact_from("open_item_by_uid.item_uid")
        late: bool = fact_from("open_item_by_uid.late")

    @phase(over="find.uids", noun="items",
           until=lambda input: _late if input.stop_late else None)
    def item(self, uid) -> Item:
        return [f"open_item_by_uid item_uid={uid}", "list_parts (full list)"]

    def notes(self, find):
        return [f"list_notes (on {find.labels[0]})"]

    def answer(self, input, item=None):
        if not item:
            return [f"say that no item named {input.name} was found", "apply no change"]
        return [f"say that {self._items_read(item)}", "apply no change"]

    def _items_read(self, item):
        return f"{len(item)} items: {', '.join(i.uid for i in item)}"


def _input(**kw):
    return Signature.Input(**{"name": "Dune", **kw})


def _module(signature=Signature, plan=PlanGenerator):
    module = types.ModuleType("skill")
    module.Signature, module.PlanGenerator = signature, plan
    return module


def test_pick_writes_the_uid_or_the_match_wording():
    assert pick("open_item_by_uid", item_uid="u1", named="Dune") == "open_item_by_uid item_uid=u1"
    # A uid containing "(" or ";" is quoted, so the chain keeps it (fix-ajir.40).
    assert pick("open_item_by_uid", item_uid="b1(eu)", named="Acme") == \
        'open_item_by_uid item_uid="b1(eu)"'
    assert pick("open_item_by_uid", item_uid="west;1", named="Sales") == \
        'open_item_by_uid item_uid="west;1"'
    assert pick("open_item_by_uid", item_uid='say "hi";', named="Note") == \
        """open_item_by_uid item_uid='say "hi";'"""
    assert pick("open_item_by_uid", item_uid=None, named="Dune") == \
        "open_item_by_uid with the uid of the Dune match"
    assert pick("open_item_by_uid", item_uid=Placeholder("find.only_uid()"), named="Dune") == \
        "open_item_by_uid with the uid of the Dune match"
    with pytest.raises(StepFormError):
        pick("open_x", label="u1", named="Dune")
    with pytest.raises(StepFormError):
        pick("open_x", x_uid="u1", named=" ")


def test_filters():
    assert num(5000000.0) == "5000000" and num(5000000.0, grouped=True) == "5,000,000"
    assert num(2.5) == "2.5" and num(None) == "none recorded" and num(None, none="none") == "none"
    assert count(1, "login") == "1 login" and count(2, "person", "people") == "2 people"
    assert listed([], 0, "group") == "none (0 groups)"
    assert listed(["a"], 3, "group") == "3 groups (a; 1 of 3 shown)"
    assert listed(["a", "b"], 2, "right", ("names as above", ["b", "a"])) == \
        "2 rights (the same names as above)"
    assert num(Placeholder("x.n")) == "<x.n>"
    assert count(Placeholder("x.n"), "login") == "<x.n> logins"
    assert listed(Placeholder("x.names"), Placeholder("x.total"), "group") == \
        "<x.total> groups (<x.names>)"


def test_a_placeholder_reads_as_where_the_fact_comes_from_and_every_test_on_it_is_true():
    from fastworkflow.planning_phases import tests_noted

    p = Placeholder("find")
    assert str(p.uids[0]) == "<find.uids[0]>" and str(p.only_uid("A")) == "<find.only_uid()>"
    with tests_noted() as tests:
        assert ", ".join(p.labels) == "<find.labels[*]>"
        assert p.ok and len(p.uids) == 1 and p.kind == "a" and p.kind != "b" and (p.total < 3)
        assert 1 < p.total and "x" in p.labels and int(p.n) == 999999
    assert tests == ["find.labels", "find.ok", "find.uids", "find.kind == 'a'",
                     "find.kind != 'b'", "find.total < 3", "find.total > 1", "'x' in find.labels"]


def test_a_fact_named_like_a_str_method_still_reads_as_the_fact():
    # title, count, format, split... read as the fact itself, not the str method of the
    # same name; index keeps its own meaning (the one item's place, not str.index).
    p = Placeholder("book")
    assert str(p.title) == "<book.title>" and str(p.count) == "<book.count>"
    assert str(p.format) == "<book.format>" and str(p.split) == "<book.split>"
    assert p.index() == 0


def test_each_method_is_a_phase_in_order_and_call_gives_the_shape():
    phases = planned(PlanGenerator)(_input())
    assert [p.name for p in phases] == ["find", "item", "answer"]          # __call__: no notes
    assert phases[0].output is Found and phases[1].output is PlanGenerator.Item
    assert phases[1].item == "uid" and phases[1].until is None and phases[2].output is None
    assert [p.name for p in planned(PlanGenerator)(_input(more=True))] == ["find", "item", "notes",
                                                                  "answer"]
    assert planned(PlanGenerator)(_input(stop_late=True))[1].until.__name__ == "_late"
    check_bindings(phases)


def test_return_empty_is_decided_when_the_phase_is_reached_calling_the_method_once():
    calls = []

    class Gate:
        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        def open(self, input, find):
            calls.append(find)
            if not find.uids:
                return []
            return [pick("open_item_by_uid", item_uid=find.uids[0], named=input.name)]

    find, opened = planned(Gate)(_input())
    assert callable(opened.when) and opened.when.only_when() == ["find.uids"]
    calls.clear()
    hit = {"find": Found(uids=["u1"], labels=["Dune"])}
    assert runs_now(opened, hit) and phase_steps(opened, hit) == ["open_item_by_uid item_uid=u1"]
    assert len(calls) == 1                                  # its steps were kept from `when`
    assert not runs_now(opened, {"find": Found(uids=[], labels=[])})
    assert concatenated_steps([find, opened]) == [
        "find_item query=Dune", "open_item_by_uid with the uid of the Dune match (only when "
        "find.uids)"]


def test_a_method_that_fails_at_run_time_fails_its_phase():
    from fastworkflow.planning_phases import PhaseError

    class Broken:
        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        def open(self, find):
            return [f"open_item_by_uid item_uid={find.uids[3]}"]

    _, opened = planned(Broken)(_input())
    facts = {"find": Found(uids=["u1"], labels=["Dune"])}
    assert runs_now(opened, facts)                          # the error is kept for steps_fn
    with pytest.raises(PhaseError, match="steps_fn failed: IndexError"):
        phase_steps(opened, facts)


def test_the_template_shows_facts_and_helpers_by_name():
    steps, _ = planning_command_parts(PlanGenerator)
    assert steps(_input(more=True)) == [
        "open_catalog", "find_item query=Dune", "open_item_by_uid item_uid=<uid>",
        "list_parts (full list)", "list_notes (on <find.labels[0]>)",
        "say that <items read> (only when item)", "apply no change (only when item)"]


def test_a_parameter_naming_a_phase_the_call_does_not_have_keeps_its_default():
    class Maybe:
        def __phases__(self, input):
            return [self.find, self.extra, self.answer] if input.more else [self.find, self.answer]

        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        class Extra(BaseModel):
            total: int = fact_from("count_checks.total")

        def extra(self) -> Extra:
            return ["count_checks"]

        def answer(self, find, extra=None):
            more = [f"say that it has {extra.total} checks"] if extra is not None else []
            return [f"say that {find.labels[0]} was found", *more]

    loaded = planning_commands.command_from_module("maybe", _module(plan=Maybe), "_commands/maybe.py")
    assert list(loaded.steps) == [
        "find_item query=<name>", "count_checks (only when more)",
        "say that <find.labels[0]> was found",
        "say that it has <extra.total> checks (only when more)"]


def test_a_method_failing_on_unknown_facts_or_writing_none_is_a_load_error():
    class Fails:
        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        def answer(self, find):
            return [f"say that {find.labels[0].upper()[:2] + 1}"]

    with pytest.raises(planning_commands.PlanningCommandFormatError, match="phase 'answer' fails while its "
                                                                "template is built"):
        planning_commands.command_from_module("fails", _module(plan=Fails), "_commands/fails.py")

    class WritesNone:
        def find(self, input) -> Found:
            return [f"find_item query={input.name} owner={None}"]

        def answer(self, find):
            return ["say that done"]

    with pytest.raises(planning_commands.PlanningCommandFormatError, match="writes None into a step"):
        planning_commands.command_from_module("none", _module(plan=WritesNone), "_commands/none.py")


def test_the_loaded_template_merges_enum_lines_without_splitting_a_bracket():
    from typing import Literal

    class KindSignature:
        """Say what one named record has."""

        class Input(BaseModel):
            name: str = Field(description="its name")
            kind: Literal["a", "b", "c"] = Field(description="its kind")

    class Said:
        def answer(self, input):
            return [f"say that {input.name} ({input.kind}_uid u1) has nothing",
                    f"list_{input.kind} (its {input.kind}s; full list)",
                    f"open_{input.kind} {input.kind}_uid=<{input.kind}s.uids[0]>"]

    loaded = planning_commands.command_from_module("said", _module(KindSignature, Said),
                                                "_commands/said.py")
    # a "(" keeps its ")" (no "<list_a (its as; or ..., by kind> full list)"): lines that
    # differ in more words than they share stay apart; a key= keeps its value
    assert list(loaded.steps) == [
        "say that <name> (<a_uid or b_uid or c_uid, by kind> u1) has nothing",
        "list_a (its as; full list) (only when kind is a)",
        "list_b (its bs; full list) (only when kind is b)",
        "list_c (its cs; full list) (only when kind is c)",
        "<open_a a_uid=<as.uids[0]> or open_b b_uid=<bs.uids[0]> or open_c c_uid=<cs.uids[0]>, "
        "by kind>"]


def test_a_plan_of_one_phase_that_reads_no_facts_runs_flat():
    class Flat:
        def find(self, input):
            return [f"find_item query={input.name}"]

    loaded = planning_commands.command_from_module("find", _module(plan=Flat), "_commands/find.py")
    assert loaded.command.plan_fn is None and planning_commands.expand(
        loaded.command, {"name": "Dune", "more": False, "stop_late": False}) == [
        "find_item query=Dune"]
    assert planning_commands.command_from_module("walk", _module(), "_commands/walk.py").command.plan_fn


def test_call_picks_the_phases_and_a_flat_call_runs_from_its_one_phase():
    class Either:
        def __phases__(self, input):
            if not input.more:
                return [self.quick]
            return [self.find, self.answer]

        def quick(self, input):
            return [f"find_item query={input.name} (the first row only)"]

        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        def answer(self, find):
            return [f"say that {find.labels[0]} was found"]

    command = planning_commands.command_from_module("either", _module(plan=Either),
                                                 "_commands/either.py").command
    args = {"name": "Dune", "more": False, "stop_late": False}
    assert planning_commands.phases_for(command, args) is None
    assert planning_commands.expand(command, args) == ["find_item query=Dune (the first row only)"]
    assert [p.name for p in planning_commands.phases_for(command, {**args, "more": True})] == [
        "find", "answer"]


def test_the_loader_refuses_the_old_forms_and_a_method_without_self():
    module = types.ModuleType("skill")
    module.Signature, module.steps = Signature, (lambda input: [])
    with pytest.raises(planning_commands.PlanningCommandFormatError, match="class PlanGenerator"):
        planning_commands.command_from_module("walk", module, "_commands/walk.py")

    class NoSelf:
        def find(input):
            return []

    with pytest.raises(planning_commands.PlanningCommandFormatError, match="self"):
        planning_commands.command_from_module("walk", _module(plan=NoSelf), "_commands/walk.py")


def test_a_fan_out_runs_per_item_and_stops_on_until():
    phases = planned(PlanGenerator)(_input(stop_late=True))
    item = phases[1]
    late = PlanGenerator.Item(uid="u1", late=True)
    assert stops_after(item, [late], {}) and not stops_after(
        item, [PlanGenerator.Item(uid="u1", late=False)], {})
    assert phase_steps(item, {}, "u7")[0] == "open_item_by_uid item_uid=u7"
    assert concatenated_steps(phases)[2] == "open_item_by_uid item_uid=<uid>"


def test_refused_from_a_response_generator_is_a_failed_output():
    class Generator:
        def __call__(self, workflow, command, input):
            raise fastworkflow.Refused("no detail view here")

    out = call_response_generator(Generator(), None, "show", None)
    assert out.command_response.success is False
    assert out.command_response.response == "no detail view here"
    assert issubclass(fastworkflow.Refused, ValueError)

    class Crash:
        def __call__(self, workflow, command, input):
            raise ValueError("HTTP 503")

    with pytest.raises(ValueError):
        call_response_generator(Crash(), None, "show", None)


def test_refused_from_a_planning_command_hook_is_its_refusal():
    class Refusing(Signature):
        """Find a named item."""

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
            if cmd_parameters.name == "Nope":
                raise fastworkflow.Refused("name the item")
            return None

    module = types.ModuleType("skill")
    module.Signature, module.PlanGenerator = Refusing, PlanGenerator
    command = planning_commands.command_from_module("walk", module, "_commands/walk.py").command
    with pytest.raises(planning_commands.PlanningCommandInputError, match="name the item"):
        planning_commands._validate_arguments(command, {"name": "Nope", "more": False, "stop_late": False}, "")
    assert planning_commands._validate_arguments(command, {"name": "Dune", "more": False, "stop_late": False}, "")


def test_the_default_generate_utterances_is_the_standard_body(monkeypatch):
    calls = []
    monkeypatch.setattr("fastworkflow.train.generate_synthetic.generate_diverse_utterances",
                        lambda seeds, name: calls.append((list(seeds), name)) or ["more"])

    class Sig:
        plain_utterances = ["show it"]

    assert command_directory.default_generate_utterances(Sig, None, "Ctx/show_item") == [
        "show item", "more"]
    assert calls == [(["show it"], "Ctx/show_item")]

    registered = {}
    directory = types.SimpleNamespace(
        register_utterance_metadata=lambda name, meta: registered.__setitem__(name, meta))
    metadata = types.SimpleNamespace(response_generation_module_path="x.py",
                                     workflow_folderpath="/wf")
    for name, sig in (("a", Sig), ("b", type("Own", (Sig,), {
            "generate_utterances": staticmethod(lambda wf, name: ["own"])}))):
        command_directory.CommandDirectory._populate_utterance_metadata_for_command(
            directory, name, metadata, "/wf", module=types.SimpleNamespace(Signature=sig))
    assert registered["a"].generated_utterances_func_name == command_directory.DEFAULT_UTTERANCES_FUNC
    assert registered["b"].generated_utterances_func_name == "Signature.generate_utterances"
    assert registered["a"].generated_utterances_module_filepath == "x.py"


def test_the_step_check_names_unknown_commands_and_parameters():
    class Cmd(BaseModel):
        query: str = ""

    sigs = {"Catalog/find_item": types.SimpleNamespace(Input=Cmd),
            "open_item_by_uid": types.SimpleNamespace(
                Input=type("I", (BaseModel,), {"__annotations__": {"item_uid": str}}))}

    class Typos:
        def find(self, input) -> Found:
            return [f"find_itme query={input.name}", f"find_item qeury={input.name} (k=v note)",
                    "go_up, then find_item other=1", "say that find_item ran", "apply no change"]

        def open(self, input, find):
            return [pick("open_item_by_uid", itme_uid=find.only_uid(input.name), named=input.name)]

    command = planning_commands.command_from_module("typos", _module(plan=Typos),
                                                 "_commands/typos.py").command
    problems = planning_commands.check_step_commands(command, sigs)
    assert [p.split(": ", 1)[1] for p in problems] == [
        "a step runs 'find_itme', which is not a command of the workflow",
        "a step passes qeury to 'find_item', whose Signature.Input has no such field",
        "a step runs 'go_up', which is not a command of the workflow",
        "a step passes itme_uid to 'open_item_by_uid', whose Signature.Input has no such field"]
    with picks_noted() as picks:
        pick("open_x_by_uid", x_uid=None, named="A")
    assert picks == [("open_x_by_uid", "x_uid")]


# ------------------------------------------------- a PlanGenerator subclassing another skill's

class Lookup:
    """A base skill's plan: find, then open the match, then answer."""

    def find(self, input) -> Found:
        return [f"find_item query={input.name}"]

    class Item(BaseModel):
        uid: str = fact_from("open_item_by_uid.item_uid")

    def item(self, input, find) -> "Item":                 # a string annotation: base's model
        if not find.uids:
            return []
        return [pick("open_item_by_uid", item_uid=find.uids[0], named=input.name)]

    def answer(self, input, item=None):
        return [f"say that {input.name} is {self._uid(item)}"]

    def _uid(self, item):
        return item.uid if item is not None else "not found"


def test_a_subclass_that_adds_phases_writes_call_with_the_order():
    class Parts(Lookup):
        class PartFacts(BaseModel):
            labels: list[str] = fact_from("list_parts.labels")

        def parts(self, item=None) -> PartFacts:
            if item is None:
                return []
            return ["list_parts (full list)"]

    from fastworkflow.planning_phases import PhaseError

    with pytest.raises(PhaseError, match=r"Parts adds the phase\(s\) parts to those of "
                       r"test_planning_command_plan.Lookup: write __phases__\(self, input\) returning the phases a call "
                       r"runs, in order \(from: self.find, self.item, self.answer, self.parts\)"):
        planned(Parts)
    with pytest.raises(planning_commands.PlanningCommandFormatError, match="write __phases__"):
        planning_commands.command_from_module("parts", _module(plan=Parts), "_commands/parts.py")

    class Ordered(Parts):                                   # base phases placed anywhere
        def __phases__(self, input):
            return [self.find, self.parts, self.item, self.answer]

    assert [p.name for p in planned(Ordered)(_input())] == ["find", "parts", "item", "answer"]


def test_call_places_base_phases_and_an_override_reads_by_its_own_signature():
    class Parts(Lookup):
        def __phases__(self, input):
            return [self.find, self.item, self.parts, self.answer]

        class PartFacts(BaseModel):
            labels: list[str] = fact_from("list_parts.labels")

        def parts(self, item=None) -> PartFacts:
            if item is None:
                return []
            return ["list_parts (full list)"]

        def answer(self, input, item=None, parts=None):           # overrides Lookup.answer
            if parts is None:
                return [f"say that no item named {input.name} was found"]
            return [f"say that {input.name} has parts {', '.join(parts.labels)}"]

    phases = planned(Parts)(_input())
    assert [p.name for p in phases] == ["find", "item", "parts", "answer"]
    assert list(inspect.signature(phases[3].steps_fn).parameters) == ["item", "parts"]
    facts = {"find": Found(uids=["u1"], labels=["Dune"])}
    assert phase_steps(phases[1], facts) == ["open_item_by_uid item_uid=u1"]
    facts["item"] = Lookup.Item(uid="u1")
    facts["parts"] = Parts.PartFacts(labels=["gear", "axle"])
    assert phase_steps(phases[3], facts) == ["say that Dune has parts gear, axle"]
    assert not runs_now(phases[1], {"find": Found(uids=[], labels=[])})


class Parts(Lookup):
    """Extends Lookup.answer's steps with super(), in run and in template mode."""

    def __phases__(self, input):
        return [self.find, self.item, self.parts, self.answer]

    class PartFacts(BaseModel):
        labels: list[str] = fact_from("list_parts.labels")
        total: int = fact_from("list_parts.total")

    def parts(self, item=None) -> PartFacts:
        if item is None:
            return []
        return ["list_parts (full list)"]

    def answer(self, input, item=None, parts=None):
        steps = super().answer(input, item)
        if parts is None:
            return steps
        return [*steps, f"say that it has {self._parts(parts)}"]

    def _parts(self, parts):
        return f"{parts.total} parts: {', '.join(parts.labels)}"


def test_an_override_extends_the_base_steps_with_super_at_run_time_and_in_the_template():
    phases = planned(Parts)(_input())
    facts = {"find": Found(uids=["u1"], labels=["Dune"]), "item": Lookup.Item(uid="u1"),
             "parts": Parts.PartFacts(labels=["gear", "axle"], total=2)}
    assert phase_steps(phases[3], facts) == ["say that Dune is u1",
                                             "say that it has 2 parts: gear, axle"]
    assert phase_steps(phases[3], {"find": facts["find"]}) == ["say that Dune is not found"]
    loaded = planning_commands.command_from_module("parts", _module(plan=Parts), "_commands/parts.py")
    # template mode: super() reaches Lookup.answer, whose helper reads as its name
    assert list(loaded.steps) == [
        "find_item query=<name>",
        "open_item_by_uid with the uid of the <name> match (only when find.uids)",
        "list_parts (full list)",
        "say that <name> is <uid>", "say that it has <parts>"]


def test_call_cannot_return_a_base_method_the_class_overrides():
    class Wrong(Parts):
        def __phases__(self, input):
            return [self.find, super().answer]

        def answer(self, input, item=None):
            return ["say that it is done"]

    from fastworkflow.planning_phases import PhaseError

    with pytest.raises(PhaseError, match=r"returns a base's answer that the class overrides: "
                                         r"return self.answer"):
        planned(Wrong)(_input())


def test_a_subclass_that_only_overrides_keeps_the_base_plan_and_inherits_call():
    class Same(Lookup):
        def answer(self, input, item=None):
            return [*super().answer(input, item), "apply no change"]

    class Chosen(Lookup):
        def __phases__(self, input):
            return [self.find, self.answer]

    class Inherits(Chosen):
        pass

    assert [p.name for p in planned(Same)(_input())] == ["find", "item", "answer"]
    assert [p.name for p in planned(Inherits)(_input())] == ["find", "answer"]
    loaded = planning_commands.command_from_module("same", _module(plan=Same), "_commands/same.py")
    assert loaded.steps[-2:] == ("say that <name> is <uid>", "apply no change")


def test_phases_from_several_bases_need_call():
    class Other:
        def notes(self, input):
            return [f"list_notes (on {input.name})"]

    class Both(Lookup, Other):
        pass

    from fastworkflow.planning_phases import PhaseError

    with pytest.raises(PhaseError, match=r"Both has phases from test_planning_command_plan.Lookup, test_planning_command_plan.*Other: write __phases__"):
        planned(Both)


def test_a_name_a_subclass_sets_to_a_non_function_is_not_a_phase():
    class NoAnswer(Lookup):
        answer = None

    assert [p.name for p in planned(NoAnswer)(_input())] == ["find", "item"]


# ------------------------------------------------- phases with no Input (fix-bh1e.11)

class NoInputSignature:
    """Summarize the catalog: its sections, then its late items."""


class NoInputPlan:
    def sections(self):
        return ["list_sections"]

    def late(self):
        return ["count_late", "say how many items are late"]


@pytest.mark.parametrize("signature", [
    NoInputSignature,
    type("EmptyInput", (NoInputSignature,), {"__doc__": NoInputSignature.__doc__,
                                             "Input": type("Input", (BaseModel,), {})})])
def test_a_planning_command_with_no_inputs_can_have_phases(signature):
    loaded = planning_commands.command_from_module("summary", _module(signature, NoInputPlan),
                                                "_commands/summary.py")
    command = loaded.command
    assert command is not None and command.params == ()
    assert loaded.steps == ("list_sections", "count_late", "say how many items are late")
    assert command.signature_text().splitlines()[0] == "summary()"
    assert planning_commands._template_grid(command) == [{}]           # one plan, one template
    phases = planning_commands.call_phases(command, {})
    assert [p.name for p in phases] == ["sections", "late"]
    assert [phase_steps(p, {}) for p in phases] == [
        ["list_sections"], ["count_late", "say how many items are late"]]
    fields = planning_commands.typed_signature(command).output_fields
    assert list(fields) == ["steps_before", "steps_after"]          # a call with no arguments


def test_a_planning_command_with_no_inputs_and_one_method_stays_fixed_steps():
    class OnePlan:
        def summary(self):
            return ["list_sections", "say how many sections there are"]

    loaded = planning_commands.command_from_module("summary", _module(NoInputSignature, OnePlan),
                                                "_commands/summary.py")
    assert planning_commands.fixed_steps(loaded.command)
    assert loaded.steps == ("list_sections", "say how many sections there are")


def test_a_no_input_phased_planning_command_call_has_no_arguments():
    import dspy
    from dspy.utils import DummyLM

    command = planning_commands.command_from_module(
        "summary", _module(NoInputSignature, NoInputPlan), "_commands/summary.py").command
    lm = DummyLM([])
    with dspy.context(lm=lm):
        call = planning_commands.typed_planning_call(command, "give me the catalog summary",
                                              "list_sections: x\ncount_late: y")
    assert lm.history == []
    assert call.args == {} and call.plan == "" and [p.name for p in call.phases] == ["sections", "late"]


def test_planner_keeps_instructions_and_the_adapt_template(request, monkeypatch):
    assert workflow_agent.PLANNER_INSTRUCTIONS.strip().startswith("Carefully review the user_query")

    folder = request.getfixturevalue("recipe_env").wf.folderpath
    _match_only(monkeypatch, "", HOURS_RECIPE)
    template, command = workflow_agent.planner_planning_command_for("When are the branches open?", folder)
    assert template == (
        "A job like this is usually done as:\n- library_hours\n- say when the branches are open\n"
        "Adapt this to the request: drop steps it does not need and add steps it asks for.")
    assert command is workflow_agent.load_planning_command_map(folder)[HOURS_RECIPE]


EXAMPLE_TODO = str(Path(__file__).resolve().parent / "example_workflow")


def _listing_headers(listing: str) -> list[str]:
    return [line for line in listing.split("\n") if line.startswith("Commands available")]


def test_the_typed_listing_is_the_planner_listing_cut_to_names_sentences_and_parameters(request):
    # The typed call (design 3.4) reads a shorter listing; the text planner keeps the full one.
    workflow = request.getfixturevalue("recipe_env").ctx.get_active_workflow()
    full = workflow_agent.planner_commands_text(workflow)
    trimmed = workflow_agent.typed_planner_commands_text(workflow)

    assert len(trimmed) < len(full)
    assert _listing_headers(trimmed) == _listing_headers(full)
    assert planning_commands.command_names(trimmed) == planning_commands.command_names(full)
    assert "  inputs:" in full and "parameters:" not in full
    assert "- book_loans\n  Say who has a book on loan, given its title.\n  parameters: title\n" in trimmed
    assert "- library_hours\n  Say when the library's branches are open, or one branch's hours.\n" \
        "  parameters: branch\n" in trimmed
    # A command with no parameters has no parameters line.
    assert "- go_up\n  Change context to the parent of the current context.\n- list_branches\n" in trimmed


@pytest.mark.usefixtures("initialized_fastworkflow")
def test_a_command_in_two_contexts_keeps_each_contexts_description(tmp_path):
    # set_properties is in TodoItem and TodoList; each context's Signature gives it its own description.
    folder = tmp_path / "todo_wf"
    shutil.copytree(EXAMPLE_TODO, folder)
    nouns = (("TodoItem", "item"), ("TodoList", "list"))
    for context, noun in nouns:
        path = folder / "_commands" / context / "set_properties.py"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(
            "class Signature:\n", f'class Signature:\n    """Set the properties of the open {noun}."""\n\n', 1),
            encoding="utf-8")
    workflow = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"todo_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    try:
        trimmed = workflow_agent.typed_planner_commands_text(workflow)
    finally:
        ctx.pop_active_workflow()

    for context, noun in nouns:
        section = f"Commands available after entering the {context} context (not callable until then):\n"
        block = trimmed.split(section, 1)[1].split("\nCommands available", 1)[0]
        assert (f"- set_properties\n  Set the properties of the open {noun}.\n"
                "  parameters: description, assign_to, status\n") in block


def test_a_parameters_line_is_not_read_as_a_command_description():
    # The typed listing gives a command with no description only its parameters line.
    listing = "Commands:\n- open_book\n  parameters: book_uid\n- go_up\n  Change context.\n"
    assert planning_commands.command_descriptions(listing) == {"go_up": "Change context."}
