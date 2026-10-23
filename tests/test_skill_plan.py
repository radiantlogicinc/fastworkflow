"""The skill-file form: class PlanGenerator (four rules), __call__, @phase, pick, the filters,
how the planner's template renders unknown facts (and the tests made on them), the load-time
step check, fastworkflow.Refused and the default generate_utterances. Offline,
workflow-neutral."""
import types

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import command_directory, skill_commands
from fastworkflow.command_executor import call_response_generator
from fastworkflow.skill import (StepFormError, count, fact_from, listed, num, phase, pick,
                                picks_noted, planned, skill_parts)
from fastworkflow.skill_phases import (Placeholder, check_bindings, concatenated_steps,
                                       phase_steps, runs_now, stops_after)


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
    def __call__(self, input):
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
    from fastworkflow.skill_phases import tests_noted

    p = Placeholder("find")
    assert str(p.uids[0]) == "<find.uids[0]>" and str(p.only_uid("A")) == "<find.only_uid()>"
    with tests_noted() as tests:
        assert ", ".join(p.labels) == "<find.labels[*]>"
        assert p.ok and len(p.uids) == 1 and p.kind == "a" and p.kind != "b" and (p.total < 3)
        assert 1 < p.total and "x" in p.labels and int(p.n) == 999999
    assert tests == ["find.labels", "find.ok", "find.uids", "find.kind == 'a'",
                     "find.kind != 'b'", "find.total < 3", "find.total > 1", "'x' in find.labels"]


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
    from fastworkflow.skill_phases import PhaseError

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
    steps, _ = skill_parts(PlanGenerator)
    assert steps(_input(more=True)) == [
        "open_catalog", "find_item query=Dune", "open_item_by_uid item_uid=<uid>",
        "list_parts (full list)", "list_notes (on <find.labels[0]>)",
        "say that <items read> (only when item)", "apply no change (only when item)"]


def test_a_parameter_naming_a_phase_the_call_does_not_have_keeps_its_default():
    class Maybe:
        def __call__(self, input):
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

    loaded = skill_commands.command_from_module("maybe", _module(plan=Maybe), "_skills/maybe.py")
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

    with pytest.raises(skill_commands.SkillFormatError, match="phase 'answer' fails while its "
                                                                "template is built"):
        skill_commands.command_from_module("fails", _module(plan=Fails), "_skills/fails.py")

    class WritesNone:
        def find(self, input) -> Found:
            return [f"find_item query={input.name} owner={None}"]

        def answer(self, find):
            return ["say that done"]

    with pytest.raises(skill_commands.SkillFormatError, match="writes None into a step"):
        skill_commands.command_from_module("none", _module(plan=WritesNone), "_skills/none.py")


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

    loaded = skill_commands.command_from_module("said", _module(KindSignature, Said),
                                                "_skills/said.py")
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

    loaded = skill_commands.command_from_module("find", _module(plan=Flat), "_skills/find.py")
    assert loaded.command.plan_fn is None and skill_commands.expand(
        loaded.command, {"name": "Dune", "more": False, "stop_late": False}) == [
        "find_item query=Dune"]
    assert skill_commands.command_from_module("walk", _module(), "_skills/walk.py").command.plan_fn


def test_call_picks_the_phases_and_a_flat_call_runs_from_its_one_phase():
    class Either:
        def __call__(self, input):
            if not input.more:
                return [self.quick]
            return [self.find, self.answer]

        def quick(self, input):
            return [f"find_item query={input.name} (the first row only)"]

        def find(self, input) -> Found:
            return [f"find_item query={input.name}"]

        def answer(self, find):
            return [f"say that {find.labels[0]} was found"]

    command = skill_commands.command_from_module("either", _module(plan=Either),
                                                 "_skills/either.py").command
    args = {"name": "Dune", "more": False, "stop_late": False}
    assert skill_commands.phases_for(command, args) is None
    assert skill_commands.expand(command, args) == ["find_item query=Dune (the first row only)"]
    assert [p.name for p in skill_commands.phases_for(command, {**args, "more": True})] == [
        "find", "answer"]


def test_the_loader_refuses_the_old_forms_and_a_method_without_self():
    module = types.ModuleType("skill")
    module.Signature, module.steps = Signature, (lambda input: [])
    with pytest.raises(skill_commands.SkillFormatError, match="class PlanGenerator"):
        skill_commands.command_from_module("walk", module, "_skills/walk.py")

    class NoSelf:
        def find(input):
            return []

    with pytest.raises(skill_commands.SkillFormatError, match="self"):
        skill_commands.command_from_module("walk", _module(plan=NoSelf), "_skills/walk.py")


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
        def __call__(self, workflow, command):
            raise fastworkflow.Refused("no detail view here")

    out = call_response_generator(Generator(), None, "show")
    assert out.command_response.success is False
    assert out.command_response.response == "no detail view here"
    assert issubclass(fastworkflow.Refused, ValueError)

    class Crash:
        def __call__(self, workflow, command):
            raise ValueError("HTTP 503")

    with pytest.raises(ValueError):
        call_response_generator(Crash(), None, "show")


def test_refused_from_a_skill_hook_is_its_refusal():
    class Refusing(Signature):
        """Find a named item."""

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
            if cmd_parameters.name == "Nope":
                raise fastworkflow.Refused("name the item")
            return True, ""

    module = types.ModuleType("skill")
    module.Signature, module.PlanGenerator = Refusing, PlanGenerator
    command = skill_commands.command_from_module("walk", module, "_skills/walk.py").command
    with pytest.raises(skill_commands.SkillInputError, match="name the item"):
        skill_commands.validate_call(command, {"name": "Nope", "more": False, "stop_late": False}, "")
    assert skill_commands.validate_call(command, {"name": "Dune", "more": False, "stop_late": False}, "")


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

    command = skill_commands.command_from_module("typos", _module(plan=Typos),
                                                 "_skills/typos.py").command
    problems = skill_commands.check_step_commands(command, sigs)
    assert [p.split(": ", 1)[1] for p in problems] == [
        "a step runs 'find_itme', which is not a command of the workflow",
        "a step passes qeury to 'find_item', whose Signature.Input has no such field",
        "a step runs 'go_up', which is not a command of the workflow",
        "a step passes itme_uid to 'open_item_by_uid', whose Signature.Input has no such field"]
    with picks_noted() as picks:
        pick("open_x_by_uid", x_uid=None, named="A")
    assert picks == [("open_x_by_uid", "x_uid")]
