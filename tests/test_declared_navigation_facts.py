"""``occupiable`` and ``descends_to`` are read from the workflow's own classes.

``occupiable`` is declared on a context's callback class, and ``descends_to`` /
``descend_parameter`` on a command's response-generator class. With neither
declared, the context is listed and the command has no navigation effect.
"""
from __future__ import annotations

from types import SimpleNamespace

import fastworkflow
from fastworkflow import ModuleType, entry_declarations
from fastworkflow import context_navigation
from fastworkflow import command_metadata_api
from fastworkflow.command_metadata_api import CommandMetadataAPI


class _FakeContext:
    occupiable = False


class _FakeContextNoDeclaration:
    enter_command = "open_thing <thing_uid>"


class _FakeContextBadValue:
    occupiable = "no"


class _FakeRouting:
    def __init__(self, contexts, modules=None, raises=()):
        self.contexts = contexts
        self._modules = modules or {}
        self._raises = set(raises)

    def get_command_class(self, command_name, module_type):
        assert module_type == ModuleType.RESPONSE_GENERATION_INFERENCE
        if command_name in self._raises:
            raise RuntimeError("signature does not import")
        return self._modules.get(command_name)


# ----------------------------------------------------------------------
# declared_occupiable
# ----------------------------------------------------------------------


def test_occupiable_false_on_context_class_is_declared(monkeypatch):
    monkeypatch.setattr(entry_declarations, "_declared_context_class",
                        lambda wf, name: _FakeContext)
    assert entry_declarations.declared_occupiable("wf", "ControlCatalog") is False


def test_occupiable_absent_from_context_class_is_not_declared(monkeypatch):
    monkeypatch.setattr(entry_declarations, "_declared_context_class",
                        lambda wf, name: _FakeContextNoDeclaration)
    assert entry_declarations.declared_occupiable("wf", "Account") is None


def test_context_without_callback_class_is_not_declared(monkeypatch):
    monkeypatch.setattr(entry_declarations, "_declared_context_class",
                        lambda wf, name: None)
    assert entry_declarations.declared_occupiable("wf", "Resource") is None


def test_non_boolean_occupiable_is_not_declared(monkeypatch):
    monkeypatch.setattr(entry_declarations, "_declared_context_class",
                        lambda wf, name: _FakeContextBadValue)
    assert entry_declarations.declared_occupiable("wf", "Account") is None


def test_unreadable_context_is_not_declared(monkeypatch):
    def boom(wf, name):
        raise RuntimeError("no routing")

    monkeypatch.setattr(entry_declarations, "_declared_context_class", boom)
    assert entry_declarations.declared_occupiable("wf", "Account") is None


# ----------------------------------------------------------------------
# _declared_descend
# ----------------------------------------------------------------------


def test_descends_to_with_gate_is_declared_on_command_module():
    class ResponseGenerator:
        descends_to = "Permission"
        descend_parameter = "permission_uid"

    routing = _FakeRouting({}, modules={"Account/list_permissions": ResponseGenerator})
    effect = context_navigation._declared_descend(routing, "Account/list_permissions")
    assert effect.declared_targets() == ("Permission",)
    assert effect.when_parameter_present == "permission_uid"


def test_descends_to_without_gate_always_descends():
    class ResponseGenerator:
        descends_to = "Group"

    routing = _FakeRouting({}, modules={"Application/open_group": ResponseGenerator})
    effect = context_navigation._declared_descend(routing, "Application/open_group")
    assert effect.declared_targets() == ("Group",)
    assert effect.when_parameter_present is None


def test_command_without_descends_to_is_not_declared():
    class ResponseGenerator:
        pass

    routing = _FakeRouting({}, modules={"Account/show_owner": ResponseGenerator})
    assert context_navigation._declared_descend(routing, "Account/show_owner") is None


def test_command_without_response_generator_is_not_declared():
    routing = _FakeRouting({}, modules={})
    assert context_navigation._declared_descend(routing, "Account/missing") is None


def test_unreadable_command_module_is_not_declared():
    routing = _FakeRouting({}, raises={"Account/broken"})
    assert context_navigation._declared_descend(routing, "Account/broken") is None


# ----------------------------------------------------------------------
# Navigation facts, as the framework reads them
# ----------------------------------------------------------------------


def _patch_navigation(monkeypatch, routing, declared_occupiable):
    monkeypatch.setattr(context_navigation.RoutingRegistry, "get_definition",
                        staticmethod(lambda wf: routing))
    monkeypatch.setattr(context_navigation, "declared_occupiable", declared_occupiable)


def test_descends_to_is_the_navigation_effect(monkeypatch):
    class ResponseGenerator:
        descends_to = "Permission"
        descend_parameter = "permission_uid"

    routing = _FakeRouting({"Account": ["Account/list_permissions"]},
                           modules={"Account/list_permissions": ResponseGenerator})
    _patch_navigation(monkeypatch, routing, lambda wf, name: None)

    effects, _ = context_navigation._navigation_effects("wf")
    effect = effects["Account/list_permissions"]
    assert effect.declared_targets() == ("Permission",)
    assert effect.when_parameter_present == "permission_uid"


def test_command_without_descends_to_has_no_navigation_effect(monkeypatch):
    class ResponseGenerator:
        pass

    routing = _FakeRouting({"Account": ["Account/show_owner"]},
                           modules={"Account/show_owner": ResponseGenerator})
    _patch_navigation(monkeypatch, routing, lambda wf, name: None)

    effects, occupiable = context_navigation._navigation_effects("wf")
    assert "Account/show_owner" not in effects
    assert "Account" not in occupiable


def test_occupiable_is_read_from_the_context_class(monkeypatch):
    routing = _FakeRouting({"ControlCatalog": [], "Account": []})

    def declared(wf, name):
        return False if name == "ControlCatalog" else None

    _patch_navigation(monkeypatch, routing, declared)
    _, occupiable = context_navigation._navigation_effects("wf")
    assert occupiable == {"ControlCatalog": False}


def test_enterable_list_leaves_out_only_the_non_occupiable_contexts(monkeypatch):
    routing = _FakeRouting({"*": [], "ControlCatalog": [], "Account": []})
    monkeypatch.setattr(command_metadata_api, "declared_occupiable",
                        lambda wf, name: False if name == "ControlCatalog" else None)
    monkeypatch.setattr(fastworkflow, "RoutingRegistry",
                        SimpleNamespace(get_definition=lambda wf: routing))

    assert CommandMetadataAPI._occupiable_context_names("wf") == {"*", "Account"}


def test_enterable_list_is_none_when_nothing_declares_occupiability(monkeypatch):
    routing = _FakeRouting({"*": [], "Account": []})
    monkeypatch.setattr(command_metadata_api, "declared_occupiable", lambda wf, name: None)
    monkeypatch.setattr(fastworkflow, "RoutingRegistry",
                        SimpleNamespace(get_definition=lambda wf: routing))

    assert CommandMetadataAPI._occupiable_context_names("wf") is None
