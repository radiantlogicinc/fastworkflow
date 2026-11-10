"""How the framework validates and writes a command's ``Input`` without running its model validators
out of context (spec 3.2).

An ``Input`` declares its cross-field rules as ``@model_validator(mode="after")``. Those validators read
the validation context (``command_text``, ``workflow``, ``command``), and only ``check_input`` (planning
side) and ``_check_input_rules`` (execution side) supply it. Any other path that builds or writes an
``Input`` must therefore either run no model validators, or run them with the context. The helpers here
are the paths that run none:

- ``set_field_unvalidated`` writes a field without pydantic's assignment validation. An ``Input`` with
  ``validate_assignment=True`` re-runs its model validators on every assignment, and pydantic does not
  carry the validation context into that re-run, so the framework's own writes (defaults, screened values,
  values a check already validated) go through here.
- ``field_checked_class`` is the model with its model validators removed and its field validators kept.
  ``bound_fields`` uses it to check name=value bindings: a field's own validators run there, as in a full
  validation, but no model validator can refuse a binding (those run later, with the context).
- ``fields_validated`` builds a model from data already checked where it was produced (an archived run's
  output), with each field checked and no model validator run.
- ``validation_class`` is the model with ``validate_assignment`` off. ``validate_input`` (used by ``check_input``)
  builds its instance from it, so an assignment a validator makes to ``self`` does not re-run the validators
  without their context. ``validate_input`` runs the model validators itself, one by one, so the writes a
  validator made before it refused are readable from its ``ValidationTrace``.

This module imports nothing from the framework, so every layer can use it.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from types import SimpleNamespace
from typing import Any, Type

from pydantic import BaseModel, ConfigDict, ValidationError, create_model, field_validator


def set_field_unvalidated(obj: BaseModel, name: str, value: Any) -> None:
    """Write *value* to field *name* of *obj* with no assignment validation (see the module docstring).

    Pydantic stores field values in the instance ``__dict__``, and marks an assigned field as set.
    """
    obj.__dict__[name] = value
    obj.__pydantic_fields_set__.add(name)


@functools.lru_cache(maxsize=None)
def validation_class(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """*model_class* with ``validate_assignment`` off: a subclass that inherits the Input's fields and its
    validators, used only to validate. Cached per class."""
    return type(model_class.__name__, (model_class,),
                {"__module__": model_class.__module__,
                 "__qualname__": model_class.__qualname__,
                 "model_config": ConfigDict(validate_assignment=False)})



@functools.lru_cache(maxsize=None)
def field_checked_class(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """A copy of *model_class* with the same fields and field validators and no model validators.
    Cached per class (see the module docstring)."""
    field_validators = {}
    for attr, decorator in model_class.__pydantic_decorators__.field_validators.items():
        info = decorator.info
        field_validators[attr] = field_validator(*info.fields, mode=info.mode,
                                                 check_fields=info.check_fields)(decorator.func)
    fields = {name: (info.annotation, info) for name, info in model_class.model_fields.items()}
    return create_model(model_class.__name__, __module__=model_class.__module__,
                        __config__=ConfigDict(**model_class.model_config),
                        __validators__=field_validators, **fields)


def is_pydantic_model(model_class: Any) -> bool:
    """True when *model_class* is a pydantic ``BaseModel`` subclass. A parameters class that is not one
    (a plain class, a test stub) has no fields or validators to check against."""
    return isinstance(model_class, type) and issubclass(model_class, BaseModel)


def _unfilled_missing(error: dict, given: dict[str, Any]) -> bool:
    """A "field required" error for a field that was not given: the caller fills it, so it is not a failure."""
    loc = error.get("loc") or ()
    return error.get("type") == "missing" and bool(loc) and loc[0] not in given


def bound_fields(model_class: Type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
    """The given *values* checked as the model checks its fields: each value through its type, the ``Field``
    constraints and the field's ``field_validator``s, with every field of the model present (the given
    values, and the defaults of the others) so a field validator that reads ``info.data`` sees what it sees
    in a full construction. Model validators do not run.

    Only names that are fields are kept. Every failing value is reported in one ``pydantic.ValidationError``,
    as the full constructor reports them. A field that was not given and is required is not a failure here
    (the caller fills it), so its "missing" error is ignored. A class that is not a pydantic model has no
    fields to check, so the values are returned as given.
    """
    if not is_pydantic_model(model_class):
        return dict(values)
    given = {name: value for name, value in values.items() if name in model_class.model_fields}
    if not given:
        return {}
    checker = field_checked_class(model_class)
    try:
        checked = checker.model_validate(given)
    except ValidationError as exc:
        errors = exc.errors()
        kept = [error for error in errors if not _unfilled_missing(error, given)]
        if len(kept) == len(errors):
            raise
        if kept:
            raise ValidationError.from_exception_data(
                model_class.__name__,
                [{key: error[key] for key in ("type", "loc", "input", "ctx") if key in error} for error in kept],
            ) from None
        # Only fields that were not given are missing, so every given value checked. Their values are
        # rebuilt in field order, each validated with the fields before it in place (as a construction does).
        shell = checker.model_construct()
        for name in model_class.model_fields:
            if name in given:
                checker.__pydantic_validator__.validate_assignment(shell, name, given[name])
        return {name: getattr(shell, name) for name in given}
    return {name: getattr(checked, name) for name in given}


def fields_validated(model_class: Type[BaseModel], data: Any) -> BaseModel:
    """A *model_class* instance built from *data* with each field checked as ``bound_fields`` checks it
    (type, constraints, the field's own validators) and no model validator run. For data whose
    model validators already ran where it was produced, such as an archived run's output (the typed
    fill). Raises ``pydantic.ValidationError`` as the plain constructor does."""
    checked = field_checked_class(model_class).model_validate(data)
    return model_class.model_construct(**{name: getattr(checked, name) for name in model_class.model_fields})


@dataclasses.dataclass
class ValidationTrace:
    """What ``validate_input`` did before it stopped: the instance its model validators ran on, and the field
    values that instance held before they ran. The validators' writes are the difference (``written``)."""

    instance: Any = None
    before: dict = dataclasses.field(default_factory=dict)

    def written(self) -> dict[str, Any]:
        """The fields the model validators changed on the instance, ``{name: value}``. Empty before they ran."""
        if self.instance is None:
            return {}
        now = self.instance.__dict__
        return {name: now[name] for name, value in self.before.items() if name in now and now[name] != value}


def validate_input(model_class: Type[BaseModel], data: Any, context: Any,
                   trace: ValidationTrace | None = None) -> BaseModel:
    """Validate *data* as an ``Input``: its fields as ``field_checked_class`` checks them, then its model
    validators in declaration order, each called with *context* as ``info.context``. Each validator
    gets the instance the previous one returned, so a write it makes to ``self`` is kept even when it
    then raises (*trace* records that instance and the values it held before the validators ran).

    Raises ``pydantic.ValidationError`` for a field's check, and whatever a model validator raises, unchanged
    (a ``fastworkflow.Refused`` is not wrapped). An Input with a model validator that is not ``mode="after"``
    is validated by pydantic in one call instead; then *trace* stays empty.
    """
    validators = list(model_class.__pydantic_decorators__.model_validators.values())
    if any(dec.info.mode != "after" for dec in validators):
        return validation_class(model_class).model_validate(data, context=context)
    checked = field_checked_class(model_class).model_validate(data, context=context)
    instance = validation_class(model_class).model_construct()
    for name in model_class.model_fields:
        set_field_unvalidated(instance, name, getattr(checked, name))
    if trace is not None:
        trace.instance = instance
        trace.before = {name: getattr(instance, name) for name in model_class.model_fields}
    info = SimpleNamespace(context=context, data={}, field_name=None, config=None)
    for dec in validators:
        if len(inspect.signature(dec.func).parameters) >= 2:
            instance = dec.func(instance, info)
        else:
            instance = dec.func(instance)
        if trace is not None:
            trace.instance = instance
    return instance


_REQUEST_REMOVED = ("'request' is removed from the validation context: read info.context['command_text']")


class ValidationContext(dict):
    """The validation context ``check_input`` passes to an ``Input``'s model validators: a dict with
    ``command_text``, ``workflow`` and ``command``. Its behaviour is a dict's, except for the removed key
    ``"request"`` (renamed ``"command_text"``, fix-o8j3): reading it raises a KeyError that names the
    replacement, instead of the bare ``KeyError: 'request'``.

    ``get`` is overridden too, because ``dict.get`` returns None for a missing key and would hide the
    removal from a validator written as ``info.context.get("request")``: that validator would then
    quietly see no command text.
    """

    def __missing__(self, key):
        if key == "request":
            raise KeyError(_REQUEST_REMOVED)
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "request":
            raise KeyError(_REQUEST_REMOVED)
        return super().get(key, default)


_CONTEXT_GUARD = "__fw_validation_context_guard__"


def _needs_context_message(inner, model_name: str, validator_name: str, where: str):
    """Wrap one ``mode="after"`` model validator (see ``guard_input_validators``)."""

    @functools.wraps(inner)
    def guarded(self, info):
        try:
            return inner(self, info)
        except (TypeError, AttributeError) as exc:
            if info.context is not None:
                raise
            raise ValueError(
                f"{where}: {model_name}.{validator_name} needs the validation context "
                "(command_text, workflow, command): validate through the framework, or build the Input "
                "with Signature.Input.model_construct(...) in tests "
                f"(it raised {type(exc).__name__}: {exc})") from exc

    setattr(guarded, _CONTEXT_GUARD, True)
    return guarded


def guard_input_validators(model_class: Type[BaseModel], where: str) -> None:
    """Load time, for a command's ``Input``: an ``after`` model validator that reads ``info.context`` and
    runs with no validation context (direct ``Signature.Input(...)`` in author or test code) raises a
    ValueError that names the validator and the fix, instead of ``'NoneType' object is not subscriptable``.

    With a context the validator runs unchanged and any exception is re-raised as it was. Applied once per
    class: a guarded validator is marked and skipped. A validator with no ``info`` parameter is left alone.
    """
    decorators = getattr(model_class, "__pydantic_decorators__", None)
    if decorators is None:
        return
    originals = dict(decorators.model_validators)
    replaced = False
    for name, dec in originals.items():
        if dec.info.mode != "after" or getattr(dec.func, _CONTEXT_GUARD, False):
            continue
        if len(inspect.signature(dec.func).parameters) < 2:
            continue
        decorators.model_validators[name] = dataclasses.replace(
            dec, func=_needs_context_message(dec.func, model_class.__name__, name, where))
        replaced = True
    if not replaced:
        return
    try:
        model_class.model_rebuild(force=True)
    except Exception:  # noqa: BLE001 - leave the Input as it was; the guard is an aid, not a gate
        decorators.model_validators.clear()
        decorators.model_validators.update(originals)
        model_class.model_rebuild(force=True)
