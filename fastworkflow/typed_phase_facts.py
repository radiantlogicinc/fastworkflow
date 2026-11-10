"""Phase outputs typed with commands' Signature models (design 3.12)."""
from __future__ import annotations

import inspect
import os
import re
import types
import typing
from dataclasses import dataclass
from types import UnionType
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from fastworkflow.utils.input_validation import fields_validated
from fastworkflow.observation_offloading.facts import (
    FactsError,
    MissingFactError,
    PAGING_PARAMETERS,
    _command_matches,
    _join_list,
    _page_key,
    _scalar_of_pages,
)
from fastworkflow.field_options import FROM_COMMAND_OUTPUT
from fastworkflow.planning_phases import FULL_LIST_PHRASES
from fastworkflow.utils.python_utils import type_text


@dataclass(frozen=True)
class TypedFieldSpec:
    """One field's command source for a typed facts model (or the whole model)."""

    field: str
    command: str
    role: str  # ``output`` or ``input``
    required: bool


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def union_members(annotation: Any) -> tuple[type[BaseModel], ...] | None:
    """Member models when *annotation* is a union of pydantic models."""
    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return None
    members = []
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if inspect.isclass(arg) and issubclass(arg, BaseModel):
            members.append(arg)
        else:
            return None
    return tuple(members) if members else None


def _type_label(annotation: Any) -> str:
    # the same text as a plain class name (no module), Optional included (python_utils.type_text)
    return type_text(annotation)


def carries_copy_field(info: Any) -> bool:
    """Whether a facts field still uses a copy edge (``fact_from`` / a plain ``"from"`` key, now retired)."""
    extra = info.json_schema_extra
    return isinstance(extra, Mapping) and "from" in extra


#: The load error's tail for a field that still copies (Phase 15, design 3.12).
COPIES_NOT_READ = "copies are no longer read (fact_from and \"from\" are removed)"


def _not_typed_text(model: type[BaseModel], name: str, info: Any) -> str:
    """The 3.12 load error for a facts field that is not typed with a command's Output or Input."""
    text = (f"{model.__name__}.{name} is {_type_label(info.annotation)}; a facts field is typed "
            f"with a command's Output or Input. Compute other values in a property or method "
            f"of {model.__name__}")
    if carries_copy_field(info):
        text += f"; {COPIES_NOT_READ}"
    return text


def handles_typed_output(model: Any) -> bool:
    """Whether :func:`fill_typed_phase_output` fills *model*: a command's Output, a union of
    them, or a facts model (design 3.12). Copy edges are not read (Phase 15)."""
    if union_members(model) is not None:
        return True
    return inspect.isclass(model) and issubclass(model, BaseModel)


def _role_for_model(model_cls: type[BaseModel], command: str,
                    signatures: Mapping[str, Any]) -> str:
    signature = signatures.get(command)
    if signature is not None:
        if model_cls is getattr(signature, "Output", None):
            return "output"
        if model_cls is getattr(signature, "Input", None):
            return "input"
    # The run-time fill passes no signatures, and a command module can be loaded under more than one
    # name, so a command's ``Signature.Input`` is known by its class name: it fills from the run's input.
    return "input" if model_cls.__name__ == "Input" else "output"


def _facts_field_type_errors(model: type[BaseModel], model_to_command: Mapping[type, str]) -> list[str]:
    """Load errors for fields of a facts model (not a command's whole Output/Input)."""
    errors: list[str] = []
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
        if FROM_COMMAND_OUTPUT in extra:
            # ``from_command_output`` on facts fields is validated in ``resolve_field_sources`` (3.3).
            continue
        inner = _strip_optional(info.annotation)
        optional_none = (
            get_origin(info.annotation) in (Union, UnionType) and type(None) in get_args(info.annotation)
        )
        if inner in model_to_command:
            continue
        if optional_none and inner is type(None):
            continue
        errors.append(_not_typed_text(model, name, info))
    return errors


def typed_field_specs(model: type[BaseModel], model_to_command: Mapping[type, str],
                      signatures: Mapping[str, Any]) -> list[TypedFieldSpec]:
    """Every typed field of *model* (empty when the whole model is one command's Output)."""
    model_to_command = _extended_mapping(model, signatures, model_to_command)
    if model in model_to_command:
        return [TypedFieldSpec("", model_to_command[model], "output", True)]
    specs: list[TypedFieldSpec] = []
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
        if FROM_COMMAND_OUTPUT in extra:
            continue
        inner = _strip_optional(info.annotation)
        optional = not info.is_required() or (
            get_origin(info.annotation) in (Union, UnionType) and type(None) in get_args(info.annotation)
        )
        if inner in model_to_command:
            cmd = model_to_command[inner]
            specs.append(TypedFieldSpec(
                name, cmd, _role_for_model(inner, cmd, signatures),
                not optional,
            ))
            continue
        if optional and inner is type(None):
            continue
        raise ValueError(_not_typed_text(model, name, info))
    return specs


def required_commands_for_output(model: Any, model_to_command: Mapping[type, str],
                                 signatures: Mapping[str, Any]) -> set[str]:
    """Commands a phase's steps must name (union: any one member per check is done by caller)."""
    members = union_members(model)
    if members is not None:
        return {model_to_command[m] for m in members if m in model_to_command}
    if not (inspect.isclass(model) and issubclass(model, BaseModel)):
        return set()
    if model in model_to_command:
        return {model_to_command[model]}
    out: set[str] = set()
    for spec in typed_field_specs(model, model_to_command, signatures):
        if spec.required:
            out.add(spec.command)
    return out


def check_typed_facts_model(
    file_path: str,
    model: Any,
    model_to_command: Mapping[type, str],
    signatures: Mapping[str, Any],
) -> list[str]:
    """Load errors for a phase output type (3.12); empty when valid or copy-based."""
    mapping = _extended_mapping(model, signatures, model_to_command)
    members = union_members(model)
    if members is not None:
        for member in members:
            if member not in mapping:
                return [
                    f"phase output {member.__name__} is not a loaded command's "
                    "Signature.Output or Signature.Input"
                ]
        return []
    if not (inspect.isclass(model) and issubclass(model, BaseModel)):
        return []
    if model in mapping:
        return []
    return _facts_field_type_errors(model, mapping)


def check_phase_steps_name_commands(
    file_path: str,
    phase_name: str,
    step_commands: Iterable[str],
    model: Any,
    model_to_command: Mapping[type, str],
    signatures: Mapping[str, Any],
) -> list[str]:
    """Load error when *step_commands* never names a command *model* needs."""
    plain = {c.split("/")[-1] for c in step_commands}
    qualified = set(step_commands)
    members = union_members(model)
    if members is not None:
        needed = {model_to_command[m] for m in members if m in model_to_command}
        if not needed:
            return []
        if any(_step_names_command(plain, qualified, cmd) for cmd in needed):
            return []
        cmd = sorted(needed)[0]
        return [
            f"phase {phase_name!r} hands on {cmd.split('/')[-1]}'s Output, but "
            f"none of its steps runs {cmd.split('/')[-1]}"
        ]
    try:
        required = required_commands_for_output(model, model_to_command, signatures)
    except ValueError as error:
        return [str(error)]
    errors: list[str] = []
    for cmd in sorted(required):
        if _step_names_command(plain, qualified, cmd):
            continue
        plain_cmd = cmd.split("/")[-1]
        errors.append(
            f"phase {phase_name!r} hands on {plain_cmd}'s Output, but none of its steps runs "
            f"{plain_cmd}"
        )
    return errors


def _step_names_command(plain: set[str], qualified: set[str], command: str) -> bool:
    if command in qualified:
        return True
    stem = command.split("/")[-1]
    return stem in plain


def _command_qname_from_class(cls: type) -> str | None:
    """Qualified command name from a ``Signature`` nested ``Input`` or ``Output`` class."""
    if cls.__name__ not in ("Input", "Output"):
        return None
    mod = getattr(cls, "__module__", "")
    framework = "fastworkflow._framework_commands."
    if mod.startswith(framework):
        # A framework command (global scope, no context): its routing name is its plain module name.
        name = mod[len(framework):]
        return name if name and "." not in name else None
    marker = "._commands."
    if marker not in mod:
        return None
    tail = mod.split(marker, 1)[1]
    parts = tail.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return None


def _model_to_command(signatures: Mapping[str, Any]) -> dict[type, str]:
    out: dict[type, str] = {}
    for qname, sig in signatures.items():
        for attr in ("Input", "Output"):
            cls = getattr(sig, attr, None)
            if inspect.isclass(cls) and issubclass(cls, BaseModel):
                out[cls] = qname
    return out


def _extended_mapping(
    model: Any,
    signatures: Mapping[str, Any],
    model_to_command: Optional[Mapping[type, str]],
) -> dict[type, str]:
    mapping = dict(model_to_command or _model_to_command(signatures))
    candidates: list[type] = []
    members = union_members(model)
    if members:
        candidates.extend(members)
    elif inspect.isclass(model) and issubclass(model, BaseModel):
        candidates.append(model)
        for info in model.model_fields.values():
            inner = _strip_optional(info.annotation)
            if inspect.isclass(inner) and issubclass(inner, BaseModel):
                candidates.append(inner)
    for cls in candidates:
        if cls in mapping:
            continue
        qname = _command_qname_from_class(cls) or _same_definition_qname(cls, mapping)
        if qname is not None:
            mapping[cls] = qname
    return mapping


def _same_definition_qname(cls: type, mapping: Mapping[type, str]) -> str | None:
    """The command of a class that is the same definition as one in *mapping*: the same file and
    qualified name, loaded under another module name (a framework command, which the workflow's
    routing and an import from another module load twice)."""
    try:
        path = os.path.realpath(inspect.getfile(cls))
    except (TypeError, OSError):
        return None
    for known, qname in mapping.items():
        if known.__qualname__ != cls.__qualname__ or known.__module__ == cls.__module__:
            continue
        try:
            if os.path.realpath(inspect.getfile(known)) == path:
                return qname
        except (TypeError, OSError):
            continue
    return None


def _matching_runs(runs: Sequence[Mapping[str, Any]], command: str, *,
                   context: Optional[str]) -> list[Mapping[str, Any]]:
    return [
        run for run in runs
        if run.get("success") and _command_matches(command, str(run.get("command") or ""))
        and (context is None or run.get("context_clause") == context)
    ]


def _output_blob_from_runs(runs: Sequence[Mapping[str, Any]], command: str, *,
                           context: Optional[str], full_list: bool) -> dict[str, Any] | None:
    """Merged Output dict from *runs* of *command* (pages joined when *full_list*)."""
    matching = _matching_runs(runs, command, context=context)
    if not matching:
        return None
    last = matching[-1]
    output = last.get("output")
    if not isinstance(output, Mapping):
        return None
    if not full_list or len(matching) <= 1:
        return dict(output)
    # The pages' keys, every one any page gave: a key an earlier page carries and the last
    # page omits is still in the facts. Its kind (list or scalar) is that of the latest page
    # that gives it a value.
    kinds: dict[str, Any] = {}
    for run in matching:
        page = run.get("output")
        if not isinstance(page, Mapping):
            continue
        for key, val in page.items():
            if val is not None or key not in kinds:
                kinds[key] = val
    joined: dict[str, Any] = {}
    for key, val in kinds.items():
        if isinstance(val, (list, tuple)):
            keyed = [
                (*_page_key(r.get("input"), PAGING_PARAMETERS, None), r["output"][key], r["output"])
                for r in matching
                if isinstance(r.get("output"), Mapping) and isinstance(r["output"].get(key), (list, tuple))
            ]
            joined[key] = _join_list(keyed) if keyed else list(val)
        else:
            keyed = [
                (*_page_key(r.get("input"), PAGING_PARAMETERS, None), r["output"].get(key), r["output"])
                for r in matching if isinstance(r.get("output"), Mapping)
            ]
            joined[key] = _scalar_of_pages(keyed)
    return joined


def _input_blob_from_runs(runs: Sequence[Mapping[str, Any]], command: str, *,
                          context: Optional[str]) -> dict[str, Any] | None:
    matching = _matching_runs(runs, command, context=context)
    if not matching:
        return None
    run_input = matching[-1].get("input")
    return dict(run_input) if isinstance(run_input, Mapping) else None


def _plan_uses_full_list(plan_text: str, command: str) -> bool:
    stem = command.split("/")[-1]
    for line in str(plan_text or "").splitlines():
        text = re.sub(r"^\d+[.)]\s*", "", line.strip().lower())
        if not text.startswith(stem.lower()):
            continue
        if any(phrase in text for phrase in FULL_LIST_PHRASES):
            return True
    return False


def fill_typed_phase_output(
    model: Any,
    runs: Sequence[Mapping[str, Any]],
    *,
    context: Optional[str] = None,
    plan_text: Optional[str] = None,
    signatures: Optional[Mapping[str, Any]] = None,
    model_to_command: Optional[Mapping[type, str]] = None,
) -> BaseModel:
    """Fill a typed phase output from archived *runs* (3.12)."""
    signatures = signatures or {}
    mapping = _extended_mapping(model, signatures, model_to_command)
    members = union_members(model)
    if members is not None:
        for run in reversed(list(runs)):
            if not run.get("success"):
                continue
            if context is not None and run.get("context_clause") != context:
                continue
            output = run.get("output")
            if not isinstance(output, Mapping):
                continue
            cmd = str(run.get("command") or "")
            for member in members:
                if member not in mapping:
                    continue
                if not _command_matches(mapping[member], cmd):
                    continue
                full_list = _plan_uses_full_list(plan_text or "", mapping[member])
                blob = _output_blob_from_runs(
                    runs, mapping[member], context=context, full_list=full_list)
                if blob is None:
                    continue
                try:
                    return fields_validated(member, blob)
                except ValidationError as error:
                    raise FactsError(
                        f"{member.__name__}: the values found do not fit the model: {error}"
                    ) from error
        raise MissingFactError(
            f"no successful run fills the phase output ({' | '.join(m.__name__ for m in members)})"
        )
    if not (inspect.isclass(model) and issubclass(model, BaseModel)):
        raise FactsError(f"cannot fill phase output {model!r}")
    specs = typed_field_specs(model, mapping, signatures)
    if len(specs) == 1 and specs[0].field == "":
        spec = specs[0]
        full_list = _plan_uses_full_list(plan_text or "", spec.command)
        if spec.role == "output":
            blob = _output_blob_from_runs(runs, spec.command, context=context, full_list=full_list)
        else:
            blob = _input_blob_from_runs(runs, spec.command, context=context)
        if blob is None:
            raise MissingFactError(
                f"{model.__name__}: no successful run of {spec.command!r} fills the phase output"
            )
        try:
            return fields_validated(model, blob)
        except ValidationError as error:
            raise FactsError(
                f"{model.__name__}: the values found do not fit the model: {error}"
            ) from error
    values: dict[str, Any] = {}
    for spec in specs:
        full_list = _plan_uses_full_list(plan_text or "", spec.command)
        if spec.role == "output":
            blob = _output_blob_from_runs(runs, spec.command, context=context, full_list=full_list)
        else:
            blob = _input_blob_from_runs(runs, spec.command, context=context)
        if blob is None:
            if spec.required:
                raise MissingFactError(
                    f"{model.__name__}.{spec.field}: no successful run of {spec.command!r} "
                    f"fills {spec.field!r}"
                )
            values[spec.field] = None
            continue
        inner = _strip_optional(model.model_fields[spec.field].annotation)
        try:
            values[spec.field] = fields_validated(inner, blob)
        except ValidationError as error:
            raise FactsError(
                f"{model.__name__}.{spec.field}: the values found do not fit: {error}"
            ) from error
    # Each value was checked above (fields_validated); the outer model is built from them without
    # re-validating its Input-typed fields, whose model validators need the command context.
    return model.model_construct(**values)


def validate_phase_output_value(model: Any, value: Any) -> BaseModel:
    """Validate one phase output dict or model for ``__result__`` binding."""
    members = union_members(model)
    if members is not None:
        return TypeAdapter(model).validate_python(value)
    if inspect.isclass(model) and issubclass(model, BaseModel):
        return model.model_validate(value)
    raise TypeError(type(model).__name__)


def preview_output_type(model: Any, model_to_command: Mapping[type, str]) -> str:
    """Phases-table text for a phase output type (3.7)."""
    members = union_members(model)
    if members is not None:
        parts = []
        for member in members:
            cmd = model_to_command.get(member, member.__name__)
            parts.append(f"`{cmd}`'s `Output`")
        return " | ".join(parts)
    if model in model_to_command:
        return f"`{model_to_command[model]}`'s `Output`"
    if inspect.isclass(model) and issubclass(model, BaseModel):
        return model.__name__
    return str(model)


def preview_fact_rows(model: type[BaseModel], model_to_command: Mapping[type, str]) -> list[tuple[str, str, str]]:
    """Facts-section rows: ``(field, type text, from text)``."""
    rows: list[tuple[str, str, str]] = []
    for name, info in model.model_fields.items():
        if carries_copy_field(info):
            continue
        inner = _strip_optional(info.annotation)
        cmd = model_to_command.get(inner)
        if cmd is None:
            continue
        role = "Output" if inner.__name__ == "Output" or inner in model_to_command else "Input"
        # Prefer Output label when class is the command's Output
        origin_text = f"`{cmd}`'s `{inner.__name__}`"
        rows.append((name, _type_label(info.annotation), origin_text))
    return rows


__all__ = [
    "TypedFieldSpec",
    "carries_copy_field",
    "check_phase_steps_name_commands",
    "check_typed_facts_model",
    "fill_typed_phase_output",
    "handles_typed_output",
    "preview_fact_rows",
    "preview_output_type",
    "required_commands_for_output",
    "typed_field_specs",
    "union_members",
    "validate_phase_output_value",
]
