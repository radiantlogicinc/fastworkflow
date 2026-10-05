import contextlib
import sys
import re
import json
import ast
from typing import Dict, List, Optional, Union, get_origin, get_args

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import ModuleType, tracing
from fastworkflow.observability.decision_signals import (
    DecisionUncertainty,
    UncertaintySignal,
    slot_binding_source,
)

from fastworkflow.utils.signatures import InputForParamExtraction


INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
INVALID = fastworkflow.get_env_var("INVALID")
PARAMETER_EXTRACTION_ERROR_MSG = None

# Versions of the mechanisms that bind a parameter, one per member of
# ``SLOT_BINDING_SOURCES``. ``signal_version`` identifies the producer of a value,
# and for an enumerated provenance that means the extractor itself: a change to the
# XML tag grammar or to the sentinel-driven merge changes what "xml_regex" or
# "stored_merge" is evidence of, without changing the recorded enum.
#
# ``llm`` is resolved at call time instead, because the model string is the thing
# that identifies that extractor and it comes from configuration.
_STATIC_SLOT_BINDING_SIGNAL_VERSIONS = {
    "stored_merge": "param-extraction/stored-merge/1",
    "xml_regex": "param-extraction/xml-regex/1",
    "db_lookup": "param-extraction/db-lookup/1",
}

# Resolved lazily and once, the same way PARAMETER_EXTRACTION_ERROR_MSG above is:
# env access on every extraction logs a warning per turn when the var is unset.
LLM_PARAM_EXTRACTION_MODEL = None


def _slot_binding_signal_version(extraction_method: str) -> str:
    """Which version of which mechanism produced a binding-source signal.

    FW-REQ-021 clause 13 asks for thresholds to be re-validated when a model,
    prompt, or artifact that produces a signal changes; naming the LLM here is what
    makes an ``llm`` binding from one model distinguishable from another's.
    """
    global LLM_PARAM_EXTRACTION_MODEL
    if static_version := _STATIC_SLOT_BINDING_SIGNAL_VERSIONS.get(extraction_method):
        return static_version
    if not LLM_PARAM_EXTRACTION_MODEL:
        LLM_PARAM_EXTRACTION_MODEL = (
            fastworkflow.get_env_var("LLM_PARAM_EXTRACTION") or "unknown"
        )
    return f"param-extraction/llm/{LLM_PARAM_EXTRACTION_MODEL}"


def slot_binding_uncertainty(diagnostics: dict) -> Optional[DecisionUncertainty]:
    """The §6.6.1 record for the slot-binding decision *diagnostics* describes.

    A pure function of the facts ``_extract_impl`` recorded, and one-way: it reads
    the capture bag and writes nothing back, so extraction cannot come to depend on
    what is being measured about it (FW-REQ-021 P0 / arch §17.3 — capture only).

    Returns None when no binding decision was reached: a command with no parameters
    class binds no slots, and an extraction that raised before choosing a mechanism
    made no decision to characterise. Neither is an uninstrumented decision, so
    neither gets a record saying it was one.
    """
    extraction_method = diagnostics.get("extraction_method")
    if extraction_method is None:
        return None

    signals: list[UncertaintySignal] = [
        slot_binding_source(
            extraction_method,
            signal_version=_slot_binding_signal_version(extraction_method),
        )
    ]

    db_lookup_events = diagnostics.get("db_lookup") or []
    # `applied` alone means the catalogue agreed with the extracted value; only a
    # rewrite makes db_lookup the thing that bound the slot, which is the sense in
    # which SLOT_BINDING_SOURCES admits it as a source.
    if any(event.get("corrected") for event in db_lookup_events):
        signals.append(
            slot_binding_source(
                "db_lookup",
                signal_version=_slot_binding_signal_version("db_lookup"),
            )
        )

    # The extractor produces one binding per slot and enumerates no alternatives to
    # it; a db_lookup that rejected a value did enumerate some, and each suggestion
    # is a value the slot could have taken instead.
    candidate_count = 1 + sum(
        len(event.get("suggestions") or []) for event in db_lookup_events
    )

    # Requirements §4.14: uncertainty is reducible when evidence gathered inside the
    # task can lower it. A missing or invalid field is re-prompted and the next
    # round fills it, which is exactly that. Anything else is left as None — nobody
    # assessed it, which is not the same as "no".
    unresolved = diagnostics.get("missing_fields") or diagnostics.get("invalid_fields")
    return DecisionUncertainty(
        decision_kind="slot-binding",
        signals=tuple(signals),
        candidate_count=candidate_count,
        reducible=True if unresolved else None,
    )


def _recorded_slot_binding_uncertainty(span, diagnostics: dict) -> Optional[dict]:
    """``slot_binding_uncertainty`` in span-attribute form, never raising.

    Skipped when *span* is None: ``start_span`` declines without a sink or an open
    turn and ``end_span`` then discards its attributes, so assembling a record
    nobody will store is pure overhead against this slice's declared budget.

    ``tracing`` wraps every one of its own calls so a broken recorder degrades to
    a log line rather than a failed turn; this assembly runs just outside that
    guard, and capture that can lose a user's turn is the behavior change Phase 0
    exists to avoid.

    None also means no slot-binding decision occurred, which is a real outcome — so
    a capture failure reports itself instead of collapsing into the same value. The
    pure function above still raises, which is what the tests exercise.
    """
    if span is None:
        return None
    try:
        uncertainty = slot_binding_uncertainty(diagnostics)
    except Exception as exc:
        logger.warning(f"slot-binding uncertainty capture failed: {exc!r}")
        return {"capture_error": type(exc).__name__}
    return None if uncertainty is None else uncertainty.model_dump(mode="json")


def _unwrap_optional(field_type):
    """
    Unwrap ``Optional[X]`` / ``Union[X, None]`` to its underlying type.

    Returns ``(inner_type, origin)`` where ``origin`` is ``get_origin`` of the
    unwrapped type (e.g. ``list`` for ``Optional[List[str]]``). Non-optional
    types are returned unchanged.
    """
    origin = get_origin(field_type)
    if origin is Union:
        if non_none_types := [
            t for t in get_args(field_type) if t is not type(None)
        ]:
            field_type = non_none_types[0]
            origin = get_origin(field_type)
    return field_type, origin


class ParameterExtraction:
    class Output(BaseModel):
        parameters_are_valid: bool
        cmd_parameters: Optional[BaseModel] = None
        error_msg: Optional[str] = None
        suggestions: Optional[Dict[str, List[str]]] = None

    def __init__(self, cme_workflow: fastworkflow.Workflow, app_workflow: fastworkflow.Workflow, command_name: str, command: str):
        self.cme_workflow = cme_workflow
        self.app_workflow = app_workflow
        self.command_name = command_name
        self.command = command

    def extract(self) -> "ParameterExtraction.Output":
        """Extract, wrapped in a ``fw.nlu.param_extraction`` span (D3 as
        amended). The span records the extraction method (stored-merge /
        xml-regex / llm), whether this is a NOT_FOUND retry round, and the
        STRUCTURED validation outcome from ``validate_parameters`` —
        missing/invalid fields, per-field db_lookup events, and the
        validate_extracted_parameters verdict. A failed extraction is a
        conversational state, not a span error: status stays ok and
        ``parameters_valid`` carries the signal; only an exception marks the
        span error. Emission never affects extraction (no-op without a host).

        It also carries the §6.6.1 ``DecisionUncertainty`` for the slot-binding
        decision, assembled by ``slot_binding_uncertainty`` from those same
        diagnostics. That record is written to the span and read by nothing else
        (FW-REQ-021 P0: representation only).
        """
        host = tracing.current_host()
        span = tracing.start_span(
            host,
            tracing.SPAN_NLU_PARAM_EXTRACTION,
            command_name=self.command_name,
            attributes={"command_name": self.command_name},
        )
        diagnostics: dict = {}
        try:
            output = self._extract_impl(diagnostics)
        except BaseException:
            # No DecisionUncertainty here: the extraction did not complete, so
            # there is no binding decision to characterise. The raw diagnostics
            # gathered so far still go out.
            tracing.end_span(
                host, span, status=tracing.STATUS_ERROR, attributes=diagnostics
            )
            raise
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK,
            attributes={
                **diagnostics,
                "decision_uncertainty": _recorded_slot_binding_uncertainty(
                    span, diagnostics
                ),
                "parameters_valid": output.parameters_are_valid,
            },
        )
        return output

    def _extract_impl(self, diagnostics: dict) -> "ParameterExtraction.Output":
        app_workflow_folderpath = self.app_workflow.folderpath
        app_command_routing_definition = fastworkflow.RoutingRegistry.get_definition(app_workflow_folderpath)

        command_parameters_class = (
            app_command_routing_definition.get_command_class(
                self.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            return self.Output(parameters_are_valid=True)

        stored_params = self._get_stored_parameters(self.cme_workflow)

        self.command = self.command.replace(self.command_name, "").strip()

        input_for_param_extraction = InputForParamExtraction.create(
            self.app_workflow, self.command_name, 
            self.command)

        # If we have missing fields (in parameter extraction error state), try to apply the command directly
        diagnostics["retry_round"] = bool(stored_params)
        # WHICH round, not just whether this is one (fix-8ko2). The flag alone
        # left a consumer counting extraction spans and hoping the count was
        # the ordinal, which it is not when a span was dropped.
        #
        # The round can be genuinely unknown, and then the attribute is OMITTED
        # rather than sent as a 0 that would read as "first attempt". See
        # _get_stored_round for when that happens; the short version is that
        # this module is not the only writer of the state it counts.
        round_ordinal = self._get_stored_round(self.cme_workflow)
        if round_ordinal is not None:
            diagnostics["retry_round_ordinal"] = round_ordinal
        if stored_params:
            new_params = self._extract_and_merge_missing_parameters(stored_params, self.command)
            diagnostics["extraction_method"] = "stored_merge"
        else:
            # Check if we're in agentic mode (not assistant mode command)
            is_agentic_mode = (
                "is_assistant_mode_command" not in self.cme_workflow.context
                and "run_as_agent" in self.app_workflow.context
                and self.app_workflow.context["run_as_agent"]
            )

            if is_agentic_mode:
                # Try regex-based extraction first in agentic mode
                new_params = self._extract_parameters_from_xml(self.command, command_parameters_class)
                diagnostics["extraction_method"] = "xml_regex"

                # If regex extraction fails, fall back to LLM-based extraction
                if new_params is None:
                    new_params = input_for_param_extraction.extract_parameters(
                        command_parameters_class,
                        self.command_name,
                        app_workflow_folderpath)
                    diagnostics["extraction_method"] = "llm"
            else:
                # Use LLM-based extraction for assistant mode
                new_params = input_for_param_extraction.extract_parameters(
                    command_parameters_class,
                    self.command_name,
                    app_workflow_folderpath)
                diagnostics["extraction_method"] = "llm"

        is_valid, error_msg, suggestions, missing_invalid_fields = \
            input_for_param_extraction.validate_parameters(
            self.app_workflow, self.command_name, new_params,
            diagnostics=diagnostics
        )

        # Set all the missing and invalid fields to appropriate sentinel values before storing
        current_values = {
            field_name: getattr(new_params, field_name, None)
            for field_name in list(type(new_params).model_fields.keys())
        }
        for field_name in missing_invalid_fields:
            if field_name in current_values:
                # Determine appropriate sentinel value based on field type
                field_info = type(new_params).model_fields[field_name]
                _, origin = _unwrap_optional(field_info.annotation)

                # Use empty list for list fields, NOT_FOUND for others
                if origin in (list, List):
                    current_values[field_name] = []
                else:
                    current_values[field_name] = NOT_FOUND
        # Reconstruct the model instance without validation
        new_params = new_params.__class__.model_construct(**current_values)

        self._store_parameters(self.cme_workflow, new_params)

        if not is_valid:
            if params_str := self._format_parameters_for_display(new_params):
                error_msg = f"Extracted parameters so far:\n{params_str}\n\n{error_msg}"

            if "run_as_agent" not in self.app_workflow.context:
                error_msg += "\nEnter 'abort' to get out of this error state and/or execute a different command."
                error_msg += "\nEnter 'you misunderstood' if the wrong command was executed."
            else:
                error_msg += "\nCheck your command name if the wrong command was executed."
            return self.Output(
                parameters_are_valid=False,
                error_msg=error_msg,
                cmd_parameters=new_params,
                suggestions=suggestions)

        self._clear_parameters(self.cme_workflow)
        return self.Output(
            parameters_are_valid=True,
            cmd_parameters=new_params)

    @staticmethod
    def _get_stored_parameters(cme_workflow: fastworkflow.Workflow):
        return cme_workflow.context.get("stored_parameters")

    @staticmethod
    def _get_stored_round(cme_workflow: fastworkflow.Workflow) -> Optional[int]:
        """How many extraction attempts already ran for the stored parameters.

        This is the 0-based ordinal of the attempt about to run: 0 is the first
        attempt, 1 is the first retry. `None` means genuinely unknown, and the
        caller omits the attribute rather than sending a 0 that would read as
        "first attempt".

        Unknown is not hypothetical. `stored_parameters` has writers this module
        does not control, and none of them know about the counter:

        - `Workflow.end_command_processing` deletes `stored_parameters` and
          leaves the counter behind. A stale counter therefore cannot be
          trusted to describe whatever is in the context now -- so the absence
          of stored parameters is the authority, and it means round 0.
        - `WorkflowExecutionContext._serialize_cme_continuation` persists
          `stored_parameters` and `_apply_cme_continuation` rebuilds it, but
          neither carries the counter. A rehydrated session therefore has real
          stored parameters whose round nobody recorded. That is unknown, and
          claiming 0 there would report a resumed third attempt as a first one.

        Both writers live in files this change may not edit (see fix-7gp9 for
        the persistence follow-up), so the rule is read-side: the counter is
        believed only when it is present, an integer, positive, and sitting
        beside the stored parameters it claims to count.
        """
        context = cme_workflow.context
        if context.get("stored_parameters") is None:
            # No error state for a round to be a round OF. Whatever an orphaned
            # counter says, the next extraction is a first attempt.
            return 0
        round_index = context.get("stored_parameter_round")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            return None
        # Storing always writes >= 1, so 0-or-negative beside stored parameters
        # is corruption, not a first attempt.
        return round_index if round_index > 0 else None

    @staticmethod
    def _store_parameters(cme_workflow: fastworkflow.Workflow, parameters):
        # Read the prior round BEFORE installing, because the answer depends on
        # whether stored parameters were already there: installing first would
        # make every call look like a continuation of itself.
        prior = ParameterExtraction._get_stored_round(cme_workflow)
        cme_workflow.context["stored_parameters"] = parameters
        if prior is None:
            # Unknown does not become known by counting up from it. Leave the
            # counter absent so the next read stays unknown instead of
            # inventing an ordinal mid-sequence.
            cme_workflow.context.pop("stored_parameter_round", None)
        else:
            cme_workflow.context["stored_parameter_round"] = prior + 1

    @staticmethod
    def _clear_parameters(cme_workflow: fastworkflow.Workflow):
        if "stored_parameters" in cme_workflow.context:
            del cme_workflow.context["stored_parameters"]
        # The round counts this error state's attempts, so it ends with it.
        # Leaving it behind would number the NEXT command's first extraction as
        # a retry of a command that already succeeded.
        if "stored_parameter_round" in cme_workflow.context:
            del cme_workflow.context["stored_parameter_round"]

    @staticmethod
    def _extract_missing_fields(input_for_param_extraction, sws, command_name, stored_params):
        stored_missing_fields = []
        is_valid, error_msg, _ = input_for_param_extraction.validate_parameters(
            sws, command_name, stored_params
        )

        if not is_valid:
            if MISSING_INFORMATION_ERRMSG in error_msg:
                missing_fields_str = error_msg.split(f"{MISSING_INFORMATION_ERRMSG}")[1].split("\n")[0]
                stored_missing_fields = [f.strip() for f in missing_fields_str.split(",")]
            if INVALID_INFORMATION_ERRMSG in error_msg:
                invalid_section = error_msg.split(f"{INVALID_INFORMATION_ERRMSG}")[1]
                if "\n" in invalid_section:
                    invalid_fields_str = invalid_section.split("\n")[0]
                    stored_missing_fields.extend(
                        invalid_field.split(" '")[0].strip()
                        for invalid_field in invalid_fields_str.split(", ")
                    )
        return stored_missing_fields

    @staticmethod
    def _merge_parameters(old_params, new_params, missing_fields):
        """
        Merge new parameters with old parameters, prioritizing new values when appropriate.
        """
        merged_data = {
            field_name: getattr(old_params, field_name, None)
            for field_name in list(type(old_params).model_fields.keys())
        }

        # all_fields = list(old_params.model_fields.keys())
        missing_fields = missing_fields or []

        for field_name in missing_fields:
            merged_data[field_name] = getattr(new_params, field_name)

        # Construct the model instance without validation
        return old_params.__class__.model_construct(**merged_data)

            # if hasattr(new_params, field_name):
            #     new_value = getattr(new_params, field_name)
            #     old_value = merged_data.get(field_name)

            #     if new_value is not None and new_value != NOT_FOUND:
            #         if isinstance(old_value, str) and INVALID in old_value and INVALID not in new_value:
            #             merged_data[field_name] = new_value

            #         elif old_value is None or old_value == NOT_FOUND:
            #             merged_data[field_name] = new_value

            #         elif isinstance(old_value, int) and old_value == INVALID_INT_VALUE:
            #             with contextlib.suppress(ValueError, TypeError):
            #                 merged_data[field_name] = int(new_value)

            #         elif isinstance(old_value, float) and old_value == INVALID_FLOAT_VALUE:
            #             with contextlib.suppress(ValueError, TypeError):
            #                 merged_data[field_name] = float(new_value)

            #         elif (field_name in missing_fields and
            #             hasattr(old_params.model_fields.get(field_name), "json_schema_extra") and
            #             old_params.model_fields.get(field_name).json_schema_extra and
            #             "db_lookup" in old_params.model_fields.get(field_name).json_schema_extra):
            #             merged_data[field_name] = new_value

            #         elif field_name in missing_fields:
            #             field_info = old_params.model_fields.get(field_name)
            #             has_pattern = hasattr(field_info, "pattern") and field_info.pattern is not None

            #             if not has_pattern:
            #                 for meta in getattr(field_info, "metadata", []):
            #                     if hasattr(meta, "pattern"):
            #                         has_pattern = True
            #                         break

            #             if not has_pattern and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            #                 has_pattern = "pattern" in field_info.json_schema_extra

            #             if has_pattern:
            #                 merged_data[field_name] = new_value

    @staticmethod
    def _format_parameters_for_display(params):
        """
        Format parameters for display in the error message.
        """
        if not params:
            return ""

        lines = []

        all_fields = list(type(params).model_fields.keys())

        for field_name in all_fields:
            value = getattr(params, field_name, None)

            if value in [
                NOT_FOUND,
                None,
                INVALID_INT_VALUE,
                INVALID_FLOAT_VALUE
            ]:
                continue

            # Skip empty lists (sentinel for missing list fields)
            if isinstance(value, list) and len(value) == 0:
                continue

            display_name = " ".join(word.capitalize() for word in field_name.split('_'))

            # Format fields appropriately based on type
            if (
                isinstance(value, bool)
                or not hasattr(value, 'value')
                and isinstance(value, (int, float))
                or not hasattr(value, 'value')
                and isinstance(value, str)
                or not hasattr(value, 'value')
            ):
                lines.append(f"{display_name}: {value}")
            else:  # Handle enum types
                lines.append(f"{display_name}: {value.value}")
        return "\n".join(lines)

    @staticmethod
    def _apply_missing_fields(command: str, default_params: BaseModel, missing_fields: list):
        global PARAMETER_EXTRACTION_ERROR_MSG
        if not PARAMETER_EXTRACTION_ERROR_MSG:
            PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

        # Work on plain dict to avoid validation during assignment
        params_data = {
            field_name: getattr(default_params, field_name, None)
            for field_name in list(type(default_params).model_fields.keys())
        }

        if "," in command:
            parts = [part.strip() for part in command.split(",")]

            if (
                len(parts) == len(missing_fields) == 1
                or len(parts) != len(missing_fields)
                and parts
                and missing_fields
            ):
                field = missing_fields[0]
                if field in params_data:
                    params_data[field] = parts[0]
            elif len(parts) == len(missing_fields) and len(missing_fields) > 1:
                for i, field in enumerate(missing_fields):
                    if i < len(parts) and field in params_data:
                        params_data[field] = parts[i]
        elif missing_fields:
            field = missing_fields[0]
            if field in params_data:
                params_data[field] = command.strip()

        # Construct model without validation
        return default_params.__class__.model_construct(**params_data)

    @staticmethod
    def _extract_parameters_from_xml(command: str, command_parameters_class: type[BaseModel]) -> Optional[BaseModel]:
        """
        Extract parameters from XML-formatted command using regex.

        Returns:
            BaseModel instance with extracted parameters, or None if parsing fails
        """
        field_names = list(command_parameters_class.model_fields.keys())

        # If no parameters are defined, return empty model immediately
        if not field_names:
            return command_parameters_class.model_construct()

        extracted_data = {}

        # Try to extract each parameter using XML tags
        if len(field_names) == 1:
            # If there's only one field, extract content from first XML tag
            pattern = r'<[^>]+>(.+?)</[^>]+>'
            if match := re.search(pattern, command, re.DOTALL):
                parameter_value = match[1].strip()
                extracted_data[field_names[0]] = parameter_value
        else:
            # Try to extract each parameter using XML tags
            for field_name in field_names:
                # Look for <field_name>value</field_name> pattern
                pattern = rf'<{re.escape(field_name)}>(.+?)</{re.escape(field_name)}>'
                # For list-typed fields, an agent may repeat the tag once per item
                # (e.g. <item_ids>a</item_ids> <item_ids>b</item_ids>). re.search
                # would keep only the first, silently dropping the rest, so collect
                # ALL occurrences with findall and hand the list parser every value.
                _, forigin = _unwrap_optional(
                    command_parameters_class.model_fields[field_name].annotation
                )
                if forigin in (list, List):
                    matches = re.findall(pattern, command, re.DOTALL)
                    if len(matches) > 1:
                        # Multiple repeated tags -> JSON array so the list parser
                        # below collects every value (not just the first).
                        extracted_data[field_name] = json.dumps([m.strip() for m in matches])
                    elif len(matches) == 1:
                        extracted_data[field_name] = matches[0].strip()
                elif match := re.search(pattern, command, re.DOTALL):
                    parameter_value = match[1].strip()
                    extracted_data[field_name] = parameter_value

        # Check if we extracted values for ALL fields (safest criteria for LLM fallback)
        all_fields_extracted = len(extracted_data) == len(field_names)

        # Check if agent used example values
        if all_fields_extracted:
            for field_name, extracted_value in extracted_data.items():
                field_info = command_parameters_class.model_fields[field_name]
                examples = getattr(field_info, "examples", None)
                if examples and extracted_value in examples:
                    all_fields_extracted = False
                    break

        if all_fields_extracted:
            # Initialize all fields with their default values (if they exist) or None
            params_data = {}
            for field_name in field_names:
                field_info = command_parameters_class.model_fields[field_name]
                if field_info.default is not PydanticUndefined:
                    params_data[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    params_data[field_name] = field_info.default_factory()
                else:
                    params_data[field_name] = None

            # Parse and type-correct extracted values
            for field_name, raw_value in extracted_data.items():
                field_info = command_parameters_class.model_fields[field_name]
                field_type, origin = _unwrap_optional(field_info.annotation)

                # Handle list types with robust parsing
                if origin in (list, List):
                    inner_type = get_args(field_type)[0] if get_args(field_type) else str
                    parsed_list = None

                    # Try JSON array format
                    if raw_value.startswith('[') and raw_value.endswith(']'):
                        with contextlib.suppress(Exception):
                            parsed = json.loads(raw_value)
                            if isinstance(parsed, list):
                                parsed_list = parsed

                    # Try Python literal format
                    if parsed_list is None:
                        with contextlib.suppress(Exception):
                            parsed = ast.literal_eval(raw_value)
                            if isinstance(parsed, list):
                                parsed_list = parsed

                    # Try comma-separated format
                    if parsed_list is None and ',' in raw_value:
                        parsed_list = [item.strip().strip('"').strip("'") for item in raw_value.split(',')]

                    # Try space-separated format
                    if parsed_list is None and ' ' in raw_value and not any(c in raw_value for c in ['[', ']', '{', '}', '"', "'"]):
                        parsed_list = [item.strip() for item in raw_value.split() if item.strip()]

                    # Single value treated as single-item list
                    if parsed_list is None:
                        cleaned = raw_value.strip().strip('"').strip("'")
                        parsed_list = [cleaned] if cleaned else []

                    # Type-convert list items to match the inner type
                    if parsed_list:
                        typed_list = []
                        for item in parsed_list:
                            if inner_type is str:
                                typed_list.append(str(item))
                            elif inner_type is int:
                                with contextlib.suppress(ValueError, TypeError):
                                    typed_list.append(int(item))
                            elif inner_type is float:
                                with contextlib.suppress(ValueError, TypeError):
                                    typed_list.append(float(item))
                            elif isinstance(inner_type, type) and issubclass(inner_type, BaseModel) and isinstance(item, dict):
                                with contextlib.suppress(Exception):
                                    typed_list.append(inner_type(**item))
                            else:
                                typed_list.append(item)
                        params_data[field_name] = typed_list
                    else:
                        params_data[field_name] = []
                else:
                    # Non-list types: use raw value as-is
                    params_data[field_name] = raw_value

            # Construct model without validation
            return command_parameters_class.model_construct(**params_data)

        return None

    @staticmethod
    def _extract_and_merge_missing_parameters(stored_params: BaseModel, command: str):
        """
        Identify fields to fill by scanning for sentinel values and merge values
        parsed from the command string into a new params instance. This preserves
        existing behavior for token/field count mismatches and leaves values as
        strings (no type coercion).
        """
        # Initialize with existing values to avoid triggering validation
        field_names = list(type(stored_params).model_fields.keys())
        params_data = {
            field_name: getattr(stored_params, field_name, None)
            for field_name in field_names
        }

        # Determine which fields still need user-provided input based on sentinels
        fields_to_fill = []
        for field_name in field_names:
            value = getattr(stored_params, field_name, None)
            if value in [
                NOT_FOUND,
                None,
                INVALID_INT_VALUE,
                INVALID_FLOAT_VALUE,
            ] or (isinstance(value, list) and len(value) == 0):
                fields_to_fill.append(field_name)

        if not fields_to_fill:
            return stored_params

        # Preserve existing mismatch handling and keep all values as strings
        if "," in command:
            parts = [part.strip() for part in command.split(",")]

            if (
                len(parts) == len(fields_to_fill) == 1
                or len(parts) != len(fields_to_fill)
                and parts
            ):
                field = fields_to_fill[0]
                if field in params_data:
                    params_data[field] = parts[0]
            elif len(parts) == len(fields_to_fill) and len(fields_to_fill) > 1:
                for i, field in enumerate(fields_to_fill):
                    if i < len(parts) and field in params_data:
                        params_data[field] = parts[i]
        else:
            field = fields_to_fill[0]
            if field in params_data:
                params_data[field] = command.strip()

        # Return a new instance without validation
        return stored_params.__class__.model_construct(**params_data)