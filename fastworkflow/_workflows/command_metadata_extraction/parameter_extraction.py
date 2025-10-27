import contextlib
import sys
import re
import ast
import json
from typing import Dict, List, Optional, Any, Type, get_args, Union
from enum import Enum

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import ModuleType

from fastworkflow.utils.signatures import InputForParamExtraction


INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
INVALID = fastworkflow.get_env_var("INVALID")
PARAMETER_EXTRACTION_ERROR_MSG = None


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
        if stored_params:
            new_params = self._extract_and_merge_missing_parameters(stored_params, self.command)
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

                # If regex extraction fails, fall back to LLM-based extraction
                if new_params is None:
                    new_params = input_for_param_extraction.extract_parameters(
                        command_parameters_class,
                        self.command_name,
                        app_workflow_folderpath)
            else:
                # Use LLM-based extraction for assistant mode
                new_params = input_for_param_extraction.extract_parameters(
                    command_parameters_class,
                    self.command_name,
                    app_workflow_folderpath)

        is_valid, error_msg, suggestions, missing_invalid_fields = \
            input_for_param_extraction.validate_parameters(
            self.app_workflow, self.command_name, new_params
        )

        # Set all the missing and invalid fields to None before storing
        current_values = {
            field_name: getattr(new_params, field_name, None)
            for field_name in list(type(new_params).model_fields.keys())
        }
        for field_name in missing_invalid_fields:
            if field_name in current_values:
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
    def _store_parameters(cme_workflow: fastworkflow.Workflow, parameters):
        cme_workflow.context["stored_parameters"] = parameters

    @staticmethod
    def _clear_parameters(cme_workflow: fastworkflow.Workflow):
        if "stored_parameters" in cme_workflow.context:
            del cme_workflow.context["stored_parameters"]

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
    def _parse_list_like_string(s: str) -> Optional[list]:
        """Parse string into list - supports JSON, Python literal, comma-separated, or single value format."""
        if not isinstance(s, str):
            return None
        text = s.strip()

        # Try JSON list
        if text.startswith("[") and text.endswith("]"):
            with contextlib.suppress(Exception):
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed

        # Try Python literal list
        with contextlib.suppress(Exception):
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed

        # Comma-separated values (including single values)
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            # Remove quotes if present and filter out empty strings
            cleaned = [
                (p[1:-1] if len(p) >= 2 and ((p[0] == p[-1] == '"') or (p[0] == p[-1] == "'")) else p)
                for p in parts
                if p.strip()  # Filter out empty strings
            ]
            return cleaned

        # Single value - treat as a list with one element
        if text:
            # Remove quotes if present
            if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
                return [text[1:-1]]
            return [text]

        return None

    @staticmethod
    def _coerce_scalar(expected_type: Type[Any], val: Any) -> tuple[bool, Optional[Any]]:
        """Coerce a value to the expected scalar type."""
        # str
        if expected_type is str:
            return True, str(val)

        # bool
        if expected_type is bool:
            if isinstance(val, bool):
                return True, val
            elif isinstance(val, str):
                lower_val = val.lower().strip()
                if lower_val in ('true', 'false'):
                    return True, lower_val == 'true'
                elif lower_val in ('0', '1'):
                    return True, lower_val == '1'
            elif isinstance(val, int):
                return True, bool(val)
            return False, None

        # int
        if expected_type is int:
            if isinstance(val, bool):
                return False, None
            if isinstance(val, int):
                return True, val
            if isinstance(val, str):
                with contextlib.suppress(Exception):
                    return True, int(val.strip())
            return False, None

        # float
        if expected_type is float:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return True, float(val)
            if isinstance(val, str):
                with contextlib.suppress(Exception):
                    return True, float(val.strip())
            return False, None

        # Enum
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            if isinstance(val, expected_type):
                return True, val
            if isinstance(val, str):
                for member in expected_type:
                    if val == member.value or val.lower() == str(member.name).lower():
                        return True, member
            return False, None

        # Unknown: accept if already instance
        return (True, val) if isinstance(val, expected_type) else (False, None)

    @staticmethod
    def _coerce_to_type(field_type: Type[Any], value: str) -> Any:
        """
        Coerce a string value to the expected field type.
        Handles list types, scalar types (int, float, bool, str), and Enums.
        """
        # Get candidate types (handle Union/Optional)
        candidate_types: List[Type[Any]] = []
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            for t in get_args(field_type):
                if t is not type(None):  # noqa: E721
                    candidate_types.append(t)
        else:
            candidate_types = [field_type]

        # Try each candidate type
        for t in candidate_types:
            # Check if it's a list type
            if hasattr(t, "__origin__") and t.__origin__ in (list, List):
                inner_args = get_args(t)
                inner_type = inner_args[0] if inner_args else str

                # Parse list-like string
                raw_list = ParameterExtraction._parse_list_like_string(value)
                if raw_list is None:
                    continue

                # Coerce each element to inner type
                coerced_list = []
                all_ok = True
                for item in raw_list:
                    ok, coerced = ParameterExtraction._coerce_scalar(inner_type, item)
                    if not ok:
                        all_ok = False
                        break
                    coerced_list.append(coerced)

                if all_ok:
                    return coerced_list

            # Try scalar types
            elif t in (str, bool, int, float):
                ok, coerced = ParameterExtraction._coerce_scalar(t, value)
                if ok:
                    return coerced

            # Try Enum
            elif isinstance(t, type) and issubclass(t, Enum):
                ok, enum_val = ParameterExtraction._coerce_scalar(t, value)
                if ok:
                    return enum_val

        # If no type conversion worked, return original value
        return value

    @staticmethod
    def _extract_parameters_from_xml(command: str, command_parameters_class: type[BaseModel]) -> Optional[BaseModel]:
        """
        Extract parameters from XML-formatted command using regex with type coercion.

        Returns:
            BaseModel instance with extracted parameters, or None if parsing fails
        """
        field_names = list(command_parameters_class.model_fields.keys())

        # If no parameters are defined, return empty model immediately
        if not field_names:
            return command_parameters_class.model_construct()

        extracted_data = {}

        # Try to extract each parameter using XML tags
        for field_name in field_names:
            # Look for <field_name>value</field_name> pattern
            pattern = rf'<{re.escape(field_name)}>(.+?)</{re.escape(field_name)}>'
            if match := re.search(pattern, command, re.DOTALL):
                raw_value = match[1].strip()

                # Get field type and coerce the value
                field_info = command_parameters_class.model_fields[field_name]
                field_type = field_info.annotation

                # Apply type coercion
                coerced_value = ParameterExtraction._coerce_to_type(field_type, raw_value)
                extracted_data[field_name] = coerced_value

        # If we extracted at least one parameter, create the model
        if extracted_data:
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

            # Update with extracted values
            params_data |= extracted_data

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
            ]:
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