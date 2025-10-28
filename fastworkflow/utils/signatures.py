import sys
import ast
import dspy
import os
from contextlib import suppress
from typing import Optional, Tuple, Union, Dict, Any, Type, List, get_args
from enum import Enum
from datetime import date
import re
from difflib import get_close_matches
import json

from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.utils.logging import logger
from fastworkflow.model_pipeline_training import get_route_layer_filepath_model
from fastworkflow.utils.fuzzy_match import find_best_matches
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.utils.command_dependency_graph import get_dependency_suggestions

MISSING_INFORMATION_ERRMSG = None
INVALID_INFORMATION_ERRMSG = None
PARAMETER_EXTRACTION_ERROR_MSG = None
NOT_FOUND = None

LLM_PARAM_EXTRACTION = None
LITELLM_API_KEY_PARAM_EXTRACTION = None

INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max


def get_trainset(subject_command_name,workflow_folderpath) -> List[Dict[str, Any]]:
        """Load labeled trainset from the command specific JSON file for LabeledFewShot"""

        trainset = []

        if not subject_command_name:
            return trainset
        
        try:
            trainset_file = f"{subject_command_name}_param_labeled.json"
            trainset_path=get_route_layer_filepath_model(workflow_folderpath,trainset_file)
            
            if os.path.exists(trainset_path):
                with open(trainset_path, "r") as f:
                    trainset_data = json.load(f)

                if 'valid_examples' in trainset_data:
                    for example_dict in trainset_data['valid_examples']:
                        try:
                            example = dspy.Example(**example_dict["fields"])
                            example = example.with_inputs(*example_dict["inputs"])
                            trainset.append(example)
                        except Exception as e:
                            logger.warning(f"Failed to parse example {example_dict}: {e}")                    
            
        except Exception as e:
            logger.warning(f"Error loading trainset for {subject_command_name}: {e}")
            
        return trainset


class DatabaseValidator:
    """Generic validator for database lookups with fuzzy matching"""
    
    @staticmethod
    def fuzzy_match(value: str, key_values: list[str], threshold: float = 0.2) -> Tuple[bool, Optional[str], List[str]]:
        """Find the closest matching value in the specified database."""
        global NOT_FOUND
        if not NOT_FOUND:
            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

        if not value or value in [None, NOT_FOUND]:
            return False, None, []

        if not key_values:
            return False, None, []     

        match = next((v for v in key_values if value.lower() == v.lower()), None)
        if match is not None:
            return True, match, []

        best_matches, _ = find_best_matches(value, key_values, threshold = 0.7)
        if len(best_matches) == 1:
            return True, best_matches[0], []
        elif len(best_matches) > 1:
            return False, None, best_matches

        lowercase_matches = get_close_matches(
            value.lower(), 
            [val.lower() for val in key_values], 
            n = 3, 
            cutoff = threshold
        )

        original_matches = [
            value 
            for match in lowercase_matches 
            for value in key_values 
            if value.lower() == match
        ]

        if original_matches:
            return False, None, original_matches

        return False, None, []


class InputForParamExtraction(BaseModel):
    """Extract parameters from a user query using step-by-step reasoning.
Today's date is {today}.
"""  
    command: str
    input_for_param_extraction_class: Optional[Type[Any]] = None
    input_for_param_extraction: Optional[Any] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create(cls, app_workflow: fastworkflow.Workflow, subject_command_name: str, subject_command: str):
        """
        Create an instance of InputForParamExtraction with a command string.
        
        Arguments:
            command: The user's request
        
        Returns:
            An instance of InputForParamExtraction   
        """
        subject_workflow_folderpath = app_workflow.folderpath
        subject_command_routing_definition = fastworkflow.RoutingRegistry.get_definition(subject_workflow_folderpath)
        
        input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
            subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
        
        input_for_param_extraction = None
        if input_for_param_extraction_class and input_for_param_extraction_class is not cls and hasattr(input_for_param_extraction_class, 'create'):
            try:
                input_for_param_extraction = input_for_param_extraction_class.create(
                    app_workflow, subject_command_name, subject_command)
            except Exception as e:
                logger.warning(f"Failed to create input_for_param_extraction: {e}")
        
        today=date.today()
        cls.__doc__ = cls.__doc__.format(today=today)
        
        return cls(
            command=subject_command,
            input_for_param_extraction_class=input_for_param_extraction_class,
            input_for_param_extraction=input_for_param_extraction
        )
    
    @staticmethod
    def create_signature_from_pydantic_model(
        pydantic_model: Type[BaseModel]
    ) -> Type[dspy.Signature]:
        """
        Create a DSPy Signature class from a Pydantic model with type annotations.
        
        Args:
            pydantic_model: The Pydantic model class to convert
            
        Returns:
            A DSPy Signature class
        """
        signature_components = {
            "statement": (str, dspy.InputField(desc="Statement according to Dhar"))
        }

        steps = []
        field_num = 1

        for attribute_name, attribute_metadata in pydantic_model.model_fields.items():
            is_optional = False
            attribute_type = attribute_metadata.annotation

            if hasattr(attribute_type, "__origin__") and attribute_type.__origin__ is Union:
                union_elements = get_args(attribute_type)
                if type(None) in union_elements:
                    is_optional = True
                    attribute_type = next((elem for elem in union_elements if elem is not type(None)), str)

            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
            if attribute_type is str:            
                default_value = NOT_FOUND
            elif attribute_type is int:
                default_value = INVALID_INT_VALUE
            elif attribute_type is float:
                default_value = -sys.float_info.max
            else:
                default_value = None

            if (
                attribute_metadata.default is not PydanticUndefined and 
                attribute_metadata.default is not None and 
                attribute_metadata.default != Ellipsis
            ):
                default_value = attribute_metadata.default

            info_text = attribute_metadata.description or f"The {attribute_name}"

            if attribute_name != "query": 
                steps.append(f"Step {field_num}: Identify the {attribute_name} ({info_text}).")
                field_num += 1

            if isinstance(attribute_type, type) and issubclass(attribute_type, Enum):
                possible_values = [f"'{option.value}'" for option in attribute_type]
                info_text += f". Valid values: {', '.join(possible_values)}"

            if attribute_metadata.examples:
                sample_values = ", ".join([f"'{sample}'" for sample in attribute_metadata.examples])
                info_text += f". Examples: {sample_values}"

            requirement_status = "Optional" if is_optional else "Required"
            info_text += f". This field is {requirement_status}."

            if is_optional:
                info_text += f" If not mentioned in the query, use: '{default_value or 'None'}'."
            elif default_value is not None:
                info_text += f" Default value: '{default_value}'."

            field_definition = dspy.OutputField(desc=info_text, default=default_value)
            signature_components[attribute_name] = (attribute_metadata.annotation, field_definition)

            steps.extend((
                f"Step {field_num}: ",
                "For missing parameter values - return the default if it is specified otherwise return None",
            ))
            
        generated_docstring = (
            f"Extract parameter values from the statement according to Dhar. Today's date is {date.today()}.\n"
            f"{chr(10).join(steps)}"
        )

        return dspy.Signature(signature_components, generated_docstring)
    
    def extract_parameters(self, CommandParameters: Type[BaseModel] = None, subject_command_name: str = None, workflow_folderpath: str = None) -> BaseModel:
        """
        Extract parameters from the command using DSPy.
        
        Returns:
            The extracted parameters
        """
        global PARAMETER_EXTRACTION_ERROR_MSG, LLM_PARAM_EXTRACTION, LITELLM_API_KEY_PARAM_EXTRACTION

        if not PARAMETER_EXTRACTION_ERROR_MSG:
            PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

        if not LLM_PARAM_EXTRACTION:
            LLM_PARAM_EXTRACTION = fastworkflow.get_env_var("LLM_PARAM_EXTRACTION")
            LITELLM_API_KEY_PARAM_EXTRACTION = fastworkflow.get_env_var("LITELLM_API_KEY_PARAM_EXTRACTION")

        lm = dspy.LM(LLM_PARAM_EXTRACTION, api_key=LITELLM_API_KEY_PARAM_EXTRACTION)
        
        model_class = CommandParameters 
        if model_class is None:
            raise ValueError("No model class provided")
        
        params_signature = self.create_signature_from_pydantic_model(model_class)
        
        class ParamExtractor(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.predictor = dspy.ChainOfThought(signature)
                
            def forward(self, command=None):
                return self.predictor(statement=command)
        
        param_extractor = ParamExtractor(params_signature)
        
        trainset = get_trainset(subject_command_name,workflow_folderpath)
        length = len(trainset)

        param_dict = {}
        field_names = list(model_class.model_fields.keys())
        with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
            optimizer = dspy.LabeledFewShot(k=length)
            compiled_model = optimizer.compile(
                student=param_extractor,
                trainset=trainset
            )

            def basic_checks(args, pred):
                for field_name in field_names:
                    # return 0 if it extracts an example value instead of correct value | None
                    extracted_param_value = getattr(pred, field_name)
                    examples = model_class.model_fields[field_name].examples
                    if examples and extracted_param_value in examples:
                        return 0.0
                return 1.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.BestOfN(
                module=compiled_model, 
                N=3, 
                reward_fn=basic_checks, 
                threshold=1.0)

            try:
                dspy_result = best_of_3(command=self.command)
                for field_name in field_names:
                    default = model_class.model_fields[field_name].default
                    param_dict[field_name] = getattr(dspy_result, field_name, default)
            except Exception as exc:
                logger.warning(PARAMETER_EXTRACTION_ERROR_MSG.format(error=exc))

        # IMPORTANT: Do *not* instantiate the original model via the regular
        # constructor – that would invoke full validation including regex
        # pattern checks.  Instead use `model_construct`, which builds the
        # object in-place without any validation.
        return model_class.model_construct(**param_dict)  # type: ignore[arg-type]
    
    def validate_parameters(self,
                            app_workflow: fastworkflow.Workflow, 
                            subject_command_name: str,
                            cmd_parameters: BaseModel) -> Tuple[bool, str, Dict[str, List[str]], List[str]]:
        """
        Check if the parameters are valid in the current context, including database lookups.
        """
        global MISSING_INFORMATION_ERRMSG, INVALID_INFORMATION_ERRMSG, NOT_FOUND
        if not MISSING_INFORMATION_ERRMSG:
            MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
        if not INVALID_INFORMATION_ERRMSG:
            INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")
        if not NOT_FOUND:
            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

        is_valid = True
        missing_fields = []
        invalid_fields = []
        all_suggestions = {}

        for field_name, field_info in type(cmd_parameters).model_fields.items():
                field_value = getattr(cmd_parameters, field_name, None)

                if field_value not in [NOT_FOUND, None, INVALID_INT_VALUE, INVALID_FLOAT_VALUE]:
                        annotation = field_info.annotation

                        # Build list of candidate concrete types (exclude NoneType from Union)
                        candidate_types: List[Type[Any]] = []
                        if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                            for t in get_args(annotation):
                                if t is not type(None):  # noqa: E721
                                    candidate_types.append(t)  # type: ignore[arg-type]
                        else:
                            candidate_types = [annotation]  # type: ignore[list-item]

                        def build_type_suggestion() -> List[str]:
                            examples = getattr(field_info, "examples", []) or []
                            example = examples[0] if examples else None
                            # Enum suggestions list valid values
                            enum_types = [t for t in candidate_types if isinstance(t, type) and issubclass(t, Enum)]
                            if enum_types:
                                opts = [f"'{opt.value}'" for t in enum_types for opt in t]
                                return [f"Please provide a value matching the expected type/format. Valid values: {', '.join(opts)}"]
                            # List suggestions
                            def _is_list_type(tt):
                                try:
                                    return hasattr(tt, "__origin__") and tt.__origin__ in (list, List)
                                except Exception:
                                    return False
                            list_types = [t for t in candidate_types if _is_list_type(t)]
                            if list_types:
                                inner_args = get_args(list_types[0])
                                inner = inner_args[0] if inner_args else str
                                inner_name = inner.__name__ if isinstance(inner, type) else str(inner)
                                hint = (
                                    f"Please provide a list of {inner_name} values. Accepted formats: "
                                    f"JSON list (e.g., [\"a\", \"b\"]), Python list (e.g., ['a', 'b']), "
                                    f"or comma-separated (e.g., a,b)."
                                )
                                return [hint]
                            # Fallback: show expected type names (handles unions)
                            name_list: List[str] = []
                            for t in candidate_types:
                                if isinstance(t, type):
                                    name_list.append(t.__name__)
                                else:
                                    name_list.append(str(t))
                            base = f"Please provide a value matching the expected type/format: {' or '.join(name_list)}"
                            if example is not None:
                                base = f"{base} (e.g., {example})"
                            return [base]

                        valid_by_type = False
                        corrected_value: Optional[Any] = None
                        def _is_list_type(tt):
                            try:
                                return hasattr(tt, "__origin__") and tt.__origin__ in (list, List)
                            except Exception:
                                return False

                        def _parse_list_like_string(s: str) -> Optional[list]:
                                if not isinstance(s, str):
                                    return None
                                text = s.strip()
                                if text.startswith("[") and text.endswith("]"):
                                        with suppress(Exception):
                                                parsed = json.loads(text)
                                                if isinstance(parsed, list):
                                                    return parsed
                                # Try Python literal list
                                with suppress(Exception):
                                        parsed = ast.literal_eval(text)
                                        if isinstance(parsed, list):
                                            return parsed
                                # Comma-separated values
                                if "," in text:
                                    parts = [p.strip() for p in text.split(",")]
                                    cleaned = [
                                        (p[1:-1] if len(p) >= 2 and ((p[0] == p[-1] == '"') or (p[0] == p[-1] == "'")) else p)
                                        for p in parts
                                    ]
                                    return cleaned
                                # Single value - treat as a list with one element
                                if text:
                                    # Remove quotes if present
                                    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
                                        return [text[1:-1]]
                                    return [text]
                                return None

                        def _coerce_scalar(expected_type: Type[Any], val: Any) -> Tuple[bool, Optional[Any]]:
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
                                        # Also handle string representations of integers
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
                                        with suppress(Exception):
                                            return True, int(val.strip())
                                    return False, None
                                # float
                                if expected_type is float:
                                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                                        return True, float(val)
                                    if isinstance(val, str):
                                        with suppress(Exception):
                                            return True, float(val.strip())
                                    return False, None
                                # Enum
                                if isinstance(expected_type, type) and issubclass(expected_type, Enum):
                                    ok, enum_val = _try_coerce_enum(expected_type, val)
                                    return (ok, enum_val if ok else None)
                                                            # Unknown: accept if already instance
                                return (True, val) if isinstance(val, expected_type) else (False, None)

                        def _try_coerce_list(list_type: Any, value: Any) -> Tuple[bool, Optional[list]]:
                            inner_args = get_args(list_type)
                            inner_type = inner_args[0] if inner_args else str
                            raw_list: Optional[list] = None
                            if isinstance(value, list):
                                raw_list = value
                            elif isinstance(value, str):
                                raw_list = _parse_list_like_string(value)
                            if raw_list is None:
                                return False, None
                            coerced_list = []
                            for item in raw_list:
                                ok, coerced = _coerce_scalar(inner_type, item)
                                if not ok:
                                    return False, None
                                coerced_list.append(coerced)
                            return True, coerced_list

                        def _try_coerce_enum(enum_cls: Type[Enum], val: Any) -> Tuple[bool, Optional[Enum]]:
                            if isinstance(val, enum_cls):
                                return True, val
                            if isinstance(val, str):
                                for member in enum_cls:
                                    if val == member.value or val.lower() == str(member.name).lower():
                                        return True, member
                            return False, None

                        for t in candidate_types:
                            # list[...] and typing.List[...] support
                            if _is_list_type(t):
                                ok, coerced_list = _try_coerce_list(t, field_value)
                                if ok:
                                    corrected_value = coerced_list
                                    valid_by_type = True
                                    break
                            # str, bool, int, float - use _coerce_scalar for consistency
                            if t in (str, bool, int, float):
                                ok, coerced = _coerce_scalar(t, field_value)
                                if ok:
                                    corrected_value = coerced
                                    valid_by_type = True
                                    break                            # Enum
                            if isinstance(t, type) and issubclass(t, Enum):
                                ok, enum_val = _try_coerce_enum(t, field_value)
                                if ok:
                                    corrected_value = enum_val
                                    valid_by_type = True
                                    break

                        if valid_by_type:
                            if corrected_value is not None:
                                pattern = next(
                                    (meta.pattern
                                        for meta in getattr(field_info, "metadata", [])
                                        if hasattr(meta, "pattern")),
                                    None,
                                )
                                if pattern and field_value is not None and field_value != NOT_FOUND:
                                    invalid_value = None
                                    if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                                        invalid_value = field_info.json_schema_extra.get("invalid_value")

                                    # if invalid_value and field_value == invalid_value:
                                    #     invalid_fields.append(f"{field_name} '{field_value}'")
                                    #     pattern_str = str(pattern)
                                    #     examples = getattr(field_info, "examples", [])
                                    #     example = examples[0] if examples else ""
                                    #     all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                                    #     is_valid = False

                                    # else:
                                    pattern_regex = re.compile(pattern)
                                    if not pattern_regex.fullmatch(str(field_value)):
                                        invalid_fields.append(f"{field_name} '{field_value}'")
                                        pattern_str = str(pattern)
                                        examples = getattr(field_info, "examples", [])
                                        example = examples[0] if examples else ""

                                        invalid_fields.append(f"{field_name} '{field_value}'")
                                        all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                                        is_valid = False
                                    else:
                                        try:
                                            setattr(cmd_parameters, field_name, corrected_value)
                                        except Exception as e:
                                            logger.critical(f"Failed to set attribute {field_name} with value {corrected_value}")
                                            raise e
                                else:
                                    try:
                                        setattr(cmd_parameters, field_name, corrected_value)
                                    except Exception as e:
                                        logger.critical(f"Failed to set attribute {field_name} with value {corrected_value}")
                                        raise e
                        else:
                            invalid_fields.append(f"{field_name} '{field_value}'")
                            all_suggestions[field_name] = build_type_suggestion()
                            is_valid = False

                is_optional = False
                attribute_type = field_info.annotation
                if hasattr(attribute_type, "__origin__") and attribute_type.__origin__ is Union:
                    union_elements = get_args(attribute_type)
                    if type(None) in union_elements:
                        is_optional = True

                is_required=True
                if is_optional:
                        is_required=False

                # Only add to missing fields if it's required AND has no value
                if is_required and \
                                field_value in [
                        NOT_FOUND, 
                        None,
                        INVALID_INT_VALUE,
                        INVALID_FLOAT_VALUE
                    ]:
                    missing_fields.append(field_name)
                    is_valid = False

        for field_name, field_info in type(cmd_parameters).model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)

            if field_value in [NOT_FOUND, None]:
                continue

            is_db_lookup = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                is_db_lookup = field_info.json_schema_extra.get("db_lookup")

            if is_db_lookup:
                if not self.input_for_param_extraction_class:
                    raise ValueError("input_for_param_extraction_class is not set.")
                matched, corrected_value, field_suggestions = (
                    self.input_for_param_extraction_class.db_lookup(
                        app_workflow, field_name, field_value)
                ) 
                # matched, corrected_value, field_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)

                if matched:
                    setattr(cmd_parameters, field_name, corrected_value)
                elif field_suggestions:
                    invalid_fields.append(f"{field_name} '{field_value}'")
                    all_suggestions[field_name] = field_suggestions
                    is_valid = False

        if is_valid:
            if not (
                self.input_for_param_extraction_class and \
                            hasattr(self.input_for_param_extraction_class, 'validate_extracted_parameters')
            ):
                return (True, "All required parameters are valid.", {}, [])

            try:
                is_valid, message = self.input_for_param_extraction_class.validate_extracted_parameters(app_workflow, subject_command_name, cmd_parameters)
            except Exception as e:
                message = f"Exception in {subject_command_name}'s validate_extracted_parameters function: {str(e)}"
                logger.critical(message)                    
                return (False, message, {}, [])

            if is_valid:
                return (True, "All required parameters are valid.", {}, [])
            return (False, message, {}, [])

        message = ''
        if missing_fields:
            message += f"{MISSING_INFORMATION_ERRMSG}" + ", ".join(missing_fields) + "\n"

            for missing_field in missing_fields:
                is_available_from = None
                if hasattr(type(cmd_parameters).model_fields.get(missing_field), "json_schema_extra") and type(cmd_parameters).model_fields.get(missing_field).json_schema_extra:
                    is_available_from = type(cmd_parameters).model_fields.get(missing_field).json_schema_extra.get("available_from")
                if is_available_from:
                    msg_prefix = "abort and "
                    if "run_as_agent" in app_workflow.context:
                        msg_prefix = ""
                    message += f"{msg_prefix}use the {' or '.join(is_available_from)} command(s) to get {missing_field} information. OR...\n"

        if invalid_fields:
            message += f"{INVALID_INFORMATION_ERRMSG}" + ", ".join(invalid_fields) + "\n"

        for field, suggestions in all_suggestions.items():
            if suggestions:
                is_format_instruction = any(("format" in str(s).lower() or "pattern" in str(s).lower()) for s in suggestions)

                if is_format_instruction:
                    message += f"\n{field}: {', '.join(suggestions)}"
                else:
                    message += f"Did you mean one of these {field}s? {', '.join(suggestions)}\n"

        combined_fields = []
        if missing_fields:
            combined_fields.extend(missing_fields)
        if invalid_fields:
            invalid_field_names = [field.split(" '")[0].strip() for field in invalid_fields]
            combined_fields.extend(invalid_field_names)

        if combined_fields:
            combined_fields_str = ", ".join(combined_fields)
            message += f"\nProvide corrected parameter values in the exact order specified below, separated by commas:\n{combined_fields_str}"
            if "run_as_agent" not in app_workflow.context:
                message += "\nFor parameter values that include a comma, provide separately from other values, and one at a time."

        return (False, message, all_suggestions, combined_fields)