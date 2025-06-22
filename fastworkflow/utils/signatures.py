import sys
import dspy
import os
from typing import Annotated, Optional, Tuple, Union, Dict, Any, Type, List, get_args
from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from pydantic_core import PydanticUndefined
import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from datetime import date
import re
import inspect
from difflib import get_close_matches
from fastworkflow import ModuleType
from fastworkflow.utils.pydantic_model_2_dspy_signature_class import TypedPredictorSignature
from dspy.teleprompt import LabeledFewShot
import json
from fastworkflow.utils.logging import logger
from fastworkflow.model_pipeline_training import train,get_route_layer_filepath_model
from fastworkflow.utils.fuzzy_match import find_best_match

MISSING_INFORMATION_ERRMSG = None
INVALID_INFORMATION_ERRMSG = None
PARAMETER_EXTRACTION_ERROR_MSG = None
NOT_FOUND = None

LLM_PARAM_EXTRACTION = None
LITELLM_API_KEY_PARAM_EXTRACTION = None


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
        """
        Find the closest matching value in the specified database.
        """
        global NOT_FOUND
        if not NOT_FOUND:
            NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

        if not value or value in [None, NOT_FOUND]:
            return False, None, []
        
        if not key_values:
            return False, None, []     
        
        best_match,_=find_best_match(value, key_values,threshold=0.7)
        if best_match:
            return True, best_match, []
        else:
            normalized_value = value.lower()
            
            for value in key_values:
                if normalized_value == value.lower():
                    return True, value, []
            
            matches = get_close_matches(normalized_value, 
                                    [val.lower() for val in key_values], 
                                    n=3, 
                                    cutoff=threshold)
            
            original_matches = []
            for match_lower in matches:
                for value in key_values:
                    if value.lower() == match_lower:
                        original_matches.append(value)
                        break
            
            if original_matches:
                return False, None, original_matches
        
        return False, None, []


class InputForParamExtraction(BaseModel):
    """Extract parameters from a user query using step-by-step reasoning.
Today's date is {today}.
"""  
    command: str
    input_for_param_extraction: Optional[Any] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def create(cls, subject_session: fastworkflow.Session, subject_command_name: str, subject_command: str):
        """
        Create an instance of InputForParamExtraction with a command string.
        
        Arguments:
            command: The user's request
        
        Returns:
            An instance of InputForParamExtraction   
        """
        subject_workflow_folderpath = subject_session.workflow_snapshot.workflow_folderpath
        subject_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(subject_workflow_folderpath)
        
        input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
            subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
        
        input_for_param_extraction = None
        if input_for_param_extraction_class and input_for_param_extraction_class is not cls and hasattr(input_for_param_extraction_class, 'create'):
            try:
                input_for_param_extraction = input_for_param_extraction_class.create(
                    subject_session.workflow_snapshot, subject_command_name, subject_command)
            except Exception as e:
                logger.warning(f"Failed to create input_for_param_extraction: {e}")
        
        today=date.today()
        cls.__doc__ = cls.__doc__.format(today=today)
        
        return cls(
            command=subject_command,
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
                "command": (str, dspy.InputField(desc="User's request"))
            }

            steps = ["query: The original user query (Always include this)."]
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
                    default_value = -sys.maxsize
                elif attribute_type is float:
                    default_value = -sys.float_info.max
                else:
                    attribute_type = None
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
                f"Step {field_num}: Check for any missing details.",
                "Return the default value for the parameters for which default value is specified.",
                "For parameters specified as enums, return the default value if the parameter value is not explicitly specified in the query",
                "Return None for the parameter value which is missing in the query",
                "Always return the query in the output.",
            ))
            generated_docstring = f"""Extract structured parameters from a user query using step-by-step reasoning. Today's date is {date.today()}.

        {chr(10).join(steps)}
        """
            instructions = generated_docstring

            return dspy.Signature(signature_components, instructions)

    @staticmethod
    def populate_defaults_dict(command_parameters_class):
        default_params = {}
        for field_name, field_info in command_parameters_class.model_fields.items():
            if (
                field_info.default is not PydanticUndefined and 
                field_info.default is not None and 
                field_info.default is not Ellipsis
            ):
                default_params[field_name] = field_info.default
            # Handle strings
            elif field_info.annotation == str:
                default_params[field_name] = NOT_FOUND
            elif field_info.annotation == int:
                default_params[field_name] = -sys.maxsize
            elif field_info.annotation == float:
                default_params[field_name] = -sys.float_info.max
            # Handle Optional[int] and Optional[float]
            elif (hasattr(field_info.annotation, "__origin__") and
                  field_info.annotation.__origin__ is Union and
                  (int in field_info.annotation.__args__ or float in field_info.annotation.__args__) and
                  type(None) in field_info.annotation.__args__):
                default_params[field_name] = None
            else:
                default_params[field_name] = None

        return command_parameters_class(**default_params)
    
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
                return self.predictor(command=command)
        
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

            try:
                dspy_result = compiled_model(command=self.command)
                for field_name in field_names:
                    default = model_class.model_fields[field_name].default
                    param_dict[field_name] = getattr(dspy_result, field_name, default)
            except Exception as exc:
                logger.warning(PARAMETER_EXTRACTION_ERROR_MSG.format(error=exc))

        try:            
            params = model_class(**param_dict)
        except ValidationError:
            params = self.populate_defaults_dict(model_class)

        return params
    
    def validate_parameters(self,
                            subject_session: fastworkflow.Session, 
                            subject_command_name: str,
                            cmd_parameters: BaseModel) -> Tuple[bool, str, Dict[str, List[str]]]:
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

        # check if the input for parameter extraction class is defined in the registary then call the process_parameters function on the instance.
        if hasattr(self.input_for_param_extraction, 'process_extracted_parameters'):
            self.input_for_param_extraction.process_extracted_parameters(subject_session.workflow_snapshot, subject_command_name, cmd_parameters)

        # Check required fields
        for field_name, field_info in type(cmd_parameters).model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)

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
                    -sys.maxsize,
                    -sys.float_info.max
                ]:
                missing_fields.append(field_name)
                is_valid = False

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

                if invalid_value and field_value == invalid_value:
                    invalid_fields.append(f"{field_name} '{field_value}'")
                    pattern_str = str(pattern)
                    examples = getattr(field_info, "examples", [])
                    example = examples[0] if examples else ""
                    all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                    is_valid = False

                else:
                    pattern_regex = re.compile(pattern)
                    if not pattern_regex.fullmatch(str(field_value)):
                        invalid_fields.append(f"{field_name} '{field_value}'")
                        pattern_str = str(pattern)
                        examples = getattr(field_info, "examples", [])
                        example = examples[0] if examples else ""
                        all_suggestions[field_name] = [f"Please use the format matching pattern {pattern_str} (e.g., {example})"]
                        is_valid = False


        for field_name, field_info in type(cmd_parameters).model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)

            if field_value in [NOT_FOUND, None]:
                continue

            is_db_lookup = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                is_db_lookup = field_info.json_schema_extra.get("db_lookup")

            if is_db_lookup:
                if not self.input_for_param_extraction:
                    raise ValueError("input_for_param_extraction is not set.")
                key_values=self.input_for_param_extraction.db_lookup(subject_session.workflow_snapshot, subject_command_name) 
                matched, corrected_value, field_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)

                if matched:
                    setattr(cmd_parameters, field_name, corrected_value)
                elif field_suggestions:
                    invalid_fields.append(f"{field_name} '{field_value}'")
                    all_suggestions[field_name] = field_suggestions
                    is_valid = False

        if is_valid:
            return (True, "All required parameters are valid.", {})

        message = ""

        if missing_fields:
            message += f"{MISSING_INFORMATION_ERRMSG}" + ", ".join(missing_fields) + "\n"

        if invalid_fields:
            message += f"{INVALID_INFORMATION_ERRMSG}" + ", ".join(invalid_fields) + "\n"

        for field, suggestions in all_suggestions.items():
            if suggestions:
                is_format_instruction = any(("format" in str(s).lower() or "pattern" in str(s).lower()) for s in suggestions)

                if is_format_instruction:
                    message += f"\n{field}: {', '.join(suggestions)}"
                else:
                    message += f"\nDid you mean one of these {field}s? {', '.join(suggestions)}"

        combined_fields = []
        if missing_fields:
            combined_fields.extend(missing_fields)
        if invalid_fields:
            invalid_field_names = [field.split(" '")[0].strip() for field in invalid_fields]
            combined_fields.extend(invalid_field_names)

        if combined_fields:
            combined_fields_str = ", ".join(combined_fields)
            message += f"\nProvide corrected parameter values in the exact order specified below, separated by commas:\n{combined_fields_str}"
            message += "\nFor parameter values that include a comma, provide separately from other values, and one at a time."

        return (False, message, all_suggestions)