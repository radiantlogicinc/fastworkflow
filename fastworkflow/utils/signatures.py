import dspy
import os
from typing import Annotated, Optional, Tuple, Union, Dict, Any, Type, List, get_args
from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator
import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from datetime import date
import re
import inspect
from difflib import get_close_matches
from fastworkflow.utils.pydantic_model_2_dspy_signature_class import TypedPredictorSignature

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")

LLM = fastworkflow.get_env_var("LLM")

DATABASES = {
}


class DatabaseValidator:
    """Generic validator for database lookups with fuzzy matching"""
    
    @staticmethod
    def fuzzy_match(value: str, database_key: str, threshold: float = 0.2) -> Tuple[bool, Optional[str], List[str]]:
        """
        Find the closest matching value in the specified database.
        """
        if not value or value in [None, NOT_FOUND]:
            return False, None, []
            
        database = DATABASES.get(database_key, [])
        if not database:
            return False, None, []
            
            
        normalized_value = value.lower()
        
        for db_value in database:
            if normalized_value == db_value.lower():
                return True, db_value, []
        
        matches = get_close_matches(normalized_value, 
                                   [db_val.lower() for db_val in database], 
                                   n=3, 
                                   cutoff=threshold)
        
        original_matches = []
        for match_lower in matches:
            for db_val in database:
                if db_val.lower() == match_lower:
                    original_matches.append(db_val)
                    break
        
        if original_matches:
            return False, None, original_matches
        
        return False, None, []


class InputForParamExtraction(BaseModel):
    """Extract parameters from a user query using step-by-step reasoning.
Today's date is {today}.
"""  
    command: str
    
    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        """
        Create an instance of InputForParamExtraction with a command string.
        
        Arguments:
            command: The user's request
        
        Returns:
            An instance of InputForParamExtraction
        """
        today=date.today()
        cls.__doc__ = cls.__doc__.format(today=today)
        
        return cls(
            command=command,
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
        signature_components = {}
        
        signature_components["query"] = (str, dspy.InputField(desc="User's request"))
        
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
            
            default_value = None
            if attribute_metadata.default is not None and attribute_metadata.default != Ellipsis:
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
        
        steps.append(f"Step {field_num}: Check for any missing details.")
        steps.append("Return the default value for the parameters for which default value is specified.")
        steps.append("For parameters specified as enums, return the default value if the parameter value is not explicitly specified in the query")
        steps.append("Return None for the parameter value which is missing in the query")
        steps.append("Always return the query in the output.")
        
        generated_docstring = f"""Extract structured parameters from a user query using step-by-step reasoning. Today's date is {date.today()}.

        {chr(10).join(steps)}
        """
        instructions = generated_docstring
        
        return dspy.Signature(signature_components, instructions)

    
    def extract_parameters(self, CommandParameters: Type[BaseModel] = None):
        """
        Extract parameters from the command using DSPy.
        
        Returns:
            The extracted parameters
        """
        lm = dspy.LM(LLM)

        model_class = CommandParameters 

        if model_class is None:
            raise ValueError("No model class provided")
            
        with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
            params_signature = self.create_signature_from_pydantic_model(
                model_class
            )
            module = dspy.ChainOfThought(params_signature)
            dspy_result = module(query=self.command)
            
        field_names = list(model_class.model_fields.keys())
            
        param_dict = {}
        for field_name in field_names:
            default = model_class.model_fields[field_name].default
            param_dict[field_name] = getattr(dspy_result, field_name, default)
                
        params = model_class(**param_dict)
            
        return params
    
    @classmethod
    def validate_parameters(cls, workflow_snapshot: WorkflowSnapshot, cmd_parameters: BaseModel) -> Tuple[bool, str, Dict[str, List[str]]]:
        """
        Check if the parameters are valid in the current context, including database lookups.
        """
        is_valid = True
        missing_fields = []
        invalid_fields = []
        all_suggestions = {}
        
        # Check required fields
        for field_name, field_info in cmd_parameters.model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)

            is_optional = False
            attribute_type = field_info.annotation
            if hasattr(attribute_type, "__origin__") and attribute_type.__origin__ is Union:
                union_elements = get_args(attribute_type)
                if type(None) in union_elements:
                    is_optional = True
            
            is_required=True
            if is_optional is True:
                is_required=False
            
            # Only add to missing fields if it's required AND has no value
            if is_required and field_value in [NOT_FOUND, None]:
                missing_fields.append(field_name)
                is_valid = False
            
            # Check pattern validation for string fields
            pattern = None
            for meta in getattr(field_info, "metadata", []):
                if hasattr(meta, "pattern"):
                    pattern = meta.pattern
                    break
                    
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
        

        for field_name, field_info in cmd_parameters.model_fields.items():
            field_value = getattr(cmd_parameters, field_name, None)
            
            if field_value in [NOT_FOUND, None]:
                continue
                
            db_key = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                db_key = field_info.json_schema_extra.get("db_validation")
            
            if db_key in DATABASES:
                matched, corrected_value, field_suggestions = DatabaseValidator.fuzzy_match(field_value, db_key)
                
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
            message += f"{MISSING_INFORMATION_ERRMSG}\n" + ", ".join(missing_fields) + "\n"
        
        if invalid_fields:
            message += f"{INVALID_INFORMATION_ERRMSG}\n" + ", ".join(invalid_fields) + "\n"
        
        message += "Please provide this information to complete your request."
        
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
            message += f"\n\nProvide your response in this exact order, separated by commas:\n{combined_fields_str}. "
            message += "\nIf any parameter value needs to include a comma, please enter the parameters one by one, i.e one at a time."
        
        return (False, message, all_suggestions)