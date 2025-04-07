from typing import Optional, Type, List, Dict, Union

from pydantic import BaseModel

import dspy

import fastworkflow
from fastworkflow.command_routing_definition import ModuleType
from fastworkflow.utils.logging import logger

from fastworkflow.utils.pydantic_model_2_dspy_signature_class import (
    TypedPredictorSignature,
)

INVALID_INT_VALUE = fastworkflow.get_env_var("INVALID_INT_VALUE")
INVALID_FLOAT_VALUE = fastworkflow.get_env_var("INVALID_FLOAT_VALUE")

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
INVALID = fastworkflow.get_env_var("INVALID")

DSPY_LM_MODEL = fastworkflow.get_env_var("DSPY_LM_MODEL")


def get_stored_parameters(session):
    return session.workflow_snapshot.context.get("stored_parameters")

def store_parameters(session, parameters):
    session.workflow_snapshot.context["stored_parameters"] = parameters

def clear_parameters(session):
    if "stored_parameters" in session.workflow_snapshot.context:
        del session.workflow_snapshot.context["stored_parameters"]


class OutputOfProcessCommand(BaseModel):
    parameter_is_valid: bool
    cmd_parameters: Optional[BaseModel] = None
    error_msg: Optional[str] = None
    suggestions: Optional[Dict[str, List[str]]] = None

def process_command(
    session: fastworkflow.Session, command: str
) -> OutputOfProcessCommand:
    sws = session.workflow_snapshot.context["subject_workflow_snapshot"]
    subject_workflow_folderpath = sws.workflow.workflow_folderpath
    subject_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(subject_workflow_folderpath)
    active_workitem_type = sws.active_workitem.path
    subject_command_name = session.workflow_snapshot.context["subject_command_name"]
    
    input_for_param_extraction_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    command_parameters_class = subject_command_routing_definition.get_command_class(
        active_workitem_type, subject_command_name, ModuleType.COMMAND_PARAMETERS_CLASS)
    
    input_for_param_extraction = input_for_param_extraction_class.create(sws, command)
    
    session_id = session.id
    stored_params = get_stored_parameters(session)
    
    if stored_params:
        is_valid, error_msg, suggestions, stored_missing_fields = extract_missing_fields(input_for_param_extraction, sws, stored_params)
    else:
        stored_missing_fields = []
    
    new_params = extract_command_parameters_from_input(
        input_for_param_extraction, 
        command_parameters_class,
        stored_missing_fields 
    )
    
    if stored_params:
        merged_params = merge_parameters(stored_params, new_params, stored_missing_fields)
    else:
        merged_params = new_params

    
    store_parameters(session, merged_params) 

    is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(sws, merged_params)
    

    if not is_valid:
        params_str = format_parameters_for_display(merged_params)
        
        if params_str:
            error_msg = f"Extracted parameters so far:\n{params_str}\n\n{error_msg}"

        error_msg += "\nEnter 'abort' if you want to abort the command."
        return OutputOfProcessCommand(parameter_is_valid=False, error_msg=error_msg, cmd_parameters=merged_params, suggestions=suggestions)
    
    clear_parameters(session)
    return OutputOfProcessCommand(parameter_is_valid=True, cmd_parameters=merged_params, suggestions={})


def extract_missing_fields(input_for_param_extraction, sws, stored_params):
    stored_missing_fields = []
    is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(sws, stored_params)
    
    if not is_valid:
        if MISSING_INFORMATION_ERRMSG in error_msg:
            missing_fields_str = error_msg.split(f"{MISSING_INFORMATION_ERRMSG}\n")[1].split("\n")[0]
            stored_missing_fields = [f.strip() for f in missing_fields_str.split(",")]
        if INVALID_INFORMATION_ERRMSG in error_msg:
            invalid_section = error_msg.split(f"{INVALID_INFORMATION_ERRMSG}\n")[1]
            if "\n" in invalid_section:
                invalid_fields_str = invalid_section.split("\n")[0]
                for invalid_field in invalid_fields_str.split(", "):
                    stored_missing_fields.append(invalid_field.split(" '")[0].strip())
    return is_valid, error_msg, suggestions, stored_missing_fields


def merge_parameters(old_params, new_params, missing_fields):
    """
    Merge new parameters with old parameters, prioritizing new values when appropriate.
    """
    merged = old_params.model_copy()
    
    all_fields = list(old_params.model_fields.keys())
    missing_fields = missing_fields or []
    
    for field_name in all_fields:
        if hasattr(new_params, field_name):
            new_value = getattr(new_params, field_name)
            old_value = getattr(merged, field_name)
            
            if new_value is not None and new_value != NOT_FOUND:
                if isinstance(old_value, str) and INVALID in old_value and INVALID not in new_value:
                    setattr(merged, field_name, new_value)

                elif old_value is None or old_value == NOT_FOUND:
                    setattr(merged, field_name, new_value)
                
                elif isinstance(old_value, int) and old_value == INVALID_INT_VALUE:
                    setattr(merged, field_name, new_value)

                elif isinstance(old_value, float) and old_value == INVALID_FLOAT_VALUE:
                    setattr(merged, field_name, new_value)
                
                elif (field_name in missing_fields and 
                      hasattr(merged.model_fields.get(field_name), "json_schema_extra") and 
                      merged.model_fields.get(field_name).json_schema_extra and 
                      "db_validation" in merged.model_fields.get(field_name).json_schema_extra):
                    setattr(merged, field_name, new_value)

                elif field_name in missing_fields:
                    field_info = merged.model_fields.get(field_name)
                    has_pattern = hasattr(field_info, "pattern") and field_info.pattern is not None
                    
                    if not has_pattern:
                        for meta in getattr(field_info, "metadata", []):
                            if hasattr(meta, "pattern"):
                                has_pattern = True
                                break
                    
                    if not has_pattern and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                        has_pattern = "pattern" in field_info.json_schema_extra
                    
                    if has_pattern:
                        setattr(merged, field_name, new_value)
    
    return merged

def format_parameters_for_display(params):
    """
    Format parameters for display in the error message.
    """
    if not params:
        return ""
        
    lines = []
    
    all_fields = list(params.model_fields.keys())
    
    for field_name in all_fields:
        value = getattr(params, field_name, None)
        
        if value is None or value == NOT_FOUND:
            continue
            
        display_name = " ".join(word.capitalize() for word in field_name.split('_'))
            
        # Format fields appropriately based on type
        if isinstance(value, bool):
            lines.append(f"{display_name}: {value}")
        elif hasattr(value, 'value'):  # Handle enum types
            lines.append(f"{display_name}: {value.value}")
        elif isinstance(value, (int, float)):
            lines.append(f"{display_name}: {value}")
        elif isinstance(value, str):
            lines.append(f"{display_name}: {value}")
        else:
            lines.append(f"{display_name}: {value}")
        
    return "\n".join(lines)


def extract_command_parameters_from_input(
    input_for_param_extraction: BaseModel,
    command_parameters_class: Type[BaseModel],
    missing_fields: list = None
) -> BaseModel:
    """
    Extract command parameters from user input.
    This implementation handles any parameter type.
    """
    # Initialize with default values based on field types and field definitions
    default_params = {}
    for field_name, field_info in command_parameters_class.model_fields.items():
        if field_info.default is not None and field_info.default is not Ellipsis:
            default_params[field_name] = field_info.default
        # Handle strings
        elif field_info.annotation == str:
            default_params[field_name] = NOT_FOUND
        elif field_info.annotation == int:
            default_params[field_name] = None 
        # Handle Optional[int]
        elif (hasattr(field_info.annotation, "__origin__") and 
              field_info.annotation.__origin__ is Union and
              int in field_info.annotation.__args__ and
              type(None) in field_info.annotation.__args__):
            default_params[field_name] = None
        else:
            default_params[field_name] = None
    
    default_params = command_parameters_class(**default_params)
    
    try:
        command = input_for_param_extraction.command
        
        if missing_fields:
            return apply_missing_fields(command, default_params, missing_fields)
                    
        if hasattr(input_for_param_extraction, 'extract_parameters'):
            try:
                return input_for_param_extraction.extract_parameters(command_parameters_class)
            except Exception as inner_e:
                logger.error(PARAMETER_EXTRACTION_ERROR_MSG.format(error=inner_e))
        
        try:
            dspy_signature_class = TypedPredictorSignature.create(
                input_for_param_extraction,
                command_parameters_class,
                prefix_instructions=input_for_param_extraction.__doc__,
            )
            lm = dspy.LM(DSPY_LM_MODEL)
            with dspy.context(lm=lm):
                extract_cmd_params = dspy.TypedChainOfThought(dspy_signature_class)
                prediction = extract_cmd_params(**input_for_param_extraction.model_dump())
                
                param_values = {}
                for field_name in command_parameters_class.model_fields:
                    if hasattr(prediction, field_name):
                        param_values[field_name] = getattr(prediction, field_name)
                    else:
                        param_values[field_name] = getattr(default_params, field_name)
                
                return command_parameters_class(**param_values)
        except Exception as dspy_error:
            logger.error(f"DSPy error: {dspy_error}")
            return default_params
            
    except Exception as e:
        logger.error(PARAMETER_EXTRACTION_ERROR_MSG.format(error=e))
        return default_params
    
def apply_missing_fields(command: str, default_params: BaseModel, missing_fields: list):
    params = default_params.model_copy()
    
    if "," in command:
        parts = [part.strip() for part in command.split(",")]
        
        if len(parts) == len(missing_fields):
            if len(missing_fields) == 1:
                field = missing_fields[0]
                if hasattr(params, field):
                    setattr(params, field, parts[0])
                    return params
            elif len(missing_fields) > 1:
                for i, field in enumerate(missing_fields):
                    if i < len(parts) and hasattr(params, field):
                        setattr(params, field, parts[i])
                return params
        else:
            if parts and missing_fields:
                field = missing_fields[0]
                if hasattr(params, field):
                    setattr(params, field, parts[0])         
            return params
                 
    # No commas in command
    else:
        if missing_fields:
            field = missing_fields[0]
            if hasattr(params, field):
                setattr(params, field, command.strip())
                return params