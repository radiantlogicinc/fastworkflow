import dspy
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, Any, Union, get_args, get_origin, Tuple, List

import fastworkflow
from fastworkflow.utils.logging import logger

def get_lm(model_env_var: str, api_key_env_var: Optional[str] = None, **kwargs):
    """get the dspy lm object"""
    model = fastworkflow.get_env_var(model_env_var)
    if not model:
        logger.critical(f"Critical Error:DSPy Language Model not provided. Set {model_env_var} environment variable.")
        raise ValueError(f"DSPy Language Model not provided. Set {model_env_var} environment variable.")
    api_key = fastworkflow.get_env_var(api_key_env_var) if api_key_env_var else None
    return dspy.LM(model=model, api_key=api_key, **kwargs) if api_key else dspy.LM(model=model, **kwargs)

def _process_field(field_info, is_input: bool) -> Tuple[Any, Any, bool]:
    """Process a single field and return its type, DSPy field, and optional status."""
    field_type = field_info.annotation
    field_desc = field_info.description or f"{'Input' if is_input else 'Output'} field"
    
    # Handle Optional types
    is_optional = False
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if type(None) in args:
            is_optional = True
            field_type = next((t for t in args if t is not type(None)), str)
    
    dspy_field = dspy.InputField(desc=field_desc) if is_input else dspy.OutputField(desc=field_desc)
    return field_type, dspy_field, is_optional


def _process_input_fields(model_class: Type[BaseModel], preserve_types: bool) -> Dict[str, Tuple]:
    """Process all input fields from a Pydantic model."""
    fields = {}
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=True)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
    return fields


def _process_output_fields(model_class: Type[BaseModel], preserve_types: bool) -> Tuple[Dict[str, Tuple], List[str]]:
    """Process all output fields from a Pydantic model and generate instructions."""
    fields = {}
    instructions = []
    
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=False)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
        
        # Generate field-specific instructions
        _add_field_instructions(field_name, field_info, instructions)
        
    return fields, instructions


def _add_field_instructions(field_name: str, field_info, instructions: List[str]) -> None:
    """Add instructions for a specific field based on its metadata."""
    if hasattr(field_info, 'default') and field_info.default is not None:
        instructions.append(f"For '{field_name}': Use '{field_info.default}' if not explicitly mentioned.")
    
    if hasattr(field_info, 'examples') and field_info.examples:
        examples_str = ", ".join(f"'{ex}'" for ex in field_info.examples)
        instructions.append(f"Examples for '{field_name}': {examples_str}")


def _create_instructions(custom_instructions: Optional[str], auto_instructions: List[str]) -> str:
    """Create the final instruction string from custom and auto-generated instructions."""
    if custom_instructions:
        return custom_instructions
    
    if not auto_instructions:
        return ""
        
    return "Extract the following fields based on the input:\n\n" + "\n".join(auto_instructions)


def dspySignature(
    Input_class: Type[BaseModel], 
    Output_class: Type[BaseModel],
    instructions: Optional[str] = None,
    preserve_types: bool = True
) -> Type[dspy.Signature]:
    """
    Dynamically creates a dspy.Signature class from Pydantic Input and Output models.

    Args:
        Input_class: A Pydantic BaseModel class defining the input fields.
        Output_class: A Pydantic BaseModel class defining the output fields.
        instructions: Optional custom instructions to include in the signature.
        preserve_types: Whether to preserve field type annotations in the signature.

    Returns:
        A new class that inherits from dspy.Signature.
    """
    if not issubclass(Input_class, BaseModel) or not issubclass(Output_class, BaseModel):
        raise TypeError("Input_class and Output_class must be subclasses of pydantic.BaseModel.")

    # Process fields from both classes
    input_fields = _process_input_fields(Input_class, preserve_types)
    output_fields, auto_instructions = _process_output_fields(Output_class, preserve_types)
    
    # Combine all fields and create instructions
    dspy_fields = {**input_fields, **output_fields}
    final_instructions = _create_instructions(instructions, auto_instructions)
    
    return dspy.Signature(dspy_fields, final_instructions.strip())

############################################
# Steps:
# 1. Define your signature and dspy function
# 2. Get prediction from DSPy module
# 3. Create output directly using ** unpacking

# dspy_signature_class = dspySignature(Signature.Input, Signature.Output)
# dspy_predict_func = dspy.Predict(dspy_signature_class)
# prediction = dspy_predict_func(input)  # Returns a dspy.Prediction object
# return Signature.Output(**prediction)