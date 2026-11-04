from string import Template
import os
import re
from pathlib import Path
from fastworkflow.build.pydantic_model_generator import generate_input_model_code, generate_output_model_code, generate_property_output_model_code
from fastworkflow.build.utterance_generator import generate_utterances
from fastworkflow.build.command_import_utils import generate_import_statements
from fastworkflow.build.class_analysis_structures import PropertyInfo, FunctionInfo
from fastworkflow.command_directory import _command_kind_from_ast
from fastworkflow.utils.logging import logger
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

GENERATED_FILE_HEADER = "# fastworkflow:generated — regenerate with fastworkflow build; edits may be overwritten"

def _is_planning_command_file(file_path: str) -> bool:
    """True when the existing file defines a top-level PlanGenerator (checked with ast, never imported)."""
    if not os.path.isfile(file_path):
        return False
    try:
        return _command_kind_from_ast(Path(file_path))[1]
    except SyntaxError:
        # Unparseable, but a half-edited planning command must not be overwritten: fall back to a line check
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return re.search(r'^class\s+PlanGenerator\b', text, re.MULTILINE) is not None
    except ValueError:
        # Defines both ResponseGenerator and PlanGenerator, or cannot be read: protect it
        return True

def _refuses_overwrite(file_path: str) -> bool:
    """Warn and return True when writing file_path must be refused because it is a planning command.
    A generated execution command with the same qualified name is the same path, so this covers it too."""
    if _is_planning_command_file(file_path):
        logger.warning(f"Not writing {file_path}: it is a planning command (defines PlanGenerator); generated execution commands do not overwrite or share a name with planning commands.")
        return True
    return False

_TYPING_NAMES = ("Any", "Dict", "List", "Optional", "Union", "Literal", "Tuple", "Set")


def get_import_block(body: str = "") -> str:
    """The imports the generated file needs, and only those: an unused import is a lint error in the file.

    The body is the rest of the file, so a name is imported when the code uses it (a parameter's
    annotation may name a typing object, and ``Field``/``ConfigDict`` appear only when the body does).
    """
    def used(name: str) -> bool:
        return re.search(rf"\b{name}\b", body) is not None

    lines = []
    if used("fastworkflow"):
        lines.append("import fastworkflow")
    lines.append("from fastworkflow.workflow import Workflow")
    typing_used = [name for name in _TYPING_NAMES if used(name)]
    if typing_used:
        lines.append(f"from typing import {', '.join(typing_used)}")
    pydantic_used = ["BaseModel"] + [name for name in ("Field", "ConfigDict") if used(name)]
    lines.append(f"from pydantic import {', '.join(pydantic_used)}")
    return "\n".join(lines) + "\n"

def create_function_command_file(function_info: FunctionInfo, output_dir: str, file_name: str = None, source_dir: str = None, overwrite: bool = False) -> str:
    """Generate a FastWorkflow command file for a global function."""
    # Determine file path
    if file_name is None:
        file_name = f"{function_info.name}.py"
    file_path = os.path.join(output_dir, file_name)

    # Check if file exists and overwrite is False
    if not overwrite and os.path.exists(file_path):
        return file_path

    # Initialize generation variables
    input_fields = ""
    output_fields = ""
    output_return = ""
    response_format = ""
    process_logic_str = ""
    docstring = function_info.docstring or f"Execute {function_info.name} function"
    has_input = True  # Default to having input

    # Process parameters for input model
    if function_info.parameters:
        input_field_lines = []
        for p in function_info.parameters:
            param_type = p.get('annotation') or 'Any'
            # No Field() - let GenAI postprocessor add it
            input_field_lines.append(f'        {p["name"]}: {param_type}')
        input_fields = "\n".join(input_field_lines)
    else:
        input_fields = "        pass"
        has_input = False  # No input needed for functions without parameters

    # Process return type for output model
    return_type = function_info.return_annotation or 'Any'
    if return_type.lower() == 'none' or not return_type:
        # No Field() - let GenAI postprocessor add it
        output_fields = '        success: bool'
        output_return = "success=True"
        response_format = "success={output.success}"
    else:
        # No Field() - let GenAI postprocessor add it
        output_fields = f'        result: {return_type}'
        output_return = "result=result_val"
        response_format = "result={output.result}"

    # Generate process logic
    param_names = [p['name'] for p in function_info.parameters] if function_info.parameters else []
    call_params = ', '.join([f"{p_name}=input.{p_name}" for p_name in param_names])

    if return_type.lower() == 'none' or not return_type:
        process_logic_str = (
            f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        {function_info.name}({call_params})"
            if param_names
            else f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        {function_info.name}()"
        )
    elif param_names:
        process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        result_val = {function_info.name}({call_params})"
    else:
        process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        result_val = {function_info.name}()"

    # Generate utterances
    plain_utterances = ",\n".join([f'        "{u}"' for u in generate_utterances(None, function_info.name, function_info.parameters, is_function=True)])

    # Build the command file content: one Signature and one ResponseGenerator whose __call__ returns the Output
    body = "class Signature:\n"
    input_annotation = "None"
    if has_input:
        body += f"    class Input(BaseModel):\n{input_fields}\n\n"
        input_annotation = "Signature.Input"
    body += f"    class Output(BaseModel):\n"
    body += f"        model_config = ConfigDict(arbitrary_types_allowed=True)\n"
    body += f"{output_fields}\n\n"
    body += f"    plain_utterances = [\n{plain_utterances}\n    ]\n\n"
    body += "class ResponseGenerator:\n"
    body += f"    def __call__(self, workflow: Workflow, command: str, input: {input_annotation}) -> Signature.Output:\n"
    body += f"        \"\"\"{docstring}\"\"\"\n"
    body += f"{process_logic_str}\n"
    body += f"        return Signature.Output({output_return})\n"
    command_file_content = GENERATED_FILE_HEADER + "\n" + get_import_block(body) + "\n\n" + body

    # Write the command file, unless it is a planning command (refused even with overwrite=True)
    if _refuses_overwrite(file_path):
        return file_path
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(command_file_content)

    return file_path

def create_command_file(class_info, method_info, output_dir, file_name=None, is_property_getter=False, is_property_setter=False, is_get_all_properties=False, all_properties_for_template: Optional[List[PropertyInfo]] = None, is_set_all_properties: bool = False, settable_properties_for_template: Optional[List[PropertyInfo]] = None, source_dir=None, overwrite=False, class_name_to_module_map: Optional[Dict[str, str]] = None, is_global_function: bool = False):
    """Generate a FastWorkflow command file matching the user's required structure."""
    import textwrap
    # Determine application module path and class name
    app_module_path = class_info.module_path.replace('/', '.').replace('.py', '')
    app_class_name = class_info.name
    # Initialize generation variables
    input_fields = ""
    output_fields = ""
    output_return = ""
    response_format = ""
    process_logic_str = "        # Default process logic if not overridden by specific command type" # Initialize process_logic_str
    docstring = method_info.docstring or f"Execute {method_info.name} method on {class_info.name}"
    has_input = True  # Default to having input
    needs_model_config = False  # Default to not needing model_config

    if is_property_getter and not is_get_all_properties: # Kept for potential individual getters, though get_properties is preferred
        input_fields = "        pass"
        prop_type = method_info.return_annotation or 'Any'
        # No Field() - let GenAI postprocessor add it
        output_fields = f'        value: {prop_type}'
        # For individual getter, process_logic retrieves the specific property
        # Ensure method_info.name for a getter is just the property name, or adjust access.
        # If method_info.name is like "Description", property name is method_info.name.lower()
        # For now, let's assume method_info.name for a getter refers to the actual property attribute name.
        actual_prop_name = method_info.name # This needs to be the actual attribute name
        process_logic_str = f"        # Individual getter logic: retrieve app_instance.{actual_prop_name}"
        output_return = f"value=app_instance.{actual_prop_name}"
        response_format = "value={output.value}"
        has_input = False  # Property getters don't need input

    elif is_property_setter and not is_set_all_properties: # Individual property setter
        # Use the property name as the input parameter name
        prop_name = method_info.name
        if method_info.parameters and len(method_info.parameters) > 0:
            param_type = method_info.parameters[0].get('annotation') or 'Any'
            # No Field() - let GenAI postprocessor add it
            input_fields = f'        {prop_name}: {param_type}'
        else:
            # No Field() - let GenAI postprocessor add it
            input_fields = f"        {prop_name}: Any"

        # No Field() - let GenAI postprocessor add it
        output_fields = '        success: bool'

        # Use attribute assignment for property setter
        process_logic_str = f"        # Set property using attribute assignment\n        app_instance.{prop_name} = input.{prop_name}"

        output_return = "success=True"
        response_format = "success={output.success}"

    elif is_get_all_properties and all_properties_for_template:
        has_input = False  # get_properties doesn't need input
        input_fields = "        pass" # No input for get_properties
        output_field_lines = []
        output_return_parts = []
        for prop in all_properties_for_template:
            prop_type = prop.type_annotation or 'Any'
            # No Field() - let GenAI postprocessor add it
            output_field_lines.append(f'        {prop.name}: {prop_type}')
            output_return_parts.append(f"{prop.name}=app_instance.{prop.name}")

        if not output_field_lines:
            output_fields = "        pass # No properties defined"
            output_return = ""
        else:
            output_fields = "\n".join(output_field_lines)
            output_return = ", ".join(output_return_parts)

        response_format = "properties={output.dict()}" # Example response format
        # docstring is already set from the synthesized method_info for GetProperties

        process_logic_lines = [
            "        # For get_properties, the primary logic is to gather attribute values,",
            "        # which is handled by constructing the output_return string that references app_instance attributes directly.",
            "        # No additional complex processing steps are typically needed in this block.",
            "        pass # Placeholder if no other pre-return logic is needed"
        ]
        process_logic_str = "\n".join(process_logic_lines)

    elif is_set_all_properties and settable_properties_for_template:
        input_field_lines = []
        for prop_info in settable_properties_for_template:
            # No Field() - let GenAI postprocessor add it
            input_field_lines.append(f"        {prop_info.name}: Optional[{prop_info.type_annotation}] = None")
        if input_field_lines:
            input_fields = "\n".join(input_field_lines)
            needs_model_config = True  # Need model_config for Optional fields
        else:
            input_fields = "        pass" 

        # No Field() - let GenAI postprocessor add it
        output_fields = "        success: bool"

        set_logic_lines = []
        for prop_info in settable_properties_for_template:
            set_logic_lines.extend(
                (
                    f"        if input.{prop_info.name} is not None:",
                    f"            app_instance.{prop_info.name} = input.{prop_info.name}",
                )
            )
        set_logic_lines.extend(
            (
                "        if input.is_complete is not None:",
                f"            app_instance.status = {class_info.name}.COMPLETE if input.is_complete else {class_info.name}.INCOMPLETE",
            )
        )
        if not set_logic_lines: 
            set_logic_lines.append("        pass # No properties to set or no inputs provided")
        process_logic_str = "\n".join(set_logic_lines)

        output_return = "success=True"
        response_format = "Set properties result: {output.success}"
                # docstring is already set from the synthesized method_info for SetProperties

    else: # For regular methods (default case)
        if method_info.parameters:
            input_field_lines = []
            for p in method_info.parameters:
                param_type = p.get('annotation') or 'Any'
                # No Field() - let GenAI postprocessor add it
                input_field_lines.append(f'        {p["name"]}: {param_type}')
            input_fields = "\n".join(input_field_lines)
        else:
            input_fields = "        pass"
            has_input = False  # No input needed for methods without parameters

        method_return_type = method_info.return_annotation or 'Any'
        regular_method_logic_lines = []
        param_names = [p['name'] for p in method_info.parameters] if method_info.parameters else []
        call_params = ', '.join([f"{p_name}=input.{p_name}" for p_name in param_names])

        if method_return_type.lower() == 'none' or not method_return_type:
            if param_names:
                regular_method_logic_lines.append(f"        app_instance.{method_info.name.lower()}({call_params})")
            else:
                regular_method_logic_lines.append(f"        app_instance.{method_info.name.lower()}()")
            # No Field() - let GenAI postprocessor add it
            output_fields = '        success: bool'
            output_return = "success=True" 
            response_format = "success={output.success}"
        else:
            if param_names:
                regular_method_logic_lines.append(f"        result_val = app_instance.{method_info.name.lower()}({call_params})")
            else:
                regular_method_logic_lines.append(f"        result_val = app_instance.{method_info.name.lower()}()")
            # No Field() - let GenAI postprocessor add it
            output_fields = f'        result: {method_return_type}'
            output_return = "result=result_val" 
            response_format = "result={output.result}"
        process_logic_str = "\n".join(regular_method_logic_lines)

    # Utterances
    plain_utterances = ",\n".join([f'        "{u}"' for u in generate_utterances(class_info.name, method_info.name, method_info.parameters)])

    # Use new import block logic
    if source_dir is None:
        raise ValueError("source_dir must be provided to create_command_file")
    app_import_source = generate_import_statements(class_info, source_dir, class_name_to_module_path=class_name_to_module_map)

    # Prepare conditional sections
    if has_input:
        # Input class without model_config
        input_class = f"    class Input(BaseModel):\n{input_fields}\n"
        input_param_type = '"Signature.Input"'
        input_param = ", input: Signature.Input"
        call_param = ", input: Signature.Input"
        call_arg = ", input"
    else:
        # No Input class
        input_class = ""
        input_param_type = "None"
        input_param = ""
        call_param = ""
        call_arg = ""

    # Build the command file content: one Signature and one ResponseGenerator whose __call__ returns the Output
    body = "class Signature:\n"
    input_annotation = "None"
    if has_input:
        body += f"    class Input(BaseModel):\n{input_fields}\n\n"
        input_annotation = "Signature.Input"
    body += f"    class Output(BaseModel):\n"
    body += f"        model_config = ConfigDict(arbitrary_types_allowed=True)\n"
    body += f"{output_fields}\n\n"
    body += f"    plain_utterances = [\n{plain_utterances}\n    ]\n\n"
    body += "class ResponseGenerator:\n"
    body += f"    def __call__(self, workflow: Workflow, command: str, input: {input_annotation}) -> Signature.Output:\n"
    body += f"        \"\"\"{docstring}\"\"\"\n"
    body += f"        app_instance: {app_class_name} = workflow.command_context_for_response_generation\n"
    body += f"{process_logic_str}\n"
    body += f"        return Signature.Output({output_return})\n"
    app_imports = [
        line for line in app_import_source.splitlines()
        if line.startswith("# You may need") or (line.startswith("from ...") and line.rsplit(" ", 1)[-1] in body)
    ]
    import_text = get_import_block(body) + "".join(f"{line}\n" for line in app_imports)
    command_file_content = GENERATED_FILE_HEADER + "\n" + import_text + "\n\n" + body

    # Write the command file
    if file_name is None:
        file_name = f"{class_info.name.lower()}_{method_info.name.lower()}.py"
    file_path = os.path.join(output_dir, file_name)
    if not overwrite and os.path.exists(file_path):
        return file_path
    # Refuse to write over a planning command (even with overwrite=True) or to collide with one
    if _refuses_overwrite(file_path):
        return file_path
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(command_file_content)
    return file_path 