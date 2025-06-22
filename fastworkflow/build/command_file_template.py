from string import Template
import os
from fastworkflow.build.pydantic_model_generator import generate_input_model_code, generate_output_model_code, generate_property_output_model_code
from fastworkflow.build.utterance_generator import generate_utterances
from fastworkflow.build.command_import_utils import generate_import_statements
from fastworkflow.build.class_analysis_structures import PropertyInfo, FunctionInfo
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

def get_import_block():
    return (
        f"import fastworkflow\n"
        f"from fastworkflow import CommandOutput, CommandResponse\n"
        f"from fastworkflow.session import Session, WorkflowSnapshot\n"
        f"from fastworkflow.utils.signatures import InputForParamExtraction\n"
        f"from fastworkflow.train.generate_synthetic import generate_diverse_utterances\n"
        f"from fastworkflow.utils.context_utils import list_context_names\n"
        f"from typing import Any, Dict, Optional\n"
        f"from pydantic import BaseModel, Field\n"
    )

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
            escaped_param_desc = (p.get('docstring') or f'Parameter {p["name"]}').replace('"', '\\"')
            input_field_lines.append(f'        {p["name"]}: {param_type} = Field(description="{escaped_param_desc}")')
        input_fields = "\n".join(input_field_lines)
    else:
        input_fields = "        pass"
        has_input = False  # No input needed for functions without parameters
    
    # Process return type for output model
    return_type = function_info.return_annotation or 'Any'
    if return_type.lower() == 'none' or not return_type:
        output_fields = '        success: bool = Field(default=True, description="Indicates successful execution.")'
        output_return = "success=True"
        response_format = "success={output.success}"
    else:
        output_fields = f'        result: {return_type} = Field(description="Result of the function call")'
        output_return = "result=result_val"
        response_format = "result={output.result}"
    
    # Generate process logic
    param_names = [p['name'] for p in function_info.parameters] if function_info.parameters else []
    call_params = ', '.join([f"{p_name}=input.{p_name}" for p_name in param_names])
    
    if return_type.lower() == 'none' or not return_type:
        if param_names:
            process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        {function_info.name}({call_params})"
        else:
            process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        {function_info.name}()"
    else:
        if param_names:
            process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        result_val = {function_info.name}({call_params})"
        else:
            process_logic_str = f"        # Call the function\n        from ..{os.path.basename(source_dir)}.{os.path.basename(function_info.module_path).replace('.py', '')} import {function_info.name}\n        result_val = {function_info.name}()"
    
    # Generate utterances
    plain_utterances = ",\n".join([f'        "{u}"' for u in generate_utterances(None, function_info.name, function_info.parameters, is_function=True)])
    
    # Build the command file content
    command_file_content = "\n" + get_import_block() + "\n\n" + "class Signature:\n"
    
    # Add Input class if needed
    if has_input:
        command_file_content += f"    class Input(BaseModel):\n{input_fields}\n\n"
        input_param_type = '"Signature.Input"'
        input_param = ", input: Signature.Input"
        call_param = ", command_parameters: Signature.Input"
        call_arg = ", command_parameters"
    else:
        input_param_type = "None"
        input_param = ""
        call_param = ""
        call_arg = ""
    
    # Add Output class
    command_file_content += f"    class Output(BaseModel):\n{output_fields}\n\n"
    
    # Add utterances
    command_file_content += f"    plain_utterances = [\n{plain_utterances}\n    ]\n\n"
    command_file_content += f"    template_utterances = []\n\n"
    
    # Add generate_utterances method
    command_file_content += """    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + result
        return utterance_list\n\n"""
    
    # Add process_extracted_parameters method
    command_file_content += f"    def process_extracted_parameters(self, workflow_snapshot: WorkflowSnapshot, command: str, cmd_parameters: {input_param_type}) -> None:\n"
    command_file_content += "        pass\n\n"
    
    # Add ResponseGenerator class
    command_file_content += "class ResponseGenerator:\n"
    command_file_content += f"    def _process_command(self, session: Session{input_param}) -> Signature.Output:\n"
    command_file_content += f"        \"\"\"{docstring}\"\"\"\n"
    command_file_content += f"{process_logic_str}\n"
    command_file_content += f"        return Signature.Output({output_return})\n\n"
    
    # Add __call__ method
    command_file_content += f"    def __call__(self, session: Session, command: str{call_param}) -> CommandOutput:\n"
    command_file_content += f"        output = self._process_command(session{call_arg})\n"
    command_file_content += """        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
"""
    
    # Write the command file
    with open(file_path, 'w') as f:
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
        output_fields = f'        value: {prop_type} = Field(description="Value of {method_info.name[4:]}")' # Assuming name is Get<PropName>
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
            escaped_param_desc = (method_info.parameters[0].get('docstring') or f'New value for {prop_name}').replace('"', '\\"')
            input_fields = f'        {prop_name}: {param_type} = Field(description="{escaped_param_desc}")'
        else:
            input_fields = f"        {prop_name}: Any = Field(description=\"New value for {prop_name}\")"

        output_fields = '        success: bool = Field(default=True, description="Indicates successful execution.")'

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
            # Ensure newlines in docstrings are also escaped for the description string
            escaped_prop_doc = (prop.docstring or f'Value of property {prop.name}').replace('"', '\\"').replace('\n', '\\\\n')
            output_field_lines.append(f'        {prop.name}: {prop_type} = Field(description="{escaped_prop_doc}")')
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
            escaped_docstring = (prop_info.docstring or f'Optional. New value for {prop_info.name}.').replace('"', '\\"')
            input_field_lines.append(f"        {prop_info.name}: Optional[{prop_info.type_annotation}] = Field(default=None, description=\"{escaped_docstring}\")")
        if input_field_lines:
            input_fields = "\n".join(input_field_lines)
            needs_model_config = True  # Need model_config for Optional fields
        else:
            input_fields = "        pass" 

        output_fields = "        success: bool = Field(description=\"True if properties update was attempted.\")"

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
                escaped_param_desc = (p.get('docstring') or f'Parameter {p["name"]}').replace('"', '\\"')
                input_field_lines.append(f'        {p["name"]}: {param_type} = Field(description="{escaped_param_desc}")')
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
            output_fields = '        success: bool = Field(default=True, description="Indicates successful execution.")'
            output_return = "success=True" 
            response_format = "success={output.success}"
        else:
            if param_names:
                regular_method_logic_lines.append(f"        result_val = app_instance.{method_info.name.lower()}({call_params})")
            else:
                regular_method_logic_lines.append(f"        result_val = app_instance.{method_info.name.lower()}()")
            output_fields = f'        result: {method_return_type} = Field(description="Result of the method call")'
            output_return = "result=result_val" 
            response_format = "result={output.result}"
        process_logic_str = "\n".join(regular_method_logic_lines)

    # Utterances
    plain_utterances = ",\n".join([f'        "{u}"' for u in generate_utterances(class_info.name, method_info.name, method_info.parameters)])
    template_utterances = "[]"

    # Use new import block logic
    if source_dir is None:
        raise ValueError("source_dir must be provided to create_command_file")
    import_block = get_import_block()
    import_block += generate_import_statements(class_info, source_dir, class_name_to_module_path=class_name_to_module_map)

    # Prepare conditional sections
    if has_input:
        # Input class without model_config
        input_class = f"    class Input(BaseModel):\n{input_fields}\n"
        input_param_type = '"Signature.Input"'
        input_param = ", input: Signature.Input"
        call_param = ", command_parameters: Signature.Input"
        call_arg = ", command_parameters"
    else:
        # No Input class
        input_class = ""
        input_param_type = "None"
        input_param = ""
        call_param = ""
        call_arg = ""

    # Build the command file content using string concatenation instead of template formatting
    command_file_content = "\n" + import_block + "\n\n" + "class Signature:\n"

    # Add Input class if needed
    command_file_content += input_class

    # Add Output class
    command_file_content += f"    class Output(BaseModel):\n{output_fields}\n\n"

    # Add utterances
    command_file_content += f"    plain_utterances = [\n{plain_utterances}\n    ]\n\n"
    command_file_content += f"    template_utterances = {template_utterances}\n\n"

    # Add generate_utterances method
    command_file_content += """    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + result
        return utterance_list\n\n"""

    # Add process_extracted_parameters method
    command_file_content += f"    def process_extracted_parameters(self, workflow_snapshot: WorkflowSnapshot, command: str, cmd_parameters: {input_param_type}) -> None:\n"
    command_file_content += "        pass\n\n"

    # Add ResponseGenerator class
    command_file_content += "class ResponseGenerator:\n"
    command_file_content += f"    def _process_command(self, session: Session{input_param}) -> Signature.Output:\n"
    command_file_content += f"        \"\"\"{docstring}\"\"\"\n"
    command_file_content += f"        app_instance = session.command_context_for_response_generation  # type: {app_class_name}\n"
    command_file_content += f"{process_logic_str}\n"
    command_file_content += f"        return Signature.Output({output_return})\n\n"

    # Add __call__ method
    command_file_content += f"    def __call__(self, session: Session, command: str{call_param}) -> CommandOutput:\n"
    command_file_content += f"        output = self._process_command(session{call_arg})\n"
    command_file_content += """        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
"""

    # Write the command file
    if file_name is None:
        file_name = f"{class_info.name.lower()}_{method_info.name.lower()}.py"
    file_path = os.path.join(output_dir, file_name)
    if not overwrite and os.path.exists(file_path):
        return file_path
    with open(file_path, 'w') as f:
        f.write(command_file_content)
    return file_path 