from string import Template
import os
from fastworkflow.build.pydantic_model_generator import generate_input_model_code, generate_output_model_code, generate_property_output_model_code
from fastworkflow.build.utterance_generator import generate_utterances
from fastworkflow.build.command_import_utils import generate_import_statements
from fastworkflow.build.class_analysis_structures import PropertyInfo
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

def get_import_block():
    return (
        f"from pydantic import ConfigDict\n\n"
        f"import fastworkflow\n"
        f"from fastworkflow import CommandOutput, CommandResponse\n"
        f"from fastworkflow.session import Session, WorkflowSnapshot\n"
        f"from fastworkflow.utils.signatures import InputForParamExtraction\n"
        f"from fastworkflow.train.generate_synthetic import generate_diverse_utterances\n"
        f"from fastworkflow.utils.context_utils import list_context_names\n"
        f"from typing import Any, Dict, Optional\n"
        f"from pydantic import BaseModel, Field\n"
    )

COMMAND_FILE_TEMPLATE = Template('''
${import_block}

class Signature:
    class Input(BaseModel):
${input_fields}
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
${output_fields}

    plain_utterances = [
${plain_utterances}
    ]

    template_utterances = [
${template_utterances}
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [command_name] + result
        return utterance_list

    def process_extracted_parameters(self, workflow_snapshot: WorkflowSnapshot, command: str, cmd_parameters: "Signature.Input") -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """${docstring}"""
        # Access the application class instance:
        app_instance = session.workflow_snapshot.context_object  # type: ${app_class_name}
${process_logic}
        return Signature.Output(${output_return})

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"${response_format}")
            ]
        )
''')

def create_command_file(class_info, method_info, output_dir, file_name=None, is_property_getter=False, is_property_setter=False, is_get_all_properties=False, all_properties_for_template: Optional[List[PropertyInfo]] = None, is_set_all_properties: bool = False, settable_properties_for_template: Optional[List[PropertyInfo]] = None, source_dir=None, overwrite=False, class_name_to_module_map: Optional[Dict[str, str]] = None):
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

    elif is_get_all_properties and all_properties_for_template:
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
        else:
            input_fields = "        pass" 
        
        output_fields = "        success: bool = Field(description=\"True if properties update was attempted.\")"
        
        set_logic_lines = []
        for prop_info in settable_properties_for_template:
            set_logic_lines.append(f"        if input.{prop_info.name} is not None:")
            set_logic_lines.append(f"            setattr(app_instance, '{prop_info.name}', input.{prop_info.name})")
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
        
        method_return_type = method_info.return_annotation or 'Any'
        regular_method_logic_lines = []
        param_names = [p['name'] for p in method_info.parameters] if method_info.parameters else []
        call_params = ', '.join([f"{p_name}=input.{p_name}" for p_name in param_names])
        
        if method_return_type.lower() == 'none' or not method_return_type:
            regular_method_logic_lines.append(f"        app_instance.{method_info.name.lower()}({call_params})")
            output_fields = '        success: bool = Field(default=True, description="Indicates successful execution.")'
            output_return = "success=True" 
            response_format = "success={output.success}"
        else:
            regular_method_logic_lines.append(f"        result_val = app_instance.{method_info.name.lower()}({call_params})")
            output_fields = f'        result: {method_return_type} = Field(description="Result of the method call")'
            output_return = "result=result_val" 
            response_format = "result={output.result}"
        process_logic_str = "\n".join(regular_method_logic_lines)
            
    # Utterances
    plain_utterances = ",\n".join([f'        "{u}"' for u in generate_utterances(class_info.name, method_info.name, method_info.parameters)])
    template_utterances = "        \"TODO: Add template utterances\""
    # Use new import block logic
    if source_dir is None:
        raise ValueError("source_dir must be provided to create_command_file")
    import_block = get_import_block()
    import_block += generate_import_statements(class_info, source_dir, class_name_to_module_path=class_name_to_module_map)
    # Fill the template
    command_file_content = COMMAND_FILE_TEMPLATE.substitute(
        import_block=import_block,
        input_fields=input_fields,
        output_fields=output_fields,
        plain_utterances=plain_utterances,
        template_utterances=template_utterances,
        docstring=docstring,
        process_logic=process_logic_str, # Added process_logic
        output_return=output_return,
        response_format=response_format,
        app_class_name=app_class_name
    )
    # Write the command file
    if file_name is None:
        file_name = f"{class_info.name.lower()}_{method_info.name.lower()}.py"
    file_path = os.path.join(output_dir, file_name)
    if not overwrite and os.path.exists(file_path):
        return file_path
    with open(file_path, 'w') as f:
        f.write(command_file_content)
    return file_path 