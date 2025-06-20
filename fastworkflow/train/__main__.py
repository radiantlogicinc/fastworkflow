import argparse
import os
import json
import shutil
from dotenv import dotenv_values

from colorama import Fore, Style
from fastworkflow.utils import python_utils
import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.model_pipeline_training import train, get_route_layer_filepath_model
from fastworkflow.utils.generate_param_examples import generate_dspy_examples
from fastworkflow.command_routing_definition import CommandRoutingDefinition
from fastworkflow.utterance_definition import UtteranceRegistry
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory


def train_workflow(workflow_path: str):
    # Ensure context model is parsed so downstream helpers have contexts
    CommandContextModel.load(workflow_path)
    CommandRoutingDefinition.build(workflow_path)
    # Ensure the command directory is persisted so that downstream helpers
    # (e.g. DSPy example generation) can read it from disk without first
    # needing to rebuild it in-memory.
    cmd_dir = CommandDirectory.load(workflow_path)
    cmd_dir.save()
    UtteranceRegistry.get_definition(workflow_path)

    #first, recursively train all child workflows
    workflows_dir = os.path.join(workflow_path, "_workflows")
    if os.path.isdir(workflows_dir):
        for child_workflow in os.listdir(workflows_dir):
            if "__pycache__" in child_workflow:
                continue
            child_workflow_path = os.path.join(workflows_dir, child_workflow)
            if os.path.isdir(child_workflow_path):
                print(f"{Fore.YELLOW}Training child workflow: {child_workflow_path}{Style.RESET_ALL}")
                train_workflow(child_workflow_path)

    if "fastworkflow" in workflow_path and "_workflows" not in workflow_path:
        return
    
    # create a session and train the main workflow
    session = fastworkflow.Session.create(
        workflow_path, 
        session_id_str=f"train_{workflow_path}", 
        for_training_semantic_router=True
    )

    _generate_dspy_examples_helper(workflow_path, session)
    
    train(session)
    session.close()

def _generate_dspy_examples_helper(workflow_path, session):
    workflow_folderpath = session.workflow_snapshot.workflow_folderpath
    json_path=get_route_layer_filepath_model(workflow_folderpath,"command_directory.json")
    # json_path = "./examples/sample_workflow/___command_info/command_directory.json"
    commands = _get_commands_with_parameters(json_path)
    for command_name in commands.keys():
        if command_name in ["wildcard"]:
            continue
        command_metadata = commands[command_name]
        module_file_path = command_metadata["parameter_path"]
        module_class_name = command_metadata["parameters_class"]

        # Import the module dynamically
        module = python_utils.get_module(module_file_path, workflow_path)
        if module:
            if "." in module_class_name:
                (outer, inner) = module_class_name.split(".")
                outer_cls = getattr(module, outer)
                fields = getattr(outer_cls, inner)
            else:
                fields = getattr(module, module_class_name)

            examples, rejected_examples = generate_dspy_examples(
            field_annotations=fields.model_fields,
            command_name=command_name,
            num_examples=15,
            validation_threshold=0.3  # You can adjust this threshold as needed
            )
            output_dir = os.path.join(workflow_path, "___command_info")
            os.makedirs(output_dir, exist_ok=True)
            
            # Format the examples for JSON
            examples_data = {
                "command_name": command_name,
                "valid_examples": examples,
                "rejected_examples": rejected_examples
            }
            
            # Save to JSON file
            output_file = os.path.join(output_dir, f"{command_name}_param_labeled.json")
            with open(output_file, 'w') as f:
                json.dump(examples_data, f, indent=2)
        else:
            None

def _get_commands_with_parameters(json_path):
    """
    Parse command_directory.json file and create a mapping between command names 
    and their parameter extraction signature module paths for commands that have
    a non-null command_parameters_class.
    
    Args:
        json_path: Path to the command_directory.json file
        
    Returns:
        dict: Dictionary mapping command names to parameter_extraction_signature_module_path
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        command_directory = json.load(f)
    
    # Extract the command metadata
    commands_metadata = command_directory.get("map_commandkey_2_metadata", {})
    
    # Initialize result dictionary
    commands_with_parameters = {}
    
    # Iterate through each command entry
    for command_key, metadata in commands_metadata.items():
        # Check if command_parameters_class is not null
        if metadata.get("command_parameters_class") is not None:
            # We want to train on the full command path including the ContextFolder prefix
            command_name = command_key.split("/")[-1]
            
            # Get the parameter extraction module path
            param_extraction_path = metadata.get("parameter_extraction_signature_module_path")
            
            # Add to result dictionary
            commands_with_parameters[command_name] = {
                "parameter_path": param_extraction_path,
                "full_command_key": command_key,
                "parameters_class": metadata.get("command_parameters_class"),
                "input_class": metadata.get("input_for_param_extraction_class")
            }
    
    return commands_with_parameters

def is_fast_workflow_trained(fastworkflow_folderpath: str):
    # Check if fastworkflow has been trained
    cme_commandinfo_folderpath = os.path.join(
        fastworkflow_folderpath,
        '_workflows',
        'command_metadata_extraction', 
        '___command_info',
        'ErrorCorrection'
    )

    return "largemodel.pth" in os.listdir(cme_commandinfo_folderpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the intent detection pipeline for a workflow"
    )
    parser.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    parser.add_argument("passwords_file_path", help="Path to the passwords file")
    args = parser.parse_args()

    if not os.path.isdir(args.workflow_folderpath):
        print(
            f"{Fore.RED}Error: The specified workflow path '{args.workflow_folderpath}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path)
    }
    fastworkflow.init(env_vars=env_vars)

    # Check if fastworkflow has been trained, and train it if not
    fastworkflow_folderpath = fastworkflow.get_fastworkflow_package_path()
    if (
        "fastworkflow" not in args.workflow_folderpath and
        not is_fast_workflow_trained(fastworkflow_folderpath)
    ):
        train_workflow(fastworkflow_folderpath)

    train_workflow(args.workflow_folderpath)