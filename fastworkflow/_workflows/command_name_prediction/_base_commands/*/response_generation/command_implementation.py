from typing import Optional
from pydantic import BaseModel
import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.model_pipeline_training import predict_single_sentence
import json
import os
from speedict import Rdict
from fastworkflow.cache_matching import get_flag, change_flag, store_utterance_cache, cache_match
import re
import Levenshtein
from fastworkflow.utils.fuzzy_match import find_best_match
# Flag values:
# 0 - Normal state (no constraints)
# 1 - Ambiguous command (low confidence, multiple potential commands)
# 2 - Misclassified command (user selected "None_of_these")

class CommandParameters(BaseModel):
    command_name: Optional[str] = None

class OutputOfProcessCommand(BaseModel):
    command_name: Optional[str] = None
    command: Optional[str] = None
    error_msg: Optional[str] = None

def get_cache_path(session_id, convo_path):
    """
    Generate cache file path based on session ID
    """
    base_dir = convo_path
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{session_id}.db")

def get_cache_path_cache(convo_path):
    """
    Generate cache file path based on session ID
    """
    base_dir = convo_path
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "cache.db")

# Store the suggested commands with the flag type
def store_suggested_commands(cache_path, command_list, flag_type):
    """
    Store the list of suggested commands for the constrained selection
    
    Args:
        cache_path: Path to the cache database
        command_list: List of suggested commands
        flag_type: Type of constraint (1=ambiguous, 2=misclassified)
    """
    db = Rdict(cache_path)
    try:
        db["suggested_commands"] = command_list
        db["flag_type"] = flag_type
    finally:
        db.close()

# Get the suggested commands
def get_suggested_commands(cache_path):
    """
    Get the list of suggested commands for the constrained selection
    """
    db = Rdict(cache_path)
    try:
        return db.get("suggested_commands", [])
    finally:
        db.close()

# Get the flag type
def get_flag_type(cache_path):
    """
    Get the type of constraint (1=ambiguous, 2=misclassified)
    """
    db = Rdict(cache_path)
    try:
        return db.get("flag_type", 1)  # Default to ambiguous if not set
    finally:
        db.close()

def get_route_layer_filepath(workflow_folderpath, model_name) -> str:
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        model_name
    )

def process_command(
    session: fastworkflow.Session, command: str
) -> OutputOfProcessCommand:
    sws = session.workflow_snapshot.context["intent_detection_sws"]
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_session_id = sws.session_id

    convo_path = os.path.join(sws_workflow_folderpath, "___convo_info")
    valid_command_names = get_valid_command_names(sws)

    # Check if the entire command is a valid command name
    normalized_command = command.replace(" ", "_").lower()
    command_name = next(
        (
            name
            for name in valid_command_names
            if normalized_command == name.lower()
        ),
        None,
    )

    def get_count(cache_path):
        db = Rdict(cache_path)
        try:
            return db.get("utterance_count")
        finally:
            db.close()

    def print_db_contents(cache_path):
        db = Rdict(cache_path)
        try:
            print("All keys in database:", list(db.keys()))
            for key in db.keys():
                print(f"Key: {key}, Value: {db[key]}")
        finally:
            db.close()

    def store_utterance(cache_path, utterance, label):
        """
        Store utterance in existing or new database
        Returns: The utterance count used
        """
        # Open the database (creates if doesn't exist)
        db = Rdict(cache_path)

        try:
            # Get existing counter or initialize to 0
            utterance_count = db.get("utterance_count", 0)

            # Create and store the utterance entry
            utterance_data = {
                "utterance": utterance,
                "label": label
            }

            db[utterance_count] = utterance_data

            # Increment and store the counter
            utterance_count += 1
            db["utterance_count"] = utterance_count

            return utterance_count - 1  # Return the count used for this utterance

        finally:
            # Always close the database
            db.close()

    # Function to read from database
    def read_utterance(cache_path, utterance_id):
        """
        Read a specific utterance from the database
        """
        db = Rdict(cache_path)
        try:
            return db.get(utterance_id)['utterance']
        finally:
            db.close()


    cache_path = get_cache_path(sws_session_id, convo_path)
    tiny_path = get_route_layer_filepath(sws_workflow_folderpath, "tinymodel.pth")
    large_path = get_route_layer_filepath(sws_workflow_folderpath, "largemodel.pth")

    threshold_path = get_route_layer_filepath(sws_workflow_folderpath, "threshold.json")
    ambiguous_threshold_path = get_route_layer_filepath(sws_workflow_folderpath, "ambiguous_threshold.json")
    with open(threshold_path, 'r') as f:
        data = json.load(f)
        confidence_threshold = data['confidence_threshold']

    with open(ambiguous_threshold_path, 'r') as f:
        data = json.load(f)
        ambiguos_confidence_threshold = data['confidence_threshold']

    modelpipeline = fastworkflow.modelpipelineregistry(
        tiny_model_path=tiny_path,
        distil_model_path=large_path,
        confidence_threshold=confidence_threshold       
    )

    path = get_cache_path_cache(convo_path)
    flag = get_flag(path)

    # Check if we're in constrained mode (flag != 0)
    if flag not in [0, None]:
        flag_type = get_flag_type(path)

        # Special cases: allow abort or "None of these" without @ prefix
        if command.lower() == "abort" or normalized_command == "abort":
            command_name = "abort"
            change_flag(path, 0)  # Reset flag
        elif command.lower() in {
            "none of these",
            "none of the above",
            "neither",
            "none",
        }:
            # User wants to see all options instead of the top 3
            error_msg = formulate_misclassified_command_error_message(valid_command_names)
            # Set flag to 2 because user is indicating none of the suggestions match
            store_suggested_commands(path, valid_command_names, 2)
            change_flag(path, 2)
            return OutputOfProcessCommand(error_msg=error_msg)
        else:
            # Only accept commands prefixed with @ that match the suggested commands
            suggested_commands = get_suggested_commands(path)

            # Create appropriate message based on flag type
            message_prefix = "The command is ambiguous" if flag_type == 1 else "The previous command was misclassified"

            if "@" in command:
                # Extract everything after @ until the next @ or end of string
                full_command_text = command.split("@", 1)[1]
                after_at = full_command_text.split("@", 1)[0] if "@" in full_command_text else full_command_text
                after_at = after_at.strip()

                # Get the first part for reference (still needed for command parts)
                tentative_command_name = after_at.split()[0] if " " in after_at else after_at

                # Try different matching strategies in order of strictness
            valid_choice = False
            matched_command = None

            # Use Levenshtein distance for fuzzy matching with the full command part after @
            matched_command, distance = find_best_match(
                command, 
                suggested_commands,
                threshold=0.3  # Adjust threshold as needed
            )
            valid_choice = matched_command is not None

            if valid_choice:
                command_name = matched_command
                # Remove the entire @command part from the input
                command = command
            else:
                # User selected an option that wasn't in the suggested list
                error_msg = f"{message_prefix}. Please select only from the provided command options:\n"
                error_msg += "\n".join(f"@{name}" for name in suggested_commands)
                error_msg += "\n\nor type 'None of these' to see all commands\nor type 'abort' to cancel"
                return OutputOfProcessCommand(error_msg=error_msg)


            # Process the selected command
            count = get_count(cache_path)
            utterance = read_utterance(cache_path, count-1)
            store_utterance_cache(path, utterance, command_name, modelpipeline)  
            change_flag(path, 0)  # Reset flag

            # If user selects None_of_these in constrained mode, show all valid commands
            if command_name == "None_of_these":
                error_msg = formulate_misclassified_command_error_message(valid_command_names)
                # Set flag to 2 because this is explicitly a misclassification case
                store_suggested_commands(path, valid_command_names, 2)
                change_flag(path, 2)
                return OutputOfProcessCommand(error_msg=error_msg)
    else:
        # Normal flow (not in constrained mode)

        # If user explicitly selects None_of_these, treat as misclassification
        if command_name == "None_of_these":
            error_msg = formulate_misclassified_command_error_message(valid_command_names)
            # Set flag to 2 because user is indicating previous command was misclassified
            store_suggested_commands(path, valid_command_names, 2)
            change_flag(path, 2)
            return OutputOfProcessCommand(error_msg=error_msg)

        if command.startswith('@'):
            tentative_command_name = command.split("@")[1].split()[0].rstrip(':-')
            normalized_command_name = tentative_command_name.lower()
            for name in valid_command_names:
                if normalized_command_name == name.lower():
                    command_name = name
                    command = command.replace(f"@{tentative_command_name}", "").strip().replace("  ", " ")
                    break
            if command_name == "None_of_these":
                error_msg = formulate_misclassified_command_error_message(valid_command_names)
                # Set flag to 2 because user is indicating previous command was misclassified
                store_suggested_commands(path, valid_command_names, 2)
                change_flag(path, 2)
                return OutputOfProcessCommand(error_msg=error_msg)

        if not command_name:
            # Try to find a match in the cache
            cache_result = cache_match(path, command, modelpipeline, 0.85)

            if cache_result:
                command_name = cache_result
                flag = get_flag(path)
                if flag is not None and flag != 0:
                    count = get_count(cache_path)
                    utterance = read_utterance(cache_path, count-1)
                    store_utterance_cache(path, utterance, command_name, modelpipeline)  
                    change_flag(path, 0)         
            else:
                # If no cache match, use the model to predict
                results = predict_single_sentence(modelpipeline, command, sws_workflow_folderpath)
                command_name = results['label']

                # If confidence is low, treat as ambiguous command (type 1)
                if results['confidence'] < ambiguos_confidence_threshold:
                    error_msg = formulate_ambiguous_command_error_message(results["topk_labels"])
                    count = store_utterance(cache_path, command, command_name)
                    # Store suggested commands and set flag to 1 (ambiguous)
                    store_suggested_commands(path, results["topk_labels"], 1)
                    change_flag(path, 1)
                    return OutputOfProcessCommand(error_msg=error_msg)

                # If model prediction is None_of_these, present all commands as options
                if command_name == "None_of_these":
                    error_msg = formulate_misclassified_command_error_message(valid_command_names)
                    # Set flag to 2 because model couldn't classify the command
                    store_suggested_commands(path, valid_command_names, 2)
                    change_flag(path, 2)
                    return OutputOfProcessCommand(error_msg=error_msg)
                else:
                    flag = get_flag(path)
                    if flag is not None and flag != 0:
                        count = get_count(cache_path)
                        utterance = read_utterance(cache_path, count-1)
                        store_utterance_cache(path, utterance, command_name, modelpipeline)  
                        change_flag(path, 0) 
        else:
            # When the command_name is already determined
            flag = get_flag(path)
            if flag is not None and flag != 0:
                count = get_count(cache_path)
                utterance = read_utterance(cache_path, count-1)
                store_utterance_cache(path, utterance, command_name, modelpipeline)  
                change_flag(path, 0)         

    # Store the final command and classification
    if command_name != "None_of_these":
        count = store_utterance(cache_path, command, command_name)

    command_parameters = CommandParameters(command_name=command_name)
    is_valid, error_msg = validate_command_name(valid_command_names, command_parameters)

    # If validation fails, set flag to 2 (misclassified)
    if not is_valid:
        store_suggested_commands(path, valid_command_names, 2)
        change_flag(path, 2)
        return OutputOfProcessCommand(error_msg=error_msg)

    return OutputOfProcessCommand(
        command_name=command_parameters.command_name,
        command=command
    )   

def get_valid_command_names(sws: WorkflowSnapshot) -> set[str]:
    valid_command_names = {'None_of_these', 'abort'}
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(sws_workflow_folderpath)
    valid_command_names |= set(sws_command_routing_definition.get_command_names(
        sws.active_workitem.path
    ))
    # valid_command_names.remove("*")
    return valid_command_names

def validate_command_name(
    valid_command_names: set[str],
    command_parameters: CommandParameters
) -> tuple[bool, str]:
    if command_parameters.command_name in valid_command_names:
        return (True, None)

    if not command_parameters.command_name and "*" in valid_command_names:
        command_parameters.command_name = "*"
        return (True, None)

    command_list = "\n".join(f"@{name}" for name in valid_command_names)
    return (
        False,
        "The command was misclassified. Please select the correct command from the list below:\n"
        f"{command_list}\n\nor type 'abort' to cancel"
    )

def formulate_ambiguous_command_error_message(route_choice_list: list[str]) -> str:
    command_list = (
        "\n".join([
            f"@{route_choice}" 
            for route_choice in route_choice_list
        ])
    )

    return (
        "The command is ambiguous. Please select from these possible options:\n"
        f"{command_list}\n\n"
        "or type 'None of these' to see all commands\n"
        "or type 'abort' to cancel"
    )
def formulate_misclassified_command_error_message(route_choice_list: list[str]) -> str:
    command_list = (
        "\n".join([
            f"@{route_choice}" 
            for route_choice in route_choice_list
        ])
    )

    return (
        "The command was misclassified. Please select the correct command from the list below:\n"
        f"{command_list}\n\nor type 'abort' to cancel"
    )