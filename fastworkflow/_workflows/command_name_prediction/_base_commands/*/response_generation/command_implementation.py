from typing import Optional

from pydantic import BaseModel
import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.model_pipeline_training import predict_single_sentence
import json
import os
from speedict import Rdict
from fastworkflow.cache_matching import get_flag,change_flag,store_utterance_cache,cache_match

class CommandParameters(BaseModel):
    command_name: Optional[str] = None

class OutputOfProcessCommand(BaseModel):
    command_name: Optional[str] = None
    command: Optional[str] = None
    error_msg: Optional[str] = None

def get_cache_path(session_id,convo_path):
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


def get_route_layer_filepath(workflow_folderpath,model_name) -> str:
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
    sws = session.workflow_snapshot.context["subject_workflow_snapshot"]
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_session_id=sws.session_id
    

    convo_path=os.path.join(sws_workflow_folderpath,"___convo_info")
   

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

    cache_path = get_cache_path(sws_session_id,convo_path)
    

        
    if not command_name and "@" in command:
        tentative_command_name = command.split("@")[1].split()[0]
        normalized_command_name = tentative_command_name.lower()
        for name in valid_command_names:
            if normalized_command_name == name.lower():
                command_name = name
                command = command.replace(f"@{tentative_command_name}", "").strip().replace("  ", " ")
                break

    if not command_name:
        
        tiny_path=get_route_layer_filepath(sws_workflow_folderpath,"tinymodel.pth")
        large_path=get_route_layer_filepath(sws_workflow_folderpath,"largemodel.pth")
        
        threshold_path=get_route_layer_filepath(sws_workflow_folderpath,"threshold.json")
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            confidence_threshold = data['confidence_threshold']

        modelpipeline=fastworkflow.modelpipelineregistry(
            tiny_model_path=tiny_path,  # Replace with your fine-tuned TinyBERT model path
            distil_model_path=large_path,
            confidence_threshold=confidence_threshold       
        )


        path=get_cache_path_cache(convo_path)
        
        cache_result=cache_match(path,command,modelpipeline,0.85)

        if cache_result:
            command_name=cache_result
            path=get_cache_path_cache(convo_path)
            flag=get_flag(path)
            if flag==1:
                count=get_count(cache_path)
                utterance=read_utterance(cache_path,count-1)
                store_utterance_cache(path,utterance,command_name)  
                change_flag(path,0)         
            
        
        else:
            results=predict_single_sentence(modelpipeline,command,sws_workflow_folderpath)
            command_name=results['label']
            if results['confidence']< 0.90:
                error_msg = formulate_ambiguous_command_error_message(results["topk_labels"])
                count=store_utterance(cache_path,command,command_name)
                path=get_cache_path_cache(convo_path)
                change_flag(path,1)
                return OutputOfProcessCommand(error_msg=error_msg)
            
            if command_name=="None_of_these":
                error_msg = formulate_ambiguous_command_error_message(valid_command_names)
                path=get_cache_path_cache(convo_path)
                change_flag(path,1)
                return OutputOfProcessCommand(error_msg=error_msg)

            else:
                path=get_cache_path_cache(convo_path)
                flag=get_flag(path)
                if flag==1:
                    count=get_count(cache_path)
                    utterance=read_utterance(cache_path,count-1)
                    store_utterance_cache(path,utterance,command_name)  
                    change_flag(path,0) 
    
    else:
        path=get_cache_path_cache(convo_path)
        flag=get_flag(path)
        if flag==1:
            count=get_count(cache_path)
            utterance=read_utterance(cache_path,count-1)
            store_utterance_cache(path,utterance,command_name)  
            change_flag(path,0)         
            
    
    if command_name!="None_of_these":
        count=store_utterance(cache_path,command,command_name)
    

    command_parameters = CommandParameters(command_name=command_name)
    is_valid, error_msg = validate_command_name(valid_command_names, command_parameters)
    if not is_valid:
        return OutputOfProcessCommand(error_msg=error_msg)

    return OutputOfProcessCommand(
        command_name=command_parameters.command_name,
        command=command
    )   

def get_valid_command_names(sws: WorkflowSnapshot) -> set[str]:
    # valid_command_names = {'abort'}
    valid_command_names = {'None_of_these','abort'}
    sws_workflow_folderpath = sws.workflow.workflow_folderpath
    sws_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(sws_workflow_folderpath)
    valid_command_names |= set(sws_command_routing_definition.get_command_names(
        sws.active_workitem.path
    ))
    valid_command_names.remove("*")
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
        "The command is ambiguous. Prefix your command with an appropriate tag from the list below:\n"
        f"{command_list}"
    )

def formulate_ambiguous_command_error_message(route_choice_list: list[str]) -> str:
    command_list = (
        "\n".join([
            f"@{route_choice}" 
            for route_choice in route_choice_list
        ])
    )

    return (
        "The command is ambiguous. Prefix your command with an appropriate tag from the list below:\n"
        f"{command_list}"
    )
