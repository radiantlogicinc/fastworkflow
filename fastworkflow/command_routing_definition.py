import importlib
import json
import os
from enum import Enum
from typing import Any, Optional, Type

from pydantic import BaseModel, field_validator, model_validator, PrivateAttr
# from speedict import Rdict

# import fastworkflow
from fastworkflow import CommandSource
from fastworkflow.command_directory import CommandDirectory, CommandMetadata, CommandKeyMap, UtteranceMetadata

from fastworkflow.utils import python_utils

class ModuleType(Enum):
    INPUT_FOR_PARAM_EXTRACTION_CLASS = 0
    COMMAND_PARAMETERS_CLASS = 1
    RESPONSE_GENERATION_INFERENCE = 2


class RoutingFailureFallbackDefinition(BaseModel):
    fallback_workflow: Optional[str] = None
    fallback_command: Optional[str] = None

    @model_validator(mode="after")
    def validate_routing_failure_fallback_definition(
        self,
    ) -> "RoutingFailureFallbackDefinition":
        if (self.fallback_workflow and self.fallback_command) or (
            not self.fallback_workflow and not self.fallback_command
        ):
            raise ValueError(
                "Exactly one of fallback workflow or fallback command must be populated."
            )

        return self


class CommandRoutingDefinition(BaseModel):
    """Command routing definition"""
    workflow_folderpath: str
    command_directory: CommandDirectory
    map_workitem_paths_2_commandkey_map: dict[
        str, CommandKeyMap
    ] = None
    # if '/fastworkflow' in workflow_folderpath:
    #     fastworkflow_folder="./fastworkflow"
    #     workflow_folderpath = os.path.join(
    #         fastworkflow_folder, "_workflows", "command_name_prediction"
    #     )
    @field_validator("map_workitem_paths_2_commandkey_map", mode="before")
    def validate_command_routing_map(
        cls, map_workitem_paths_2_commandkey_map
    ):
        if not map_workitem_paths_2_commandkey_map:
            raise ValueError(
                "map_workitem_paths_2_commandkey_map cannot be empty"
            )

        for (
            workitem_path,
            command_key_map,
        ) in map_workitem_paths_2_commandkey_map.items():
            if not workitem_path:
                raise ValueError("workitem_path cannot be an empty string")

            if isinstance(command_key_map, dict):
                command_key_map[workitem_path] = (
                    CommandKeyMap(**command_key_map)
                )
            elif not isinstance(
                map_workitem_paths_2_commandkey_map[workitem_path],
                CommandKeyMap,
            ):
                raise ValueError(
                    f"Invalid value for command key map for '{workitem_path}'"
                )

        return map_workitem_paths_2_commandkey_map

    def get_command_names(self, workitem_path: str) -> list[str]:
        if workitem_path in self.map_workitem_paths_2_commandkey_map:
            # To discard "abort", "*" and "none_of_these" from the list of valid commands
            
            # m=list()
            # i=0
            # for key in self.map_workitem_paths_2_commandkey_map[workitem_path]:
            #     i+=1
            #     if i==0:
            #         continue
            #     for k in key[1]:
            #         y=k
            #         k=key[1][k]
            #         x=self.command_directory.get_command_metadata(k)
            #         x=x.workflow_folderpath
            #         if x==None:
                        # m.append(y)
            # return m

            return list(
                self.map_workitem_paths_2_commandkey_map[
                    workitem_path
                ].map_command_2_command_key.keys()
            )
        else:
            raise ValueError(
                f"Command routing definition not found for workitem type '{workitem_path}'"
            )

    _command_class_cache: dict[str, Type[Any]] = PrivateAttr(default_factory=dict)
    def get_command_class(self, workitem_path: str, command_name: str, module_type: ModuleType):
        cache_key = f"{workitem_path}:{command_name}:{module_type}"
        if cache_key in self._command_class_cache:
            return self._command_class_cache[cache_key]
        result = self._compute_command_class(workitem_path, command_name, module_type)
        self._command_class_cache[cache_key] = result
        return result

    def get_command_class_object(
        self, workitem_path: str, command_name: str, module_type: ModuleType
    ) -> Optional[Type[object]]:
        command_class = self.get_command_class(workitem_path, command_name, module_type)
        return command_class() if command_class else None

    def get_utterance_metadata(
        self, workitem_path: str, command_name: str
    ) -> UtteranceMetadata:
        if workitem_path not in self.map_workitem_paths_2_commandkey_map:
            raise ValueError(
                f"Command routing definition not found for workitem type '{workitem_path}'"
            )
        command_key_map = self.map_workitem_paths_2_commandkey_map[workitem_path]
        if command_name not in command_key_map.map_command_2_command_key:
            raise ValueError(
                f"Command name not found for command '{command_name}'"
            )
        command_key = command_key_map.map_command_2_command_key[command_name]
        return self.command_directory.get_utterance_metadata(command_key)

    def save(self):
        """Save the command routing definition to JSON, excluding the command_directory"""
        save_path = f"{CommandDirectory.get_commandinfo_folderpath(self.workflow_folderpath)}/command_routing_definition.json"
        # Create a dict without command_directory
        save_dict = self.model_dump(exclude={'command_directory'})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4)

    @classmethod
    def load(cls, workflow_folderpath):
        """Load the command routing definition from JSON"""
        load_path = f"{CommandDirectory.get_commandinfo_folderpath(workflow_folderpath)}/command_routing_definition.json"
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["workflow_folderpath"] = workflow_folderpath

        command_directory = CommandDirectory.load(workflow_folderpath) or CommandDirectory(
            workflow_folderpath=workflow_folderpath)
        data["command_directory"] = command_directory

        return cls.model_validate(data)

    def _compute_command_class(
        self, workitem_path: str, command_name: str, module_type: ModuleType
    ) -> Optional[Type[Any]]:  # sourcery skip: extract-method, last-if-guard
        if workitem_path not in self.map_workitem_paths_2_commandkey_map:
            raise ValueError(
                f"Command routing definition not found for workitem type '{workitem_path}'"
            )
        command_key_map = (
            self.map_workitem_paths_2_commandkey_map[workitem_path]
        )
        if command_name not in command_key_map.map_command_2_command_key:
            raise ValueError(
                f"Command '{command_name}' not found for workitem type '{workitem_path}'"
            )

        command_metadata = self.command_directory.get_command_metadata(
            command_key_map.map_command_2_command_key[command_name]
        )

        if module_type == ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS:
            module_file_path = (
                command_metadata.parameter_extraction_signature_module_path
            )
            module_class_name = (
                command_metadata.input_for_param_extraction_class
            )
        elif module_type == ModuleType.COMMAND_PARAMETERS_CLASS:
            module_file_path = (
                command_metadata.parameter_extraction_signature_module_path
            )
            module_class_name = command_metadata.command_parameters_class
        elif module_type == ModuleType.RESPONSE_GENERATION_INFERENCE:
            module_file_path = command_metadata.response_generation_module_path
            module_class_name = command_metadata.response_generation_class_name
        else:
            raise ValueError(f"Invalid module type '{module_type}'")

        module = python_utils.get_module(module_file_path, self.workflow_folderpath)
        return getattr(module, module_class_name) if module else None

    @classmethod
    def _populate_command_routing_definition(
        cls,
        parent_workitem_path: str,
        workflow_folderpath: str,
        command_directory: CommandDirectory,
        map_workitem_paths_2_commandkeymap: dict[
            str, CommandKeyMap
        ],
    ):
        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workitem_path = f"{parent_workitem_path}/{os.path.basename(workflow_folderpath.rstrip('/'))}"
        # if workitem_path.startswith("_"):
        #     raise ValueError(f"{workitem_path} starts with an '_'. Names starting with an _ are reserved")
        
        #read the built-in commands from the _workflows folder in fastworkflow. 
        fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
        ############################################################
        # There needs to be a better way of getting the fastworkflow folder from __file__
        # if '/fastworkflow' in fastworkflow_folder:
        #             fastworkflow_folder="./fastworkflow"
        #############################################################

        #############################################################
        # commandname_prediction_workflow_folderpath = os.path.join(
        #     fastworkflow_folder, "_workflows", "command_name_prediction"
        # )
        # cls._populate_command_key_map(
        #     parent_workitem_path,
        #     commandname_prediction_workflow_folderpath,
        #     command_directory,
        #     map_workitem_paths_2_commandkeymap,
        #     workitem_path,
        #     CommandSource.BASE_COMMANDS,
        # )

        # read the _base_commands folder if it exists
        cls._populate_command_key_map(
            parent_workitem_path,
            workflow_folderpath,
            command_directory,
            map_workitem_paths_2_commandkeymap,
            workitem_path,
            CommandSource.BASE_COMMANDS,
        )

        # read the _commands folder if it exists
        cls._populate_command_key_map(
            parent_workitem_path,
            workflow_folderpath,
            command_directory,
            map_workitem_paths_2_commandkeymap,
            workitem_path,
            CommandSource.COMMANDS,
        )

        # Recursively process subfolders
        for command in os.listdir(workflow_folderpath):
            subfolder_path = os.path.join(workflow_folderpath, command)
            if os.path.isdir(subfolder_path) and not command.startswith("_"):
                cls._populate_command_routing_definition(
                    workitem_path,
                    subfolder_path,
                    command_directory,
                    map_workitem_paths_2_commandkeymap,
                )

    @classmethod
    def _populate_command_key_map(
        cls,
        parent_workitem_path: str,
        workflow_folderpath: str,
        command_directory: CommandDirectory,
        map_workitem_paths_2_commandkeymap: dict[
            str, CommandKeyMap
        ],
        workitem_path: str,
        command_source: CommandSource,
    ):
        map_command_2_command_key: dict[str, str] = {}

        # copy the base commands
        if parent_workitem_path and parent_workitem_path in map_workitem_paths_2_commandkeymap:
            if command_key_map := map_workitem_paths_2_commandkeymap[
                parent_workitem_path
            ]:
                for (
                    command,
                    command_key,
                ) in command_key_map.map_command_2_command_key.items():
                    command_metadata = command_directory.get_command_metadata(command_key)
                    if command_metadata.command_source == CommandSource.BASE_COMMANDS:
                        map_command_2_command_key[command] = command_key

        commands_folder = os.path.join(workflow_folderpath, command_source.value)
        if os.path.exists(commands_folder):
            for command in os.listdir(commands_folder):
                subfolder_path = os.path.join(commands_folder, command)

                if not os.path.isdir(subfolder_path):
                    continue
                if command.startswith("_"):
                    continue

                # parameter extraction
                parameter_extraction_signature_module_path: Optional[str] = None
                input_for_param_extraction_class: Optional[str] = None
                command_parameters_class: Optional[str] = None
                # response generation
                response_generation_module_path: str
                response_generation_class_name: str

                parameter_extraction_folderpath = os.path.join(
                    subfolder_path, "parameter_extraction"
                )
                if os.path.exists(parameter_extraction_folderpath):
                    signature_filepath = os.path.join(
                        parameter_extraction_folderpath, "signatures.py"
                    )
                    if os.path.exists(signature_filepath):
                        parameter_extraction_signature_module_path = signature_filepath
                        input_for_param_extraction_class = "InputForParamExtraction"
                        command_parameters_class = "CommandParameters"

                response_generation_folderpath = os.path.join(
                    subfolder_path, "response_generation"
                )
                if not os.path.exists(response_generation_folderpath):
                    raise ValueError(
                        f"Response generation folder not found at '{response_generation_folderpath}'"
                    )
                inference_filepath = os.path.join(
                    response_generation_folderpath, "inference.py"
                )
                if not os.path.exists(inference_filepath):
                    raise ValueError(
                        f"Response generation inference file not found at '{inference_filepath}'"
                    )
                response_generation_module_path = inference_filepath
                response_generation_class_name = "ResponseGenerator"
                
                # if '/fastworkflow' in workflow_folderpath:
                #     workflow_folderpath="./fastworkflow/_workflows/command_name_prediction"
                # else:
                #     workflow_folderpath=None

                command_metadata = CommandMetadata(
                    command_source=command_source,
                    workflow_folderpath= workflow_folderpath if '/fastworkflow' in workflow_folderpath else None,
                    parameter_extraction_signature_module_path=parameter_extraction_signature_module_path,
                    input_for_param_extraction_class=input_for_param_extraction_class,
                    command_parameters_class=command_parameters_class,
                    # response generation
                    response_generation_module_path=response_generation_module_path,
                    response_generation_class_name=response_generation_class_name,
                )

                command_key = f"{command_source.value}/{command}"
                command_directory.register_command_metadata(
                    command_key, command_metadata)
                map_command_2_command_key[command] = command_key 

        if workitem_path in map_workitem_paths_2_commandkeymap:
            map_workitem_paths_2_commandkeymap[workitem_path].map_command_2_command_key.update(map_command_2_command_key)
        elif map_command_2_command_key:
            map_workitem_paths_2_commandkeymap[workitem_path] = (
                CommandKeyMap(
                    map_command_2_command_key=map_command_2_command_key
                )
            )

class CommandRoutingRegistry:
    """ This class is used to register command routing definitions for different workflows """
    @classmethod
    def get_definition(cls, workflow_folderpath: str) -> CommandRoutingDefinition:
        if workflow_folderpath in cls._command_routing_definitions:
            return cls._command_routing_definitions[workflow_folderpath]
        
        command_directory = CommandDirectory.load(workflow_folderpath)
        if not command_directory:
            return CommandRoutingRegistry.create_definition(workflow_folderpath)
        
        command_routing_definition = CommandRoutingDefinition.load(workflow_folderpath)
        cls._command_routing_definitions[workflow_folderpath] = command_routing_definition
        return command_routing_definition

    @classmethod
    def create_definition(cls, workflow_folderpath: str) -> Optional[CommandRoutingDefinition]:
        command_directory = CommandDirectory.load(workflow_folderpath) or CommandDirectory(
            workflow_folderpath=workflow_folderpath)

        map_workitem_paths_2_commandkeymap: dict[
            str, CommandKeyMap
        ] = {}

        # load the command directory from fastworkflow/_workflows (built-in commands)
        # fastworkflow_folderpath = os.path.abspath(os.path.dirname(__file__))
        # if "fastworkflow" not in workflow_folderpath:
            # commandname_prediction_folderpath = f"{fastworkflow_folderpath}/_workflows/command_name_prediction"
            # commandname_prediction_cmddir = CommandDirectory.load(commandname_prediction_folderpath)
            # command_directory.map_commandkey_2_metadata.update(**commandname_prediction_cmddir.map_commandkey_2_metadata)
            # commandname_prediction_crdef = CommandRoutingDefinition.load(commandname_prediction_folderpath)
            # map_workitem_paths_2_commandkeymap.update(**commandname_prediction_crdef.map_workitem_paths_2_commandkey_map)
            
            # parameter_extraction_folderpath = f"{fastworkflow_folderpath}/_workflows/parameter_extraction"
            # parameter_extraction_cmddir = CommandDirectory.load(parameter_extraction_folderpath)
            # command_directory.map_commandkey_2_metadata.update(**parameter_extraction_cmddir.map_commandkey_2_metadata)
            # parameter_extraction_crdef = CommandRoutingDefinition.load(parameter_extraction_folderpath)
            # map_workitem_paths_2_commandkeymap.update(**parameter_extraction_crdef.map_workitem_paths_2_commandkey_map)

        CommandRoutingDefinition._populate_command_routing_definition(
            "", workflow_folderpath, command_directory, map_workitem_paths_2_commandkeymap
        )

        if not map_workitem_paths_2_commandkeymap:
            return None

        command_routing_definition = CommandRoutingDefinition(
            workflow_folderpath=workflow_folderpath,
            command_directory=command_directory,
            map_workitem_paths_2_commandkey_map=map_workitem_paths_2_commandkeymap,
        )

        command_directory.save()
        command_routing_definition.save()

        cls._command_routing_definitions[workflow_folderpath] = command_routing_definition
        return command_routing_definition

    _command_routing_definitions: dict[str, CommandRoutingDefinition] = {}
