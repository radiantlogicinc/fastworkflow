import importlib
import os
from enum import Enum
from typing import Any, Optional, Type

from pydantic import BaseModel, ConfigDict, field_validator, model_validator, PrivateAttr
from speedict import Rdict

import fastworkflow
from fastworkflow import CommandSource


class ModuleType(Enum):
    INPUT_FOR_PARAM_EXTRACTION_CLASS = 0
    COMMAND_PARAMETERS_CLASS = 1
    RESPONSE_GENERATION_INFERENCE = 2


class CommandMetadata(BaseModel):
    command_source: CommandSource
    """Command metadata"""
    # parameter extraction
    parameter_extraction_signature_module_path: Optional[str] = None
    input_for_param_extraction_class: Optional[str] = None
    command_parameters_class: Optional[str] = None
    # response generation
    response_generation_module_path: str
    response_generation_class_name: str

    @model_validator(mode="after")
    def validate_parameter_extraction_fields(self) -> "CommandMetadata":
        rg_fields = [
            self.response_generation_module_path,
            self.response_generation_class_name,
        ]

        if not (all(rg_fields)):
            raise ValueError("Response generation modules must be populated.")

        return self


class CommandExecutionMetadata(BaseModel):
    """Command execution metadata"""

    map_commands_2_metadata: dict[str, CommandMetadata] = None

    @field_validator("map_commands_2_metadata", mode="before")
    def validate_command_metadata_map(cls, map_commands_2_metadata):
        if not map_commands_2_metadata:
            raise ValueError("Command metadata dictionary cannot be empty.")
        for command_name, metadata in map_commands_2_metadata.items():
            if not command_name:
                raise ValueError("Command name cannot be an empty string.")

            if isinstance(metadata, dict):
                metadata = CommandMetadata(**metadata)
                map_commands_2_metadata[command_name] = metadata
            elif not isinstance(metadata, CommandMetadata):
                raise ValueError(
                    f"Invalid value for command metadata for '{command_name}'"
                )

        return map_commands_2_metadata

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


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
    map_workitem_types_2_commandexecutionmetadata: dict[
        str, CommandExecutionMetadata
    ] = None

    @field_validator("map_workitem_types_2_commandexecutionmetadata", mode="before")
    def validate_command_routing_map(
        cls, map_workitem_types_2_commandexecutionmetadata
    ):
        if not map_workitem_types_2_commandexecutionmetadata:
            raise ValueError(
                "map_workitem_types_2_commandexecutionmetadata cannot be empty"
            )

        for (
            workitem_type,
            command_execution_metadata,
        ) in map_workitem_types_2_commandexecutionmetadata.items():
            if not workitem_type:
                raise ValueError("workitem_type cannot be an empty string")

            if isinstance(command_execution_metadata, dict):
                map_workitem_types_2_commandexecutionmetadata[workitem_type] = (
                    CommandExecutionMetadata(**command_execution_metadata)
                )
            elif not isinstance(
                map_workitem_types_2_commandexecutionmetadata[workitem_type],
                CommandExecutionMetadata,
            ):
                raise ValueError(
                    f"Invalid value for command execution metadata for '{workitem_type}'"
                )

        return map_workitem_types_2_commandexecutionmetadata

    def write(self, filename: str):
        with open(filename, "w") as f:
            json_data = self.model_dump_json(indent=4)
            f.write(json_data)

    def get_command_names(self, workitem_type: str) -> list[str]:
        if workitem_type in self.map_workitem_types_2_commandexecutionmetadata:
            return list(
                self.map_workitem_types_2_commandexecutionmetadata[
                    workitem_type
                ].map_commands_2_metadata.keys()
            )
        else:
            raise ValueError(
                f"Command routing definition not found for workitem type '{workitem_type}'"
            )

    def _compute_command_class(
        self, workitem_type: str, command_name: str, module_type: ModuleType
    ) -> Optional[Type[Any]]:
        if workitem_type in self.map_workitem_types_2_commandexecutionmetadata:
            command_execution_metadata = (
                self.map_workitem_types_2_commandexecutionmetadata[workitem_type]
            )
            if command_name in command_execution_metadata.map_commands_2_metadata:
                command_metadata = command_execution_metadata.map_commands_2_metadata[
                    command_name
                ]

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

                if not module_file_path:
                    return None

                module_name = os.path.splitext(os.path.basename(module_file_path))[0]

                workflow_folder_syspath = (
                    f"{self.workflow_folderpath}/"
                    if not self.workflow_folderpath.endswith("/")
                    else ...
                )
                relative_module_filepath = os.path.relpath(
                    module_file_path, os.path.dirname(workflow_folder_syspath)
                )

                package_name = (
                    os.path.dirname(relative_module_filepath)
                    .replace("/", ".")
                    .replace("\\", ".")
                )
                full_module_name = f".{package_name}.{module_name}"

                parent_folder = os.path.dirname(workflow_folder_syspath)
                fastworkflow_relpath = parent_folder[parent_folder.rfind("fastworkflow"):] if "fastworkflow" in parent_folder else parent_folder
                workflow_package_name = (
                    fastworkflow_relpath.replace("/", ".")
                    .replace("\\", ".")
                ).replace("..", "")

                spec = importlib.util.find_spec(full_module_name, package=workflow_package_name)
                if spec is None:
                    raise ImportError(f"Module {full_module_name} not found")
                module = importlib.import_module(full_module_name, package=workflow_package_name)

                # Get the class from the module
                return getattr(module, module_class_name)
            else:
                raise ValueError(
                    f"Command '{command_name}' not found for workitem type '{workitem_type}'"
                )
        else:
            raise ValueError(
                f"Command routing definition not found for workitem type '{workitem_type}'"
            )

    _command_class_cache: dict[str, Type[Any]] = PrivateAttr(default_factory=dict)
    def get_command_class(self, workitem_type: str, command_name: str, module_type: ModuleType):
        cache_key = f"{workitem_type}:{command_name}:{module_type}"
        if cache_key in self._command_class_cache:
            return self._command_class_cache[cache_key]
        result = self._compute_command_class(workitem_type, command_name, module_type)
        self._command_class_cache[cache_key] = result
        return result

    def get_command_class_object(
        self, workitem_type: str, command_name: str, module_type: ModuleType
    ) -> Optional[Type[object]]:
        command_class = self.get_command_class(workitem_type, command_name, module_type)
        if command_class:
            return command_class()
        else:
            return None

    @classmethod
    def _populate_command_routing_definition(
        cls,
        parent_workitem_type: str,
        workflow_folderpath: str,
        map_workitem_types_2_commandexecutionmetadata: dict[
            str, CommandExecutionMetadata
        ],
    ):
        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workitem_type = os.path.basename(workflow_folderpath.rstrip("/"))
        # if workitem_type.startswith("_"):
        #     raise ValueError(f"{workitem_type} starts with an '_'. Names starting with an _ are reserved")

        # read the _base_commands folder if it exists
        cls._populate_command_execution_metadata(
            parent_workitem_type,
            workflow_folderpath,
            map_workitem_types_2_commandexecutionmetadata,
            workitem_type,
            CommandSource.BASE_COMMANDS,
        )

        # read the _commands folder if it exists
        cls._populate_command_execution_metadata(
            parent_workitem_type,
            workflow_folderpath,
            map_workitem_types_2_commandexecutionmetadata,
            workitem_type,
            CommandSource.COMMANDS,
        )

        # Recursively process subfolders
        for command in os.listdir(workflow_folderpath):
            subfolder_path = os.path.join(workflow_folderpath, command)
            if os.path.isdir(subfolder_path) and not command.startswith("_"):
                cls._populate_command_routing_definition(
                    workitem_type,
                    subfolder_path,
                    map_workitem_types_2_commandexecutionmetadata,
                )

    @classmethod
    def _populate_command_execution_metadata(
        cls,
        parent_workitem_type: str,
        workflow_folderpath: str,
        map_workitem_types_2_commandexecutionmetadata: dict[
            str, CommandExecutionMetadata
        ],
        workitem_type: str,
        command_source: CommandSource,
    ):
        map_commands_2_metadata: dict[str, CommandExecutionMetadata] = {}

        # copy the base commands
        if parent_workitem_type:
            command_execution_metadata = map_workitem_types_2_commandexecutionmetadata[
                parent_workitem_type
            ]
            for (
                command,
                command_metadata,
            ) in command_execution_metadata.map_commands_2_metadata.items():
                if command_metadata.command_source == CommandSource.BASE_COMMANDS:
                    map_commands_2_metadata[command] = command_metadata

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

                map_commands_2_metadata[command] = CommandMetadata(
                    command_source=command_source,
                    parameter_extraction_signature_module_path=parameter_extraction_signature_module_path,
                    input_for_param_extraction_class=input_for_param_extraction_class,
                    command_parameters_class=command_parameters_class,
                    # response generation
                    response_generation_module_path=response_generation_module_path,
                    response_generation_class_name=response_generation_class_name,
                )

        if workitem_type in map_workitem_types_2_commandexecutionmetadata:
            command_execution_metadata = map_workitem_types_2_commandexecutionmetadata[
                workitem_type
            ]
            for command, command_metadata in map_commands_2_metadata.items():
                command_execution_metadata.map_commands_2_metadata[command] = (
                    command_metadata
                )
        else:
            if map_commands_2_metadata:
                map_workitem_types_2_commandexecutionmetadata[workitem_type] = (
                    CommandExecutionMetadata(
                        map_commands_2_metadata=map_commands_2_metadata
                    )
                )

class CommandRoutingRegistry:
    """ This class is used to register command routing definitions for different workflows """
    @classmethod
    def get_definition(cls, workflow_folderpath: str) -> CommandRoutingDefinition:
        if workflow_folderpath in cls._command_routing_definitions:
            return cls._command_routing_definitions[workflow_folderpath]
        
        commandroutingdefinitiondb_folderpath_dir = cls._get_commandroutingdefinition_db_folderpath()
        commandroutingdefinitiondb = Rdict(commandroutingdefinitiondb_folderpath_dir)
        command_routing_definition = commandroutingdefinitiondb.get(workflow_folderpath, None)
        commandroutingdefinitiondb.close()

        if command_routing_definition:
            cls._command_routing_definitions[workflow_folderpath] = command_routing_definition
            return command_routing_definition
        
        return CommandRoutingRegistry._create_definition(workflow_folderpath)

    @classmethod
    def _create_definition(cls, workflow_folderpath: str) -> CommandRoutingDefinition:
        map_workitem_types_2_commandexecutionmetadata: dict[
            str, CommandExecutionMetadata
        ] = {}

        CommandRoutingDefinition._populate_command_routing_definition(
            "", workflow_folderpath, map_workitem_types_2_commandexecutionmetadata
        )

        command_routing_definition = CommandRoutingDefinition(
            workflow_folderpath=workflow_folderpath,
            map_workitem_types_2_commandexecutionmetadata=map_workitem_types_2_commandexecutionmetadata,
        )

        commandroutingdefinitiondb_folderpath_dir = cls._get_commandroutingdefinition_db_folderpath()
        commandroutingdefinitiondb = Rdict(commandroutingdefinitiondb_folderpath_dir)
        commandroutingdefinitiondb[workflow_folderpath] = command_routing_definition
        commandroutingdefinitiondb.close()

        cls._command_routing_definitions[workflow_folderpath] = command_routing_definition
        return command_routing_definition
    
    @classmethod
    def _get_commandroutingdefinition_db_folderpath(cls) -> str:
        """get the command routing definition db folder path"""
        SPEEDDICT_FOLDERNAME = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        commandroutingdefinition_db_folderpath = os.path.join(
            SPEEDDICT_FOLDERNAME,
            "commandroutingdefinitions"
        )
        os.makedirs(commandroutingdefinition_db_folderpath, exist_ok=True)
        return commandroutingdefinition_db_folderpath

    _command_routing_definitions: dict[str, CommandRoutingDefinition] = {}
