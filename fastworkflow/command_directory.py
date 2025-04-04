
import importlib
import os
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from fastworkflow import CommandSource
from fastworkflow.utils import python_utils


class CommandMetadata(BaseModel):
    command_source: CommandSource
    """Command metadata"""
    # parameter extraction
    workflow_folderpath: Optional[str] = None
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


class UtteranceMetadata(BaseModel):
    workflow_folderpath: Optional[str] = None
    plain_utterances: list[str]
    template_utterances: list[str]
    generated_utterances_module_filepath: str
    generated_utterances_func_name: str

    def get_generated_utterances_func(self, workflow_folderpath: str) -> list[str]:      
        module = python_utils.get_module(
            self.generated_utterances_module_filepath, 
            self.workflow_folderpath if self.workflow_folderpath else workflow_folderpath
        )

        # Get the function from the module and execute it
        return getattr(module, self.generated_utterances_func_name)

    @field_validator("plain_utterances", mode="before")
    def parse_plain_utterances(cls, plain_utterances: list[str]):
        for str in plain_utterances:
            if not str:
                raise ValueError("Plain utterance string cannot be empty")
        return plain_utterances

    @field_validator("template_utterances", mode="before")
    def parse_template_utterances(cls, template_utterances: list[str]):
        for str in template_utterances:
            if not str:
                raise ValueError("Plain utterance string cannot be empty")
        return template_utterances

    @field_validator("generated_utterances_module_filepath", mode="before")
    def parse_generated_utterances_module_path(
        cls, generated_utterances_module_filepath: str
    ):
        if not generated_utterances_module_filepath:
            raise ValueError("Generated utterances module path cannot be empty")
        return generated_utterances_module_filepath

    @field_validator("generated_utterances_func_name", mode="before")
    def parse_generatd_utterances_func_name(cls, generated_utterances_func_name: str):
        if not generated_utterances_func_name:
            raise ValueError("Generated utterances function name cannot be empty")
        return generated_utterances_func_name


class CommandDirectory(BaseModel):
    """
    Centralized directory for all command implementations.
    Each command is uniquely identified by a string - <workflow_path/cmd_name>.
    """
    workflow_folderpath: str

    map_commandkey_2_metadata: dict[str, CommandMetadata] = {}
    map_command_key_2_utterance_metadata: dict[str, UtteranceMetadata] = {}

    def register_command_metadata(self, command_key: str, metadata: CommandMetadata):
        if not (command_key and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_commandkey_2_metadata[command_key] = metadata

    def get_command_keys(self) -> list[str]:
        """Retrieve all command keys registered in the command directory."""
        return list(self.map_commandkey_2_metadata.keys())

    def get_command_metadata(self, command_key: str) -> CommandMetadata:
        if command_key not in self.map_commandkey_2_metadata:
            raise KeyError(f"Command key '{command_key}' not found.")
        return self.map_commandkey_2_metadata[command_key]

    def register_utterance_metadata(self, command_key: str, metadata: UtteranceMetadata):
        if not (command_key and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_command_key_2_utterance_metadata[command_key] = metadata

    def get_utterance_keys(self) -> list[str]:
        """Retrieve all utterance keys registered in the command directory."""
        return list(self.map_command_key_2_utterance_metadata.keys())

    def get_utterance_metadata(self, command_key: str) -> UtteranceMetadata:
        if command_key not in self.map_command_key_2_utterance_metadata:
            raise KeyError(f"Command key '{command_key}' not found.")
        return self.map_command_key_2_utterance_metadata[command_key]

    @classmethod
    def get_command_name(cls, command_key: str) -> str:
        return command_key.split("/")[-1]

    @classmethod
    def get_commandinfo_folderpath(cls, workflow_folderpath: str):
        command_info_folderpath = f"{workflow_folderpath}/___command_info"
        os.makedirs(command_info_folderpath, exist_ok=True)
        return command_info_folderpath

    def save(self):
        commandroutinginfo_folderpath = CommandDirectory.get_commandinfo_folderpath(self.workflow_folderpath)
        with open(f"{commandroutinginfo_folderpath}/command_directory.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, workflow_folderpath: str):
        commandroutinginfo_folderpath = CommandDirectory.get_commandinfo_folderpath(workflow_folderpath)
        try:
            with open(f"{commandroutinginfo_folderpath}/command_directory.json", "r", encoding="utf-8") as f:
                command_directory = CommandDirectory(
                    workflow_folderpath=workflow_folderpath
                )
                return command_directory.model_validate_json(f.read())
        except FileNotFoundError:
            return None

    @field_validator("map_commandkey_2_metadata", mode="before")
    def validate_map_commandkey_2_metadata(cls, map_commandkey_2_metadata):
        if not map_commandkey_2_metadata:
            raise ValueError("map_commandkey_2_metadata cannot be empty.")
        if not isinstance(map_commandkey_2_metadata, dict):
            raise ValueError("map_commandkey_2_metadata must be a dictionary.")
        for command_key, metadata in map_commandkey_2_metadata.items():
            if not command_key:
                raise ValueError("Command key cannot be an empty string.")
            if isinstance(metadata, dict):
                metadata = CommandMetadata(**metadata)
                map_commandkey_2_metadata[command_key] = metadata
            if not isinstance(metadata, CommandMetadata):
                raise ValueError(f"Invalid metadata for command key '{command_key}'")
        return map_commandkey_2_metadata

    @field_validator("map_command_key_2_utterance_metadata", mode="before")
    def parse_map_command_2_utterances(
        cls, map_command_key_2_utterance_metadata: dict[str, UtteranceMetadata]
    ):
        for key, value in map_command_key_2_utterance_metadata.items():
            if isinstance(value, dict):
                map_command_key_2_utterance_metadata[key] = UtteranceMetadata(**value)
            elif not isinstance(map_command_key_2_utterance_metadata[key], UtteranceMetadata):
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return map_command_key_2_utterance_metadata

class CommandKeyMap(BaseModel):
    """Map workflow commands to command keys in the command directory"""

    map_command_2_command_key: dict[str, str] = None

    @field_validator("map_command_2_command_key", mode="before")
    def validate_command_metadata_map(cls, map_command_2_command_key):
        if not map_command_2_command_key:
            raise ValueError("map_command_2_command_key cannot be empty.")
        for command_name, command_key in map_command_2_command_key.items():
            if not command_name:
                raise ValueError("Command name cannot be an empty string.")

            if not isinstance(command_key, str):
                raise ValueError(
                    f"Invalid value for commandkey for '{command_name}'"
                )

        return map_command_2_command_key

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
