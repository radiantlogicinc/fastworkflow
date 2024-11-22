import importlib
import json
import os

from pydantic import BaseModel, field_validator
from speedict import Rdict

import fastworkflow
from fastworkflow import CommandSource


class Utterances(BaseModel):
    command_source: CommandSource
    plain_utterances: list[str]
    template_utterances: list[str]

    generated_utterances_module_filepath: str
    generated_utterances_func_name: str

    def get_generated_utterances_func(self, workflow_folderpath: str) -> list[str]:
        # Import the module with parent package information
        module_name = os.path.splitext(
            os.path.basename(self.generated_utterances_module_filepath)
        )[0]
        relative_module_filepath = os.path.relpath(
            self.generated_utterances_module_filepath, workflow_folderpath
        )
        package_name = (
            os.path.dirname(relative_module_filepath)
            .replace("/", ".")
            .replace("\\", ".")
        )
        full_module_name = f".{package_name}.{module_name}"

        workflow_folder_syspath = (
            f"{workflow_folderpath}/"
            if not workflow_folderpath.endswith("/")
            else ...
        )
        workflow_package_name = (
            os.path.dirname(workflow_folder_syspath)
            .split("site-packages/", 1)[-1]
            .replace("/", ".")
            .replace("\\", ".")
        ).replace("..", "")

        spec = importlib.util.find_spec(full_module_name, package=workflow_package_name)
        if spec is None:
            raise ImportError(f"Module {full_module_name} not found")
        module = importlib.import_module(full_module_name, package=workflow_package_name)

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


class CommandUtterances(BaseModel):
    map_command_2_utterances: dict[str, Utterances]

    @field_validator("map_command_2_utterances", mode="before")
    def parse_map_command_2_utterances(
        cls, map_command_2_utterances: dict[str, Utterances]
    ):
        for key, value in map_command_2_utterances.items():
            if isinstance(value, dict):
                map_command_2_utterances[key] = Utterances(**value)
            elif not isinstance(map_command_2_utterances[key], Utterances):
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return map_command_2_utterances


class UtteranceDefinition(BaseModel):
    workitem_type_2_commandutterances: dict[str, CommandUtterances]

    @field_validator("workitem_type_2_commandutterances", mode="before")
    def parse_workitem_type_2_commandutterances(
        cls, workitem_type_2_commandutterances: dict[str, CommandUtterances]
    ):
        for key, value in workitem_type_2_commandutterances.items():
            if isinstance(value, dict):
                workitem_type_2_commandutterances[key] = CommandUtterances(**value)
            elif not isinstance(
                workitem_type_2_commandutterances[key], CommandUtterances
            ):
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return workitem_type_2_commandutterances

    def get_command_names(self, workitem_type: str) -> list[str]:
        if workitem_type in self.workitem_type_2_commandutterances:
            return list(
                self.workitem_type_2_commandutterances[
                    workitem_type
                ].map_command_2_utterances.keys()
            )
        else:
            raise ValueError(
                f"Utterance definition not found for workitem type '{workitem_type}'"
            )

    def get_command_utterances(
        self, workitem_type: str, command_name: str
    ) -> Utterances:
        if workitem_type in self.workitem_type_2_commandutterances:
            command_utterances = self.workitem_type_2_commandutterances[workitem_type]
            if command_name in command_utterances.map_command_2_utterances:
                return command_utterances.map_command_2_utterances[command_name]
            else:
                raise ValueError(
                    f"Utterance definition not found for command '{command_name}'"
                )
        else:
            raise ValueError(
                f"Utterance definition not found for workitem type '{workitem_type}'"
            )

    def get_sample_utterances(self, workitem_type: str) -> list[str]:
        command_names = self.get_command_names(workitem_type)
        sample_utterances = []
        for command_name in command_names:
            command_utterances = self.get_command_utterances(workitem_type, command_name)
            if command_utterances.template_utterances:
                sample_utterances.append(command_utterances.template_utterances[0])
            else:
                if command_utterances.plain_utterances:
                    sample_utterances.append(command_utterances.plain_utterances[0])
        return sample_utterances

    def write(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def _populate_utterance_definition(
        cls,
        parent_workitem_type: str,
        workflow_folderpath: str,
        workitem_type_2_commandutterances: dict[str, CommandUtterances],
    ):
        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workitem_type = os.path.basename(workflow_folderpath.rstrip("/"))
        # if workitem_type.startswith("_"):
        #     raise ValueError(f"{workitem_type} starts with an '_'. Names starting with an _ are reserved")

        # read the _base_commands folder if it exists
        cls._populate_command_utterances(
            parent_workitem_type,
            workflow_folderpath,
            workitem_type_2_commandutterances,
            workitem_type,
            CommandSource.BASE_COMMANDS,
        )

        # read the _commands folder if it exists
        cls._populate_command_utterances(
            parent_workitem_type,
            workflow_folderpath,
            workitem_type_2_commandutterances,
            workitem_type,
            CommandSource.COMMANDS,
        )

        # Recursively process subfolders
        for command in os.listdir(workflow_folderpath):
            subfolder_path = os.path.join(workflow_folderpath, command)
            if os.path.isdir(subfolder_path) and not command.startswith("_"):
                cls._populate_utterance_definition(
                    workitem_type, subfolder_path, workitem_type_2_commandutterances
                )

    @classmethod
    def _populate_command_utterances(
        cls,
        parent_workitem_type: str,
        workflow_folderpath: str,
        workitem_type_2_commandutterances: dict[str, CommandUtterances],
        workitem_type: str,
        command_source: CommandSource,
    ):
        map_command_2_utterances: dict[str, Utterances] = {}
        # copy the base commands
        if parent_workitem_type:
            command_utterances = workitem_type_2_commandutterances[parent_workitem_type]
            for (
                command,
                utterances,
            ) in command_utterances.map_command_2_utterances.items():
                if utterances.command_source == CommandSource.BASE_COMMANDS:
                    map_command_2_utterances[command] = utterances

        commands_folder = os.path.join(workflow_folderpath, command_source.value)
        if os.path.exists(commands_folder):
            for command in os.listdir(commands_folder):
                if command == "*":
                    continue

                subfolder_path = os.path.join(commands_folder, command)

                if not os.path.isdir(subfolder_path):
                    continue
                if command.startswith("_"):
                    continue

                plain_utterances: list[str] = []
                template_utterances: list[str] = []
                generated_utterances_module_filepath: str = ""
                generated_utterances_func_name: str = ""

                utterances_folderpath = os.path.join(subfolder_path, "utterances")
                if os.path.exists(utterances_folderpath):
                    plain_utterances_filepath = os.path.join(
                        utterances_folderpath, "plain_utterances.json"
                    )
                    if os.path.exists(plain_utterances_filepath):
                        with open(plain_utterances_filepath, "r") as f:
                            plain_utterances = json.load(f)

                    template_utterance_filepath = os.path.join(
                        utterances_folderpath, "template_utterances.json"
                    )
                    if os.path.exists(template_utterance_filepath):
                        with open(template_utterance_filepath, "r") as f:
                            template_utterances = json.load(f)

                    generated_utterances_module_filepath = os.path.join(
                        utterances_folderpath, "generate_utterances.py"
                    )
                    if not os.path.exists(generated_utterances_module_filepath):
                        raise ValueError(
                            f"Generated utterances module filepath '{generated_utterances_module_filepath}' not found"
                        )
                    else:
                        generated_utterances_func_name = "generate_utterances"

                map_command_2_utterances[command] = Utterances(
                    command_source=command_source,
                    plain_utterances=plain_utterances,
                    template_utterances=template_utterances,
                    generated_utterances_module_filepath=generated_utterances_module_filepath,
                    generated_utterances_func_name=generated_utterances_func_name,
                )

        if workitem_type in workitem_type_2_commandutterances:
            command_utterances = workitem_type_2_commandutterances[workitem_type]
            for command, utterances in map_command_2_utterances.items():
                if command in command_utterances:
                    command_utterances[command].plain_utterances.extend(
                        utterances.plain_utterances
                    )
                    command_utterances[command].template_utterances.extend(
                        utterances.template_utterances
                    )
                    command_utterances[command].generated_utterances_module_filepath = (
                        utterances.generated_utterances_module_filepath
                    )
                    command_utterances[command].generated_utterances_func_name = (
                        utterances.generated_utterances_func_name
                    )
                else:
                    command_utterances.map_command_2_utterances[command] = utterances
        else:
            workitem_type_2_commandutterances[workitem_type] = CommandUtterances(
                map_command_2_utterances=map_command_2_utterances
            )

    class Config:
        arbitrary_types_allowed = True

class UtteranceRegistry:   
    @classmethod
    def get_definition(cls, workflow_folderpath: str) -> UtteranceDefinition:
        if workflow_folderpath in cls._utterance_definitions:
            return cls._utterance_definitions[workflow_folderpath]
        
        utterancedefinitiondb_folderpath_dir = cls._get_utterancedefinition_db_folderpath()
        utterancedefinitiondb = Rdict(utterancedefinitiondb_folderpath_dir)
        utterance_definition = utterancedefinitiondb.get(workflow_folderpath, None)
        utterancedefinitiondb.close()

        if utterance_definition:
            cls._utterance_definitions[workflow_folderpath] = utterance_definition
            return utterance_definition
        
        return UtteranceRegistry._create_definition(workflow_folderpath)

    @classmethod
    def _create_definition(cls, workflow_folderpath: str) -> UtteranceDefinition:
        workitem_type_2_commandutterances: dict[str, CommandUtterances] = {}
        UtteranceDefinition._populate_utterance_definition(
            "", workflow_folderpath, workitem_type_2_commandutterances
        )

        utterance_definition = UtteranceDefinition(
            workitem_type_2_commandutterances=workitem_type_2_commandutterances
        )

        utterancedefinitiondb_folderpath_dir = cls._get_utterancedefinition_db_folderpath()
        utterancedefinitiondb = Rdict(utterancedefinitiondb_folderpath_dir)
        utterancedefinitiondb[workflow_folderpath] = utterance_definition
        utterancedefinitiondb.close()

        cls._utterance_definitions[workflow_folderpath] = utterance_definition
        return utterance_definition

    @classmethod
    def _get_utterancedefinition_db_folderpath(cls) -> str:
        """get the utterance definition db folder path"""
        SPEEDDICT_FOLDERNAME = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        utterancedefinition_db_folderpath = os.path.join(
            SPEEDDICT_FOLDERNAME,
            "utterancedefinitions"
        )
        os.makedirs(utterancedefinition_db_folderpath, exist_ok=True)
        return utterancedefinition_db_folderpath

    _utterance_definitions: dict[str, UtteranceDefinition] = {}
