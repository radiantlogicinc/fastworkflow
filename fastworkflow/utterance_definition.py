import json
import os
from typing import Optional

from pydantic import BaseModel

from fastworkflow import CommandRoutingRegistry, CommandSource
import fastworkflow
from fastworkflow.command_directory import CommandDirectory, UtteranceMetadata


class UtteranceDefinition(BaseModel):
    workflow_folderpath: str
    def get_command_names(self, workitem_path: str) -> list[str]:
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(self.workflow_folderpath)
        return command_routing_definition.get_command_names(workitem_path)

    def get_command_utterances(
        self, workitem_path: str, command_name: str
    ) -> UtteranceMetadata:
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(self.workflow_folderpath)
        return command_routing_definition.get_utterance_metadata(
            workitem_path, command_name
        )

    def get_sample_utterances(self, workitem_path: str) -> list[str]:
        command_names = self.get_command_names(workitem_path)
        sample_utterances = []
        for command_name in command_names:
            if command_name=="*":
                continue
            if command_name=="abort":
                continue
            if command_name=="None_of_these":
                continue
            command_utterances = self.get_command_utterances(workitem_path, command_name)
            if command_utterances.template_utterances:
                sample_utterances.append(command_utterances.template_utterances[0])
            elif command_utterances.plain_utterances:
                sample_utterances.append(command_utterances.plain_utterances[0])
        return sample_utterances

    @classmethod
    def _populate_utterance_definition(
        cls,
        workflow_folderpath: str,
        command_directory: CommandDirectory,
    ):
        for command_key in command_directory.get_command_keys():
            # if command_key.endswith("/*"):
            #     continue
            
            command_metadata = command_directory.get_command_metadata(command_key)
            command_source = command_metadata.command_source
            commands_folder = os.path.join(
                workflow_folderpath 
                if command_metadata.workflow_folderpath is None 
                else command_metadata.workflow_folderpath, 
                command_source.value)
            command = command_key.split("/")[-1]
            subfolder_path = os.path.join(commands_folder, command)

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

            utterance_metadata = UtteranceMetadata(
                workflow_folderpath=commands_folder,
                plain_utterances=plain_utterances,
                template_utterances=template_utterances,
                generated_utterances_module_filepath=generated_utterances_module_filepath,
                generated_utterances_func_name=generated_utterances_func_name,
            )

            command_directory.register_utterance_metadata(
                command_key, utterance_metadata
            )

    class Config:
        arbitrary_types_allowed = True

class UtteranceRegistry:   
    @classmethod
    def create_definition(cls, workflow_folderpath: str) -> Optional[CommandDirectory]:
        command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
            workflow_folderpath)
        if not command_routing_definition:
            return None
        UtteranceDefinition._populate_utterance_definition(
            workflow_folderpath,
            command_routing_definition.command_directory)

        command_routing_definition.command_directory.save()
        return command_routing_definition.command_directory

    @classmethod
    def get_definition(cls, workflow_folderpath: str) -> UtteranceDefinition:
        return UtteranceDefinition(workflow_folderpath=workflow_folderpath)