import json
import os
from typing import Optional

from pydantic import BaseModel, ConfigDict

from fastworkflow import CommandRoutingRegistry, CommandSource
from fastworkflow.command_routing_definition import ModuleType
import fastworkflow
from fastworkflow.command_directory import CommandDirectory, UtteranceMetadata
from fastworkflow.utils import python_utils


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
            command_metadata = command_directory.get_command_metadata(command_key)
            
            # Get the Signature class from the command module
            module = python_utils.get_module(
                command_metadata.response_generation_module_path,
                command_metadata.workflow_folderpath or workflow_folderpath
            )
            if not module:
                continue

            Signature = getattr(module, "Signature", None)
            if not Signature:
                continue

            # Extract utterances from the Signature class
            plain_utterances = getattr(Signature, "plain_utterances", [])
            template_utterances = getattr(Signature, "template_utterances", [])

            # Get generation function if it exists
            generated_utterances_module_filepath = ""
            generated_utterances_func_name = ""
            if hasattr(Signature, "generate_utterances"):
                generated_utterances_module_filepath = command_metadata.response_generation_module_path
                generated_utterances_func_name = "Signature.generate_utterances"

            utterance_metadata = UtteranceMetadata(
                workflow_folderpath=command_metadata.workflow_folderpath or workflow_folderpath,
                plain_utterances=plain_utterances,
                template_utterances=template_utterances,
                generated_utterances_module_filepath=generated_utterances_module_filepath,
                generated_utterances_func_name=generated_utterances_func_name,
            )

            command_directory.register_utterance_metadata(
                command_key, utterance_metadata
            )

    model_config = ConfigDict(arbitrary_types_allowed=True)

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