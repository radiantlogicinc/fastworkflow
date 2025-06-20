from pydantic import BaseModel, ConfigDict

from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_context_model import CommandContextModel


class UtteranceDefinition(BaseModel):
    """
    Provides access to the utterances (sample phrases) for commands within a workflow.
    It retrieves command availability from the ContextModel and utterance
    details from the CommandDirectory.
    """
    workflow_folderpath: str
    context_model: CommandContextModel
    command_directory: CommandDirectory

    def get_command_names(self, context: str) -> list[str]:
        """Gets the names of commands available in a given context."""
        return self.context_model.commands(context)

    def get_command_utterances(self, command_name: str):
        """Gets the utterance metadata for a single, specific command."""
        try:
            return self.command_directory.get_utterance_metadata(command_name)
        except KeyError as e:
            raise ValueError(
                f"Could not find utterance metadata for command '{command_name}'. "
                "It might be missing from the _commands directory."
            ) from e

    def get_sample_utterances(self, command_context: str) -> list[str]:
        """Gets a sample utterance for each command in the given context."""
        command_names = self.get_command_names(command_context)
        sample_utterances = []
        for command_name in command_names:
            if command_name in {"wildcard", "abort", "misunderstood_intent"}:
                continue

            command_utterances = self.get_command_utterances(command_name)
            if not command_utterances:
                continue

            if command_utterances.template_utterances:
                sample_utterances.append(f"{command_name}: {command_utterances.template_utterances[0]}")
            elif command_utterances.plain_utterances:
                sample_utterances.append(f"{command_name}: {command_utterances.plain_utterances[0]}")
        
        return sample_utterances

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UtteranceRegistry:
    """A simple registry to get an UtteranceDefinition for a workflow."""
    _definitions: dict[str, UtteranceDefinition] = {}

    @classmethod
    def get_definition(cls, workflow_folderpath: str) -> UtteranceDefinition:
        """
        Gets the utterance definition for a workflow.
        If it doesn't exist, it will be created and cached.
        """
        if workflow_folderpath not in cls._definitions:
            # Ensure command directory is parsed once so utterance metadata is available.
            cmd_dir = CommandDirectory.load(workflow_folderpath)
            ctx_model = CommandContextModel.load(workflow_folderpath)
            cls._definitions[workflow_folderpath] = UtteranceDefinition(
                workflow_folderpath=workflow_folderpath,
                context_model=ctx_model,
                command_directory=cmd_dir,
            )
        
        return cls._definitions[workflow_folderpath]

    @classmethod
    def clear_registry(cls):
        """Clears the registry. Useful for testing."""
        cls._definitions.clear()