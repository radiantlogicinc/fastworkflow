from abc import ABC, abstractmethod
import fastworkflow

class CommandExecutorInterface(ABC):
    @abstractmethod
    def invoke_command(
        self,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        pass

    @abstractmethod
    def perform_action(
        self,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:
        pass