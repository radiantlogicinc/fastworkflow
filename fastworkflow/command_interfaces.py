from abc import ABC, abstractmethod
import fastworkflow

class CommandExecutorInterface(ABC):
    @abstractmethod
    def invoke_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        pass

    @abstractmethod
    def perform_action(
        self,
        session: fastworkflow.Session,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:
        pass