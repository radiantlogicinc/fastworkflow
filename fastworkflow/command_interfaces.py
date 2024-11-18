from abc import ABC, abstractmethod
import fastworkflow

class CommandRouterInterface(ABC):
    @abstractmethod
    def route_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        pass

class CommandExecutorInterface(ABC):
    @abstractmethod
    def invoke_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command_name: str,
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