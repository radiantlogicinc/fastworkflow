from __future__ import annotations

from pathlib import Path
from typing import Set
from pydantic import BaseModel

import fastworkflow

from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
class Signature:
    class Output(BaseModel):
        valid_command_names: list[str]

    # Constants from plain_utterances.json
    plain_utterances = [
        "show me all the utterances",
        "what can you do?",
        "what are my options?",
        "what are my choices?",
        "what are my capabilities?",
        "what can i do?",
        "what can i use?",
        "what are my tools?",
        "now what?",
        "list commands",
        "list utterances"
    ]

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(session.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        
        utterance_list: list[str] = [command_name] + result

        return utterance_list


class ResponseGenerator:
    def _process_command(
        self, session: fastworkflow.Session
    ) -> Signature.Output:
        """
        Provides helpful information about this type of work-item.
        If the workitem_path is not provided, it provides information about the current work-item.

        :param input: The input parameters for the function.
        """
        sub_sess = session.workflow_context["subject_session"]
        subject_crd = fastworkflow.RoutingRegistry.get_definition(
            sub_sess.workflow_folderpath)
        
        crd = fastworkflow.RoutingRegistry.get_definition(
            session.workflow_folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        fully_qualified_command_names = (
            set(cme_command_names) | 
            set(subject_crd.get_command_names(sub_sess.current_command_context_name))
        ) - {'wildcard'}

        valid_command_names = [
            fully_qualified_command_name.split('/')[-1] 
            for fully_qualified_command_name in fully_qualified_command_names
        ]

        return Signature.Output(valid_command_names=sorted(valid_command_names))

    def __call__(
        self,
        session: fastworkflow.Session,
        command: str,
    ) -> fastworkflow.CommandOutput:
        output = self._process_command(session)

        subject_session = session.workflow_context["subject_session"]
        context_name = (
            'global' if subject_session.current_command_context_name == '*'
            else subject_session.current_command_context_name
        )

        response = "\n".join([f"{cmd}" for cmd in output.valid_command_names])
        response = (
            f"Commands available in the current context ({context_name}):\n"
            f"{response}\n"
        )

        return fastworkflow.CommandOutput(
            session_id=session.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=response,
                )
            ]
        ) 
