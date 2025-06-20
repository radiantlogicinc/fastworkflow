from __future__ import annotations

from pathlib import Path
from typing import Set
from pydantic import BaseModel

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.command_router import CommandRouter
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
class Signature:
    class Output(BaseModel):
        command_context_name: str
        utterances: list[str]

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
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
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
        # Get the current workitem type for the subject workflow snapshot
        sub_sess = session.workflow_snapshot.workflow_context["subject_session"]
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(sub_sess.workflow_snapshot.workflow_folderpath)
        cur_cmd_ctxt_cls_name = sub_sess.current_command_context_name
        utterances = utterance_definition.get_sample_utterances(cur_cmd_ctxt_cls_name)

        return Signature.Output(
            command_context_name=cur_cmd_ctxt_cls_name, 
            utterances=utterances
        )

    def __call__(
        self,
        session: fastworkflow.Session,
        command: str,
    ) -> fastworkflow.CommandOutput:
        output = self._process_command(session)

        # Format the list of commands
        utterance_list = "\n".join([f"- {cmd}" for cmd in output.utterances])

        # Create the response
        response = (
            f"Here are some example commands available in this task ({output.command_context_name}):\n"
            f"{utterance_list}\n"
            f"Your chat message must fall within the scope of these utterances."
        )

        return fastworkflow.CommandOutput(
            session_id=session.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=response,
                )
            ]
        ) 
    # def _ensure_router(self, snapshot: WorkflowSnapshot) -> CommandRouter:
    #     """Get or build a CommandRouter attached to the snapshot."""
    #     router: CommandRouter | None = getattr(snapshot, "_command_router", None)  # type: ignore[attr-defined]
    #     if router is None:
    #         # Assume commands live in `<workflow_path>/_commands`
    #         commands_root = Path(snapshot.workflow_folderpath) / "_commands"
    #         router = CommandRouter(commands_root)
    #         router.scan()
    #         setattr(snapshot, "_command_router", router)
    #     return router
