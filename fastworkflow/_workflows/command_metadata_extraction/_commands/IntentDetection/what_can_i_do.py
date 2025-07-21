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
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(
        self, workflow: fastworkflow.Workflow
    ) -> Signature.Output:
        """
        Provides helpful information about this type of work-item.
        If the workitem_path is not provided, it provides information about the current work-item.

        :param input: The input parameters for the function.
        """
        app_workflow = workflow.context["app_workflow"]
        subject_crd = fastworkflow.RoutingRegistry.get_definition(
            app_workflow.folderpath)
        
        crd = fastworkflow.RoutingRegistry.get_definition(
            workflow.folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        # ------------------------------------------------------------------
        # Build the union of command names that *should* be visible, then
        # filter out any command that (a) is the special wildcard helper or
        # (b) has no user-facing utterances in *any* command directory.
        # ------------------------------------------------------------------

        candidate_commands: set[str] = (
            set(cme_command_names)
            | set(subject_crd.get_command_names(app_workflow.current_command_context_name))
        )

        def _has_utterances(fq_cmd: str) -> bool:
            """Return True if *fq_cmd* has at least one utterance definition in
            either the subject workflow or the CME workflow."""
            return (
                subject_crd.command_directory.get_utterance_metadata(fq_cmd) is not None
                or crd.command_directory.get_utterance_metadata(fq_cmd) is not None
            )

        visible_commands = [
            fq_cmd for fq_cmd in candidate_commands
            if fq_cmd != "wildcard" and _has_utterances(fq_cmd)
        ]

        valid_command_names = [
            cmd.split("/")[-1] for cmd in sorted(visible_commands)
        ]

        return Signature.Output(valid_command_names=valid_command_names)

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str,
    ) -> fastworkflow.CommandOutput:
        output = self._process_command(workflow)

        response = "\n".join(output.valid_command_names)
        response = (
            f"Commands available in the current context:\n"
            f"{response}\n"
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=response,
                )
            ]
        ) 
