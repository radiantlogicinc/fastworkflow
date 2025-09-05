from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel
import json

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.command_metadata_api import CommandMetadataAPI

class Signature:
    """List all the commands available in the current context along with their metadata"""
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
        "what can i do here?",
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
    # def _get_enhanced_command_info(self, workflow: fastworkflow.Workflow) -> Dict[str, Any]:
    #     """
    #     Get enhanced command information for agent mode.
    #     Returns structured JSON with context info, command details, etc.
    #     """
    #     app_workflow = workflow.context["app_workflow"]

    #     # Get the current context information
    #     context_info = {
    #         "name": app_workflow.current_command_context_name,
    #         "display_name": app_workflow.current_command_context_displayname,
    #         "description": "",
    #         "inheritance": [],
    #         "containment": []
    #     }

    #     # Try to load context hierarchy information if available
    #     with contextlib.suppress(Exception):
    #         inheritance_path = Path(app_workflow.folderpath) / "context_inheritance_model.json"
    #         if inheritance_path.exists():
    #             with open(inheritance_path) as f:
    #                 inheritance_data = json.load(f)
    #                 context_info["inheritance"] = inheritance_data.get(context_info["name"], [])

    #         containment_path = Path(app_workflow.folderpath) / "context_containment_model.json"
    #         if containment_path.exists():
    #             with open(containment_path) as f:
    #                 containment_data = json.load(f)
    #                 context_info["containment"] = containment_data.get(context_info["name"], [])

    #     from fastworkflow.command_metadata_api import CommandMetadataAPI

    #     meta = CommandMetadataAPI.get_enhanced_command_info(
    #         subject_workflow_path=app_workflow.folderpath,
    #         cme_workflow_path=workflow.folderpath,
    #         active_context_name=app_workflow.current_command_context_name,
    #     )

    #     return {
    #         "context": context_info,
    #         "commands": meta.get("commands", [])
    #     }
    
    # def _process_command(
    #     self, workflow: fastworkflow.Workflow
    # ) -> Signature.Output:
    #     """
    #     Provides helpful information about this type of work-item.
    #     If the workitem_path is not provided, it provides information about the current work-item.

    #     :param input: The input parameters for the function.
    #     """
    #     app_workflow = workflow.context["app_workflow"]
    #     subject_crd = fastworkflow.RoutingRegistry.get_definition(
    #         app_workflow.folderpath)
        
    #     crd = fastworkflow.RoutingRegistry.get_definition(
    #         workflow.folderpath)
    #     cme_command_names = crd.get_command_names('IntentDetection')

    #     # ------------------------------------------------------------------
    #     # Build the union of command names that *should* be visible, then
    #     # filter out any command that (a) is the special wildcard helper or
    #     # (b) has no user-facing utterances in *any* command directory.
    #     # ------------------------------------------------------------------

    #     candidate_commands: set[str] = (
    #         set(cme_command_names)
    #         | set(subject_crd.get_command_names(app_workflow.current_command_context_name))
    #     )

    #     def _has_utterances(fq_cmd: str) -> bool:
    #         """Return True if *fq_cmd* has at least one utterance definition in
    #         either the subject workflow or the CME workflow."""
    #         return (
    #             subject_crd.command_directory.get_utterance_metadata(fq_cmd) is not None
    #             or crd.command_directory.get_utterance_metadata(fq_cmd) is not None
    #         )

    #     visible_commands = [
    #         fq_cmd for fq_cmd in candidate_commands
    #         if fq_cmd != "wildcard" and _has_utterances(fq_cmd)
    #     ]

    #     valid_command_names = [
    #         cmd.split("/")[-1] for cmd in sorted(visible_commands)
    #     ]

    #     return Signature.Output(valid_command_names=valid_command_names)

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str,
    ) -> fastworkflow.CommandOutput:
        # Check if we're in agent mode by looking for chat session run_as_agent flag
        is_agent_mode = False
        with contextlib.suppress(Exception):
            if fastworkflow.chat_session:
                is_agent_mode = fastworkflow.chat_session.run_as_agent
        
        app_workflow = workflow.context["app_workflow"]
        response = CommandMetadataAPI.get_command_display_text(
            subject_workflow_path=app_workflow.folderpath,
            cme_workflow_path=workflow.folderpath,
            active_context_name=app_workflow.current_command_context_name,
            for_agents=is_agent_mode,
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=response,
                )
            ]
        ) 
