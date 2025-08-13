from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Set, Dict, Any, List
from pydantic import BaseModel
import json

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
    def _get_enhanced_command_info(self, workflow: fastworkflow.Workflow) -> Dict[str, Any]:
        """
        Get enhanced command information for agent mode.
        Returns structured JSON with context info, command details, etc.
        """
        app_workflow = workflow.context["app_workflow"]

        # Get the current context information
        context_info = {
            "name": app_workflow.current_command_context_name,
            "display_name": app_workflow.current_command_context_displayname,
            "description": "",  # Could be extracted from context class docstring
            "inheritance": [],  # Could be extracted from context_inheritance_model.json
            "containment": []   # Could be extracted from context_containment_model.json
        }

        # Try to load context hierarchy information if available
        with contextlib.suppress(Exception):
            inheritance_path = Path(app_workflow.folderpath) / "context_inheritance_model.json"
            if inheritance_path.exists():
                with open(inheritance_path) as f:
                    inheritance_data = json.load(f)
                    context_info["inheritance"] = inheritance_data.get(context_info["name"], [])

            containment_path = Path(app_workflow.folderpath) / "context_containment_model.json"
            if containment_path.exists():
                with open(containment_path) as f:
                    containment_data = json.load(f)
                    context_info["containment"] = containment_data.get(context_info["name"], [])
        # Get command information
        subject_crd = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
        cme_crd = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)

        # Get available commands
        cme_command_names = cme_crd.get_command_names('IntentDetection')
        subject_command_names = subject_crd.get_command_names(app_workflow.current_command_context_name)

        candidate_commands = set(cme_command_names) | set(subject_command_names)

        # Filter and build command details
        commands = []
        for fq_cmd in candidate_commands:
            if fq_cmd == "wildcard":
                continue

            # Check if command has utterances
            utterance_meta = (
                subject_crd.command_directory.get_utterance_metadata(fq_cmd) or
                cme_crd.command_directory.get_utterance_metadata(fq_cmd)
            )

            if not utterance_meta:
                continue

            cmd_name = fq_cmd.split("/")[-1]

            # Get command signature information if available
            signature_info = {}
            with contextlib.suppress(Exception):
                # Try to get command module for signature extraction
                cmd_module = None
                try:
                    cmd_module = subject_crd.get_command_module(fq_cmd)
                except Exception:
                    with contextlib.suppress(Exception):
                        cmd_module = cme_crd.get_command_module(fq_cmd)
                
                if cmd_module and hasattr(cmd_module, 'Signature'):
                    sig_class = cmd_module.Signature
                    if hasattr(sig_class, 'Input'):
                        input_class = sig_class.Input
                        signature_info["inputs"] = [
                            {
                                "name": field_name,
                                "type": str(field_info.annotation),
                                "description": field_info.description or "",
                                "examples": getattr(field_info, 'examples', []),
                                "default": str(field_info.default) if field_info.default else None
                            }
                            for field_name, field_info in input_class.model_fields.items()
                        ]

                    # Get plain utterances if available
                    if hasattr(sig_class, 'plain_utterances'):
                        signature_info["plain_utterances"] = sig_class.plain_utterances
            # Get docstring from utterance metadata if available
            docstring = ""
            if hasattr(utterance_meta, 'docstring'):
                docstring = utterance_meta.docstring or ""

            commands.append({
                "qualified_name": fq_cmd,
                "name": cmd_name,
                "signature_docstring": docstring,
                **signature_info
            })

        return {
            "context": context_info,
            "commands": sorted(commands, key=lambda x: x["name"])
        }
    
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
        # Check if we're in agent mode by looking for chat session run_as_agent flag
        is_agent_mode = False
        with contextlib.suppress(Exception):
            if fastworkflow.chat_session:
                is_agent_mode = fastworkflow.chat_session.run_as_agent
        
        if is_agent_mode:
            # Return enhanced JSON structure for agent mode
            enhanced_info = self._get_enhanced_command_info(workflow)
            response = json.dumps(enhanced_info, indent=2)
        else:
            # Return traditional text format for assistant mode
            output = self._process_command(workflow)

            # Include the current context display name in the header so callers see
            # which context is active (e.g., "TodoListManager" or "global/*").
            app_workflow = workflow.context["app_workflow"]
            context_name_for_display = app_workflow.current_command_context_displayname
            response_body = "\n".join(output.valid_command_names)
            response = (
                f"Commands available in the current context: {context_name_for_display}\n"
                f"{response_body}\n"
            )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=response,
                )
            ]
        ) 
