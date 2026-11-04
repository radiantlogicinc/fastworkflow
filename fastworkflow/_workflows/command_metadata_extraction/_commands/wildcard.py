from typing import Any, Optional

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import Action, NLUPipelineStage
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.nlu_labels import PARAMETER_VALUE_PLACEHOLDERS

from ..intent_detection import CommandNamePrediction
from ..parameter_extraction import ParameterExtraction

class Signature:
    class Output(BaseModel):
        """The routing result of one message. The executor reads it and builds the turn's CommandOutput.

        handled: a CME or core command already answered the message (or a routing hint did), so the
            turn ends here with ``response``. A handled answer carries the artifact ``command_handled``.
        routed: True routes the message on to an application command (run by the executor next);
            False refuses it, and ``response`` is the answer (success=False).
        command / command_name / cmd_parameters / ran_on_context_*: the extraction, when the message
            is routed on to the application command ``command_name`` (run by the executor next).
        """

        handled: bool = False
        routed: bool = True
        response: str = ""
        command: Optional[str] = None
        command_name: Optional[str] = None
        cmd_parameters: Optional[Any] = Field(default=None, exclude=True)
        ran_on_context_name: Optional[str] = None
        ran_on_context_clause: Optional[str] = None
        # The nested command's Output, when a CME command answered (the handled path only).
        output: Optional[Any] = Field(default=None, exclude=True)

    # These are the PARAMETER_EXTRACTION stage's bare-value literals. They belong
    # to nlu_labels.PARAMETER_VALUE_LABEL, not to this command's INTENT_DETECTION
    # escalation class; see fastworkflow/nlu_labels.py for the stage split.
    plain_utterances = list(PARAMETER_VALUE_PLACEHOLDERS)

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        # Only the humanised command name. The placeholders above are excluded on
        # purpose: training them into the 'wildcard' label taught the escalation
        # classifier that a bare value like "france" means "escalate to my
        # parent". The trainer labels them under PARAMETER_VALUE_LABEL instead.
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ]


def _global_head_trained(app_workflow) -> bool:
    """Whether the workflow has a global head to ask: '*' has execution commands, and the trainer
    wrote its artifacts (``___command_info/global``, the folder ``CommandRouter`` reads for '*').
    Without one the global-first step is skipped silently, so no model is loaded for it."""
    import os

    from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER

    try:
        commands = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath).get_command_names("*")
    except Exception:  # noqa: BLE001 - a workflow whose routing cannot be read has no global head to ask
        return False
    if not commands:
        return False
    return os.path.isfile(os.path.join(
        app_workflow.folderpath, "___command_info", GLOBAL_CONTEXT_FOLDER, "threshold.json"))


def _confident_global_command(output) -> bool:
    """A confident answer naming one application command: not a decline, not an ambiguity, not a
    bare value, and not a framework (CME) command. The reserved labels resolve to no command name."""
    return (output.command_name is not None
            and output.error_msg is None
            and not output.answered_bare_value
            and not output.is_cme_command)


class ResponseGenerator:
    def __call__(
        self, 
        workflow: fastworkflow.Workflow, 
        command: str,
        input: None,
    ) -> Signature.Output:  # sourcery skip: hoist-if-from-if
        app_workflow = workflow.context["app_workflow"]   # type: fastworkflow.Workflow
        cmd_ctxt_obj_name = app_workflow.current_command_context_name
        nlu_pipeline_stage = workflow.context.get(
            "NLU_Pipeline_Stage", 
            NLUPipelineStage.INTENT_DETECTION)

        # The record the command runs on: the current object, unless an exact command name
        # resolves to an enclosing record or the global scope on the lookup path (3.9).
        bound_record = app_workflow.current_command_context
        typed_name = False  # the first token is a command name (exact, or a core command)
        if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
            from fastworkflow.context_navigation import command_on_path
            from fastworkflow.workflow_agent import core_command_names

            # A leading '/' is the deterministic-mode prefix, not part of the name (as in
            # workflow_agent._explicit_agent_command and command_executor). It is kept in `token`
            # so the slice below stays aligned, and stripped for every name comparison.
            token = command.split(None, 1)[0].split("(", 1)[0] if command.strip() else ""
            bare = token.lstrip("/")
            # A framework core command name is answered by the current scope first, always.
            core_short = {name.split("/")[-1].lower() for name in core_command_names()}
            typed_name = bool(bare) and bare.lower() in core_short
            found = (command_on_path(app_workflow, bare)
                     if bare and not typed_name else None)
            if found is not None:
                typed_name = True
            if found is not None and len(found[1]) == 1:
                scope, (name,) = found
                # The scope's own predict matches the short name exactly, as it does here.
                command = name.split("/")[-1] + command.strip()[len(token):]
                cmd_ctxt_obj_name = scope.context_name
                bound_record = scope.record

        predictor = CommandNamePrediction(workflow)
        cnp_output = None
        # Confident global first (owner-approved, step 3 bars): for natural language, the global
        # head '*' is asked before the current context's head. A confident command of '*' is the
        # answer and runs bound to the root object, as the climb binds it. Anything else is discarded
        # and today's flow runs unchanged. An exact name never reaches here (the lookup above).
        if (nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION
                and not typed_name
                and app_workflow.current_command_context_name != "*"
                and app_workflow.current_command_context is not None
                and app_workflow.root_command_context is not None
                and _global_head_trained(app_workflow)):
            global_output = predictor.predict("*", command, nlu_pipeline_stage)
            if _confident_global_command(global_output):
                cnp_output = global_output
                cmd_ctxt_obj_name = "*"
                bound_record = app_workflow.root_command_context
        if cnp_output is None:
            cnp_output = predictor.predict(cmd_ctxt_obj_name, command, nlu_pipeline_stage)

        if cnp_output.error_msg:
            workflow_context = workflow.context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION
            workflow_context["command"] = command
            workflow.context = workflow_context
            return Signature.Output(
                handled=False,
                routed=False,
                response=(
                    f"Ambiguous intent error for command '{command}'\n"
                    f"{cnp_output.error_msg}"
                ),
            )
        else:
            if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION and \
                cnp_output.command_name != 'ErrorCorrection/you_misunderstood':
                workflow_context = workflow.context
                workflow_context["command"] = command
                workflow.context = workflow_context
        
        if cnp_output.is_cme_command:
            workflow_context = workflow.context
            if cnp_output.command_name == 'ErrorCorrection/you_misunderstood':
                workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
                workflow_context["command"] = command
            elif (
                nlu_pipeline_stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION or
                cnp_output.command_name == 'ErrorCorrection/abort'
            ):
                workflow.end_command_processing()
            workflow.context = workflow_context

            startup_action = Action(
                command_name=cnp_output.command_name,
                command=command,
            )
            command_output = CommandExecutor.perform_action(workflow, startup_action)
            handled = (
                nlu_pipeline_stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION or
                cnp_output.command_name == 'ErrorCorrection/abort'
            )
            return Signature.Output(
                handled=handled,
                routed=command_output.success,
                response=command_output.command_response.response,
                output=command_output.command_response.output,
            )
        
        if nlu_pipeline_stage in {
                NLUPipelineStage.INTENT_DETECTION,
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
            }:
            app_workflow.command_context_for_response_generation = bound_record

            if cnp_output.command_name is None:
                # The hint the first declining context composed.
                # Every context on the chain declines the same token for the
                # same reason, so the first one is the whole story; it is kept
                # rather than recomputed because the walk overwrites cnp_output.
                routing_hint = cnp_output.routing_hint
                # A head that answered `parameter_value` has said the utterance is a
                # bare value, so no ancestor is asked (design 2.18, 3.9 step 4).
                while not cnp_output.command_name and \
                        not cnp_output.answered_bare_value and \
                        app_workflow.command_context_for_response_generation is not None and \
                            not app_workflow.is_command_context_for_response_generation_root:
                    app_workflow.command_context_for_response_generation = \
                        app_workflow.get_parent(app_workflow.command_context_for_response_generation)
                    cnp_output = predictor.predict(
                        fastworkflow.Workflow.get_command_context_name(app_workflow.command_context_for_response_generation), 
                        command, nlu_pipeline_stage)
                    routing_hint = routing_hint or cnp_output.routing_hint
            
                if cnp_output.command_name is None:
                    # The misunderstanding stage is included: its reply is
                    # matched against this context's full command set, so a
                    # real command owned elsewhere is declined there too, and
                    # the hint ends that stage the same way.
                    if nlu_pipeline_stage in (
                            NLUPipelineStage.INTENT_DETECTION,
                            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION,
                    ) and routing_hint:
                        # The name IS a command of this workflow; the walk simply
                        # never passed a context that owns it. "Nothing matched"
                        # is true and useless here, so say where it lives and how
                        # to get there. A hint only: nothing below navigates, and
                        # navigating on a guess about what was meant would change
                        # the workflow's state on the strength of that guess.
                        #
                        # Not a misunderstanding either, so no clarification
                        # stage: that stage matches the next message against this
                        # context's commands only, and the entering command the
                        # hint names usually lives elsewhere. Ending command
                        # processing routes that next message normally.
                        workflow.end_command_processing()
                        return Signature.Output(handled=True, routed=False, response=routing_hint)

                    if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
                        # out of scope commands
                        workflow_context = workflow.context
                        workflow_context["NLU_Pipeline_Stage"] = \
                            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
                        workflow_context["command"] = command
                        workflow.context = workflow_context

                        startup_action = Action(
                            command_name='ErrorCorrection/you_misunderstood',
                            command=command,
                        )
                        command_output = CommandExecutor.perform_action(workflow, startup_action)
                        return Signature.Output(
                            handled=True,
                            routed=command_output.success,
                            response=command_output.command_response.response,
                            output=command_output.command_response.output,
                        )

                    return Signature.Output(handled=False, routed=False, response=cnp_output.error_msg)

            # move to the parameter extraction stage
            workflow_context = workflow.context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.PARAMETER_EXTRACTION
            workflow.context = workflow_context

        if nlu_pipeline_stage == NLUPipelineStage.PARAMETER_EXTRACTION:
            cnp_output.command_name = workflow.context["command_name"]
        else:
            workflow_context = workflow.context
            workflow_context["command_name"] = cnp_output.command_name
            workflow.context = workflow_context

        command_name = cnp_output.command_name
        extraction_text = None
        if nlu_pipeline_stage == NLUPipelineStage.PARAMETER_EXTRACTION:
            first_token = command.split(None, 1)[0].split("(", 1)[0] if command.strip() else ""
            if ParameterExtraction._is_own_command_token(first_token, command_name):
                # The pending command sent again in full is a new extraction, not
                # an answer to the pending round: it replaces the stored command
                # and the parameters collected for it (fix-hia1.17).
                ParameterExtraction._clear_parameters(workflow)
                workflow_context = workflow.context
                workflow_context["command"] = command
                workflow.context = workflow_context
            else:
                # An answer to the pending round: what gets merged is this text,
                # not the stored command it answers.
                extraction_text = f'{command_name}: {command}'
        # Use the preserved original command (with parameters) if available
        preserved_command = f'{command_name}: {workflow.context.get("command", command)}'
        extractor = ParameterExtraction(
            workflow, app_workflow, command_name, extraction_text or preserved_command)
        pe_output = extractor.extract()
        if not pe_output.parameters_are_valid:
            return Signature.Output(
                handled=False,
                routed=False,
                response=(
                    f"PARAMETER EXTRACTION ERROR FOR COMMAND '{command_name}'\n"
                    f"{pe_output.error_msg}"
                ),
            )

        workflow.end_command_processing()

        # The walk bound an enclosing record: the command runs on that record, so it is
        # filed under it (design 3.9, "Filing the run"). Absent when it runs here.
        ran_on_name = ran_on_clause = None
        ran_on = app_workflow.command_context_for_response_generation
        if ran_on is not app_workflow.current_command_context:
            from fastworkflow.context_identity import identity_of
            from fastworkflow.observation_offloading.labels import context_clause

            ran_on_name, ran_on_label = identity_of(app_workflow, ran_on)
            ran_on_clause = (
                context_clause(ran_on_name, ran_on_label) if ran_on is not None else "")

        return Signature.Output(
            handled=False,
            routed=True,
            response="",
            command=preserved_command,
            command_name=command_name,
            cmd_parameters=pe_output.cmd_parameters,
            ran_on_context_name=ran_on_name,
            ran_on_context_clause=ran_on_clause,
        ) 