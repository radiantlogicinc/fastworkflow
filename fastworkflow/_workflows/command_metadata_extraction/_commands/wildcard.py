import fastworkflow
from fastworkflow import Action, CommandOutput, CommandResponse, NLUPipelineStage
from fastworkflow.command_executor import CommandExecutor

from ..intent_detection import CommandNamePrediction
from ..parameter_extraction import ParameterExtraction


class Signature:
    plain_utterances = [
        "3",
        "france",
        "16.7,.002",
        "John Doe, 56, 281-995-6423",
        "/path/to/my/object",
        "id=3636",
        "25.73 and Howard St",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + Signature.plain_utterances


class ResponseGenerator:
    def __call__(
        self, 
        workflow: fastworkflow.Workflow, 
        command: str,
    ) -> CommandOutput:  # sourcery skip: hoist-if-from-if
        app_workflow = workflow.context["app_workflow"]   # type: fastworkflow.Workflow
        cmd_ctxt_obj_name = app_workflow.current_command_context_name
        nlu_pipeline_stage = workflow.context.get(
            "NLU_Pipeline_Stage", 
            NLUPipelineStage.INTENT_DETECTION)

        predictor = CommandNamePrediction(workflow)           
        cnp_output = predictor.predict(cmd_ctxt_obj_name, command, nlu_pipeline_stage)

        if cnp_output.error_msg:
            workflow_context = workflow.context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION
            workflow_context["command"] = command
            workflow.context = workflow_context
            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response=(
                            f"Ambiguous intent error for command '{command}'\n"
                            f"{cnp_output.error_msg}"
                        ),
                        success=False
                    )
                ]
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
            if (
                nlu_pipeline_stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION or
                cnp_output.command_name == 'ErrorCorrection/abort'
            ):
                command_output.command_responses[0].artifacts["command_handled"] = True     
                # Set the additional attributes
                command_output.command_name = cnp_output.command_name
            return command_output
        
        if nlu_pipeline_stage in {
                NLUPipelineStage.INTENT_DETECTION,
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
            }:
            app_workflow.command_context_for_response_generation = \
                app_workflow.current_command_context

            if cnp_output.command_name is None:
                while not cnp_output.command_name and \
                    app_workflow.command_context_for_response_generation is not None and \
                        not app_workflow.is_command_context_for_response_generation_root:
                    app_workflow.command_context_for_response_generation = \
                        app_workflow.get_parent(app_workflow.command_context_for_response_generation)
                    cnp_output = predictor.predict(
                        fastworkflow.Workflow.get_command_context_name(app_workflow.command_context_for_response_generation), 
                        command, nlu_pipeline_stage)
            
                if cnp_output.command_name is None:
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
                        command_output.command_responses[0].artifacts["command_handled"] = True
                        return command_output

                    return CommandOutput(
                        command_responses=[
                            CommandResponse(
                                response=cnp_output.error_msg,
                                success=False
                            )
                        ]
                    )

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
        # Use the preserved original command (with parameters) if available
        preserved_command = f'{command_name}: {workflow.context.get("command", command)}'
        extractor = ParameterExtraction(workflow, app_workflow, command_name, preserved_command)
        pe_output = extractor.extract()
        if not pe_output.parameters_are_valid:
            return CommandOutput(
                command_name = command_name,
                command_responses=[
                    CommandResponse(
                        response=(
                            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{command_name}'\n"
                            f"{pe_output.error_msg}"
                        ),
                        success=False
                    )
                ]
            )

        workflow.end_command_processing()

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response="",
                    artifacts={
                        "command": preserved_command,
                        "command_name": command_name,
                        "cmd_parameters": pe_output.cmd_parameters,
                    },
                )
            ]
        ) 