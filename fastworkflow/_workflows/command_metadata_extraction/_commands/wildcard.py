import contextlib
from enum import Enum
import sys
from typing import Dict, List, Optional, Type, Union
import json
import os

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from speedict import Rdict

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import Action, CommandOutput, CommandResponse, ModuleType, NLUPipelineStage
from fastworkflow.cache_matching import cache_match, store_utterance_cache
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_routing import RoutingDefinition
import fastworkflow.command_routing
from fastworkflow.model_pipeline_training import (
    predict_single_sentence,
    get_artifact_path,
    CommandRouter
)

from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.fuzzy_match import find_best_matches
from fastworkflow.utils.signatures import InputForParamExtraction


INVALID_INT_VALUE = -sys.maxsize
INVALID_FLOAT_VALUE = -sys.float_info.max

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
INVALID = fastworkflow.get_env_var("INVALID")
PARAMETER_EXTRACTION_ERROR_MSG = None


class CommandNamePrediction:
    class Output(BaseModel):
        command_name: Optional[str] = None
        error_msg: Optional[str] = None
        is_cme_command: bool = False

    def __init__(self, cme_workflow: fastworkflow.Workflow):
        self.cme_workflow = cme_workflow
        self.app_workflow = cme_workflow.context["app_workflow"]
        self.app_workflow_folderpath = self.app_workflow.folderpath
        self.app_workflow_id = self.app_workflow.id

        self.convo_path = os.path.join(self.app_workflow_folderpath, "___convo_info")
        self.cache_path = self._get_cache_path(self.app_workflow_id, self.convo_path)
        self.path = self._get_cache_path_cache(self.convo_path)

    def predict(self, command_context_name: str, command: str, nlu_pipeline_stage: NLUPipelineStage) -> "CommandNamePrediction.Output":
        # sourcery skip: extract-duplicate-method

        model_artifact_path = f"{self.app_workflow_folderpath}/___command_info/{command_context_name}"
        command_router = CommandRouter(model_artifact_path)

        # Re-use the already-built ModelPipeline attached to the router
        # instead of instantiating a fresh one.  This avoids reloading HF
        # checkpoints and transferring tensors each time we see a new
        # message for the same context.
        modelpipeline = command_router.modelpipeline

        crd = fastworkflow.RoutingRegistry.get_definition(
            self.cme_workflow.folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        valid_command_names = set()
        if nlu_pipeline_stage == NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION:
            valid_command_names = self._get_suggested_commands(self.path)
        elif nlu_pipeline_stage in (
                NLUPipelineStage.INTENT_DETECTION, NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION):
            app_crd = fastworkflow.RoutingRegistry.get_definition(
                self.app_workflow_folderpath)
            valid_command_names = (
                set(cme_command_names) | 
                set(app_crd.get_command_names(command_context_name))
            )

        command_name_dict = {
            fully_qualified_command_name.split('/')[-1]: fully_qualified_command_name 
            for fully_qualified_command_name in valid_command_names
        }

        if nlu_pipeline_stage == NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION:
            # what_can_i_do is special in INTENT_AMBIGUITY_CLARIFICATION
            # We will not predict, just match plain utterances with exact or fuzzy match
            command_name_dict |= {
                plain_utterance: 'IntentDetection/what_can_i_do'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'IntentDetection/what_can_i_do'
                ].plain_utterances
            }

        if nlu_pipeline_stage != NLUPipelineStage.INTENT_DETECTION:
            # abort is special. 
            # We will not predict, just match plain utterances with exact or fuzzy match
            command_name_dict |= {
                plain_utterance: 'ErrorCorrection/abort'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'ErrorCorrection/abort'
                ].plain_utterances
            }

        if nlu_pipeline_stage != NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION:
            # you_misunderstood is special. 
            # We will not predict, just match plain utterances with exact or fuzzy match
            command_name_dict |= {
                plain_utterance: 'ErrorCorrection/you_misunderstood'
                for plain_utterance in crd.command_directory.map_command_2_utterance_metadata[
                    'ErrorCorrection/you_misunderstood'
                ].plain_utterances
            }

        # See if the command starts with a command name followed by a space
        tentative_command_name = command.split(" ", 1)[0]
        normalized_command_name = tentative_command_name.lower()
        command_name = None
        if normalized_command_name in command_name_dict:
            command_name = normalized_command_name
            command = command.replace(f"{tentative_command_name}", "").strip().replace("  ", " ")
        else:
            # Use Levenshtein distance for fuzzy matching with the full command part after @
            best_matched_commands, _ = find_best_matches(
                command.replace(" ", "_"),
                command_name_dict.keys(),
                threshold=0.3  # Adjust threshold as needed
            )
            if best_matched_commands:
                command_name = best_matched_commands[0]

        if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
            if not command_name:
                if cache_result := cache_match(self.path, command, modelpipeline, 0.85):
                    command_name = cache_result
                else:
                    predictions=command_router.predict(command)

                    if len(predictions)==1:
                        command_name = predictions[0].split('/')[-1]
                    else:
                        # If confidence is low, treat as ambiguous command (type 1)
                        error_msg = self._formulate_ambiguous_command_error_message(predictions)
                        # Store suggested commands
                        self._store_suggested_commands(self.path, predictions, 1)
                        return CommandNamePrediction.Output(error_msg=error_msg)

        elif nlu_pipeline_stage in (
            NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
        ) and not command_name:
            command_name = "what_can_i_do"

        if not command_name or command_name == "wildcard":
            fully_qualified_command_name=None
            is_cme_command=False
        else:
            fully_qualified_command_name = command_name_dict[command_name]
            is_cme_command=(
                fully_qualified_command_name in cme_command_names or 
                fully_qualified_command_name in crd.get_command_names('ErrorCorrection')
            )

        if (
            nlu_pipeline_stage
            in (
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION,
            )
            and not fully_qualified_command_name.endswith('abort')
            and not fully_qualified_command_name.endswith('what_can_i_do')
            and not fully_qualified_command_name.endswith('you_misunderstood')
        ):
            command = self.cme_workflow.context["command"]
            store_utterance_cache(self.path, command, command_name, modelpipeline)

        return CommandNamePrediction.Output(
            command_name=fully_qualified_command_name,
            is_cme_command=is_cme_command
        )

    @staticmethod
    def _get_cache_path(workflow_id, convo_path):
        """
        Generate cache file path based on workflow ID
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{workflow_id}.db")

    @staticmethod
    def _get_cache_path_cache(convo_path):
        """
        Generate cache file path based on workflow ID
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "cache.db")

    # Store the suggested commands with the flag type
    @staticmethod
    def _store_suggested_commands(cache_path, command_list, flag_type):
        """
        Store the list of suggested commands for the constrained selection

        Args:
            cache_path: Path to the cache database
            command_list: List of suggested commands
            flag_type: Type of constraint (1=ambiguous, 2=misclassified)
        """
        db = Rdict(cache_path)
        try:
            db["suggested_commands"] = command_list
            db["flag_type"] = flag_type
        finally:
            db.close()

    # Get the suggested commands
    @staticmethod
    def _get_suggested_commands(cache_path):
        """
        Get the list of suggested commands for the constrained selection
        """
        db = Rdict(cache_path)
        try:
            return db.get("suggested_commands", [])
        finally:
            db.close()

    @staticmethod
    def _get_count(cache_path):
        db = Rdict(cache_path)
        try:
            return db.get("utterance_count", 0)  # Default to 0 if key doesn't exist
        finally:
            db.close()

    @staticmethod
    def _print_db_contents(cache_path):
        db = Rdict(cache_path)
        try:
            print("All keys in database:", list(db.keys()))
            for key in db.keys():
                print(f"Key: {key}, Value: {db[key]}")
        finally:
            db.close()

    @staticmethod
    def _store_utterance(cache_path, utterance, label):
        """
        Store utterance in existing or new database
        Returns: The utterance count used
        """
        # Open the database (creates if doesn't exist)
        db = Rdict(cache_path)

        try:
            # Get existing counter or initialize to 0
            utterance_count = db.get("utterance_count", 0)

            # Create and store the utterance entry
            utterance_data = {
                "utterance": utterance,
                "label": label
            }

            db[utterance_count] = utterance_data

            # Increment and store the counter
            utterance_count += 1
            db["utterance_count"] = utterance_count

            return utterance_count - 1  # Return the count used for this utterance

        finally:
            # Always close the database
            db.close()

    # Function to read from database
    @staticmethod
    def _read_utterance(cache_path, utterance_id):
        """
        Read a specific utterance from the database
        """
        db = Rdict(cache_path)
        try:
            return db.get(utterance_id)['utterance']
        finally:
            db.close()

    @staticmethod
    def _formulate_ambiguous_command_error_message(route_choice_list: list[str]) -> str:
        command_list = (
            "\n".join([
                f"{route_choice.split('/')[-1].lower()}"
                for route_choice in route_choice_list if route_choice != 'wildcard'
            ])
        )

        return (
            "The command is ambiguous. Please select from these possible options:\n"
            f"{command_list}\n\n"
            "or type 'what can i do' to see all commands\n"
            "or type 'abort' to cancel"
        )

class ParameterExtraction:
    class Output(BaseModel):
        parameters_are_valid: bool
        cmd_parameters: Optional[BaseModel] = None
        error_msg: Optional[str] = None
        suggestions: Optional[Dict[str, List[str]]] = None

    def __init__(self, cme_workflow: fastworkflow.Workflow, app_workflow: fastworkflow.Workflow, command_name: str, command: str):
        self.cme_workflow = cme_workflow
        self.app_workflow = app_workflow
        self.command_name = command_name
        self.command = command

    def extract(self) -> "ParameterExtraction.Output":
        app_workflow_folderpath = self.app_workflow.folderpath
        app_command_routing_definition = fastworkflow.RoutingRegistry.get_definition(app_workflow_folderpath)

        command_parameters_class = (
            app_command_routing_definition.get_command_class(
                self.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            return self.Output(parameters_are_valid=True)

        stored_params = self._get_stored_parameters(self.cme_workflow)

        self.command = self.command.replace(self.command_name, "").strip()

        input_for_param_extraction = InputForParamExtraction.create(
            self.app_workflow, self.command_name, 
            self.command)

        if stored_params:
            _, _, _, stored_missing_fields = self._extract_missing_fields(input_for_param_extraction, self.app_workflow, self.command_name, stored_params)
        else:
            stored_missing_fields = []
            
        # If we have missing fields (in parameter extraction error state), try to apply the command directly
        if stored_missing_fields:
            # Apply the command directly as parameter values
            direct_params = self._apply_missing_fields(self.command, stored_params, stored_missing_fields)
            new_params = direct_params
        else:
            # Otherwise use the LLM-based extraction
            new_params = input_for_param_extraction.extract_parameters(
                command_parameters_class, 
                self.command_name, 
                app_workflow_folderpath)

        if stored_params:
            merged_params = self._merge_parameters(stored_params, new_params, stored_missing_fields)
        else:
            merged_params = new_params

        self._store_parameters(self.cme_workflow, merged_params)

        is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(
            self.app_workflow, self.command_name, merged_params
        )

        if not is_valid:
            if params_str := self._format_parameters_for_display(merged_params):
                error_msg = f"Extracted parameters so far:\n{params_str}\n\n{error_msg}"

            error_msg += "\nEnter 'abort' to get out of this error state and/or execute a different command."
            error_msg += "\nEnter 'you misunderstood' if the wrong command was executed."
            return self.Output(
                parameters_are_valid=False,
                error_msg=error_msg,
                cmd_parameters=merged_params,
                suggestions=suggestions)

        self._clear_parameters(self.cme_workflow)
        return self.Output(
            parameters_are_valid=True,
            cmd_parameters=merged_params)

    @staticmethod
    def _get_stored_parameters(cme_workflow: fastworkflow.Workflow):
        return cme_workflow.context.get("stored_parameters")

    @staticmethod
    def _store_parameters(cme_workflow: fastworkflow.Workflow, parameters):
        cme_workflow.context["stored_parameters"] = parameters

    @staticmethod
    def _clear_parameters(cme_workflow: fastworkflow.Workflow):
        if "stored_parameters" in cme_workflow.context:
            del cme_workflow.context["stored_parameters"]

    @staticmethod
    def _extract_missing_fields(input_for_param_extraction, sws, command_name, stored_params):
        stored_missing_fields = []
        is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(
            sws, command_name, stored_params
        )

        if not is_valid:
            if MISSING_INFORMATION_ERRMSG in error_msg:
                missing_fields_str = error_msg.split(f"{MISSING_INFORMATION_ERRMSG}")[1].split("\n")[0]
                stored_missing_fields = [f.strip() for f in missing_fields_str.split(",")]
            if INVALID_INFORMATION_ERRMSG in error_msg:
                invalid_section = error_msg.split(f"{INVALID_INFORMATION_ERRMSG}")[1]
                if "\n" in invalid_section:
                    invalid_fields_str = invalid_section.split("\n")[0]
                    stored_missing_fields.extend(
                        invalid_field.split(" '")[0].strip()
                        for invalid_field in invalid_fields_str.split(", ")
                    )
        return is_valid, error_msg, suggestions, stored_missing_fields

    @staticmethod
    def _merge_parameters(old_params, new_params, missing_fields):
        """
        Merge new parameters with old parameters, prioritizing new values when appropriate.
        """
        global PARAMETER_EXTRACTION_ERROR_MSG
        if not PARAMETER_EXTRACTION_ERROR_MSG:
            PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

        merged = old_params.model_copy()

        try:
            all_fields = list(old_params.model_fields.keys())
            missing_fields = missing_fields or []

            for field_name in all_fields:
                if hasattr(new_params, field_name):
                    new_value = getattr(new_params, field_name)
                    old_value = getattr(merged, field_name)

                    if new_value is not None and new_value != NOT_FOUND:
                        if isinstance(old_value, str) and INVALID in old_value and INVALID not in new_value:
                            setattr(merged, field_name, new_value)

                        elif old_value is None or old_value == NOT_FOUND:
                            setattr(merged, field_name, new_value)

                        elif isinstance(old_value, int) and old_value == INVALID_INT_VALUE:
                            with contextlib.suppress(ValueError, TypeError):
                                setattr(merged, field_name, int(new_value))

                        elif isinstance(old_value, float) and old_value == INVALID_FLOAT_VALUE:
                            with contextlib.suppress(ValueError, TypeError):
                                setattr(merged, field_name, float(new_value))

                        elif (field_name in missing_fields and
                            hasattr(merged.model_fields.get(field_name), "json_schema_extra") and
                            merged.model_fields.get(field_name).json_schema_extra and
                            "db_lookup" in merged.model_fields.get(field_name).json_schema_extra):
                            setattr(merged, field_name, new_value)

                        elif field_name in missing_fields:
                            field_info = merged.model_fields.get(field_name)
                            has_pattern = hasattr(field_info, "pattern") and field_info.pattern is not None

                            if not has_pattern:
                                for meta in getattr(field_info, "metadata", []):
                                    if hasattr(meta, "pattern"):
                                        has_pattern = True
                                        break

                            if not has_pattern and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                                has_pattern = "pattern" in field_info.json_schema_extra

                            if has_pattern:
                                setattr(merged, field_name, new_value)
        except Exception as exc:
            logger.warning(PARAMETER_EXTRACTION_ERROR_MSG.format(error=exc))

        return merged

    @staticmethod
    def _format_parameters_for_display(params):
        """
        Format parameters for display in the error message.
        """
        if not params:
            return ""

        lines = []

        all_fields = list(params.model_fields.keys())

        for field_name in all_fields:
            value = getattr(params, field_name, None)

            if value in [
                NOT_FOUND, 
                None,
                INVALID_INT_VALUE,
                INVALID_FLOAT_VALUE
            ]:
                continue

            display_name = " ".join(word.capitalize() for word in field_name.split('_'))

            # Format fields appropriately based on type
            if (
                isinstance(value, bool)
                or not hasattr(value, 'value')
                and isinstance(value, (int, float))
                or not hasattr(value, 'value')
                and isinstance(value, str)
                or not hasattr(value, 'value')
            ):
                lines.append(f"{display_name}: {value}")
            else:  # Handle enum types
                lines.append(f"{display_name}: {value.value}")
        return "\n".join(lines)

    @staticmethod
    def _apply_missing_fields(command: str, default_params: BaseModel, missing_fields: list):
        global PARAMETER_EXTRACTION_ERROR_MSG
        if not PARAMETER_EXTRACTION_ERROR_MSG:
            PARAMETER_EXTRACTION_ERROR_MSG = fastworkflow.get_env_var("PARAMETER_EXTRACTION_ERROR_MSG")

        params = default_params.model_copy()

        try:
            if "," in command:
                parts = [part.strip() for part in command.split(",")]

                if len(parts) == len(missing_fields):
                    if len(missing_fields) == 1:
                        field = missing_fields[0]
                        if hasattr(params, field):
                            setattr(params, field, parts[0])
                            return params
                    elif len(missing_fields) > 1:
                        for i, field in enumerate(missing_fields):
                            if i < len(parts) and hasattr(params, field):
                                setattr(params, field, parts[i])
                        return params
                else:
                    if parts and missing_fields:
                        field = missing_fields[0]
                        if hasattr(params, field):
                            setattr(params, field, parts[0])
                    return params

            elif missing_fields:
                field = missing_fields[0]
                if hasattr(params, field):
                    setattr(params, field, command.strip())
                    return params
        
        except Exception as exc:
            # logger.warning(PARAMETER_EXTRACTION_ERROR_MSG.format(error=exc))
            pass

        return params    


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
            else:
                workflow.end_command_processing()
            workflow.context = workflow_context

            startup_action = Action(
                command_name=cnp_output.command_name,
                command=command,
            )
            command_output = CommandExecutor.perform_action(workflow, startup_action)
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
            workflow_context["command_name"] = cnp_output.command_name
            workflow.context = workflow_context

        command_name = workflow.context["command_name"]
        extractor = ParameterExtraction(workflow, app_workflow, command_name, command)
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
                        "command": command,
                        "command_name": command_name,
                        "cmd_parameters": pe_output.cmd_parameters,
                    },
                )
            ]
        ) 