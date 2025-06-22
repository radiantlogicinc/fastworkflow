from enum import Enum
import sys
from typing import Dict, List, Optional, Type, Union
import json
import os

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from speedict import Rdict

import fastworkflow
from fastworkflow import Action, CommandOutput, CommandResponse
from fastworkflow.cache_matching import cache_match, change_flag, get_flag, store_utterance_cache
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_routing_definition import CommandRoutingDefinition, ModuleType
import fastworkflow.command_routing_definition
from fastworkflow.model_pipeline_training import (
    predict_single_sentence,
    get_artifact_path,
)
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.fuzzy_match import find_best_match
from fastworkflow.utils.signatures import InputForParamExtraction


INVALID_INT_VALUE = fastworkflow.get_env_var("INVALID_INT_VALUE")
INVALID_FLOAT_VALUE = fastworkflow.get_env_var("INVALID_FLOAT_VALUE")

MISSING_INFORMATION_ERRMSG = fastworkflow.get_env_var("MISSING_INFORMATION_ERRMSG")
INVALID_INFORMATION_ERRMSG = fastworkflow.get_env_var("INVALID_INFORMATION_ERRMSG")

NOT_FOUND = fastworkflow.get_env_var("NOT_FOUND")
INVALID = fastworkflow.get_env_var("INVALID")

class NLUPipelineStage(Enum):
    """Specifies the stages of the NLU Pipeline processing."""
    INTENT_DETECTION = 0
    INTENT_AMBIGUITY_CLARIFICATION = 1
    PARAMETER_EXTRACTION = 2


class CommandNamePrediction:
    class Output(BaseModel):
        command_name: Optional[str] = None
        error_msg: Optional[str] = None
        is_cme_command: bool = False

    def __init__(self, session: fastworkflow.Session):
        self.session = session
        self.sub_sess = session.workflow_snapshot.workflow_context["subject_session"]
        self.sub_sess_workflow_folderpath = self.sub_sess.workflow_snapshot.workflow_folderpath
        self.sub_session_id = self.sub_sess.id

        self.convo_path = os.path.join(self.sub_sess_workflow_folderpath, "___convo_info")
        self.cache_path = self._get_cache_path(self.sub_session_id, self.convo_path)
        self.path = self._get_cache_path_cache(self.convo_path)

        # Determine current command context for the subject session.  The
        # global context "*" maps to a folder named "_global".
        ctx_name = self.sub_sess.current_command_context_name or "*"

        tiny_path = get_artifact_path(self.sub_sess_workflow_folderpath, ctx_name, "tinymodel.pth")
        large_path = get_artifact_path(self.sub_sess_workflow_folderpath, ctx_name, "largemodel.pth")
        threshold_path = get_artifact_path(self.sub_sess_workflow_folderpath, ctx_name, "threshold.json")
        ambiguous_threshold_path = get_artifact_path(self.sub_sess_workflow_folderpath, ctx_name, "ambiguous_threshold.json")
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            confidence_threshold = data['confidence_threshold']
        with open(ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.ambiguos_confidence_threshold = data['confidence_threshold']

        self.modelpipeline = fastworkflow.ModelPipelineRegistry(
            tiny_model_path=tiny_path,
            distil_model_path=large_path,
            confidence_threshold=confidence_threshold
        )

        cme_workflow_folderpath = self.session.workflow_snapshot.workflow_folderpath
        tiny_path = get_artifact_path(cme_workflow_folderpath, "ErrorCorrection", "tinymodel.pth")
        large_path = get_artifact_path(cme_workflow_folderpath, "ErrorCorrection", "largemodel.pth")
        threshold_path = get_artifact_path(cme_workflow_folderpath, "ErrorCorrection", "threshold.json")
        ambiguous_threshold_path = get_artifact_path(cme_workflow_folderpath, "ErrorCorrection", "ambiguous_threshold.json")
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            confidence_threshold = data['confidence_threshold']
        with open(ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.ambiguos_confidence_threshold = data['confidence_threshold']

        self.err_corr_modelpipeline = fastworkflow.ModelPipelineRegistry(
            tiny_model_path=tiny_path,
            distil_model_path=large_path,
            confidence_threshold=confidence_threshold
        )

    def predict(self, command_context_name: str, command: str, nlu_pipeline_stage: NLUPipelineStage) -> "CommandNamePrediction.Output":
        # sourcery skip: extract-duplicate-method
        crd = fastworkflow.CommandRoutingRegistry.get_definition(
            self.session.workflow_snapshot.workflow_folderpath)
        
        if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
            modelpipeline = self.modelpipeline
            cme_command_names = crd.get_command_names('IntentDetection')
            label_encoder_path = get_artifact_path(self.sub_sess_workflow_folderpath, command_context_name, "label_encoder.pkl")

            subject_crd = fastworkflow.CommandRoutingRegistry.get_definition(
                self.sub_sess_workflow_folderpath)
            valid_command_names = (
                set(cme_command_names) | 
                set(subject_crd.get_command_names(command_context_name))
            )
        else:
        # if stage is INTENT_AMBIGUITY_CLARIFICATION or PARAMETER_EXTRACTION
            modelpipeline = self.err_corr_modelpipeline
            cme_workflow_folderpath = self.session.workflow_snapshot.workflow_folderpath
            label_encoder_path = get_artifact_path(cme_workflow_folderpath, "ErrorCorrection", "label_encoder.pkl")
            cme_command_names = crd.get_command_names('ErrorCorrection')
            valid_command_names = crd.get_command_names('ErrorCorrection')

        # Check if the entire command is a valid command name
        normalized_command = command.replace(" ", "_").lower()
        command_name = next(
            (
                name
                for name in valid_command_names
                if normalized_command == name.lower()
            ),
            None,
        )

        flag = get_flag(self.path)

        # Check if we're in constrained mode (flag != 0)
        if flag not in [0, None]:
            flag_type = self._get_flag_type(self.path)

            # Special cases: allow abort or "None of these" without @ prefix
            if command.lower() == "abort" or normalized_command == "abort":
                command_name = "Core/abort"
                change_flag(self.path, 0)  # Reset flag
            elif any(phrase in command.lower() for phrase in [
                "none of these",
                "none of the above", 
                "neither",
                "none",
                "misunderstood"
            ]):
                # User wants to see all options instead of the top 3
                error_msg = self._formulate_misclassified_command_error_message(valid_command_names)
                # Set flag to 2 because user is indicating none of the suggestions match
                self._store_suggested_commands(self.path, valid_command_names, 2)
                change_flag(self.path, 2)
                return CommandNamePrediction.Output(error_msg=error_msg)
            else:
                # Only accept commands prefixed with @ that match the suggested commands
                suggested_commands = self._get_suggested_commands(self.path)

                # Create appropriate message based on flag type
                message_prefix = "The command is ambiguous" if flag_type == 1 else "The previous command was misclassified"

                if "@" in command:
                    # Extract everything after @ until the next @ or end of string
                    full_command_text = command.split("@", 1)[1]
                    after_at = full_command_text.split("@", 1)[0] if "@" in full_command_text else full_command_text
                    after_at = after_at.strip()

                    # Get the first part for reference (still needed for command parts)
                    tentative_command_name = after_at.split()[0] if " " in after_at else after_at

                # Try different matching strategies in order of strictness
                valid_choice = False
                matched_command = None

                # Use Levenshtein distance for fuzzy matching with the full command part after @
                matched_command, distance = find_best_match(
                    command,
                    suggested_commands,
                    threshold=0.3  # Adjust threshold as needed
                )
                valid_choice = matched_command is not None

                if valid_choice:
                    command_name = matched_command
                    # Remove the entire @command part from the input
                    command = command
                else:
                    # User selected an option that wasn't in the suggested list
                    error_msg = f"{message_prefix}. Please select only from the provided command options:\n"
                    error_msg += "\n".join(f"@{name}" for name in suggested_commands)
                    error_msg += "\n\nor type 'none of these' to see all commands\nor type 'abort' to cancel"
                    return CommandNamePrediction.Output(error_msg=error_msg)

                # Process the selected command
                count = self._get_count(self.cache_path)
                if count > 0:
                    utterance = self._read_utterance(self.cache_path, count-1)
                    store_utterance_cache(self.path, utterance, command_name, modelpipeline)
                change_flag(self.path, 0)  # Reset flag

                # If user selects none_of_these in constrained mode, show all valid commands
                if command_name == "Core/misunderstood_intent":
                    error_msg = self._formulate_misclassified_command_error_message(valid_command_names)
                    # Set flag to 2 because this is explicitly a misclassification case
                    self._store_suggested_commands(self.path, valid_command_names, 2)
                    change_flag(self.path, 2)
                    return CommandNamePrediction.Output(error_msg=error_msg)
        else:
            # Normal flow (not in constrained mode)

            # If user explicitly selects none_of_these, treat as misclassification
            if command_name == "Core/misunderstood_intent":
                error_msg = self._formulate_misclassified_command_error_message(valid_command_names)
                # Set flag to 2 because user is indicating previous command was misclassified
                self._store_suggested_commands(self.path, valid_command_names, 2)
                change_flag(self.path, 2)
                return CommandNamePrediction.Output(error_msg=error_msg)

            if command.startswith('@'):
                tentative_command_name = command.split("@")[1].split()[0].rstrip(':-')
                normalized_command_name = tentative_command_name.lower()
                for name in valid_command_names:
                    if normalized_command_name == name.lower():
                        command_name = name
                        command = command.replace(f"@{tentative_command_name}", "").strip().replace("  ", " ")
                        break
                if command_name == "Core/misunderstood_intent":
                    error_msg = self._formulate_misclassified_command_error_message(valid_command_names)
                    # Set flag to 2 because user is indicating previous command was misclassified
                    self._store_suggested_commands(self.path, valid_command_names, 2)
                    change_flag(self.path, 2)
                    return CommandNamePrediction.Output(error_msg=error_msg)

            if not command_name:
                if cache_result := cache_match(
                    self.path, command, modelpipeline, 0.85
                ):
                    command_name = cache_result
                    flag = get_flag(self.path)
                    if flag is not None and flag != 0:
                        count = self._get_count(self.cache_path)
                        utterance = self._read_utterance(self.cache_path, count-1)
                        store_utterance_cache(self.path, utterance, command_name, modelpipeline)
                        change_flag(self.path, 0)
                else:
                    # If no cache match, use the model to predict
                    results = predict_single_sentence(modelpipeline, command, label_encoder_path)
                    command_name = results['label']

                    # If confidence is low, treat as ambiguous command (type 1)
                    if results['confidence'] < self.ambiguos_confidence_threshold:
                        error_msg = self._formulate_ambiguous_command_error_message(results["topk_labels"])
                        count = self._store_utterance(self.cache_path, command, command_name)
                        # Store suggested commands and set flag to 1 (ambiguous)
                        self._store_suggested_commands(self.path, results["topk_labels"], 1)
                        change_flag(self.path, 1)
                        return CommandNamePrediction.Output(error_msg=error_msg)

                    # If model prediction is none_of_these, present all commands as options
                    if command_name == "Core/misunderstood_intent":
                        error_msg = self._formulate_misclassified_command_error_message(valid_command_names)
                        # Set flag to 2 because model couldn't classify the command
                        self._store_suggested_commands(self.path, valid_command_names, 2)
                        change_flag(self.path, 2)
                        return CommandNamePrediction.Output(error_msg=error_msg)
                    else:
                        flag = get_flag(self.path)
                        if flag is not None and flag != 0:
                            count = self._get_count(self.cache_path)
                            utterance = self._read_utterance(self.cache_path, count-1)
                            store_utterance_cache(self.path, utterance, command_name, modelpipeline)
                            change_flag(self.path, 0)
            else:
                # When the command_name is already determined
                flag = get_flag(self.path)
                if flag is not None and flag != 0:
                    count = self._get_count(self.cache_path)
                    utterance = self._read_utterance(self.cache_path, count-1)
                    store_utterance_cache(self.path, utterance, command_name, modelpipeline)
                    change_flag(self.path, 0)

        # Store the final command and classification
        if command_name:
            if command_name != "Core/misunderstood_intent":
                count = self._store_utterance(self.cache_path, command, command_name)

            class ValidateCommandNameSignature(BaseModel):
                command_name: str
            command_parameters = ValidateCommandNameSignature(command_name=command_name)
            is_valid, error_msg = self._validate_command_name(valid_command_names, command_parameters)

            # If validation fails, set flag to 2 (misclassified)
            if not is_valid:
                self._store_suggested_commands(self.path, valid_command_names, 2)
                change_flag(self.path, 2)
                return CommandNamePrediction.Output(error_msg=error_msg)

            return CommandNamePrediction.Output(
                command_name=command_parameters.command_name,
                is_cme_command=command_parameters.command_name in cme_command_names
            )
        
        return CommandNamePrediction.Output(command_name=None, is_cme_command=False)

    @staticmethod
    def _get_cache_path(session_id, convo_path):
        """
        Generate cache file path based on session ID
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{session_id}.db")

    @staticmethod
    def _get_cache_path_cache(convo_path):
        """
        Generate cache file path based on session ID
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

    # Get the flag type
    @staticmethod
    def _get_flag_type(cache_path):
        """
        Get the type of constraint (1=ambiguous, 2=misclassified)
        """
        db = Rdict(cache_path)
        try:
            return db.get("flag_type", 1)  # Default to ambiguous if not set
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
    def _validate_command_name(
        valid_command_names: list[str],
        command_parameters: "CommandNamePrediction.ValidateCommandNameSignature"
    ) -> tuple[bool, str]:
        """
        Validate if the command name is valid and get the required parameters.
        """
        if command_parameters.command_name in valid_command_names:
            return (True, None)

        if not command_parameters.command_name and "*" in valid_command_names:
            command_parameters.command_name = "*"
            return (True, None)

        command_list = "\n".join(f"@{name}" for name in valid_command_names)
        return (
            False,
            "Please select the correct command from the list below:\n"
            f"{command_list}\n\nor type 'abort' to cancel"
        )

    @staticmethod
    def _formulate_ambiguous_command_error_message(route_choice_list: list[str]) -> str:
        command_list = (
            "\n".join([
                f"@{route_choice}"
                for route_choice in route_choice_list
            ])
        )

        return (
            "The command is ambiguous. Please select from these possible options:\n"
            f"{command_list}\n\n"
            "or type 'none of these' to see all commands\n"
            "or type 'abort' to cancel"
        )

    @staticmethod
    def _formulate_misclassified_command_error_message(route_choice_list: list[str]) -> str:
        command_list = (
            "\n".join([
                f"@{route_choice}"
                for route_choice in route_choice_list
            ])
        )

        return (
            "Please select the correct command from the list below:\n"
            f"{command_list}\n\nor type 'abort' to cancel"
        )

class ParameterExtraction:
    class Output(BaseModel):
        parameters_are_valid: bool
        cmd_parameters: Optional[BaseModel] = None
        error_msg: Optional[str] = None
        suggestions: Optional[Dict[str, List[str]]] = None

    def __init__(self, session: fastworkflow.Session, subject_session: fastworkflow.Session, command_name: str, command: str):
        self.session = session
        self.subject_session = subject_session
        self.command_name = command_name
        self.command = command

    def extract(self) -> "ParameterExtraction.Output":
        subject_workflow_folderpath = self.subject_session.workflow_snapshot.workflow_folderpath
        subject_command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(subject_workflow_folderpath)

        command_parameters_class = (
            subject_command_routing_definition.get_command_class(
                self.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            return self.Output(parameters_are_valid=True)

        stored_params = self._get_stored_parameters(self.session)

        input_for_param_extraction = InputForParamExtraction.create(self.subject_session, self.command_name, self.command)

        if stored_params:
            _, _, _, stored_missing_fields = self._extract_missing_fields(input_for_param_extraction, self.subject_session, self.command_name, stored_params)
        else:
            stored_missing_fields = []

        new_params = self._extract_command_parameters_from_input(
            input_for_param_extraction,
            command_parameters_class,
            stored_missing_fields,
            self.command_name,
            subject_workflow_folderpath
        )

        if stored_params:
            merged_params = self._merge_parameters(stored_params, new_params, stored_missing_fields)
        else:
            merged_params = new_params

        self._store_parameters(self.session, merged_params)

        is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(
            self.subject_session, self.command_name, merged_params
        )

        if not is_valid:
            if params_str := self._format_parameters_for_display(merged_params):
                error_msg = f"Extracted parameters so far:\n{params_str}\n\n{error_msg}"

            error_msg += "\nEnter 'abort' if you want to abort the command."
            error_msg += "\nEnter 'you misunderstood' if the wrong command was executed."
            return self.Output(
                parameters_are_valid=False,
                error_msg=error_msg,
                cmd_parameters=merged_params,
                suggestions=suggestions)

        self._clear_parameters(self.session)
        return self.Output(
            parameters_are_valid=True,
            cmd_parameters=merged_params)

    @staticmethod
    def _get_stored_parameters(session):
        return session.workflow_snapshot.workflow_context.get("stored_parameters")

    @staticmethod
    def _store_parameters(session, parameters):
        session.workflow_snapshot.workflow_context["stored_parameters"] = parameters

    @staticmethod
    def _clear_parameters(session):
        if "stored_parameters" in session.workflow_snapshot.workflow_context:
            del session.workflow_snapshot.workflow_context["stored_parameters"]

    @staticmethod
    def _extract_missing_fields(input_for_param_extraction, sws, command_name, stored_params):
        stored_missing_fields = []
        is_valid, error_msg, suggestions = input_for_param_extraction.validate_parameters(
            sws, command_name, stored_params
        )

        if not is_valid:
            if MISSING_INFORMATION_ERRMSG in error_msg:
                missing_fields_str = error_msg.split(f"{MISSING_INFORMATION_ERRMSG}\n")[1].split("\n")[0]
                stored_missing_fields = [f.strip() for f in missing_fields_str.split(",")]
            if INVALID_INFORMATION_ERRMSG in error_msg:
                invalid_section = error_msg.split(f"{INVALID_INFORMATION_ERRMSG}\n")[1]
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
        merged = old_params.model_copy()

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
                        setattr(merged, field_name, new_value)

                    elif isinstance(old_value, float) and old_value == INVALID_FLOAT_VALUE:
                        setattr(merged, field_name, new_value)

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
                -sys.maxsize,
                -sys.float_info.max
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
    def _extract_command_parameters_from_input(
        input_for_param_extraction: BaseModel,
        command_parameters_class: Type[BaseModel],
        missing_fields: list = None,
        subject_command_name: str = None,
        subject_workflow_folderpath: str = None,
    ) -> BaseModel:
        """
        Extract command parameters from user input.
        This implementation handles any parameter type.
        """
        if missing_fields:
            default_params = InputForParamExtraction.populate_defaults_dict(
                command_parameters_class)
            return ParameterExtraction._apply_missing_fields(
                input_for_param_extraction.command, default_params, missing_fields)

        return input_for_param_extraction.extract_parameters(command_parameters_class, subject_command_name, subject_workflow_folderpath)

    @staticmethod
    def _apply_missing_fields(command: str, default_params: BaseModel, missing_fields: list):
        params = default_params.model_copy()

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

        return params    

class ResponseGenerator:
    def __call__(
        self, 
        session: fastworkflow.Session, 
        command: str,
    ) -> CommandOutput:
        session.workflow_snapshot.is_complete = False

        subject_session = session.workflow_snapshot.workflow_context["subject_session"]   # type: fastworkflow.Session
        cmd_ctxt_obj_name = subject_session.current_command_context_name
        nlu_pipeline_stage = session.workflow_snapshot.workflow_context.get(
            "NLU_Pipeline_Stage", 
            NLUPipelineStage.INTENT_DETECTION)

        predictor = CommandNamePrediction(session)           
        cnp_output = predictor.predict(cmd_ctxt_obj_name, command, nlu_pipeline_stage)

        if cnp_output.is_cme_command:
            session.workflow_snapshot.is_complete = True
            session.workflow_snapshot.workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_DETECTION

            startup_action = Action(
                workitem_path="/command_metadata_extraction",
                command_name=cnp_output.command_name,
                command=command,
            )
            command_executor = CommandExecutor()
            command_output = command_executor.perform_action(session, startup_action)
            if len(command_output.command_responses) > 1:
                raise ValueError("Multiple command responses returned from command_metadata_extraction workflow")    
            # set command_handled to true
            command_output.command_responses[0].artifacts["command_handled"] = True
            return command_output
        
        if nlu_pipeline_stage in {
                NLUPipelineStage.INTENT_DETECTION,
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION
            }:
            subject_session.command_context_for_response_generation = \
                subject_session.current_command_context

            if cnp_output.command_name is None:
                while not cnp_output.command_name and \
                    subject_session.command_context_for_response_generation is not None and \
                        not subject_session.is_command_context_for_response_generation_root:
                    subject_session.command_context_for_response_generation = \
                        subject_session.get_parent(subject_session.command_context_for_response_generation)
                    cnp_output = predictor.predict(
                        fastworkflow.Session.get_command_context_name(subject_session.command_context_for_response_generation), 
                        command, nlu_pipeline_stage)
            
                if not cnp_output.command_name:
                    session.workflow_snapshot.workflow_context["NLU_Pipeline_Stage"] = \
                        NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION

                    return CommandOutput(
                        command_responses=[
                            CommandResponse(
                                response=cnp_output.error_msg,
                                success=False
                            )
                        ]
                    )

            # move to the parameter extraction stage
            workflow_context = session.workflow_snapshot.workflow_context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.PARAMETER_EXTRACTION
            workflow_context["command_name"] = cnp_output.command_name
            workflow_context["command"] = command
            session.workflow_snapshot.workflow_context = workflow_context

        command_name = session.workflow_snapshot.workflow_context["command_name"]
        extractor = ParameterExtraction(session, subject_session, command_name, command)
        pe_output = extractor.extract()
        if not pe_output.parameters_are_valid:
            return CommandOutput(
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

        session.workflow_snapshot.is_complete = True
        session.workflow_snapshot.workflow_context["NLU_Pipeline_Stage"] = \
            NLUPipelineStage.INTENT_DETECTION

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response="",
                    artifacts={
                        "command_name": command_name,
                        "cmd_parameters": pe_output.cmd_parameters,
                    },
                )
            ]
        ) 