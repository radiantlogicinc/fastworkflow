from typing import Optional
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from speedict import Rdict

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import NLUPipelineStage
from fastworkflow.cache_matching import cache_match, store_utterance_cache
from fastworkflow.model_pipeline_training import (
    CommandRouter
)

from fastworkflow.utils.fuzzy_match import find_best_matches


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

        # See if the command starts with a command name followed by a space or a '('
        tentative_command_name = command.split(" ", 1)[0].split("(", 1)[0]
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
                    # predictions = majority_vote_predictions(command_router, command)

                    if len(predictions)==1:
                        command_name = predictions[0].split('/')[-1]
                    else:
                        # If confidence is low, treat as ambiguous command (type 1)
                        error_msg = self._formulate_ambiguous_command_error_message(
                            predictions, "run_as_agent" in self.app_workflow.context)

                        # Store suggested commands
                        self._store_suggested_commands(self.path, predictions, 1)
                        return CommandNamePrediction.Output(error_msg=error_msg)

        elif nlu_pipeline_stage in (
            NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
        ) and not command_name:
            command_name = "what can i do?"

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
    def _formulate_ambiguous_command_error_message(
        route_choice_list: list[str], run_as_agent: bool) -> str:
        command_list = (
            "\n".join([
                f"{route_choice.split('/')[-1].lower()}"
                for route_choice in route_choice_list if route_choice != 'wildcard'
            ])
        )

        return (
            "The command is ambiguous. "
            + (
                "Choose the correct command name from these possible options and update your command:\n"
                if run_as_agent
                else "Please choose a command name from these possible options:\n"
            )
            + f"{command_list}\n\nor type 'what can i do' to see all commands\n"
            + ("or type 'abort' to cancel" if run_as_agent else '')
        )


# TODO - generation is deterministic. They all return the same answer
# TODO - Need 'temperature' for intent detection pipeline
def majority_vote_predictions(command_router, command: str, n_predictions: int = 5) -> list[str]:
    """
    Generate N prediction sets in parallel and return the set that wins the majority vote.
    
    This function improves prediction reliability by running multiple parallel predictions
    and selecting the most common result through majority voting. This helps reduce
    the impact of random variations in model predictions.
    
    Args:
        command_router: The CommandRouter instance to use for predictions
        command: The input command string
        n_predictions: Number of parallel predictions to generate (default: 5)
                      Can be configured via N_PARALLEL_PREDICTIONS environment variable
        
    Returns:
        The prediction set that received the majority vote. Falls back to a single
        prediction if all parallel predictions fail.
        
    Note:
        Uses ThreadPoolExecutor with max_workers limited to min(n_predictions, 10)
        to avoid overwhelming the system with too many concurrent threads.
    """
    def get_single_prediction():
        """Helper function to get a single prediction"""
        return command_router.predict(command)
    
    # Generate N predictions in parallel
    prediction_sets = []
    with ThreadPoolExecutor(max_workers=min(n_predictions, 10)) as executor:
        # Submit all prediction tasks
        futures = [executor.submit(get_single_prediction) for _ in range(n_predictions)]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                prediction_set = future.result()
                prediction_sets.append(prediction_set)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                # Continue with other predictions even if one fails
    
    if not prediction_sets:
        # Fallback to single prediction if all parallel predictions failed
        logger.warning("All parallel predictions failed, falling back to single prediction")
        return command_router.predict(command)
    
    # Convert lists to tuples so they can be hashed and counted
    prediction_tuples = [tuple(sorted(pred_set)) for pred_set in prediction_sets]
    
    # Count occurrences of each unique prediction set
    vote_counts = Counter(prediction_tuples)
    
    # Get the prediction set with the most votes
    winning_tuple = vote_counts.most_common(1)[0][0]
    
    # Convert back to list and return
    return list(winning_tuple)
