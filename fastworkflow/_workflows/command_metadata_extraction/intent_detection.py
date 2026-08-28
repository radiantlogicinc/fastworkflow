from typing import Optional
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow import NLUPipelineStage, tracing
from fastworkflow.cache_matching import cache_match, store_utterance_cache
from fastworkflow.kvstore import KVStore
from fastworkflow.model_pipeline_training import (
    CommandRouter
)
from fastworkflow.nlu_labels import is_escalation, is_non_routable

from fastworkflow.utils.fuzzy_match import find_best_matches


# A low-confidence top-k prediction containing an escalation label remains an
# ambiguity: prompt with the local command candidates and log that the parent signal
# was discarded. This is product behaviour, not workflow configuration.


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
        """Predict, wrapped in a ``fw.nlu.intent`` span (D3 as amended).

        One span per prediction attempt — the wildcard command's parent-chain
        walk calls this once per context, and each attempt is recorded with
        the context it ran against. The span carries which matching layer
        decided (exact prefix / fuzzy pre-match / embedding cache /
        classifier), the classifier's confidence and threshold when it ran,
        and the candidate set on an ambiguity. Emission never affects the
        prediction: the helpers no-op without a bound host/sink.
        """
        host = tracing.current_host()
        span = tracing.start_span(
            host,
            tracing.SPAN_NLU_INTENT,
            attributes={
                "context": command_context_name,
                "stage": nlu_pipeline_stage.name,
                "utterance": command,
            },
        )
        nlu_trace: dict = {}
        try:
            output = self._predict_impl(
                command_context_name, command, nlu_pipeline_stage, nlu_trace
            )
        except BaseException:
            tracing.end_span(
                host, span, status=tracing.STATUS_ERROR, attributes=nlu_trace
            )
            raise
        tracing.end_span(
            host,
            span,
            status=tracing.STATUS_OK,
            attributes={
                **nlu_trace,
                "command_name": output.command_name,
                "is_cme_command": output.is_cme_command,
                "ambiguous": output.error_msg is not None,
                # None = no local prediction; the caller walks up the context
                # chain (or files a misunderstanding) — exactly the routing
                # signal a debugging agent needs.
                "resolved": output.command_name is not None,
            },
        )
        return output

    def _predict_impl(
        self,
        command_context_name: str,
        command: str,
        nlu_pipeline_stage: NLUPipelineStage,
        nlu_trace: dict,
    ) -> "CommandNamePrediction.Output":
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
            nlu_trace["matcher_layer"] = "exact_prefix"
        else:
            # Use Levenshtein distance for fuzzy matching with the full command part after @
            # No match is ([], None), never (None, None) — len() here is safe.
            best_matched_commands, _ = find_best_matches(
                command.replace(" ", "_"),
                command_name_dict.keys(),
                threshold=0.3  # Adjust threshold as needed
            )
            if (
                len(best_matched_commands) > 1
                and nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION
            ):
                # Commands sharing a prefix tie at distance 0, because scoring
                # compares only the leading len(input) characters, and
                # command_name_dict iterates a set — so picking [0] would choose
                # nondeterministically between them across processes. Leave the
                # name unset so the classifier and its ambiguity prompt decide.
                # The clarification stages are deliberately excluded: they have
                # no classifier to fall back to.
                logger.warning(
                    f"Fuzzy pre-match tied across {best_matched_commands} for "
                    f"utterance '{command}' in context '{command_context_name}'. "
                    "Deferring to the classifier instead of picking one."
                )
                nlu_trace["fuzzy_prematch_tie"] = [str(c) for c in best_matched_commands]
            elif best_matched_commands:
                command_name = best_matched_commands[0]
                nlu_trace["matcher_layer"] = "fuzzy_prematch"

        if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
            if not command_name:
                if cache_result := cache_match(self.path, command, modelpipeline, 0.85):
                    command_name = cache_result
                    nlu_trace["matcher_layer"] = "embedding_cache"
                    nlu_trace["cache_similarity_threshold"] = 0.85
                else:
                    predictions, classifier_details = (
                        command_router.predict_with_details(command)
                    )
                    # predictions = majority_vote_predictions(command_router, command)
                    nlu_trace["matcher_layer"] = "classifier"
                    nlu_trace["classifier"] = classifier_details

                    if len(predictions)==1:
                        command_name = predictions[0].split('/')[-1]
                    else:
                        # If confidence is low, treat as ambiguous command (type 1)
                        if escalation_signals := self.escalation_signals_in(predictions):
                            logger.warning(
                                f"Top-k escalation signal discarded in "
                                f"context '{command_context_name}' for utterance "
                                f"'{command}'. predictions={predictions}, "
                                f"suppressed={escalation_signals}. The classifier ranked "
                                "an escalation label alongside local candidates; the user "
                                "will be prompted with the local candidates only and the "
                                "'this belongs to an ancestor context' signal is dropped."
                            )
                            nlu_trace["escalation_labels_discarded"] = [
                                str(label) for label in escalation_signals
                            ]

                        error_msg = self._formulate_ambiguous_command_error_message(
                            predictions, "run_as_agent" in self.app_workflow.context)

                        # Store suggested commands
                        nlu_trace["candidates"] = [str(p) for p in predictions]
                        self._store_suggested_commands(self.path, predictions, 1)
                        return CommandNamePrediction.Output(error_msg=error_msg)

        elif nlu_pipeline_stage in (
            NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
        ) and not command_name:
            command_name = "what can i do?"
            nlu_trace["matcher_layer"] = "clarification_default"

        fully_qualified_command_name = self.resolve_fully_qualified_command_name(
            command_name, command_name_dict)
        if fully_qualified_command_name is None:
            is_cme_command=False
        else:
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
            # A reserved label resolves to None; the clarification cache keys on a
            # real command name, so there is nothing to store for it.
            and fully_qualified_command_name is not None
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
        return os.path.join(base_dir, f"{workflow_id}.sqlite3")

    @staticmethod
    def _get_cache_path_cache(convo_path):
        """
        Generate cache file path based on workflow ID
        """
        base_dir = convo_path
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "cache.sqlite3")

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
        with KVStore(cache_path) as db:
            # predict() returns a numpy ndarray of labels; JSON needs plain strs.
            db["suggested_commands"] = [str(c) for c in list(command_list)]
            db["flag_type"] = int(flag_type)

    # Get the suggested commands
    @staticmethod
    def _get_suggested_commands(cache_path):
        """
        Get the list of suggested commands for the constrained selection
        """
        with KVStore(cache_path) as db:
            return db.get("suggested_commands", [])

    @staticmethod
    def _get_count(cache_path):
        with KVStore(cache_path) as db:
            return db.get("utterance_count", 0)  # Default to 0 if key doesn't exist

    @staticmethod
    def _print_db_contents(cache_path):
        with KVStore(cache_path) as db:
            print("All keys in database:", list(db.keys()))
            for key in db.keys():
                print(f"Key: {key}, Value: {db[key]}")

    @staticmethod
    def _store_utterance(cache_path, utterance, label):
        """
        Store utterance in existing or new database
        Returns: The utterance count used
        """
        with KVStore(cache_path) as db:
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

    # Function to read from database
    @staticmethod
    def _read_utterance(cache_path, utterance_id):
        """
        Read a specific utterance from the database
        """
        with KVStore(cache_path) as db:
            return db.get(utterance_id)['utterance']
    @staticmethod
    def resolve_fully_qualified_command_name(
        command_name: Optional[str], command_name_dict: dict[str, str]) -> Optional[str]:
        """Map a predicted label to a fully qualified command name, or None.

        Reserved labels (`wildcard`, `parameter_value`) name no command, so they
        must resolve to None rather than be looked up in `command_name_dict` —
        which would raise KeyError. None is what drives the parent-chain walk in
        the CME wildcard command.
        """
        if not command_name or is_non_routable(command_name):
            return None
        return command_name_dict[command_name]

    @staticmethod
    def escalation_signals_in(route_choice_list: list[str]) -> list[str]:
        """Return the escalation labels present in a prediction list."""
        return [
            route_choice for route_choice in route_choice_list
            if is_escalation(route_choice)
        ]

    @staticmethod
    def _formulate_ambiguous_command_error_message(
        route_choice_list: list[str], run_as_agent: bool) -> str:
        command_list = (
            "\n".join([
                f"{route_choice.split('/')[-1].lower()}"
                for route_choice in route_choice_list if not is_non_routable(route_choice)
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
