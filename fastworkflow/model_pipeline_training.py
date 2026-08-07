import os
from typing import Callable, ClassVar, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import torch 
# from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import json
import os
from torch.utils.data import random_split
import fastworkflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional, Tuple,Union
import pickle
from pathlib import Path
from collections import Counter

from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.train import heldout_evaluation
from fastworkflow.train.determinism import (
    ContextTrainingStatus,
    get_provenance_recorder,
)
from fastworkflow.train.selective_training import contexts_for_training
from fastworkflow.train import class_balance
from fastworkflow.utils.logging import logger
from fastworkflow.nlu_labels import (
    PARAMETER_VALUE_LABEL,
    PARAMETER_VALUE_PLACEHOLDERS,
    WILDCARD_LABEL,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transformers 5.x emits a tqdm "Loading weights" progress bar on every
# from_pretrained() call. fastWorkflow loads one intent-detection model per
# context, so these bars are noise when running/training a workflow. Suppress them.
try:
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
except Exception:  # noqa: BLE001 - older transformers may not expose this
    pass

dataset=None
label_encoder=LabelEncoder()


class TrainingDataError(ValueError):
    """Raised before model fitting when a label cannot be trained and evaluated."""


def split_training_data(
    dataset: list[tuple[str, int]],
    decode_label: Optional[Callable[[int], str]] = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a deterministic, class-aware train/evaluation split.

    Every label must contribute at least one row to both sets. This replaces the
    unstratified 25% split that could randomly put every row for a small class in the
    evaluation set, leaving the classifier unable to learn that command.

    ``decode_label`` turns the encoded label back into its command name for the
    error message. Without it the abort names labels the developer cannot map to
    commands without reverse-engineering the LabelEncoder, which is the state
    this argument exists to prevent.
    """
    label_counts = Counter(label for _, label in dataset)
    starved = sorted(
        label
        for label, count in label_counts.items()
        if count < heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL
    )
    if starved:
        named = [decode_label(label) for label in starved] if decode_label else starved
        raise TrainingDataError(
            f"Each intent label needs at least "
            f"{heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL} rows (one for training "
            f"and one for evaluation); labels with fewer rows: {named}"
        )

    label_count = len(label_counts)
    test_size = max((len(dataset) + 3) // 4, label_count)
    if len(dataset) - test_size < label_count:
        raise TrainingDataError(
            f"Cannot split {len(dataset)} rows across {label_count} labels while keeping "
            "at least one row per label in both training and evaluation sets."
        )

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        stratify=[label for _, label in dataset],
    )
    return list(train_data), list(test_data)


def save_label_encoder(filepath):
    global label_encoder
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(filepath):
    global label_encoder
    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)


def find_optimal_confidence_threshold(model, test_loader, device, min_threshold=0.5129, max_top3_usage=0.3, step_size=0.01, k_val=3):
    """
    Find optimal confidence threshold above the escalation threshold while limiting top@3 usage.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: torch device
        min_threshold: Minimum threshold (escalation threshold)
        max_top3_usage: Maximum allowed top@3 usage (default 0.3 or 30%)
        step_size: Step size for threshold search
    """
    # Get confidence statistics
    stats, confidences, predictions, labels, failed_cases = analyze_model_confidence(
        model, test_loader, device
    )
    
    # Set search range starting from escalation threshold
    start_threshold = min_threshold
    end_threshold = min(stats['successful']['max'], 0.95)
    
    best_metrics = None
    optimal_threshold = None
    best_score = 0
    
    # Store results for all thresholds
    thresholds = []
    f1_scores = []
    top3_usages = []
    combined_scores = []
    
    def calculate_score(f1, top3_usage):
        """
        Scoring function that:
        1. Prioritizes F1 score
        2. Heavily penalizes exceeding max_top3_usage
        """
        if top3_usage > max_top3_usage:
            return f1 * (1 - 2 * (top3_usage - max_top3_usage))  # Strong penalty for exceeding limit
        return f1
    
    # Test different correct thresholds 
    model.eval()
    with torch.no_grad():
        for threshold in tqdm(np.arange(start_threshold, end_threshold, step_size), 
                            desc="Finding optimal threshold"):
            true_labels = []
            predicted_labels = []
            top3_count = 0
            total = 0
            correct_top1 = 0
            correct_top3 = 0
            
            for encodings, labels, _ in test_loader:
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                top_probs, top_preds = torch.topk(probs, k=k_val, dim=1) #TODO remove the hardcode k value set it based on len of y
                max_confidences = top_probs[:, 0]
                
                batch_size = labels.size(0)
                total += batch_size
                
                for i in range(batch_size):
                    if max_confidences[i] >= threshold:
                        # Use top-1 prediction
                        pred = top_preds[i, 0]
                        if pred == labels[i]:
                            correct_top1 += 1
                    else:
                        # Use top-3 predictions
                        top3_count += 1
                        if labels[i] in top_preds[i]:
                            pred = labels[i]
                            correct_top3 += 1
                        else:
                            pred = top_preds[i, 0]
                    
                    true_labels.append(labels[i].cpu().item())
                    predicted_labels.append(pred.cpu().item())
            
            # Calculate metrics
            f1 = f1_score(true_labels, predicted_labels, average='weighted')
            top3_usage = top3_count / total
            top1_accuracy = correct_top1 / (total - top3_count) if (total - top3_count) > 0 else 0
            top3_accuracy = correct_top3 / top3_count if top3_count > 0 else 0
            
            # Calculate combined score
            combined_score = calculate_score(f1, top3_usage)
            
            thresholds.append(threshold)
            f1_scores.append(f1)
            top3_usages.append(top3_usage)
            combined_scores.append(combined_score)
            
            # Update best threshold if current score is better and top3 usage is within limit
            if combined_score > best_score and top3_usage <= max_top3_usage:
                best_score = combined_score
                optimal_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_score': f1,
                    'top3_usage': top3_usage,
                    'top1_accuracy': top1_accuracy,
                    'top3_accuracy': top3_accuracy,
                    'combined_score': combined_score
                }
    return optimal_threshold, best_metrics



def analyze_model_confidence(model, test_loader, device, model_name=""):
    model.eval()
    failed_cases = []
    failed_confidences = []
    successful_confidences = []
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for encodings, labels, _ in tqdm(test_loader, desc=f"Analyzing {model_name} confidence"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidence = torch.max(probs, dim=1).values

            # Store all results
            all_confidences.extend(confidence.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Analyze correct and incorrect predictions
            correct_mask = (predictions == labels)
            incorrect_mask = ~correct_mask

            for idx in torch.where(incorrect_mask)[0]:
                failed_confidences.append(confidence[idx].item())
                failed_cases.append({
                    'true_label': labels[idx].item(),
                    'predicted_label': predictions[idx].item(),
                    'confidence': confidence[idx].item()
                })

            successful_confidences.extend(confidence[correct_mask].cpu().numpy())

    stats = {
        'failed': {
            'min': np.min(failed_confidences) if failed_confidences else None,
            'max': np.max(failed_confidences) if failed_confidences else None,
            'mean': np.mean(failed_confidences) if failed_confidences else None,
            'median': np.median(failed_confidences) if failed_confidences else None
        },
        'successful': {
            'min': np.min(successful_confidences) if successful_confidences else None,
            'max': np.max(successful_confidences) if successful_confidences else None,
            'mean': np.mean(successful_confidences) if successful_confidences else None,
            'median': np.median(successful_confidences) if successful_confidences else None
        }
    }
    return stats, all_confidences, all_predictions, all_labels, failed_cases

def find_optimal_threshold(tiny_stats, test_loader, pipeline):
    # Generate threshold range based on confidence statistics
    min_threshold = tiny_stats['failed']['mean']
    max_threshold = tiny_stats['successful']['mean']
    if min_threshold and max_threshold:
        thresholds = np.linspace(min_threshold, max_threshold, 20)
    else:
        return (
            {            
                'threshold': -1,
                'f1': -1,
                'ndcg': -1,
                'distil_usage': -1
            }, [{            
                'threshold': -1,
                'f1': -1,
                'ndcg': -1,
                'distil_usage': -1
            }]
        )

    results = []
    for threshold in tqdm(thresholds, desc="Finding optimal threshold"):
        pipeline.confidence_threshold = threshold
        f1, ndcg, stats = pipeline.evaluate(test_loader)
        results.append({
            'threshold': threshold,
            'f1': f1,
            'ndcg': ndcg,
            'distil_usage': stats['distil_percentage']
        })
    
    # Find threshold with best balance of performance and efficiency
    alpha = 0.15
    best_result = max(results, key=lambda x: x['f1'] * x['ndcg'] * (1 - alpha * (x['distil_usage'] / 100)))

    return best_result, results



def get_route_layer_filepath_model(workflow_folderpath,model_name) -> str:
    command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        model_name
    )
class CommandRouter:
    _instances_cache: ClassVar[dict[str, "CommandRouter"]] = {}

    def __new__(cls, model_artifacts_folderpath: str):
        """Return a cached instance if we've already created a CommandRouter for *model_artifacts_folderpath*.
        The path is normalised (the '*' replacement) so logically identical paths map to the same key.
        This avoids re-reading JSON threshold files **and**, more importantly, re-building the underlying
        ModelPipeline with expensive model loading.
        """
        # Normalise the path in the same way __init__ will do so that the cache key matches.
        normalised_path = model_artifacts_folderpath.replace('*', GLOBAL_CONTEXT_FOLDER)
        cached = cls._instances_cache.get(normalised_path)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        cls._instances_cache[normalised_path] = instance
        return instance

    def __init__(self, model_artifacts_folderpath: str):
        # Avoid re-initialising if we are returning a cached instance.
        if getattr(self, "_initialised", False):
            return
        if '*' in model_artifacts_folderpath:
            model_artifacts_folderpath = model_artifacts_folderpath.replace('*', GLOBAL_CONTEXT_FOLDER)
            
        self.tiny_path = f"{model_artifacts_folderpath}/tinymodel.pth"
        self.large_path = f"{model_artifacts_folderpath}/largemodel.pth"
        threshold_path = f"{model_artifacts_folderpath}/threshold.json"
        tiny_ambiguous_threshold_path = f"{model_artifacts_folderpath}/tiny_ambiguous_threshold.json"
        large_ambiguous_threshold_path = f"{model_artifacts_folderpath}/large_ambiguous_threshold.json"
        self.label_encoder_path = f"{model_artifacts_folderpath}/label_encoder.pkl"
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            self.confidence_threshold = data['confidence_threshold']
            
        with open(tiny_ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.tiny_ambiguous_confidence_threshold = data['confidence_threshold']
        with open(large_ambiguous_threshold_path, 'r') as f:
            data = json.load(f)
            self.large_ambiguous_confidence_threshold = data['confidence_threshold']

        self.modelpipeline = ModelPipeline(
            tiny_model_path=self.tiny_path,
            distil_model_path=self.large_path,
            confidence_threshold=self.confidence_threshold
        )

        self._initialised = True

    def predict(self, command: str) -> list[str]:
        """
        if we are confident we will return a single label otherwise we will return a list
        """
        results = predict_single_sentence(self.modelpipeline, command, self.label_encoder_path)
        if (
            results['used_distil']
            and results['confidence'] > self.large_ambiguous_confidence_threshold
            or not results['used_distil']
            and results['confidence'] > self.tiny_ambiguous_confidence_threshold
        ):
            return [results['label']]
        else:
            return results['topk_labels']
            
        
class ModelPipeline:
    # ------------------------------------------------------------------
    # Singleton-like caching ------------------------------------------------
    # ------------------------------------------------------------------
    _instances_cache: ClassVar[dict[tuple[str, str, float, str], "ModelPipeline"]] = {}

    def __new__(cls,
                tiny_model_path: str,
                distil_model_path: str,
                confidence_threshold: float = 0.65,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        key = (tiny_model_path, distil_model_path, confidence_threshold, device)
        existing = cls._instances_cache.get(key)
        if existing is not None:
            return existing
        instance = super().__new__(cls)
        cls._instances_cache[key] = instance
        return instance

    def __init__(
        self,
        tiny_model_path: str,
        distil_model_path: str,
        confidence_threshold: float = 0.65,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # __init__ will be called every time __new__ returns an instance – including
        # when we served a cached instance.  Guard against double-initialisation.
        if getattr(self, "_initialised", False):
            return

        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load TinyBERT
        self.tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_model_path)
        self.tiny_model = AutoModelForSequenceClassification.from_pretrained(
            tiny_model_path
        ).to(device)

        # Load DistilBERT
        self.distil_tokenizer = AutoTokenizer.from_pretrained(distil_model_path)
        self.distil_model = AutoModelForSequenceClassification.from_pretrained(
            distil_model_path
        ).to(device)

        # Set models to evaluation mode
        self.tiny_model.eval()
        self.distil_model.eval()

        # Determine top-k value once for this pipeline (≤3, but never > num_labels)
        num_labels = self.tiny_model.config.num_labels
        self.k_val = min(3, num_labels)

        self._initialised = True

    def calculate_ndcg_at_k(self, batch_top_k_preds: List[List[int]], batch_top_k_scores: List[List[float]], true_labels: List[int], k: int = 3) -> float:
        batch_ndcg = 0.0
        
        for pred_top_k, conf_top_k, true_label in zip(batch_top_k_preds, batch_top_k_scores, true_labels):
            # Calculate relevance for top k predictions (1 if correct, 0 if incorrect)
            relevance = [1 if pred == true_label else 0 for pred in pred_top_k]
            
            # Calculate DCG
            dcg = 0.0
            for i in range(min(k, len(pred_top_k))):
                if relevance[i] == 1:
                    dcg += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32))
            
            # Calculate IDCG (always 1/log2(2) since we only have one relevant document)
            idcg = 1 / torch.log2(torch.tensor(2, dtype=torch.float32))
            
            # Calculate NDCG for this sample
            sample_ndcg = dcg / idcg if idcg != 0 else 0
            batch_ndcg += sample_ndcg
            
        # Return average NDCG for the batch
        return batch_ndcg / len(true_labels)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        k_val: int | None = None
    ) -> Dict:
        all_predictions = []
        all_confidences = []
        all_top_k_predictions = []  # Store top k predictions for each sample
        all_top_k_scores = []      # Store top k confidence scores for each sample
        all_logits = []
        all_used_distil = []
        k = k_val or self.k_val

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Predict with TinyBERT
            tiny_inputs = self.tiny_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            tiny_outputs = self.tiny_model(**tiny_inputs)
            tiny_logits = tiny_outputs.logits
            tiny_probs = torch.softmax(tiny_logits, dim=1)
            tiny_confidence, tiny_predictions = torch.max(tiny_probs, dim=1)

            # Get top k predictions and scores for TinyBERT
            tiny_top_k_scores, tiny_top_k_preds = torch.topk(tiny_probs, k=k, dim=1)

            # Identify low-confidence samples
            need_distil = tiny_confidence < self.confidence_threshold

            # Initialize with TinyBERT results
            batch_predictions = tiny_predictions.clone()
            batch_confidences = tiny_confidence.clone()
            batch_logits = tiny_logits.clone()
            batch_used_distil = need_distil.clone()
            batch_top_k_preds = tiny_top_k_preds.clone()
            batch_top_k_scores = tiny_top_k_scores.clone()

            # Predict with DistilBERT for low-confidence samples
            if need_distil.any():
                distil_texts = [text for text, flag in zip(batch_texts, need_distil) if flag]

                distil_inputs = self.distil_tokenizer(
                    distil_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                distil_outputs = self.distil_model(**distil_inputs)
                distil_logits = distil_outputs.logits
                distil_probs = torch.softmax(distil_logits, dim=1)
                distil_confidence, distil_predictions = torch.max(distil_probs, dim=1)

                # Get top k predictions and scores for DistilBERT
                distil_top_k_scores, distil_top_k_preds = torch.topk(distil_probs, k=k, dim=1)

                # Update results for low-confidence samples
                distil_idx = 0
                for j in range(len(batch_predictions)):
                    if need_distil[j]:
                        batch_predictions[j] = distil_predictions[distil_idx]
                        batch_confidences[j] = distil_confidence[distil_idx]
                        batch_logits[j] = distil_logits[distil_idx]
                        batch_top_k_preds[j] = distil_top_k_preds[distil_idx]
                        batch_top_k_scores[j] = distil_top_k_scores[distil_idx]
                        distil_idx += 1

            # Store results
            all_predictions.extend(batch_predictions.cpu().tolist())
            all_confidences.extend(batch_confidences.cpu().tolist())
            all_logits.append(batch_logits.cpu())
            all_used_distil.extend(batch_used_distil.cpu().tolist())
            all_top_k_predictions.extend(batch_top_k_preds.cpu().tolist())
            all_top_k_scores.extend(batch_top_k_scores.cpu().tolist())

        return {
            "predictions": all_predictions,
            "confidences": all_confidences,
            "logits": torch.cat(all_logits, dim=0).to(self.device),
            "used_distil": all_used_distil,
            "top_k_predictions": all_top_k_predictions,
            "top_k_scores": all_top_k_scores
        }

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, Dict]:
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_logits = []
        all_top_k_predictions = []
        all_top_k_scores = []
        total_used_distil = 0
        total_samples = 0
        total_ndcg = 0.0
        num_batches = 0

        for batch in tqdm(test_loader, desc="Evaluating"):
            # Batch now comes with raw *texts* so we can skip expensive decode→encode
            encodings, labels, texts = batch

            results = self.predict_batch(texts)

            all_predictions.extend(results['predictions'])
            all_labels.extend(labels.cpu().tolist())
            all_confidences.extend(results['confidences'])
            all_logits.append(results['logits'])
            all_top_k_predictions.extend(results['top_k_predictions'])
            all_top_k_scores.extend(results['top_k_scores'])
            total_used_distil += sum(results['used_distil'])
            total_samples += len(labels)

            # Calculate NDCG@3 for current batch
            batch_ndcg = self.calculate_ndcg_at_k(
                results['top_k_predictions'],
                results['top_k_scores'],
                labels.cpu().tolist()
            )
            total_ndcg += batch_ndcg
            num_batches += 1

        # Calculate F1 Score
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Calculate average NDCG@3 across all batches
        avg_ndcg = total_ndcg / num_batches

        # Model usage stats
        stats = {
            "total_samples": total_samples,
            "distil_usage": total_used_distil,
            "distil_percentage": (total_used_distil / total_samples) * 100,
            "tiny_percentage": ((total_samples - total_used_distil) / total_samples) * 100
        }

        return f1, avg_ndcg, stats


#for single utterance prediction
def predict_single_sentence(
    pipeline: ModelPipeline,
    text: str,
    path: str,
    #label_encoder: LabelEncoder
) -> Dict[str, Union[int, str, float, bool]]:

    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if not text.strip():
        raise ValueError("Input text cannot be empty")


    # `path` is expected to be the absolute path to the label_encoder artefact
    global label_encoder
    load_label_encoder(path)
    k_val=len(label_encoder.classes_)
    k_val = 3 if k_val>2 else 2
    # Make prediction using the pipeline's batch prediction method
    results = pipeline.predict_batch([text],k_val=k_val)
    # Get the numeric prediction
    numeric_prediction = results["predictions"][0]

    label_names = label_encoder.inverse_transform(results['top_k_predictions'][0])

    # Convert numeric prediction back to original label name
    label_name = label_encoder.inverse_transform([numeric_prediction])[0]

    return {
        "prediction": numeric_prediction,
        "label": label_name,
        "confidence": results["confidences"][0],
        "used_distil": results["used_distil"][0],
        "topk_labels":label_names
    }

# ---------------------------------------------------------------------
# Helper utilities for artefact locations (per-context)
# ---------------------------------------------------------------------
GLOBAL_CONTEXT_FOLDER = "global"

# Workflow folderpath (resolved) -> artifact version currently being written.
# Installed by the trainer for the duration of a run; empty at every other time.
_active_artifact_version: dict[str, str] = {}


def set_active_artifact_version(workflow_folderpath: str, version_id: Optional[str]) -> None:
    """Route `get_artifact_path` writes for *workflow_folderpath* into *version_id*.

    Pass ``None`` to clear. The trainer installs this for the duration of a run so a
    retrain assembles a NEW version instead of overwriting the live one in place (R4 /
    finding F5). Keeping it here rather than threading a parameter through means the six
    existing `get_artifact_path` call sites need no changes.
    """
    key = str(Path(workflow_folderpath).resolve())
    if version_id is None:
        _active_artifact_version.pop(key, None)
    else:
        _active_artifact_version[key] = version_id


def get_artifact_path(workflow_folderpath: str, context_name: str, filename: str) -> str:
    """Return the absolute path for a model/artefact for *context_name*.

    Under versioning the file lives at:
        <workflow>/___command_info/versions/<version>/<context_name>/<filename>
    which every legacy reader still reaches through the per-context compatibility entry
    at ``<workflow>/___command_info/<context_name>/``.

    When no version is active and none has been published (a workflow that has never been
    trained under R4) this falls back to the historical unversioned path, so a partially
    migrated tree keeps working.

    The directory is created if it does not yet exist. The special context name "*" is
    mapped to GLOBAL_CONTEXT_FOLDER.
    """
    from fastworkflow import RoutingRegistry
    from fastworkflow.train import artifact_versioning as av

    ctx_folder = context_name if context_name != "*" else GLOBAL_CONTEXT_FOLDER
    crd = RoutingRegistry.get_definition(workflow_folderpath)
    # Preserve the ___command_info mkdir side effect that callers have always relied on.
    info_root = Path(crd.command_directory.get_commandinfo_folderpath(workflow_folderpath))

    key = str(Path(workflow_folderpath).resolve())
    version_id = _active_artifact_version.get(key) or av.resolve_current_version(
        workflow_folderpath
    )
    if version_id is None:
        base_dir = info_root / ctx_folder
        base_dir.mkdir(parents=True, exist_ok=True)
        return str(base_dir / filename)

    return str(
        av.context_artifact_dir(workflow_folderpath, version_id, context_name) / filename
    )


def is_workflow_trained(workflow_folderpath: str) -> Tuple[bool, List[str]]:
    """Return ``(is_trained, missing_context_folders)`` for *workflow_folderpath*.

    A workflow is considered trained when every routing context declared in its
    ``___command_info/routing_definition.json`` has a ``threshold.json`` model
    artifact under ``___command_info/<context>/`` (the special context ``"*"``
    maps to ``GLOBAL_CONTEXT_FOLDER``).

    This is a lightweight, filesystem-only check used to fail fast *before*
    starting a chat session on an untrained workflow. Without it, intent
    detection crashes on the first command when ``CommandRouter`` cannot open
    the missing ``threshold.json`` (e.g. ``___command_info/global/threshold.json``).
    """
    command_info_root = os.path.join(workflow_folderpath, "___command_info")
    routing_def_path = os.path.join(command_info_root, "routing_definition.json")
    if not os.path.isfile(routing_def_path):
        return False, ["<workflow not built>"]

    try:
        with open(routing_def_path, "r") as f:
            routing_definition = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, ["<unreadable routing_definition.json>"]

    contexts = routing_definition.get("contexts") or {}
    if not contexts:
        return False, ["<no contexts in routing_definition.json>"]

    # Only check the contexts that training actually produces artifacts for. This
    # mirrors the context selection in train(): the internal
    # command_metadata_extraction contexts (e.g. IntentDetection, ErrorCorrection)
    # are NOT trained per app workflow - their models live in the internal CME
    # workflow - and the global wildcard context "*" is always included.
    try:
        internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        internal_contexts = set(
            fastworkflow.CommandContextModel.load(internal_wf_path)._command_contexts.keys()
        )
    except Exception:
        internal_contexts = set()

    contexts_to_check = (set(contexts.keys()) - internal_contexts) | {"*"}

    missing: List[str] = []
    for context_name in contexts_to_check:
        context_folder = GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        threshold_path = os.path.join(command_info_root, context_folder, "threshold.json")
        if not os.path.isfile(threshold_path):
            missing.append(context_folder)

    return (not missing), sorted(missing)

# ---------------------------------------------------------------------

# After training loop is complete
def save_model(model,tokenizer, save_path):
    # Save the model
    model.save_pretrained(save_path)
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def evaluate_model(model, data_loader, device, k_val):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_ndcg = 0
    num_batches = 0

    with torch.no_grad():
        for encodings, labels, _ in tqdm(data_loader, desc="Evaluating"):
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Calculate test loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            ndcg = calculate_ndcg_at_k(outputs.logits, labels, k=k_val) # TODO this should be min of 3
            total_ndcg += ndcg
            num_batches += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches  # Average test loss
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_ndcg = total_ndcg / num_batches

    return f1, avg_ndcg, avg_loss

def calculate_ndcg_at_k(logits, true_labels, k=3):
    probs = torch.softmax(logits, dim=1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)
    batch_size = logits.shape[0]
    relevance = torch.zeros_like(top_k_probs)
    for i in range(batch_size):
        relevance[i] = (top_k_indices[i] == true_labels[i]).float()
    discounts = 1 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32)).to(device)
    dcg = torch.sum(relevance * discounts, dim=1)
    idcg = discounts[0]
    ndcg = dcg / idcg
    return ndcg.mean().item()

def _requires_utterances(cmd_dir: object, cmd_name: str) -> bool:
    """Return True if *cmd_name* defines `Signature.Input` and therefore
    needs utterance-based intent classification.

    This is determined after lazy-hydrating the command so the metadata is
    accurate.  A command with *no* `Signature.Input` is expected to be
    dispatched exclusively via `perform_action` and should be excluded from
    model training.
    """
    # Ensure metadata is fully populated before inspection
    cmd_dir.ensure_command_hydrated(cmd_name)
    metadata = cmd_dir.get_command_metadata(cmd_name)
    return bool(metadata.input_for_param_extraction_class)


def _get_utterances(workflow: fastworkflow.Workflow,
                    workflow_folderpath: str, 
                    cmd_dir: object, cmd: str) -> list[str]:
    """Safely retrieve utterances for *cmd* when required.

    If the command does not have `Signature.Input`, we return an empty list
    (no utterances needed) **without** calling `get_utterance_metadata`.
    """
    if not _requires_utterances(cmd_dir, cmd):
        return []

    um = cmd_dir.get_utterance_metadata(cmd)
    if not um:
        raise KeyError(
            f"Could not find utterance metadata for command '{cmd}'. "
            "It might be missing from the _commands directory."
        )

    func = um.get_generated_utterances_func(workflow_folderpath)
    return func(workflow, cmd) if func else []


def _get_cached_command_utterances(
    workflow: fastworkflow.Workflow,
    workflow_folderpath: str,
    cmd_dir: object,
    cmd: str,
    command_cache: dict[str, list[str]],
) -> list[str]:
    """Generate once per fully-qualified command, then reuse across contexts.

    A generated command function receives `(workflow, command_name)` and no context.
    Context inheritance changes where its rows are used, not how its rows are generated.
    Keying this cache by context used to redraw the same command for every context and
    made its last generation overwrite all earlier provenance.
    """
    if cmd not in command_cache:
        command_cache[cmd] = _get_utterances(
            workflow, workflow_folderpath, cmd_dir, cmd
        )
    return command_cache[cmd]


def _record_context_training(
    context_name: str,
    command_name: str,
    cmd_dir: object,
    utterances: list[str],
) -> None:
    """Record context row use or the trainer's exact skip reason."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    if not _requires_utterances(cmd_dir, command_name):
        recorder.record_context(
            context_name=context_name,
            command_name=command_name,
            status=ContextTrainingStatus.SKIPPED_NO_INPUT,
            reason="command has no Signature.Input",
        )
        return

    if not utterances:
        recorder.record_context(
            context_name=context_name,
            command_name=command_name,
            status=ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
            reason="utterance generator returned no rows",
        )
        return

    generation = recorder.records.get(command_name)
    fell_back = bool(generation and generation.fell_back)
    recorder.record_context(
        context_name=context_name,
        command_name=command_name,
        status=(
            ContextTrainingStatus.INCLUDED_FALLBACK
            if fell_back
            else ContextTrainingStatus.INCLUDED
        ),
        row_count=len(utterances),
        reason=generation.fallback_reason if fell_back and generation else None,
    )


def _record_wildcard_context_training(
    context_name: str,
    escalation_rows: Optional[list[str]],
    *,
    own_row_count: int,
    raw_candidate_count: int,
    deduplicated_candidate_count: int,
    always_include_rows: list[str],
    selected_budget: Optional[int],
    coverage_floor: int,
) -> None:
    """Record escalation-class rows and every denominator used to select them."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    included = escalation_rows is not None
    recorder.record_context(
        context_name=context_name,
        command_name=WILDCARD_LABEL,
        status=(
            ContextTrainingStatus.INCLUDED
            if included
            else ContextTrainingStatus.SKIPPED_NO_UTTERANCES
        ),
        row_count=len(escalation_rows or []),
        reason=(
            "reserved escalation class"
            if included
            else "context has no non-local ancestor utterances"
        ),
        own_row_count=own_row_count,
        raw_candidate_count=raw_candidate_count,
        deduplicated_candidate_count=deduplicated_candidate_count,
        always_include_count=len(always_include_rows),
        selected_budget=selected_budget,
        coverage_floor=coverage_floor,
        coverage_floor_applied=(
            coverage_floor > own_row_count if included else False
        ),
    )


def _record_parameter_value_context_training(
    context_name: str,
    parameter_value_rows: list[str],
    own_row_count: int,
) -> None:
    """Record bare-value reserved rows without wildcard-only budget fields."""
    recorder = get_provenance_recorder()
    if recorder is None:
        return

    recorder.record_context(
        context_name=context_name,
        command_name=PARAMETER_VALUE_LABEL,
        status=(
            ContextTrainingStatus.INCLUDED
            if parameter_value_rows
            else ContextTrainingStatus.SKIPPED_NO_UTTERANCES
        ),
        row_count=len(parameter_value_rows),
        reason="reserved bare-value class",
        own_row_count=own_row_count,
        raw_candidate_count=len(PARAMETER_VALUE_PLACEHOLDERS),
        deduplicated_candidate_count=len(parameter_value_rows),
        always_include_count=0,
    )


def cache_ancestor_utterances(
    context_name: str, 
    crd: RoutingDefinition, 
    workflow: fastworkflow.Workflow,
    cache: dict,
    command_cache: Optional[dict[str, list[str]]] = None,
) -> list[str]:
    """
    Collects utterances for the 'wildcard' context.
    
    This includes the base 'wildcard' command's utterances, plus all utterances
    from every command (except 'wildcard') belonging to any of the current 
    context's ancestors.
    """
    workflow_folderpath = workflow.folderpath
    cmd_dir = crd.command_directory
    if command_cache is None:
        command_cache = {}

    ancestor_utterances = set()
    ancestor_contexts = crd.context_model.get_ancestor_contexts(context_name)
    for ancestor_ctx in ancestor_contexts:
        if ancestor_ctx in cache:
            for cmd_name in cache[ancestor_ctx]:
                ancestor_utterances |= set(cache[ancestor_ctx][cmd_name])    
            continue
        cache[ancestor_ctx] = {}
        ancestor_commands = crd.context_model.commands(ancestor_ctx)
        for cmd in ancestor_commands:
            if cmd.split('/')[-1] == 'wildcard':
                continue            
            utterances = _get_cached_command_utterances(
                workflow, workflow_folderpath, cmd_dir, cmd, command_cache
            )
            cache[ancestor_ctx][cmd] = utterances   
            ancestor_utterances |= set(utterances)    

    return ancestor_utterances


def _score_heldout_context(
    report: heldout_evaluation.HeldoutReport,
    heldout_records: list[heldout_evaluation.LabeledUtterance],
    benchmark_cases: list[heldout_evaluation.BenchmarkCase],
    predict_labels: Optional[heldout_evaluation.PredictFn],
) -> None:
    """Score independent persona-holdout and fixed-benchmark populations."""
    if heldout_records:
        if predict_labels is None:
            raise ValueError("persona holdout scoring requires a predictor")
        # Escalation-class rows are appended to the training tuples before the
        # persona split is drawn, so they arrive here carrying real persona
        # attribution and used to be scored as routes. Split them onto their own
        # axis (D2) rather than dropping them: in a context with ancestors they
        # are most of the population, and losing them would trade a wrong number
        # for a blind spot.
        routable, escalation_class = heldout_evaluation.partition_by_routability(
            heldout_records)
        if routable:
            report.routing = heldout_evaluation.score_routing(
                routable, predict_labels)
        if escalation_class:
            report.holdout_escalation = heldout_evaluation.score_escalation(
                escalation_class, predict_labels)
        if not routable:
            report.notes.append(
                f"Persona holdout for this context is entirely escalation-class "
                f"({len(escalation_class)} rows); no held-out routing was scored."
            )
    else:
        report.notes.append(
            "Persona holdout unavailable; persona-held-out routing was not scored. "
            "Fixed benchmark routing and escalation remain independently measurable."
        )

    # `kind` is required. Omitting it raised a TypeError that the guard
    # below turned into a note on the report, so escalation silently
    # never scored while routing kept working -- the failure looked like
    # "this workflow has no escalation cases" (bd fix-588).
    escalation_cases = heldout_evaluation.benchmark_cases_for_context(
        benchmark_cases, report.context, "escalation")
    routing_cases = heldout_evaluation.benchmark_cases_for_context(
        benchmark_cases, report.context, "routing")
    if (escalation_cases or routing_cases) and predict_labels is None:
        raise ValueError("benchmark scoring requires a predictor")

    if escalation_cases:
        report.escalation = heldout_evaluation.score_escalation(
            escalation_cases, predict_labels)
    if routing_cases:
        report.benchmark_routing = heldout_evaluation.score_routing(
            routing_cases, predict_labels)
        print(
            f"Benchmark routing [{report.context}]: "
            f"{report.benchmark_routing.top1_correct}"
            f"/{report.benchmark_routing.total} top-1"
        )


def preflight_benchmark(
    workflow_folderpath: str,
    crd: Optional[RoutingDefinition] = None,
) -> list[heldout_evaluation.BenchmarkCase]:
    """Load and validate the workflow's benchmark file, raising on a seed leak.

    Returns the loaded cases (empty when the workflow ships no benchmark) so a caller
    that is about to score can reuse them.

    Runs off the command directory and routing definition alone — no workflow instance,
    no model, no LLM — which is what lets `train.__main__.train_workflow` call it before
    it spends anything. `train` calls it again as a second line of defence for direct
    callers that bypass the orchestrator.

    ------------------------------------------------------------------
    R1b: a benchmark that shares phrasings with the training seeds measures
    memorisation, so it is checked BEFORE any training happens. Failing fast is the
    point: discovering the leak after the run would mean every number it produced has
    to be thrown away, and the run costs LLM calls and GPU minutes.
    ------------------------------------------------------------------
    """
    benchmark_path = heldout_evaluation.default_benchmark_path(workflow_folderpath)
    if not os.path.isfile(benchmark_path):
        return []

    if crd is None:
        crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    cmd_dir = crd.command_directory

    benchmark_cases = heldout_evaluation.load_benchmark_file(benchmark_path)
    seed_utterances_by_command: dict[str, list[str]] = {}
    for command_key in cmd_dir.get_utterance_keys():
        if metadata := cmd_dir.get_utterance_metadata(command_key):
            seed_utterances_by_command[command_key] = list(metadata.plain_utterances)
    # Deliberately NOT caught: an unnoticed leak silently turns the whole evaluation
    # into a memorisation score, which is the failure R1 exists to remove.
    heldout_evaluation.assert_benchmark_disjoint_from_seeds(
        benchmark_cases, seed_utterances_by_command)
    # A close-but-not-equal match is a judgement call, so it is reported, not enforced.
    for warning in heldout_evaluation.find_near_duplicate_benchmark_cases(
        benchmark_cases, seed_utterances_by_command
    ):
        logger.warning(warning)

    # A benchmark case can also be defective in a way that has nothing to do with
    # leakage: a routing case naming a label the context does not have can never
    # pass, and an "escalation" case whose command is not actually absent here and
    # present in an ancestor is not testing escalation at all. Either one drags the
    # reported score down for a reason that is not the model's fault, so both are
    # reported before the run rather than discovered as a mysteriously low number.
    core_command_names = set(cmd_dir.core_command_names)
    context_label_space = {
        context_name: set(command_list) | core_command_names
        for context_name, command_list in crd.contexts.items()
    }
    ancestor_map = {
        context_name: list(crd.context_model.get_ancestor_contexts(context_name))
        for context_name in crd.contexts
    }
    for problem in (
        heldout_evaluation.validate_routing_cases(
            benchmark_cases, context_label_space)
        + heldout_evaluation.validate_escalation_cases(
            benchmark_cases, context_label_space, ancestor_map)
    ):
        logger.warning(f"Benchmark defect: {problem}")

    print(f"Benchmark: {len(benchmark_cases)} case(s) from {benchmark_path}")
    return benchmark_cases


def train(workflow: fastworkflow.Workflow,
          contexts_to_train: Optional[set[str]] = None):
    """Train intent-classification models **per command context**.

    ``contexts_to_train`` is R5's selective-retraining hook. ``None`` trains every
    context. Automatic planning passes a set when unchanged contexts can safely be
    carried forward. A set restricts the run
    to those contexts, and the CALLER then becomes responsible for carrying every other
    context's artifacts forward into the version being assembled
    (``train.selective_training.carry_forward_contexts``). Skipping a context here and
    forgetting that is not a slow workflow, it is an un-trained one: `publish_version`
    removes the compatibility entry of any context its version does not contain.
    """

    import time

    from fastworkflow.train.determinism import get_training_seed, seed_everything

    # Seed before anything samples: utterance generation picks personas, torch initialises
    # classifier heads, and the train/test split shuffles. Seeding is necessary but NOT
    # sufficient for reproducibility -- the iteration order of the sets feeding the split
    # matters just as much, which is why the loops below sort (fix-9mo).
    seed = seed_everything(get_training_seed())
    print(f"Training seed: {seed}")

    heldout_reports: list[heldout_evaluation.HeldoutReport] = []

    workflow_folderpath = workflow.folderpath
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath, load_cached=False)
    cmd_dir = crd.command_directory

    # Second line of defence. `train.__main__.train_workflow` runs the same preflight
    # before it spends anything on parameter-example generation; this call covers direct
    # callers of `train()` and re-reads the file in case it changed since.
    benchmark_cases = preflight_benchmark(workflow_folderpath, crd)

    context_utterance_cache = {}
    # Generation identity is the fully-qualified command. Context identity belongs to
    # the labelled-row use recorded below, not to the LLM draw or utterance cache key.
    command_utterance_cache: dict[str, list[str]] = {}

    core_cmds = set(cmd_dir.core_command_names)

    # Get contexts specific to this workflow (not from command_metadata_extraction)
    # The set itself is computed by `selective_training.contexts_for_training` so that the
    # R5 planner and this loop cannot disagree about which contexts exist. A context the
    # planner never considered would be neither retrained nor carried forward, and that
    # presents as part of a workflow silently becoming untrained.
    context_set_for_training = contexts_for_training(workflow_folderpath)

    if contexts_to_train is not None:
        requested = set(contexts_to_train)
        skipped = sorted(context_set_for_training - requested)
        context_set_for_training = context_set_for_training & requested
        if skipped:
            print(f"Selective training: skipping {len(skipped)} context(s) whose "
                  f"artifacts the caller will carry forward: {', '.join(skipped)}")

    wildcard_utterances = set(_get_utterances(
        workflow, workflow.folderpath, crd.command_directory, 'wildcard'))

    # Only iterate through contexts defined in this specific workflow.
    # sorted(): context_set_for_training is a set, and a context visited first as an
    # ANCESTOR gets its cache populated from context_model.commands(), which excludes the
    # core commands. Iterating in hash order therefore decided, per interpreter start,
    # which contexts lost their core-command labels (AR5). Sorting makes the order stable;
    # the cache-fill below makes the outcome order-independent as well.
    for ctx_name in sorted(context_set_for_training):
        ctx_cmd_list = crd.contexts[ctx_name]
        print(f"\n=== Training model for workflow_folderpath: {workflow_folderpath.split('/')[-1]} and context: {ctx_name} ===\n")
        
        # Build the label set for this context
        train_cmds: set[str] = set(ctx_cmd_list) | core_cmds
        if not train_cmds:
            print(f"Skipping context {ctx_name} - no commands to train")
            continue  # nothing to train

        context_utterances = set()
        utterance_command_tuples: list[tuple[str, str]] = []
        # One loop over every label this context trains, using the cache when it already
        # holds the command and GENERATING when it does not. The previous shape had two
        # branches: a cached branch that silently skipped commands absent from the cache,
        # and a generate-everything branch that only ran when the cached branch produced
        # nothing at all. A context previously visited as an ancestor has a cache holding
        # only context_model.commands(), so the cached branch dropped every core command
        # for it -- making those commands unroutable in that context (AR5 / fix-9mo).
        map_cmd_2_uttlist = context_utterance_cache.setdefault(ctx_name, {})
        for cmd_name in sorted(train_cmds):
            # Split form matches cache_ancestor_utterances, so both paths agree on what
            # the reserved label is regardless of qualification.
            if cmd_name.split('/')[-1] == WILDCARD_LABEL:
                continue
            if cmd_name in map_cmd_2_uttlist:
                print(f"Getting cached utterances for context: {ctx_name}, command: {cmd_name}\n")
            else:
                print(f"Generating utterances for context: {ctx_name}, command: {cmd_name} ...\n")
                map_cmd_2_uttlist[cmd_name] = _get_cached_command_utterances(
                    workflow,
                    workflow_folderpath,
                    cmd_dir,
                    cmd_name,
                    command_utterance_cache,
                )
            cmd_utterances = map_cmd_2_uttlist[cmd_name]
            _record_context_training(
                ctx_name, cmd_name, cmd_dir, cmd_utterances
            )
            utterance_command_tuples.extend(
                list(zip(cmd_utterances, [cmd_name] * len(cmd_utterances)))
            )
            context_utterances |= set(cmd_utterances)

        if not context_utterances:
            print(f"Skipping context {ctx_name} - no utterances available")
            continue  # skip empty

        ancestor_utterances = cache_ancestor_utterances(
            ctx_name,
            crd,
            workflow,
            context_utterance_cache,
            command_utterance_cache,
        )
        net_ancestor_utterances = ancestor_utterances - context_utterances
        own_rows = len(utterance_command_tuples)
        grouped_ancestor_rows = class_balance.group_ancestor_utterances(
            crd.context_model.get_ancestor_contexts(ctx_name),
            context_utterance_cache,
            skip_labels=(WILDCARD_LABEL,),
        )
        raw_candidate_count, deduplicated_candidate_count = (
            class_balance.reserved_candidate_counts(
                grouped_ancestor_rows,
                exclude=context_utterances,
            )
        )
        coverage_floor = class_balance.coverage_floor_of(grouped_ancestor_rows)

        # WILDCARD_LABEL is the ESCALATION signal: "an ancestor context can serve this".
        # It is emitted only where that can be true. In a context with no ancestors the
        # class would collapse to the single humanised command name, and a one-row class
        # cannot satisfy the class-aware split below: it needs one training row and one
        # evaluation row. Escalation is meaningless
        # there anyway: the runtime parent walk terminates immediately at the response
        # generation root, so the turn can only ever reach you_misunderstood, which an
        # unconfident classifier already reaches.
        escalation_rows: Optional[list[str]] = None
        always_include_rows: list[str] = []
        budget: Optional[int] = None
        if net_ancestor_utterances:
            # R7.2: bound the escalation class against this context's own row count, so
            # training time stays linear in workflow size, but never below one row per
            # ancestor command -- an ancestor that contributes no row cannot be escalated
            # to at all. Round-robin over (ancestor context, command) is what stops one
            # verbose ancestor from spending the budget the others need.
            #
            # `exclude` carries the `net_` semantics the flat set expression used to carry:
            # an utterance that means something HERE must not also train the "ask my
            # parent" class, or the same string would be trained under two labels.
            # `own_rows` is counted BEFORE the escalation rows are appended, which is what
            # makes the ratio mean "multiplier on this context's own training cost".
            # 1.0 is the fixed cost invariant: escalation may add at most as many rows
            # as the context's real commands, so reserved rows can at most double cost.
            # It is not an accuracy-tuned value; R7.3 weighting measured null and is
            # intentionally not shipped.
            budget = class_balance.reserved_class_budget(
                own_rows,
                coverage_floor,
                ratio=1.0,
            )
            always_include_rows = sorted(
                wildcard_utterances - context_utterances
            )
            escalation_rows = sorted(
                class_balance.select_reserved_rows(
                    grouped_ancestor_rows,
                    budget,
                    always_include=always_include_rows,
                    exclude=context_utterances,
                )
            )
            utterance_command_tuples.extend(
                list(zip(escalation_rows, [WILDCARD_LABEL] * len(escalation_rows)))
            )
        _record_wildcard_context_training(
            ctx_name,
            escalation_rows,
            own_row_count=own_rows,
            raw_candidate_count=raw_candidate_count,
            deduplicated_candidate_count=deduplicated_candidate_count,
            always_include_rows=always_include_rows,
            selected_budget=budget,
            coverage_floor=coverage_floor,
        )

        # PARAMETER_VALUE_LABEL is the PARAMETER_EXTRACTION stage's bare-value catcher and
        # is emitted in EVERY context: a user can type a bare value anywhere. These are the
        # seven literals that used to be trained into the wildcard class, which is what
        # taught the escalation classifier that "france" means "escalate to my parent".
        parameter_value_rows = sorted(set(PARAMETER_VALUE_PLACEHOLDERS) - context_utterances)
        if parameter_value_rows:
            utterance_command_tuples.extend(
                list(zip(parameter_value_rows,
                         [PARAMETER_VALUE_LABEL] * len(parameter_value_rows)))
            )
        _record_parameter_value_context_training(
            ctx_name, parameter_value_rows, own_rows
        )


        # ------------------------------------------------------------------
        # R1a: reserve WHOLE PERSONAS before training, for a real generalisation
        # measure. The class-aware split below still divides rows from the same
        # generated corpus, so utterances written by the same persona, from the same
        # seed utterance, can land on both sides of it. Scoring that is close to scoring
        # memorisation, which is why the F1 it produces has never been a safe basis
        # for "did this change help?". It is still computed and still used to
        # calibrate the ambiguity thresholds -- it is only its use as a QUALITY
        # metric that is unsound -- so it is reported under a name that says what it is.
        # ------------------------------------------------------------------
        heldout_records: list[heldout_evaluation.LabeledUtterance] = []
        heldout_personas: list[str] = []
        split_notes: list[str] = []
        if (recorder := get_provenance_recorder()) is not None:
            # utterance text -> persona id, across every command generated this run.
            # Anything absent is a hand-written seed utterance, which split_by_persona
            # always keeps in training: a developer's declared input is not
            # generalisation data.
            persona_by_utterance: dict[str, str] = {}
            for provenance in recorder.records.values():
                persona_by_utterance.update(provenance.utterance_personas)

            labeled = [
                heldout_evaluation.LabeledUtterance(
                    utterance=utterance,
                    label=label,
                    persona=persona_by_utterance.get(
                        utterance, heldout_evaluation.SEED_PERSONA_ID),
                )
                for utterance, label in utterance_command_tuples
            ]
            split = heldout_evaluation.split_by_persona(labeled, seed=seed)
            if split.heldout:
                # Held-out personas are removed from training entirely. Without this
                # the score below would be measuring the training set.
                utterance_command_tuples = [
                    (record.utterance, record.label) for record in split.train
                ]
                heldout_records = list(split.heldout)
                heldout_personas = list(split.heldout_personas)
                split_notes = list(split.notes)
                print(
                    f"Held out {len(heldout_records)} utterances from "
                    f"{len(heldout_personas)} personas for evaluation"
                )
            else:
                split_notes = list(split.notes)
                print(
                    "No held-out personas available for this context; "
                    "only the in-distribution score will be reported"
                )

        print("Utterances generation complete! Beginning model pipeline training\n")

        # ==================================================================================
        # Original training procedure below, with only artefact paths changed to per-context
        # ==================================================================================

        # unpack the test data and train data
        X, y = zip(*utterance_command_tuples)
        num= len(set(y))
        k_val = 3 if num>2 else 2
        # Base models are configurable so downstream apps can swap them without
        # code changes. The defaults are transformers 5.x-compatible BERT/DistilBERT
        # checkpoints that ship a `model_type` and a loadable tokenizer.
        tiny_model_name = fastworkflow.get_env_var(
            "INTENT_DETECTION_TINY_MODEL", default="google/bert_uncased_L-4_H-128_A-2")
        large_model_name = fastworkflow.get_env_var(
            "INTENT_DETECTION_LARGE_MODEL", default="distilbert-base-uncased")

        model_name = tiny_model_name
        print(f"\nLoading {model_name}...")
        tiny_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tiny_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)

        model_name = large_model_name
        print(f"Loading {model_name}...")
        distil_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #large_model = AutoModel.from_pretrained(model_name).to(device)
        large_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)
        global label_encoder
        dataset = list(zip(X, y))
        #label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Now create the dataset with encoded labels
        dataset = list(zip(X, y_encoded))
        train_data, test_data = split_training_data(
            dataset, lambda encoded: label_encoder.inverse_transform([encoded])[0]
        )

        # ---------------------------------------------------------------
        # Collate fn that keeps raw *texts* so we can avoid decode→encode
        # later during evaluation / fallback inference.
        # ---------------------------------------------------------------
        def make_collate_fn(tok):
            def _fn(batch):
                texts = [item[0] for item in batch]
                labels_tensor = torch.tensor([item[1] for item in batch], dtype=torch.long)
                encodings = tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                return encodings, labels_tensor, texts
            return _fn

        train_loader = DataLoader(
            train_data,
            batch_size=10,
            shuffle=True,
            collate_fn=make_collate_fn(tiny_tokenizer)
        )

        test_loader = DataLoader(
            test_data,
            batch_size=10,
            shuffle=False,
            collate_fn=make_collate_fn(tiny_tokenizer)
        )

        # -----------------------------------------------------------------
        # Preserve the Tiny-BERT test loader before we overwrite *test_loader*
        # for Distil-BERT.  All Tiny-BERT analysis & threshold-tuning should
        # continue to use this cached version to avoid tokenizer mismatch.
        # -----------------------------------------------------------------
        tiny_test_loader = test_loader

        #batch_size = 64  # Increased batch size
        optimizer = AdamW(tiny_model.parameters(), lr=1e-4)  # Slightly higher learning rate
        num_epochs = 12
        
        from time import time
        print("Starting training...")
        tiny_model.train()
        best_ndcg = 0
        best_f1 = 0
        training_start_time = time()
        training_losses = []  # Store training loss for each epoch
        test_losses = []
        for epoch in range(num_epochs):
            epoch_start_time = time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")

            for batch_idx, (encodings, labels, _) in enumerate(progress_bar):
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = tiny_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            avg_train_loss = total_loss / len(train_loader)
            training_losses.append(avg_train_loss)  # Append training loss for the epoch

            # Evaluate after each epoch
            f1, ndcg, avg_test_loss = evaluate_model(tiny_model, test_loader, device, k_val)
            test_losses.append(avg_test_loss)
            epoch_time = time() - epoch_start_time
            print(f"Epoch {epoch + 1} Results:")
            print(f"F1 Score: {f1:.4f}")
            print(f"NDCG@3: {ndcg:.4f}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")

        # Save paths updated to use context-specific folders
        tiny_path = get_artifact_path(workflow_folderpath, ctx_name, "tinymodel.pth")
        save_model(tiny_model, tiny_tokenizer, tiny_path)
        total_training_time = time() - training_start_time


        train_loader = DataLoader(
            train_data,
            batch_size=10,
            shuffle=True,
            collate_fn=make_collate_fn(distil_tokenizer)
        )

        test_loader = DataLoader(
            test_data,
            batch_size=10,
            shuffle=False,
            collate_fn=make_collate_fn(distil_tokenizer)
        )

        optimizer = AdamW(large_model.parameters(), lr=5e-5)
        num_epochs = 5

        print("Started training distilBert...")
        large_model.train()
        best_ndcg = 0
        best_f1 = 0
        num_epochs=5
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")

            for batch_idx, (encodings, labels, _) in enumerate(progress_bar):
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = large_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            # Evaluate after each epoch
            f1, ndcg, avg_loss = evaluate_model(large_model, test_loader, device, k_val)
            print(f"Epoch {epoch + 1} Results:")
            print(f"F1 Score: {f1:.4f}")
            print(f"NDCG@3: {ndcg:.4f}")

        # Save paths updated to use context-specific folders
        large_path = get_artifact_path(workflow_folderpath, ctx_name, "largemodel.pth")
        save_model(large_model, distil_tokenizer, large_path)

        pipeline = ModelPipeline(
            tiny_model_path=tiny_path,  
            distil_model_path=large_path,
            confidence_threshold=0.65
        )
        
        # Save paths updated to use context-specific folders
        label_path = get_artifact_path(workflow_folderpath, ctx_name, "label_encoder.pkl")
        save_label_encoder(label_path)

        print("\nAnalyzing TinyBERT confidence patterns...")
        tiny_stats, tiny_confidences, tiny_predictions, tiny_labels, tiny_failed = analyze_model_confidence(tiny_model, tiny_test_loader, device, "TinyBERT")

        print("\nTinyBERT Confidence Statistics:")
        print("\nFalse Classifications:")
        if tiny_stats['failed']['min'] is not None:
            print(f"Minimum Confidence: {tiny_stats['failed']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if tiny_stats['failed']['max'] is not None:
            print(f"Maximum Confidence: {tiny_stats['failed']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if tiny_stats['failed']['mean'] is not None:
            print(f"Mean Confidence: {tiny_stats['failed']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if tiny_stats['failed']['median'] is not None:
            print(f"Median Confidence: {tiny_stats['failed']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nTrue Classifications:")
        if tiny_stats['successful']['min'] is not None:
            print(f"Minimum Confidence: {tiny_stats['successful']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if tiny_stats['successful']['max'] is not None:
            print(f"Maximum Confidence: {tiny_stats['successful']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if tiny_stats['successful']['mean'] is not None:
            print(f"Mean Confidence: {tiny_stats['successful']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if tiny_stats['successful']['median'] is not None:
            print(f"Median Confidence: {tiny_stats['successful']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nAnalyzing DistilBERT confidence patterns...")
        large_stats, large_confidences, large_predictions, large_labels, large_failed = analyze_model_confidence(large_model, tiny_test_loader, device, "DistilBERT")

        print("\nTinyBERT Confidence Statistics:")
        print("\nFalse Classifications:")
        if large_stats['failed']['min'] is not None:
            print(f"Minimum Confidence: {large_stats['failed']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if large_stats['failed']['max'] is not None:
            print(f"Maximum Confidence: {large_stats['failed']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if large_stats['failed']['mean'] is not None:
            print(f"Mean Confidence: {large_stats['failed']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if large_stats['failed']['median'] is not None:
            print(f"Median Confidence: {large_stats['failed']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nTrue Classifications:")
        if large_stats['successful']['min'] is not None:
            print(f"Minimum Confidence: {large_stats['successful']['min']:.4f}")
        else:
            print("Minimum Confidence: N/A")
        if large_stats['successful']['max'] is not None:
            print(f"Maximum Confidence: {large_stats['successful']['max']:.4f}")
        else:
            print("Maximum Confidence: N/A")
        if large_stats['successful']['mean'] is not None:
            print(f"Mean Confidence: {large_stats['successful']['mean']:.4f}")
        else:
            print("Mean Confidence: N/A")
        if large_stats['successful']['median'] is not None:
            print(f"Median Confidence: {large_stats['successful']['median']:.4f}")
        else:
            print("Median Confidence: N/A")

        print("\nFinding optimal threshold...")
        best_result, all_results = find_optimal_threshold(tiny_stats, tiny_test_loader, pipeline)
        print("\nOptimal Threshold Results:")
        print(f"Threshold: {best_result['threshold']:.4f}")
        print(f"F1 Score: {best_result['f1']:.4f}")
        print(f"NDCG@3: {best_result['ndcg']:.4f}")
        print(f"DistilBERT Usage: {best_result['distil_usage']:.2f}%")

        pipeline.confidence_threshold = best_result['threshold']

        threshold = best_result['threshold']
        # Save paths updated to use context-specific folders
        threshold_path = get_artifact_path(workflow_folderpath, ctx_name, "threshold.json")
        with open(threshold_path, 'w') as f:
            json.dump({'confidence_threshold': threshold}, f)

        f1, ndcg, stats = pipeline.evaluate(tiny_test_loader)

        print("\nEvaluation Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"NDCG@3: {ndcg:.4f}")
        print("\nModel Usage Statistics:")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"DistilBERT Usage: {stats['distil_percentage']:.2f}%")
        print(f"TinyBERT Usage: {stats['tiny_percentage']:.2f}%")


        
        if large_stats['failed']['mean'] is not None:
            large_ambiguous_threshold = large_stats['failed']['mean']
        else:
            large_ambiguous_threshold = 0.0
        # Save paths updated to use context-specific folders
        large_ambiguous_threshold_path = get_artifact_path(workflow_folderpath, ctx_name, "large_ambiguous_threshold.json")
        with open(large_ambiguous_threshold_path, 'w') as f:
            json.dump({'confidence_threshold': large_ambiguous_threshold}, f)
        
        if tiny_stats['failed']['mean'] is not None:
            tiny_ambiguous_threshold = tiny_stats['failed']['mean']
        else:
            tiny_ambiguous_threshold = 0.0
        # Save paths updated to use context-specific folders
        tiny_ambiguous_threshold_path = get_artifact_path(workflow_folderpath, ctx_name, "tiny_ambiguous_threshold.json")
        with open(tiny_ambiguous_threshold_path, 'w') as f:
            json.dump({'confidence_threshold': tiny_ambiguous_threshold}, f)

    
        text = "list commands"
        result = predict_single_sentence(pipeline, text, label_path)
        print(f"Predicted label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Used DistilBERT: {'Yes' if result['used_distil'] else 'No'}")

        # ------------------------------------------------------------------
        # R1a/R1b: score the reserved personas through the REAL runtime path.
        # CommandRouter.predict is what intent detection actually calls, thresholds
        # and all, so this measures what a user would experience rather than what the
        # raw classifier head emits. Every artifact it needs was written above; the
        # model directory is taken from the threshold path so this keeps working
        # whether or not artifacts are being routed into a version.
        # ------------------------------------------------------------------
        report = heldout_evaluation.HeldoutReport(
            context=ctx_name,
            in_distribution_f1=f1,
            seed=seed,
            heldout_personas=heldout_personas,
            notes=split_notes,
        )
        try:
            has_context_benchmark = any(
                case.context == ctx_name
                and case.kind in {"routing", "escalation"}
                for case in benchmark_cases
            )
            predict_labels = None
            if heldout_records or has_context_benchmark:
                router = CommandRouter(os.path.dirname(threshold_path))

                def predict_labels(utterance: str, _router=router) -> list[str]:
                    """Adapt `CommandRouter.predict` to the scorer's contract.

                    `predict` returns either a one-element list holding a numpy string or
                    the raw numpy top-k array. The scorer needs a plain `list[str]`: a
                    numpy array raises "truth value of an array is ambiguous" the moment
                    anything tests it for emptiness.
                    """
                    return [str(label) for label in _router.predict(utterance)]

            # `kind` is required. Omitting it raised a TypeError that the guard
            # below turned into a note on the report, so escalation silently
            # never scored while routing kept working -- the failure looked like
            # "this workflow has no escalation cases" (bd fix-588).
            _score_heldout_context(
                report, heldout_records, benchmark_cases, predict_labels)
        except Exception as exc:
            # A scoring failure must never destroy a completed training run: the
            # models are already on disk and usable. Record it and move on.
            report.notes.append(f"held-out evaluation failed: {exc}")
            logger.error(
                f"Held-out evaluation failed for context '{ctx_name}': {exc}")
        heldout_reports.append(report)

    # End of context loop

    if heldout_reports:
        print(heldout_evaluation.format_report(heldout_reports))
        _write_heldout_report(workflow_folderpath, heldout_reports)
    return None


def _write_heldout_report(
    workflow_folderpath: str,
    heldout_reports: list[heldout_evaluation.HeldoutReport],
) -> Optional[str]:
    """Persist the held-out evaluation report, saying so loudly when it cannot.

    Returns the path written, or None on failure.

    Not fatal — the models are already on disk and usable, and a failed report write is
    not a reason to discard a completed run. Not silent either: a failure leaves
    `heldout_evaluation.json` stale or absent while the run reports success, and
    `selective_training.merge_heldout_evaluation` then merges this run's contexts into a
    baseline describing different artifacts, while `capture_heldout_evaluation` returns
    None on the next run. Both consequences are invisible from the training output, which
    is why the write cannot simply be suppressed.
    """
    try:
        report_path = heldout_evaluation.write_report(
            workflow_folderpath, heldout_reports)
    except OSError as exc:
        message = (
            f"Could not write the held-out evaluation report for "
            f"{workflow_folderpath}: {type(exc).__name__}: {exc}. The scores printed "
            f"above are NOT on disk; heldout_evaluation.json is stale or absent, so the "
            f"next selective run merges against the wrong baseline."
        )
        logger.error(message)
        print(f"WARNING: {message}", flush=True)
        return None
    print(f"Held-out evaluation report: {report_path}")
    return report_path