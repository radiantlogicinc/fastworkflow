import os
from typing import Optional
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
from typing import List, Dict, Tuple,Union
import pickle
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=None
label_encoder=LabelEncoder()

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
    command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(
        workflow_folderpath
    )
    cmddir = command_routing_definition.command_directory
    return os.path.join(
        cmddir.get_commandinfo_folderpath(workflow_folderpath),
        model_name
    )

class ModelPipeline:
    def __init__(
        self,
        tiny_model_path: str,
        distil_model_path: str,
        confidence_threshold: float = 0.65,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
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
        num_labels = AutoModelForSequenceClassification.from_pretrained(tiny_model_path).config.num_labels
        self.k_val = min(3, num_labels)

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
_GLOBAL_CONTEXT_FOLDER = "_global"

def get_artifact_path(workflow_folderpath: str, context_name: str, filename: str) -> str:
    """Return the absolute path for a model/artefact for *context_name*.

    The file lives under:
        <workflow>/___command_info/<context_name>/<filename>

    The directory is created if it does not yet exist.  The special
    context name "*" is mapped to a folder named "_global".
    """
    from fastworkflow import CommandRoutingRegistry

    ctx_folder = context_name if context_name != "*" else _GLOBAL_CONTEXT_FOLDER
    crd = CommandRoutingRegistry.get_definition(workflow_folderpath)
    base_dir = Path(crd.command_directory.get_commandinfo_folderpath(workflow_folderpath)) / ctx_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / filename)

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

def train(session: fastworkflow.Session):
    """Train intent-classification models **per command context**.

    A separate Tiny-/Distil-BERT pair (and label-encoder / thresholds) is
    produced inside ``___command_info/<context_name>/`` for every context in
    the application workflow.  The dataset for a context is built from:

        effective_commands(context) ∪ core_commands  – {"wildcard"}
    """

    import time

    workflow_folderpath = session.workflow_snapshot.workflow_folderpath
    crd = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)
    cmd_dir = crd.command_directory

    # Helper to pull utterances for a command
    def _get_utterances(cmd: str) -> list[str]:
        um = cmd_dir.get_utterance_metadata(cmd)
        func = um.get_generated_utterances_func(workflow_folderpath)
        return func(session, cmd) if func else []

    core_cmds = set(cmd_dir.core_command_names)

    # Get contexts specific to this workflow (not from command_metadata_extraction)
    internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    internal_contexts = set(fastworkflow.CommandContextModel.load(internal_wf_path)._command_contexts.keys())

    if "command_metadata_extraction" in workflow_folderpath:
        context_set_for_training = (set(internal_contexts) - {'Core', '*'})
    else:
        context_set_for_training = (set(crd.contexts.keys()) - internal_contexts) | {'*'}

    # Only iterate through contexts defined in this specific workflow
    for ctx_name in context_set_for_training:
        ctx_cmd_list = crd.contexts[ctx_name]
        print(f"\n\n=== Training model for context: {ctx_name} ===\n")
        
        # Build the label set for this context
        train_cmds: set[str] = set(ctx_cmd_list) | core_cmds
        if not train_cmds:
            print(f"Skipping context {ctx_name} - no commands to train")
            continue  # nothing to train

        utterance_command_tuples: list[tuple[str, str]] = []
        for cmd_name in sorted(train_cmds):
            utterance_list = _get_utterances(cmd_name)
            utterance_command_tuples.extend(zip(utterance_list, [cmd_name] * len(utterance_list)))

        if not utterance_command_tuples:
            print(f"Skipping context {ctx_name} - no utterances available")
            continue  # skip empty
            
        # ==================================================================================
        # Original training procedure below, with only artefact paths changed to per-context
        # ==================================================================================

        # unpack the test data and train data
        X, y = zip(*utterance_command_tuples)
        num= len(set(y))
        k_val = 3 if num>2 else 2
        model_name = "prajjwal1/bert-tiny"
        print(f"\nLoading {model_name}...")
        tiny_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tiny_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num).to(device)


        model_name = "distilbert-base-uncased"
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
        train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)

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

        model = AutoModelForSequenceClassification.from_pretrained(tiny_path).to(device)

        optimal_threshold, best_metrics = find_optimal_confidence_threshold(
            model, 
            tiny_test_loader, 
            device,
            min_threshold=pipeline.confidence_threshold,
            max_top3_usage=0.3,
            k_val=k_val
        )
        ambiguous_threshold = optimal_threshold or 0.0

        # Save paths updated to use context-specific folders
        ambiguous_threshold_path = get_artifact_path(workflow_folderpath, ctx_name, "ambiguous_threshold.json")
        with open(ambiguous_threshold_path, 'w') as f:
            json.dump({'confidence_threshold': ambiguous_threshold}, f)

        if ambiguous_threshold > 0.0:
            print("\nOptimal Threshold Results:")
            print(f"Threshold: {best_metrics['threshold']:.3f}")
            print(f"F1 Score: {best_metrics['f1_score']:.3f}")
            print(f"Top-3 Usage: {best_metrics['top3_usage']:.3f}")
            print(f"Top-1 Accuracy: {best_metrics['top1_accuracy']:.3f}")
            print(f"Top-3 Accuracy: {best_metrics['top3_accuracy']:.3f}")

        text = "list commands"
        try:
            result = predict_single_sentence(pipeline, text, label_path)
            print(f"Predicted label: {result['label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Used DistilBERT: {'Yes' if result['used_distil'] else 'No'}")
        except ValueError as e:
            print(f"Error: {e}")
            
    # End of context loop
    return None