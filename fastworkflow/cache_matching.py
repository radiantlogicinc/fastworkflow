import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fastworkflow
import torch
from speedict import Rdict
import mmh3  # mmh33 implementation
from datetime import datetime
from functools import lru_cache

# ---------------------------------------------------------------------
# In-process memoisation for expensive DistilBERT embeddings.
# Key = (id(model_pipeline), text).  The cache is deliberately small –
# only the last 256 distinct (pipeline, text) pairs – to keep memory
# bounded while covering the overwhelmingly common repeat utterances
# seen during ambiguity clarification loops.
# ---------------------------------------------------------------------

_MAX_EMBED_CACHE_SIZE = 256

@lru_cache(maxsize=_MAX_EMBED_CACHE_SIZE)
def _cached_embedding(model_id: int, text: str):
    """Return embedding array for *text* using model_pipeline with *model_id*."""
    pipeline = _MODEL_ID_2_REF[model_id]()  # weakref
    if pipeline is None:
        # The pipeline object has been GC'd – recompute via a dummy call
        raise RuntimeError("ModelPipeline instance no longer alive; cache invalid.")
    return _compute_embedding(text, pipeline)

import weakref
_MODEL_ID_2_REF: dict[int, weakref.ReferenceType] = {}


def _compute_embedding(text: str, model_pipeline):
    """Actual embedding computation (was body of old get_embedding)."""
    model = model_pipeline.distil_model
    tokenizer = model_pipeline.distil_tokenizer
    device = model_pipeline.device
    model = model.distilbert

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def store_utterance_cache(cache_path, utterance, label, model_pipeline=None):
    """
    Store utterance in the new format with mmh3 and command mapping
    
    Args:
        cache_path (str): Path to the cache database
        utterance (str): The input utterance to store
        label (str): The true label for the utterance
        model_pipeline: Optional model pipeline to compute embedding
        
    Returns:
        The hash key of the stored utterance
    """
    # Open the database
    db = Rdict(cache_path)
    try:
        # Generate hash for utterance using mmh3
        utterance_hash = str(mmh3.hash(utterance))
        
        # Get the cache or initialize
        cache = db.get("cache", {})
        
        # Get current timestamp for feedback date
        current_time = datetime.now().isoformat()
        
        # Compute embedding if model_pipeline provided
        embedding = None
        if model_pipeline is not None:
            embedding = get_embedding(utterance, model_pipeline)[0].tolist()
        
        if utterance_hash in cache:
            # Update existing entry
            if embedding is not None:
                cache[utterance_hash]["embedding"] = embedding
                
            if label in cache[utterance_hash]["command_mapping"]:
                # Increment frequency for this label
                cache[utterance_hash]["command_mapping"][label]["frequency"] += 1
                cache[utterance_hash]["command_mapping"][label]["feedback_date"] = current_time
            else:
                # Add new label mapping
                cache[utterance_hash]["command_mapping"][label] = {
                    "frequency": 1,
                    "feedback_date": current_time
                }
        else:
            # Create new entry
            cache[utterance_hash] = {
                "embedding": embedding if embedding is not None else [],
                "utterance": utterance,  # Store original utterance for reference
                "command_mapping": {
                    label: {
                        "frequency": 1,
                        "feedback_date": current_time
                    }
                }
            }
        
        # Save updated cache to database
        db["cache"] = cache
        
        return utterance_hash
        
    finally:
        # Always close the database
        db.close()

def get_embedding(text: str, model_pipeline):
    """Return (possibly cached) embedding for *text* using *model_pipeline*."""
    model_id = id(model_pipeline)
    if model_id not in _MODEL_ID_2_REF:
        _MODEL_ID_2_REF[model_id] = weakref.ref(model_pipeline)
    try:
        return _cached_embedding(model_id, text)
    except RuntimeError:
        # Pipeline was garbage-collected; recompute and re-cache.
        return _compute_embedding(text, model_pipeline)

def cache_match(cache_path, utterance, model_pipeline, threshold=0.90, return_details=False):
    """
    Match an utterance against cached examples using embedding similarity.
    Use new data structure with multiple label support.
    
    Args:
        cache_path (str): Path to the cache database
        utterance (str): The input utterance to match
        model_pipeline: Model pipeline containing the DistilBERT model
        threshold (float): Similarity threshold for a successful match
        return_details (bool): Whether to return detailed matching information
        
    Returns:
        If match found: true_label or (true_label, similarity) if return_details=True
        If no match: None
    """
    # Open the database
    db = Rdict(cache_path)
    try:
        # Get the cache dictionary
        cache = db.get("cache", {})

        # If no entries, return None
        if not cache:
            return None

        # Get embedding for the query utterance
        query_embedding = get_embedding(utterance, model_pipeline)

        # Reshape query embedding for cosine_similarity
        query_embedding = query_embedding.reshape(1, -1)

        # Check cache for similar utterances
        best_similarity = 0
        cache_match = None

        # Find the best matching cached utterance
        for hash_key, entry in cache.items():
            # Skip entries without embeddings
            if not entry.get("embedding"):
                continue

            # Reshape cached embedding for cosine_similarity
            cached_embedding = np.array(entry["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                cache_match = hash_key

        # If good cache match found, determine the best label
        if best_similarity >= threshold and cache_match is not None:
            command_mapping = cache[cache_match]["command_mapping"]

            # If only one label, return it directly
            if len(command_mapping) == 1:
                true_label = next(iter(command_mapping.keys()))
            else:
                # Find label with highest frequency
                max_frequency = 0
                max_freq_labels = []

                for label, info in command_mapping.items():
                    freq = info["frequency"]
                    if freq > max_frequency:
                        max_frequency = freq
                        max_freq_labels = [label]
                    elif freq == max_frequency:
                        max_freq_labels.append(label)

                # If multiple labels with same frequency, choose most recent one
                if len(max_freq_labels) > 1:
                    true_label = max(
                        max_freq_labels,
                        key=lambda l: command_mapping[l]["feedback_date"]
                    )
                else:
                    true_label = max_freq_labels[0]

            return (true_label, best_similarity) if return_details else true_label
        # No good match found
        return None
    finally:
        # Always close the database
        db.close()