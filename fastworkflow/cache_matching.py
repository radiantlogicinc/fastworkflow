# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import fastworkflow
# import torch
# from speedict import Rdict  

# def get_flag(cache_path):
#         """
#         Read a specific utterance from the database
#         """
#         db = Rdict(cache_path)
#         try:
#             return db.get("flag")
#         finally:
#             db.close()
# def change_flag(cache_path,value):
#         """
#         Read a specific utterance from the database
#         """
#         db = Rdict(cache_path)
#         try:
#             db["flag"]=value
            
#         finally:
#             db.close()

# def store_utterance_cache(cache_path, utterance, label):
#     """
#     Store utterance in existing or new database
#     Initializes a flag key when creating new database
#     Returns: The utterance count used
#     """
#     # Open the database (creates if doesn't exist)
#     db = Rdict(cache_path)
#     try:
#         # Check if this is a new database by looking for utterance_count
#         is_new_db = "utterance_count" not in db
        
#         # If new database, initialize flag
#         if is_new_db:
#             db["flag"] = 0
        
#         # Get existing counter or initialize to 0
#         utterance_count = db.get("utterance_count", 0)
        
#         # Create and store the utterance entry
#         utterance_data = {
#             "utterance": utterance,
#             "label": label
#         }
#         db[utterance_count] = utterance_data
        
#         # Increment and store the counter
#         utterance_count += 1
#         db["utterance_count"] = utterance_count
        
#         return utterance_count - 1  # Return the count used for this utterance
#     finally:
#         # Always close the database
#         db.close()
# def get_embedding(text, model_pipeline):
#     """Get DistilBERT embedding for a text using the model pipeline"""
#     model = model_pipeline.distil_model
#     tokenizer = model_pipeline.distil_tokenizer
#     device = model_pipeline.device
#     model=model.distilbert
    
#     model.eval()
#     with torch.no_grad():
#         inputs = tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=128,
#             return_tensors="pt"
#         ).to(device)
        
#         outputs = model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# def cache_match(cache_path, utterance, model_pipeline, threshold=0.90, return_details=False):
#     """
#     Match an utterance against cached misclassified examples using embedding similarity.
    
#     Args:
#         cache_path (str): Path to the cache database
#         utterance (str): The input utterance to match
#         model_pipeline: Model pipeline containing the DistilBERT model
#         threshold (float): Similarity threshold for a successful match
#         return_details (bool): Whether to return detailed matching information
        
#     Returns:
#         If match found: true_label or (true_label, similarity) if return_details=True
#         If no match: None
#     """
#     # Open the database
#     db = Rdict(cache_path)
#     try:
#         # Get the utterance count 
#         utterance_count = db.get("utterance_count", 0)
        
#         # If no entries, return None
#         if utterance_count == 0:
#             return None
            
#         # Collect cache entries (excluding the counter)
#         cache = {}
#         for i in range(utterance_count):
#             if i in db:  # Check if key exists
#                 cache[i] = db[i]
                
#         # If no valid entries, return None
#         if not cache:
#             return None
            
#         # Calculate embeddings for cached utterances
#         cache_embeddings = {}
#         for key, entry in cache.items():
#             cache_embeddings[key] = get_embedding(entry['utterance'], model_pipeline)
        
#         # Get embedding for the query utterance
#         query_embedding = get_embedding(utterance, model_pipeline)
        
#         # Check cache for similar utterances
#         best_similarity = 0
#         cache_match = None
        
#         # Reshape query embedding for cosine_similarity
#         query_embedding = query_embedding.reshape(1, -1)
        
#         # Find the best matching cached utterance
#         for key, cached_embedding in cache_embeddings.items():
#             # Reshape cached embedding for cosine_similarity
#             cached_embedding = cached_embedding.reshape(1, -1)
#             similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]
            
#             if similarity > best_similarity:
#                 best_similarity = similarity
#                 cache_match = key
        
#         # If good cache match found, return the true label
#         if best_similarity >= threshold and cache_match is not None:
#             cached_data = cache[cache_match]
#             true_label = cached_data['label']
            
#             if return_details:
#                 return (true_label, best_similarity)
#             return true_label
            
#         # No good match found
#         return None
#     finally:
#         # Always close the database
#         db.close()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fastworkflow
import torch
from speedict import Rdict
import mmh3  # MurmurHash3 implementation
from datetime import datetime

def get_flag(cache_path):
    """
    Read a specific utterance from the database
    """
    db = Rdict(cache_path)
    try:
        return db.get("flag")
    finally:
        db.close()

def change_flag(cache_path, value):
    """
    Read a specific utterance from the database
    """
    db = Rdict(cache_path)
    try:
        db["flag"] = value
    finally:
        db.close()

def store_utterance_cache(cache_path, utterance, label, model_pipeline=None):
    """
    Store utterance in the new format with MurmurHash and command mapping
    
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
        # Check if this is a new database by looking for flag
        if "flag" not in db:
            db["flag"] = 0
            
        # Generate hash for utterance using MurmurHash
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

def get_embedding(text, model_pipeline):
    """Get DistilBERT embedding for a text using the model pipeline"""
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
            
            if return_details:
                return (true_label, best_similarity)
            return true_label
            
        # No good match found
        return None
    finally:
        # Always close the database
        db.close()