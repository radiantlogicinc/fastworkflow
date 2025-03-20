import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fastworkflow
import torch
from speedict import Rdict  

def get_flag(cache_path):
        """
        Read a specific utterance from the database
        """
        db = Rdict(cache_path)
        try:
            return db.get("flag")
        finally:
            db.close()
def change_flag(cache_path,value):
        """
        Read a specific utterance from the database
        """
        db = Rdict(cache_path)
        try:
            db["flag"]=value
            
        finally:
            db.close()

def store_utterance_cache(cache_path, utterance, label):
    """
    Store utterance in existing or new database
    Initializes a flag key when creating new database
    Returns: The utterance count used
    """
    # Open the database (creates if doesn't exist)
    db = Rdict(cache_path)
    try:
        # Check if this is a new database by looking for utterance_count
        is_new_db = "utterance_count" not in db
        
        # If new database, initialize flag
        if is_new_db:
            db["flag"] = 0
        
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
def get_embedding(text, model_pipeline):
    """Get DistilBERT embedding for a text using the model pipeline"""
    model = model_pipeline.distil_model
    tokenizer = model_pipeline.distil_tokenizer
    device = model_pipeline.device
    model=model.distilbert
    
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
    Match an utterance against cached misclassified examples using embedding similarity.
    
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
        # Get the utterance count 
        utterance_count = db.get("utterance_count", 0)
        
        # If no entries, return None
        if utterance_count == 0:
            return None
            
        # Collect cache entries (excluding the counter)
        cache = {}
        for i in range(utterance_count):
            if i in db:  # Check if key exists
                cache[i] = db[i]
                
        # If no valid entries, return None
        if not cache:
            return None
            
        # Calculate embeddings for cached utterances
        cache_embeddings = {}
        for key, entry in cache.items():
            cache_embeddings[key] = get_embedding(entry['utterance'], model_pipeline)
        
        # Get embedding for the query utterance
        query_embedding = get_embedding(utterance, model_pipeline)
        
        # Check cache for similar utterances
        best_similarity = 0
        cache_match = None
        
        # Reshape query embedding for cosine_similarity
        query_embedding = query_embedding.reshape(1, -1)
        
        # Find the best matching cached utterance
        for key, cached_embedding in cache_embeddings.items():
            # Reshape cached embedding for cosine_similarity
            cached_embedding = cached_embedding.reshape(1, -1)
            similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                cache_match = key
        
        # If good cache match found, return the true label
        if best_similarity >= threshold and cache_match is not None:
            cached_data = cache[cache_match]
            true_label = cached_data['label']
            
            if return_details:
                return (true_label, best_similarity)
            return true_label
            
        # No good match found
        return None
    finally:
        # Always close the database
        db.close()