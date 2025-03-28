import re
import Levenshtein

def normalize_text(text):
        """
        Normalize text by removing spaces, @ symbol, underscores, and converting to lowercase
        """
        cleaned_text = re.sub(r'[@\s_]', '', str(text).lower())
        return cleaned_text

def normalized_levenshtein_distance(s1, s2):
    """Calculate normalized Levenshtein distance"""
    distance = Levenshtein.distance(s1, s2)
    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 0.0
    return distance / max_length

def find_best_match(input_text, text_list, threshold=0.4):
    """Find best match using normalized Levenshtein distance"""
    # Convert text_list to a list if it's not already one
    text_list = list(text_list)
    
    normalized_input = normalize_text(input_text)
    normalized_list = [normalize_text(text) for text in text_list]
    
    # If empty list, return None
    if not normalized_list:
        return None, None
    
    normalized_distances = [
        normalized_levenshtein_distance(normalized_input, normalized) 
        for normalized in normalized_list
    ]
    
    best_match_index = min(
        range(len(normalized_list)),
        key=lambda i: normalized_distances[i]
    )
    
    best_distance = normalized_distances[best_match_index]
    
    if best_distance <= threshold:
        return (
            text_list[best_match_index],
            best_distance
        )
    
    return None, None