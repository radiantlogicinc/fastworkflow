import re
from typing import Optional
import Levenshtein

def normalize_text(text):
        """
        Normalize text by removing spaces, @ symbol, underscores, and converting to lowercase
        """
        return re.sub(r'[@\s_]', '', str(text).lower())

def normalized_levenshtein_distance(s1, s2):
        """Calculate normalized Levenshtein distance"""
        distance = Levenshtein.distance(s1, s2)
        max_length = max(len(s1), len(s2))
        return 0.0 if max_length == 0 else distance / max_length

def find_best_matches(input_text: str, 
                    text_list: list[str], 
                    threshold: float=0.4
                    ) -> tuple[list[str], float] | tuple[None, None]:
    """Find best match(es) using normalized Levenshtein distance.

    The function now returns *all* entries whose normalized Levenshtein
    distance equals the minimum distance computed for the provided
    ``input_text``. This is useful when multiple candidates tie for the
    smallest distance.

    Returns
    -------
    tuple[list[str], float] | tuple[None, None]
        A tuple containing the list of best-matching original strings
        and the corresponding distance. If the best distance exceeds
        the ``threshold`` value, ``(None, None)`` is returned.
    """

    # Ensure we have a concrete list (e.g., when a generator is passed)
    text_list = list(text_list)

    normalized_input = normalize_text(input_text)
    normalized_list = [normalize_text(text) for text in text_list]

    # If empty list, return None
    if not normalized_list:
        return ([], None)

    len_input = len(normalized_input)
    # Compute distances between the input and every candidate truncated to len of input
    normalized_distances = [
        normalized_levenshtein_distance(normalized_input, normalized[:len_input])
        for normalized in normalized_list
    ]

    # Determine the minimal distance observed
    best_distance = min(normalized_distances)

    # Gather *all* indices whose distance equals the minimal distance
    # Use a small tolerance to account for floating-point precision issues
    tolerance = 1e-12
    best_match_indices = [
        i for i, dist in enumerate(normalized_distances)
        if abs(dist - best_distance) <= tolerance
    ]

    # Map indices back to the original (non-normalized) texts
    best_matches = [text_list[i] for i in best_match_indices]

    if best_distance <= threshold:
        return (best_matches, best_distance)

    return ([], None)