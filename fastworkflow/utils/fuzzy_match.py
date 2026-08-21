import re
from typing import Optional
import Levenshtein

def normalize_text(text):
        """
        Normalize text by removing spaces, @ symbol, underscores, and converting to lowercase
        """
        return re.sub(r'[@\s_]', '', str(text).lower())

def normalized_levenshtein_distance(
        s1, s2, max_normalized: Optional[float] = None):
        """Calculate normalized Levenshtein distance.

        ``max_normalized`` is an optional upper bound. Distances that cannot
        beat it are not fully computed; a value strictly greater than
        ``max_normalized`` is returned instead. Callers must treat that
        sentinel as "worse", never as a tie at the bound.
        """
        max_length = max(len(s1), len(s2))
        if max_length == 0:
            return 0.0
        if max_normalized is None:
            return Levenshtein.distance(s1, s2) / max_length
        # score_cutoff is an integer edit count. Bare int() floors products
        # such as (1/49)*49 == 0.999... to 0, which then abandons a later
        # candidate that ties the bound. Nudge up by less than one edit so
        # 0.999... becomes 1 and 0.8 stays 0.
        cutoff = int(max_normalized * max_length + 1e-9)
        distance = Levenshtein.distance(s1, s2, score_cutoff=cutoff)
        if distance > cutoff:
            return max_normalized + 1.0
        return distance / max_length

def best_window_distance(
        normalized_input: str,
        normalized_candidate: str,
        max_normalized: float = 1.0,
        ) -> float:
    """Smallest normalized distance between the input and any equal-length window
    of the candidate.

    Exact containment is distance 0.0 and returns immediately. Remaining
    windows use Levenshtein's ``score_cutoff`` so a window that cannot beat
    ``max_normalized`` (or the best window seen so far) is abandoned early.
    If no window beats ``max_normalized``, a value strictly greater than it
    is returned so the caller does not invent a tie at the bound.
    """
    if normalized_input in normalized_candidate:
        return 0.0

    len_input = len(normalized_input)
    n_windows = max(len(normalized_candidate) - len_input, 0) + 1
    best: Optional[float] = None
    bound = max_normalized
    for i in range(n_windows):
        dist = normalized_levenshtein_distance(
            normalized_input,
            normalized_candidate[i:i + len_input],
            max_normalized=bound,
        )
        if dist > bound:
            continue
        if best is None or dist < best:
            best = dist
            bound = dist
            if best == 0.0:
                return 0.0
    return best if best is not None else max_normalized + 1.0

def find_best_matches(input_text: str, 
                    text_list: list[str], 
                    threshold: float=0.4,
                    best_window: bool=False
                    ) -> tuple[list[str], Optional[float]]:
    """Find best match(es) using normalized Levenshtein distance.

    The function now returns *all* entries whose normalized Levenshtein
    distance equals the minimum distance computed for the provided
    ``input_text``. This is useful when multiple candidates tie for the
    smallest distance.

    Parameters
    ----------
    best_window
        By default each candidate is scored on its *leading* ``len(input_text)``
        characters, so a candidate that contains the input anywhere else is
        scored as though it did not contain it at all: searching for
        ``Garrison`` scores ``Barry Jones`` (0.375) ahead of ``Aaron Garrison``
        (0.750). When true, every window of that length is scored and the best
        one wins, so a containing candidate always reaches distance 0.0.

        This is opt-in rather than the default because it can only *lower*
        distances, and therefore admits strictly more matches at any fixed
        threshold. Callers that route on the first match instead of treating
        ties as ambiguity depend on the narrower leading-window behaviour.

    Returns
    -------
    tuple[list[str], Optional[float]]
        A tuple containing the list of best-matching original strings
        and the corresponding distance. If the best distance exceeds
        the ``threshold`` value, or if ``text_list`` is empty,
        ``([], None)`` is returned — never ``(None, None)``. Callers
        must treat an empty list as "no match" (truthiness or ``len``);
        ``best_matches is None`` is always false.
    """

    # Ensure we have a concrete list (e.g., when a generator is passed)
    text_list = list(text_list)

    normalized_input = normalize_text(input_text)
    normalized_list = [normalize_text(text) for text in text_list]

    if not normalized_list:
        return ([], None)

    if best_window:
        # Containment is distance 0.0. Once any candidate contains the
        # input, later candidates can only tie by also containing it —
        # there is no need to slide windows over the rest of the list.
        contained = [
            text for text, normalized in zip(text_list, normalized_list)
            if normalized_input in normalized
        ]
        if contained:
            return (contained, 0.0)

        normalized_distances = []
        best_so_far = 1.0
        for normalized in normalized_list:
            dist = best_window_distance(
                normalized_input, normalized, max_normalized=best_so_far)
            if dist < best_so_far:
                best_so_far = dist
            normalized_distances.append(dist)
    else:
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