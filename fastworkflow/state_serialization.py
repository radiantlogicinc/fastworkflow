"""One strict serializer for durable session state.

Strictness has to live at the *first* serializer. `WorkflowExecutionContext`
used to return ``json.loads(json.dumps(payload, default=str))``, so an
unsupported trajectory or artifact object had already become an ordinary string
before anything downstream could object — and a strict check at the store then
validated the lossy result and passed it. Checking at the store is checking
after the loss has happened.

So this module is the boundary: state either encodes losslessly or it does not
encode at all, and a caller that cannot encode keeps the runtime live rather
than writing something it will silently mis-restore.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class StateEncodingError(ValueError):
    """State could not be encoded losslessly. Never write a partial result."""


def validate_state(state: Any) -> None:
    """Raise StateEncodingError unless ``state`` round-trips through JSON exactly.

    Rejects, with the path to the offending value:
      * a non-dict at the top level;
      * non-string dictionary keys, at any depth;
      * values outside None/bool/int/float/str/list/dict;
      * NaN and infinities, which JSON has no representation for;
      * cycles;
      * a mutable container reachable by more than one path, because it restores
        as independent copies and any code that relied on the sharing is wrong
        afterwards in a way nothing reports.
    """
    if not isinstance(state, dict):
        raise StateEncodingError(
            f"state must be a dict at the top level, got {type(state).__name__}"
        )
    _walk(state, path="$", seen_containers={})


def encode_state(state: Any) -> str:
    """Validate and encode canonically: sorted keys, compact, no NaN.

    Canonical so that two equal states produce identical bytes and therefore
    identical digests — an unstable encoding would force a durable write on
    every retirement.
    """
    validate_state(state)
    return json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def state_digest(state: Any) -> str:
    """SHA-256 over the canonical encoding."""
    return hashlib.sha256(encode_state(state).encode("utf-8")).hexdigest()


def _walk(value: Any, *, path: str, seen_containers: dict[int, str]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise StateEncodingError(
                f"{path}: {value!r} has no JSON representation"
            )
        return

    if isinstance(value, (dict, list)):
        if (previous := seen_containers.get(id(value))) is not None:
            raise StateEncodingError(
                f"{path}: same {type(value).__name__} is also reachable at "
                f"{previous}; it would restore as two independent copies"
            )
        seen_containers[id(value)] = path

        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise StateEncodingError(
                        f"{path}: dictionary key {key!r} is "
                        f"{type(key).__name__}, not str"
                    )
                _walk(item, path=f"{path}.{key}", seen_containers=seen_containers)
        else:
            for index, item in enumerate(value):
                _walk(
                    item,
                    path=f"{path}[{index}]",
                    seen_containers=seen_containers,
                )
        return

    raise StateEncodingError(
        f"{path}: {type(value).__name__} is not JSON-native. Convert it in the "
        "serialization hook rather than letting it be coerced to a string."
    )
