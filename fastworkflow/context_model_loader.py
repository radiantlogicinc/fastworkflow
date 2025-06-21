from __future__ import annotations

"""Utility to load and validate a *v2* `context_inheritance_model.json`.

The v2 schema wraps *inheritance* and *aggregation* information in two top-level
keys:

```
{
  "inheritance": {
      "*": {"base": []},
      "Order": {"base": ["*"]}
  },
  "aggregation": {
      "OrderLine": {"container": ["Order"]}
  }
}
```

Only ``inheritance`` is mandatory. If the JSON lacks an ``aggregation`` block
we transparently add an empty one so callers can rely on its presence.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional

__all__ = ["ContextModelLoader", "ContextModelLoaderError"]


class ContextModelLoaderError(Exception):
    """Raised for invalid or malformed context model files."""


class ContextModelLoader:
    """Loads and validates a v2 *command context model* JSON file."""

    def __init__(self, model_path: str | Path = "_commands/context_inheritance_model.json") -> None:
        self.model_path = Path(model_path)
        self._model_data: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Load and validate the context model file, returning the parsed dict."""

        if self._model_data is not None:
            return self._model_data  # cached

        if not self.model_path.exists():
            raise ContextModelLoaderError(f"Context model file not found: {self.model_path}")

        try:
            with self.model_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ContextModelLoaderError(
                f"Invalid JSON in context model file: {self.model_path}\n{exc}"
            ) from exc

        # Basic structure validation
        if not isinstance(data, dict):
            raise ContextModelLoaderError("Root of context model must be a JSON object (dict).")

        if "inheritance" not in data:
            raise ContextModelLoaderError("Missing required 'inheritance' key in context model")

        if not isinstance(data["inheritance"], dict):
            raise ContextModelLoaderError("'inheritance' must map to an object/dictionary")

        # Ensure aggregation present and sane
        if "aggregation" not in data:
            data["aggregation"] = {}
        elif not isinstance(data["aggregation"], dict):
            raise ContextModelLoaderError("'aggregation' must map to an object/dictionary")

        self._model_data = data
        return data

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def inheritance(self) -> Dict[str, Any]:
        """Return the inheritance mapping; load the model if necessary."""
        if self._model_data is None:
            self.load()
        # mypy hint
        assert self._model_data is not None
        return self._model_data["inheritance"]

    @property
    def aggregation(self) -> Dict[str, Any]:
        """Return the aggregation mapping (guaranteed to exist)."""
        if self._model_data is None:
            self.load()
        assert self._model_data is not None
        return self._model_data["aggregation"] 