from __future__ import annotations

"""Utility to load and validate a flat `context_inheritance_model.json`.

The schema is a simple mapping of context names to their base classes:

```
{
  "*": {"base": []},
  "Order": {"base": ["*"]},
  "OrderLine": {"base": ["*"]},
}
```

Each context has a "base" list containing its base classes.
An empty dict is valid and indicates no contexts are defined.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, List, Set

__all__ = ["ContextModelLoader", "ContextModelLoaderError"]


class ContextModelLoaderError(Exception):
    """Raised for invalid or malformed context model files."""


class ContextModelLoader:
    """Loads and validates a flat command context model JSON file."""

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

        # Validate context entries
        for context_name, context_data in data.items():
            if not isinstance(context_data, dict):
                raise ContextModelLoaderError(f"Context '{context_name}' must map to an object/dictionary")
            
            if "base" not in context_data:
                raise ContextModelLoaderError(f"Context '{context_name}' is missing required 'base' key")
                
            if not isinstance(context_data["base"], list):
                raise ContextModelLoaderError(f"'base' for context '{context_name}' must be a list")

        self._model_data = data
        return data

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def contexts(self) -> Dict[str, Dict[str, Any]]:
        """Return all contexts; load the model if necessary."""
        if self._model_data is None:
            self.load()
        # mypy hint
        assert self._model_data is not None
        
        return self._model_data

    def bases(self, context: str) -> List[str]:
        """Return the base classes for a given context."""
        if self._model_data is None:
            self.load()
        assert self._model_data is not None
        
        if context not in self._model_data:
            return []
        
        return self._model_data[context].get("base", [])
