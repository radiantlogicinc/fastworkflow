import json
import os
from typing import Any, Iterator


class Rdict:
    """
    Minimal filesystem-backed dict interface compatible with tests expecting speedict.Rdict.
    Stores content in a single JSON file under the provided directory path.
    """
    def __init__(self, dir_path: str):
        self._dir = os.path.abspath(dir_path)
        os.makedirs(self._dir, exist_ok=True)
        self._file = os.path.join(self._dir, "db.json")
        self._data: dict[str, Any] = {}
        if os.path.exists(self._file):
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # Ensure keys are strings internally
                    self._data = {str(k): v for k, v in raw.items()}
            except Exception:
                self._data = {}

    def __contains__(self, key: str) -> bool:
        return str(key) in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[str(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[str(key)] = value
        self._flush()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(str(key), default)

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            # Best effort – ignore write errors in test shim
            pass