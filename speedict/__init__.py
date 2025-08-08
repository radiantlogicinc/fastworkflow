import os
import json
import threading
from typing import Any, Iterator


class Rdict:
    """
    Minimal file-backed dictionary used by the project for simple persistent key-value storage.

    Behavior:
    - The constructor takes a directory path. A single JSON file (kv.json) is created inside
      that directory to store all key-value pairs.
    - All mutations are immediately persisted to disk for simplicity and test determinism.
    - Close is a no-op for API compatibility.
    - Only JSON-serializable keys/values are supported. Keys are stored as strings in the
      JSON representation. Non-string keys are converted to strings when persisting and
      when accessing via membership/get/set.
    """

    def __init__(self, directory_path: str):
        self._dir = directory_path
        self._file_path = os.path.join(self._dir, "kv.json")
        self._lock = threading.RLock()
        os.makedirs(self._dir, exist_ok=True)
        # Initialize storage
        if not os.path.exists(self._file_path):
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        with self._lock:
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                return {}
            except (FileNotFoundError, json.JSONDecodeError):
                # If file is missing or corrupted, reset to empty
                return {}

    def _save(self) -> None:
        with self._lock:
            tmp_path = self._file_path + ".tmp"
            # Write to temp file first to avoid partial writes leaving corrupt JSON
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            # Atomic replace
            os.replace(tmp_path, self._file_path)

    def _key(self, key: Any) -> str:
        # Normalize all keys to string for JSON mapping
        return str(key)

    # Mapping protocol -----------------------------------------------------
    def __contains__(self, key: Any) -> bool:  # pragma: no cover - trivial
        return self._key(key) in self._data

    def __getitem__(self, key: Any) -> Any:
        norm = self._key(key)
        if norm not in self._data:
            raise KeyError(key)
        return self._data[norm]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[self._key(key)] = value
        self._save()

    def get(self, key: Any, default: Any = None) -> Any:  # pragma: no cover - trivial
        return self._data.get(self._key(key), default)

    def keys(self) -> Iterator[str]:  # pragma: no cover - trivial
        return iter(self._data.keys())

    def values(self) -> Iterator[Any]:  # pragma: no cover - trivial
        return iter(self._data.values())

    def items(self) -> Iterator[tuple[str, Any]]:  # pragma: no cover - trivial
        return iter(self._data.items())

    def pop(self, key: Any, default: Any = None) -> Any:  # pragma: no cover - trivial
        norm = self._key(key)
        if norm in self._data:
            val = self._data.pop(norm)
            self._save()
            return val
        return default

    def popitem(self) -> tuple[str, Any]:  # pragma: no cover - trivial
        item = self._data.popitem()
        self._save()
        return item

    def clear(self) -> None:  # pragma: no cover - trivial
        self._data.clear()
        self._save()

    def update(self, other: dict[str, Any]) -> None:  # pragma: no cover - trivial
        for k, v in other.items():
            self._data[self._key(k)] = v
        self._save()

    # Context manager and lifecycle ---------------------------------------
    def close(self) -> None:  # API compatibility (no resources to release)
        pass

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        self.close()
        return False