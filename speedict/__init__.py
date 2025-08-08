import os
import json
import threading
from typing import Any, Optional


class Rdict:
    """
    Minimal, file-backed dictionary compatible with the subset of the speedict.Rdict API
    used in this codebase. Each instance writes to a single JSON file under the provided
    directory path. Keys are coerced to strings for JSON compatibility.
    """

    _FILENAME = "__rstore.json"

    def __init__(self, directory_path: str):
        self._dir = os.path.abspath(directory_path)
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, self._FILENAME)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self._path):
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                except Exception:
                    # Corrupt or incompatible file; fall back to empty
                    self._data = {}
            else:
                self._data = {}

    def _save(self) -> None:
        with self._lock:
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self._path)

    def __contains__(self, key: Any) -> bool:  # type: ignore[override]
        skey = self._stringify_key(key)
        with self._lock:
            return skey in self._data

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        skey = self._stringify_key(key)
        with self._lock:
            return self._data[skey]

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        skey = self._stringify_key(key)
        with self._lock:
            self._data[skey] = value
            # Persist on each write to mimic durability semantics
            self._save()

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        skey = self._stringify_key(key)
        with self._lock:
            return self._data.get(skey, default)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def close(self) -> None:
        # No-op, file is already persisted on set
        pass

    def _stringify_key(self, key: Any) -> str:
        # Keep it simple and deterministic
        try:
            return str(key)
        except Exception:
            # Fallback to repr if str fails for any reason
            return repr(key)