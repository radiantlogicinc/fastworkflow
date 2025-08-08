import os
import pickle
from typing import Any, Optional


class Rdict:
    """
    Minimal, file-backed dictionary used as a local shim for speedict.Rdict.

    - Persists a Python dict to <path>/rdb.pkl
    - Supports __contains__, __getitem__, __setitem__, get, close
    - Creates the parent directory if it does not exist
    - Not safe for concurrent writers; adequate for test usage
    """

    def __init__(self, path: str):
        self._root_path = os.path.abspath(path)
        os.makedirs(self._root_path, exist_ok=True)
        self._db_path = os.path.join(self._root_path, "rdb.pkl")
        self._data: dict[Any, Any] = {}
        if os.path.exists(self._db_path):
            try:
                with open(self._db_path, "rb") as fh:
                    self._data = pickle.load(fh)
            except Exception:
                # If corrupted or unreadable, start fresh to avoid hard failures in tests
                self._data = {}

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        return self._data.get(key, default)

    def close(self) -> None:
        # Persist to disk on close
        try:
            with open(self._db_path, "wb") as fh:
                pickle.dump(self._data, fh)
        except Exception:
            # As a shim, do not raise; best-effort persistence
            pass