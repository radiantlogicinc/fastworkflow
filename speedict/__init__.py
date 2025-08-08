import os
import shelve
import threading
from typing import Any, Iterable


class Rdict:
    """
    Minimal persistent dictionary using Python's shelve module.

    Behaves like a dict for the limited operations used in the codebase:
    - __contains__, __getitem__, __setitem__, __delitem__
    - get, keys, items, clear, close

    The constructor accepts a directory-like path. Data will be stored under a
    file named 'db' inside that directory to be compatible with existing usage
    that passes folder paths.
    """

    def __init__(self, path: str):
        self._lock = threading.RLock()

        # If a directory path is provided, ensure it exists and store under 'db'
        if path.endswith(os.sep) or not os.path.splitext(path)[1] or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            db_path = os.path.join(path, "db")
        else:
            # Treat as a file path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            db_path = path

        # writeback=False to avoid caching entire objects; explicit assignments persist
        self._shelf = shelve.open(db_path, flag="c", writeback=False)

    def _k(self, key: Any) -> str:
        # Shelve/dbm require string keys; normalise to str to support int keys
        return str(key)

    # Mapping protocol
    def __contains__(self, key: Any) -> bool:  # type: ignore[override]
        with self._lock:
            return self._k(key) in self._shelf

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        with self._lock:
            return self._shelf[self._k(key)]

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        with self._lock:
            self._shelf[self._k(key)] = value
            self._shelf.sync()

    def __delitem__(self, key: Any) -> None:  # type: ignore[override]
        with self._lock:
            del self._shelf[self._k(key)]
            self._shelf.sync()

    def __iter__(self) -> Iterable[str]:  # type: ignore[override]
        with self._lock:
            return iter(list(self._shelf.keys()))

    # Dict helpers
    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return self._shelf.get(self._k(key), default)

    def keys(self):
        with self._lock:
            return list(self._shelf.keys())

    def items(self):
        with self._lock:
            return list(self._shelf.items())

    def clear(self) -> None:
        with self._lock:
            self._shelf.clear()
            self._shelf.sync()

    def close(self) -> None:
        with self._lock:
            try:
                self._shelf.sync()
            finally:
                self._shelf.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False