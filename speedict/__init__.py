import os
import pickle
from typing import Any, Dict


class Rdict:
    """A simple persistent dictionary implementation."""
    
    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {}
        
        # Ensure the directory exists
        os.makedirs(self.path, exist_ok=True)
        
        # Load existing data if available
        self._load()
    
    def _get_file_path(self, key: str) -> str:
        """Get the file path for a given key."""
        # Simple hash-based file naming
        import hashlib
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.path, f"{key_hash}.pkl")
    
    def _load(self):
        """Load all existing data from disk."""
        if not os.path.exists(self.path):
            return
            
        for filename in os.listdir(self.path):
            if filename.endswith('.pkl'):
                try:
                    filepath = os.path.join(self.path, filename)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and 'key' in data and 'value' in data:
                            self._data[data['key']] = data['value']
                except Exception:
                    # Skip corrupted files
                    pass
    
    def _save_key(self, key: str, value: Any):
        """Save a single key-value pair to disk."""
        filepath = self._get_file_path(key)
        data = {'key': key, 'value': value}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _delete_key(self, key: str):
        """Delete a key from disk."""
        filepath = self._get_file_path(key)
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def __getitem__(self, key: str) -> Any:
        """Get value by key."""
        if key not in self._data:
            raise KeyError(key)
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set value by key."""
        self._data[key] = value
        self._save_key(key, value)
    
    def __delitem__(self, key: str):
        """Delete key."""
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]
        self._delete_key(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with default."""
        return self._data.get(key, default)
    
    def keys(self):
        """Get all keys."""
        return self._data.keys()
    
    def values(self):
        """Get all values."""
        return self._data.values()
    
    def items(self):
        """Get all key-value pairs."""
        return self._data.items()
    
    def close(self):
        """Close the database connection. For compatibility with the original Rdict."""
        # In this simple implementation, there's nothing to close
        # but we provide the method for API compatibility
        pass