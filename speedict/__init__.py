class Rdict(dict):
    """Stub replacement for speedict.Rdict to satisfy tests without a real DB."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass