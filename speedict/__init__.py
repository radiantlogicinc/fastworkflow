class Rdict(dict):
    """A minimal stub replacement for speedict.Rdict for testing purposes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # speedict.Rdict supports context manager usage to auto-commit; mimic.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No persistence layer, just ignore.
        pass