# Minimal DSPy stub for tests
# Provides just enough surface for code paths exercised in unit tests.

class Signature:  # type: ignore
    def __init__(self, fields=None, instructions: str = ""):
        self.fields = fields or {}
        self.instructions = instructions

class Tool:  # compatibility placeholder
    def __init__(self, name: str, func, **kwargs):
        self.name = name
        self.func = func

class LM:  # placeholder language model wrapper
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

class JSONAdapter:  # no-op adapter
    pass

class context:  # simple context manager stub
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class ChainOfThought:
    def __init__(self, signature: Signature):
        self.signature = signature
    def __call__(self, **kwargs):
        # Return a lightweight object with attributes for requested outputs
        class _Result: pass
        result = _Result()
        for name, (_type, field) in (self.signature.fields or {}).items():
            setattr(result, name, None)
        return result

class LabeledFewShot:
    def __init__(self, k: int = 0):
        self.k = k
    def compile(self, student, trainset=None):
        return student

class ReAct:  # agent stub
    def __init__(self, signature, tools=None, max_iters: int = 1):
        self.signature = signature
        self.tools = tools or []
        self.max_iters = max_iters
    def __call__(self, **kwargs):
        return ""

# Simple namespace for settings.configure
class settings:
    lm = None
    @classmethod
    def configure(cls, lm):
        cls.lm = lm