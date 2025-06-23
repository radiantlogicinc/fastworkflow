class Signature:
    def __init__(self, *args, **kwargs):
        pass

class InputField:
    def __init__(self, *args, **kwargs):
        pass

class OutputField:
    def __init__(self, *args, **kwargs):
        pass

class Example(dict):
    def with_inputs(self, *args, **kwargs):
        return self

class Module:
    def __init__(self, *args, **kwargs):
        pass

class ChainOfThought:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        class Dummy:
            def __getattr__(self, item):
                return None
        return Dummy()

class LabeledFewShot:
    def __init__(self, k=0):
        self.k = k
    def compile(self, student=None, trainset=None):
        return student

class LM:
    def __init__(self, *args, **kwargs):
        pass

class JSONAdapter:
    pass

import contextlib

@contextlib.contextmanager
def context(*args, **kwargs):
    yield