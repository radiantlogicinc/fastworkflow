"""Experiment lifecycle and setup APIs."""

__all__ = [
    "AttemptBootstrap",
    "AttemptClaim",
    "AttemptRun",
    "BenchmarkPinDigestMismatch",
    "ExperimentAborted",
    "ExperimentController",
    "ExperimentHarness",
    "ExperimentTask",
    "Grader",
    "LM_CACHE_VAR",
    "MissingExperimentLifecycleFeature",
    "UTTERANCE_CACHE_SCOPE_VAR",
    "channel_for",
    "derived_outcome",
    "experiment_store_readiness",
]


def __getattr__(name: str):
    # PEP 562: runner imports WorkflowExecutionContext, which imports dspy and litellm.
    import importlib

    if name in __all__:
        module = importlib.import_module("fastworkflow.experiment.runner")
        for export in __all__:
            globals()[export] = getattr(module, export)
        return globals()[name]
    try:
        module = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{name}":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
        raise
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(__all__) | {key for key in globals() if not key.startswith("_")} | {"runner"})
