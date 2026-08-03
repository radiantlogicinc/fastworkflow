"""R2's acceptance test: two training runs at the same seed produce the same artifacts.

`test_utterance_cache_determinism.py` proved half of this — that two runs train on the
same UTTERANCES once R6's cache is on. Its own docstring names what it deliberately
left out: `generate_dspy_examples`, a second unseeded LLM call that writes
`<command>_param_labeled.json` at temperature 0.9, and whatever nondeterminism remains
inside fine-tuning itself. This module closes both (bd fix-czb).

The measurement that made the parameter path a blocker, on `examples/hello_world` at
`TRAINING_SEED=42`, two consecutive runs, utterance cache on and parameter cache off:

    add_two_numbers_param_labeled.json   DIFFERENT   0 of 30 example utterances shared

Nothing in that request is seeded and the sampling temperature is deliberately high,
so no `TRAINING_SEED` could have made it reproducible. Reuse is what closes it.

An integration test in the strict sense of `.cursor/rules/testing_rules.mdc`: the real
bundled workflow, the real trainer, the real generator, the real LLM. It therefore
needs a real synthetic-data key and skips cleanly without one, following
`test_train_modern_stack.py`'s convention.

Both caches are installed here by the test rather than by the trainer. Once
`train_workflow` installs the parameter-example cache itself — the one-line change
this issue hands to the integrating agent, mirroring how R6's cache is installed —
that one wins, and either way both point at the same workflow.
"""

import importlib.util
import json
import os
import shutil

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.__main__ import train_workflow
from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME
from fastworkflow.train.param_example_cache import (
    CACHE_DIRNAME as PARAM_CACHE_DIRNAME,
    ParamExampleCache,
    set_param_example_cache,
)
from fastworkflow.train.utterance_cache import (
    MODE_REUSE,
    UtteranceCache,
    set_utterance_cache,
)


HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")

# Written by `save_pretrained` (a directory) and by the threshold calibration.
# Everything under a trained context that a second run at the same seed must
# reproduce byte for byte.
_MODEL_FILE_SUFFIXES = (".pth", ".pkl", ".json", ".safetensors", ".bin", ".txt")


def _datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def _resolve_env_vars() -> dict:
    """Build the training env, exactly as `test_train_modern_stack.py` does."""
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}

    local_env = os.path.join("env", ".env")
    local_pwd = os.path.join("passwords", ".env")
    if os.path.exists(local_env):
        env_vars.update(dotenv_values(local_env))
    if os.path.exists(local_pwd):
        env_vars.update(dotenv_values(local_pwd))

    for key in (
        "LLM_SYNDATA_GEN",
        "LITELLM_API_KEY_SYNDATA_GEN",
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
    ):
        val = os.environ.get(key)
        if val and "<" not in val:
            env_vars[key] = val
    return env_vars


def _command_info(workflow_path: str) -> str:
    return os.path.join(workflow_path, COMMAND_INFO_FOLDERNAME)


def _param_cache_root(workflow_path: str) -> str:
    return os.path.join(_command_info(workflow_path), PARAM_CACHE_DIRNAME)


def _param_files(workflow_path: str) -> dict[str, bytes]:
    """Filename -> raw bytes for every `<command>_param_labeled.json`."""
    info = _command_info(workflow_path)
    snapshot = {}
    for name in sorted(os.listdir(info)):
        if name.endswith("_param_labeled.json"):
            with open(os.path.join(info, name), "rb") as f:
                snapshot[name] = f.read()
    return snapshot


def _model_files(workflow_path: str) -> dict[str, bytes]:
    """Relative path -> raw bytes for every trained artifact.

    `versions/` is skipped: it holds one immutable copy per run, so comparing it would
    compare two different version ids rather than two runs' outputs. The per-context
    entries at the top level are compatibility pointers INTO the current version, so
    walking them with `followlinks` reads the real bytes of whichever version each run
    published.
    """
    info = _command_info(workflow_path)
    snapshot: dict[str, bytes] = {}
    for root, dirs, files in os.walk(info, followlinks=True):
        dirs[:] = [
            d for d in dirs
            if d not in {"versions", "__pycache__", PARAM_CACHE_DIRNAME}
        ]
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, info)
            if rel.startswith("utterance_cache" + os.sep):
                continue
            # `save_pretrained` writes tinymodel.pth / largemodel.pth as DIRECTORIES,
            # so the weights are files inside a path component ending in `.pth`.
            inside_a_model_dir = ".pth" + os.sep in rel and name.endswith(
                _MODEL_FILE_SUFFIXES
            )
            is_calibration = name in {
                "threshold.json",
                "tiny_ambiguous_threshold.json",
                "large_ambiguous_threshold.json",
                "label_encoder.pkl",
            }
            if inside_a_model_dir or is_calibration:
                with open(full, "rb") as f:
                    snapshot[rel] = f.read()
    return snapshot


def _snapshot_param_cache(workflow_path: str) -> dict[str, str]:
    """Filename -> contents for every cache entry, so a rewrite is detectable."""
    root = _param_cache_root(workflow_path)
    if not os.path.isdir(root):
        return {}
    snapshot = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                snapshot[name] = f.read()
    return snapshot


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Train an isolated copy of `hello_world` twice at the same seed.

    A COPY, not the bundled example: other tests (e.g. `test_fastapi_service.py`)
    depend on the real one being pre-trained, and training rewrites its artifacts.
    """
    if not _datasets_available():
        pytest.skip("datasets package not installed; intent-detection training is skipped.")

    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip(
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run the synthetic "
            "generation required for model training."
        )

    workflow_path = str(
        tmp_path_factory.mktemp("param_example_determinism") / "hello_world"
    )
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )

    fastworkflow.init(env_vars=env_vars)

    results = []
    for _ in range(2):
        set_utterance_cache(UtteranceCache(workflow_path, mode=MODE_REUSE))
        set_param_example_cache(ParamExampleCache(workflow_path, mode=MODE_REUSE))
        try:
            train_workflow(workflow_path)
        finally:
            set_utterance_cache(None)
            set_param_example_cache(None)
        results.append(
            {
                "param_files": _param_files(workflow_path),
                "model_files": _model_files(workflow_path),
                "param_cache": _snapshot_param_cache(workflow_path),
            }
        )

    yield results, workflow_path, env_vars

    if os.path.isdir("./___workflow_contexts"):
        shutil.rmtree("./___workflow_contexts")


def test_the_param_example_cache_is_populated_by_the_first_run(two_runs):
    """Guard against the whole module passing because nothing was cached."""
    results, workflow_path, _env = two_runs
    assert results[0]["param_cache"], (
        f"No parameter-example cache entries were written under "
        f"{_param_cache_root(workflow_path)}; the rest of this module would pass "
        "vacuously."
    )
    assert any(name.endswith(".json") for name in results[0]["param_cache"])


def test_param_labeled_files_are_byte_identical_across_two_runs(two_runs):
    """The direct inverse of the measurement that opened bd fix-czb."""
    results, _workflow_path, _env = two_runs
    first, second = results[0]["param_files"], results[1]["param_files"]

    assert first, "no <command>_param_labeled.json was written at all"
    assert set(first) == set(second), (
        "the two runs wrote different sets of parameter-example files: "
        f"{sorted(set(first) ^ set(second))}"
    )

    differing = [name for name in sorted(first) if first[name] != second[name]]
    assert not differing, (
        "parameter-example files that changed between two runs at the same seed: "
        f"{differing}"
    )


def test_the_examples_are_a_real_non_empty_set(two_runs):
    """Two identical EMPTY files would satisfy the test above and mean nothing."""
    results, _workflow_path, _env = two_runs
    for name, raw in results[0]["param_files"].items():
        payload = json.loads(raw)
        assert payload["valid_examples"], f"{name} cached an empty example set"
        assert all(
            example.get("fields", {}).get("command")
            for example in payload["valid_examples"]
        ), f"{name} holds examples with no command utterance"


def test_the_second_run_did_not_rewrite_the_param_example_cache(two_runs):
    """Reuse, not regenerate-and-overwrite.

    Every entry carries a `created_at`, so a regenerated entry would differ even if
    the LLM happened to return the same text.
    """
    results, _workflow_path, _env = two_runs
    assert results[0]["param_cache"] == results[1]["param_cache"]


def test_the_param_example_cache_survived_the_stale_artifact_prune(two_runs):
    """`_prune_stale_artifacts` runs at the end of every train; the cache must live.

    It walks `___command_info` and removes unrecognised entries. `param_example_cache`
    is in `artifact_versioning.RESERVED_TOPLEVEL_NAMES`, which is what exempts it —
    and a cache that a training run deletes is a cache that never helps a second run.
    """
    _results, workflow_path, _env = two_runs
    root = _param_cache_root(workflow_path)
    assert os.path.isdir(root)
    assert [name for name in os.listdir(root) if name.endswith(".json")]
    # And it was not mistaken for a trained context and turned into a compatibility
    # pointer by `publish_version`, which would leave it holding model artifacts.
    assert not os.path.islink(root)
    assert not os.path.exists(os.path.join(root, "threshold.json"))


def test_trained_model_artifacts_are_byte_identical_across_two_runs(two_runs):
    """R2's full acceptance criterion, and the part nobody had measured.

    With both LLM paths pinned, the remaining question is whether fine-tuning itself
    is reproducible. Measured on this repo's reference machine (CUDA present, so the
    trainer selects it — `model_pipeline_training.py:34`) the answer is yes: 24 of 24
    artifacts byte-identical across two runs at `TRAINING_SEED=42`.

    That is an empirical result, not a guarantee. `seed_everything` seeds
    `random`/numpy/torch/CUDA, but nothing sets `torch.use_deterministic_algorithms`
    or `cudnn.deterministic`, so a different GPU, a different cuDNN, or a change to
    batch shapes could reintroduce drift. If this fails, that is the finding — do not
    delete the assertion; it is the only place the claim is checked.
    """
    results, _workflow_path, _env = two_runs
    first, second = results[0]["model_files"], results[1]["model_files"]

    assert first, "no trained model artifacts were found to compare"
    assert set(first) == set(second), (
        f"the two runs produced different artifact sets: {sorted(set(first) ^ set(second))}"
    )
    differing = [name for name in sorted(first) if first[name] != second[name]]
    assert not differing, (
        f"{len(differing)} of {len(first)} trained artifacts differ between two runs "
        f"at the same TRAINING_SEED: {differing}"
    )
