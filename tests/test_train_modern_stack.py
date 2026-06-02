"""Regression test: `fastworkflow train` on the modern stack.

This exercises the full intent-detection training pipeline end-to-end on the
bundled ``hello_world`` example and asserts that:

1. Trained per-context model artifacts are produced under ``___command_info/``
   (``tinymodel.pth``/``largemodel.pth`` etc.), not just the build JSONs.
2. A representative utterance routes to the expected command.

It is designed to run under the modern stack (transformers 5.x, dspy 3.x,
openai 2.x). Producing model artifacts requires the optional ``datasets``
package AND a real synthetic-data-generation key, so the test skips cleanly
when those are unavailable (e.g. CI without secrets configured).
"""

import os
import shutil
import importlib.util

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.__main__ import train_workflow
from fastworkflow.model_pipeline_training import CommandRouter


HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")

# Artifacts written by the intent-detection trainer for each context.
_MODEL_ARTIFACTS = [
    "tinymodel.pth",
    "largemodel.pth",
    "threshold.json",
    "tiny_ambiguous_threshold.json",
    "large_ambiguous_threshold.json",
    "label_encoder.pkl",
]


def _datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def _resolve_env_vars() -> dict:
    """Build the training env.

    Defaults to the bundled example env files; overlays the repo-local
    ``env/.env`` + ``passwords/.env`` (real keys) when present so the full
    synthetic-generation path is exercised locally.
    """
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}

    local_env = os.path.join("env", ".env")
    local_pwd = os.path.join("passwords", ".env")
    if os.path.exists(local_env):
        env_vars.update(dotenv_values(local_env))
    if os.path.exists(local_pwd):
        env_vars.update(dotenv_values(local_pwd))

    # Allow CI (or any caller) to override model strings / keys via process env,
    # e.g. LITELLM_API_KEY_SYNDATA_GEN provided as a CI secret. Importing
    # fastworkflow auto-loads the bundled *example* passwords (placeholders) into
    # os.environ, so we must ignore placeholder values here and never let them
    # clobber the real keys resolved from the local env/passwords files above.
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


def _command_info_path(workflow_path: str) -> str:
    return os.path.join(workflow_path, "___command_info")


def _cleanup(workflow_path: str, env_vars: dict) -> None:
    command_info = _command_info_path(workflow_path)
    if os.path.isdir(command_info):
        shutil.rmtree(command_info)
    speeddict = env_vars.get("SPEEDDICT_FOLDERNAME")
    if speeddict and os.path.isdir(os.path.join(workflow_path, speeddict)):
        shutil.rmtree(os.path.join(workflow_path, speeddict))
    if os.path.isdir("./___workflow_contexts"):
        shutil.rmtree("./___workflow_contexts")


def _find_model_dirs(command_info_path: str) -> list[str]:
    """Return all directories under ``command_info_path`` that hold a trained model."""
    # tinymodel.pth / largemodel.pth are written via save_pretrained, so they are
    # *directories* (not files), hence the check against both dirs and files.
    return [
        root
        for root, dirs, files in os.walk(command_info_path)
        if "tinymodel.pth" in dirs or "tinymodel.pth" in files
    ]


@pytest.fixture(scope="module")
def trained_hello_world():
    if not _datasets_available():
        pytest.skip("datasets package not installed; intent-detection training is skipped.")

    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip(
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run synthetic "
            "utterance generation required for model training."
        )

    workflow_path = HELLO_WORLD_PATH
    _cleanup(workflow_path, env_vars)

    fastworkflow.init(env_vars=env_vars)
    train_workflow(workflow_path)

    yield workflow_path, env_vars

    _cleanup(workflow_path, env_vars)


def test_train_produces_model_artifacts(trained_hello_world):
    """The trained model subdirectories (not just JSONs) must be produced."""
    workflow_path, _env_vars = trained_hello_world
    command_info = _command_info_path(workflow_path)

    assert os.path.isdir(command_info), f"{command_info} was not created"

    # Build JSONs must exist.
    for shared in ("command_directory.json", "routing_definition.json"):
        assert os.path.exists(os.path.join(command_info, shared)), (
            f"{shared} was not generated in {command_info}"
        )

    # At least one trained per-context model directory must exist with the full
    # artifact set.
    model_dirs = _find_model_dirs(command_info)
    assert model_dirs, (
        f"No trained model artifacts (tinymodel.pth) found under {command_info}. "
        "Training stopped at the build phase."
    )
    for model_dir in model_dirs:
        for artifact in _MODEL_ARTIFACTS:
            assert os.path.exists(os.path.join(model_dir, artifact)), (
                f"{artifact} missing from trained model dir {model_dir}"
            )


def test_trained_model_routes_utterance(trained_hello_world):
    """A representative utterance must route to the add_two_numbers command."""
    workflow_path, _env_vars = trained_hello_world
    command_info = _command_info_path(workflow_path)

    # The hello_world `add_two_numbers` command lives in the global context.
    model_dirs = _find_model_dirs(command_info)
    assert model_dirs, "No trained model directories to route against."

    routed = False
    for model_dir in model_dirs:
        router = CommandRouter(model_dir)
        labels = router.predict("add 2 and 3")
        if any("add_two_numbers" in label for label in labels):
            routed = True
            break

    assert routed, (
        "Utterance 'add 2 and 3' did not route to 'add_two_numbers' in any "
        f"trained context under {command_info}."
    )
