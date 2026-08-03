"""The R6 acceptance test: two training runs at the same seed train on the same data.

The measurement that made this issue p0 (spec §11, M4) was two back-to-back training
runs of `examples/hello_world` at `TRAINING_SEED=42`, same code and environment,
which produced **0 of 5** commands with identical utterance sets — differing in both
text and row count. `TRAINING_SEED` seeds `random`, `numpy`, `torch`, persona
selection and the train/test split; it cannot seed the LLM, so the training DATA
itself varied run to run and no downstream artifact could be reproducible.

This test reproduces that harness and asserts the opposite outcome now that
generated utterances are persisted and reused.

It is an integration test in the strict sense of `.cursor/rules/testing_rules.mdc`:
the real bundled workflow, the real trainer, the real generator, the real LLM. It
therefore needs a real synthetic-data key and skips cleanly without one, following
`test_train_modern_stack.py`'s convention.

Scope, stated plainly: this asserts identical *utterance sets and provenance*, which
is what R6 owns. It does NOT assert byte-identical model artifacts. Two other
unseeded LLM paths remain outside R6 — `generate_dspy_examples`, which writes
`<command>_param_labeled.json` at temperature 0.9, and any nondeterminism in
fine-tuning itself. R2 cannot be closed on this test alone.
"""

import importlib.util
import json
import os
import shutil

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.__main__ import train_workflow
from fastworkflow.train.determinism import (
    COMMAND_INFO_FOLDERNAME,
    PROVENANCE_FILENAME,
)
from fastworkflow.train.utterance_cache import (
    CACHE_DIRNAME,
    MODE_REUSE,
    UtteranceCache,
    set_utterance_cache,
)


HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")


def _datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def _resolve_env_vars() -> dict:
    """Build the training env, exactly as `test_train_modern_stack.py` does.

    Defaults to the bundled example env files; overlays the repo-local ``env/.env``
    + ``passwords/.env`` (real keys) when present so the full synthetic-generation
    path is exercised locally.
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


def _provenance_path(workflow_path: str) -> str:
    return os.path.join(workflow_path, COMMAND_INFO_FOLDERNAME, PROVENANCE_FILENAME)


def _cache_root(workflow_path: str) -> str:
    return os.path.join(workflow_path, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME)


def _read_provenance(workflow_path: str) -> dict:
    with open(_provenance_path(workflow_path)) as f:
        return json.load(f)


def _command_records(provenance: dict) -> dict:
    """Return generation records from schema v2 or the legacy flat schema."""
    commands = provenance.get("commands")
    return commands if isinstance(commands, dict) else provenance


def _utterance_sets(provenance: dict) -> dict[str, list[str]]:
    """Per command, the exact set of utterances it trained on.

    `utterance_personas` is keyed by utterance text and covers the command-name
    token, the hand-written seeds and every generated row, so its key set IS the
    command's training data. This is the comparison the M4 measurement made.
    """
    return {
        command: sorted(record.get("utterance_personas", {}))
        for command, record in _command_records(provenance).items()
    }


def _snapshot_cache(workflow_path: str) -> dict[str, str]:
    """Filename -> contents for every cache entry, so a rewrite is detectable."""
    root = _cache_root(workflow_path)
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
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run synthetic "
            "utterance generation required for model training."
        )

    workflow_path = str(
        tmp_path_factory.mktemp("utterance_cache_determinism") / "hello_world"
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
        # Installed per run: the trainer clears the sink when it finishes, and once
        # `train_workflow` installs its own cache (the integration this issue hands
        # over) that one wins — either way both point at the same workflow.
        set_utterance_cache(UtteranceCache(workflow_path, mode=MODE_REUSE))
        try:
            train_workflow(workflow_path)
        finally:
            set_utterance_cache(None)
        results.append(
            {
                "provenance": _read_provenance(workflow_path),
                "provenance_bytes": open(_provenance_path(workflow_path), "rb").read(),
                "cache": _snapshot_cache(workflow_path),
            }
        )

    yield results, workflow_path, env_vars

    if os.path.isdir("./___workflow_contexts"):
        shutil.rmtree("./___workflow_contexts")


def test_the_cache_is_populated_by_the_first_run(two_runs):
    """Guard against the whole suite passing because nothing was cached."""
    results, workflow_path, _env = two_runs
    assert results[0]["cache"], (
        f"No utterance cache entries were written under {_cache_root(workflow_path)}; "
        "the rest of this module would pass vacuously."
    )
    assert any(name.endswith(".json") for name in results[0]["cache"])


def test_two_runs_at_the_same_seed_produce_identical_utterance_sets(two_runs):
    """The R6 acceptance criterion, and the direct inverse of the M4 measurement."""
    results, _workflow_path, _env = two_runs
    first, second = _utterance_sets(results[0]["provenance"]), _utterance_sets(
        results[1]["provenance"]
    )

    assert set(first) == set(second), (
        "the two runs trained different sets of commands: "
        f"{sorted(set(first) ^ set(second))}"
    )

    differing = [
        f"{command}  A={len(first[command])} rows  B={len(second[command])} rows"
        for command in sorted(first)
        if first[command] != second[command]
    ]
    assert not differing, (
        "commands whose utterance sets changed between two runs at the same seed:\n"
        + "\n".join(differing)
    )


def test_row_counts_are_identical(two_runs):
    """Row COUNT drift was half of what M4 measured; assert it separately.

    `final_count` counts rows including duplicates, which the set comparison above
    would not catch.
    """
    results, _workflow_path, _env = two_runs
    first_records = _command_records(results[0]["provenance"])
    second_records = _command_records(results[1]["provenance"])
    for command, record in sorted(first_records.items()):
        other = second_records[command]
        assert record["final_count"] == other["final_count"], (
            f"{command}: {record['final_count']} rows then "
            f"{other['final_count']} rows"
        )
        assert record["generated_count"] == other["generated_count"]


def test_provenance_is_byte_identical(two_runs):
    """The strongest available form: the whole record round-trips unchanged.

    This is why cache hit/miss status is deliberately NOT recorded in provenance —
    it is a property of the run, not of the data, and recording it would make this
    assertion impossible to satisfy by construction.
    """
    results, _workflow_path, _env = two_runs
    assert results[0]["provenance_bytes"] == results[1]["provenance_bytes"]


def test_the_second_run_did_not_rewrite_the_cache(two_runs):
    """Reuse, not regenerate-and-overwrite.

    Every entry carries a `created_at`, so a regenerated entry would differ even if
    the LLM happened to return the same text.
    """
    results, _workflow_path, _env = two_runs
    assert results[0]["cache"] == results[1]["cache"]


def test_no_command_fell_back_in_either_run(two_runs):
    """A fallen-back command is degraded data, and is never cached.

    If this fails the run above was rate-limited, and the identical-utterance result
    would be identical *degraded* data rather than evidence that reuse works.
    """
    results, _workflow_path, _env = two_runs
    for index, result in enumerate(results):
        fell_back = [
            command
            for command, record in _command_records(result["provenance"]).items()
            if record.get("fell_back")
        ]
        assert not fell_back, f"run {index + 1} degraded for: {fell_back}"


def test_trained_artifacts_still_exist_after_both_runs(two_runs):
    """Reuse must not have short-circuited training itself."""
    _results, workflow_path, _env = two_runs
    command_info = os.path.join(workflow_path, COMMAND_INFO_FOLDERNAME)
    model_dirs = [
        root
        for root, dirs, files in os.walk(command_info)
        if "tinymodel.pth" in dirs or "tinymodel.pth" in files
    ]
    assert model_dirs, f"No trained model artifacts under {command_info}"


def test_the_cache_survived_the_stale_artifact_prune(two_runs):
    """`_prune_stale_artifacts` runs at the end of every train; the cache must live.

    It walks `___command_info` and removes unrecognised entries. `utterance_cache` is
    in `artifact_versioning.RESERVED_TOPLEVEL_NAMES`, which is what exempts it — and
    a cache that a training run deletes is a cache that never helps a second run.
    """
    _results, workflow_path, _env = two_runs
    root = _cache_root(workflow_path)
    assert os.path.isdir(root)
    assert [name for name in os.listdir(root) if name.endswith(".json")]
    # And it was not mistaken for a trained context and turned into a compatibility
    # pointer by `publish_version`, which would leave it holding model artifacts.
    assert not os.path.islink(root)
    assert not os.path.exists(os.path.join(root, "threshold.json"))
