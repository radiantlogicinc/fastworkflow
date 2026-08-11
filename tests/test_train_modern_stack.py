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

import json
import os
import shutil
import importlib.util

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.__main__ import train_workflow
from fastworkflow.train import heldout_evaluation
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.train.determinism import get_training_seed
from fastworkflow.train.generate_synthetic import (
    utterance_fingerprint,
)
from fastworkflow.train.utterance_cache import MODE_REUSE, UtteranceCache


# Every test in this module drives a real `train_workflow`, so every one of them costs
# minutes of fine-tuning plus (when a key is configured) real DSPy parameter-example
# calls. Marked at module scope so `-m "not slow"` deselects the whole file; the
# `requires_llm_key` marker names the other reason a caller may want to exclude them.
# Without this there was no way to deselect the ~7 full training runs in the suite.
# bd fix-k0i.42.
pytestmark = [pytest.mark.slow, pytest.mark.requires_llm_key]

HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")
ADD_TWO_NUMBERS_COMMAND = "add_two_numbers"
ROUTING_UTTERANCE = "add 2 and 3"

# ROUTING_UTTERANCE is planted verbatim in `_fixed_generated_corpus` so that the corpus
# -- and therefore the trained model -- is identical on every run (the flake this fixture
# exists to remove). Routing it back is consequently a MEMORISATION check: a regression
# that destroyed generalisation while preserving the training rows would still pass it.
# HELDOUT_ROUTING_UTTERANCE is deliberately absent from the corpus and from
# hello_world's seed list, so routing it is evidence about the model rather than about
# the fixture. bd fix-k0i.30.
HELDOUT_ROUTING_UTTERANCE = "what is 2 plus 3"
_FIXTURE_PERSONA_IDS = [f"fixture:{index}" for index in range(4)]

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
    # Persistent state is rooted at FASTWORKFLOW_STATE_ROOT (isolated per-test by
    # the conftest fixture), not under the workflow folder, so nothing
    # workflow-local remains to remove. The legacy CWD folder is purged
    # defensively for older runs.
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


def _fixed_generated_corpus(command_name: str) -> list[str]:
    """Return 20 deterministic generated rows for one hello-world intent."""
    if command_name == ADD_TWO_NUMBERS_COMMAND:
        return [
            "sum the values 8 and 5",
            "calculate 12 plus 4",
            "what is 9 added to 6",
            "total the numbers 11 and 7",
            "combine 3 with 14 mathematically",
            ROUTING_UTTERANCE,
            "please add 17 and 2",
            "find the sum of 21 and 9",
            "compute 4 plus 13",
            "give me the total of 6 and 18",
            "add together 5 and 16",
            "what do 7 and 19 sum to",
            "calculate the addition of 23 and 1",
            "sum up 10 with 15",
            "please total 22 and 8",
            "combine the two numbers 13 and 3",
            "work out 18 plus 12",
            "add the values 20 and 6",
            "tell me the sum of 14 and 9",
            "compute the total when adding 16 and 7",
        ]

    intent = command_name.split("/")[-1].replace("_", " ")
    return [
        f"{prefix} {intent}{suffix}"
        for prefix, suffix in (
            ("please", ""),
            ("can you", " for me"),
            ("I need to", ""),
            ("help me", ""),
            ("go ahead and", ""),
            ("would you", " now"),
            ("kindly", ""),
            ("let me", ""),
            ("I want to", ""),
            ("could you", " please"),
            ("please now", ""),
            ("when ready", ""),
            ("for me", ""),
            ("quickly", ""),
            ("today", ""),
            ("at once", ""),
            ("right away", ""),
            ("in this session", ""),
            ("show me how to", ""),
            ("perform", ""),
        )
    ]


def _preseed_fixed_utterance_cache(workflow_path: str) -> dict[str, list[str]]:
    """Populate the fresh workflow's real R6 cache before its first train."""
    cmd_dir = CommandDirectory.load(workflow_path)
    cache = UtteranceCache(workflow_path, mode=MODE_REUSE)
    seed = get_training_seed()
    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    num_personas = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS", int)
    utterances_per_persona = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA", int)
    personas_per_batch = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH", int)
    expected_corpora = {}
    commands = set(cmd_dir.get_utterance_keys()) | set(cmd_dir.core_command_names)

    for command_name in sorted(commands):
        if command_name.split("/")[-1] == "wildcard":
            continue
        cmd_dir.ensure_command_hydrated(command_name)
        metadata = cmd_dir.get_utterance_metadata(command_name)
        corpus = _fixed_generated_corpus(command_name)
        fingerprint = utterance_fingerprint(
            metadata.plain_utterances,
            command_name,
            num_personas,
            utterances_per_persona,
            personas_per_batch,
            model,
        )
        attribution = {
            utterance: _FIXTURE_PERSONA_IDS[index // 5]
            for index, utterance in enumerate(corpus)
        }
        assert cache.store(
            fingerprint,
            seed,
            corpus,
            utterance_personas=attribution,
            persona_ids=_FIXTURE_PERSONA_IDS,
            persona_dataset_size=len(_FIXTURE_PERSONA_IDS),
        )
        expected_corpora[command_name] = corpus

    return expected_corpora


def _assert_fixed_corpus_was_used(
    workflow_path: str,
    expected_corpora: dict[str, list[str]],
) -> None:
    """Prove every intent hit the preseed instead of making a first LLM draw."""
    provenance_path = os.path.join(
        _command_info_path(workflow_path), "training_provenance.json")
    with open(provenance_path) as f:
        payload = json.load(f)
    provenance = payload.get("commands", payload)

    for command_name, corpus in expected_corpora.items():
        record = provenance[command_name]
        assert record["generated_count"] == len(corpus)
        assert record["fell_back"] is False
        assert set(record["persona_ids"]) == set(_FIXTURE_PERSONA_IDS)
        assert set(corpus) <= set(record["utterance_personas"]), (
            f"{command_name} regenerated utterances instead of using the R6 preseed"
        )


@pytest.fixture(scope="module")
def trained_hello_world(tmp_path_factory):
    if not _datasets_available():
        pytest.skip("datasets package not installed; intent-detection training is skipped.")

    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip(
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run synthetic "
            "utterance generation required for model training."
        )

    # Train into an isolated COPY of the example rather than the real
    # fastworkflow/examples/hello_world. _cleanup() rmtree's ___command_info at
    # both setup and teardown, which would otherwise destroy the real example's
    # trained model that other tests (e.g. test_fastapi_service.py) rely on being
    # pre-trained — a hidden inter-test dependency (see bd fix-0hb). The copy
    # excludes generated/runtime dirs so training starts clean.
    workflow_path = str(tmp_path_factory.mktemp("train_hello_world") / "hello_world")
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
    _cleanup(workflow_path, env_vars)

    fastworkflow.init(env_vars=env_vars)
    # try/finally so that a failure in setup (or in any test using this fixture) still
    # removes `./___workflow_contexts` from the repo root. bd fix-k0i.48.
    try:
        expected_corpora = _preseed_fixed_utterance_cache(workflow_path)
        train_workflow(workflow_path)
        _assert_fixed_corpus_was_used(workflow_path, expected_corpora)

        yield workflow_path, env_vars
    finally:
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


def test_benchmark_that_leaks_a_seed_utterance_fails_fast(tmp_path_factory):
    """A benchmark sharing a phrasing with the training seeds must abort the run (R1b).

    The check runs BEFORE training rather than alongside the scoring, and that ordering is
    the whole point: a leak discovered afterwards means every number the run produced has
    to be thrown away, having already spent the LLM calls and GPU time. It compares
    normalised text, so the realistic version of the mistake -- pasting a failing benchmark
    case into a seed list with the capitalisation changed -- is caught too.
    """
    if not _datasets_available():
        pytest.skip("datasets package not installed; training entry point is skipped.")

    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip("No real LITELLM_API_KEY_SYNDATA_GEN available; training is skipped.")

    workflow_path = str(tmp_path_factory.mktemp("leak_hello_world") / "hello_world")
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"
        ),
    )
    _cleanup(workflow_path, env_vars)
    fastworkflow.init(env_vars=env_vars)

    # try/finally, because `_cleanup` also removes `./___workflow_contexts` from the
    # REPO ROOT -- `train_workflow` creates it in the process's cwd. Leaving the cleanup
    # after the assertions meant one failure here littered the working tree, and the
    # next run inherited that directory. bd fix-k0i.48.
    try:
        cmd_dir = CommandDirectory.load(workflow_path)
        leaked = None
        for command_key in cmd_dir.get_utterance_keys():
            metadata = cmd_dir.get_utterance_metadata(command_key)
            if metadata and metadata.plain_utterances:
                leaked = metadata.plain_utterances[0]
                break
        assert leaked, "hello_world has no seed utterances to leak; test cannot run."

        benchmark_path = heldout_evaluation.default_benchmark_path(workflow_path)
        os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
        with open(benchmark_path, "w") as f:
            json.dump(
                {
                    "schema_version": heldout_evaluation.BENCHMARK_SCHEMA_VERSION,
                    "cases": [
                        {
                            "utterance": leaked,
                            "kind": "routing",
                            "expected_label": "add_two_numbers",
                            "context": "*",
                        }
                    ],
                },
                f,
            )

        with pytest.raises(heldout_evaluation.BenchmarkLeakError) as excinfo:
            train_workflow(workflow_path)
        assert leaked in str(excinfo.value)
    finally:
        _cleanup(workflow_path, env_vars)


def test_benchmark_routing_cases_are_scored(tmp_path_factory):
    """A valid benchmark must actually be scored, not silently skipped (R1b).

    Regression test for bd fix-588. The trainer called `benchmark_cases_for_context`
    without its required `kind` argument; the resulting TypeError was caught by the guard
    that exists to stop a scoring failure from destroying a completed training run, and
    became a note on the report. Routing from the persona holdout kept working, so the
    symptom was indistinguishable from "this workflow declares no benchmark cases".

    Every other test here either trains without a benchmark file or expects the leak
    guard to abort before scoring, which is exactly why nothing caught it. This one
    trains with a clean benchmark and asserts a score came back.
    """
    if not _datasets_available():
        pytest.skip("datasets package not installed; intent-detection training is skipped.")

    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip("No real LITELLM_API_KEY_SYNDATA_GEN available; training is skipped.")

    workflow_path = str(tmp_path_factory.mktemp("bench_hello_world") / "hello_world")
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"
        ),
    )
    _cleanup(workflow_path, env_vars)
    fastworkflow.init(env_vars=env_vars)

    # try/finally for the same reason as the leak test above: `_cleanup` removes
    # `./___workflow_contexts` from the repo root. bd fix-k0i.48.
    try:
        # Phrasings deliberately unlike hello_world's seeds, so the disjointness guard passes.
        benchmark_path = heldout_evaluation.default_benchmark_path(workflow_path)
        os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
        with open(benchmark_path, "w") as f:
            json.dump(
                {
                    "schema_version": heldout_evaluation.BENCHMARK_SCHEMA_VERSION,
                    "cases": [
                        {"utterance": "what is 41 plus 1", "kind": "routing",
                         "expected_label": "add_two_numbers", "context": "*"},
                        {"utterance": "please total 19 with 23", "kind": "routing",
                         "expected_label": "add_two_numbers", "context": "*"},
                    ],
                },
                f,
            )

        train_workflow(workflow_path)

        report_path = os.path.join(
            _command_info_path(workflow_path), heldout_evaluation.REPORT_FILENAME
        )
        with open(report_path) as f:
            payload = json.load(f)

        scored = [
            entry for entry in payload["contexts"]
            if (entry.get("benchmark_routing") or {}).get("total", 0) > 0
        ]
        assert scored, (
            "The benchmark's routing cases were never scored. Check the report notes for a "
            f"swallowed scoring failure: {payload}"
        )
        for entry in scored:
            benchmark_routing = entry["benchmark_routing"]
            assert benchmark_routing["total"] == 2
            assert benchmark_routing["top1_correct"] <= 2

        # The guard must not have fired: a note here means scoring raised and was swallowed.
        for entry in payload["contexts"]:
            assert not any(
                "evaluation failed" in note for note in entry.get("notes", [])
            ), f"Held-out evaluation raised and was swallowed: {entry['notes']}"
    finally:
        _cleanup(workflow_path, env_vars)


def test_training_produces_heldout_evaluation_report(trained_hello_world):
    """Training must emit a whole-persona held-out score, not just in-distribution F1.

    This is the epic's headline requirement (R1a). The module implementing it shipped
    before it was called by anything, so the trainer went on reporting only the
    same-distribution F1 -- a number computed on utterances from the personas it trained
    on. A report file whose routing score is populated is what proves the evaluation
    actually ran against the real CommandRouter path rather than being silently skipped.
    """
    workflow_path, _env_vars = trained_hello_world
    report_path = os.path.join(
        _command_info_path(workflow_path), heldout_evaluation.REPORT_FILENAME
    )
    assert os.path.isfile(report_path), (
        f"No held-out evaluation report at {report_path}; R1a is not wired into training."
    )

    with open(report_path) as f:
        payload = json.load(f)

    contexts = payload.get("contexts")
    assert contexts, f"Held-out evaluation report has no contexts: {payload}"

    # At least one context must have produced a real routing score. Held-out evaluation
    # degrades to notes-only when a workflow has too few personas to reserve any, so an
    # empty holdout is legitimate -- but hello_world generates enough that a total of
    # zero everywhere means the split or the scoring silently no-opped. Scoring failures
    # are deliberately swallowed during training so a completed run is never destroyed by
    # them, which is exactly why they need asserting on here.
    scored = [
        entry for entry in contexts
        if (entry.get("routing") or {}).get("total", 0) > 0
    ]
    assert scored, (
        "No context produced a held-out routing score. Either every persona split was "
        f"empty or scoring failed and was swallowed. Report: {payload}"
    )
    for entry in scored:
        routing = entry["routing"]
        assert routing["top1_correct"] <= routing["total"]
        assert 0.0 <= routing["top1"] <= 1.0

    assert payload["totals"]["routing_total"] > 0


def _routes_to_add_two_numbers(command_info: str, utterance: str) -> bool:
    """True when *utterance* reaches `add_two_numbers` in any trained context."""
    for model_dir in _find_model_dirs(command_info):
        router = CommandRouter(model_dir)
        labels = router.predict(utterance)
        if any("add_two_numbers" in label for label in labels):
            return True
    return False


def test_trained_model_routes_utterance(trained_hello_world):
    """A representative utterance must route to the add_two_numbers command.

    ROUTING_UTTERANCE is a training row (see `_fixed_generated_corpus`), so this is a
    memorisation/smoke check: it proves the artifacts load and the label reaches the
    encoder, not that the model generalises. The generalisation claim is
    `test_trained_model_routes_a_heldout_paraphrase` below; do not cite this one as
    routing-quality evidence. bd fix-k0i.30.
    """
    workflow_path, _env_vars = trained_hello_world
    command_info = _command_info_path(workflow_path)

    # The hello_world `add_two_numbers` command lives in the global context.
    model_dirs = _find_model_dirs(command_info)
    assert model_dirs, "No trained model directories to route against."

    assert _routes_to_add_two_numbers(command_info, ROUTING_UTTERANCE), (
        f"Utterance {ROUTING_UTTERANCE!r} did not route to 'add_two_numbers' in any "
        f"trained context under {command_info}."
    )


def test_the_heldout_probe_is_actually_held_out(trained_hello_world):
    """The probe below is only evidence if it is absent from everything trained on.

    Cheap and worth pinning: the natural way to "fix" a flaky generalisation test is to
    paste its probe into the fixed corpus, which silently converts it back into the
    memorisation check it was written to replace.
    """
    workflow_path, _env_vars = trained_hello_world
    trained_rows = set(_fixed_generated_corpus(ADD_TWO_NUMBERS_COMMAND))

    cmd_dir = CommandDirectory.load(workflow_path)
    for command_key in cmd_dir.get_utterance_keys():
        cmd_dir.ensure_command_hydrated(command_key)
        metadata = cmd_dir.get_utterance_metadata(command_key)
        trained_rows.update(_fixed_generated_corpus(command_key))
        trained_rows.update(metadata.plain_utterances)
        trained_rows.update(metadata.template_utterances)

    normalised = {row.strip().casefold() for row in trained_rows}
    assert HELDOUT_ROUTING_UTTERANCE.strip().casefold() not in normalised, (
        f"{HELDOUT_ROUTING_UTTERANCE!r} is in the training data, so routing it proves "
        f"only that the model memorised its training rows"
    )


def test_trained_model_routes_a_heldout_paraphrase(trained_hello_world):
    """The routing claim that is not memorisation.

    `HELDOUT_ROUTING_UTTERANCE` never appears in the preseeded corpus or in
    hello_world's seed list, so a regression that keeps reproducing training rows while
    losing every paraphrase fails here and nowhere else in this module. The fixed corpus
    stays fixed -- the determinism it buys is real (spec section 11 documents this test
    flaking on fresh LLM draws) -- and this probe is measured against it. bd fix-k0i.30.
    """
    workflow_path, _env_vars = trained_hello_world
    command_info = _command_info_path(workflow_path)

    model_dirs = _find_model_dirs(command_info)
    assert model_dirs, "No trained model directories to route against."

    assert _routes_to_add_two_numbers(command_info, HELDOUT_ROUTING_UTTERANCE), (
        f"Held-out paraphrase {HELDOUT_ROUTING_UTTERANCE!r} did not route to "
        f"'add_two_numbers' in any trained context under {command_info}. The model "
        f"reproduces its training rows but does not generalise."
    )
