""""Fail before spending" has to mean before the first paid call (bd fix-k0i.16).

R1b's benchmark-leak check aborts a training run whose benchmark shares a phrasing with
the training seeds, because such a benchmark measures memorisation. The check lived inside
`model_pipeline_training.train`, which `train_workflow` reaches only *after*
`_generate_dspy_examples_helper` has already made real DSPy parameter-example calls for
every command with parameters. The regression test for the leak used `pytest.raises` and so
was position-blind: it passed while the run was already spending.

The old "already up to date" early return skipped the check entirely, which was benign only
because nothing retrained on that path — the leak then went unreported until something did.

No mocks (`.cursor/rules/testing_rules.mdc`). These tests copy the real
`tests/hello_world_workflow`, write a real benchmark file at the real default path, and
call the shipped `preflight_benchmark` and `train_workflow`. No LLM key is needed, and that
is itself the point: if the run needed a key to reach the abort, the abort would be too
late.
"""

import importlib.util
import inspect
import json
import os
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.model_pipeline_training import preflight_benchmark
from fastworkflow.train import heldout_evaluation
from fastworkflow.train.__main__ import train_workflow

HELLO_WORLD_PATH = os.path.join("tests", "hello_world_workflow")


def _resolve_env_vars() -> dict:
    env_vars = {
        **dotenv_values(os.path.join("fastworkflow", "examples", "fastworkflow.env")),
        **dotenv_values(
            os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
        ),
    }
    for override in (os.path.join("env", ".env"), os.path.join("passwords", ".env")):
        if os.path.exists(override):
            env_vars.update(dotenv_values(override))
    return env_vars


@pytest.fixture(scope="module", autouse=True)
def _initialised_fastworkflow():
    fastworkflow.init(env_vars=_resolve_env_vars())


@pytest.fixture
def hello_world(tmp_path: Path) -> str:
    workflow_path = str(tmp_path / "hello_world")
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"
        ),
    )
    return workflow_path


def _first_seed_utterance(workflow_path: str) -> str:
    cmd_dir = CommandDirectory.load(workflow_path)
    for command_key in cmd_dir.get_utterance_keys():
        metadata = cmd_dir.get_utterance_metadata(command_key)
        if metadata and metadata.plain_utterances:
            return metadata.plain_utterances[0]
    pytest.fail("hello_world has no seed utterances to leak; the test cannot run")


def _write_benchmark(workflow_path: str, utterance: str, kind: str = "routing") -> str:
    path = heldout_evaluation.default_benchmark_path(workflow_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": heldout_evaluation.BENCHMARK_SCHEMA_VERSION,
                "cases": [
                    {
                        "utterance": utterance,
                        "kind": kind,
                        "expected_label": "add_two_numbers",
                        "context": "*",
                    }
                ],
            },
            f,
        )
    return path


def _param_label_files(workflow_path: str) -> list[str]:
    """The artifacts `_generate_dspy_examples_helper` writes — one paid LLM draw each."""
    command_info = os.path.join(workflow_path, "___command_info")
    if not os.path.isdir(command_info):
        return []
    return sorted(
        name for name in os.listdir(command_info) if name.endswith("_param_labeled.json")
    )


# ---------------------------------------------------------------------------
# The preflight in isolation
# ---------------------------------------------------------------------------


def test_preflight_needs_no_workflow_instance_no_model_and_no_llm(hello_world: str):
    """It has to be callable before anything expensive exists, or it cannot run first.

    Off the command directory and routing definition alone: that constraint is what makes
    hoisting it above the paid step possible at all.
    """
    cases = preflight_benchmark(hello_world)
    assert cases == [], "hello_world ships no benchmark file"

    clean = "please add seventeen and twenty five together for me right now"
    _write_benchmark(hello_world, clean)
    cases = preflight_benchmark(hello_world)
    assert len(cases) == 1
    assert cases[0].utterance == clean


def test_preflight_raises_on_a_leaked_seed_phrasing(hello_world: str):
    leaked = _first_seed_utterance(hello_world)
    _write_benchmark(hello_world, leaked)

    with pytest.raises(heldout_evaluation.BenchmarkLeakError) as excinfo:
        preflight_benchmark(hello_world)
    assert leaked in str(excinfo.value)


def test_preflight_catches_a_recapitalised_leak(hello_world: str):
    """The realistic version of the mistake: pasting a failing case in and editing the case."""
    leaked = _first_seed_utterance(hello_world)
    _write_benchmark(hello_world, leaked.upper())

    with pytest.raises(heldout_evaluation.BenchmarkLeakError):
        preflight_benchmark(hello_world)


# ---------------------------------------------------------------------------
# The ordering property, through the shipped orchestrator
# ---------------------------------------------------------------------------


def test_a_leaked_benchmark_aborts_before_any_paid_parameter_example_call(
    hello_world: str,
):
    """The finding itself, asserted by position rather than by exception type alone.

    `_param_labeled.json` is written once per command with parameters, immediately after
    the DSPy draw that produced it, so its absence is direct evidence that no paid call was
    made. Needing no API key to reach the abort is the other half of the same evidence.
    """
    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets package not installed; train_workflow returns before the "
                    "pre-flight and there is no ordering to assert.")

    assert _param_label_files(hello_world) == [], "precondition"
    leaked = _first_seed_utterance(hello_world)
    _write_benchmark(hello_world, leaked)

    with pytest.raises(heldout_evaluation.BenchmarkLeakError):
        train_workflow(hello_world)

    assert _param_label_files(hello_world) == [], (
        "parameter examples were generated before the leak aborted the run; the leak "
        "check is still downstream of the money"
    )


def test_the_leak_is_reported_even_when_nothing_needs_retraining(hello_world: str):
    """The second half of fix-k0i.16: the no-op early return used to skip the check.

    Reaching the real early return needs a trained workflow, so this asserts the property
    that makes it impossible to skip — the check sits above every return in
    `train_workflow`, including the one for an up-to-date workflow. A leak that only
    surfaces once something happens to need retraining surfaces at the worst moment.
    """
    source = inspect.getsource(train_workflow)
    preflight_at = source.index("preflight_benchmark(")
    paid_at = source.index("_generate_dspy_examples_helper(")
    noop_return_at = source.index("_repair_noop_publication(")

    assert preflight_at < paid_at, "the benchmark check must precede DSPy generation"
    assert preflight_at < noop_return_at, (
        "the benchmark check must precede the already-up-to-date early return"
    )


def test_a_clean_benchmark_does_not_stop_the_run(hello_world: str):
    """The guard must not fire on a benchmark that is doing its job.

    Training itself needs a key and is not attempted here; what matters is that the
    pre-flight lets the run continue past it.
    """
    _write_benchmark(
        hello_world, "could you total up eleven plus four hundred and two"
    )
    cases = preflight_benchmark(hello_world)
    assert len(cases) == 1
