"""The pilot's cache report: keys from the trainer's own functions, lookups only, no LLM, no training.

Runs on a private copy of messaging_app_4 with a tiny real cache file written through the
cache's own store(). Nothing calls a model.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.train import generate_synthetic as gs
from fastworkflow.train import not_here_pilot as pilot
from fastworkflow.train import utterance_cache as uc

MESSAGING = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "messaging_app_4"
PERSONAS, PER_PERSONA, PER_BATCH = 4, 5, 1
ENV = {
    "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS": str(PERSONAS),
    "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA": str(PER_PERSONA),
    "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH": str(PER_BATCH),
    "LLM_SYNDATA_GEN": "mistral/unit-test-model",
    "LITELLM_API_KEY_SYNDATA_GEN": "unit-test-key-unused",
}


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init(dict(ENV))
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def copy_of_messaging(tmp_path) -> str:
    destination = tmp_path / "messaging_copy"
    shutil.copytree(MESSAGING, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    yield str(destination)
    RoutingRegistry.clear_registry()


def _cache_files(folder: str) -> set[str]:
    base = Path(folder) / "___command_info"
    found = set()
    for sub in ("utterance_cache", "param_example_cache"):
        if (base / sub).is_dir():
            found |= {str(p) for p in (base / sub).iterdir()}
    return found


def test_an_empty_cache_is_all_misses_and_the_estimates_follow_the_batching(copy_of_messaging):
    report = pilot.cache_report(copy_of_messaging)

    utterance = report["utterance_cache"]
    assert utterance["commands"] > 0
    assert utterance["hits"] == 0
    assert len(utterance["misses"]) == utterance["commands"]
    per_miss = math.ceil(PERSONAS / PER_BATCH)
    assert all(m["estimated_llm_calls"] == per_miss for m in utterance["misses"])
    assert utterance["estimated_llm_calls"] == per_miss * utterance["commands"]
    assert all(m["differs"] == "no stored variant for this command" for m in utterance["misses"])

    param = report["param_example_cache"]
    assert param["hits"] == 0
    assert param["estimated_llm_calls"] == len(param["misses"])
    assert report["estimated_llm_calls_total"] == (
        utterance["estimated_llm_calls"] + param["estimated_llm_calls"])


def test_a_stored_variant_with_other_inputs_is_reported_by_field(copy_of_messaging):
    first = pilot.cache_report(copy_of_messaging)["utterance_cache"]["misses"][0]["command"]
    cmd_dir = pilot.prepare_registry(copy_of_messaging)
    seeds = list(cmd_dir.get_utterance_metadata(first).plain_utterances)
    current = gs.utterance_fingerprint(seeds, first, PERSONAS, PER_PERSONA, PER_BATCH, ENV["LLM_SYNDATA_GEN"])
    # the same inputs in every field but one: the report must name exactly that field
    older_inputs = {**current.inputs, "model": "some/older-model"}
    cache_dir = Path(copy_of_messaging) / "___command_info" / "utterance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_key = "0" * uc.VARIANT_KEY_LENGTH  # stored variants are filed under a hex key of that length
    (cache_dir / f"{uc.slugify(first)}.{old_key}.json").write_text(json.dumps({
        "command_name": first,
        "variant_key": old_key,
        "fingerprint_inputs": older_inputs,
    }))

    misses = {m["command"]: m for m in pilot.cache_report(copy_of_messaging)["utterance_cache"]["misses"]}

    assert misses[first]["differs"] == (
        f'differs from {uc.slugify(first)}.{old_key}.json in model: "some/older-model" -> '
        f'"{ENV["LLM_SYNDATA_GEN"]}"')


def test_a_stored_entry_written_by_the_cache_itself_is_a_hit(copy_of_messaging):
    first = pilot.cache_report(copy_of_messaging)["utterance_cache"]["misses"][0]["command"]
    cmd_dir = pilot.prepare_registry(copy_of_messaging)
    seeds = list(cmd_dir.get_utterance_metadata(first).plain_utterances)
    fingerprint = gs.utterance_fingerprint(seeds, first, PERSONAS, PER_PERSONA, PER_BATCH, ENV["LLM_SYNDATA_GEN"])
    cache = uc.UtteranceCache(copy_of_messaging, mode=uc.MODE_REUSE)
    seed = gs.get_training_seed()
    assert cache.store(fingerprint, seed, [f"{first} sample one", f"{first} sample two"],
                       utterance_personas={}, persona_ids=[], persona_dataset_size=PERSONAS)

    report = pilot.cache_report(copy_of_messaging)

    assert report["utterance_cache"]["hits"] == 1
    assert first not in {m["command"] for m in report["utterance_cache"]["misses"]}


def test_the_report_writes_nothing_into_the_caches(copy_of_messaging):
    before = _cache_files(copy_of_messaging)

    pilot.cache_report(copy_of_messaging)

    assert _cache_files(copy_of_messaging) == before


def test_fill_and_report_refuse_without_their_flags(tmp_path):
    import subprocess
    import sys

    scratch = tmp_path / "scratch"
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.train.not_here_pilot", "--scratch", str(scratch),
         "--fill-cache-only", "--cache-from", str(tmp_path), "--env-file", "x.env"],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "allow-llm-for-fill" in result.stderr
    assert not scratch.exists()
