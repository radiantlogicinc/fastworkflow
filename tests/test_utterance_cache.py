"""Integration tests for the generated-utterance cache (spec R6, decision D6).

The finding these cover is spec §11 M4: two training runs of
`examples/hello_world` at the same `TRAINING_SEED` produced **0 of 5** commands with
identical utterance sets, because the generator is a live LLM that no seed reaches.
Reuse of persisted utterances is what closes that gap, and the hazard reuse
introduces is the opposite one — training on stale data after a developer edits a
command's seed utterances. Most of what follows is about that hazard.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures
and no patching of fastWorkflow internals. The generator is exercised through its
real entry point with a locally defined completion backend and persona source, so
nothing here needs an API key or the network. The end-to-end "two training runs
agree" test lives in `test_utterance_cache_determinism.py`, which does need a key
and skips without one.

One structural note. The cache fingerprint records the completion backend and the
persona source by qualified name, precisely so an injected generator can never share
an entry with the real one. That means any two calls here that are SUPPOSED to share
an entry must go through the same `RecordingCompletion` class and the same
`_local_persona_dataset` function — hence the single backend class with a
`forbidden` switch, rather than a separate "this must not be called" class.

Historical note: the cache once offered cross-seed aggregation, sorting seeds
ascending, de-duplicating by first occurrence, and merging persona attribution.
That mode was unreachable through the shipped trainer and is intentionally absent;
the supported modes either reuse one exact seed or regenerate it.
"""

import contextlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import litellm

import fastworkflow
from fastworkflow.train import artifact_versioning
from fastworkflow.train.determinism import (
    COMMAND_INFO_FOLDERNAME,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    UtteranceProvenance,
)
from fastworkflow.train.generate_synthetic import (
    _apply_cached_entry,
    generate_diverse_utterances_with_provenance,
    utterance_fingerprint,
)
from fastworkflow.train.utterance_cache import (
    CACHE_DIRNAME,
    CACHE_FORMAT_VERSION,
    CACHE_MODES,
    CACHE_README_FILENAME,
    DEFAULT_CACHE_MODE,
    MODE_REGENERATE,
    MODE_REUSE,
    PRODUCTION_COMPLETION_BACKEND,
    PRODUCTION_PERSONA_SOURCE,
    UtteranceCache,
    UtteranceCacheEntry,
    callable_identity,
    compute_fingerprint,
    get_utterance_cache,
    normalize_mode,
    resolve_cache_mode,
    set_utterance_cache,
    source_digest,
)


COMMAND_NAME = "add_two_numbers"
SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]

LOCAL_PERSONAS = [
    {"persona": f"A person who is persona number {i}."} for i in range(64)
]

# Flips `_local_persona_dataset` into an assertion. Module-level state rather than a
# second loader function because the loader's qualified name is part of the cache
# key: swapping in a different function would change the key and turn a hit into a
# miss, which is the opposite of what the test is trying to observe.
_persona_loader_state = {"forbidden": False}


def _local_persona_dataset():
    """The persona source for every test in this module.

    `len()` plus `[i]['persona']` is the entire interface generate_synthetic uses.
    Real persona strings, no download.
    """
    if _persona_loader_state["forbidden"]:
        raise AssertionError(
            "PersonaHub was loaded even though a cache entry should have been reused"
        )
    return LOCAL_PERSONAS


@contextlib.contextmanager
def persona_loads_forbidden():
    """Assert that the persona source is not touched inside this block."""
    _persona_loader_state["forbidden"] = True
    try:
        yield
    finally:
        _persona_loader_state["forbidden"] = False


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _rate_limit_error() -> litellm.exceptions.RateLimitError:
    """A real litellm RateLimitError, constructed locally rather than mocked."""
    return litellm.exceptions.RateLimitError(
        message="rate limited by the test", llm_provider="test", model="test/model"
    )


class RecordingCompletion:
    """The one completion backend these tests inject.

    Derives its output from the PERSONA DESCRIPTION in the prompt, not from the
    `Persona_N` header: the header is a position within the batch and is identical
    across seeds, whereas the description differs because seeding changes which
    PersonaHub rows are selected. This keeps exact-seed reuse observable: two
    different seeds produce different text, and reuse must return only the requested
    seed's draw.

    `forbidden=True` makes any call an assertion failure, which is how a cache hit is
    proved. It is a flag on this class rather than a separate class because the class
    name is part of the cache fingerprint.
    """

    def __init__(self, tag: str = "line", forbidden: bool = False,
                 extra_lines: tuple[str, ...] = ()) -> None:
        self.tag = tag
        self.forbidden = forbidden
        self.extra_lines = tuple(extra_lines)
        self.calls = 0

    def __call__(self, **kwargs):
        if self.forbidden:
            raise AssertionError(
                "the LLM was called even though a cache entry should have been reused"
            )
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        header = prompt.split("[Persona_")[1]
        name = header.split("]")[0]
        description = header.split("]", 1)[1].strip().splitlines()[0]
        marker = description.rstrip(".").split()[-1]
        lines = list(self.extra_lines) + [
            f"{self.tag} a from persona {marker}",
            f"{self.tag} b from persona {marker}",
        ]
        return _llm_response(f"[Persona_{name}]\n" + "\n".join(lines) + "\n")


@pytest.fixture
def fast_retry_env():
    """Legacy fixture name: cache tests use successful local completions, so no retry tuning."""
    return None


@pytest.fixture
def clean_cache_sink():
    """Guarantee the process-wide cache sink is restored after each test."""
    previous = get_utterance_cache()
    set_utterance_cache(None)
    yield
    set_utterance_cache(previous)


@pytest.fixture
def workflow_dir(tmp_path):
    path = tmp_path / "workflow"
    path.mkdir()
    return str(path)


def _generate(**overrides):
    """Run the real generation path with local, network-free inputs."""
    kwargs = {
        "num_personas": 2,
        "utterances_per_persona": 2,
        "personas_per_batch": 1,
        "seed": 42,
        "persona_dataset_loader": _local_persona_dataset,
        "completion_fn": RecordingCompletion(),
    }
    kwargs |= overrides
    seed_utterances = kwargs.pop("seed_utterances", SEED_UTTERANCES)
    command_name = kwargs.pop("command_name", COMMAND_NAME)
    return generate_diverse_utterances_with_provenance(
        seed_utterances, command_name, **kwargs
    )


def _fingerprint(**overrides):
    """The fingerprint `_generate` would compute, for direct cache manipulation."""
    kwargs = {
        "seed_utterances": SEED_UTTERANCES,
        "command_name": COMMAND_NAME,
        "num_personas": 2,
        "utterances_per_persona": 2,
        "personas_per_batch": 1,
        "model": "mistral/mistral-small-latest",
        "completion_fn": None,
        "persona_dataset_loader": None,
    }
    kwargs |= overrides
    return utterance_fingerprint(
        kwargs["seed_utterances"],
        kwargs["command_name"],
        kwargs["num_personas"],
        kwargs["utterances_per_persona"],
        kwargs["personas_per_batch"],
        kwargs["model"],
        completion_fn=kwargs["completion_fn"],
        persona_dataset_loader=kwargs["persona_dataset_loader"],
    )


def _generated_only(utterances: list[str]) -> set[str]:
    """Drop the command-name token and the hand-written seeds."""
    return set(utterances) - {COMMAND_NAME, *SEED_UTTERANCES}


# ---------------------------------------------------------------------------
# Fingerprint: what must invalidate
# ---------------------------------------------------------------------------

def test_import_before_init_resolves_generation_counts_at_call_time():
    """Importing the generator must not freeze absent workflow config as None."""
    code = """
import re
from types import SimpleNamespace

import fastworkflow
from fastworkflow.train.generate_synthetic import (
    generate_diverse_utterances_with_provenance,
)

fastworkflow.init({
    "LLM_SYNDATA_GEN": "test/model",
    "LITELLM_API_KEY_SYNDATA_GEN": "test-key",
    "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS": "3",
    "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA": "2",
    "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH": "2",
})

def completion_fn(**kwargs):
    prompt = kwargs["messages"][0]["content"]
    names = re.findall(r"\\[(Persona_\\d+)\\]", prompt)
    content = "".join(
        f"[{name}]\\n{name} first phrasing\\n{name} second phrasing\\n"
        for name in names
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )

utterances, provenance = generate_diverse_utterances_with_provenance(
    ["do the thing"],
    "do_thing",
    seed=42,
    completion_fn=completion_fn,
    persona_dataset_loader=lambda: [
        {"persona": f"persona {index}"} for index in range(10)
    ],
)
assert provenance.generator_config["num_personas"] == 3
assert provenance.generator_config["utterances_per_persona"] == 2
assert provenance.generator_config["personas_per_batch"] == 2
assert provenance.generated_count == 6
assert len(utterances) == 8
"""
    env = os.environ.copy()
    for key in (
        "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS",
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA",
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH",
    ):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fingerprint_is_stable_for_identical_inputs():
    assert _fingerprint().variant_key == _fingerprint().variant_key


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"seed_utterances": ["add 2 and 3"]}, id="seed-removed"),
        pytest.param(
            {"seed_utterances": SEED_UTTERANCES + ["total these"]}, id="seed-added"
        ),
        pytest.param(
            {"seed_utterances": ["add 2 and 3", "sum these number"]}, id="seed-edited"
        ),
        pytest.param(
            {"seed_utterances": list(reversed(SEED_UTTERANCES))}, id="seed-reordered"
        ),
        pytest.param({"command_name": "subtract_two_numbers"}, id="command-name"),
        pytest.param({"num_personas": 3}, id="num-personas"),
        pytest.param({"utterances_per_persona": 3}, id="utterances-per-persona"),
        pytest.param({"personas_per_batch": 2}, id="personas-per-batch"),
        pytest.param({"model": "openai/gpt-4o-mini"}, id="model"),
        pytest.param(
            {"completion_fn": RecordingCompletion()}, id="completion-backend"
        ),
        pytest.param(
            {"persona_dataset_loader": _local_persona_dataset}, id="persona-source"
        ),
    ],
)
def test_fingerprint_changes_when_a_generation_input_changes(override):
    """Each of these changes what the LLM would produce, so each must miss.

    The seed-utterance rows are the ones that matter most: silently reusing
    generated text after the developer rewrote the seeds it was derived from is
    worse than having no cache at all.
    """
    assert _fingerprint(**override).variant_key != _fingerprint().variant_key


def test_fingerprint_covers_the_prompt_building_source():
    """A prompt edit must invalidate; the digest of the generator source is how."""
    baseline = _fingerprint()
    changed = compute_fingerprint(
        command_name=COMMAND_NAME,
        seed_utterances=SEED_UTTERANCES,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        model="mistral/mistral-small-latest",
        generator_source_digest="a-different-prompt",
    )
    assert changed.variant_key != baseline.variant_key
    assert baseline.inputs["generator_source_digest"] not in ("", None)


def test_fingerprint_covers_the_api_base_without_storing_it():
    """A proxy move is a generator change; the hostname itself is not recorded."""
    with_base = compute_fingerprint(
        command_name=COMMAND_NAME,
        seed_utterances=SEED_UTTERANCES,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        model="m",
        api_base="https://internal-proxy.example.invalid/v1",
    )
    without_base = compute_fingerprint(
        command_name=COMMAND_NAME,
        seed_utterances=SEED_UTTERANCES,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        model="m",
    )
    assert with_base.variant_key != without_base.variant_key
    assert "internal-proxy" not in json.dumps(with_base.inputs)


def test_production_backends_are_named_by_constants_not_qualnames():
    """A litellm upgrade must not invalidate every cache in the world."""
    assert callable_identity(None, PRODUCTION_COMPLETION_BACKEND) == (
        PRODUCTION_COMPLETION_BACKEND
    )
    assert callable_identity(None, PRODUCTION_PERSONA_SOURCE) == (
        PRODUCTION_PERSONA_SOURCE
    )
    assert "_local_persona_dataset" in callable_identity(
        _local_persona_dataset, PRODUCTION_PERSONA_SOURCE
    )


def test_callable_objects_are_identified_by_class_not_by_address():
    """Two instances must key the same, or a callable object would never hit."""
    first = callable_identity(RecordingCompletion(), PRODUCTION_COMPLETION_BACKEND)
    second = callable_identity(
        RecordingCompletion(tag="other"), PRODUCTION_COMPLETION_BACKEND
    )
    assert first == second
    assert "RecordingCompletion" in first
    assert "0x" not in first


def test_source_digest_is_stable_and_never_raises():
    digest = source_digest(_local_persona_dataset)
    assert digest == source_digest(_local_persona_dataset)
    assert digest != source_digest(_llm_response)
    # A builtin has no readable source; that must degrade, not explode.
    assert source_digest(len)


# ---------------------------------------------------------------------------
# Storage location and prune exemption
# ---------------------------------------------------------------------------

def test_cache_lives_beside_the_versions_directory(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    assert cache.root == os.path.join(
        workflow_dir, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME
    )
    # Reading must not create anything: inspecting an unbuilt workflow stays
    # read-only, matching artifact_versioning.command_info_root.
    assert not os.path.exists(cache.root)


def test_cache_dirname_is_exempt_from_the_stale_artifact_prune():
    """`_prune_stale_artifacts` skips RESERVED_TOPLEVEL_NAMES; the cache is in it.

    The constant is duplicated in artifact_versioning to keep that module free of
    heavy imports, so the two must be asserted to agree.
    """
    assert CACHE_DIRNAME == artifact_versioning.UTTERANCE_CACHE_DIRNAME
    assert CACHE_DIRNAME in artifact_versioning.RESERVED_TOPLEVEL_NAMES


def test_cache_is_never_a_valid_artifact_version_id():
    with pytest.raises(ValueError):
        artifact_versioning.version_dir("/tmp/nope", CACHE_DIRNAME)


def test_store_writes_the_expensive_to_regenerate_readme(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    cache.store(_fingerprint(), 42, ["one utterance"])
    readme = os.path.join(cache.root, CACHE_README_FILENAME)
    assert os.path.isfile(readme)
    with open(readme) as f:
        assert "fixed" in f.read()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_store_then_lookup_round_trips(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    assert cache.store(
        fingerprint,
        42,
        ["first phrasing", "second phrasing"],
        utterance_personas={"first phrasing": "3", "second phrasing": "17"},
        persona_ids=["3", "17"],
        persona_dataset_size=200_000,
    )

    entry = UtteranceCache(workflow_dir).lookup(fingerprint, 42)
    assert entry is not None
    assert entry.generated_utterances == ["first phrasing", "second phrasing"]
    assert entry.utterance_personas == {"first phrasing": "3", "second phrasing": "17"}
    assert entry.persona_ids == ["3", "17"]
    assert entry.persona_dataset_size == 200_000
    assert entry.seed == 42


def test_duplicate_utterances_survive_a_round_trip(workflow_dir):
    """Generation does not de-duplicate, so neither may reuse.

    A duplicate row is a second training row; dropping it on read would make a
    reused run train on less data than the run that produced the entry, which is
    exactly the byte-comparison R2 needs.
    """
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["same text", "other", "same text"])
    entry = UtteranceCache(workflow_dir).lookup(fingerprint, 42)
    assert entry.usable_utterances() == ["same text", "other", "same text"]


def test_lookup_misses_on_a_different_seed(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["only for 42"])
    assert cache.lookup(fingerprint, 43) is None
    assert cache.lookup(fingerprint, 42) is not None


def test_lookup_misses_when_seed_utterances_were_edited(workflow_dir):
    """The failure mode the fingerprint exists to prevent."""
    cache = UtteranceCache(workflow_dir)
    cache.store(_fingerprint(), 42, ["derived from the original seeds"])
    edited = _fingerprint(seed_utterances=["add 2 and 3", "please total these"])
    assert cache.lookup(edited, 42) is None


def test_entry_file_is_human_readable(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["a readable utterance"])

    with open(cache.entry_path(fingerprint)) as f:
        payload = json.load(f)
    assert payload["command_name"] == COMMAND_NAME
    assert payload["cache_format_version"] == CACHE_FORMAT_VERSION
    assert payload["fingerprint_inputs"]["seed_utterances"] == SEED_UTTERANCES
    assert payload["entries"]["42"]["generated_utterances"] == ["a readable utterance"]
    assert COMMAND_NAME in os.path.basename(cache.entry_path(fingerprint))


def test_a_qualified_command_name_produces_a_safe_filename(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint(command_name="IntentDetection/what_can_i_do")
    assert cache.store(fingerprint, 42, ["something"])
    name = os.path.basename(cache.entry_path(fingerprint))
    assert "/" not in name
    assert "IntentDetection_what_can_i_do" in name
    assert UtteranceCache(workflow_dir).lookup(fingerprint, 42) is not None


def test_two_seeds_accumulate_in_one_variant_file(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["from seed 42"])
    cache.store(fingerprint, 7, ["from seed 7"])

    reread = UtteranceCache(workflow_dir)
    assert reread.lookup(fingerprint, 42).generated_utterances == ["from seed 42"]
    assert reread.lookup(fingerprint, 7).generated_utterances == ["from seed 7"]
    # One variant file plus the README: seeds accumulate inside a file, not beside it.
    assert len(os.listdir(cache.root)) == 2


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_corrupt_entry_causes_a_miss_not_a_crash(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["will be clobbered"])

    with open(cache.entry_path(fingerprint), "w") as f:
        f.write("{ this is not json at all")

    reader = UtteranceCache(workflow_dir)
    assert reader.lookup(fingerprint, 42) is None
    assert reader.stats["unreadable"] == 1


def test_wrongly_shaped_entry_causes_a_miss(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["will be clobbered"])

    with open(cache.entry_path(fingerprint), "w") as f:
        json.dump({"entries": "not a mapping of entries"}, f)

    assert UtteranceCache(workflow_dir).lookup(fingerprint, 42) is None


def test_a_corrupt_entry_is_replaced_by_the_next_successful_store(workflow_dir):
    """Recovery must be automatic; nobody should have to delete a file by hand."""
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    with contextlib.suppress(OSError):
        os.makedirs(cache.root, exist_ok=True)
    with open(cache.entry_path(fingerprint), "w") as f:
        f.write("not json")

    assert cache.store(fingerprint, 42, ["freshly generated"])
    assert UtteranceCache(workflow_dir).lookup(
        fingerprint, 42
    ).generated_utterances == ["freshly generated"]


def test_entry_from_a_future_format_version_is_ignored(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["from a newer build"])

    path = cache.entry_path(fingerprint)
    with open(path) as f:
        payload = json.load(f)
    payload["cache_format_version"] = CACHE_FORMAT_VERSION + 1
    with open(path, "w") as f:
        json.dump(payload, f)

    assert UtteranceCache(workflow_dir).lookup(fingerprint, 42) is None


def test_entry_filed_under_the_wrong_key_is_ignored(workflow_dir):
    """A renamed or hand-edited file must not be trusted."""
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    cache.store(fingerprint, 42, ["belongs to another variant"])

    path = cache.entry_path(fingerprint)
    with open(path) as f:
        payload = json.load(f)
    payload["variant_key"] = "0" * 24
    with open(path, "w") as f:
        json.dump(payload, f)

    assert UtteranceCache(workflow_dir).lookup(fingerprint, 42) is None


def test_an_empty_utterance_set_is_never_stored_and_never_returned(workflow_dir):
    """A hit that returns [] would be F3 reached through the cache."""
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    assert cache.store(fingerprint, 42, []) is False
    assert cache.store(fingerprint, 42, ["   ", ""]) is False
    assert not os.path.exists(cache.entry_path(fingerprint))

    # Even a hand-written empty entry must not be handed back.
    cache.store(fingerprint, 42, ["real"])
    path = cache.entry_path(fingerprint)
    with open(path) as f:
        payload = json.load(f)
    payload["entries"]["42"]["generated_utterances"] = []
    with open(path, "w") as f:
        json.dump(payload, f)
    assert UtteranceCache(workflow_dir).lookup(fingerprint, 42) is None


def test_store_into_an_unwritable_location_does_not_raise(tmp_path):
    """A read-only artifact tree must degrade to "no cache", not fail the run."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    # A regular FILE where ___command_info must be a directory, so makedirs raises
    # an OSError subclass rather than succeeding.
    (blocked / COMMAND_INFO_FOLDERNAME).write_text("not a directory")

    cache = UtteranceCache(str(blocked))
    assert cache.store(_fingerprint(), 42, ["something"]) is False
    assert cache.stats["write_failed"] == 1


def test_lookup_on_an_absent_cache_is_a_plain_miss(workflow_dir):
    cache = UtteranceCache(workflow_dir)
    assert cache.lookup(_fingerprint(), 42) is None
    assert cache.stats["miss"] == 1
    assert cache.stats["hit"] == 0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_mode_defaults_to_reuse_and_normalizes_unknown_values():
    assert normalize_mode(None) == DEFAULT_CACHE_MODE == MODE_REUSE
    assert CACHE_MODES == frozenset({MODE_REUSE, MODE_REGENERATE})
    assert normalize_mode("nonsense-mode") == DEFAULT_CACHE_MODE


def test_mode_is_not_configurable_from_env_file_or_shell(monkeypatch):
    """The public CLI passes regeneration directly; environment values are ignored."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init(previous)
        monkeypatch.delenv("UTTERANCE_CACHE_MODE", raising=False)
        assert resolve_cache_mode() == MODE_REUSE

        monkeypatch.setenv("UTTERANCE_CACHE_MODE", "regenerate")
        assert resolve_cache_mode() == MODE_REUSE

        fastworkflow.init({**previous, "UTTERANCE_CACHE_MODE": "off"})
        assert resolve_cache_mode() == MODE_REUSE
    finally:
        fastworkflow.init(previous)


def test_regenerate_mode_ignores_an_entry_but_refreshes_it(workflow_dir):
    stale = UtteranceCache(workflow_dir, mode=MODE_REUSE)
    fingerprint = _fingerprint()
    stale.store(fingerprint, 42, ["the stale draw"])

    refreshing = UtteranceCache(workflow_dir, mode=MODE_REGENERATE)
    assert refreshing.lookup(fingerprint, 42) is None
    assert refreshing.store(fingerprint, 42, ["the fresh draw"]) is True

    assert UtteranceCache(workflow_dir).lookup(
        fingerprint, 42
    ).generated_utterances == ["the fresh draw"]


# ---------------------------------------------------------------------------
# Aggregation across seeds (R6)
# ---------------------------------------------------------------------------

# Ascending seed order, first occurrence wins, so the result is reproducible.


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_writers_do_not_corrupt_an_entry(workflow_dir):
    """Many threads, one variant file: it must always parse afterwards.

    Different seeds so the merge-forward path is exercised. The guarantee is "never
    corrupt", not "never lose an update" — a lost update means one command
    regenerates next run, which is the safe direction.
    """
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    errors: list[BaseException] = []

    def write(seed: int) -> None:
        try:
            for _ in range(5):
                cache.store(fingerprint, seed, [f"utterance from seed {seed}"])
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    reread = UtteranceCache(workflow_dir)
    for seed in range(8):
        entry = reread.lookup(fingerprint, seed)
        assert entry is not None, f"seed {seed} was lost"
        assert entry.generated_utterances == [f"utterance from seed {seed}"]
    # No temporaries left behind by the atomic replaces.
    assert not [name for name in os.listdir(cache.root) if ".tmp-" in name]


def test_separate_cache_instances_writing_different_variants_coexist(workflow_dir):
    """Two commands generated in parallel write to different files by construction."""
    first = UtteranceCache(workflow_dir)
    second = UtteranceCache(workflow_dir)
    fp_a = _fingerprint(command_name="alpha")
    fp_b = _fingerprint(command_name="beta")

    assert first.entry_path(fp_a) != second.entry_path(fp_b)
    first.store(fp_a, 42, ["alpha text"])
    second.store(fp_b, 42, ["beta text"])

    reread = UtteranceCache(workflow_dir)
    assert reread.lookup(fp_a, 42).generated_utterances == ["alpha text"]
    assert reread.lookup(fp_b, 42).generated_utterances == ["beta text"]


# ---------------------------------------------------------------------------
# The generation path: reuse, invalidation, and degradation together
# ---------------------------------------------------------------------------

def test_second_generation_reuses_the_first_without_calling_the_llm(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """The acceptance property, at the level of one command.

    Run one generates; run two must return identical utterances and identical
    provenance without touching either the LLM or PersonaHub.
    """
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    completion = RecordingCompletion()
    first_utterances, first_provenance = _generate(completion_fn=completion)
    assert completion.calls > 0
    assert first_provenance.fell_back is False
    assert cache.stats["stored"] == 1

    with persona_loads_forbidden():
        second_utterances, second_provenance = _generate(
            completion_fn=RecordingCompletion(forbidden=True)
        )

    assert second_utterances == first_utterances
    assert second_provenance.persona_ids == first_provenance.persona_ids
    assert second_provenance.utterance_personas == first_provenance.utterance_personas
    assert second_provenance.generated_count == first_provenance.generated_count
    assert second_provenance.final_count == first_provenance.final_count
    assert second_provenance.fell_back is False
    assert cache.stats["hit"] == 1


def test_editing_a_seed_utterance_forces_regeneration(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """Stale reuse after a seed edit is worse than no cache; prove it cannot happen."""
    set_utterance_cache(UtteranceCache(workflow_dir))

    original = RecordingCompletion()
    _generate(completion_fn=original)
    assert original.calls > 0

    after_edit = RecordingCompletion(forbidden=True)
    with pytest.raises(AssertionError, match="cache entry should have been reused"):
        _generate(
            seed_utterances=["add 2 and 3", "please total these"],
            completion_fn=after_edit,
        )


def test_the_cached_prefix_is_rebuilt_from_live_seeds_not_from_the_cache(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """Only generated text is cached; the command name and seeds are re-derived.

    A hit therefore cannot resurrect a seed utterance that has been deleted from the
    command file — the fingerprint would miss first, but the storage shape means it
    is impossible even if it somehow did not.
    """
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)
    _generate()

    fingerprint = _fingerprint(
        completion_fn=RecordingCompletion(),
        persona_dataset_loader=_local_persona_dataset,
    )
    with open(cache.entry_path(fingerprint)) as f:
        payload = json.load(f)
    stored = payload["entries"]["42"]["generated_utterances"]
    assert stored
    assert COMMAND_NAME not in stored
    for seed_utterance in SEED_UTTERANCES:
        assert seed_utterance not in stored


def test_a_rate_limited_command_is_not_cached_and_still_falls_back_to_seeds(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """Wave 1's F3 fix must survive, and degraded data must not be frozen in."""
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    def always_rate_limited(**_kwargs):
        raise _rate_limit_error()

    utterances, provenance = _generate(completion_fn=always_rate_limited)
    assert utterances == [COMMAND_NAME] + SEED_UTTERANCES
    assert provenance.fell_back is True
    assert cache.stats["stored"] == 0
    assert not [
        name for name in os.listdir(cache.root) if name.endswith(".json")
    ] if os.path.isdir(cache.root) else True

    # The next run must try again rather than reuse the degraded set.
    second_utterances, second_provenance = _generate(
        completion_fn=always_rate_limited
    )
    assert second_utterances == [COMMAND_NAME] + SEED_UTTERANCES
    assert second_provenance.fell_back is True


def test_a_partial_batch_failure_is_not_cached(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """Keeping what earlier batches produced is right; persisting it is not."""
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    def one_good_then_rate_limited(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        index = int(prompt.split("[Persona_")[1].split("]")[0])
        if index > 1:
            raise _rate_limit_error()
        return _llm_response("[Persona_1]\nfirst batch alpha\nfirst batch beta\n")

    utterances, provenance = _generate(
        num_personas=3, completion_fn=one_good_then_rate_limited
    )
    assert provenance.fell_back is True
    assert "first batch alpha" in utterances
    assert cache.stats["stored"] == 0


def test_generation_without_an_installed_cache_is_unchanged(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """A workflow author calling the generator directly gets the old behaviour."""
    assert get_utterance_cache() is None
    completion = RecordingCompletion()
    utterances, provenance = _generate(completion_fn=completion)
    assert completion.calls > 0
    assert provenance.fell_back is False
    assert utterances[: 1 + len(SEED_UTTERANCES)] == [COMMAND_NAME] + SEED_UTTERANCES
    assert not os.path.exists(
        os.path.join(workflow_dir, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME)
    )


def test_seed_duplicated_by_a_cached_utterance_stays_attributed_to_the_seed(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """Attribution is re-derived on a hit, so seed absorption still happens.

    Copying the stored map verbatim would leave the duplicate attributed to a
    persona, and a whole-persona holdout would score a memorised row as a
    generalisation success.
    """
    set_utterance_cache(UtteranceCache(workflow_dir))

    _generate(
        num_personas=1,
        completion_fn=RecordingCompletion(extra_lines=(SEED_UTTERANCES[0],)),
    )
    with persona_loads_forbidden():
        _utterances, provenance = _generate(
            num_personas=1, completion_fn=RecordingCompletion(forbidden=True)
        )
    assert provenance.utterance_personas[SEED_UTTERANCES[0]] == SEED_PERSONA_ID


def test_a_cached_entry_without_attribution_degrades_conservatively():
    """An entry missing its persona map must not pin text to one persona."""
    entry = UtteranceCacheEntry(
        seed=42,
        generated_utterances=["unattributed text"],
        persona_ids=["3", "9"],
    )
    provenance = UtteranceProvenance(command_name=COMMAND_NAME, seed=42)
    applied = _apply_cached_entry(provenance, entry)

    assert applied == ["unattributed text"]
    attribution = provenance.utterance_personas["unattributed text"]
    assert attribution.startswith(UNRESOLVED_PERSONA_PREFIX)
    assert "3" in attribution and "9" in attribution


def test_reuse_mode_returns_only_the_requested_seed(
    workflow_dir, clean_cache_sink, fast_retry_env
):
    """The default must return only this seed's draw, never another cached seed."""
    set_utterance_cache(UtteranceCache(workflow_dir, mode=MODE_REUSE))

    at_42, _ = _generate(seed=42)
    at_7, _ = _generate(seed=7)
    assert _generated_only(at_42) != _generated_only(at_7)
    assert not _generated_only(at_42) <= _generated_only(at_7)


def test_cache_summary_reports_reuse(workflow_dir, clean_cache_sink, fast_retry_env):
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)
    _generate()
    with persona_loads_forbidden():
        _generate(completion_fn=RecordingCompletion(forbidden=True))

    summary = cache.format_summary()
    assert "1 reused" in summary
    assert "1 written" in summary
    assert cache.root in summary
