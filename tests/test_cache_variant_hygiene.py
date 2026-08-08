"""Integration tests for generated-data cache ergonomics (bd fix-k0i.36).

Two costs the caches imposed on whoever maintains a workflow:

1. **A miss never said why.** The counters recorded that reuse did not happen; nothing
   said which fingerprint input diverged. Each entry stores `fingerprint_inputs`
   precisely so that question is answerable, so answering it by hand-diffing JSON was a
   choice, not a limitation. Now a miss logs the diverging inputs at INFO.

2. **Orphaned variants grew without bound.** Every edit to a command's seed utterances,
   model string, parameter model or prompt code mints a new variant key and therefore a
   new file, and `_prune_stale_artifacts` deliberately skips these directories. On an
   actively-tuned 160-command workflow that accumulated forever. A write now prunes that
   command's older variants down to the most recent few.

Both mechanisms are shared by the two generated-data caches rather than implemented
twice, so both are exercised through both caches here.

The dangerous half of a retention policy is deleting the wrong file, so most of what
follows is about the boundary: never another command's data, never a file this module
did not write, never the file just written, and never an exception that could fail a
training run.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: real cache objects,
real files on disk, no Mock fixtures. Log assertions read real `LogRecord`s off the real
`fastWorkflow` logger through a stdlib handler, because that logger does not propagate
to root and so is invisible to `caplog`.
"""

import json
import logging
import os
import time

import pytest

import fastworkflow
from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME
from fastworkflow.train.param_example_cache import ParamExampleCache
from fastworkflow.train.param_example_cache import (
    compute_fingerprint as param_compute_fingerprint,
)
from fastworkflow.train.utterance_cache import (
    MAX_VARIANTS_PER_COMMAND,
    MODE_REGENERATE,
    MODE_REUSE,
    VARIANT_KEY_LENGTH,
    UtteranceCache,
    compute_fingerprint,
    describe_fingerprint_divergence,
    prune_orphaned_variants,
    slugify,
    stored_variants,
)
from fastworkflow.utils.logging import logger as fastworkflow_logger

COMMAND_NAME = "add_two_numbers"
SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]
MODEL = "mistral/mistral-small-latest"


class _LogSink(logging.Handler):
    """Collects real `LogRecord`s off the real, non-propagating fastWorkflow logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def log_sink():
    sink = _LogSink()
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(logging.DEBUG)
    fastworkflow_logger.addHandler(sink)
    try:
        yield sink
    finally:
        fastworkflow_logger.removeHandler(sink)
        fastworkflow_logger.setLevel(previous_level)


@pytest.fixture(autouse=True)
def training_env():
    previous = dict(fastworkflow._env_vars)
    fastworkflow.init({**previous, "LLM_SYNDATA_GEN": MODEL, "TRAINING_SEED": "42"})
    yield
    fastworkflow.init(previous)


@pytest.fixture
def workflow_dir(tmp_path):
    path = tmp_path / "workflow"
    path.mkdir()
    return str(path)


def _fingerprint(**overrides):
    kwargs = {
        "command_name": COMMAND_NAME,
        "seed_utterances": list(SEED_UTTERANCES),
        "num_personas": 2,
        "utterances_per_persona": 2,
        "personas_per_batch": 1,
        "model": MODEL,
    }
    kwargs |= overrides
    return compute_fingerprint(**kwargs)


def _param_fingerprint(**overrides):
    kwargs = {
        "command_name": COMMAND_NAME,
        "field_annotations_text": "first_number: float",
        "field_details": [{"name": "first_number", "type": "float"}],
        "num_examples": 3,
        "validation_threshold": 0.3,
        "temperature": 0.9,
        "model": MODEL,
    }
    kwargs |= overrides
    return param_compute_fingerprint(**kwargs)


def _param_example(command: str = "add 1 and 2") -> list[dict]:
    return [{"fields": {"command": command}, "inputs": ["command"]}]


def _stored_names(cache) -> list[str]:
    return sorted(
        name for name in os.listdir(cache.root) if name.endswith(".json")
    )


# ---------------------------------------------------------------------------
# A miss must say which input diverged
# ---------------------------------------------------------------------------

def test_a_seed_utterance_edit_is_named_as_the_reason_for_the_miss(
    workflow_dir, log_sink
):
    """The most common and most expensive reason reuse stops happening."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(), 42, ["derived from the original seeds"])
    log_sink.records.clear()

    edited = _fingerprint(seed_utterances=["add 2 and 3", "please total these"])
    assert cache.lookup(edited, 42) is None

    messages = log_sink.messages(logging.INFO)
    assert len(messages) == 1, messages
    assert COMMAND_NAME in messages[0]
    assert "seed_utterances" in messages[0]


@pytest.mark.parametrize(
    ("override", "expected_key"),
    [
        pytest.param({"model": "openai/gpt-4o-mini"}, "model", id="model"),
        pytest.param({"num_personas": 9}, "num_personas", id="num-personas"),
        pytest.param(
            {"utterances_per_persona": 9},
            "utterances_per_persona",
            id="utterances-per-persona",
        ),
        pytest.param(
            {"generator_source_digest": "an-edited-prompt"},
            "generator_source_digest",
            id="prompt-source",
        ),
    ],
)
def test_each_diverging_input_is_named_by_its_own_key(
    workflow_dir, log_sink, override, expected_key
):
    """"Why did this regenerate?" has one answer per input, and each must be reachable."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(), 42, ["the original draw"])
    log_sink.records.clear()

    assert cache.lookup(_fingerprint(**override), 42) is None
    messages = log_sink.messages(logging.INFO)
    assert len(messages) == 1, messages
    assert expected_key in messages[0]


def test_the_report_shows_the_old_value_and_the_new_one(workflow_dir, log_sink):
    """A key name alone still leaves the developer looking the values up."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(), 42, ["the original draw"])
    log_sink.records.clear()

    assert cache.lookup(_fingerprint(model="openai/gpt-4o-mini"), 42) is None
    message = log_sink.messages(logging.INFO)[0]
    assert MODEL in message
    assert "openai/gpt-4o-mini" in message


def test_the_report_compares_against_the_closest_variant_not_the_newest(
    workflow_dir, log_sink
):
    """The useful answer is the smallest one.

    A workflow that has been tuned for a while has several stored variants. Diffing
    against the most recently written one would report every input that has changed
    since, when what the developer needs is the single edit that cost them this run's
    reuse.
    """
    cache = UtteranceCache(workflow_dir)
    # Far away in three inputs, and written LAST so recency and closeness disagree.
    assert cache.store(_fingerprint(model="openai/gpt-4o-mini"), 42, ["one"])
    assert cache.store(_fingerprint(), 42, ["two"])
    time.sleep(0.01)
    assert cache.store(
        _fingerprint(
            model="anthropic/claude", num_personas=99, utterances_per_persona=99
        ),
        42,
        ["three"],
    )
    log_sink.records.clear()

    assert cache.lookup(_fingerprint(num_personas=3), 42) is None
    message = log_sink.messages(logging.INFO)[0]
    assert "num_personas" in message
    assert "utterances_per_persona" not in message


def test_a_matching_variant_with_a_missing_seed_says_which_seeds_exist(
    workflow_dir, log_sink
):
    """A different kind of miss, and a different useful answer.

    Nothing about the configuration changed; this seed was simply never generated. A
    fingerprint-input diff would report no differences at all and read as a bug.
    """
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    assert cache.store(fingerprint, 42, ["only for 42"])
    assert cache.store(fingerprint, 7, ["only for 7"])
    log_sink.records.clear()

    assert cache.lookup(fingerprint, 99) is None
    message = log_sink.messages(logging.INFO)[0]
    assert "seed 99" in message
    assert "42" in message and "7" in message


def test_a_first_run_reports_no_divergence(workflow_dir, log_sink):
    """160 commands each logging "nothing cached yet" would bury the lines that matter."""
    cache = UtteranceCache(workflow_dir)
    assert cache.lookup(_fingerprint(), 42) is None
    assert cache.stats["miss"] == 1
    assert log_sink.messages(logging.INFO) == []


def test_regenerate_mode_reports_no_divergence(workflow_dir, log_sink):
    """Reads are off by request, so there is no divergence to explain."""
    seeded = UtteranceCache(workflow_dir, mode=MODE_REUSE)
    assert seeded.store(_fingerprint(), 42, ["the stale draw"])

    regenerating = UtteranceCache(workflow_dir, mode=MODE_REGENERATE)
    log_sink.records.clear()
    assert regenerating.lookup(_fingerprint(), 42) is None
    assert regenerating.stats["miss"] == 1
    assert log_sink.messages(logging.INFO) == []


def test_another_commands_variant_is_never_offered_as_the_explanation(
    workflow_dir, log_sink
):
    """Reporting a different command's inputs would be worse than reporting nothing."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(command_name="subtract_two_numbers"), 42, ["other"])
    log_sink.records.clear()

    assert cache.lookup(_fingerprint(), 42) is None
    assert log_sink.messages(logging.INFO) == []


def test_the_parameter_example_cache_explains_its_misses_too(workflow_dir, log_sink):
    """Both summaries are printed by the same run; one of them being mute is a gap."""
    cache = ParamExampleCache(workflow_dir)
    assert cache.store(_param_fingerprint(), 42, _param_example())
    log_sink.records.clear()

    edited = _param_fingerprint(
        field_annotations_text="first_number: float\ncomment: str"
    )
    assert cache.lookup(edited, 42) is None
    messages = log_sink.messages(logging.INFO)
    assert len(messages) == 1, messages
    assert "field_annotations_digest" in messages[0]
    assert COMMAND_NAME in messages[0]


def test_the_divergence_helper_is_silent_on_a_directory_that_does_not_exist():
    """Reading an unbuilt workflow must not create anything, or raise."""
    assert describe_fingerprint_divergence("/nonexistent/path", _fingerprint()) is None


# ---------------------------------------------------------------------------
# Superseded variants must be collected
# ---------------------------------------------------------------------------

def _store_distinct_variants(cache, count: int) -> None:
    """Write *count* variants of one command, each a different model string."""
    for index in range(count):
        assert cache.store(
            _fingerprint(model=f"test/model-{index}"), 42, [f"draw {index}"]
        )
        # Coarse mtime granularity would otherwise make "most recent" ambiguous.
        time.sleep(0.01)


def test_superseded_variants_of_a_command_are_collected_on_write(workflow_dir):
    """The growth the cache had no answer for at all."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 3)

    assert len(_stored_names(cache)) == MAX_VARIANTS_PER_COMMAND
    assert cache.stats["pruned"] == 3


def test_the_variant_just_written_always_survives(workflow_dir):
    """Pruning the file you are writing would make the cache useless, loudly."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)

    newest = _fingerprint(model="test/model-newest")
    assert cache.store(newest, 42, ["the newest draw"])
    assert UtteranceCache(workflow_dir).lookup(newest, 42) is not None


def test_the_most_recent_variants_are_the_ones_kept(workflow_dir):
    """Recency, not filename order: the survivors must be the last ones written."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)

    reader = UtteranceCache(workflow_dir)
    kept = [
        index
        for index in range(MAX_VARIANTS_PER_COMMAND + 2)
        if reader.lookup(_fingerprint(model=f"test/model-{index}"), 42) is not None
    ]
    assert kept == list(
        range(2, MAX_VARIANTS_PER_COMMAND + 2)
    ), "the oldest variants should have been the ones collected"


def test_a_write_reports_what_it_collected(workflow_dir, log_sink):
    """Silent deletion of generated data a developer paid for is not acceptable."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND)
    log_sink.records.clear()

    assert cache.store(_fingerprint(model="test/model-extra"), 42, ["extra"])
    pruning = [m for m in log_sink.messages(logging.INFO) if "Pruned" in m]
    assert len(pruning) == 1, pruning
    assert COMMAND_NAME in pruning[0]
    assert str(MAX_VARIANTS_PER_COMMAND) in pruning[0]


def test_the_run_summary_counts_what_was_collected(workflow_dir):
    """The end-of-run line is where a developer notices data disappearing."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 1)
    assert "1 superseded pruned" in cache.format_summary()


def test_a_run_that_collected_nothing_does_not_mention_pruning(workflow_dir):
    """A summary that always mentions pruning trains people to skip the line."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(), 42, ["only variant"])
    assert "pruned" not in cache.format_summary()


def test_seeds_of_one_variant_are_never_collected(workflow_dir):
    """A seed sweep writes many entries into ONE file, and must keep every one.

    Retention is per variant, not per entry. Confusing the two would delete the other
    seeds of the sweep that is running.
    """
    cache = UtteranceCache(workflow_dir)
    fingerprint = _fingerprint()
    for seed in range(MAX_VARIANTS_PER_COMMAND + 5):
        assert cache.store(fingerprint, seed, [f"draw for seed {seed}"])

    assert cache.stats["pruned"] == 0
    reader = UtteranceCache(workflow_dir)
    for seed in range(MAX_VARIANTS_PER_COMMAND + 5):
        assert reader.lookup(fingerprint, seed) is not None


def test_another_commands_variants_are_never_collected(workflow_dir):
    """Retention is scoped to the command being written, and to nothing else."""
    cache = UtteranceCache(workflow_dir)
    other = _fingerprint(command_name="subtract_two_numbers")
    assert cache.store(other, 42, ["belongs to another command"])
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)

    assert UtteranceCache(workflow_dir).lookup(other, 42) is not None


def test_a_slug_collision_cannot_delete_the_colliding_commands_data(workflow_dir):
    """Two command names can produce the same filename stem; `slugify` says so.

    Harmless for addressing, because the variant key disambiguates — and it must stay
    harmless for deletion, which is why retention matches on the STORED command name.
    """
    cache = UtteranceCache(workflow_dir)
    first, second = "Ctx/do_thing", "Ctx do_thing"
    assert slugify(first) == slugify(second)

    victim = _fingerprint(command_name=second)
    assert cache.store(victim, 42, ["the colliding command's draw"])
    for index in range(MAX_VARIANTS_PER_COMMAND + 2):
        assert cache.store(
            _fingerprint(command_name=first, model=f"test/model-{index}"),
            42,
            [f"draw {index}"],
        )
        time.sleep(0.01)

    assert UtteranceCache(workflow_dir).lookup(victim, 42) is not None


def test_the_readme_and_foreign_files_are_never_collected(workflow_dir):
    """This directory is meant to be inspected, so people leave things in it."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, 1)

    foreign = [
        "README.md",
        f"{slugify(COMMAND_NAME)}.notavariantkey.json",
        f"{slugify(COMMAND_NAME)}.{'z' * VARIANT_KEY_LENGTH}.json",
        f"{slugify(COMMAND_NAME)}.{'0' * VARIANT_KEY_LENGTH}.json.bak",
        "notes.txt",
    ]
    for name in foreign:
        with open(os.path.join(cache.root, name), "w", encoding="utf-8") as f:
            f.write("{}")

    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)
    for name in foreign:
        assert os.path.isfile(os.path.join(cache.root, name)), name


def test_an_unparsable_variant_file_is_left_alone(workflow_dir):
    """Deleting a file nobody could read would destroy the evidence of why."""
    cache = UtteranceCache(workflow_dir)
    corrupt = _fingerprint(model="test/model-corrupt")
    assert cache.store(corrupt, 42, ["will be corrupted"])
    with open(cache.entry_path(corrupt), "w", encoding="utf-8") as f:
        f.write("{ not json at all")

    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)
    assert os.path.isfile(cache.entry_path(corrupt))


def test_the_parameter_example_cache_collects_its_variants_too(workflow_dir):
    """Every parameter-model edit orphans a variant, and nothing else collects them."""
    cache = ParamExampleCache(workflow_dir)
    for index in range(MAX_VARIANTS_PER_COMMAND + 2):
        assert cache.store(
            _param_fingerprint(field_annotations_text=f"field_{index}: str"),
            42,
            _param_example(f"example {index}"),
        )
        time.sleep(0.01)

    assert len(_stored_names(cache)) == MAX_VARIANTS_PER_COMMAND
    assert cache.stats["pruned"] == 2
    assert "2 superseded pruned" in cache.format_summary()


# ---------------------------------------------------------------------------
# The shared helpers, at their edges
# ---------------------------------------------------------------------------

def test_pruning_a_directory_that_does_not_exist_is_a_no_op():
    """A cache must never be the reason a training run fails."""
    assert prune_orphaned_variants("/nonexistent/path", COMMAND_NAME) == []


def test_pruning_with_a_nonsensical_keep_count_deletes_nothing(workflow_dir):
    """Any bug in the caller must fail towards keeping data, not towards deleting it."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, 2)
    before = _stored_names(cache)

    assert prune_orphaned_variants(cache.root, COMMAND_NAME, keep=0) == []
    assert prune_orphaned_variants(cache.root, COMMAND_NAME, keep=-1) == []
    assert _stored_names(cache) == before


def test_stored_variants_reads_the_command_name_out_of_the_file(workflow_dir):
    """The whole safety argument for retention rests on this."""
    cache = UtteranceCache(workflow_dir)
    assert cache.store(_fingerprint(), 42, ["mine"])
    assert cache.store(_fingerprint(command_name="other_command"), 42, ["theirs"])

    mine = stored_variants(cache.root, COMMAND_NAME)
    assert len(mine) == 1
    assert mine[0][1]["command_name"] == COMMAND_NAME
    assert stored_variants(cache.root, "never_stored") == []


def test_retention_does_not_touch_the_versions_directory(workflow_dir):
    """Artifact versions are immutable snapshots; the cache has no business in them."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)
    versions = os.path.join(workflow_dir, COMMAND_INFO_FOLDERNAME, "versions")
    assert not os.path.exists(versions)


def test_a_surviving_variant_is_still_readable_after_a_prune(workflow_dir):
    """Deleting siblings must not disturb the file being kept."""
    cache = UtteranceCache(workflow_dir)
    _store_distinct_variants(cache, MAX_VARIANTS_PER_COMMAND + 2)

    survivor = _fingerprint(model=f"test/model-{MAX_VARIANTS_PER_COMMAND + 1}")
    with open(cache.entry_path(survivor), encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["entries"]["42"]["generated_utterances"] == [
        f"draw {MAX_VARIANTS_PER_COMMAND + 1}"
    ]
