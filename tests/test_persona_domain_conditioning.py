"""Integration tests for the domain-conditioned persona pool (bd fix-k0i.38, spec R9a/F12).

The finding: when fewer PersonaHub rows matched an application's ``domain_keywords``
than the diversity floor wanted, the top-up appended **every** unmatched row. The pool
became the whole ~200k corpus and `PersonaSource.select` sampled it uniformly, so a
matched persona's chance of writing anything was about ``matched/200000`` — the
conditioning was nullified at exactly the moment it was thinnest, and the note called
that "partial". Compounding it, ``num_personas_hint`` was never supplied by any
production call path, so the floor collapsed to ``max(0 * 4, 4) == 4`` and the whole
branch only fired below four matches.

What is pinned here:

* the padded pool is only as large as the floor asks for, never the corpus;
* the matching rows are drawn FIRST, so conditioning survives its own padding;
* the pool is identical on every run and at every training seed, because the utterance
  cache has no fingerprint input that would record a pool that moved;
* the per-command persona count reaches the pool sizer from configuration;
* the note names the real numbers, since a developer cannot fix a thin pool they were
  not told about.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures.
The corpus is a real ``datasets``-shaped object built from the retail workflow's own
shipped seed utterances, and the loader is a parameter of every API under test, so
nothing is patched.
"""

import os

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.duplicate_detection import utterances_from_workflow
from fastworkflow.train.generate_synthetic import select_persona_indices
from fastworkflow.train.personas import (
    DEFAULT_MIN_POOL_MULTIPLE,
    NUM_PERSONAS_ENV_VAR,
    PERSONA_FILENAME,
    AppPersonaSource,
    DomainConditionedPersonaSource,
    Persona,
    PersonaHubSource,
    persona_source_for_workflow,
    resolve_personas,
    set_persona_source,
)

RETAIL_PATH = os.path.join("fastworkflow", "examples", "retail_workflow")

#: A word that appears in only a couple of the retail seed utterances, so the matched
#: set is genuinely below the floor and the top-up branch is the one under test.
THIN_KEYWORD = "zip"


def _resolve_env_vars() -> dict:
    """Same resolution order as `tests/test_personas.py`."""
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}

    local_env = os.path.join("env", ".env")
    local_pwd = os.path.join("passwords", ".env")
    if os.path.exists(local_env):
        env_vars.update(dotenv_values(local_env))
    if os.path.exists(local_pwd):
        env_vars.update(dotenv_values(local_pwd))
    return env_vars


@pytest.fixture(scope="module", autouse=True)
def _initialised_fastworkflow():
    fastworkflow.init(env_vars=_resolve_env_vars())


@pytest.fixture(autouse=True)
def _no_installed_source():
    """The installed source is process-wide, so a leak would change later tests."""
    set_persona_source(None)
    yield
    set_persona_source(None)


class _RowDataset:
    """PersonaHub's access shape: ``len()`` and ``ds[i]['persona']``."""

    def __init__(self, texts):
        self._rows = [{"persona": text} for text in texts]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows)


@pytest.fixture(scope="module")
def retail_persona_corpus():
    """Real sentences from the retail workflow's shipped command files."""
    seeds = utterances_from_workflow(RETAIL_PATH)
    texts = [
        f"A person who says: {utterance}"
        for command in sorted(seeds)
        for utterance in seeds[command]
    ]
    assert len(texts) > 40, "retail workflow should supply plenty of distinct sentences"
    return _RowDataset(texts)


def _thin_source(corpus, **overrides):
    """A domain source whose keyword matches far fewer rows than the floor wants."""
    kwargs = {"keywords": [THIN_KEYWORD], "origin": "test"}
    kwargs |= overrides
    return DomainConditionedPersonaSource(lambda: corpus, **kwargs)


def _matched_rows(corpus, keyword: str) -> list[int]:
    return [
        index
        for index in range(len(corpus))
        if keyword in corpus[index]["persona"].lower().split()
    ]


def test_the_keyword_under_test_really_is_thin(retail_persona_corpus):
    """Guard the premise: if the keyword stopped being rare these tests would vacuously pass."""
    matched = _matched_rows(retail_persona_corpus, THIN_KEYWORD)
    assert 0 < len(matched) < DEFAULT_MIN_POOL_MULTIPLE * 2
    assert len(matched) < len(retail_persona_corpus)


# ---------------------------------------------------------------------------
# The pool must be padded to the floor, not to the corpus
# ---------------------------------------------------------------------------

def test_a_thin_pool_is_padded_to_the_floor_and_not_to_the_whole_corpus(
    retail_persona_corpus,
):
    """The reported defect, stated as a number.

    Appending every unmatched row made the pool the entire corpus, which is what
    reduced a matched persona's chance of being drawn to noise.
    """
    source = _thin_source(retail_persona_corpus, num_personas_hint=2)
    wanted = max(2 * DEFAULT_MIN_POOL_MULTIPLE, DEFAULT_MIN_POOL_MULTIPLE)

    assert source.pool_size() == wanted
    assert source.pool_size() < len(retail_persona_corpus)


def test_the_matching_rows_are_drawn_before_any_padding(retail_persona_corpus):
    """Conditioning must survive its own padding.

    A command asking for as many personas as there are matches must get ONLY matched
    personas; uniform sampling over a padded pool would have handed it mostly generic
    ones.
    """
    matched = _matched_rows(retail_persona_corpus, THIN_KEYWORD)
    source = _thin_source(retail_persona_corpus, num_personas_hint=8)
    set_persona_source(source)

    selection = resolve_personas(
        len(matched), 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    assert sorted(int(pid) for pid in selection.persona_ids) == sorted(matched)
    for text in selection.personas:
        assert THIN_KEYWORD in text.lower().split()


def test_every_command_gets_the_matching_rows_not_one_command_in_five(
    retail_persona_corpus,
):
    """The per-command consequence, across several commands and seeds.

    With a pool of `matched + padding` sampled uniformly, most commands would draw no
    matched persona at all. Priority ordering makes every command spend its budget on
    the matched rows first.
    """
    matched = set(_matched_rows(retail_persona_corpus, THIN_KEYWORD))
    source = _thin_source(retail_persona_corpus, num_personas_hint=6)
    set_persona_source(source)

    for seed in (0, 42, 12345):
        for command_name in ("cancel_pending_order", "get_user_details", "wildcard"):
            selection = resolve_personas(
                len(matched), seed, command_name, lambda: retail_persona_corpus
            )
            assert {int(pid) for pid in selection.persona_ids} == matched, (
                f"command {command_name!r} at seed {seed} drew unconditioned personas "
                f"while matched rows were still available"
            )


def test_a_request_larger_than_the_matched_set_is_filled_from_the_padding(
    retail_persona_corpus,
):
    """Padding still does its job: the request is filled, and the extras are generic."""
    matched = set(_matched_rows(retail_persona_corpus, THIN_KEYWORD))
    source = _thin_source(retail_persona_corpus, num_personas_hint=8)
    set_persona_source(source)

    requested = len(matched) + 3
    selection = resolve_personas(
        requested, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    drawn = {int(pid) for pid in selection.persona_ids}

    assert len(selection.persona_ids) == requested
    assert matched <= drawn
    assert len(selection.persona_ids) == len(set(selection.persona_ids)), (
        "a persona was drawn twice, so the two-tier draw is sampling with replacement"
    )


def test_the_padded_pool_is_the_same_on_every_run_and_at_every_seed(
    retail_persona_corpus,
):
    """The utterance cache has no fingerprint input that records the pool.

    So a pool that varied per run — or per training seed — would produce misses, or
    worse hits, that nothing on disk could explain. The padding is therefore derived
    from the keywords alone.
    """
    first = _thin_source(retail_persona_corpus, num_personas_hint=5)
    second = _thin_source(retail_persona_corpus, num_personas_hint=5)
    assert first.rows == second.rows

    set_persona_source(first)
    at_seed_42 = resolve_personas(
        4, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    set_persona_source(second)
    again_at_seed_42 = resolve_personas(
        4, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    assert at_seed_42.persona_ids == again_at_seed_42.persona_ids


def test_the_padding_is_not_simply_the_first_rows_of_the_corpus(retail_persona_corpus):
    """Slicing would give every thin-pool workflow the same handful of generic personas.

    They would also all be neighbours in the corpus, which for a corpus with any
    ordering at all is the least diverse padding available.
    """
    source = _thin_source(retail_persona_corpus, num_personas_hint=6)
    matched = set(_matched_rows(retail_persona_corpus, THIN_KEYWORD))
    padding = [row for row in source.rows if row not in matched]

    assert padding
    assert padding != list(range(len(padding)))
    assert padding == sorted(padding), "the padding order must be stable"


def test_the_note_names_the_real_numbers(retail_persona_corpus):
    """A thin pool a developer is not told the size of is a silent quality regression."""
    matched = _matched_rows(retail_persona_corpus, THIN_KEYWORD)
    source = _thin_source(retail_persona_corpus, num_personas_hint=3)
    note = next(n for n in source.notes() if "topped up" in n)

    assert f"topped up to {source.pool_size()}" in note
    assert f"{len(matched)} matching one(s) are still drawn first" in note
    assert "personas.json" in note


# ---------------------------------------------------------------------------
# The hint must reach the pool sizer from configuration
# ---------------------------------------------------------------------------

def test_the_pool_floor_uses_the_configured_per_command_persona_count(
    retail_persona_corpus,
):
    """`num_personas_hint` was dead in production, so the floor was always 4.

    `train/__main__.py` builds the source with no hint and has no reason to know about
    persona pools, so the source reads the same setting `generate_synthetic` resolves
    `num_personas` from.
    """
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, NUM_PERSONAS_ENV_VAR: "9"})
        source = _thin_source(retail_persona_corpus)
        assert source.pool_size() == 9 * DEFAULT_MIN_POOL_MULTIPLE
    finally:
        fastworkflow.init(previous)


def test_an_explicit_hint_still_wins_over_configuration(retail_persona_corpus):
    """A caller that knows the real count must not be second-guessed by the env file."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, NUM_PERSONAS_ENV_VAR: "9"})
        source = _thin_source(retail_persona_corpus, num_personas_hint=2)
        assert source.pool_size() == 2 * DEFAULT_MIN_POOL_MULTIPLE
    finally:
        fastworkflow.init(previous)


def test_a_malformed_persona_count_does_not_fail_the_run(retail_persona_corpus):
    """A persona pool is not worth aborting a multi-hour training run over."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, NUM_PERSONAS_ENV_VAR: "not-a-number"})
        source = _thin_source(retail_persona_corpus)
        assert source.pool_size() == DEFAULT_MIN_POOL_MULTIPLE
    finally:
        fastworkflow.init(previous)


def test_a_workflow_persona_file_reaches_the_pool_sizer(tmp_path, retail_persona_corpus):
    """The production shape: `persona_source_for_workflow` with no hint argument."""
    with open(os.path.join(str(tmp_path), PERSONA_FILENAME), "w", encoding="utf-8") as f:
        f.write('{"schema_version": 1, "domain_keywords": ["' + THIN_KEYWORD + '"]}')

    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, NUM_PERSONAS_ENV_VAR: "7"})
        source = persona_source_for_workflow(
            str(tmp_path), dataset_loader=lambda: retail_persona_corpus
        )
        assert source.pool_size() == 7 * DEFAULT_MIN_POOL_MULTIPLE
    finally:
        fastworkflow.init(previous)


# ---------------------------------------------------------------------------
# Nothing else may move
# ---------------------------------------------------------------------------

def test_the_default_personahub_draw_is_still_byte_identical(retail_persona_corpus):
    """The two-tier draw must be unreachable for every source that padded nothing.

    `select_persona_indices` is the reference implementation, and a workflow with no
    persona file must keep selecting exactly what it selects, or every provenance
    record ever written becomes incomparable with every new one.
    """
    loader = lambda: retail_persona_corpus  # noqa: E731 - a one-expression loader
    assert PersonaHubSource(loader).priority_pool_size() == 0

    for seed in (0, 1, 42, 12345):
        for command_name in ("cancel_pending_order", "IntentDetection/go_up", "wildcard"):
            for num_personas in (1, 4, 7):
                expected = select_persona_indices(
                    len(retail_persona_corpus), num_personas, seed, command_name
                )
                selection = resolve_personas(
                    num_personas=num_personas,
                    seed=seed,
                    command_name=command_name,
                    dataset_loader=loader,
                )
                assert selection.persona_ids == [str(i) for i in expected]


def test_an_app_supplied_pool_is_never_treated_as_two_tiers():
    """Every persona the developer wrote down is one they wanted."""
    source = AppPersonaSource(
        [Persona(id="app:a", text="One."), Persona(id="app:b", text="Two.")]
    )
    assert source.priority_pool_size() == 0


def test_a_pool_that_needed_no_padding_is_one_uniform_population(retail_persona_corpus):
    """`priority_pool_size == pool_size` must take the plain uniform branch.

    Otherwise the common, healthy domain-conditioned case would go down a different
    sampling path from the one the reference implementation pins.
    """
    source = DomainConditionedPersonaSource(
        lambda: retail_persona_corpus,
        keywords=["order"],
        num_personas_hint=1,
        origin="test",
    )
    assert source.priority_pool_size() == source.pool_size()
    assert not any("topped up" in note for note in source.notes())

    set_persona_source(source)
    selection = resolve_personas(
        3, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    expected = select_persona_indices(
        source.pool_size(), 3, 42, "cancel_pending_order"
    )
    assert selection.persona_ids == [str(source.rows[i]) for i in expected]


def test_no_matches_at_all_still_falls_back_to_the_whole_corpus(retail_persona_corpus):
    """A mistyped keyword must not stop a run, and must not become a one-row pool."""
    source = DomainConditionedPersonaSource(
        lambda: retail_persona_corpus,
        keywords=["nonexistentkeywordxyzzy"],
        num_personas_hint=4,
        origin="test",
    )
    assert source.pool_size() == len(retail_persona_corpus)
    assert source.priority_pool_size() == 0
    assert any("no personahub persona matched" in n.lower() for n in source.notes())


def test_a_floor_larger_than_the_corpus_takes_every_row_available(
    retail_persona_corpus,
):
    """The padding must be capped by what exists, not by what was asked for."""
    source = _thin_source(retail_persona_corpus, num_personas_hint=1000)
    assert source.pool_size() == len(retail_persona_corpus)
    assert any("topped up" in note for note in source.notes())
