"""Integration tests for app-supplied and domain-conditioned personas (spec R9a / F12).

Two things dominate this file, because they are the two ways R9a can do damage:

1. **The default path must not move.** With no ``personas.json`` and no
   ``SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE``, `resolve_personas` must select exactly what
   `generate_synthetic.select_persona_indices` selects, for every seed and command name.
   That equivalence is asserted against the real function, not described in prose.
2. **Persona ids must survive `heldout_evaluation.expand_persona_id`.** The whole-persona
   holdout (decision D1) parses composite ids; an app-supplied id containing the separator
   would be split into fragments and the holdout would leak exactly the way D1 exists to
   prevent. Every id this module can emit is round-tripped through the real parser.

The PersonaHub corpus is ~200k rows and is downloaded on first use, so tests that need a
corpus use a real ``datasets``-shaped stand-in built from the retail workflow's own command
metadata. That is a genuine dataset object of the shape the loader returns, not a mock of
`load_persona_dataset`: nothing is patched, the loader is a parameter of the API under test.
"""

import json
import os

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.train.duplicate_detection import utterances_from_workflow
from fastworkflow.train.determinism import (
    PERSONA_ID_SEPARATOR,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    get_training_seed,
)
from fastworkflow.train.generate_synthetic import select_persona_indices
from fastworkflow.train.heldout_evaluation import expand_persona_id
from fastworkflow.train.personas import (
    APP_PERSONA_ID_PREFIX,
    DEFAULT_MIN_POOL_MULTIPLE,
    PERSONA_FILENAME,
    SOURCE_APP_SUPPLIED,
    SOURCE_DOMAIN_CONDITIONED,
    SOURCE_PERSONAHUB,
    AppPersonaSource,
    DomainConditionedPersonaSource,
    Persona,
    PersonaConfigError,
    PersonaHubSource,
    active_persona_source_label,
    get_persona_source,
    load_persona_config,
    persona_config_path,
    persona_source_for_workflow,
    persona_source_needs_corpus,
    resolve_personas,
    set_persona_source,
    validate_persona_id,
)

RETAIL_PATH = os.path.join("fastworkflow", "examples", "retail_workflow")
HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")


def _resolve_env_vars() -> dict:
    """Same resolution order as `test_train_modern_stack._resolve_env_vars`."""
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
    """Every test starts with nothing installed and leaves nothing behind.

    The source is process-wide (it has to be — `generate_diverse_utterances` is public API
    and cannot grow a parameter), so a leaked source would silently change what a later
    test, or a later training run in the same process, generates.
    """
    set_persona_source(None)
    yield
    set_persona_source(None)


class _RowDataset:
    """A stand-in with PersonaHub's access shape: ``len()`` and ``ds[i]['persona']``.

    Built from real text supplied by the caller. This is the shape
    `generate_synthetic.load_persona_dataset` returns after
    ``load_dataset(...)['train']``, and the loader is a parameter of every API under test,
    so nothing is patched or monkeyed.
    """

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
    """A corpus of real sentences from the retail workflow's shipped command files.

    Real text matters for the domain-conditioning tests: they assert that filtering picks
    out rows mentioning retail vocabulary, which needs prose that actually varies.
    """
    seeds = utterances_from_workflow(RETAIL_PATH)
    texts = [
        f"A person who says: {utterance}"
        for command in sorted(seeds)
        for utterance in seeds[command]
    ]
    assert len(texts) > 40, "retail workflow should supply plenty of distinct sentences"
    return _RowDataset(texts)


# ---------------------------------------------------------------------------
# The default path must not move
# ---------------------------------------------------------------------------


def test_default_selection_matches_generate_synthetic_exactly(retail_persona_corpus):
    """The equivalence that keeps a workflow with no persona file byte-identical.

    Asserted against the real `select_persona_indices` across several seeds, command names
    and sample sizes, because it is the single most expensive thing R9a could break: a
    silent change here would make every previously-recorded provenance file
    incomparable with every new one.
    """
    loader = lambda: retail_persona_corpus  # noqa: E731 - a one-expression loader
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
                assert selection.personas == [
                    retail_persona_corpus[i]["persona"] for i in expected
                ]


def test_default_source_is_personahub_and_reports_itself(retail_persona_corpus):
    selection = resolve_personas(
        num_personas=3,
        seed=42,
        command_name="cancel_pending_order",
        dataset_loader=lambda: retail_persona_corpus,
    )
    assert selection.source == SOURCE_PERSONAHUB
    assert selection.pool_size == len(retail_persona_corpus)
    assert selection.fingerprint


def test_workflow_without_a_persona_file_gets_personahub():
    source = persona_source_for_workflow(RETAIL_PATH, dataset_loader=lambda: None)
    assert isinstance(source, PersonaHubSource)
    assert not os.path.exists(persona_config_path(RETAIL_PATH)), (
        "a shipped example must not acquire a personas.json as a side effect"
    )


def test_resolve_with_seed_none_uses_the_configured_training_seed(retail_persona_corpus):
    loader = lambda: retail_persona_corpus  # noqa: E731
    explicit = resolve_personas(
        num_personas=3,
        seed=get_training_seed(),
        command_name="get_user_details",
        dataset_loader=loader,
    )
    implicit = resolve_personas(
        num_personas=3,
        seed=None,
        command_name="get_user_details",
        dataset_loader=loader,
    )
    assert implicit.persona_ids == explicit.persona_ids


# ---------------------------------------------------------------------------
# App-supplied personas
# ---------------------------------------------------------------------------


def _write_persona_file(folder, payload) -> str:
    path = os.path.join(str(folder), PERSONA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def test_app_supplied_personas_are_used_verbatim(tmp_path):
    personas = [
        "A returns clerk who processes exchanges all day.",
        "A shopper who never reads the confirmation email.",
        "A fraud analyst reviewing suspicious orders.",
    ]
    _write_persona_file(tmp_path, {"schema_version": 1, "personas": personas})

    source = persona_source_for_workflow(str(tmp_path))
    assert isinstance(source, AppPersonaSource)

    set_persona_source(source)
    selection = resolve_personas(
        num_personas=3,
        seed=42,
        command_name="cancel_pending_order",
        dataset_loader=_loader_that_must_not_be_called,
    )
    assert sorted(selection.personas) == sorted(personas)
    assert selection.source == SOURCE_APP_SUPPLIED
    assert all(pid.startswith(APP_PERSONA_ID_PREFIX) for pid in selection.persona_ids)


def _loader_that_must_not_be_called():
    """Guard: app-supplied personas must never trigger a PersonaHub download.

    A 200k-row download at the top of a training run that does not need it is a real cost,
    and one an app-supplied persona set is partly bought to avoid.
    """
    raise AssertionError(
        "PersonaHub was loaded even though the application supplied its own personas."
    )


def test_app_supplied_personas_accept_explicit_ids(tmp_path):
    _write_persona_file(
        tmp_path,
        {
            "schema_version": 1,
            "personas": [
                {"id": "returns-clerk", "persona": "A returns clerk."},
                {"id": "fraud-analyst", "text": "A fraud analyst."},
            ],
        },
    )
    config = load_persona_config(persona_config_path(str(tmp_path)))
    assert [p.id for p in config.personas] == [
        f"{APP_PERSONA_ID_PREFIX}returns-clerk",
        f"{APP_PERSONA_ID_PREFIX}fraud-analyst",
    ]


def test_app_supplied_personas_accept_a_bare_list(tmp_path):
    _write_persona_file(tmp_path, ["A returns clerk.", "A fraud analyst."])
    config = load_persona_config(persona_config_path(str(tmp_path)))
    assert len(config.personas) == 2


def test_app_supplied_selection_is_deterministic_and_command_specific(tmp_path):
    personas = [f"Persona number {i} who shops online." for i in range(12)]
    _write_persona_file(tmp_path, {"personas": personas})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))

    first = resolve_personas(4, 42, "cancel_pending_order", _loader_that_must_not_be_called)
    again = resolve_personas(4, 42, "cancel_pending_order", _loader_that_must_not_be_called)
    other = resolve_personas(4, 42, "get_user_details", _loader_that_must_not_be_called)

    assert first.persona_ids == again.persona_ids
    assert first.persona_ids != other.persona_ids, (
        "two commands drawing the same personas would remove the diversity the persona "
        "mechanism exists to provide"
    )


def test_small_app_persona_set_is_reported(tmp_path):
    _write_persona_file(tmp_path, {"personas": ["The only user."]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    selection = resolve_personas(4, 42, "cancel_pending_order", _loader_that_must_not_be_called)
    assert len(selection.personas) == 1
    assert any("holdout" in note for note in selection.notes)


def test_fingerprint_changes_when_persona_text_changes(tmp_path):
    """R6's cache must invalidate on an edited persona, not just a renamed one."""
    _write_persona_file(tmp_path, {"personas": [{"id": "a", "persona": "First wording."}]})
    before = persona_source_for_workflow(str(tmp_path)).fingerprint()

    _write_persona_file(tmp_path, {"personas": [{"id": "a", "persona": "Second wording."}]})
    after = persona_source_for_workflow(str(tmp_path)).fingerprint()

    assert before != after


def test_fingerprint_is_stable_for_identical_content(tmp_path):
    payload = {"personas": [{"id": "a", "persona": "First wording."}]}
    _write_persona_file(tmp_path, payload)
    first = persona_source_for_workflow(str(tmp_path)).fingerprint()
    second = persona_source_for_workflow(str(tmp_path)).fingerprint()
    assert first == second


# ---------------------------------------------------------------------------
# Domain conditioning
# ---------------------------------------------------------------------------


def test_domain_keywords_filter_the_corpus(tmp_path, retail_persona_corpus):
    _write_persona_file(
        tmp_path, {"schema_version": 1, "domain_keywords": ["order", "shipping"]}
    )
    source = persona_source_for_workflow(
        str(tmp_path), dataset_loader=lambda: retail_persona_corpus
    )
    assert isinstance(source, DomainConditionedPersonaSource)
    assert 0 < source.pool_size() < len(retail_persona_corpus)

    set_persona_source(source)
    selection = resolve_personas(
        4, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    assert selection.source == SOURCE_DOMAIN_CONDITIONED
    for text in selection.personas:
        assert "order" in text.lower() or "shipping" in text.lower()


def test_domain_conditioned_ids_stay_personahub_row_indices(tmp_path, retail_persona_corpus):
    """Filtering must not renumber the corpus.

    A provenance record naming persona 8137 has to mean the same row whether or not the
    run was domain-conditioned, or two runs cannot be compared at all.
    """
    _write_persona_file(tmp_path, {"domain_keywords": ["order"]})
    source = persona_source_for_workflow(
        str(tmp_path), dataset_loader=lambda: retail_persona_corpus
    )
    set_persona_source(source)
    selection = resolve_personas(3, 42, "cancel_pending_order", lambda: retail_persona_corpus)
    for persona_id, text in zip(selection.persona_ids, selection.personas):
        assert retail_persona_corpus[int(persona_id)]["persona"] == text


def test_a_prose_domain_is_usable_without_keywords(tmp_path, retail_persona_corpus):
    _write_persona_file(tmp_path, {"domain": "order shipping and returns"})
    source = persona_source_for_workflow(
        str(tmp_path), dataset_loader=lambda: retail_persona_corpus
    )
    assert isinstance(source, DomainConditionedPersonaSource)
    assert source.pool_size() > 0


def test_unmatched_keywords_fall_back_to_the_whole_corpus_and_say_so(
    tmp_path, retail_persona_corpus
):
    """A mistyped keyword must not abort a multi-hour run, and must not pass silently."""
    _write_persona_file(tmp_path, {"domain_keywords": ["nonexistentkeywordxyzzy"]})
    source = persona_source_for_workflow(
        str(tmp_path), dataset_loader=lambda: retail_persona_corpus
    )
    assert source.pool_size() == len(retail_persona_corpus)
    assert any("no personahub persona matched" in n.lower() for n in source.notes())


def test_a_thin_domain_pool_is_topped_up_and_reported(tmp_path, retail_persona_corpus):
    source = DomainConditionedPersonaSource(
        lambda: retail_persona_corpus,
        keywords=["zip"],
        num_personas_hint=1000,
        origin="test",
    )
    assert source.pool_size() == len(retail_persona_corpus)
    assert any("topped up" in note for note in source.notes())


def test_explicit_personas_win_over_domain_keywords(tmp_path):
    """Supplying personas is the stronger statement; keywords must not dilute the set."""
    _write_persona_file(
        tmp_path,
        {"personas": ["The only curated user."], "domain_keywords": ["order"]},
    )
    source = persona_source_for_workflow(str(tmp_path))
    assert isinstance(source, AppPersonaSource)


# ---------------------------------------------------------------------------
# Persona id discipline — the holdout must survive every id this module can emit
# ---------------------------------------------------------------------------


def test_emitted_ids_round_trip_through_the_holdout_parser(tmp_path, retail_persona_corpus):
    """Every id shape, through the real `expand_persona_id`.

    If a persona id does not survive this, `split_by_persona` reserves a fragment as if it
    were a persona and the whole-persona holdout leaks — the exact defect D1 exists to
    prevent.
    """
    _write_persona_file(
        tmp_path,
        {"personas": [{"id": "returns clerk (EU)", "persona": "A clerk."},
                      {"id": "fraud-analyst.2", "persona": "An analyst."}]},
    )
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    app_selection = resolve_personas(2, 42, "cancel_pending_order", _loader_that_must_not_be_called)

    set_persona_source(None)
    hub_selection = resolve_personas(
        2, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )

    for persona_id in app_selection.persona_ids + hub_selection.persona_ids:
        assert expand_persona_id(persona_id) == frozenset({persona_id})


def test_ids_that_would_break_the_holdout_are_rejected(tmp_path):
    for bad_id in (
        f"clerk{PERSONA_ID_SEPARATOR}analyst",
        SEED_PERSONA_ID,
        f"{UNRESOLVED_PERSONA_PREFIX}17",
        "",
        "   ",
    ):
        with pytest.raises(PersonaConfigError):
            validate_persona_id(bad_id)


def test_a_persona_file_with_a_separator_in_an_id_fails_to_load(tmp_path):
    _write_persona_file(
        tmp_path,
        {"personas": [{"id": f"a{PERSONA_ID_SEPARATOR}b", "persona": "Someone."}]},
    )
    with pytest.raises(PersonaConfigError):
        load_persona_config(persona_config_path(str(tmp_path)))


def test_the_seed_persona_id_is_never_emitted(tmp_path):
    """`SEED_PERSONA_ID` marks hand-written seeds, which are never held out.

    The prefix makes a collision structurally impossible; this pins that it stays so.
    """
    _write_persona_file(tmp_path, {"personas": [{"id": SEED_PERSONA_ID.strip("_"), "persona": "x"}]})
    config = load_persona_config(persona_config_path(str(tmp_path)))
    assert config.personas[0].id != SEED_PERSONA_ID


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------


def test_unknown_schema_version_is_rejected(tmp_path):
    _write_persona_file(tmp_path, {"schema_version": 99, "personas": ["x"]})
    with pytest.raises(PersonaConfigError):
        load_persona_config(persona_config_path(str(tmp_path)))


def test_an_empty_persona_file_is_rejected(tmp_path):
    _write_persona_file(tmp_path, {"schema_version": 1})
    with pytest.raises(PersonaConfigError):
        load_persona_config(persona_config_path(str(tmp_path)))


def test_duplicate_persona_ids_are_rejected(tmp_path):
    _write_persona_file(
        tmp_path,
        {"personas": [{"id": "a", "persona": "One."}, {"id": "a", "persona": "Two."}]},
    )
    with pytest.raises(PersonaConfigError):
        load_persona_config(persona_config_path(str(tmp_path)))


def test_a_persona_without_text_is_rejected(tmp_path):
    _write_persona_file(tmp_path, {"personas": [{"id": "a", "persona": "   "}]})
    with pytest.raises(PersonaConfigError):
        load_persona_config(persona_config_path(str(tmp_path)))


def test_a_broken_persona_file_fails_the_run_rather_than_falling_back(tmp_path):
    """Silently training on generic personas would make provenance describe a run that
    did not happen. See `PersonaConfigError`."""
    _write_persona_file(tmp_path, {"schema_version": 1, "personas": [42]})
    with pytest.raises(PersonaConfigError):
        persona_source_for_workflow(str(tmp_path))


# ---------------------------------------------------------------------------
# Installed-source plumbing
# ---------------------------------------------------------------------------


def test_installed_source_is_process_wide_and_clearable(tmp_path):
    _write_persona_file(tmp_path, {"personas": ["Someone."]})
    source = persona_source_for_workflow(str(tmp_path))
    set_persona_source(source)
    assert get_persona_source() is source
    set_persona_source(None)
    assert get_persona_source() is None


def test_app_source_needs_at_least_one_persona():
    with pytest.raises(PersonaConfigError):
        AppPersonaSource([])


def test_selection_is_capped_by_the_pool(tmp_path):
    _write_persona_file(tmp_path, {"personas": ["One.", "Two."]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    selection = resolve_personas(50, 42, "cancel_pending_order", _loader_that_must_not_be_called)
    assert len(selection.personas) == 2
    assert len(selection.persona_ids) == 2


def test_zero_personas_requested_yields_an_empty_selection(retail_persona_corpus):
    selection = resolve_personas(
        0, 42, "cancel_pending_order", lambda: retail_persona_corpus
    )
    assert selection.personas == []
    assert selection.persona_ids == []


def test_min_pool_multiple_is_a_positive_integer():
    assert isinstance(DEFAULT_MIN_POOL_MULTIPLE, int) and DEFAULT_MIN_POOL_MULTIPLE >= 1


def test_persona_model_rejects_nothing_it_should_accept():
    persona = Persona(id=f"{APP_PERSONA_ID_PREFIX}0", text="Someone.")
    assert expand_persona_id(persona.id) == frozenset({persona.id})


# ---------------------------------------------------------------------------
# Utterance-cache interaction (R6 / decision D6)
# ---------------------------------------------------------------------------


def test_cache_label_is_none_for_the_default_draw():
    """A workflow with no persona file must keep its existing cache variant.

    If this returns anything, every cached utterance in every existing workflow is
    invalidated the day R9a lands, and the first training run after the upgrade
    regenerates everything from the LLM for no reason.
    """
    assert active_persona_source_label() is None


def test_cache_label_distinguishes_an_app_supplied_source(tmp_path):
    """An app persona set must be a different cache variant.

    Without this a workflow that adds a personas.json keeps being served utterances
    generated from generic PersonaHub personas, and the feature silently does nothing.
    """
    _write_persona_file(tmp_path, {"personas": ["A returns clerk."]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    label = active_persona_source_label()
    assert label and label.startswith(SOURCE_APP_SUPPLIED)


def test_cache_label_changes_with_the_persona_text(tmp_path):
    _write_persona_file(tmp_path, {"personas": [{"id": "a", "persona": "First."}]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    first = active_persona_source_label()

    _write_persona_file(tmp_path, {"personas": [{"id": "a", "persona": "Second."}]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    assert active_persona_source_label() != first


def test_cache_label_for_domain_conditioning_needs_no_corpus(tmp_path):
    """The label is computed before the cache decides whether to generate at all.

    Loading PersonaHub to compute it would undo the reason the cache exists, so the
    loader here raises if touched.
    """
    _write_persona_file(tmp_path, {"domain_keywords": ["retail", "shopping"]})
    set_persona_source(
        persona_source_for_workflow(
            str(tmp_path), dataset_loader=_loader_that_must_not_be_called
        )
    )
    label = active_persona_source_label()
    assert label and label.startswith(SOURCE_DOMAIN_CONDITIONED)


def test_cache_label_changes_with_the_domain_keywords(tmp_path):
    _write_persona_file(tmp_path, {"domain_keywords": ["retail"]})
    set_persona_source(
        persona_source_for_workflow(str(tmp_path), dataset_loader=_loader_that_must_not_be_called)
    )
    first = active_persona_source_label()

    _write_persona_file(tmp_path, {"domain_keywords": ["banking"]})
    set_persona_source(
        persona_source_for_workflow(str(tmp_path), dataset_loader=_loader_that_must_not_be_called)
    )
    assert active_persona_source_label() != first


def test_personahub_fingerprint_needs_no_corpus():
    assert PersonaHubSource(_loader_that_must_not_be_called).fingerprint()


def test_only_an_app_supplied_set_can_skip_the_datasets_package(tmp_path):
    """A workflow that wrote its own personas must not need the optional `datasets` dep.

    Generation refuses to run without it today, which is correct for a PersonaHub draw
    and wrong for a persona set that never touches the corpus.
    """
    assert persona_source_needs_corpus() is True

    _write_persona_file(tmp_path, {"personas": ["A returns clerk."]})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))
    assert persona_source_needs_corpus() is False

    _write_persona_file(tmp_path, {"domain_keywords": ["retail"]})
    set_persona_source(
        persona_source_for_workflow(
            str(tmp_path), dataset_loader=_loader_that_must_not_be_called
        )
    )
    assert persona_source_needs_corpus() is True


# ---------------------------------------------------------------------------
# End to end through the real generation loop
# ---------------------------------------------------------------------------


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``.

    Copied from `test_train_modern_stack` so this file skips under the same conditions the
    rest of the training suite does.
    """
    return bool(value) and "<" not in value and "your-" not in value.lower()


@pytest.mark.skipif(
    not _looks_like_real_key(_resolve_env_vars().get("LITELLM_API_KEY_SYNDATA_GEN")),
    reason=(
        "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot exercise synthetic "
        "utterance generation."
    ),
)
def test_app_personas_drive_real_generation_and_land_in_provenance(tmp_path):
    """The composition this feature exists for, against the real LLM.

    `resolve_personas` is the hook `generate_diverse_utterances_with_provenance` calls, and
    `generate_utterances_for_personas` is the generation loop it hands the result to. This
    runs that composition end to end and then checks the part that would fail silently: the
    persona ids attributed to each generated utterance must be app ids, and every one of
    them must survive `expand_persona_id`, or the whole-persona holdout would reserve a
    fragment of an id and leak.
    """
    import litellm

    from fastworkflow.train.determinism import UtteranceProvenance
    from fastworkflow.train.generate_synthetic import generate_utterances_for_personas

    # `generate_utterances_for_personas` is the inner loop; installing the key is the
    # caller's job and `generate_diverse_utterances_with_provenance` does it at its top.
    litellm.api_key = fastworkflow.get_env_var("LITELLM_API_KEY_SYNDATA_GEN")

    personas = [
        "A retail returns clerk who processes exchanges and refunds all day.",
        "An online shopper who orders frequently and often changes their mind.",
    ]
    _write_persona_file(tmp_path, {"schema_version": 1, "personas": personas})
    set_persona_source(persona_source_for_workflow(str(tmp_path)))

    selection = resolve_personas(
        num_personas=2,
        seed=42,
        command_name="cancel_pending_order",
        dataset_loader=_loader_that_must_not_be_called,
    )
    assert selection.source == SOURCE_APP_SUPPLIED

    provenance = UtteranceProvenance(
        command_name="cancel_pending_order",
        seed=42,
        persona_ids=list(selection.persona_ids),
    )
    seed_utterances = [
        "Can you cancel my pending order?",
        "I want to cancel my order because I no longer need it.",
    ]
    generated = generate_utterances_for_personas(
        seed_utterances,
        "cancel_pending_order",
        selection.personas,
        selection.persona_ids,
        provenance,
        utterances_per_persona=3,
        personas_per_batch=1,
        model=fastworkflow.get_env_var("LLM_SYNDATA_GEN"),
        seed=42,
    )

    if provenance.fell_back:
        pytest.skip(f"generation fell back: {provenance.fallback_reason}")

    assert generated, "the LLM returned no utterances at all"
    assert provenance.utterance_personas, "no utterance was attributed to a persona"

    app_ids = set(selection.persona_ids)
    for persona_id in provenance.utterance_personas.values():
        contributors = expand_persona_id(persona_id)
        assert contributors <= app_ids, (
            f"utterance attributed to {persona_id!r}, which does not expand to the "
            f"app-supplied persona ids {sorted(app_ids)}"
        )
