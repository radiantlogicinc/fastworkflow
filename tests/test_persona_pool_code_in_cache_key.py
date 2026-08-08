"""Integration tests for bd fix-6r5: a persona source's own row matching is in the key.

The residue fix-k0i.43 left behind. That issue widened
`generate_synthetic._DIGESTED_GENERATION_SOURCES` to cover the persona SELECTION
algorithm — `derived_seed`, `select_persona_indices`, `resolve_personas` and
`PersonaSource.select`. What stayed uncovered was the SOURCE's own matching behaviour:
`DomainConditionedPersonaSource._matches` decides which PersonaHub rows a domain keyword
selects, and `rows` decides how a thin match is padded. Editing either changed the
persona pool a command is generated from while every cache entry stayed valid, so a
developer could tune the matching, retrain, and be silently served the utterances the
old matching produced.

The formulation chosen is documented on `personas.pool_source_digest`: digest the code
the ACTIVE SOURCE's CLASS defines, and carry it as the third component of
`active_persona_source_label` — the fingerprint input that was already a function of the
installed source. What that buys, and what these tests pin, is:

* the row filter and the padding rule really are in the shipped variant key;
* the digest is a function of code TEXT only, so the difference between two sources can
  be attributed to a method body rather than to the name of the class holding it;
* it is deterministic in a fresh process, which is the whole point of a cache key;
* the default PersonaHub draw's key does not move, so no existing workflow's cached
  utterances are invalidated (decision D6);
* a future subclass is covered by reflection rather than by remembering to edit a list.

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures. The
persona sources are real subclasses of the shipped ones, the corpus is a real
``datasets``-shaped object, the cache is a real `UtteranceCache` writing real JSON, and
the completion backend is the production `completion_fn` injection point.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import fastworkflow
from fastworkflow.train.generate_synthetic import (
    generate_diverse_utterances_with_provenance,
    utterance_fingerprint,
)
from fastworkflow.train import personas
from fastworkflow.train.personas import (
    DEFAULT_MIN_POOL_MULTIPLE,
    SOURCE_DOMAIN_CONDITIONED,
    AppPersonaSource,
    DomainConditionedPersonaSource,
    Persona,
    PersonaHubSource,
    PersonaSource,
    active_persona_source_label,
    implementation_functions,
    set_persona_source,
)
from fastworkflow.train.utterance_cache import (
    UNAVAILABLE_SOURCE_DIGEST,
    UtteranceCache,
    get_utterance_cache,
    set_utterance_cache,
    source_digest,
)

COMMAND_NAME = "cancel_pending_order"
SEED_UTTERANCES = ["cancel my order", "please cancel order #123"]
MODEL = "mistral/mistral-small-latest"
KEYWORDS = ["zip"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Sources that differ in exactly one method body.
#
# The three below are the load-bearing construction of this file, so the shape matters.
# `_MatchesOnZip` and `_MatchesOnOrder` differ ONLY in the word inside `_matches`;
# `_AlsoMatchesOnZip` has a `_matches` that is character-for-character identical to
# `_MatchesOnZip`'s and differs only in the name of the class holding it. Together they
# let a test say "this digest changed because that method body changed", which neither
# pair could establish alone: two arbitrary subclasses would differ for any number of
# reasons, including their names.
#
# Do not add a docstring or a comment inside these three `_matches` bodies. A class
# docstring is fine — only functions are digested — but a difference inside the method
# text would make `_AlsoMatchesOnZip` stop being a control.
# ---------------------------------------------------------------------------


class _MatchesOnZip(DomainConditionedPersonaSource):
    """Matches the keyword as a whole word, which is roughly what the shipped one does."""

    def _matches(self, text: str) -> bool:
        return "zip" in text.lower().split()


class _MatchesOnOrder(DomainConditionedPersonaSource):
    """The same filter tuned to a different word: the edit fix-6r5 is about."""

    def _matches(self, text: str) -> bool:
        return "order" in text.lower().split()


class _AlsoMatchesOnZip(DomainConditionedPersonaSource):
    """Behaviourally and textually identical to `_MatchesOnZip`, under another name."""

    def _matches(self, text: str) -> bool:
        return "zip" in text.lower().split()


class _MatchesOnZipWithAComment(DomainConditionedPersonaSource):
    """`_MatchesOnZip` plus a comment, and nothing else."""

    def _matches(self, text: str) -> bool:
        # the only difference from _MatchesOnZip is this line
        return "zip" in text.lower().split()


class _PadsWithTheFirstRows(DomainConditionedPersonaSource):
    """Overrides the pool through the ``rows`` PROPERTY rather than a plain method."""

    @property
    def rows(self) -> list[int]:
        return list(range(min(4, len(self.dataset))))


class _PadsWithTheLastRows(DomainConditionedPersonaSource):
    """The same override, padding from the other end of the corpus."""

    @property
    def rows(self) -> list[int]:
        return list(range(max(0, len(self.dataset) - 4), len(self.dataset)))


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


def _corpus():
    """Rows matching "zip" and rows matching "order" are DISJOINT, deliberately.

    Overlapping sets would make `_MatchesOnZip` and `_MatchesOnOrder` select the same
    pool, and every attribution test below would pass without the digest doing anything.
    """
    def text(index: int) -> str:
        if index % 3 == 0:
            return f"A person {index} who knows every zip in the county."
        if index % 3 == 1:
            return f"A person {index} who places an order every week."
        return f"A person {index} who reads books."

    return _RowDataset([text(index) for index in range(24)])


def _loader():
    """A named module-level loader, so `callable_identity` is stable across calls."""
    return _corpus()


def _source(source_class, keywords=KEYWORDS):
    return source_class(_loader, keywords, num_personas_hint=2, origin="test")


@pytest.fixture(autouse=True)
def training_env():
    """Pin the generation env so a fingerprint does not depend on the local machine."""
    previous = dict(fastworkflow._env_vars)
    fastworkflow.init({
        **previous,
        "LLM_SYNDATA_GEN": MODEL,
        "LITELLM_API_KEY_SYNDATA_GEN": "test-key-not-used-by-the-injected-backend",
        "TRAINING_SEED": "42",
    })
    yield
    fastworkflow.init(previous)


@pytest.fixture(autouse=True)
def clean_process_wide_sinks():
    """Both sinks are process-wide, so a leak would change a later test."""
    previous_source = None
    previous_cache = get_utterance_cache()
    set_persona_source(previous_source)
    set_utterance_cache(None)
    yield
    set_persona_source(None)
    set_utterance_cache(previous_cache)


class CountingCompletion:
    """The production `completion_fn` injection point, answering with real utterances."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        from types import SimpleNamespace

        self.calls += 1
        persona_name = kwargs["messages"][0]["content"].split("[")[1].split("]")[0]
        body = "\n".join(
            (
                "cancel the order I placed yesterday",
                "kill order 4471 please",
            )
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"[{persona_name}]\n{body}\n")
                )
            ]
        )


# ---------------------------------------------------------------------------
# The premise: the two filters really do select different rows
# ---------------------------------------------------------------------------

def test_the_two_test_filters_really_do_produce_different_pools():
    """Guard the premise. If both filters matched the same rows there would be nothing
    for the digest to notice, and every test below would pass vacuously."""
    assert _source(_MatchesOnZip).rows != _source(_MatchesOnOrder).rows


def test_the_two_row_overrides_really_do_produce_different_pools():
    assert _source(_PadsWithTheFirstRows).rows != _source(_PadsWithTheLastRows).rows


# ---------------------------------------------------------------------------
# The row filter and the padding rule are inside the digest
# ---------------------------------------------------------------------------

def test_editing_the_row_filter_changes_the_pool_digest():
    """The reported defect, stated as the smallest possible edit.

    `_MatchesOnZip` and `_MatchesOnOrder` differ only in the word inside `_matches`.
    """
    assert (
        _source(_MatchesOnZip).pool_source_digest()
        != _source(_MatchesOnOrder).pool_source_digest()
    )


def test_the_digest_is_a_function_of_code_text_and_not_of_the_class_name():
    """The control that makes the test above attributable to the method body.

    `_AlsoMatchesOnZip._matches` is character-for-character `_MatchesOnZip._matches`
    under a different class name. If class or method NAMES were hashed in, these would
    differ too — and then "the digest changed" would say nothing about which edit caused
    it. Two sources whose code is identical would also generate identical utterances
    from identical configuration, so sharing one cache entry is the correct outcome.
    """
    assert (
        _source(_MatchesOnZip).pool_source_digest()
        == _source(_AlsoMatchesOnZip).pool_source_digest()
    )


def test_a_comment_only_edit_to_the_row_filter_still_invalidates():
    """R6's coarseness rule, applied here: any edit, including a comment.

    Under-invalidation costs trust in the measurement, and this project has paid that
    bill once already. A digest that ignored comments would also be a digest that has to
    decide what a "real" edit is.
    """
    assert (
        _source(_MatchesOnZip).pool_source_digest()
        != _source(_MatchesOnZipWithAComment).pool_source_digest()
    )


def test_the_padding_rule_is_digested_through_the_property_accessor():
    """`rows` is a ``property``, which `inspect.getsource` does not accept.

    Handing the property object itself to `source_digest` would record
    ``source-unavailable`` for it — a digest that has stopped noticing edits while still
    looking like a digest — and both overrides below would then collapse onto one value.
    """
    assert (
        _source(_PadsWithTheFirstRows).pool_source_digest()
        != _source(_PadsWithTheLastRows).pool_source_digest()
    )
    accessor = DomainConditionedPersonaSource.rows.fget
    assert accessor in implementation_functions(DomainConditionedPersonaSource)
    assert source_digest(accessor) != UNAVAILABLE_SOURCE_DIGEST


def test_the_two_methods_fix_6r5_names_are_in_the_digested_set():
    """The issue names `rows` and `_matches` specifically."""
    functions = implementation_functions(DomainConditionedPersonaSource)
    assert DomainConditionedPersonaSource._matches in functions
    assert DomainConditionedPersonaSource.rows.fget in functions


# ---------------------------------------------------------------------------
# The residue: a constant `rows` reads is not in any function's TEXT (bd fix-l48)
# ---------------------------------------------------------------------------

def test_retuning_the_pool_floor_constant_moves_the_digest_and_the_variant_key(
    monkeypatch
):
    """`rows` reads `DEFAULT_MIN_POOL_MULTIPLE`, so the VALUE has to be in the key.

    Digesting source text cannot reach it: retuning the constant changes the pool a
    command is generated from while every digested function's text stays identical.
    Asserted against the shipped `utterance_fingerprint` as well as the digest, because
    the digest only matters where the cache reads it.
    """
    before = _source(_MatchesOnZip)
    before_digest = before.pool_source_digest()
    before_key = _variant_key(before)

    monkeypatch.setattr(
        personas, "DEFAULT_MIN_POOL_MULTIPLE", DEFAULT_MIN_POOL_MULTIPLE * 2
    )
    # Built after the change, so the pool floor the source will actually use and the
    # value its digest reports are the same one -- as they are when the constant is
    # edited in the file rather than here.
    after = _source(_MatchesOnZip)

    assert after.pool_source_digest() != before_digest
    assert _variant_key(after) != before_key


def test_an_entry_written_under_the_old_pool_floor_is_not_reused(tmp_path, monkeypatch):
    """The consequence, through the real cache and the real generation path (fix-l48).

    The harm the issue describes end to end: retune the floor, retrain, and be served
    the utterances the old, smaller pool wrote — with the retuning sitting in the diff
    apparently doing nothing.
    """
    workflow_dir = str(tmp_path / "workflow")
    os.mkdir(workflow_dir)
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    set_persona_source(_source(_MatchesOnZip))
    first = CountingCompletion()
    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=first,
        persona_dataset_loader=_loader,
    )
    assert first.calls > 0
    assert provenance.fell_back is False
    assert cache.stats["stored"] == 1, cache.stats

    monkeypatch.setattr(
        personas, "DEFAULT_MIN_POOL_MULTIPLE", DEFAULT_MIN_POOL_MULTIPLE * 2
    )
    set_persona_source(_source(_MatchesOnZip))
    after_the_retune = CountingCompletion()
    generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=after_the_retune,
        persona_dataset_loader=_loader,
    )
    assert after_the_retune.calls > 0, (
        "the entry written under the previous pool floor was reused, so retuning "
        "DEFAULT_MIN_POOL_MULTIPLE silently kept serving the old pool's personas"
    )
    assert cache.stats["hit"] == 0, cache.stats


# ---------------------------------------------------------------------------
# Determinism — the property a cache key cannot do without
# ---------------------------------------------------------------------------

def test_two_independently_built_sources_of_one_class_agree():
    """No instance state, and no `id()`, may reach the digest.

    Constructed twice on purpose: an identity-based representation would pass a
    single-object comparison and miss on the next run — the failure bd fix-k0i.46
    reports for the sibling cache's key.
    """
    assert (
        _source(_MatchesOnZip).pool_source_digest()
        == _source(_MatchesOnZip).pool_source_digest()
    )


def test_the_digest_does_not_depend_on_the_configuration_it_is_paired_with():
    """The code digest and the content fingerprint are separate components on purpose.

    Two sources of one class with different keywords must share a pool-code digest, or
    `describe_fingerprint_divergence` could not tell a developer whether their persona
    file or their persona code moved.
    """
    assert (
        _source(_MatchesOnZip, ["zip"]).pool_source_digest()
        == _source(_MatchesOnZip, ["banking", "loans"]).pool_source_digest()
    )


def test_the_digest_needs_no_corpus():
    """It is computed before the cache decides whether to generate at all.

    Loading PersonaHub to compute a cache key would undo the reason the cache exists, so
    the loader here raises if it is touched.
    """
    def loader_that_must_not_be_called():
        raise AssertionError(
            "computing a pool-code digest loaded the persona corpus"
        )

    source = DomainConditionedPersonaSource(
        loader_that_must_not_be_called, KEYWORDS, num_personas_hint=2
    )
    assert source.pool_source_digest()

    # And through the label, which is where the cache actually reads it from.
    set_persona_source(source)
    assert active_persona_source_label().endswith(source.pool_source_digest())


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_the_digest_is_the_same_in_a_fresh_process(hash_seed):
    """"Deterministic" means across PROCESSES, not merely within one.

    A digest built from set iteration or from ``__dict__`` order could agree with itself
    all day inside one interpreter and still move between runs, because the string hash
    salt is per-process — which is precisely how the sibling cache's key went stale
    invisibly (bd fix-k0i.46). ``PYTHONHASHSEED`` is varied to force the point.
    """
    probe = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {root!r})
        from fastworkflow.train.personas import DomainConditionedPersonaSource
        source = DomainConditionedPersonaSource(
            lambda: [], {keywords!r}, num_personas_hint=2, origin="probe")
        print(source.pool_source_digest())
        """
    ).format(root=PROJECT_ROOT, keywords=KEYWORDS)

    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    in_a_fresh_process = result.stdout.strip().splitlines()[-1]
    assert in_a_fresh_process == _source(
        DomainConditionedPersonaSource
    ).pool_source_digest()


# ---------------------------------------------------------------------------
# The label, and the shipped utterance-cache key
# ---------------------------------------------------------------------------

def test_the_label_carries_the_name_the_content_and_the_code():
    """Three components, so a reported miss is readable rather than merely reported."""
    set_persona_source(_source(_MatchesOnZip))
    name, content, code = active_persona_source_label().split("#")

    assert name == SOURCE_DOMAIN_CONDITIONED
    assert content == _source(_MatchesOnZip).fingerprint()
    assert code == _source(_MatchesOnZip).pool_source_digest()


def test_editing_the_filter_moves_only_the_code_component():
    """The keywords did not change, so the content fingerprint must not either.

    This is what makes the third component diagnostic: a developer reading a miss can
    see that their persona file is untouched and their persona code is not.
    """
    set_persona_source(_source(_MatchesOnZip))
    before = active_persona_source_label().split("#")
    set_persona_source(_source(_MatchesOnOrder))
    after = active_persona_source_label().split("#")

    assert before[:2] == after[:2]
    assert before[2] != after[2]


def test_editing_the_keywords_moves_only_the_content_component():
    """The mirror image, so neither component is doing the other's work."""
    set_persona_source(_source(_MatchesOnZip, ["zip"]))
    before = active_persona_source_label().split("#")
    set_persona_source(_source(_MatchesOnZip, ["banking"]))
    after = active_persona_source_label().split("#")

    assert before[0] == after[0]
    assert before[1] != after[1]
    assert before[2] == after[2]


def _variant_key(source):
    set_persona_source(source)
    return utterance_fingerprint(
        SEED_UTTERANCES,
        COMMAND_NAME,
        2,
        2,
        1,
        MODEL,
        persona_dataset_loader=_loader,
    ).variant_key


def test_editing_the_filter_changes_the_shipped_variant_key():
    """Asserted against the real `utterance_fingerprint`, not a reconstruction of it.

    Before this fix both keys were `domain_conditioned#<same keyword hash>` and the two
    filters shared one cache entry, which is the whole of bd fix-6r5.
    """
    assert _variant_key(_source(_MatchesOnZip)) != _variant_key(
        _source(_MatchesOnOrder)
    )


def test_the_variant_key_is_still_stable_for_an_unchanged_source():
    """Over-invalidation costs money, so prove the key is self-consistent."""
    assert _variant_key(_source(_MatchesOnZip)) == _variant_key(
        _source(_MatchesOnZip)
    )


def test_an_entry_written_under_the_old_filter_is_not_reused(tmp_path):
    """The consequence, through the real cache and the real generation path.

    A run that reused the entry would train on utterances written by personas the tuned
    filter would no longer select — with the tuning sitting right there in the diff,
    apparently doing nothing.
    """
    workflow_dir = str(tmp_path / "workflow")
    os.mkdir(workflow_dir)
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    set_persona_source(_source(_MatchesOnZip))
    first = CountingCompletion()
    _utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=first,
        persona_dataset_loader=_loader,
    )
    assert first.calls > 0
    assert provenance.fell_back is False
    assert cache.stats["stored"] == 1, cache.stats

    set_persona_source(_source(_MatchesOnOrder))
    after_the_edit = CountingCompletion()
    generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=after_the_edit,
        persona_dataset_loader=_loader,
    )
    assert after_the_edit.calls > 0, (
        "the entry written under the previous row filter was reused, so tuning the "
        "filter silently kept serving the personas it replaced"
    )
    assert cache.stats["hit"] == 0, cache.stats


def test_an_unchanged_source_still_reuses_the_entry_it_wrote(tmp_path):
    """The other direction: the fix must not turn every run into a regeneration."""
    workflow_dir = str(tmp_path / "workflow")
    os.mkdir(workflow_dir)
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    for _run in range(2):
        set_persona_source(_source(_MatchesOnZip))
        generate_diverse_utterances_with_provenance(
            SEED_UTTERANCES,
            COMMAND_NAME,
            num_personas=2,
            utterances_per_persona=2,
            personas_per_batch=1,
            seed=42,
            completion_fn=CountingCompletion(),
            persona_dataset_loader=_loader,
        )
    assert cache.stats["hit"] == 1, cache.stats


# ---------------------------------------------------------------------------
# Nothing that was working may move
# ---------------------------------------------------------------------------

def test_the_default_personahub_draw_still_has_no_persona_label():
    """Decision D6. A label here invalidates every cached utterance in every workflow.

    The default source has no configuration to condition on — its pool is the whole
    corpus — so there is nothing for a pool-code digest to protect, and the cost of
    adding one would be paid by every workflow that has never seen a `personas.json`.
    """
    assert active_persona_source_label() is None


def test_the_default_draws_fingerprint_input_carries_no_pool_digest():
    """The same claim one level down, where the key is actually built.

    `persona_source` for a default run must remain the bare corpus identity: if a pool
    digest leaked in, every existing entry would miss on the first run after this fix.
    """
    persona_source = utterance_fingerprint(
        SEED_UTTERANCES,
        COMMAND_NAME,
        2,
        2,
        1,
        MODEL,
        persona_dataset_loader=_loader,
    ).inputs["persona_source"]

    assert persona_source == f"{_loader.__module__}.{_loader.__qualname__}"
    assert "#" not in persona_source


def test_an_app_supplied_source_is_covered_too():
    """Every source that emits a label gets its code digested, not just the domain one."""
    source = AppPersonaSource([Persona(id="app:a", text="A returns clerk.")])
    set_persona_source(source)
    label = active_persona_source_label()

    assert label.count("#") == 2
    assert label.endswith(source.pool_source_digest())


def test_the_three_shipped_sources_have_distinct_pool_digests():
    """Otherwise the component is decorative."""
    digests = {
        PersonaHubSource(_loader).pool_source_digest(),
        AppPersonaSource([Persona(id="app:a", text="One.")]).pool_source_digest(),
        _source(DomainConditionedPersonaSource).pool_source_digest(),
    }
    assert len(digests) == 3


# ---------------------------------------------------------------------------
# The mechanism must survive the next subclass
# ---------------------------------------------------------------------------

def _all_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _all_subclasses(subclass)


def test_no_subclass_overrides_the_pool_digest():
    """`pool_source_digest` is final in the same sense `select` is.

    A subclass that overrode it would be declaring its own cache key, which is the one
    thing a source may not do: it could weaken the digest to a constant and nothing else
    in the system would notice.
    """
    for subclass in _all_subclasses(PersonaSource):
        assert (
            subclass.pool_source_digest is PersonaSource.pool_source_digest
        ), f"{subclass.__qualname__} overrides pool_source_digest"


def test_every_function_a_source_class_defines_is_digested():
    """The maintenance hazard that produced fix-6r5, closed by reflection.

    fix-k0i.43 named four functions in a list; `_matches` and `rows` were two more and
    nothing failed when they were absent. This walks every `PersonaSource` subclass that
    exists — including the ones defined in this file, which is a fair test of the next
    subclass nobody has written yet — and asserts that every function it defines is
    covered.
    """
    for subclass in _all_subclasses(PersonaSource):
        covered = set(implementation_functions(subclass))
        for name, member in vars(subclass).items():
            if isinstance(member, property):
                expected = [member.fget]
            elif isinstance(member, (staticmethod, classmethod)):
                expected = [member.__func__]
            elif callable(member) and not isinstance(member, type):
                expected = [member]
            else:
                continue
            for function in expected:
                assert function in covered, (
                    f"{subclass.__qualname__}.{name} defines behaviour that no cache "
                    f"key would notice an edit to"
                )
