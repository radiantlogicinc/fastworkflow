"""The prompt's format example uses the literal word "utterance" as its placeholder, and
models periodically copy that scaffolding into their answer. Neither pre-existing filter
catches it -- "utterance" is nine characters, so the `len(u) > 3` guard passes it, and it
does not start with '[' -- so the token was trained as a positive example of whatever
command was being generated. It was measured at 21 of 412 rows (5.1%) on retail, spread
across seven real commands at once, which means the classifier was shown one identical
string carrying seven conflicting labels.

These tests pin the filter that removes it, and -- more importantly -- pin the boundary,
because a filter that drops real utterances is a worse bug than the one it fixes.

bd fix-iy0.
"""

import json
import os
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.train.determinism import UtteranceProvenance
from fastworkflow.train.generate_synthetic import (
    _DIGESTED_GENERATION_SOURCES,
    _is_template_echo,
    generate_diverse_utterances_with_provenance,
    generate_utterances_for_personas,
    generation_source_digest,
    utterance_fingerprint,
    _TEMPLATE_ECHOES,
)
from fastworkflow.train.utterance_cache import (
    UtteranceCache,
    get_utterance_cache,
    set_utterance_cache,
    source_digest,
)

COMMAND_NAME = "add_two_numbers"
SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]
MODEL = "mistral/mistral-small-latest"

# The scaffolding the prompt's format example shows the model, in the shapes it actually
# arrives in. "..." is deliberately absent: it is three characters, so the pre-existing
# `len(u) > 3` guard already drops it, and including it here would let this test pass
# with the echo filter deleted.
SCAFFOLDING_LINES = (
    "utterance",
    "utterances",
    "- utterance",
    "**utterance**",
    "Persona_Name",
    "Next_Persona_Name",
)

REAL_LINES = ("add 7 and 11 for me", "what do 4 and 9 come to")


class ScaffoldingCompletion:
    """A completion backend that answers half in real utterances, half in scaffolding.

    This is the real `completion_fn` parameter of the production generation loop, not a
    mock of anything: the loop was given the injection point precisely so it can be
    driven without a network call (see its docstring). The response interleaves the two
    kinds of line because the measured defect was a model half-filling the template --
    real utterances AND copied placeholders in the same block.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        persona_name = (
            kwargs["messages"][0]["content"].split("[")[1].split("]")[0]
        )
        body = "\n".join((REAL_LINES[0], *SCAFFOLDING_LINES, REAL_LINES[1]))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"[{persona_name}]\n{body}\n")
                )
            ]
        )


def _local_persona_dataset():
    """`len()` plus `[i]['persona']` is the whole interface generation uses."""
    return [{"persona": f"A person who is persona number {i}."} for i in range(8)]


@pytest.fixture
def training_env():
    """Pin the generation env so the fingerprint does not depend on the local machine."""
    previous = dict(fastworkflow._env_vars)
    fastworkflow.init({
        **previous,
        "LLM_SYNDATA_GEN": MODEL,
        "LITELLM_API_KEY_SYNDATA_GEN": "test-key-not-used-by-the-injected-backend",
        "TRAINING_SEED": "42",
    })
    yield
    fastworkflow.init(previous)


@pytest.fixture
def clean_cache_sink():
    """Guarantee the process-wide cache sink is restored after each test."""
    previous = get_utterance_cache()
    set_utterance_cache(None)
    yield
    set_utterance_cache(previous)


@pytest.mark.parametrize(
    "echo",
    [
        "utterance",
        "Utterance",
        "UTTERANCE",
        "utterances",
        "- utterance",
        "* utterance",
        "1. utterance",
        "2) utterance",
        "  utterance  ",
        "**utterance**",
        "`utterance`",
        "<utterance>",
        "...",
        "persona_name",
        "Next_Persona_Name",
    ],
)
def test_scaffolding_copied_from_the_prompt_is_dropped(echo):
    assert _is_template_echo(echo)


@pytest.mark.parametrize(
    "real",
    [
        # The whole risk of this filter is that it eats real user language. Every string
        # here CONTAINS the trigger word but is a legitimate thing to say to a workflow.
        "what utterance did I just say",
        "show me the utterances for this command",
        "utterance count",
        "list all product types",
        "cancel my order",
        "32 * 4",
        "id=3636",
        "...and then what",
        "why is the persona_name field empty",
    ],
)
def test_real_utterances_that_merely_mention_the_word_survive(real):
    assert not _is_template_echo(real)


def test_the_echo_set_is_matched_whole_not_as_a_substring():
    """A substring match would delete every utterance discussing utterances. The measured
    defect is an exact copy of a one-word placeholder, so an exact match is what fixes it."""
    assert _is_template_echo("utterance")
    assert not _is_template_echo("utterance about my order")
    assert not _is_template_echo("my utterance")


def test_the_filter_is_inside_the_cache_fingerprint(training_env):
    """The post-filter decides which generated rows become training data just as surely as
    the prompt decides which rows exist. If only the generator were digested, editing the
    filter would leave every cached entry looking valid under a filter that had changed.

    Asserted against the SHIPPED fingerprint. This test previously rebuilt the digest join
    in its own body and asserted that its own reconstruction contained what it had just
    put there, which is true for any implementation of `utterance_fingerprint` including
    one that digests nothing -- deleting `_is_template_echo` from
    `_DIGESTED_GENERATION_SOURCES` failed nothing. bd fix-k0i.27.
    """
    digest = utterance_fingerprint(
        SEED_UTTERANCES,
        COMMAND_NAME,
        2,
        2,
        1,
        MODEL,
    ).inputs["generator_source_digest"]

    # Each component is labelled `qualname:digest` so a stale-cache miss names the
    # function that moved, so assert on the labelled form rather than the bare hash.
    assert f"_is_template_echo:{source_digest(_is_template_echo)}" in digest
    assert (
        f"generate_utterances_for_personas:"
        f"{source_digest(generate_utterances_for_personas)}"
    ) in digest
    # Two distinct functions must not collapse to one digest, or the join is decorative.
    assert source_digest(_is_template_echo) != source_digest(
        generate_utterances_for_personas)
    # The digest is the real join, not a substring of some longer opaque value: it must
    # be exactly what `generation_source_digest` produces.
    assert digest == generation_source_digest()
    assert _is_template_echo in _DIGESTED_GENERATION_SOURCES


def test_editing_the_filter_would_change_the_cache_key(training_env):
    """The consequence the digest exists for, stated as a property of the real key.

    `source_digest` hashes a function's source TEXT, so a filter edit necessarily moves
    the component this test locates inside the shipped fingerprint -- which is what makes
    every existing cache entry miss. Pinning the component's position (rather than a
    hard-coded hash, which would need updating on every unrelated edit) is what keeps this
    a claim about `utterance_fingerprint` and not about today's bytes.
    """
    fingerprint = utterance_fingerprint(
        SEED_UTTERANCES, COMMAND_NAME, 2, 2, 1, MODEL)
    components = fingerprint.inputs["generator_source_digest"].split("+")
    filter_components = [
        component for component in components
        if component.startswith("_is_template_echo:")
    ]
    assert len(filter_components) == 1, (
        f"expected exactly one _is_template_echo component in the fingerprint, got "
        f"{filter_components} out of {components}"
    )
    # Substituting a different digest for that one component must change the variant key,
    # which is what "editing the filter invalidates every entry" means operationally.
    assert filter_components[0].split(":", 1)[1] == source_digest(_is_template_echo)
    assert fingerprint.variant_key == utterance_fingerprint(
        SEED_UTTERANCES, COMMAND_NAME, 2, 2, 1, MODEL).variant_key


def test_every_entry_in_the_echo_set_is_lowercase():
    """`_is_template_echo` casefolds before comparing, so an uppercase entry in the set
    would be unreachable and silently never match."""
    for entry in _TEMPLATE_ECHOES:
        assert entry == entry.casefold(), f"unreachable echo entry: {entry!r}"


# ---------------------------------------------------------------------------
# The filter inside the real generation loop (bd fix-k0i.26)
#
# Every test above calls `_is_template_echo` directly, which pins the predicate but says
# nothing about whether the loop applies it. Deleting
# `utterances = [u for u in utterances if not _is_template_echo(u)]` from
# `generate_utterances_for_personas` left the whole suite green -- so the 5.1%-of-rows
# contamination this filter exists to remove could return silently. These tests drive the
# real generation entry points with the production `completion_fn` injection point.
# ---------------------------------------------------------------------------


def _run_generation_loop(backend, personas=("A person who returns things.",)):
    provenance = UtteranceProvenance(command_name=COMMAND_NAME, seed=42)
    generated = generate_utterances_for_personas(
        SEED_UTTERANCES,
        COMMAND_NAME,
        list(personas),
        [str(index) for index in range(len(personas))],
        provenance,
        utterances_per_persona=2,
        personas_per_batch=1,
        model=MODEL,
        seed=42,
        completion_fn=backend,
    )
    return generated, provenance


def test_scaffolding_emitted_by_the_generator_never_reaches_the_returned_utterances():
    """The loop must drop the echo, not merely be able to recognise it.

    A row that survives here is trained as a POSITIVE example of whatever command was
    being generated. The measured damage was one identical string carrying seven
    conflicting labels across seven real commands at once.
    """
    backend = ScaffoldingCompletion()
    generated, _provenance = _run_generation_loop(backend)

    assert backend.calls == 1, "the injected backend was not the one that was called"
    for line in SCAFFOLDING_LINES:
        assert line not in generated, (
            f"the prompt's own placeholder {line!r} became a training utterance for "
            f"{COMMAND_NAME}"
        )
    # And the real rows in the same response survived: a filter that ate them would be a
    # worse bug than the one it fixes.
    for line in REAL_LINES:
        assert line in generated


def test_a_dropped_echo_is_not_attributed_to_a_persona_either():
    """Attribution is what the held-out evaluation splits on, so it must not see the echo.

    `_attribute_utterance` runs over the post-filter list. An echo that reached the
    attribution map would appear in `training_provenance.json` as a real row produced by a
    real persona, which is how a filtered row would still end up counted.
    """
    _generated, provenance = _run_generation_loop(ScaffoldingCompletion())

    for line in SCAFFOLDING_LINES:
        assert line not in provenance.utterance_personas
    assert set(REAL_LINES) <= set(provenance.utterance_personas)
    assert provenance.fell_back is False


def test_the_echo_is_dropped_for_every_persona_batch():
    """One echo per batch, so a filter applied only to the first section is caught.

    The loop filters inside the per-section loop; hoisting the filter out of it, or
    applying it before the sections are split, would leave later batches contaminated.
    """
    personas = tuple(f"A person who is persona number {index}." for index in range(3))
    generated, _provenance = _run_generation_loop(
        ScaffoldingCompletion(), personas=personas)

    assert generated.count(REAL_LINES[0]) == len(personas), (
        "each batch should have contributed its real rows"
    )
    for line in SCAFFOLDING_LINES:
        assert line not in generated


def test_scaffolding_is_absent_from_the_cache_entry_the_run_writes(
    tmp_path, training_env, clean_cache_sink
):
    """A cached echo is permanent: every later run at this seed reuses it.

    The filter running but the cache being written from the pre-filter list would put the
    contamination beyond the reach of the fix -- so this asserts on the JSON that actually
    lands under `___command_info/utterance_cache/`, through the real
    `generate_diverse_utterances_with_provenance` path the trainer calls.
    """
    workflow_dir = str(tmp_path / "workflow")
    os.mkdir(workflow_dir)
    cache = UtteranceCache(workflow_dir)
    set_utterance_cache(cache)

    utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        COMMAND_NAME,
        num_personas=2,
        utterances_per_persona=2,
        personas_per_batch=1,
        seed=42,
        completion_fn=ScaffoldingCompletion(),
        persona_dataset_loader=_local_persona_dataset,
    )

    assert provenance.fell_back is False
    assert cache.stats["stored"] == 1, (
        f"nothing was cached, so this test would pass vacuously: {cache.stats}"
    )

    entry_files = [
        os.path.join(cache.root, name)
        for name in os.listdir(cache.root)
        if name.endswith(".json")
    ]
    assert len(entry_files) == 1, entry_files
    payload = json.loads(open(entry_files[0], encoding="utf-8").read())
    cached_rows = [
        row
        for entry in payload["entries"].values()
        for row in entry["generated_utterances"]
    ]
    assert cached_rows, f"the cache entry holds no generated rows: {payload}"

    for line in SCAFFOLDING_LINES:
        assert line not in cached_rows, (
            f"{line!r} was written to the utterance cache, so every future run at this "
            f"seed will train on it even after the filter is fixed"
        )
        assert line not in utterances
    assert set(REAL_LINES) <= set(cached_rows)


# ---------------------------------------------------------------------------
# fix-k0i.26 (S3 tail): scaffolding that arrives numbered or punctuated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "echo",
    [
        "utterance 1",
        "Utterance 12",
        "Utterance:",
        "utterance:",
        "Persona_3",
        "persona 2",
        "1. Utterance:",
        "- utterance 4",
        "**utterance**",
    ],
    ids=[
        "numbered", "numbered-two-digit", "label-colon", "lower-colon",
        "persona-underscore-index", "persona-space-index", "marker-and-colon",
        "marker-and-number", "bold",
    ],
)
def test_numbered_and_punctuated_scaffolding_is_dropped(echo):
    """These three shapes slipped through: the index is a SUFFIX so the leading
    list-marker pattern could not reach it, ':' was absent from the strip set, and
    a bare persona header was not in the echo vocabulary at all."""
    assert _is_template_echo(echo) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "add 2 and 3",
        "what is 2 plus 3",
        "set priority to 3",
        "call user 5",
        "delete item 7",
        "show me utterance history",
        "my persona is a chef",
        "...and then what",
    ],
)
def test_real_phrasings_that_end_in_a_number_are_kept(utterance):
    """The trailing-index strip is the risky half of this fix.

    It runs only as a SECOND attempt, after the whole string has already failed to
    match, so the only strings it can reach are ones that are otherwise nothing but
    scaffolding. A real phrasing ending in a number keeps it.
    """
    assert _is_template_echo(utterance) is False


def test_bare_ellipsis_is_still_dropped():
    """Regression guard for the fix itself.

    '...' is an echo, and the label-punctuation strip added for 'Utterance:' empties
    it to '' -- which matches nothing. The unpunctuated form is therefore tested
    first, and this is the test that fails if that ordering is ever reversed.
    """
    assert _is_template_echo("...") is True
