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

import pytest

from fastworkflow.train.generate_synthetic import (
    _is_template_echo,
    generate_utterances_for_personas,
    _TEMPLATE_ECHOES,
)
from fastworkflow.train.utterance_cache import source_digest


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


def test_the_filter_is_inside_the_cache_fingerprint():
    """The post-filter decides which generated rows become training data just as surely as
    the prompt decides which rows exist. If only the generator were digested, editing the
    filter would leave every cached entry looking valid under a filter that had changed."""
    combined = "+".join((
        source_digest(generate_utterances_for_personas),
        source_digest(_is_template_echo),
    ))
    assert source_digest(_is_template_echo) in combined
    assert source_digest(generate_utterances_for_personas) in combined
    # Two distinct functions must not collapse to one digest, or the join is decorative.
    assert source_digest(_is_template_echo) != source_digest(
        generate_utterances_for_personas)


def test_every_entry_in_the_echo_set_is_lowercase():
    """`_is_template_echo` casefolds before comparing, so an uppercase entry in the set
    would be unreachable and silently never match."""
    for entry in _TEMPLATE_ECHOES:
        assert entry == entry.casefold(), f"unreachable echo entry: {entry!r}"
