"""Guard tests for the *installation* of the two LLM-generation caches, as opposed to their
mechanism.

These exist because of a real gap. Both caches reach their generators through a module-level
handle, and both of their dedicated test modules install that handle themselves so they can
run without a full train. That is reasonable for testing the mechanism, but it means the
entire suite passes identically whether or not `train_workflow` installs the cache in the
production path -- so `fastworkflow train` could silently go back to being non-reproducible
with every test green. The param-example cache was in exactly that state for a while: module
written, 39 mechanism tests passing, production path unwired.

Source inspection is a weak form of test and is used here deliberately, because the strong
form (a real train) costs money and several minutes per run and already exists in
tests/test_param_example_determinism.py. What these add is a cheap tripwire on the wiring
that the expensive tests structurally cannot see.

bd fix-czb, bd fix-551.9.
"""

import inspect

import pytest

from fastworkflow.train.__main__ import train_workflow


@pytest.fixture(scope="module")
def train_workflow_source():
    return inspect.getsource(train_workflow)


@pytest.mark.parametrize(
    "handle",
    [
        "param_example_cache.set_param_example_cache",
        "utterance_cache.set_utterance_cache",
        "determinism.set_provenance_recorder",
    ],
)
def test_train_workflow_installs_the_module_level_handle(train_workflow_source, handle):
    assert handle in train_workflow_source, (
        f"{handle} is not called in train_workflow. The generator it feeds cannot be handed "
        f"a workflow path, so without this call the production train path silently runs "
        f"uncached -- and no mechanism test will notice, because they install the handle "
        f"themselves."
    )


@pytest.mark.parametrize(
    "handle",
    [
        "param_example_cache.set_param_example_cache(None)",
        "utterance_cache.set_utterance_cache(None)",
        "determinism.set_provenance_recorder(None)",
    ],
)
def test_every_installed_handle_is_torn_down(train_workflow_source, handle):
    """A handle left installed leaks across workflows. train_workflow recurses into child
    workflows, so a leaked handle would point a child's generator at its parent's cache
    directory."""
    assert handle in train_workflow_source


def test_teardown_happens_in_a_finally_block(train_workflow_source):
    """If teardown is not in `finally`, a training failure leaks the handle into whatever
    runs next in the same process -- which, in the test suite, is another workflow."""
    assert train_workflow_source.count("finally:") >= 2, (
        "expected at least two finally blocks: one bracketing DSPy example generation and "
        "one bracketing train()"
    )


def test_the_param_cache_is_installed_before_dspy_generation_runs(train_workflow_source):
    """Ordering is the whole point: installing the handle after the generator has already
    been called would be a no-op that still passes a 'handle is installed' check."""
    install = train_workflow_source.index("set_param_example_cache(param_cache)")
    generate = train_workflow_source.index("_generate_dspy_examples_helper(workflow)")
    teardown = train_workflow_source.index("set_param_example_cache(None)")
    assert install < generate < teardown
