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

The inspection is over the PARSED program, not over its text. The original version of this
module grepped `inspect.getsource(train_workflow)` for substrings, which made every tripwire
here satisfiable by a comment: commenting out `utterance_cache.set_utterance_cache(cache)`
left the substring in the source and the test green, with production running uncached. The
teardown check was weaker still -- `source.count("finally:") >= 2` is satisfied by any two
unrelated `finally` blocks, including two that contain no teardown at all. Comments and
strings do not survive `ast.parse`, a call inside a `finally` handler is a structural fact,
and call ORDER is a comparison of node positions, so all three become claims about the
program rather than about its formatting. bd fix-czb, bd fix-551.9, bd fix-k0i.47.
"""

import ast
import inspect
import textwrap

import pytest

from fastworkflow.train.__main__ import train_workflow


# handle -> (installed with, torn down with, the call that must happen in between)
#
# The middle column is the point of the whole module: a handle installed AFTER the
# generator it feeds has already run is a no-op that still satisfies "the handle is
# installed". `_generate_dspy_examples_helper` is the parameter-example generator;
# `train` is what reaches synthetic utterance generation and the provenance recorder.
_INSTALLATION_ORDER = {
    "param_example_cache.set_param_example_cache": (
        "param_cache",
        "_generate_dspy_examples_helper",
    ),
    "utterance_cache.set_utterance_cache": ("cache", "train"),
    "determinism.set_provenance_recorder": ("recorder", "train"),
}


def _dotted_name(node: ast.AST) -> str:
    """Render `a.b.c` / `f` from a Call's func expression, or "" for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _call_source(node: ast.Call) -> str:
    """`handle(arg)` for a call, using the argument's own dotted/None rendering."""
    arguments = []
    for argument in node.args:
        if isinstance(argument, ast.Constant) and argument.value is None:
            arguments.append("None")
        else:
            arguments.append(_dotted_name(argument) or "?")
    arguments.extend(f"{keyword.arg}=..." for keyword in node.keywords)
    return f"{_dotted_name(node.func)}({', '.join(arguments)})"


@pytest.fixture(scope="module")
def train_workflow_tree() -> ast.FunctionDef:
    """`train_workflow`'s body as an AST, with its decorator wrapper unwrapped.

    `train_workflow` is decorated, so `inspect.getsource` on the module attribute would
    return the wrapper. `__wrapped__` (set by functools.wraps) names the real function.
    """
    target = inspect.unwrap(train_workflow)
    source = textwrap.dedent(inspect.getsource(target))
    module = ast.parse(source)
    function = module.body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)), (
        f"expected to parse a function definition, got {type(function).__name__}"
    )
    return function


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _calls_named(tree: ast.AST, handle: str) -> list[ast.Call]:
    return [call for call in _calls(tree) if _dotted_name(call.func) == handle]


def _position(node: ast.AST) -> tuple[int, int]:
    return (node.lineno, node.col_offset)


def _finally_calls(tree: ast.AST) -> list[ast.Call]:
    """Every call that appears inside the `finalbody` of some `try` in *tree*."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for statement in node.finalbody:
                found.extend(_calls(statement))
    return found


@pytest.mark.parametrize("handle", sorted(_INSTALLATION_ORDER))
def test_train_workflow_installs_the_module_level_handle(train_workflow_tree, handle):
    """The handle must be CALLED in train_workflow's body, not merely mentioned in it."""
    installs = [
        call
        for call in _calls_named(train_workflow_tree, handle)
        if not (
            len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value is None
        )
    ]
    assert installs, (
        f"{handle} is not called with a cache/recorder in train_workflow. The generator "
        f"it feeds cannot be handed a workflow path, so without this call the production "
        f"train path silently runs uncached -- and no mechanism test will notice, because "
        f"they install the handle themselves. A commented-out call does not count: this "
        f"is an AST check, so comments are not part of the program."
    )


@pytest.mark.parametrize("handle", sorted(_INSTALLATION_ORDER))
def test_every_installed_handle_is_torn_down_inside_a_finally_block(
    train_workflow_tree, handle
):
    """`handle(None)` must sit in a `finally`, not merely somewhere in the function.

    Two separate failures are covered. A handle left installed leaks across workflows --
    train_workflow recurses into child workflows, so a leaked handle would point a
    child's generator at its parent's cache directory. And a teardown on the happy path
    only is no teardown at all: a training failure would leak the handle into whatever
    runs next in the same process, which in the test suite is another workflow.
    """
    teardowns = [
        call
        for call in _finally_calls(train_workflow_tree)
        if _dotted_name(call.func) == handle
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value is None
    ]
    assert teardowns, (
        f"{handle}(None) is not called inside any `finally` block of train_workflow. "
        f"Counting `finally:` occurrences would pass here; being inside the handler is "
        f"what actually guarantees the handle is cleared when training raises."
    )


@pytest.mark.parametrize(
    ("handle", "install_argument", "generator"),
    [
        pytest.param(handle, argument, generator, id=handle)
        for handle, (argument, generator) in sorted(_INSTALLATION_ORDER.items())
    ],
)
def test_the_handle_is_installed_before_the_generator_it_feeds_runs(
    train_workflow_tree, handle, install_argument, generator
):
    """Ordering is the whole point: install < generate < teardown.

    Installing the handle after the generator has already been called would be a no-op
    that still passes a "handle is installed" check, and tearing it down before the
    generator runs is the same bug with the statements in a different order. This was
    only checked for the parameter-example cache; the utterance cache and the provenance
    recorder -- the two that make a training run reproducible at all -- had no ordering
    check. bd fix-k0i.47.
    """
    installs = [
        call
        for call in _calls_named(train_workflow_tree, handle)
        if [_dotted_name(argument) for argument in call.args] == [install_argument]
    ]
    assert installs, (
        f"no {handle}({install_argument}) call found; the install argument this test "
        f"names has been renamed, so the ordering claim below cannot be checked"
    )
    generates = _calls_named(train_workflow_tree, generator)
    assert generates, f"no call to {generator}() found in train_workflow"
    teardowns = [
        call
        for call in _calls_named(train_workflow_tree, handle)
        if len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value is None
    ]
    assert teardowns, f"no {handle}(None) teardown found in train_workflow"

    install_at = min(_position(call) for call in installs)
    generate_at = min(_position(call) for call in generates)
    teardown_at = max(_position(call) for call in teardowns)
    assert install_at < generate_at < teardown_at, (
        f"{handle} must be installed before {generator}() runs and torn down after it; "
        f"got install at {install_at}, {generator} at {generate_at}, teardown at "
        f"{teardown_at}"
    )


def test_the_generator_call_is_bracketed_by_the_try_that_tears_the_handle_down(
    train_workflow_tree,
):
    """Each generator must run INSIDE the `try` whose `finally` clears its handle.

    Install-then-generate-then-teardown in that textual order is still leaky if the
    generator is not inside the protected block -- an exception from it would skip a
    teardown that merely happens to appear later in the function. Checking containment
    is what makes the guarantee structural.
    """
    protected: dict[str, set[str]] = {}
    for node in ast.walk(train_workflow_tree):
        if not isinstance(node, ast.Try):
            continue
        cleared = {
            _dotted_name(call.func)
            for call in _finally_calls(node)
            if len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value is None
        }
        if not cleared:
            continue
        guarded = {
            _dotted_name(call.func)
            for statement in node.body
            for call in _calls(statement)
        }
        for handle in cleared:
            protected.setdefault(handle, set()).update(guarded)

    for handle, (_argument, generator) in sorted(_INSTALLATION_ORDER.items()):
        assert generator in protected.get(handle, set()), (
            f"{generator}() does not run inside the try whose finally calls "
            f"{handle}(None). A failure in {generator}() would leak the handle into the "
            f"next workflow trained in this process. Guarded calls found for {handle}: "
            f"{sorted(protected.get(handle, set()))}"
        )


def test_the_ast_helpers_see_the_real_calls_they_claim_to_see(train_workflow_tree):
    """Anti-vacuity guard for this module itself.

    Every assertion above is a search over parsed nodes, and a search that silently finds
    nothing is the failure mode these tests were written to remove in the first place. So
    pin that the walk really does reach `train_workflow`'s calls, that `_dotted_name`
    renders the dotted handles rather than empty strings, and that `_finally_calls` is
    narrower than "every call in the function".
    """
    all_calls = _calls(train_workflow_tree)
    assert len(all_calls) > 20, (
        f"only {len(all_calls)} calls parsed out of train_workflow; the AST walk is not "
        f"seeing the function body"
    )
    rendered = {_call_source(call) for call in all_calls}
    for handle in _INSTALLATION_ORDER:
        assert any(name.startswith(f"{handle}(") for name in rendered), (
            f"{handle} was not rendered by _call_source; the dotted-name helper is broken"
        )
    finally_calls = _finally_calls(train_workflow_tree)
    assert finally_calls, "no calls found in any finally block"
    assert len(finally_calls) < len(all_calls), (
        "every call in train_workflow appears to be inside a finally block, which means "
        "the finalbody walk is matching too much"
    )
