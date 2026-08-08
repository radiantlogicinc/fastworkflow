"""Integration tests for the selective-retraining closure (R5, bd fix-551.10).

The closure rule is the whole point of this module, so these tests exercise the two
inheritance axes independently:

  * ``base``   -- a changed command reaching a context through command inheritance
  * ``parent`` -- a changed command in an ancestor invalidating a descendant's
                  ``wildcard`` class, even though the descendant's own label space
                  never mentions that command

The closure functions are pure over plain data, so these are real end-to-end
exercises of the rule rather than mocks of anything.
"""

import hashlib
import importlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastworkflow.train import selective_training as st
from fastworkflow.train.selective_training import (
    CommandFingerprint,
    TrainingPlan,
    changed_commands,
    close_dirty_contexts,
    descendants_of,
    format_plan,
)


# A hierarchy with both axes present.
#
#   Workspace (root)
#     |- ReviewTicket      parent: Workspace
#     |- Account           parent: Workspace
#     |- AdminAccount      parent: Workspace, base: Account
#
# AdminAccount inherits Account/close through `base`, so a change to Account/close
# reaches AdminAccount's label space without any hierarchy relationship between them.
CONTEXT_COMMANDS = {
    "Workspace": {"Workspace/bulk_decide", "Workspace/list_all"},
    "ReviewTicket": {"ReviewTicket/certify_approve"},
    "Account": {"Account/close"},
    "AdminAccount": {"Account/close", "AdminAccount/force_close"},
}

CONTEXT_ANCESTORS = {
    "Workspace": [],
    "ReviewTicket": ["Workspace"],
    "Account": ["Workspace"],
    "AdminAccount": ["Workspace"],
}


def test_ancestor_change_invalidates_every_descendant_wildcard_class():
    """A command changing in an ancestor dirties every descendant.

    None of the descendants list Workspace/bulk_decide in their own label space; they
    are pulled in solely because ancestor utterances constitute their wildcard class.
    This is the dependency that makes naive per-command retraining wrong.
    """
    reasons = close_dirty_contexts(
        {"Workspace/bulk_decide"}, CONTEXT_COMMANDS, CONTEXT_ANCESTORS
    )

    assert set(reasons) == {"Workspace", "ReviewTicket", "Account", "AdminAccount"}
    assert any("label space" in r for r in reasons["Workspace"])
    for descendant in ("ReviewTicket", "Account", "AdminAccount"):
        assert any("wildcard class is stale" in r for r in reasons[descendant])
        assert any("Workspace" in r for r in reasons[descendant])


def test_base_inheritance_is_a_second_closure_axis():
    """A changed command reaches every context that inherits it via `base`.

    AdminAccount has no hierarchy relationship to Account -- they are siblings under
    Workspace -- so only command inheritance can explain its inclusion.
    """
    reasons = close_dirty_contexts({"Account/close"}, CONTEXT_COMMANDS, CONTEXT_ANCESTORS)

    assert set(reasons) == {"Account", "AdminAccount"}
    assert "Workspace" not in reasons
    assert "ReviewTicket" not in reasons
    assert any("Account/close" in r for r in reasons["AdminAccount"])


def test_base_and_parent_axes_compose():
    """A command reaches a context that is on neither its base nor its parent chain.

    This is the composition the adversarial review (AR2) identified as the real gap in
    the original closure rule: expanding over `parent` alone misses it, and expanding
    over `base` alone misses it too. Only the union catches it.

        Account            owns Account/close
        AdminAccount       base:   [Account]        -> inherits Account/close
        AdminSession       parent: [AdminAccount]   -> wildcard class carries it

    AdminSession is not a descendant of Account and does not inherit from it, yet its
    wildcard class is built from AdminAccount's utterances, which include Account/close.
    """
    context_commands = {
        "Account": {"Account/close"},
        "AdminAccount": {"Account/close", "AdminAccount/force_close"},
        "AdminSession": {"AdminSession/end"},
    }
    context_ancestors = {
        "Account": [],
        "AdminAccount": [],
        "AdminSession": ["AdminAccount"],
    }

    reasons = close_dirty_contexts({"Account/close"}, context_commands, context_ancestors)

    assert set(reasons) == {"Account", "AdminAccount", "AdminSession"}
    assert any("wildcard class is stale" in r for r in reasons["AdminSession"])
    assert any("AdminAccount" in r for r in reasons["AdminSession"])


def test_leaf_change_stays_local():
    reasons = close_dirty_contexts(
        {"ReviewTicket/certify_approve"}, CONTEXT_COMMANDS, CONTEXT_ANCESTORS
    )
    assert set(reasons) == {"ReviewTicket"}


def test_no_dirty_commands_closes_to_nothing():
    assert close_dirty_contexts(set(), CONTEXT_COMMANDS, CONTEXT_ANCESTORS) == {}


def test_a_context_can_be_dirty_on_both_axes_at_once():
    reasons = close_dirty_contexts(
        {"Workspace/bulk_decide", "Account/close"}, CONTEXT_COMMANDS, CONTEXT_ANCESTORS
    )
    assert len(reasons["AdminAccount"]) == 2


def test_descendants_of_finds_the_full_subtree():
    assert descendants_of("Workspace", CONTEXT_ANCESTORS) == {
        "ReviewTicket",
        "Account",
        "AdminAccount",
    }
    assert descendants_of("Account", CONTEXT_ANCESTORS) == set()


def _fingerprint(name, source="a", seeds="s"):
    return CommandFingerprint(
        command_name=name,
        source_path=f"/tmp/{name}.py",
        source_sha256=source,
        seed_utterances_sha256=seeds,
    )


def test_changed_commands_detects_edits_additions_and_removals():
    previous = {"a": _fingerprint("a"), "b": _fingerprint("b")}
    current = {
        "a": _fingerprint("a"),
        "b": _fingerprint("b", source="edited"),
        "c": _fingerprint("c"),
    }

    assert changed_commands(previous, current) == {"b", "c"}

    # A removal counts too: the removed command's utterances were part of some other
    # context's wildcard class, so that context is now stale.
    assert changed_commands(previous, {"a": _fingerprint("a")}) == {"b"}


def test_a_seed_list_edit_alone_marks_a_command_dirty():
    previous = {"a": _fingerprint("a", seeds="one")}
    current = {"a": _fingerprint("a", seeds="two")}
    assert changed_commands(previous, current) == {"a"}


def test_fingerprints_round_trip_through_json():
    original = _fingerprint("Account/close")
    restored = CommandFingerprint.model_validate(json.loads(original.model_dump_json()))
    assert restored == original


# ---------------------------------------------------------------------
# Fail-closed comparison: an unresolved fingerprint must never compare equal
#
# This is the M12 post-mortem's headline -- "a comparison that can fail must not fail
# equal". Two consecutive runs that both failed to read a command's source produced two
# identical `(None, None)` fingerprints, so the command looked unchanged forever and its
# context was carried forward on a stale model with no signal anywhere. The shipped fix is
# the `resolved` flag; nothing in the suite exercised it, so reverting
# `training_inputs_differ` to a plain field comparison -- the exact regression -- passed
# everything. bd fix-k0i.28.
# ---------------------------------------------------------------------


def _unresolved(name, reason="source file could not be read"):
    return st._unresolvable(name, reason)


def test_an_unresolvable_fingerprint_records_why_it_could_not_be_computed():
    """The reason is the only thing that tells a developer why a run retrained."""
    fingerprint = _unresolved("Account/close", "source file could not be read: /gone.py")

    assert fingerprint.resolved is False
    assert fingerprint.unresolved_reason == "source file could not be read: /gone.py"
    assert fingerprint.source_sha256 is None
    assert fingerprint.seed_utterances_sha256 is None


def test_two_unresolved_fingerprints_differ_even_though_their_fields_are_identical():
    """The `(None, None) == (None, None)` bug, asserted directly.

    Both sides carry exactly the same field values -- that is the whole point. A plain
    field comparison returns "unchanged" here, which is an inability to check being
    reported as a check that passed.
    """
    previous = _unresolved("Account/close")
    current = _unresolved("Account/close")

    assert previous.source_sha256 == current.source_sha256 is None
    assert previous.seed_utterances_sha256 == current.seed_utterances_sha256 is None
    assert current.training_inputs_differ(previous) is True
    assert previous.training_inputs_differ(current) is True


def test_an_unresolved_fingerprint_differs_from_itself():
    """Not reflexive, deliberately: "I could not check this" is never "this is the same"."""
    fingerprint = _unresolved("Account/close")
    assert fingerprint.training_inputs_differ(fingerprint) is True


@pytest.mark.parametrize("unresolved_side", ["previous", "current"])
def test_either_side_being_unresolved_is_enough_to_differ(unresolved_side):
    """A command that stopped being readable, and one that started being readable again.

    Both directions matter: the first is a source that vanished under a still-published
    model, the second is a baseline recorded during an outage. Neither may be treated as
    "unchanged", because in both cases one of the two hashes does not exist.
    """
    resolved = _fingerprint("Account/close")
    unresolved = _unresolved("Account/close")
    previous, current = (
        (unresolved, resolved) if unresolved_side == "previous"
        else (resolved, unresolved)
    )

    assert current.training_inputs_differ(previous) is True


def test_two_resolved_identical_fingerprints_still_compare_equal():
    """The other half of the contract: failing closed must not mean failing always.

    Without this, "return True" would satisfy every assertion above and turn every
    automatic incremental run into a full retrain.
    """
    previous = _fingerprint("Account/close")
    current = _fingerprint("Account/close")

    assert previous.resolved is True
    assert current.training_inputs_differ(previous) is False


def test_changed_commands_reports_an_unresolved_command_as_dirty():
    """The closure input, not just the predicate: `changed_commands` must pass it through.

    This is where the fail-closed comparison actually reaches the plan -- a command absent
    from the dirty set is a command whose context can be carried forward.
    """
    previous = {"a": _unresolved("a"), "b": _fingerprint("b")}
    current = {"a": _unresolved("a"), "b": _fingerprint("b")}

    assert changed_commands(previous, current) == {"a"}


def test_unresolved_state_survives_the_json_round_trip():
    """The baseline is READ BACK from `training_signature.json` before it is compared.

    If `resolved` did not round-trip it would validate back to its `True` default, and the
    fail-closed branch would be bypassed for every comparison against a stored baseline --
    which is every comparison the trainer actually makes.
    """
    original = _unresolved("Account/close", "could not hydrate command metadata: boom")
    restored = CommandFingerprint.model_validate(json.loads(original.model_dump_json()))

    assert restored.resolved is False
    assert restored.unresolved_reason == "could not hydrate command metadata: boom"
    assert restored.training_inputs_differ(original) is True

    signature = st.TrainingSignature(command_fingerprints={"Account/close": original})
    reloaded = st.TrainingSignature.model_validate(
        json.loads(signature.model_dump_json()))
    assert reloaded.command_fingerprints["Account/close"].resolved is False
    assert changed_commands(
        reloaded.command_fingerprints, signature.command_fingerprints
    ) == {"Account/close"}


def test_format_plan_states_both_retrained_and_carried_forward():
    """R5 requires the developer to be able to see the closure, not just trust it."""
    plan = TrainingPlan(
        dirty_commands=["Workspace/bulk_decide"],
        contexts_to_train=["ReviewTicket", "Workspace"],
        contexts_carried_forward=["Unrelated"],
        reasons={
            "Workspace": ["label space contains changed command(s): Workspace/bulk_decide"],
            "ReviewTicket": ["wildcard class is stale: ancestor 'Workspace' changed"],
        },
    )

    rendered = format_plan(plan)
    assert "retraining 2 context(s)" in rendered
    assert "carrying forward 1 context(s): Unrelated" in rendered
    assert "wildcard class is stale" in rendered


def test_full_retrain_plan_renders_distinctly():
    plan = TrainingPlan(contexts_to_train=["A", "B"], is_full_retrain=True)
    assert "full retrain of 2 context(s)" in format_plan(plan)


@pytest.mark.parametrize(
    ("source_path", "signature_field"),
    [
        ("fastworkflow/model_pipeline_training.py", "trainer_source_digest"),
        ("fastworkflow/train/generate_synthetic.py", "generator_source_digest"),
        ("fastworkflow/train/class_balance.py", "class_balance_source_digest"),
    ],
)
def test_helper_edits_in_training_modules_change_global_signature(
    tmp_path, source_path, signature_field
):
    """Every helper module that shapes training participates in the global signature."""
    source_copy = tmp_path / source_path.rsplit("/", 1)[-1]
    source_copy.write_bytes(
        (Path(__file__).resolve().parents[1] / source_path).read_bytes()
    )
    module_name = f"_selective_digest_{source_copy.stem}"

    original_spec = importlib.util.spec_from_file_location(module_name, source_copy)
    original_digest = st._module_source_digest(module_name, original_spec)

    # This helper is deliberately outside the formerly digested function bodies.
    with source_copy.open("a", encoding="utf-8") as source:
        source.write(
            "\n\ndef _selective_training_digest_test_helper():\n"
            "    return 'changed helper implementation'\n"
        )
    changed_spec = importlib.util.spec_from_file_location(module_name, source_copy)
    changed_digest = st._module_source_digest(module_name, changed_spec)

    assert changed_digest != original_digest
    previous = st.TrainingSignature(**{signature_field: original_digest})
    current = st.TrainingSignature(**{signature_field: changed_digest})
    differences = st._diff_global_inputs(previous, current)
    assert len(differences) == 1
    assert differences[0].startswith(f"{signature_field} changed:")


def test_module_source_digest_includes_comments(tmp_path):
    """Comments-only invalidation is conservative and intentional."""
    source_path = tmp_path / "comment_digest.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    module_name = "_selective_comment_digest"
    original_spec = importlib.util.spec_from_file_location(module_name, source_path)
    original_digest = st._module_source_digest(module_name, original_spec)

    source_path.write_text("VALUE = 1\n# rationale changed\n", encoding="utf-8")
    changed_spec = importlib.util.spec_from_file_location(module_name, source_path)

    assert st._module_source_digest(module_name, changed_spec) != original_digest


def test_module_source_digest_reads_zipimport_source(tmp_path):
    """Zip-installed source remains digestible without importing the module."""
    module_name = "_selective_zip_digest"
    archive_path = tmp_path / "training_source.zip"
    source = "def helper():\n    return 'from zip source'\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{module_name}.py", source)

    sys.path.insert(0, str(archive_path))
    importlib.invalidate_caches()
    try:
        assert st._module_source_digest(module_name) == hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
    finally:
        sys.path.remove(str(archive_path))
        importlib.invalidate_caches()


class _SourcelessLoader:
    def get_source(self, _module_name):
        return None


def test_sourceless_frozen_module_forces_never_matching_global_input():
    """Frozen builds without retrievable source retrain rather than carry stale models."""
    module_name = "_selective_frozen_digest"
    frozen_spec = SimpleNamespace(origin="frozen", loader=_SourcelessLoader())

    def read_frozen_source():
        return st._module_source_digest(module_name, frozen_spec)

    first = st._global_input(read_frozen_source, "frozen trainer source")
    second = st._global_input(read_frozen_source, "frozen trainer source")

    assert first.startswith("unavailable:")
    assert second.startswith("unavailable:")
    assert first != second
