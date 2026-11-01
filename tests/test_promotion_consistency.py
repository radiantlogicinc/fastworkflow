"""AR3's promotion-time closure check, and why the shipped substitute was not one.

R4-as-amended (AR3) asks for a check that runs *at promotion* and re-derives R5's
retraining closure as a postcondition, so that a closure bug is caught before the
version becomes current instead of after its stale models start answering users. What
shipped instead was the plan-time signature diff, and the finding this file exists for
(bd fix-k0i.7) is that the diff cannot serve that purpose: both of its sides come out of
the same ``build_context_maps`` call as the closure itself. A deterministic bug in
``commands()`` or the ancestor map reproduces identically in the baseline and in
the current signature, the diff comes out clean, and the bad version publishes.

The first section below demonstrates exactly that, on a real workflow, using the shipped
comparison functions -- it is the premise everything else rests on. The rest exercises
``verify_version_consistency``, which re-derives the closure from files instead:

* laws that hold of any correct hierarchy (ancestry is transitively closed; a context's
  wildcard sources are the union of its ancestors' label spaces),
* the recorded per-context digests against the command fingerprints actually present in
  the same version,
* every carried-forward context against the version its models were carried from --
  the comparison the shipped ``wildcard_sources`` list could not make, because it holds
  command NAMES and an ancestor command whose content changed keeps its name.

No mocks (repo rule ``.cursor/rules/testing_rules.mdc``): every test copies the real
``tests/todo_list_workflow``, builds its real routing definition, computes its real
training signature, and writes real manifests through the shipped writer. The
"buggy closure" inputs are produced by feeding the shipped
``compute_context_signatures`` the ancestor map a buggy ``get_ancestor_contexts`` would
have returned, so everything downstream of the simulated bug is the real code path.

``todo_list_workflow`` is used because it is the only real workflow in the repo with a
three-generation hierarchy (``TodoItem -> TodoList -> TodoListManager -> *``). Ancestry
truncation is invisible in a two-generation one: with a single hop there is nothing for a
truncated chain to omit.

Phase 9 step 5: ``context_hierarchy_model.json`` is retired, so nothing computes an ancestor
map any more and the closure takes none. The AR3 check still reads the ancestry a version
RECORDED (old versions carry it; new ones write ``[]``), so the ancestry-law tests below
write the recorded chain into the manifest directly instead of deriving it from the file.
"""

import ast
import inspect
import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.train import __main__ as train_orchestration
from fastworkflow.train import artifact_versioning as av
from fastworkflow.train import selective_training as st
from fastworkflow.utils.logging import logger as fastworkflow_logger


TODO_WORKFLOW_PATH = os.path.join("tests", "todo_list_workflow")

# The grandparent whose command edits must reach TodoItem's wildcard class, and one of
# its commands. TodoItem neither declares nor inherits it: it arrives purely through the
# ancestor axis, which is the axis a name-based comparison cannot police.
GRANDPARENT_CONTEXT = "TodoListManager"
GRANDPARENT_COMMAND_FILE = os.path.join(
    "_commands", "TodoListManager", "create_todo_list.py"
)
GRANDCHILD_CONTEXT = "TodoItem"

# The chain the retired hierarchy file declared for todo_list_workflow, as a version that
# recorded it would carry it (transitively closed). Written into manifests by the tests that
# exercise AR3's ancestry laws; nothing reads it back from the file.
TODO_RECORDED_ANCESTRY = {
    "*": [],
    "TodoListManager": ["*"],
    "TodoList": ["*", "TodoListManager"],
    "TodoItem": ["*", "TodoList", "TodoListManager"],
}


def _resolve_env_vars() -> dict:
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join(
        "fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}
    for local in (os.path.join("env", ".env"), os.path.join("passwords", ".env")):
        if os.path.exists(local):
            env_vars.update(dotenv_values(local))
    return env_vars


@pytest.fixture(scope="module")
def env_vars() -> dict:
    values = _resolve_env_vars()
    fastworkflow.init(env_vars=values)
    return values


@pytest.fixture
def fastworkflow_logs(caplog):
    """Capture the real `fastWorkflow` logger, which does not propagate to root.

    `utils/logging.py` sets `propagate = False`, so `caplog` alone sees nothing. The
    compatibility policy below is "say so and continue", and a test that cannot see the
    saying would pass on a silent pass.
    """
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(logging.DEBUG)
    fastworkflow_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        fastworkflow_logger.removeHandler(caplog.handler)
        fastworkflow_logger.setLevel(previous_level)


def _copy_workflow(destination_root: Path) -> str:
    workflow_path = str(destination_root / "todo_list_workflow")
    shutil.copytree(
        TODO_WORKFLOW_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    return workflow_path


def _rebuild(workflow_path: str) -> None:
    """Re-derive the routing artifacts after editing a workflow's sources.

    Every cached layer has to be dropped, not just the on-disk JSON: the registry
    memoises per path, so a signature computed after an edit would otherwise describe the
    pre-edit command directory and report nothing changed -- which looks exactly like the
    bug these tests exist to catch.
    """
    RoutingRegistry.clear_registry()
    CommandContextModel.load(workflow_path)
    RoutingDefinition.build(workflow_path)
    RoutingRegistry.get_definition(workflow_path, load_cached=False)


def _signature(workflow_path: str, seed: int = 42, recorded=None) -> st.TrainingSignature:
    contexts = st.contexts_for_training(workflow_path)
    signature, _unresolved = st.compute_training_signature(
        workflow_path, contexts, seed=seed, recorded_samples=recorded)
    return signature


def _recorded(signature: st.TrainingSignature) -> dict:
    """The not-here samples a version recorded, as the planner hands them on (design 2.50)."""
    return {name: context.not_here_samples for name, context in signature.contexts.items()}


def _write_context_artifacts(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
        (folder / artifact).touch()


def _stage_version(
    workflow_path: str,
    signature: st.TrainingSignature,
    plan: st.TrainingPlan,
    contexts: list[str] | None = None,
    recorded_ancestry: bool = False,
) -> str:
    """Do what a training run does between `train()` and `publish_version`.

    The artifact files are empty placeholders: nothing in the consistency check reads
    their contents, only whether a carried-forward context has any, and training them for
    real would put an LLM call and a fine-tune in the middle of a pure-file test. What is
    real here is everything the check does read -- the version layout and the manifest,
    written by the shipped writer from a real signature.
    """
    version_id = av.new_version_id()
    for context_name in contexts if contexts is not None else sorted(signature.contexts):
        _write_context_artifacts(
            av.version_dir(workflow_path, version_id)
            / av.context_folder_name(context_name)
        )
    st.save_training_signature(workflow_path, version_id, signature)
    av.write_manifest(
        workflow_path,
        version_id,
        seed=42,
        contexts_retrained=sorted(plan.contexts_to_train),
        contexts_carried_forward=sorted(plan.contexts_carried_forward),
        **st.context_contribution_manifest_fields(signature, plan),
    )
    if recorded_ancestry:
        manifest = av.read_manifest(workflow_path, version_id)
        for context_name, record in manifest[av.CONTEXT_CONTRIBUTIONS_KEY].items():
            record["ancestors"] = list(TODO_RECORDED_ANCESTRY.get(context_name, []))
        _rewrite_manifest(workflow_path, version_id, manifest)
    return version_id


def _full_retrain_plan(signature: st.TrainingSignature) -> st.TrainingPlan:
    return st.TrainingPlan(
        contexts_to_train=sorted(signature.contexts), is_full_retrain=True)


def _carry_forward_plan(
    signature: st.TrainingSignature, carried: list[str], source: str
) -> st.TrainingPlan:
    return st.TrainingPlan(
        contexts_to_train=sorted(set(signature.contexts) - set(carried)),
        contexts_carried_forward=sorted(carried),
        carry_forward_from=source,
    )


def _rewrite_manifest(workflow_path: str, version_id: str, manifest: dict) -> None:
    """Put *manifest* back on disk verbatim.

    Written directly rather than through `write_manifest`, which merges and would keep
    the value being perturbed. What is on disk after this is exactly what the buggy run
    being simulated would have left there.
    """
    (av.version_dir(workflow_path, version_id) / av.MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _append_utterance(path: str, utterance: str) -> None:
    source = Path(path).read_text()
    marker = "plain_utterances = ["
    assert marker in source, f"{path} has no plain_utterances list to edit"
    index = source.index(marker) + len(marker)
    Path(path).write_text(f'{source[:index]}\n        "{utterance}",{source[index:]}')


def _replace_utterance(path: str, old: str, new: str) -> None:
    source = Path(path).read_text()
    assert source.count(f'"{old}"') == 1, f"{path} must hold {old!r} exactly once"
    Path(path).write_text(source.replace(f'"{old}"', f'"{new}"', 1))


def _truncate_recorded_ancestry(workflow_path: str, version_id: str) -> None:
    """Record the chain a one-hop `get_ancestor_contexts` would have written.

    A plausible, deterministic bug: TodoItem keeps only its immediate parent, while TodoList
    still names its own ancestors. The recorded contradiction is what AR3 must refuse.
    """
    manifest = av.read_manifest(workflow_path, version_id)
    manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]["ancestors"] = ["TodoList"]
    _rewrite_manifest(workflow_path, version_id, manifest)


@pytest.fixture
def workflow(tmp_path: Path, env_vars) -> str:
    workflow_path = _copy_workflow(tmp_path)
    _rebuild(workflow_path)
    yield workflow_path
    RoutingRegistry.clear_registry()


# ---------------------------------------------------------------------
# The premise: the plan-time diff shares inputs with the thing it checks
# ---------------------------------------------------------------------


def test_the_hierarchy_file_is_no_longer_a_source_of_ancestry(workflow: str):
    """Phase 9 step 5 (was: the real hierarchy has a chain long enough to truncate).

    The signature no longer records a parent chain, and the grandparent's commands still
    reach the grandchild's wildcard class, now as one of every foreign command.
    """
    signature = _signature(workflow)

    assert any(
        command.startswith(f"{GRANDPARENT_CONTEXT}/")
        for command in signature.contexts[GRANDCHILD_CONTEXT].wildcard_sources
    ), "the grandparent's commands must reach the grandchild's wildcard class"


def test_a_command_edit_dirties_only_its_own_heads_while_the_recorded_samples_hold(workflow: str):
    """Phase 9 step 6 (was: a content edit to a grandparent command dirtied every head).

    Design 2.50: a foreign head reads one recorded sample of a command, so an edit that leaves the
    sample's seed text in place dirties only the command's own heads. The grandchild keeps its
    sample, so the plan carries it forward, and the signature diff is silent. Whether that carry is
    honest is the publish check's question, answered by the sample, not by the fingerprints.
    """
    baseline_signature = _signature(workflow)

    _append_utterance(
        os.path.join(workflow, GRANDPARENT_COMMAND_FILE),
        "start a brand new checklist for the offsite",
    )
    _rebuild(workflow)
    current_signature = _signature(workflow, recorded=_recorded(baseline_signature))

    changed = st.changed_commands(
        baseline_signature.command_fingerprints, current_signature.command_fingerprints)
    assert changed, "precondition: the edit must move a command fingerprint"
    assert baseline_signature.contexts[GRANDCHILD_CONTEXT].not_here_samples[
        f"{GRANDPARENT_CONTEXT}/create_todo_list"] == current_signature.contexts[
        GRANDCHILD_CONTEXT].not_here_samples[f"{GRANDPARENT_CONTEXT}/create_todo_list"], (
        "precondition: the recorded sample is still a seed of the edited command")

    context_commands, unresolved = st.build_context_maps(
        workflow, sorted(current_signature.contexts))
    assert unresolved == {}
    reasons = st.close_dirty_contexts(changed, context_commands)
    assert GRANDPARENT_CONTEXT in reasons
    assert GRANDCHILD_CONTEXT not in reasons

    assert st._diff_context_signature(
        baseline_signature.contexts[GRANDCHILD_CONTEXT],
        current_signature.contexts[GRANDCHILD_CONTEXT],
    ) == [], "the recorded sample holds, so the grandchild's signature does not move"


def test_recorded_wildcard_sources_cannot_see_an_ancestor_command_edit(workflow: str):
    """The manifest-format half of the finding, stated as a fact about real data.

    AR3 asks for per-context utterance-set FINGERPRINTS. What was recorded is command
    NAMES, and an edit to an ancestor command changes no name -- so the recorded
    description of TodoItem's wildcard class is byte-identical either side of an edit
    that changes what that class is trained on.
    """
    before_signature = _signature(workflow)
    before = before_signature.contexts[GRANDCHILD_CONTEXT]

    _append_utterance(
        os.path.join(workflow, GRANDPARENT_COMMAND_FILE),
        "spin up a fresh list for the sprint",
    )
    _rebuild(workflow)
    after_signature = _signature(workflow, recorded=_recorded(before_signature))
    after = after_signature.contexts[GRANDCHILD_CONTEXT]

    assert before.wildcard_sources == after.wildcard_sources
    assert st._diff_context_signature(before, after) == []

    # The fingerprints AR3 asked for do move, which is what makes the check possible.
    fields_before = st.context_contribution_manifest_fields(
        _signature(workflow), _full_retrain_plan(after_signature))
    assert (
        st.command_utterance_fingerprint(
            after_signature.command_fingerprints[
                f"{GRANDPARENT_CONTEXT}/create_todo_list"]
        )
        in json.dumps(fields_before)
    )


# ---------------------------------------------------------------------
# The postcondition, against real versions on disk
# ---------------------------------------------------------------------


def test_a_real_workflows_version_publishes_and_verifies(workflow: str):
    """The happy path. A guard that refuses everything would pass every test below."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert consistency.unverifiable_reasons == []
    assert consistency.verified is True

    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_the_manifest_records_utterance_set_fingerprints_per_context(workflow: str):
    """AR3's format requirement, asserted on the file a real run writes."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    contributions = manifest[av.CONTEXT_CONTRIBUTIONS_KEY]
    fingerprints = manifest[av.COMMAND_UTTERANCE_FINGERPRINTS_KEY]

    assert manifest[av.CONTRIBUTION_FORMAT_KEY] == av.CONTRIBUTION_FORMAT_VERSION
    assert set(contributions) == set(signature.contexts)
    assert set(fingerprints) == set(signature.command_fingerprints)

    record = contributions[GRANDCHILD_CONTEXT]
    if av.NOT_HERE_SAMPLES_KEY in record:  # design 2.50: a sparse head's digest is over its samples
        assert record[av.NOT_HERE_SAMPLES_KEY]
        assert record["ancestor_utterances_sha256"] == av.not_here_samples_fingerprint(
            record[av.NOT_HERE_SAMPLES_KEY])
    else:
        assert record["ancestor_utterances_sha256"] == av.utterance_set_fingerprint(
            fingerprints, record["wildcard_sources"]
        )
    assert record["own_utterances_sha256"] == av.utterance_set_fingerprint(
        fingerprints, record["label_space"]
    )
    assert record["ancestor_utterances_sha256"] != record["own_utterances_sha256"]


def test_a_truncated_ancestor_chain_is_refused_at_promotion(workflow: str):
    """The bug the plan-time diff cancels out, caught by a law it cannot cancel.

    Ancestry is transitive whatever the hierarchy says, so a chain that stops one hop
    short for TodoItem while TodoList still names its own ancestors contradicts itself on
    paper. Re-running `get_ancestor_contexts` would return the same truncated chain and
    notice nothing; reading what was recorded catches it.
    """
    signature = _signature(workflow)
    version_id = _stage_version(
        workflow, signature, _full_retrain_plan(signature), recorded_ancestry=True)
    _truncate_recorded_ancestry(workflow, version_id)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "not transitively closed" in problem for problem in consistency.problems
    ), consistency.problems
    assert consistency.verified is False

    with pytest.raises(av.ArtifactConsistencyError) as excinfo:
        av.publish_version(workflow, version_id)
    assert GRANDCHILD_CONTEXT in str(excinfo.value)
    assert av.resolve_current_version(workflow) is None, (
        "a refused promotion must not have advanced the pointer"
    )


def test_wildcard_sources_that_are_not_the_ancestors_union_are_refused(workflow: str):
    """The other structural law: what an ancestor teaches must reach its descendants.

    Provoked by dropping one ancestor's command from the recorded wildcard sources --
    what a `commands()` that under-reported in the ancestor role would leave behind.
    """
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    record = manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]
    dropped = next(
        command
        for command in record["wildcard_sources"]
        if command.startswith(f"{GRANDPARENT_CONTEXT}/")
    )
    record["wildcard_sources"] = [
        command for command in record["wildcard_sources"] if command != dropped
    ]
    record["ancestor_utterances_sha256"] = av.utterance_set_fingerprint(
        manifest[av.COMMAND_UTTERANCE_FINGERPRINTS_KEY], record["wildcard_sources"]
    )
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "union of its ancestors' label spaces" in problem
        for problem in consistency.problems
    ), consistency.problems
    assert dropped in " ".join(consistency.problems)


def test_an_ancestor_the_version_does_not_describe_is_refused(workflow: str):
    """The silent-empty case: `context_commands.get(ancestor, set())` returns nothing.

    An ancestor outside the candidate set contributes no commands to its descendants'
    wildcard classes and nothing anywhere says so. Recomputing the same way reproduces
    the same silence; requiring every named ancestor to be described does not.
    """
    signature = _signature(workflow)
    version_id = _stage_version(
        workflow, signature, _full_retrain_plan(signature), recorded_ancestry=True)

    manifest = av.read_manifest(workflow, version_id)
    manifest[av.CONTEXT_CONTRIBUTIONS_KEY].pop("TodoList")
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "which version" in problem and "does not describe" in problem
        for problem in consistency.problems
    ), consistency.problems


def test_a_context_recorded_as_its_own_ancestor_is_refused(workflow: str):
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    record = manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]
    record["ancestors"] = sorted({*record["ancestors"], GRANDCHILD_CONTEXT})
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any("its own ancestor" in problem for problem in consistency.problems), (
        consistency.problems
    )


def test_a_digest_that_disagrees_with_the_versions_fingerprints_is_refused(
    workflow: str,
):
    """"Recorded fingerprints disagree with the fingerprints actually present."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT][
        "ancestor_utterances_sha256"
    ] = "0" * 64
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "disagrees with the command fingerprints present" in problem
        or "disagrees with the not-here samples" in problem
        for problem in consistency.problems
    ), consistency.problems


def test_a_command_with_no_recorded_fingerprint_is_refused(workflow: str):
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    orphan = manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]["label_space"][0]
    manifest[av.COMMAND_UTTERANCE_FINGERPRINTS_KEY].pop(orphan)
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "records no training-input fingerprint" in problem
        for problem in consistency.problems
    ), consistency.problems


# ---------------------------------------------------------------------
# Carried-forward contexts, checked against the version they came from
# ---------------------------------------------------------------------


def test_a_carried_forward_context_whose_ancestor_command_changed_is_refused(
    workflow: str,
):
    """The money case, and the one only a recorded sample can catch.

    TodoItem is carried forward while the sample it recorded for a TodoListManager command it never
    declares and never inherits has moved: the seed text was edited away, and the head would have
    had to choose again (design 2.50). The recorded NAMES are identical either side of that edit, so
    a name-based comparison sees nothing. The expectation here comes from the other version's
    manifest, not from asking the planner again.
    """
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    recorded_text = baseline_signature.contexts[GRANDCHILD_CONTEXT].not_here_samples[
        f"{GRANDPARENT_CONTEXT}/create_todo_list"][0]
    _replace_utterance(
        os.path.join(workflow, GRANDPARENT_COMMAND_FILE),
        recorded_text,
        "kick off a new list for the launch",
    )
    _rebuild(workflow)
    current_signature = _signature(workflow, recorded=_recorded(baseline_signature))

    assert (
        current_signature.contexts[GRANDCHILD_CONTEXT].wildcard_sources
        == baseline_signature.contexts[GRANDCHILD_CONTEXT].wildcard_sources
    ), "precondition: the names a name-based check compares did not move"

    plan = _carry_forward_plan(current_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, current_signature, plan)

    with pytest.raises(av.ArtifactConsistencyError) as excinfo:
        av.publish_version(workflow, version_id)
    message = str(excinfo.value)
    assert f"{GRANDPARENT_CONTEXT}/create_todo_list" in message
    assert "not-here samples" in message
    assert av.resolve_current_version(workflow) == baseline, (
        "the previously current version must still be current after a refusal"
    )


def test_an_honest_carry_past_an_unrecorded_grandparent_edit_still_publishes(workflow: str):
    """Design 2.50: the same edit, with the recorded sample still a seed, is an honest carry."""
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    _append_utterance(
        os.path.join(workflow, GRANDPARENT_COMMAND_FILE),
        "kick off a new list for the launch",
    )
    _rebuild(workflow)
    current_signature = _signature(workflow, recorded=_recorded(baseline_signature))

    plan = _carry_forward_plan(current_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, current_signature, plan)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_a_carried_forward_context_whose_own_command_changed_is_refused(workflow: str):
    """The same rule on the closure's first axis, so a hole there cannot ship either."""
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    _append_utterance(
        os.path.join(workflow, GRANDPARENT_COMMAND_FILE),
        "open up another list for the review",
    )
    _rebuild(workflow)
    current_signature = _signature(workflow)

    plan = _carry_forward_plan(current_signature, [GRANDPARENT_CONTEXT], baseline)
    version_id = _stage_version(workflow, current_signature, plan)

    with pytest.raises(av.ArtifactConsistencyError) as excinfo:
        av.publish_version(workflow, version_id)
    assert "own label space" in str(excinfo.value)


def test_an_honestly_carried_forward_context_still_publishes(workflow: str):
    """The guard must not refuse a correct selective run, or it refuses every run."""
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(baseline_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, baseline_signature, plan)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert consistency.unverifiable_reasons == []

    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_a_plan_from_the_real_planner_produces_a_verifiable_version(workflow: str):
    """Close the loop: the plan object a training run actually hands the writer.

    Every other carry-forward test here builds its plan by hand, which would keep passing
    if `compute_training_plan` and the writer disagreed about what a carried context is.
    """
    signature = _signature(workflow)
    baseline = _stage_version(workflow, signature, _full_retrain_plan(signature))
    av.publish_version(workflow, baseline)

    plan, current_signature = st.compute_training_plan(
        workflow,
        st.contexts_for_training(workflow),
        changed_only=True,
        seed=42,
        carry_forward_from=baseline,
    )
    assert plan.contexts_carried_forward, "precondition: the planner carried something"

    version_id = _stage_version(workflow, current_signature, plan)
    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert consistency.unverifiable_reasons == []
    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_a_carried_forward_context_with_no_artifacts_is_refused(workflow: str):
    """Publishing removes the compatibility entry of a context the version lacks.

    A version that claims to have carried a context forward and has no bytes for it would
    therefore un-train that part of the workflow while reporting success.
    """
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(baseline_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(
        workflow,
        baseline_signature,
        plan,
        contexts=[c for c in sorted(baseline_signature.contexts)
                  if c != GRANDCHILD_CONTEXT],
    )

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "no model artifacts there" in problem for problem in consistency.problems
    ), consistency.problems


def test_a_carried_forward_context_with_an_unresolved_fingerprint_is_refused(
    workflow: str,
):
    """An input that could not be fingerprinted cannot prove a model is still current."""
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(baseline_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, baseline_signature, plan)

    manifest = av.read_manifest(workflow, version_id)
    record = manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]
    unresolvable = record["wildcard_sources"][0]
    manifest[av.COMMAND_UTTERANCE_FINGERPRINTS_KEY][unresolvable] = (
        av.UNRESOLVED_FINGERPRINT
    )
    for context_record in manifest[av.CONTEXT_CONTRIBUTIONS_KEY].values():
        for field, key in (
            ("own_utterances_sha256", "label_space"),
            ("ancestor_utterances_sha256", "wildcard_sources"),
        ):
            context_record[field] = av.utterance_set_fingerprint(
                manifest[av.COMMAND_UTTERANCE_FINGERPRINTS_KEY], context_record[key]
            )
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "could not be fingerprinted" in problem for problem in consistency.problems
    ), consistency.problems


def test_carrying_forward_without_naming_a_source_version_is_refused(workflow: str):
    signature = _signature(workflow)
    plan = st.TrainingPlan(
        contexts_to_train=sorted(set(signature.contexts) - {GRANDCHILD_CONTEXT}),
        contexts_carried_forward=[GRANDCHILD_CONTEXT],
    )
    version_id = _stage_version(workflow, signature, plan)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "names no source version" in problem for problem in consistency.problems
    ), consistency.problems


def test_a_damaged_manifest_refuses_promotion(workflow: str):
    """Consistent with retention (bd fix-k0i.33): damage is not "nothing recorded"."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))
    (av.version_dir(workflow, version_id) / av.MANIFEST_FILENAME).write_text(
        "{not json at all", encoding="utf-8")

    with pytest.raises(av.ArtifactConsistencyError) as excinfo:
        av.publish_version(workflow, version_id)
    assert "damaged manifest" in str(excinfo.value)
    assert av.resolve_current_version(workflow) is None


def test_a_damaged_carry_forward_source_manifest_refuses_promotion(workflow: str):
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(baseline_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, baseline_signature, plan)
    (av.version_dir(workflow, baseline) / av.MANIFEST_FILENAME).write_text(
        "}{", encoding="utf-8")

    with pytest.raises(av.ArtifactConsistencyError) as excinfo:
        av.publish_version(workflow, version_id)
    assert "cannot be read as an object" in str(excinfo.value)
    assert av.resolve_current_version(workflow) == baseline


# ---------------------------------------------------------------------
# Independence, and the compatibility policy
# ---------------------------------------------------------------------


def test_the_check_runs_with_the_workflow_sources_deleted(workflow: str):
    """Independence, demonstrated rather than asserted.

    If the postcondition consulted `build_context_maps` -- or anything else that reads
    `_commands` and `context_hierarchy_model.json` -- it could not run at all once those
    are gone. It runs, and reaches the same verdict, because every input it compares is
    already inside the version.
    """
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))
    assert av.verify_version_consistency(workflow, version_id).verified is True

    shutil.rmtree(os.path.join(workflow, "_commands"))
    os.unlink(os.path.join(workflow, "context_hierarchy_model.json"))
    RoutingRegistry.clear_registry()

    assert av.verify_version_consistency(workflow, version_id).verified is True
    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_the_check_catches_the_bug_with_the_workflow_sources_deleted(workflow: str):
    """The same, for a refusal: the evidence is in the version, not in the workflow."""
    signature = _signature(workflow)
    version_id = _stage_version(
        workflow, signature, _full_retrain_plan(signature), recorded_ancestry=True)
    _truncate_recorded_ancestry(workflow, version_id)

    shutil.rmtree(os.path.join(workflow, "_commands"))
    RoutingRegistry.clear_registry()

    with pytest.raises(av.ArtifactConsistencyError):
        av.publish_version(workflow, version_id)


def test_a_version_written_before_this_check_publishes_and_says_it_was_unchecked(
    workflow: str, fastworkflow_logs
):
    """The compatibility policy: unverifiable, said out loud, not silently passed.

    Refusing would strand every workflow trained by an earlier build -- including the
    repair path, which republishes whatever is already current on an up-to-date run.
    """
    signature = _signature(workflow)
    version_id = av.new_version_id()
    for context_name in sorted(signature.contexts):
        _write_context_artifacts(
            av.version_dir(workflow, version_id)
            / av.context_folder_name(context_name)
        )
    av.write_manifest(workflow, version_id, seed=42)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert consistency.verified is False, (
        "'nothing was recorded' must not be reported as 'checked and consistent'"
    )
    assert any(
        "predates the promotion-time consistency check" in reason
        for reason in consistency.unverifiable_reasons
    )

    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id
    assert "without the promotion-time closure check" in fastworkflow_logs.text


def test_a_carry_forward_source_written_before_this_check_is_unverifiable(
    workflow: str, fastworkflow_logs
):
    """Half-upgraded state: the new version records contributions, the old one does not.

    The laws and the recomputation still apply to the new version; only the cross-version
    comparison is impossible, and only that is skipped.
    """
    signature = _signature(workflow)
    baseline = av.new_version_id()
    for context_name in sorted(signature.contexts):
        _write_context_artifacts(
            av.version_dir(workflow, baseline) / av.context_folder_name(context_name)
        )
    av.write_manifest(workflow, baseline, seed=42)
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, signature, plan)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert any(
        "records no usable contribution block" in reason
        for reason in consistency.unverifiable_reasons
    ), consistency.unverifiable_reasons

    av.publish_version(workflow, version_id)
    assert "without the promotion-time closure check" in fastworkflow_logs.text


def test_a_pruned_carry_forward_source_is_unverifiable_not_refused(workflow: str):
    """Rollback republishes an old version by id; its source is long gone.

    Refusing there would make rolling back impossible after two more training runs, which
    is the one thing versioning exists to guarantee.
    """
    baseline_signature = _signature(workflow)
    baseline = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, baseline)

    plan = _carry_forward_plan(baseline_signature, [GRANDCHILD_CONTEXT], baseline)
    version_id = _stage_version(workflow, baseline_signature, plan)
    av.publish_version(workflow, version_id)

    assert av.prune_versions(workflow, version_ids=[baseline], dry_run=False) == [
        baseline
    ]
    # Undo the publish so the next one is a real advance rather than a repair.
    later = _stage_version(
        workflow, baseline_signature, _full_retrain_plan(baseline_signature))
    av.publish_version(workflow, later)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.problems == []
    assert any(
        "no longer on disk" in reason for reason in consistency.unverifiable_reasons
    ), consistency.unverifiable_reasons
    av.publish_version(workflow, version_id)
    assert av.resolve_current_version(workflow) == version_id


def test_an_unrecognised_contribution_format_is_unverifiable_not_consistent(
    workflow: str,
):
    """Forward compatibility must not read a newer manifest under this build's rules."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))
    av.write_manifest(
        workflow,
        version_id,
        **{av.CONTRIBUTION_FORMAT_KEY: av.CONTRIBUTION_FORMAT_VERSION + 1},
    )

    consistency = av.verify_version_consistency(workflow, version_id)
    assert consistency.verified is False
    assert consistency.problems == []
    assert any(
        "this build does not understand" in reason
        for reason in consistency.unverifiable_reasons
    )


def test_a_malformed_contribution_block_is_refused_not_ignored(workflow: str):
    """A block that claims to be checkable and is not must not degrade to a pass."""
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))

    manifest = av.read_manifest(workflow, version_id)
    manifest[av.CONTEXT_CONTRIBUTIONS_KEY][GRANDCHILD_CONTEXT]["ancestors"] = "TodoList"
    _rewrite_manifest(workflow, version_id, manifest)

    consistency = av.verify_version_consistency(workflow, version_id)
    assert any(
        "is not a list of command/context names" in problem
        for problem in consistency.problems
    ), consistency.problems


def test_the_trainer_records_what_the_gate_checks_before_it_publishes(env_vars):
    """A postcondition nothing writes the inputs for is a postcondition that never runs.

    Every version a real run produces must carry the contribution block, or the gate
    degrades to the compatibility path for good and the finding is only half fixed.
    Asserted structurally against the shipped `train_workflow`, in the manner of
    `test_training_orchestration.py::test_training_report_gate_runs_before_publication`.
    """
    source = inspect.getsource(train_orchestration.train_workflow)
    tree = ast.parse(inspect.cleandoc(source))

    stamped = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_manifest"
        and any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Attribute)
            and keyword.value.func.attr == "context_contribution_manifest_fields"
            for keyword in node.keywords
        )
    ]
    assert len(stamped) == 1, (
        "exactly one write_manifest call must merge the contribution fields in"
    )
    assert source.index("context_contribution_manifest_fields(") < source.index(
        "artifact_versioning.publish_version(workflow_path, version_id)"
    ), "the fields must be on disk before the gate that reads them runs"


# ---------------------------------------------------------------------
# The repair path must stay runnable
# ---------------------------------------------------------------------


def test_republishing_the_current_version_repairs_rather_than_promotes(workflow: str):
    """`_repair_noop_publication` republishes whatever is current, manifest or not.

    Gating a repair on the manifest would turn a lost manifest into an unroutable
    workflow: the pointer already names this version, so the artifacts are live either
    way and refusing would only remove the reader paths to them.
    """
    signature = _signature(workflow)
    version_id = _stage_version(workflow, signature, _full_retrain_plan(signature))
    av.publish_version(workflow, version_id)

    (av.version_dir(workflow, version_id) / av.MANIFEST_FILENAME).write_text(
        "{corrupt", encoding="utf-8")
    os.unlink(av.command_info_root(workflow) / GRANDCHILD_CONTEXT)

    av.publish_version(workflow, version_id)

    assert av.resolve_current_version(workflow) == version_id
    assert (
        av.command_info_root(workflow) / GRANDCHILD_CONTEXT / "threshold.json"
    ).exists(), "the repair must have restored the reader path it exists to restore"
