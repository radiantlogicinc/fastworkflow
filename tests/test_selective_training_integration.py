"""Selective retraining (R5) against real workflows and real artifact versions.

``tests/test_selective_training.py`` covers the closure rule as pure data. This file
covers the part that data cannot: whether the closure, when handed a real workflow's
real context models and a real published artifact version, still pulls in everything
it must.

Two tiers, and the split is deliberate:

* the planner tests build a real ``messaging_app_4`` copy, real routing definitions
  and a real published version, and need no network -- deciding what to retrain is
  pure computation over source files, so it can be tested exhaustively and cheaply;
* one end-to-end test actually trains, twice, and is gated on a real key. It is the
  only test that can answer "did the model that was supposed to change, change, and
  did the ones that were carried forward survive byte-identical".

The workflow is ``messaging_app_4`` because it is the only bundled example with more
than one trainable context: it has ``PremiumUser(base=User)`` for the command-
inheritance axis and ``User``/``PremiumUser`` under ``ChatRoom`` for the context-
ancestry axis. ``hello_world`` and ``retail_workflow`` both train exactly one context
(``*``), so no closure is observable in them at all.
"""

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.model_pipeline_training import (
    CommandRouter,
    TrainingDataError,
    get_artifact_path,
)
from fastworkflow.nlu_labels import WILDCARD_LABEL
from fastworkflow.train import (
    artifact_versioning,
    determinism,
    heldout_evaluation,
    training_report,
    utterance_cache,
)
from fastworkflow.train import selective_training as st
from fastworkflow.train.__main__ import _repair_noop_publication
from fastworkflow.train.determinism import (
    ContextTrainingStatus,
    ProvenanceRecorder,
    UtteranceProvenance,
    get_training_seed,
)
from fastworkflow.train.generate_synthetic import (
    utterance_fingerprint,
)
from fastworkflow.train.utterance_cache import MODE_REUSE, UtteranceCache


MESSAGING_APP_PATH = os.path.join("fastworkflow", "examples", "messaging_app_4")
NEW_PRIORITY_UTTERANCE = (
    "flag this note to sara@fastworkflow.ai as urgent priority mail"
)

# NEW_PRIORITY_UTTERANCE is planted verbatim in the changed command's preseeded corpus so
# the second train is deterministic, which makes routing it a MEMORISATION check.
# HELDOUT_PRIORITY_UTTERANCE is in neither corpus nor seed list, so routing it is evidence
# that the retrained model learned the intent rather than the row. bd fix-k0i.30.
HELDOUT_PRIORITY_UTTERANCE = (
    "mark this memo to dana@fastworkflow.ai as urgent priority mail"
)
PRIORITY_COMMAND = "PremiumUser/send_priority_message"
USER_MESSAGE_COMMAND = "User/send_message"

# Contexts messaging_app_4 trains, as artifact folder names resolve them.
ALL_CONTEXTS = ("*", "ChatRoom", "PremiumUser", "User")

# The grandchild context the AR2 compose case needs and messaging_app_4 does not ship.
PREMIUM_SESSION_CONTEXT = "PremiumSession"
PREMIUM_SESSION_COMMAND = "PremiumSession/end_session"


def _datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def _resolve_env_vars() -> dict:
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join(
        "fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}
    for local in (os.path.join("env", ".env"), os.path.join("passwords", ".env")):
        if os.path.exists(local):
            env_vars.update(dotenv_values(local))
    for key in (
        "LLM_SYNDATA_GEN",
        "LITELLM_API_KEY_SYNDATA_GEN",
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
    ):
        val = os.environ.get(key)
        if val and "<" not in val:
            env_vars[key] = val
    return env_vars


def _copy_workflow(destination_root, name: str = "messaging_app_4") -> str:
    workflow_path = str(destination_root / name)
    shutil.copytree(
        MESSAGING_APP_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    return workflow_path


_PREMIUM_SESSION_APPLICATION = '''\
"""A session owned by a PremiumUser, so the workflow has a third generation."""


class PremiumSession:
    def __init__(self, user):
        self.user = user

    def end_session(self) -> str:
        return f"ended {self.user.name}'s session"
'''

_PREMIUM_SESSION_CONTEXT_CLASS = '''\
from ...application.session import PremiumSession
from ...application.user import PremiumUser


class Context:
    @classmethod
    def get_parent(cls, command_context_object: PremiumSession) -> PremiumUser:
        return command_context_object.user
'''

_PREMIUM_SESSION_COMMAND = '''\
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from ...application.session import PremiumSession


class Signature:
    plain_utterances = [
        "close out this session",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> fastworkflow.CommandOutput:
        session: PremiumSession = workflow.command_context_for_response_generation
        return fastworkflow.CommandOutput(
            command_responses=[
                fastworkflow.CommandResponse(response=session.end_session())
            ]
        )
'''


def _add_premium_session_context(workflow_path: str) -> None:
    """Give the copied workflow the grandchild the AR2 compose case needs.

    ``messaging_app_4`` ships two children of ``ChatRoom`` and no grandchild, so the
    composition AR2 identified as the real gap in the closure rule cannot be reproduced in
    it at all::

        User            owns    User/send_message
        PremiumUser     base:   [User]          -> inherits User/send_message
        PremiumSession  parent: [PremiumUser]   -> wildcard class carries it

    ``PremiumSession`` is not a descendant of ``User`` and does not inherit from it, yet
    its wildcard class is assembled from ``PremiumUser``'s utterances, which include
    ``User/send_message`` *because of* the ``base`` edge. Expanding over ``parent`` alone
    misses it and expanding over ``base`` alone misses it; only the union catches it, and
    only if the planner derives an ancestor's commands from the RESOLVED command list
    rather than from the raw base graph -- which is the anti-decision AR2 names.

    Built into a copy rather than into the bundled example, because adding a context to
    the shipped workflow would invalidate its checked-in trained artifacts. bd fix-k0i.49.
    """
    (Path(workflow_path) / "application" / "session.py").write_text(
        _PREMIUM_SESSION_APPLICATION, encoding="utf-8")

    commands_dir = Path(workflow_path) / "_commands" / PREMIUM_SESSION_CONTEXT
    commands_dir.mkdir(parents=True)
    (commands_dir / f"_{PREMIUM_SESSION_CONTEXT}.py").write_text(
        _PREMIUM_SESSION_CONTEXT_CLASS, encoding="utf-8")
    (commands_dir / "end_session.py").write_text(
        _PREMIUM_SESSION_COMMAND, encoding="utf-8")

    hierarchy_path = Path(workflow_path) / "context_hierarchy_model.json"
    hierarchy = json.loads(hierarchy_path.read_text())
    hierarchy[PREMIUM_SESSION_CONTEXT] = {"parent": ["PremiumUser"]}
    hierarchy_path.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")


def _rebuild(workflow_path: str) -> None:
    """Re-derive the routing artifacts after editing a workflow's sources.

    Every cached layer has to be dropped, not just the on-disk JSON: the registry
    memoises per path, so a plan computed after an edit would otherwise be computed
    against the pre-edit command directory and report nothing changed. That failure
    would look exactly like the bug these tests exist to catch, which is why it is
    done here once rather than ad hoc in each test.
    """
    RoutingRegistry.clear_registry()
    CommandContextModel.load(workflow_path)
    RoutingDefinition.build(workflow_path)
    RoutingRegistry.get_definition(workflow_path, load_cached=False)


def _seed_utterance_cache(workflow_path: str) -> None:
    """Leave behind the cache entries a real baseline run would have written.

    Same placeholder philosophy as the artifact files below: the carry-forward guard
    (`_shared_commands_absent_from_cache`) reads only whether an entry EXISTS for a
    command, never its contents, because it runs before this run's variant key is
    known. Without them the fixture describes a state a real run cannot produce -- a
    published baseline whose generated utterances were never cached -- and the
    planner correctly refuses to carry anything forward, which is not what the tests
    below are about.
    """
    cache_root = os.path.join(
        workflow_path, determinism.COMMAND_INFO_FOLDERNAME,
        utterance_cache.CACHE_DIRNAME,
    )
    os.makedirs(cache_root, exist_ok=True)
    context_commands, _ancestors, _ = st.build_context_maps(
        workflow_path, st.contexts_for_training(workflow_path))
    for commands in context_commands.values():
        for command_name in commands:
            path = os.path.join(
                cache_root, f"{utterance_cache.slugify(command_name)}.baseline.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"entries": {}}, f)


def _publish_baseline(workflow_path: str, seed: int = 42) -> tuple[str, st.TrainingSignature]:
    """Publish a version holding a complete-looking artifact set for every context.

    The artifact files are empty placeholders: nothing in the planner reads their
    contents, only whether the full set is present, and training them for real would
    put an LLM call in the middle of a pure-computation test. What is real here is
    everything the planner does read -- the version layout, the published pointer,
    and the recorded signature.
    """
    _seed_utterance_cache(workflow_path)
    contexts = st.contexts_for_training(workflow_path)
    version_id = artifact_versioning.new_version_id()
    for context_name in contexts:
        folder = artifact_versioning.version_dir(
            workflow_path, version_id
        ) / artifact_versioning.context_folder_name(context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()

    artifact_versioning.write_manifest(workflow_path, version_id, seed=seed)
    signature, _unresolved = st.compute_training_signature(
        workflow_path, contexts, seed=seed)
    st.save_training_signature(workflow_path, version_id, signature)
    artifact_versioning.publish_version(workflow_path, version_id)
    return version_id, signature


def _plan(workflow_path: str, version_id: str, seed: int = 42) -> st.TrainingPlan:
    plan, _signature = st.compute_training_plan(
        workflow_path,
        st.contexts_for_training(workflow_path),
        changed_only=True,
        seed=seed,
        carry_forward_from=version_id,
    )
    return plan


def _append_utterance(path: str, utterance: str) -> None:
    source = open(path).read()
    marker = "plain_utterances = ["
    assert marker in source, f"{path} has no plain_utterances list to edit"
    index = source.index(marker) + len(marker)
    open(path, "w").write(
        f'{source[:index]}\n        "{utterance}",{source[index:]}')


def _fixed_generated_corpus(command_name: str) -> list[str]:
    """Return a deterministic local corpus with a distinct vocabulary per intent."""
    if command_name == USER_MESSAGE_COMMAND:
        return [
            f"{prefix} send a regular message to teammate{i}@fastworkflow.ai"
            for i, prefix in enumerate((
                "please", "can you", "I need to", "help me", "go ahead and",
                "would you", "kindly", "let me", "I want to", "could you",
                "please now", "when ready", "for me", "quickly", "today",
                "at once", "right away", "in this chat", "privately", "directly",
            ))
        ]
    if command_name == PRIORITY_COMMAND:
        return [
            f"{prefix} send an urgent priority message to teammate{i}@fastworkflow.ai"
            for i, prefix in enumerate((
                "please", "can you", "I need to", "help me", "go ahead and",
                "would you", "kindly", "let me", "I want to", "could you",
                "please now", "when ready", "for me", "quickly", "today",
                "at once", "right away", "in this chat", "privately", "directly",
            ))
        ]

    intent = command_name.split("/")[-1].replace("_", " ")
    return [
        f"{prefix} {intent}{suffix}"
        for prefix, suffix in (
            ("please", ""),
            ("can you", " for me"),
            ("I need to", ""),
            ("help me", ""),
            ("go ahead and", ""),
            ("would you", " now"),
            ("kindly", ""),
            ("let me", ""),
            ("I want to", ""),
            ("could you", " please"),
            ("please now", ""),
            ("when ready", ""),
            ("for me", ""),
            ("quickly", ""),
            ("today", ""),
            ("at once", ""),
            ("right away", ""),
            ("in this chat", ""),
            ("show me how to", ""),
            ("perform", ""),
        )
    ]


def _preseed_fixed_utterance_cache(
    workflow_path: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Seed both sides of the edit so neither train draws a fresh LLM corpus.

    The changed command intentionally gets a second cache variant. R6 fingerprints
    the ordered seed list, so the edit below must miss the baseline variant; seeding
    the post-edit variant in advance preserves that invalidation while making the
    replacement corpus deterministic.
    """
    cmd_dir = CommandDirectory.load(workflow_path)
    cache = UtteranceCache(workflow_path, mode=MODE_REUSE)
    seed = get_training_seed()
    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    num_personas = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS", int)
    utterances_per_persona = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA", int)
    personas_per_batch = fastworkflow.get_env_var(
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH", int)
    baseline_corpora = {}

    commands = set(cmd_dir.get_utterance_keys()) | set(cmd_dir.core_command_names)
    for command_name in sorted(commands):
        if command_name.split("/")[-1] == "wildcard":
            continue
        cmd_dir.ensure_command_hydrated(command_name)
        metadata = cmd_dir.get_utterance_metadata(command_name)
        corpus = _fixed_generated_corpus(command_name)
        fingerprint = utterance_fingerprint(
            metadata.plain_utterances,
            command_name,
            num_personas,
            utterances_per_persona,
            personas_per_batch,
            model,
        )
        assert cache.store(fingerprint, seed, corpus)
        baseline_corpora[command_name] = corpus

    metadata = cmd_dir.get_utterance_metadata(PRIORITY_COMMAND)
    changed_corpus = (
        _fixed_generated_corpus(PRIORITY_COMMAND)
        + [
            (
                f"flag priority note {index} to teammate{index}@fastworkflow.ai "
                "as urgent priority mail"
            )
            for index in range(24)
        ]
        + [NEW_PRIORITY_UTTERANCE]
    )
    changed_fingerprint = utterance_fingerprint(
        [NEW_PRIORITY_UTTERANCE, *metadata.plain_utterances],
        PRIORITY_COMMAND,
        num_personas,
        utterances_per_persona,
        personas_per_batch,
        model,
    )
    assert cache.store(changed_fingerprint, seed, changed_corpus)
    return baseline_corpora, changed_corpus


def _assert_fixed_corpus_was_used(
    workflow_path: str,
    expected_corpora: dict[str, list[str]],
) -> None:
    """Prove the run consumed the preseed rather than silently missing it."""
    provenance_path = os.path.join(
        workflow_path, "___command_info", "training_provenance.json")
    payload = json.loads(open(provenance_path).read())
    provenance = payload.get("commands", payload)
    for command_name, corpus in expected_corpora.items():
        record = provenance[command_name]
        assert record["generated_count"] == len(corpus), (
            f"{command_name} did not use its fixed cached corpus"
        )
        assert set(corpus) <= set(record["utterance_personas"]), (
            f"{command_name} regenerated utterances instead of using the preseed"
        )


@pytest.fixture(scope="module")
def env_vars() -> dict:
    values = _resolve_env_vars()
    fastworkflow.init(env_vars=values)
    return values


@pytest.fixture
def baseline(tmp_path, env_vars):
    """A messaging_app_4 copy with a published baseline version, ready to diff."""
    workflow_path = _copy_workflow(tmp_path)
    _rebuild(workflow_path)
    version_id, _signature = _publish_baseline(workflow_path)
    yield workflow_path, version_id
    RoutingRegistry.clear_registry()


@pytest.fixture
def baseline_with_grandchild(tmp_path, env_vars):
    """The same, plus the `PremiumSession` grandchild the AR2 compose case needs.

    A separate fixture rather than a change to `baseline`, because every other test in
    this file asserts on the exact set of four contexts messaging_app_4 ships and adding a
    fifth to all of them would say nothing about the closure. bd fix-k0i.49.
    """
    workflow_path = _copy_workflow(tmp_path)
    _add_premium_session_context(workflow_path)
    _rebuild(workflow_path)
    version_id, _signature = _publish_baseline(workflow_path)
    yield workflow_path, version_id
    RoutingRegistry.clear_registry()


# ---------------------------------------------------------------------
# The closure, against a real workflow
# ---------------------------------------------------------------------

def test_an_unchanged_workflow_carries_every_context_forward(baseline):
    """The premise of the feature: no edit, no retraining.

    If this fails everything else is moot -- a planner that cannot recognise "nothing
    changed" saves nothing, and one that reports changes at random would train the
    right things for the wrong reason and hide a real detection bug.
    """
    workflow_path, version_id = baseline
    plan = _plan(workflow_path, version_id)

    assert plan.contexts_to_train == []
    assert set(plan.contexts_carried_forward) == {
        "*", "ChatRoom", "PremiumUser", "User"}
    assert plan.dirty_commands == []


def test_a_leaf_command_edit_retrains_only_its_own_context(baseline):
    """``PremiumUser`` is a leaf: nothing inherits from it, nothing descends from it."""
    workflow_path, version_id = baseline
    _append_utterance(
        os.path.join(
            workflow_path, "_commands/PremiumUser/send_priority_message.py"),
        "flag this note to sara@fastworkflow.ai as urgent priority mail",
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert plan.contexts_to_train == ["PremiumUser"]
    assert set(plan.contexts_carried_forward) == {"*", "ChatRoom", "User"}
    assert any(
        "send_priority_message" in reason
        for reason in plan.reasons["PremiumUser"]
    )


def test_base_axis_a_changed_base_command_retrains_every_deriving_context(baseline):
    """``PremiumUser`` declares ``base: [User]``, so ``User``'s commands are its own.

    The command file edited here lives under ``_commands/User/``. Nothing under
    ``_commands/PremiumUser/`` was touched, and a per-directory notion of "what
    changed" would therefore leave PremiumUser on a model that has never seen the
    edited command's new phrasing while still being expected to route it.
    """
    workflow_path, version_id = baseline
    _append_utterance(
        os.path.join(workflow_path, "_commands/User/send_message.py"),
        "shoot a quick note over to dana@fastworkflow.ai",
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert "User" in plan.contexts_to_train
    assert "PremiumUser" in plan.contexts_to_train, (
        "PremiumUser inherits User's commands through `base`; leaving it on the old "
        "model is silent staleness"
    )


def test_parent_axis_an_ancestor_command_change_retrains_its_descendants(baseline):
    """``User`` and ``PremiumUser`` sit under ``ChatRoom`` in the context hierarchy.

    A ChatRoom command is not in either descendant's label space. It is in their
    *wildcard* class, which is assembled from ancestor utterances -- so the edit
    changes what those models must recognise as "belongs upstairs" without appearing
    anywhere in their own label spaces. This is the axis that pure per-command
    reasoning gets wrong.
    """
    workflow_path, version_id = baseline
    _append_utterance(
        os.path.join(workflow_path, "_commands/ChatRoom/add_user.py"),
        "please enroll priya@fastworkflow.ai into this room",
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert "ChatRoom" in plan.contexts_to_train
    for descendant in ("User", "PremiumUser"):
        assert descendant in plan.contexts_to_train, (
            f"{descendant} descends from ChatRoom; its wildcard class is now stale"
        )
        assert any(
            "wildcard" in reason for reason in plan.reasons[descendant]
        ), f"{descendant} was retrained but not for the wildcard reason"
    assert plan.contexts_carried_forward == ["*"]


def test_a_context_inheritance_model_edit_alone_retrains_the_deriving_context(baseline):
    """No command file changes here at all -- only ``context_inheritance_model.json``.

    Removing ``PremiumUser``'s ``base`` entry deletes ``User``'s commands from
    PremiumUser's label space. Every command file on disk is byte-identical, so the
    fingerprint diff sees nothing; only the recorded label space does.
    """
    workflow_path, version_id = baseline
    model_path = os.path.join(
        workflow_path, "_commands", "context_inheritance_model.json")
    open(model_path, "w").write(json.dumps({}))
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert plan.dirty_commands == [], "no command file was edited"
    assert "PremiumUser" in plan.contexts_to_train
    assert any(
        "label space changed" in reason
        for reason in plan.reasons["PremiumUser"]
    )


def test_a_new_command_retrains_its_context_and_every_descendant(baseline):
    """Adding a file is a label-space change, not just a fingerprint change."""
    workflow_path, version_id = baseline
    shutil.copy(
        os.path.join(workflow_path, "_commands/ChatRoom/list_users.py"),
        os.path.join(workflow_path, "_commands/ChatRoom/list_recent_users.py"),
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert any("list_recent_users" in name for name in plan.dirty_commands)
    for context_name in ("ChatRoom", "User", "PremiumUser"):
        assert context_name in plan.contexts_to_train


# ---------------------------------------------------------------------
# AR2: base and parent compose, against a real workflow (bd fix-k0i.49)
#
# `tests/test_selective_training.py::test_base_and_parent_axes_compose` exercises the
# shipped `close_dirty_contexts` but with hand-built context maps, so it says nothing about
# how the PLANNER derives those maps. The derivation is where the composition can be lost:
# taking an ancestor's contribution from the raw base graph rather than from the resolved
# `commands()` leaves the grandchild clean while every existing test still passes. No
# bundled example had a grandchild to notice it with.
# ---------------------------------------------------------------------


def test_the_grandchild_fixture_really_has_both_axes(baseline_with_grandchild):
    """The compose case is only observable if the fixture actually composes.

    Checked separately from the closure below so that a fixture that quietly stopped
    declaring the `base` edge, or a `PremiumSession` that never became a context, fails as
    a fixture problem rather than as a spurious "the closure is fine" pass.
    """
    workflow_path, _version_id = baseline_with_grandchild
    crd = RoutingRegistry.get_definition(workflow_path)

    assert PREMIUM_SESSION_CONTEXT in st.contexts_for_training(workflow_path)

    # base axis: PremiumUser's resolved commands include the command User owns.
    assert USER_MESSAGE_COMMAND in crd.context_model.commands("PremiumUser")
    # ...and PremiumSession does NOT inherit it -- it has no `base` entry at all.
    assert USER_MESSAGE_COMMAND not in crd.context_model.commands(
        PREMIUM_SESSION_CONTEXT)
    # parent axis: PremiumUser is PremiumSession's ancestor, and User is not.
    ancestors = crd.context_model.get_ancestor_contexts(PREMIUM_SESSION_CONTEXT)
    assert "PremiumUser" in ancestors
    assert "ChatRoom" in ancestors
    assert "User" not in ancestors


def test_a_base_inherited_command_edit_dirties_the_grandchild(baseline_with_grandchild):
    """The composition, end to end through `compute_training_plan`.

    `User/send_message` is edited. `PremiumSession` neither owns it nor inherits it nor
    descends from `User`, so neither axis alone reaches it -- it is dirty only because its
    wildcard class is built from `PremiumUser`'s utterances and `PremiumUser` carries the
    command through `base`. Leaving it on its old model desyncs what that model treats as
    "belongs upstairs", with no error and no visible symptom.
    """
    workflow_path, version_id = baseline_with_grandchild
    _append_utterance(
        os.path.join(workflow_path, "_commands/User/send_message.py"),
        "shoot a quick note over to dana@fastworkflow.ai",
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert plan.dirty_commands == [USER_MESSAGE_COMMAND]
    assert PREMIUM_SESSION_CONTEXT in plan.contexts_to_train, (
        "the grandchild's wildcard class is assembled from PremiumUser's utterances, "
        "which include User/send_message through `base`; carrying it forward is silent "
        "staleness"
    )
    reasons = plan.reasons[PREMIUM_SESSION_CONTEXT]
    assert any("wildcard class is stale" in reason for reason in reasons), reasons
    assert any("PremiumUser" in reason for reason in reasons), reasons
    # The reason must name the ANCESTOR that carries it, not User: a developer reading
    # this has to be able to follow the two hops.
    assert all(
        "label space contains" not in reason for reason in reasons
    ), f"the grandchild does not own the command; got {reasons}"
    # And the closure is still a closure. Without this, "dirty everything" would satisfy
    # the assertion above and the compose claim would be indistinguishable from a planner
    # that gave up: `ChatRoom` owns none of the edited command and has no ancestors, and
    # `*` carries only core commands.
    assert set(plan.contexts_carried_forward) == {"*", "ChatRoom"}


def test_an_unchanged_grandchild_workflow_carries_every_context_forward(
    baseline_with_grandchild,
):
    """The grandchild must be carryable too, or the fixture proves nothing about staleness.

    A planner that always dirtied `PremiumSession` would pass the compose test for the
    wrong reason.
    """
    workflow_path, version_id = baseline_with_grandchild
    plan = _plan(workflow_path, version_id)

    assert plan.contexts_to_train == []
    assert set(plan.contexts_carried_forward) == {
        "*", "ChatRoom", "PremiumUser", "User", PREMIUM_SESSION_CONTEXT}


def test_a_leaf_edit_in_the_grandchild_stays_in_the_grandchild(
    baseline_with_grandchild,
):
    """`PremiumSession` is a leaf: nothing inherits from it, nothing descends from it."""
    workflow_path, version_id = baseline_with_grandchild
    _append_utterance(
        os.path.join(
            workflow_path, "_commands", PREMIUM_SESSION_CONTEXT, "end_session.py"),
        "wrap up and log me out of this session",
    )
    _rebuild(workflow_path)

    plan = _plan(workflow_path, version_id)

    assert plan.dirty_commands == [PREMIUM_SESSION_COMMAND]
    assert plan.contexts_to_train == [PREMIUM_SESSION_CONTEXT]
    assert set(plan.contexts_carried_forward) == {
        "*", "ChatRoom", "PremiumUser", "User"}


# ---------------------------------------------------------------------
# Safety: everything ambiguous must retrain
# ---------------------------------------------------------------------

def test_an_incomplete_artifact_set_forces_a_retrain(baseline):
    """A half-written context directory must never be carried forward.

    An interrupted run leaves exactly this state. Carrying it forward would publish a
    version that looks trained and raises FileNotFoundError on the first turn, and
    every later selective run would inherit it again.
    """
    workflow_path, version_id = baseline
    (artifact_versioning.version_dir(workflow_path, version_id)
     / "ChatRoom" / "threshold.json").unlink()

    plan = _plan(workflow_path, version_id)

    assert plan.contexts_to_train == ["ChatRoom"]
    assert any(
        "no complete model artifact set" in reason
        for reason in plan.reasons["ChatRoom"]
    )


def test_an_unreadable_command_source_is_unresolved_and_forces_a_retrain(baseline):
    """"I could not check this command" must never be reported as "it did not change".

    The `(None, None) == (None, None)` failure: a command whose source cannot be read has
    no hashes, and two consecutive runs that both failed to read it produced two identical
    empty fingerprints -- so it looked unchanged forever and every context carrying it was
    carried forward on a stale model. Nothing in the suite exercised the fail-closed
    branch, so reverting `training_inputs_differ` to a plain field comparison reintroduced
    exactly that and passed everything.

    The file is removed WITHOUT rebuilding, deliberately: a rebuild would drop the command
    from the directory entirely, which is a removal (already covered by
    `changed_commands`). What is under test is the command the directory still knows about
    whose bytes have become unreadable. bd fix-k0i.28.
    """
    workflow_path, version_id = baseline
    os.remove(os.path.join(workflow_path, "_commands/User/send_message.py"))

    fingerprints = st.compute_command_fingerprints(workflow_path)
    unreadable = fingerprints[USER_MESSAGE_COMMAND]
    assert unreadable.resolved is False
    assert unreadable.source_sha256 is None
    assert "could not be read" in unreadable.unresolved_reason
    # It must differ from the baseline's readable fingerprint AND from another unresolved
    # one -- the second is the comparison that used to fail equal.
    baseline_signature = st.load_training_signature(workflow_path, version_id)
    assert baseline_signature.command_fingerprints[USER_MESSAGE_COMMAND].resolved is True
    assert unreadable.training_inputs_differ(
        baseline_signature.command_fingerprints[USER_MESSAGE_COMMAND])
    assert unreadable.training_inputs_differ(unreadable)

    plan = _plan(workflow_path, version_id)

    assert USER_MESSAGE_COMMAND in plan.dirty_commands
    assert "User" in plan.contexts_to_train
    assert "PremiumUser" in plan.contexts_to_train, (
        "PremiumUser inherits the unreadable command through `base`, so it cannot be "
        "proven clean either"
    )


def test_a_command_unreadable_on_both_runs_is_still_dirty_on_the_second(baseline):
    """The regression's actual shape: TWO runs that both failed to read the same file.

    One unresolved side against one resolved side would be caught by a plain field
    comparison too, because the hashes differ. What a plain comparison cannot catch is two
    unresolved sides, which is what a persistent permissions or packaging problem produces
    -- and which is the state a workflow sits in for as long as the problem lasts.
    """
    workflow_path, version_id = baseline
    source = os.path.join(workflow_path, "_commands/User/send_message.py")
    os.remove(source)

    # Record an unresolved baseline, exactly as the first failing run would have.
    first_run = st.compute_command_fingerprints(workflow_path)
    unresolved_baseline = st.TrainingSignature(
        **{
            **st.load_training_signature(workflow_path, version_id).model_dump(),
            "command_fingerprints": {
                name: fingerprint.model_dump()
                for name, fingerprint in first_run.items()
            },
        }
    )
    st.save_training_signature(workflow_path, version_id, unresolved_baseline)

    plan = _plan(workflow_path, version_id)

    assert USER_MESSAGE_COMMAND in plan.dirty_commands, (
        "the command was unreadable on both runs, so the two fingerprints are identical "
        "and a field comparison calls it unchanged -- which is an inability to check "
        "being reported as a passing check"
    )
    for context_name in ("User", "PremiumUser"):
        assert context_name in plan.contexts_to_train


def test_a_changed_seed_forces_a_full_retrain(baseline):
    """The seed is a global input: it decides every context's data, so nothing carries."""
    workflow_path, version_id = baseline

    plan = _plan(workflow_path, version_id, seed=1234)

    assert plan.is_full_retrain
    assert plan.contexts_carried_forward == []
    assert any("seed changed" in reason for reason in plan.global_reasons)


def test_an_unreadable_baseline_signature_forces_a_full_retrain(baseline):
    """A corrupt baseline is an inability to check, which is not a passing check."""
    workflow_path, version_id = baseline
    st.signature_path(workflow_path, version_id).write_text("{not json")

    plan = _plan(workflow_path, version_id)

    assert plan.is_full_retrain
    assert plan.contexts_carried_forward == []


def test_no_previous_version_forces_a_full_retrain(baseline):
    """First train of a workflow: there is nothing to carry forward from."""
    workflow_path, _version_id = baseline

    plan, _signature = st.compute_training_plan(
        workflow_path,
        st.contexts_for_training(workflow_path),
        changed_only=True,
        seed=42,
        carry_forward_from=None,
    )

    assert plan.is_full_retrain
    assert plan.contexts_carried_forward == []


def test_default_plan_automatically_reuses_unchanged_contexts(baseline):
    """The trainer decides automatically; users do not manage a selective flag."""
    workflow_path, version_id = baseline

    plan, _signature = st.compute_training_plan(
        workflow_path,
        st.contexts_for_training(workflow_path),
        seed=42,
        carry_forward_from=version_id,
    )

    assert not plan.is_full_retrain
    assert plan.contexts_to_train == []
    assert set(plan.contexts_carried_forward) == {
        "*", "ChatRoom", "PremiumUser", "User"}


def test_carry_forward_refuses_to_publish_an_incomplete_version(baseline):
    """The failure must be loud.

    Publishing a version that is missing a context makes `publish_version` remove
    that context's compatibility entry -- the workflow silently loses a model. So a
    carry-forward that cannot be completed raises instead of returning what it
    managed to do.
    """
    workflow_path, _version_id = baseline
    plan = st.TrainingPlan(
        contexts_carried_forward=["ChatRoom"],
        carry_forward_from=None,
    )

    with pytest.raises(st.SelectiveTrainingError):
        st.carry_forward_contexts(workflow_path, plan, "20260101T000000Z-abcdef")


def test_carry_forward_places_real_artifacts_in_the_new_version(baseline):
    """The carried context must actually exist in the version being assembled."""
    workflow_path, version_id = baseline
    new_version = artifact_versioning.new_version_id()
    artifact_versioning.write_manifest(workflow_path, new_version, seed=42)
    plan = st.TrainingPlan(
        contexts_carried_forward=["ChatRoom", "User"],
        carry_forward_from=version_id,
    )

    carried = st.carry_forward_contexts(workflow_path, plan, new_version)

    assert carried == ["ChatRoom", "User"]
    for context_name in carried:
        assert st.context_artifacts_complete(
            workflow_path, new_version, context_name)


# ---------------------------------------------------------------------
# AR3's crash invariant, through the REAL reader paths (bd fix-k0i.29)
#
# `train_workflow` carries unchanged contexts forward BEFORE it publishes, and publishes
# `current.json` LAST. Both orderings were enforced only by comments: nothing killed a run
# between the two and asserted that the old version was still what every reader saw. The
# two reader paths are checked independently because they are derived independently --
# `get_artifact_path` reads the authoritative pointer, while `intent_detection.py` builds
# `f"{workflow}/___command_info/{context}"` as a literal string and resolves through the
# per-context compatibility entries.
# ---------------------------------------------------------------------


def _mark_version_artifacts(workflow_path: str, version_id: str, marker: float) -> None:
    """Give one version's threshold files distinguishable content.

    `_publish_baseline` writes empty placeholders, which is enough for the planner's
    completeness check but cannot answer "which version am I reading?" from content.
    """
    for context_name in st.contexts_for_training(workflow_path):
        folder = artifact_versioning.version_dir(
            workflow_path, version_id
        ) / artifact_versioning.context_folder_name(context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            path = folder / artifact
            if artifact.endswith(".json"):
                path.write_text(
                    json.dumps(
                        {"confidence_threshold": marker, "version": version_id}),
                    encoding="utf-8",
                )
            elif not path.exists():
                path.touch()


def _literal_reader_path(workflow_path: str, context_name: str) -> str:
    """The path `intent_detection.py` builds by hand, as an f-string, for one context."""
    folder = artifact_versioning.context_folder_name(context_name)
    return f"{workflow_path}/___command_info/{folder}/threshold.json"


def _assemble_next_version(workflow_path: str, from_version: str) -> str:
    """Do everything a selective train does before `publish_version`, and stop there."""
    new_version = artifact_versioning.new_version_id()
    artifact_versioning.write_manifest(
        workflow_path, new_version, seed=42, previous_version=from_version)
    retrained = artifact_versioning.version_dir(
        workflow_path, new_version
    ) / artifact_versioning.context_folder_name("PremiumUser")
    retrained.mkdir(parents=True, exist_ok=True)
    for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
        path = retrained / artifact
        if artifact.endswith(".json"):
            path.write_text(
                json.dumps({"confidence_threshold": 0.99, "version": new_version}),
                encoding="utf-8",
            )
        else:
            path.touch()
    plan = st.TrainingPlan(
        contexts_to_train=["PremiumUser"],
        contexts_carried_forward=["*", "ChatRoom", "User"],
        carry_forward_from=from_version,
    )
    assert st.carry_forward_contexts(workflow_path, plan, new_version) == [
        "*", "ChatRoom", "User"]
    return new_version


def test_a_crash_between_carry_forward_and_publish_leaves_every_reader_on_the_old_version(
    baseline,
):
    """The version is fully assembled and the run dies. Nothing may have moved.

    This is the ordering `train_workflow` enforces with a comment: carry forward, then
    publish. A crash in the window between them -- an OOM in the confusion scan, a failed
    signature write, a SIGKILL -- must leave the workflow exactly as runnable as it was,
    because publishing a version that is missing a context silently un-trains part of the
    workflow and there is no signal that would surface it.
    """
    workflow_path, version_id = baseline
    _mark_version_artifacts(workflow_path, version_id, 0.11)
    new_version = _assemble_next_version(workflow_path, version_id)

    # ---- the crash: `publish_version` is never reached ----

    assert artifact_versioning.resolve_current_version(workflow_path) == version_id
    for context_name in ALL_CONTEXTS:
        derived = get_artifact_path(workflow_path, context_name, "threshold.json")
        literal = _literal_reader_path(workflow_path, context_name)
        assert f"versions{os.sep}{version_id}{os.sep}" in derived, derived
        assert new_version not in derived, derived
        # The two reader paths are derived independently; they must land on the same file.
        assert os.path.realpath(literal) == os.path.realpath(derived), (
            f"{context_name}: the literal reader path and get_artifact_path disagree"
        )
        payload = json.loads(open(literal, encoding="utf-8").read())
        assert payload["version"] == version_id, (
            f"{context_name} is being read out of the unpublished version"
        )
    # The assembled version is still on disk and protected -- it is the thing a retry
    # resumes into, and pruning it would make the crash cost a full retrain.
    assert artifact_versioning.version_dir(workflow_path, new_version).is_dir()


def test_a_crash_midway_through_the_compat_swaps_keeps_the_old_version_authoritative(
    baseline, monkeypatch
):
    """A fault injected BETWEEN compatibility swaps, not before the first one.

    `publish_version` swaps the per-context entries one at a time and writes
    `current.json` last, so a crash inside the loop leaves reader paths mixed. The
    invariant that has to hold is that the AUTHORITATIVE pointer never moved: the
    trainer's own path derivation (`get_artifact_path`) still names the old version, so a
    rerun reads and writes against the version that is actually complete, and no literal
    reader path is left dangling on a `threshold.json` that does not exist -- this repo's
    best-known failure signature.
    """
    workflow_path, version_id = baseline
    _mark_version_artifacts(workflow_path, version_id, 0.11)
    new_version = _assemble_next_version(workflow_path, version_id)

    real_point_compat_entry = artifact_versioning._point_compat_entry
    swapped: list[str] = []

    def fail_after_the_first_swap(dest, target):
        if swapped:
            raise OSError("injected crash midway through the compatibility swaps")
        swapped.append(dest.name)
        real_point_compat_entry(dest, target)

    monkeypatch.setattr(
        artifact_versioning, "_point_compat_entry", fail_after_the_first_swap)
    with pytest.raises(OSError, match="injected crash"):
        artifact_versioning.publish_version(workflow_path, new_version)
    monkeypatch.undo()

    assert swapped, "the injected failure fired before any swap happened"
    assert artifact_versioning.resolve_current_version(workflow_path) == version_id
    assert json.loads(
        artifact_versioning.pointer_path(workflow_path).read_text()
    )["version_id"] == version_id

    for context_name in ALL_CONTEXTS:
        derived = get_artifact_path(workflow_path, context_name, "threshold.json")
        assert f"versions{os.sep}{version_id}{os.sep}" in derived, derived
        # Never a dangling entry: the runtime opens the literal path directly.
        literal = _literal_reader_path(workflow_path, context_name)
        assert os.path.isfile(literal), (
            f"{context_name}'s compatibility entry is dangling after the crash; the "
            f"runtime would raise FileNotFoundError on threshold.json"
        )

    # The state IS mixed -- that is why it needs repairing rather than merely surviving.
    mixed = [
        context_name for context_name in ALL_CONTEXTS
        if new_version in os.path.realpath(
            _literal_reader_path(workflow_path, context_name))
    ]
    assert mixed, (
        "no compatibility entry was swapped, so this test is not exercising the "
        "mid-publish window at all"
    )

    # The repair is the production one: the next run finds nothing to train and calls
    # `_repair_noop_publication`, which republishes the current version.
    _repair_noop_publication(workflow_path, version_id)

    assert artifact_versioning.resolve_current_version(workflow_path) == version_id
    for context_name in ALL_CONTEXTS:
        derived = get_artifact_path(workflow_path, context_name, "threshold.json")
        literal = _literal_reader_path(workflow_path, context_name)
        assert os.path.realpath(literal) == os.path.realpath(derived), (
            f"{context_name}: the readers still disagree after the repair"
        )
        payload = json.loads(open(literal, encoding="utf-8").read())
        assert payload["version"] == version_id
    # ...and the abandoned half-routed version is retired, so the next run does not
    # inherit a version nothing points at.
    assert not artifact_versioning.version_dir(workflow_path, new_version).is_dir()


def test_the_noop_repair_refuses_when_there_is_no_current_version(baseline):
    """Repair cannot invent a version to route to.

    Reached when the pointer is gone and no symlink survives. Publishing "whatever looks
    newest" would be the wrong recovery: a half-assembled version from an interrupted run
    is exactly what looks newest.
    """
    workflow_path, _version_id = baseline

    with pytest.raises(TrainingDataError) as excinfo:
        _repair_noop_publication(workflow_path, None)
    assert "no current artifact version" in str(excinfo.value)


def _generation_record(
    command_name: str, marker: str, generated_count: int
) -> UtteranceProvenance:
    return UtteranceProvenance(
        command_name=command_name,
        seed=42,
        persona_ids=[marker],
        utterance_personas={marker: marker},
        generated_count=generated_count,
        final_count=generated_count,
    )


def _capture_versioned_provenance(
    recorder: ProvenanceRecorder, workflow_path: str, version_id: str
) -> dict:
    top_level_path = recorder.save()
    shutil.copy2(
        top_level_path,
        artifact_versioning.version_dir(workflow_path, version_id)
        / os.path.basename(top_level_path),
    )
    captured = st.capture_training_provenance(workflow_path, version_id)
    assert captured is not None
    return captured


def test_provenance_merge_restores_only_carried_context_records(baseline):
    """Carried siblings remain reportable while fresh retrained records always win.

    ``User/send_message`` is intentionally present in both ``User`` and
    ``PremiumUser`` because the real workflow declares ``PremiumUser(base=User)``.
    This exercises the dangerous overlap: the old generation record must not replace
    the fresh one, and the old PremiumUser context row must not replace its fresh row.
    """
    workflow_path, version_id = baseline
    previous_recorder = ProvenanceRecorder(workflow_path)
    previous_recorder.record(
        _generation_record("ChatRoom/add_user", "old-chat-room", 11))
    previous_recorder.record(
        _generation_record(USER_MESSAGE_COMMAND, "old-shared", 12))
    previous_recorder.record(
        _generation_record("PremiumUser/deleted_command", "old-deleted", 13))
    previous_recorder.record_context(
        context_name="ChatRoom",
        command_name="ChatRoom/add_user",
        status=ContextTrainingStatus.INCLUDED,
        row_count=11,
    )
    previous_recorder.record_context(
        context_name="ChatRoom",
        command_name=WILDCARD_LABEL,
        status=ContextTrainingStatus.INCLUDED,
        row_count=7,
        own_row_count=11,
        raw_candidate_count=21,
        deduplicated_candidate_count=13,
        always_include_count=1,
        selected_budget=11,
        coverage_floor=4,
        coverage_floor_applied=False,
    )
    previous_recorder.record_context(
        context_name="User",
        command_name=USER_MESSAGE_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=12,
    )
    previous_recorder.record_context(
        context_name="PremiumUser",
        command_name=USER_MESSAGE_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=12,
    )
    previous_recorder.record_context(
        context_name="PremiumUser",
        command_name="PremiumUser/deleted_command",
        status=ContextTrainingStatus.INCLUDED,
        row_count=13,
    )
    previous = _capture_versioned_provenance(
        previous_recorder, workflow_path, version_id)

    fresh_recorder = ProvenanceRecorder(workflow_path)
    fresh_recorder.record(
        _generation_record(USER_MESSAGE_COMMAND, "fresh-shared", 21))
    fresh_recorder.record(
        _generation_record(PRIORITY_COMMAND, "fresh-priority", 22))
    fresh_recorder.record_context(
        context_name="PremiumUser",
        command_name=USER_MESSAGE_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=21,
    )
    fresh_recorder.record_context(
        context_name="PremiumUser",
        command_name=PRIORITY_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=22,
    )
    fresh_recorder.save()
    plan = st.TrainingPlan(
        contexts_to_train=["PremiumUser"],
        contexts_carried_forward=["ChatRoom", "User"],
        carry_forward_from=version_id,
    )

    assert st.merge_training_provenance(workflow_path, plan, previous)

    commands = ProvenanceRecorder.load(workflow_path)
    contexts = ProvenanceRecorder.load_context_records(workflow_path)
    assert set(commands) == {
        "ChatRoom/add_user",
        USER_MESSAGE_COMMAND,
        PRIORITY_COMMAND,
    }
    assert commands[USER_MESSAGE_COMMAND].generated_count == 21
    assert commands[USER_MESSAGE_COMMAND].persona_ids == ["fresh-shared"]
    assert contexts[("ChatRoom", "ChatRoom/add_user")].row_count == 11
    carried_wildcard = contexts[("ChatRoom", WILDCARD_LABEL)]
    assert carried_wildcard.row_count == 7
    assert carried_wildcard.raw_candidate_count == 21
    assert carried_wildcard.selected_budget == 11
    assert carried_wildcard.always_include_count == 1
    assert carried_wildcard.coverage_floor == 4
    assert carried_wildcard.coverage_floor_applied is False
    assert contexts[("User", USER_MESSAGE_COMMAND)].row_count == 12
    assert contexts[("PremiumUser", USER_MESSAGE_COMMAND)].row_count == 21
    assert ("PremiumUser", "PremiumUser/deleted_command") not in contexts
    report = training_report.build_report(
        workflow_path, min_rows=1, min_seeds=0)
    chat_room_row = next(
        row for row in report.rows if row.command_name == "ChatRoom/add_user")
    assert chat_room_row.status is not training_report.RowStatus.MISSING


def test_provenance_merge_does_not_hide_a_missing_fresh_shared_record(baseline):
    """A command retrained through base inheritance must get fresh provenance.

    Copying its old command record merely because ``User`` was carried would turn a
    genuine recorder failure in the retrained ``PremiumUser`` context into a false OK.
    """
    workflow_path, version_id = baseline
    previous_recorder = ProvenanceRecorder(workflow_path)
    previous_recorder.record(
        _generation_record(USER_MESSAGE_COMMAND, "old-shared", 12))
    for context_name in ("User", "PremiumUser"):
        previous_recorder.record_context(
            context_name=context_name,
            command_name=USER_MESSAGE_COMMAND,
            status=ContextTrainingStatus.INCLUDED,
            row_count=12,
        )
    previous = _capture_versioned_provenance(
        previous_recorder, workflow_path, version_id)

    fresh_recorder = ProvenanceRecorder(workflow_path)
    fresh_recorder.record_context(
        context_name="PremiumUser",
        command_name=USER_MESSAGE_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=21,
    )
    fresh_recorder.save()
    plan = st.TrainingPlan(
        contexts_to_train=["PremiumUser"],
        contexts_carried_forward=["User"],
        carry_forward_from=version_id,
    )

    assert st.merge_training_provenance(workflow_path, plan, previous)

    commands = ProvenanceRecorder.load(workflow_path)
    contexts = ProvenanceRecorder.load_context_records(workflow_path)
    assert USER_MESSAGE_COMMAND not in commands
    assert contexts[("User", USER_MESSAGE_COMMAND)].row_count == 12
    assert contexts[("PremiumUser", USER_MESSAGE_COMMAND)].row_count == 21


def test_the_wildcard_context_is_in_the_candidate_set(baseline):
    """``*`` has no _commands directory of its own but is trained like any context.

    It is the one context whose name does not come from ``crd.contexts``, so it is
    also the one most easily dropped from a candidate set -- and a context missing
    from the candidate set is neither retrained nor carried forward.
    """
    workflow_path, _version_id = baseline
    assert "*" in st.contexts_for_training(workflow_path)


# ---------------------------------------------------------------------
# Held-out report merging
# ---------------------------------------------------------------------

def test_the_heldout_report_keeps_the_contexts_that_were_not_retrained(baseline):
    """``train()`` rewrites the report from only what it trained.

    Without the merge, a selective run turns a four-context evaluation report into a
    one-context report that still looks complete. The models are fine; the evidence
    about them is what disappears.
    """
    workflow_path, version_id = baseline
    report_path = (st._heldout_path(workflow_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "schema_version": heldout_evaluation.REPORT_SCHEMA_VERSION,
        "totals": {},
        "contexts": [
            {"context": "ChatRoom", "in_distribution_f1": 0.8,
             "routing": {"total": 10, "top1_correct": 8, "in_list_correct": 9}},
            {"context": "PremiumUser", "in_distribution_f1": 0.6,
             "routing": {"total": 10, "top1_correct": 5, "in_list_correct": 6}},
        ],
    }))
    previous = st.capture_heldout_evaluation(workflow_path)

    # What train() would leave behind having retrained only PremiumUser.
    report_path.write_text(json.dumps({
        "schema_version": heldout_evaluation.REPORT_SCHEMA_VERSION,
        "totals": {"contexts": 1},
        "contexts": [
            {"context": "PremiumUser", "in_distribution_f1": 0.7,
             "routing": {"total": 10, "top1_correct": 7, "in_list_correct": 8}},
        ],
    }))
    plan = st.TrainingPlan(
        contexts_to_train=["PremiumUser"],
        contexts_carried_forward=["ChatRoom"],
        carry_forward_from=version_id,
    )

    assert st.merge_heldout_evaluation(workflow_path, plan, previous)

    merged = json.loads(report_path.read_text())
    entries = {entry["context"]: entry for entry in merged["contexts"]}
    assert set(entries) == {"ChatRoom", "PremiumUser"}
    assert entries["ChatRoom"]["carried_forward"] is True
    assert entries["ChatRoom"]["carried_forward_from"] == version_id
    # PremiumUser keeps the number this run measured, not the old one.
    assert entries["PremiumUser"]["in_distribution_f1"] == 0.7
    assert "carried_forward" not in entries["PremiumUser"]
    assert merged["totals"]["routing_total"] == 20
    assert merged["totals"]["routing_top1_correct"] == 15


def test_heldout_merge_rejects_legacy_top1_semantics(baseline):
    """A selective run must not mix inflated v1 top-1 counts into a v2 report."""
    workflow_path, version_id = baseline
    report_path = st._heldout_path(workflow_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {
        "schema_version": heldout_evaluation.REPORT_SCHEMA_VERSION - 1,
        "totals": {},
        "contexts": [
            {
                "context": "ChatRoom",
                "routing": {
                    "total": 10,
                    "top1_correct": 9,
                    "in_list_correct": 9,
                },
            }
        ],
    }
    current = {
        "schema_version": heldout_evaluation.REPORT_SCHEMA_VERSION,
        "totals": {"contexts": 1},
        "contexts": [
            {
                "context": "PremiumUser",
                "routing": {
                    "total": 10,
                    "top1_correct": 7,
                    "in_list_correct": 8,
                },
            }
        ],
    }
    report_path.write_text(json.dumps(current))
    plan = st.TrainingPlan(
        contexts_to_train=["PremiumUser"],
        contexts_carried_forward=["ChatRoom"],
        carry_forward_from=version_id,
    )

    assert not st.merge_heldout_evaluation(workflow_path, plan, previous)
    assert json.loads(report_path.read_text()) == current


def test_the_heldout_merge_does_not_duplicate_an_untouched_report(baseline):
    """A run that retrained nothing leaves the report intact; merging must be a no-op.

    Re-inserting entries that are already there would double every context and
    double-count it in the totals -- a report that reads as twice the evaluation
    evidence that actually exists.
    """
    workflow_path, version_id = baseline
    report_path = st._heldout_path(workflow_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": heldout_evaluation.REPORT_SCHEMA_VERSION,
        "totals": {},
        "contexts": [
            {"context": "ChatRoom", "in_distribution_f1": 0.8,
             "routing": {"total": 10, "top1_correct": 8, "in_list_correct": 9}},
        ],
    }
    report_path.write_text(json.dumps(payload))
    previous = st.capture_heldout_evaluation(workflow_path)
    plan = st.TrainingPlan(
        contexts_carried_forward=["ChatRoom"],
        carry_forward_from=version_id,
    )

    st.merge_heldout_evaluation(workflow_path, plan, previous)

    merged = json.loads(report_path.read_text())
    assert [entry["context"] for entry in merged["contexts"]] == ["ChatRoom"]
    assert merged["totals"]["routing_total"] == 10


# ---------------------------------------------------------------------
# End to end: does the model that was supposed to change, change?
# ---------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_llm_key
def test_selective_retrain_updates_the_changed_context_and_preserves_the_others(
    tmp_path, env_vars
):
    """Two real trains: initial full, then automatic incremental after one command edit.

    The planner tests above prove the closure decides correctly. This proves the
    decision is carried out: that the retrained context's artifacts are genuinely new
    bytes, that the carried-forward contexts' artifacts are the SAME bytes rather
    than quietly regenerated or lost, and that the published version still contains
    every context. That last one is the failure with no symptom -- a version missing
    a context publishes cleanly and un-trains part of the workflow.
    """
    if not _datasets_available():
        pytest.skip("datasets package not installed; training is skipped.")
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip(
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run the real "
            "DSPy parameter-example generation that precedes model training."
        )

    from fastworkflow.train.__main__ import train_workflow

    workflow_path = _copy_workflow(tmp_path)
    _rebuild(workflow_path)
    baseline_corpora, changed_corpus = _preseed_fixed_utterance_cache(
        workflow_path)
    train_workflow(workflow_path)
    _assert_fixed_corpus_was_used(workflow_path, baseline_corpora)

    first_version = artifact_versioning.resolve_current_version(workflow_path)
    assert first_version, "the first train published no version"
    before = {
        context: _artifact_signature(workflow_path, first_version, context)
        for context in ("*", "ChatRoom", "User", "PremiumUser")
    }

    _append_utterance(
        os.path.join(
            workflow_path, "_commands/PremiumUser/send_priority_message.py"),
        NEW_PRIORITY_UTTERANCE,
    )
    RoutingRegistry.clear_registry()
    train_workflow(workflow_path)
    _assert_fixed_corpus_was_used(
        workflow_path,
        {
            **{
                command: corpus
                for command, corpus in baseline_corpora.items()
                if command in {
                    "IntentDetection/go_up",
                    "IntentDetection/reset_context",
                    "IntentDetection/what_can_i_do",
                    "IntentDetection/what_is_current_context",
                    USER_MESSAGE_COMMAND,
                }
            },
            PRIORITY_COMMAND: changed_corpus,
        },
    )

    second_version = artifact_versioning.resolve_current_version(workflow_path)
    assert second_version != first_version

    manifest = artifact_versioning.read_manifest(workflow_path, second_version)
    assert manifest["contexts_retrained"] == ["PremiumUser"]
    assert manifest["contexts_carried_forward"] == ["*", "ChatRoom", "User"]

    after = {
        context: _artifact_signature(workflow_path, second_version, context)
        for context in ("*", "ChatRoom", "User", "PremiumUser")
    }
    assert after["PremiumUser"] != before["PremiumUser"], (
        "the edited command's context was not actually retrained"
    )
    for context in ("*", "ChatRoom", "User"):
        assert after[context] == before[context], (
            f"{context} was carried forward but its artifacts changed"
        )
    for context in ("*", "ChatRoom", "User", "PremiumUser"):
        assert st.context_artifacts_complete(
            workflow_path, second_version, context), (
            f"{context} is missing from the published version; publishing it "
            f"un-trains that part of the workflow"
        )

    # New bytes are necessary but not sufficient: the point of retraining is that the
    # model now recognises the phrasing that was added. Asserted through the real
    # CommandRouter on the published artifacts, which is the path the runtime uses.
    model_dir = os.path.join(
        workflow_path, "___command_info",
        artifact_versioning.context_folder_name("PremiumUser"))
    router = CommandRouter(model_dir)
    labels = router.predict(NEW_PRIORITY_UTTERANCE)
    assert any("send_priority_message" in label for label in labels), (
        f"the retrained PremiumUser model does not route the added utterance; "
        f"got {labels}"
    )

    # NEW_PRIORITY_UTTERANCE is a training row of the changed corpus (it has to be, or the
    # second train would draw fresh utterances and this test would flake), so the
    # assertion above is a memorisation check. This one is not: the paraphrase appears in
    # neither corpus nor seed list, so a regression that reproduced its training rows
    # while losing every rewording of them fails here and nowhere else. bd fix-k0i.30.
    assert HELDOUT_PRIORITY_UTTERANCE not in changed_corpus
    assert HELDOUT_PRIORITY_UTTERANCE not in _fixed_generated_corpus(PRIORITY_COMMAND)
    heldout_labels = router.predict(HELDOUT_PRIORITY_UTTERANCE)
    assert any("send_priority_message" in label for label in heldout_labels), (
        f"the retrained PremiumUser model routes its training rows but not the held-out "
        f"paraphrase {HELDOUT_PRIORITY_UTTERANCE!r}; got {heldout_labels}"
    )


def _artifact_signature(workflow_path: str, version_id: str, context: str) -> set:
    """(relative path, sha256) for every artifact file of one context.

    Content-addressed rather than (size, mtime): carry-forward hardlinks preserve size
    and mtime exactly, so a stat-based signature could not tell a carried context from a
    retrained one that happened to produce the same-sized bytes. bd fix-k0i.48.
    """
    folder = artifact_versioning.version_dir(
        workflow_path, version_id
    ) / artifact_versioning.context_folder_name(context)
    signature = set()
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            signature.add((os.path.relpath(path, folder), digest))
    return signature
