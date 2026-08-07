"""Selective retraining: deciding which contexts must be retrained when a command changes.

Retraining a single command is **not local**. "Only this command changed" is never
only one context, because two independent inheritance axes decide the blast radius:

``base`` — command inheritance, declared in ``_commands/context_inheritance_model.json``.
    ``CommandContextModel.commands(ctx)`` already merges every base context's commands
    into ``ctx``'s effective label space, so a changed command appears directly in the
    label space of every context that inherits it.

``parent`` — context ancestry, declared in ``context_hierarchy_model.json``.
    A context's ``wildcard`` class is assembled from its *ancestors'* utterances
    (see ``model_pipeline_training.cache_ancestor_utterances``). A command changing in
    context ``A`` therefore changes the *negative* class of every descendant of ``A``,
    even though none of those descendants list that command in their own label space.

The second axis is the one that makes naive per-command retraining wrong. It desyncs
wildcard classes across the hierarchy with no error, no exception, and no visible
symptom -- the models simply disagree about what "belongs upstairs".

The closure is computed once rather than iterated to a fixed point: a context that is
dirty *only* because its wildcard class changed has no changed commands of its own, so
it does not propagate dirtiness any further. ``get_ancestor_contexts`` already returns
the full transitive chain, so one pass covers arbitrarily deep hierarchies.

Why the closure alone is not enough
-----------------------------------
The command-level closure above answers "which contexts are affected by a command
whose *file* changed". It does not answer "is this context's model still the model
this workflow would produce today", and those are different questions. Three change
classes move a context's label space without any command file changing:

* an edit to ``_commands/context_inheritance_model.json`` (``base``) changes which
  commands a context carries, and therefore which commands reach the *wildcard*
  classes of that context's descendants -- with no command file touched anywhere;
* an edit to ``context_hierarchy_model.json`` (``parent``) changes whose utterances
  form a context's wildcard class, and can change whether ``WILDCARD_LABEL`` is a
  label in that context **at all**: the trainer drops the escalation class where
  ``ancestor_utterances - context_utterances`` is empty, so gaining or losing an
  ancestor can add or remove a class;
* a change to the reserved-label inputs themselves -- ``PARAMETER_VALUE_PLACEHOLDERS``
  is a code constant that is trained as a class in **every** context.

So this module records a full ``TrainingSignature`` alongside each artifact version
and refuses to carry a context forward unless that context's recorded signature is
*identical* to the one the workflow would produce now. The closure decides what is
definitely dirty; the signature comparison decides what is provably clean. A context
is carried forward only when both agree, plus a third check that its artifacts are
actually present and complete in the version being carried from.

Safe by construction
--------------------
Every path that cannot answer "this context is provably unchanged" retrains. There is
no branch anywhere below that skips a context on incomplete information: a missing
baseline, an unreadable file, an unknown context, an unhydratable command, a changed
global input, or an incomplete artifact directory all resolve to *retrain*. The cost
of a false retrain is time; the cost of a false skip is a silently stale model that
nothing in the package would detect (spec F1/F10 -- there is no signal that would
surface one). Those are not comparable, so the tie always breaks the same way.

Where the baseline lives, and why it is versioned
------------------------------------------------
The signature is written to ``versions/<id>/training_signature.json`` and the baseline
is read from whichever version is **current**. Storing it at the top level of
``___command_info`` instead would break internal recovery: moving the artifacts back
without their signature would leave a top-level signature describing
the newest train, so the next automatic incremental run would compare today's sources
against a baseline that does not describe the artifacts it is about to carry forward,
and would carry forward stale models while reporting "nothing changed". Versioning the
signature makes rollback self-consistent at no cost.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ValidationError

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.nlu_labels import PARAMETER_VALUE_PLACEHOLDERS, WILDCARD_LABEL
from fastworkflow.train import (
    artifact_versioning,
    determinism,
    heldout_evaluation,
    utterance_cache,
)
from fastworkflow.train.personas import active_persona_source_label
from fastworkflow.utils.logging import logger


FINGERPRINT_FILENAME = "command_fingerprints.json"

# Written into each artifact version; read from the CURRENT version to obtain the
# baseline the next automatic incremental run diffs against.
SIGNATURE_FILENAME = "training_signature.json"

# Bump BY HAND whenever the trained-artifact layout, the label-assignment rules, or
# anything else about what `train()` writes changes in a way that makes an older
# context directory not interchangeable with a freshly trained one. A signature
# recorded under a different value never matches, so the next run is a full retrain.
#
# This exists because there is no reliable automatic detector for "the trainer now
# produces different artifacts". The module digests below cover the trainer and its
# local generation/class-balancing helpers, but this constant remains the backstop
# for changes outside those modules. Bump it when in doubt: the cost is one slow run.
ARTIFACT_FORMAT_VERSION = 1

# Full-module digests are intentionally conservative. A helper or comments-only edit
# may trigger a full retrain; missing a helper edit could silently carry stale models.
TRAINER_SOURCE_MODULE = "fastworkflow.model_pipeline_training"
GENERATOR_SOURCE_MODULE = "fastworkflow.train.generate_synthetic"
CLASS_BALANCE_SOURCE_MODULE = "fastworkflow.train.class_balance"

# Every file `train()` writes per context. All of them must exist before a context is
# eligible to be carried forward -- a half-written directory from an interrupted run
# would otherwise be inherited by every subsequent selective run.
REQUIRED_CONTEXT_ARTIFACTS: tuple[str, ...] = (
    "tinymodel.pth",
    "largemodel.pth",
    "threshold.json",
    "tiny_ambiguous_threshold.json",
    "large_ambiguous_threshold.json",
    "label_encoder.pkl",
)


class SelectiveTrainingError(RuntimeError):
    """Raised when a selective run cannot be completed safely.

    Deliberately fatal. The alternative -- carrying on and publishing a version that
    is missing a context -- would make `publish_version` remove that context's
    compatibility entry, silently un-training part of a workflow.
    """


class CommandFingerprint(BaseModel):
    """Identity of one command's training inputs.

    ``source_sha256`` covers the command file itself; ``seed_utterances_sha256`` covers
    the declared seed list separately so that a pure prose edit to a docstring can be
    distinguished from a change that actually alters the training data.

    ``resolved`` is False when the fingerprint could not be computed -- the command
    would not hydrate, or its source file could not be read. An unresolved fingerprint
    compares unequal to everything including itself, so the command is always dirty.
    Without that, two consecutive runs that both failed to read a command's source
    would produce two identical ``(None, None)`` fingerprints and the command would
    look unchanged forever, which is precisely the silent-staleness failure.
    """

    command_name: str
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    seed_utterances_sha256: Optional[str] = None
    resolved: bool = True
    unresolved_reason: Optional[str] = None

    def training_inputs_differ(self, other: "CommandFingerprint") -> bool:
        if not self.resolved or not other.resolved:
            return True
        return (
            self.source_sha256 != other.source_sha256
            or self.seed_utterances_sha256 != other.seed_utterances_sha256
        )


class ContextSignature(BaseModel):
    """Everything that decides one context's label space and its wildcard class.

    ``label_space`` is the set of labels the trainer iterates for this context --
    ``crd.contexts[ctx]`` (already ``base``-resolved) unioned with the core commands.

    ``ancestors`` and ``wildcard_sources`` describe the *escalation* class.
    ``wildcard_sources`` is the union of ``context_model.commands(a)`` over every
    ancestor ``a``, which is exactly what ``cache_ancestor_utterances`` walks. Both are
    recorded because they fail independently: a context can keep the same ancestor set
    while an ancestor's own command list changes underneath it (someone adds a ``base``
    entry to the ancestor), and that changes this context's wildcard class without
    changing anything else observable here.

    ``expects_wildcard_label`` records whether ``WILDCARD_LABEL`` is expected to be a
    class at all. The trainer's real condition is
    ``ancestor_utterances - context_utterances != {}``, which is a property of
    generated TEXT and cannot be evaluated without generating. This is the
    command-level approximation of it. It is recorded for legibility rather than for
    safety: every input the real condition depends on -- the ancestor set, the
    ancestor command lists, the local command list, and each command's fingerprint --
    is compared exactly elsewhere in this signature, so a change that flips the real
    condition necessarily changes something already compared.
    """

    context_name: str
    label_space: list[str] = []
    ancestors: list[str] = []
    wildcard_sources: list[str] = []
    expects_wildcard_label: bool = False


class TrainingSignature(BaseModel):
    """The complete set of inputs that decide what a training run produces.

    Split into a `globals`-style header and a per-context body: a header mismatch
    invalidates every context at once (a different base model, a different seed, a
    different trainer/generator/helper module), while a body mismatch invalidates
    one context.
    """

    format_version: int = ARTIFACT_FORMAT_VERSION
    seed: Optional[int] = None
    tiny_model: Optional[str] = None
    large_model: Optional[str] = None
    syndata_model: Optional[str] = None
    persona_source: Optional[str] = None
    generator_source_digest: Optional[str] = None
    trainer_source_digest: Optional[str] = None
    class_balance_source_digest: Optional[str] = None
    parameter_value_placeholders_sha256: Optional[str] = None
    command_fingerprints: dict[str, CommandFingerprint] = {}
    contexts: dict[str, ContextSignature] = {}

    def global_inputs(self) -> dict:
        """The header fields, as a plain dict, for a field-by-field diff."""
        return {
            "format_version": self.format_version,
            "seed": self.seed,
            "tiny_model": self.tiny_model,
            "large_model": self.large_model,
            "syndata_model": self.syndata_model,
            "persona_source": self.persona_source,
            "generator_source_digest": self.generator_source_digest,
            "trainer_source_digest": self.trainer_source_digest,
            "class_balance_source_digest": self.class_balance_source_digest,
            "parameter_value_placeholders_sha256":
                self.parameter_value_placeholders_sha256,
        }


class TrainingPlan(BaseModel):
    """The computed retraining closure, in a form a developer can audit.

    ``reasons`` is the point of this object. The developer must be able to see *why*
    each context was pulled in -- particularly the ancestor-driven entries, which are
    the ones that look wrong until you know about the wildcard class.

    ``global_reasons`` records why automatic planning chose a full retrain.
    """

    dirty_commands: list[str] = []
    contexts_to_train: list[str] = []
    contexts_carried_forward: list[str] = []
    reasons: dict[str, list[str]] = {}
    global_reasons: list[str] = []
    is_full_retrain: bool = False
    carry_forward_from: Optional[str] = None


def _sha256_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _sha256_strings(values: Iterable[str]) -> str:
    # Sorted so that a reordering of the seed list is not reported as a change:
    # the trainer consumes the seeds as a set-like collection.
    payload = "\n".join(sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_source_digest(module_name: str, module_spec=None) -> str:
    """Hash the complete source module, including helpers and comments.

    ``module_spec`` is an explicit seam for callers that already resolved a module
    (and for tests using a temporary source copy). Normal callers leave it unset.

    Source loaders are consulted before importing the module. That avoids the
    ``model_pipeline_training`` <-> ``selective_training`` import cycle and lets
    zipimport loaders provide bytes from inside an archive. Frozen or sourceless
    modules cannot prove their training code is unchanged, so this raises and
    ``_global_input`` supplies a fresh never-matching value for each run.
    """
    spec = module_spec or importlib.util.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise OSError(f"no importable module spec for {module_name}")

    origin = spec.origin
    source_origin = str(origin or "")
    if all(
        (
            source_origin,
            source_origin not in {"built-in", "frozen"},
            Path(source_origin).suffix in {".py", ".pyw"},
        )
    ):
        get_data = getattr(spec.loader, "get_data", None)
        if callable(get_data):
            try:
                source_bytes = get_data(source_origin)
            except (ImportError, OSError):
                pass
            else:
                return hashlib.sha256(source_bytes).hexdigest()

    get_source = getattr(spec.loader, "get_source", None)
    if callable(get_source):
        try:
            source = get_source(module_name)
        except (ImportError, OSError):
            source = None
        if source is not None:
            return hashlib.sha256(source.encode("utf-8")).hexdigest()

    raise OSError(f"source is unavailable for module {module_name}")


def _unresolvable(command_name: str, reason: str) -> CommandFingerprint:
    """A fingerprint that can never compare equal, so the command is always dirty."""
    return CommandFingerprint(
        command_name=command_name,
        resolved=False,
        unresolved_reason=reason,
    )


def compute_command_fingerprints(workflow_folderpath: str) -> dict[str, CommandFingerprint]:
    """Fingerprint every command's training inputs for *workflow_folderpath*.

    ``ensure_command_hydrated`` is called first, and that call is load-bearing rather
    than defensive. ``map_command_2_utterance_metadata`` is populated lazily: a command
    directory restored from ``command_directory.json`` carries metadata only for the
    commands something has already touched this process. Reading the map directly
    would therefore hand back an empty fingerprint for every un-hydrated command --
    and two runs that both left a command un-hydrated would agree, so an edited
    command would be reported unchanged. Hydration order is not a property anything
    else guarantees, so this cannot be left to luck.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    cmd_dir = crd.command_directory

    fingerprints: dict[str, CommandFingerprint] = {}
    for command_name in cmd_dir.get_commands():
        try:
            cmd_dir.ensure_command_hydrated(command_name)
        except Exception as exc:  # noqa: BLE001 - any failure means "unknown"
            fingerprints[command_name] = _unresolvable(
                command_name, f"could not hydrate command metadata: {exc}")
            continue

        source_path: Optional[str] = None
        seed_hash: Optional[str] = None

        utterance_metadata = cmd_dir.map_command_2_utterance_metadata.get(command_name)
        if utterance_metadata is not None:
            source_path = utterance_metadata.generated_utterances_module_filepath
            seed_hash = _sha256_strings(
                list(utterance_metadata.plain_utterances)
                + list(utterance_metadata.template_utterances)
            )
        else:
            # A command with no `Signature.generate_utterances` contributes no rows to
            # any classifier, so it has no seed list to hash -- but its FILE still
            # decides that, and an edit adding `generate_utterances` must be seen.
            # Fall back to the command's own module path.
            try:
                source_path = cmd_dir.get_command_metadata(
                    command_name).response_generation_module_path
            except (KeyError, AttributeError) as exc:
                fingerprints[command_name] = _unresolvable(
                    command_name, f"no utterance metadata and no source path: {exc}")
                continue

        if not source_path:
            fingerprints[command_name] = _unresolvable(
                command_name, "command metadata declares no source path")
            continue

        source_sha = _sha256_file(source_path)
        if source_sha is None:
            # The file is named but unreadable. Treating that as "no change" would be
            # the unsafe direction, so it is an unresolved fingerprint instead.
            fingerprints[command_name] = _unresolvable(
                command_name, f"source file could not be read: {source_path}")
            continue

        fingerprints[command_name] = CommandFingerprint(
            command_name=command_name,
            source_path=source_path,
            source_sha256=source_sha,
            seed_utterances_sha256=seed_hash,
        )

    return fingerprints


def fingerprint_path(workflow_folderpath: str) -> str:
    return os.path.join(
        CommandDirectory.get_commandinfo_folderpath(workflow_folderpath),
        FINGERPRINT_FILENAME,
    )


def save_command_fingerprints(
    workflow_folderpath: str, fingerprints: dict[str, CommandFingerprint]
) -> str:
    path = fingerprint_path(workflow_folderpath)
    payload = {name: fp.model_dump() for name, fp in sorted(fingerprints.items())}
    Path(path).write_text(json.dumps(payload, indent=2))
    return path


def load_command_fingerprints(workflow_folderpath: str) -> dict[str, CommandFingerprint]:
    path = Path(fingerprint_path(workflow_folderpath))
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        name: CommandFingerprint.model_validate(value)
        for name, value in payload.items()
    }


def changed_commands(
    previous: dict[str, CommandFingerprint],
    current: dict[str, CommandFingerprint],
) -> set[str]:
    """Commands whose training inputs differ between two fingerprint sets.

    Additions and removals both count as changes. A removal matters because the
    removed command's utterances were part of some other context's wildcard class.
    """
    dirty = set(previous.keys()) ^ set(current.keys())
    for name in set(previous) & set(current):
        if current[name].training_inputs_differ(previous[name]):
            dirty.add(name)
    return dirty


def close_dirty_contexts(
    dirty_commands: set[str],
    context_commands: dict[str, set[str]],
    context_ancestors: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Close a set of changed commands upwards over the context hierarchy.

    Pure function over plain data so the rule can be tested without building a
    workflow. ``context_commands`` must already have ``base`` inheritance merged in
    (which is what ``CommandContextModel.commands`` returns).

    Returns context name -> the list of reasons it must be retrained.
    """
    reasons: dict[str, list[str]] = {}
    if not dirty_commands:
        return reasons

    for context_name, own_commands in context_commands.items():
        # Axis 1: the command is in this context's own (base-merged) label space.
        if hit := sorted(own_commands & dirty_commands):
            reasons.setdefault(context_name, []).append(
                f"label space contains changed command(s): {', '.join(hit)}"
            )

        # Axis 2: an ancestor's command changed, so this context's wildcard class
        # -- which is built from ancestor utterances -- is now stale.
        for ancestor in context_ancestors.get(context_name, []):
            ancestor_commands = context_commands.get(ancestor, set())
            if ancestor_hit := sorted(ancestor_commands & dirty_commands):
                reasons.setdefault(context_name, []).append(
                    f"wildcard class is stale: ancestor '{ancestor}' changed "
                    f"command(s): {', '.join(ancestor_hit)}"
                )

    return reasons


def build_context_maps(
    workflow_folderpath: str, context_names: Iterable[str]
) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Build the plain-data maps the closure and the signature consume.

    Returns ``(context_commands, context_ancestors, unresolved)`` where ``unresolved``
    maps a context name to the reasons its model could not be described. The previous
    shape of this function swallowed lookup failures and substituted a smaller command
    set, which silently weakened the closure -- a context whose ancestors could not be
    resolved appeared to have none, so no ancestor change could ever dirty it. Failures
    are now reported so the caller can force those contexts to retrain.

    Ancestor command lists come from ``context_model.commands``, NOT from
    ``crd.contexts``, because that is what ``cache_ancestor_utterances`` walks: core
    commands are deliberately absent from an ancestor's contribution to a wildcard
    class. The context's OWN label space does include them, which is why the two are
    built from different sources here.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    context_model = crd.context_model
    core_commands = set(crd.command_directory.core_command_names)

    context_commands: dict[str, set[str]] = {}
    context_ancestors: dict[str, list[str]] = {}
    unresolved: dict[str, list[str]] = {}

    for context_name in context_names:
        try:
            effective = set(context_model.commands(context_name))
        except Exception as exc:  # noqa: BLE001
            # `crd.contexts` is the trainer's own source for the label space, so it is
            # the right fallback -- but the disagreement itself is reported, because a
            # context the model cannot resolve is a context whose closure we cannot
            # trust.
            effective = set(crd.contexts.get(context_name, []))
            unresolved.setdefault(context_name, []).append(
                f"context model could not resolve commands: {exc}")
        context_commands[context_name] = effective | core_commands

        try:
            context_ancestors[context_name] = list(
                context_model.get_ancestor_contexts(context_name)
            )
        except Exception as exc:  # noqa: BLE001
            context_ancestors[context_name] = []
            unresolved.setdefault(context_name, []).append(
                f"context model could not resolve ancestors: {exc}")

    return context_commands, context_ancestors, unresolved


def descendants_of(
    context_name: str, context_ancestors: dict[str, list[str]]
) -> set[str]:
    """Every context that lists *context_name* somewhere in its ancestor chain."""
    return {
        candidate
        for candidate, ancestors in context_ancestors.items()
        if context_name in ancestors
    }


def contexts_for_training(workflow_folderpath: str) -> set[str]:
    """The contexts ``train()`` produces artifacts for, for *workflow_folderpath*.

    This lives here rather than inline in ``train()`` so that the planner and the
    trainer cannot disagree about the candidate set. They must not: a context the
    planner never considered is a context that is neither retrained nor carried
    forward, and ``publish_version`` removes the compatibility entry of any context
    a version does not contain -- so the disagreement would present as part of a
    workflow silently becoming untrained.

    The internal ``command_metadata_extraction`` contexts are excluded from an app
    workflow because their models live in the internal workflow, and the global
    wildcard context ``"*"`` is always included.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    internal_wf_path = fastworkflow.get_internal_workflow_path(
        "command_metadata_extraction")
    internal_contexts = set(
        fastworkflow.CommandContextModel.load(internal_wf_path)._command_contexts.keys()
    )

    if "command_metadata_extraction" in workflow_folderpath:
        # `ErrorCorrection` is excluded because the internal workflow declares it as a
        # context but trains no classifier for it.
        return set(internal_contexts) - {'ErrorCorrection'}
    return (set(crd.contexts.keys()) - internal_contexts) | {'*'}


# ---------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------

def _global_input(getter, label: str) -> Optional[str]:
    """Read one global signature input, or return a value that can never match.

    A failure returns a fresh uuid rather than a fixed sentinel. A fixed sentinel
    would compare equal to the previous run's sentinel, so two consecutive runs that
    both failed to read (say) a trainer module digest would agree that nothing
    changed -- turning an inability to check into a claim that the check passed.
    """
    try:
        value = getter()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Selective training could not read the '{label}' signature input "
            f"({exc}); this run cannot carry any context forward."
        )
        return f"unavailable:{uuid.uuid4().hex}"
    return None if value is None else str(value)


def compute_context_signatures(
    context_commands: dict[str, set[str]],
    context_ancestors: dict[str, list[str]],
    context_names: Iterable[str],
) -> dict[str, ContextSignature]:
    """Describe each context's label space and wildcard-class inputs."""
    signatures: dict[str, ContextSignature] = {}
    for context_name in context_names:
        own = context_commands.get(context_name, set())
        ancestors = list(context_ancestors.get(context_name, []))

        wildcard_sources: set[str] = set()
        for ancestor in ancestors:
            # Mirrors cache_ancestor_utterances, which skips the reserved wildcard
            # command when collecting an ancestor's utterances.
            wildcard_sources |= {
                command
                for command in context_commands.get(ancestor, set())
                if command.split("/")[-1] != WILDCARD_LABEL
            }

        signatures[context_name] = ContextSignature(
            context_name=context_name,
            label_space=sorted(own),
            ancestors=sorted(ancestors),
            wildcard_sources=sorted(wildcard_sources),
            expects_wildcard_label=bool(wildcard_sources - own),
        )
    return signatures


def compute_training_signature(
    workflow_folderpath: str,
    context_names: Iterable[str],
    seed: Optional[int] = None,
) -> tuple[TrainingSignature, dict[str, list[str]]]:
    """Build the signature this workflow would train under right now.

    Returns ``(signature, unresolved)``; ``unresolved`` names contexts the context
    model could not describe, which the caller must retrain regardless of the diff.
    """
    contexts = sorted(context_names)
    context_commands, context_ancestors, unresolved = build_context_maps(
        workflow_folderpath, contexts)

    signature = TrainingSignature(
        format_version=ARTIFACT_FORMAT_VERSION,
        seed=(
            int(seed)
            if seed is not None
            else _global_int(determinism.get_training_seed)
        ),
        tiny_model=_global_input(
            lambda: fastworkflow.get_env_var(
                "INTENT_DETECTION_TINY_MODEL",
                default="google/bert_uncased_L-4_H-128_A-2"),
            "INTENT_DETECTION_TINY_MODEL"),
        large_model=_global_input(
            lambda: fastworkflow.get_env_var(
                "INTENT_DETECTION_LARGE_MODEL", default="distilbert-base-uncased"),
            "INTENT_DETECTION_LARGE_MODEL"),
        syndata_model=_global_input(
            lambda: fastworkflow.get_env_var("LLM_SYNDATA_GEN"), "LLM_SYNDATA_GEN"),
        persona_source=_global_input(
            active_persona_source_label, "persona source"),
        generator_source_digest=_global_input(
            lambda: _module_source_digest(GENERATOR_SOURCE_MODULE),
            "synthetic-generation module source"),
        trainer_source_digest=_global_input(
            lambda: _module_source_digest(TRAINER_SOURCE_MODULE),
            "trainer module source"),
        class_balance_source_digest=_global_input(
            lambda: _module_source_digest(CLASS_BALANCE_SOURCE_MODULE),
            "class-balance module source"),
        # The bare-value literals are a code constant that becomes a real class in
        # EVERY context (`PARAMETER_VALUE_LABEL`). Editing the list changes every
        # context's training data without touching a single workflow file.
        parameter_value_placeholders_sha256=_sha256_strings(
            PARAMETER_VALUE_PLACEHOLDERS),
        command_fingerprints=compute_command_fingerprints(workflow_folderpath),
        contexts=compute_context_signatures(
            context_commands, context_ancestors, contexts),
    )
    return signature, unresolved


def _global_int(getter) -> Optional[int]:
    try:
        return int(getter())
    except Exception:  # noqa: BLE001
        return None


# Mirrored here so the safety check remains explicit and cheap. Importing
# `utterance_cache` does not pull litellm/datasets in at import time.
# `test_selective_training.py` asserts the constant still agrees with its public value.
_MODE_REUSE: str = "reuse"


def _resolve_cache_mode() -> str:
    """The R6 utterance-cache mode this run will use, or a never-matching value."""
    try:
        return utterance_cache.resolve_cache_mode()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Could not resolve the utterance cache mode ({exc}); treating this run "
            f"as ineligible for selective training."
        )
        return f"unavailable:{uuid.uuid4().hex}"


def signature_path(workflow_folderpath: str, version_id: str) -> Path:
    """Path of the signature recorded inside *version_id*."""
    return artifact_versioning.version_dir(
        workflow_folderpath, version_id) / SIGNATURE_FILENAME


def save_training_signature(
    workflow_folderpath: str, version_id: str, signature: TrainingSignature
) -> str:
    path = signature_path(workflow_folderpath, version_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signature.model_dump(), indent=2, sort_keys=True))
    return str(path)


def load_training_signature(
    workflow_folderpath: str, version_id: Optional[str]
) -> Optional[TrainingSignature]:
    """Load *version_id*'s signature, or None when there is not a usable one."""
    if not version_id:
        return None
    try:
        path = signature_path(workflow_folderpath, version_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return TrainingSignature.model_validate(json.loads(path.read_text()))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Ignoring unreadable training signature {path}: {exc}. "
            f"The next run will be a full retrain."
        )
        return None


def context_artifacts_complete(
    workflow_folderpath: str, version_id: str, context_name: str
) -> bool:
    """True when *version_id* holds a complete artifact set for *context_name*.

    Checked before a context is carried forward, because the closure and the
    signature both reason about SOURCES and neither of them knows whether the bytes
    they would carry forward exist. An interrupted run leaves a directory holding
    some of these files; publishing a version that inherited it would produce a
    workflow that looks trained and raises `FileNotFoundError` on the first turn.
    """
    try:
        folder = artifact_versioning.version_dir(
            workflow_folderpath, version_id
        ) / artifact_versioning.context_folder_name(context_name)
    except ValueError:
        return False
    if not folder.is_dir():
        return False
    # save_pretrained writes tinymodel.pth / largemodel.pth as DIRECTORIES, the
    # threshold files as files. `exists()` covers both without asserting which.
    return all((folder / name).exists() for name in REQUIRED_CONTEXT_ARTIFACTS)


def _diff_global_inputs(
    previous: TrainingSignature, current: TrainingSignature
) -> list[str]:
    previous_inputs = previous.global_inputs()
    current_inputs = current.global_inputs()
    return [
        f"{key} changed: {previous_inputs[key]!r} -> {current_inputs[key]!r}"
        for key in sorted(current_inputs)
        if previous_inputs.get(key) != current_inputs[key]
    ]


def _diff_context_signature(
    previous: Optional[ContextSignature], current: ContextSignature
) -> list[str]:
    if previous is None:
        return ["no recorded signature in the version being carried from"]
    differences: list[str] = []
    if previous.label_space != current.label_space:
        added = sorted(set(current.label_space) - set(previous.label_space))
        removed = sorted(set(previous.label_space) - set(current.label_space))
        differences.append(
            "label space changed"
            + (f"; added {', '.join(added)}" if added else "")
            + (f"; removed {', '.join(removed)}" if removed else "")
        )
    if previous.ancestors != current.ancestors:
        differences.append(
            f"ancestors changed: {previous.ancestors} -> {current.ancestors}")
    if previous.wildcard_sources != current.wildcard_sources:
        added = sorted(set(current.wildcard_sources) - set(previous.wildcard_sources))
        removed = sorted(set(previous.wildcard_sources) - set(current.wildcard_sources))
        differences.append(
            "wildcard class sources changed"
            + (f"; added {', '.join(added)}" if added else "")
            + (f"; removed {', '.join(removed)}" if removed else "")
        )
    if previous.expects_wildcard_label != current.expects_wildcard_label:
        differences.append(
            f"escalation ({WILDCARD_LABEL}) class presence changed: "
            f"{previous.expects_wildcard_label} -> {current.expects_wildcard_label}"
        )
    return differences


def _full_retrain(candidates: list[str], reason: str) -> TrainingPlan:
    return TrainingPlan(
        contexts_to_train=candidates,
        is_full_retrain=True,
        global_reasons=[reason],
        reasons={ctx: [reason] for ctx in candidates},
    )


def compute_training_plan(
    workflow_folderpath: str,
    candidate_contexts: Iterable[str],
    only_commands: Optional[Iterable[str]] = None,
    only_contexts: Optional[Iterable[str]] = None,
    changed_only: bool = True,
    seed: Optional[int] = None,
    carry_forward_from: Optional[str] = None,
    cache_mode: Optional[str] = None,
) -> tuple[TrainingPlan, TrainingSignature]:
    """Decide which contexts to retrain and which to carry forward.

    The default is automatic change detection. It falls back to a full retrain whenever
    no trustworthy baseline exists, a global input changed, artifacts are incomplete, or
    the utterance cache cannot safely support carry-forward.

    ``only_commands`` and ``only_contexts`` are retained as a programmatic affordance
    but are deliberately NOT exposed on the CLI. They express a developer's ASSERTION
    about what changed, and an assertion that is wrong in the unsafe direction is the
    exact failure this module exists to prevent. They are additive here: whatever they
    name is added to the dirty set computed from fingerprints, never substituted for
    it, and every safety check below still applies.

    Returns ``(plan, signature)``. The signature is returned rather than recomputed by
    the caller because it must be written into the version that this plan produces,
    and recomputing it after training could pick up a source edit made mid-run.
    """
    candidates = sorted(candidate_contexts)

    current_signature, unresolved = compute_training_signature(
        workflow_folderpath, candidates, seed=seed)

    if not (only_commands or only_contexts or changed_only):
        return (
            _full_retrain(candidates, "full retrain (no selector given)"),
            current_signature,
        )

    previous_signature = load_training_signature(
        workflow_folderpath, carry_forward_from)
    if carry_forward_from is None:
        return (
            _full_retrain(
                candidates,
                "full retrain (no published artifact version to carry forward from)"),
            current_signature,
        )
    if previous_signature is None:
        return (
            _full_retrain(
                candidates,
                f"full retrain (artifact version {carry_forward_from} records no "
                f"usable training signature to diff against)"),
            current_signature,
        )

    # `--regenerate-utterances` (and the `off`/`aggregate` cache modes) change the
    # generated utterances of every command the run touches. A retrained context would
    # then be built on fresh text while a carried-forward context keeps the old text --
    # including in its wildcard class, which is assembled from its ancestors'
    # utterances. That is exactly the internally inconsistent version AR3 warns about,
    # so the two features are mutually exclusive rather than merely discouraged.
    cache_mode = cache_mode or _resolve_cache_mode()
    if cache_mode != _MODE_REUSE:
        plan = _full_retrain(
            candidates,
            f"full retrain (utterance cache mode is {cache_mode!r}; only "
            f"{_MODE_REUSE!r} can carry a context forward safely)")
        return plan, current_signature

    if global_differences := _diff_global_inputs(
        previous_signature, current_signature
    ):
        # One header field decides every context, so there is nothing to carry.
        plan = _full_retrain(
            candidates, "full retrain (a global training input changed)")
        plan.global_reasons.extend(global_differences)
        return plan, current_signature

    dirty: set[str] = set(only_commands or ())
    dirty |= changed_commands(
        previous_signature.command_fingerprints,
        current_signature.command_fingerprints,
    )

    context_commands, context_ancestors, _ = build_context_maps(
        workflow_folderpath, candidates)
    reasons = close_dirty_contexts(dirty, context_commands, context_ancestors)

    # An explicitly named context is retrained, and so is every descendant, because
    # the named context's utterances constitute those descendants' wildcard class.
    for context_name in only_contexts or []:
        reasons.setdefault(context_name, []).append("explicitly requested")
        for descendant in sorted(descendants_of(context_name, context_ancestors)):
            reasons.setdefault(descendant, []).append(
                f"wildcard class is stale: ancestor '{context_name}' was retrained"
            )

    for context_name in candidates:
        # A context the context model could not describe is a context whose closure
        # was computed on incomplete information.
        for reason in unresolved.get(context_name, []):
            reasons.setdefault(context_name, []).append(reason)

        # The signature comparison is INDEPENDENT of the closure above and catches
        # what the closure structurally cannot: changes to the context models
        # themselves, which move a label space without moving any command file.
        for difference in _diff_context_signature(
            previous_signature.contexts.get(context_name),
            current_signature.contexts[context_name],
        ):
            reasons.setdefault(context_name, []).append(difference)

        # Last: are the bytes we would carry forward actually there and complete?
        if context_name not in reasons and not context_artifacts_complete(
            workflow_folderpath, carry_forward_from, context_name
        ):
            reasons.setdefault(context_name, []).append(
                f"artifact version {carry_forward_from} has no complete model "
                f"artifact set for this context"
            )

    to_train = sorted(ctx for ctx in candidates if ctx in reasons)
    carried = sorted(ctx for ctx in candidates if ctx not in reasons)

    # The mode check above asks whether the cache is ALLOWED to be reused. It does not
    # ask whether there is anything in it. A deleted cache directory, a fresh clone, or
    # a cache that was never committed all leave mode == reuse with no entries, and then
    # every command in a retrained context is redrawn from the LLM while the carried
    # contexts keep the superseded text in their wildcard class -- the internally
    # inconsistent version AR3 says versioning cannot detect and this must prevent.
    if to_train and carried:
        if uncached := _shared_commands_absent_from_cache(
            workflow_folderpath, to_train, carried, context_commands, context_ancestors
        ):
            plan = _full_retrain(
                candidates,
                f"full retrain ({len(uncached)} command(s) contributing to both a "
                f"retrained and a carried-forward context have no cached utterances, "
                f"so they would be redrawn from the LLM while the carried contexts "
                f"keep the superseded text: "
                f"{', '.join(sorted(uncached)[:5])}"
                f"{' ...' if len(uncached) > 5 else ''})",
            )
            return plan, current_signature

    # `is_full_retrain` stays False even when every context turned out dirty, so the
    # per-context reasons are still rendered. "Everything was dirty, and here is why"
    # and "no selector was given" are different answers to the same question, and a
    # developer inspecting an unexpectedly broad automatic retrain needs the first one.
    plan = TrainingPlan(
        dirty_commands=sorted(dirty),
        contexts_to_train=to_train,
        contexts_carried_forward=carried,
        reasons={ctx: reasons[ctx] for ctx in to_train},
        is_full_retrain=False,
        carry_forward_from=carry_forward_from,
    )
    if not carried:
        plan.global_reasons.append(
            "every context was dirty; nothing could be carried forward")
    return plan, current_signature


def carry_forward_contexts(
    workflow_folderpath: str,
    plan: TrainingPlan,
    to_version: str,
) -> list[str]:
    """Link each carried-forward context's artifacts into *to_version*.

    This is what makes a selective run produce a COMPLETE version. Without it the new
    version would hold only the retrained contexts, and `publish_version` -- which
    removes compatibility entries for contexts a version does not have -- would
    un-train the rest of the workflow while reporting success.

    Raises `SelectiveTrainingError` rather than returning partial success. The caller
    runs this BEFORE publishing, so a failure here leaves the previous version current
    and complete.
    """
    if not plan.contexts_carried_forward:
        return []
    if not plan.carry_forward_from:
        raise SelectiveTrainingError(
            "The plan carries contexts forward but names no source version. Refusing "
            "to publish a version that would be missing "
            f"{len(plan.contexts_carried_forward)} context(s)."
        )

    carried: list[str] = []
    for context_name in plan.contexts_carried_forward:
        if not artifact_versioning.carry_forward_context(
            workflow_folderpath,
            plan.carry_forward_from,
            to_version,
            context_name,
        ):
            raise SelectiveTrainingError(
                f"Could not carry context {context_name!r} forward from version "
                f"{plan.carry_forward_from} into {to_version}. Publishing now would "
                f"remove that context's model. Re-run after restoring the previous "
                f"artifact set."
            )
        carried.append(context_name)
    return carried


def _training_provenance_path(
    workflow_folderpath: str, version_id: Optional[str] = None
) -> Path:
    if version_id is not None:
        return (
            artifact_versioning.version_dir(workflow_folderpath, version_id)
            / determinism.PROVENANCE_FILENAME
        )
    return (
        Path(workflow_folderpath)
        / determinism.COMMAND_INFO_FOLDERNAME
        / determinism.PROVENANCE_FILENAME
    )


def capture_training_provenance(
    workflow_folderpath: str, version_id: Optional[str] = None
) -> Optional[dict]:
    """Snapshot provenance BEFORE a selective ``train()`` overwrites it.

    The returned object is the raw JSON envelope so records unknown to this version
    survive a merge unchanged. ``None`` means the file was absent, unreadable, or not
    a JSON object. When ``version_id`` is supplied, the snapshot comes from that
    immutable artifact version rather than the mutable top-level reporting copy; a
    selective trainer should pass ``plan.carry_forward_from``.
    """
    try:
        path = _training_provenance_path(workflow_folderpath, version_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _v2_provenance_sections(payload: dict) -> Optional[tuple[dict, dict]]:
    """Return validated command/context maps from a schema-v2 envelope."""
    if payload.get("schema_version") != determinism.PROVENANCE_SCHEMA_VERSION:
        return None
    commands = payload.get("commands")
    context_training = payload.get("context_training")
    if not isinstance(commands, dict) or not isinstance(context_training, dict):
        return None
    if any(
        not isinstance(command_name, str) or not isinstance(record, dict)
        for command_name, record in commands.items()
    ):
        return None
    for context_name, command_map in context_training.items():
        if not isinstance(context_name, str) or not isinstance(command_map, dict):
            return None
        if any(
            not isinstance(command_name, str) or not isinstance(record, dict)
            for command_name, record in command_map.items()
        ):
            return None
    return commands, context_training


def merge_training_provenance(
    workflow_folderpath: str, plan: TrainingPlan, previous: Optional[dict]
) -> bool:
    """Merge carried contexts from prior provenance into the fresh v2 envelope.

    Only ``plan.contexts_carried_forward`` contributes old context records. An old
    command-generation record is copied only when that command belongs solely to
    carried contexts; if it also appears in a retrained context, the fresh run must
    supply it and an older record is never allowed to hide its absence. Existing
    current command and context records always win.

    Returns ``True`` after writing a merged envelope. Returns ``False`` without
    changing the fresh file for a full retrain, no carried contexts, absent prior
    provenance, a non-v2/invalid envelope, or an I/O failure.
    """
    if plan.is_full_retrain or not plan.contexts_carried_forward or not previous:
        return False

    path = _training_provenance_path(workflow_folderpath)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            f"Could not merge carried-forward training provenance into {path} "
            f"({exc}); the fresh provenance was left unchanged."
        )
        return False
    if not isinstance(current, dict):
        logger.warning(
            f"Could not merge carried-forward training provenance into {path}: "
            "the fresh provenance is not a JSON object."
        )
        return False

    previous_sections = _v2_provenance_sections(previous)
    current_sections = _v2_provenance_sections(current)
    if previous_sections is None or current_sections is None:
        logger.warning(
            f"Could not merge carried-forward training provenance into {path}: "
            "both the previous and fresh provenance must use valid schema-v2 "
            "envelopes. The fresh provenance was left unchanged."
        )
        return False
    previous_commands, previous_contexts = previous_sections
    current_commands, current_contexts = current_sections

    carried_contexts = set(plan.contexts_carried_forward)
    if missing_contexts := sorted(carried_contexts - set(previous_contexts)):
        logger.warning(
            f"Could not merge carried-forward training provenance into {path}: "
            "the previous provenance has no context-training records for "
            f"{', '.join(missing_contexts)}. The fresh provenance was left unchanged."
        )
        return False

    retrained_commands: set[str] = set()
    for context_name in plan.contexts_to_train:
        retrained_commands.update(previous_contexts.get(context_name, {}))
        retrained_commands.update(current_contexts.get(context_name, {}))

    carried_commands: set[str] = set()
    for context_name in sorted(carried_contexts):
        previous_command_map = previous_contexts[context_name]
        carried_commands.update(previous_command_map)
        current_command_map = current_contexts.setdefault(context_name, {})
        for command_name, record in previous_command_map.items():
            current_command_map.setdefault(command_name, record)

    for command_name in sorted(carried_commands - retrained_commands):
        previous_record = previous_commands.get(command_name)
        if previous_record is not None:
            current_commands.setdefault(command_name, previous_record)

    current["commands"] = {
        command_name: current_commands[command_name]
        for command_name in sorted(current_commands)
    }
    current["context_training"] = {
        context_name: {
            command_name: current_contexts[context_name][command_name]
            for command_name in sorted(current_contexts[context_name])
        }
        for context_name in sorted(current_contexts)
    }

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        logger.warning(
            f"Could not write merged carried-forward training provenance to {path} "
            f"({exc}); the fresh provenance was left unchanged."
        )
        return False
    return True


def _heldout_path(workflow_folderpath: str) -> Path:
    return Path(
        CommandDirectory.get_commandinfo_folderpath(workflow_folderpath)
    ) / "heldout_evaluation.json"


def capture_heldout_evaluation(workflow_folderpath: str) -> Optional[dict]:
    """Snapshot the held-out evaluation report BEFORE ``train()`` overwrites it.

    ``train()`` rewrites ``heldout_evaluation.json`` wholesale from the contexts it
    just trained, so a selective run narrows a four-context report to a one-context
    report. That does not make any model stale, but it destroys the evidence a
    developer uses to decide whether the workflow routes well, and it does so
    silently -- the file still looks complete, it just describes less of the
    workflow than it did yesterday.
    """
    path = _heldout_path(workflow_folderpath)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def merge_heldout_evaluation(
    workflow_folderpath: str, plan: TrainingPlan, previous: Optional[dict]
) -> bool:
    """Re-insert carried-forward contexts into the freshly written held-out report.

    Each re-inserted entry is tagged ``carried_forward`` with the version it came
    from, because the numbers were measured against models that were trained
    earlier -- they are still the numbers for the models this version ships, but a
    reader deserves to know they were not measured on this run.

    Returns False and leaves the fresh report untouched if either report has an
    unexpected shape. Merging is a reporting nicety and the fresh report is never
    *wrong*, only narrow, so an unrecognised schema must not fail a training run.
    """
    if plan.is_full_retrain or not plan.contexts_carried_forward or not previous:
        return False

    path = _heldout_path(workflow_folderpath)
    try:
        current = json.loads(path.read_text())
        previous_schema = previous["schema_version"]
        current_schema = current["schema_version"]
        if previous_schema != current_schema:
            logger.warning(
                f"Could not merge carried-forward contexts into {path}: held-out "
                f"report schema changed from {previous_schema} to {current_schema}. "
                "The report describes only the contexts this run retrained."
            )
            return False
        previous_entries = {
            entry["context"]: entry for entry in previous["contexts"]}
        merged = list(current["contexts"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            f"Could not merge carried-forward contexts into {path} ({exc}). The "
            f"report describes only the contexts this run retrained."
        )
        return False

    # A run that retrained NOTHING leaves the previous report in place untouched, so
    # every carried-forward context is already there. Re-inserting would duplicate
    # each context and double-count it in the totals.
    already_present = {
        entry.get("context") for entry in merged if isinstance(entry, dict)}

    for context_name in plan.contexts_carried_forward:
        entry = previous_entries.get(context_name)
        if entry is None or context_name in already_present:
            continue
        entry = dict(entry)
        entry["carried_forward"] = True
        entry["carried_forward_from"] = plan.carry_forward_from
        merged.append(entry)

    merged.sort(key=lambda entry: entry.get("context") or "")
    current["contexts"] = merged
    try:
        current["totals"] = _recompute_heldout_totals(merged)
    except (KeyError, TypeError, ValidationError) as exc:
        # ValidationError joins the set now that the totals are recomputed by
        # validating entries into HeldoutReport: a malformed entry must still
        # degrade to stale totals with a warning, not abort the merge.
        logger.warning(
            f"Merged carried-forward contexts into {path} but could not recompute "
            f"the totals ({exc}); they describe only the retrained contexts."
        )
    current.setdefault("metric_notes", {})["selective_training"] = (
        "This run automatically reused unchanged contexts. Entries marked "
        "'carried_forward' were measured on an earlier run against the same model "
        "artifacts this version ships."
    )
    path.write_text(json.dumps(current, indent=2))
    return True


def _commands_contributing_to(
    context_name: str,
    context_commands: dict[str, set[str]],
    context_ancestors: dict[str, set[str]],
) -> set[str]:
    """Every command whose utterances end up in *context_name*'s training data.

    Its own commands, plus its ancestors' -- an ancestor's utterances constitute
    this context's wildcard class, which is why a change upstream is a change here.
    """
    contributing = set(context_commands.get(context_name) or ())
    for ancestor in context_ancestors.get(context_name) or ():
        contributing |= set(context_commands.get(ancestor) or ())
    return contributing


def _shared_commands_absent_from_cache(
    workflow_folderpath: str,
    to_train: Iterable[str],
    carried: Iterable[str],
    context_commands: dict[str, set[str]],
    context_ancestors: dict[str, set[str]],
) -> set[str]:
    """Commands feeding BOTH a retrained and a carried-forward context, with no cache.

    Absence of any variant file for a command is a sound proxy for "this will be
    regenerated": a hit requires a file. The converse does not hold -- a file whose
    variant key no longer matches still misses -- but that case is a fingerprint
    change, which the signature diff already turns into a retrain. This check exists
    for the case the signature diff cannot see, where nothing about the inputs
    changed and the cached bytes simply are not there.
    """
    shared: set[str] = set()
    retrained_side: set[str] = set()
    for context_name in to_train:
        retrained_side |= _commands_contributing_to(
            context_name, context_commands, context_ancestors)
    for context_name in carried:
        shared |= retrained_side & _commands_contributing_to(
            context_name, context_commands, context_ancestors)

    if not shared:
        return set()

    cache_root = os.path.join(
        workflow_folderpath, determinism.COMMAND_INFO_FOLDERNAME,
        utterance_cache.CACHE_DIRNAME,
    )
    if not os.path.isdir(cache_root):
        return shared

    cached_stems = {
        name.split(".", 1)[0]
        for name in os.listdir(cache_root)
        if name.endswith(".json")
    }
    return {
        command for command in shared
        if utterance_cache.slugify(command) not in cached_stems
    }


def _recompute_heldout_totals(entries: list[dict]) -> dict:
    """Re-aggregate the report totals over a merged context list.

    Every total in the report is a plain sum or a ratio of two sums, so merging is
    exact -- there is no approximation here, and a merged report is numerically what
    a full retrain would have produced given the same per-context results.

    Delegates to ``aggregate_totals`` rather than re-summing, because this used to
    be a parallel implementation and drifted from it: it dropped keys the fresh
    path always emits and turned a missing mean into a literal 0.0, so a merged
    report and a full-retrain report had different schemas and a diff between them
    read as a collapse in quality. Sharing the function makes that class of
    divergence unrepresentable.
    """
    reports = [
        heldout_evaluation.HeldoutReport.model_validate(entry) for entry in entries
    ]
    return heldout_evaluation.aggregate_totals(reports)


def format_plan(plan: TrainingPlan) -> str:
    """Render the plan so the developer can see the closure that was computed.

    R5 requires the output to state which contexts were retrained and which were
    carried forward. Showing the reasons is what lets a developer check the closure
    rather than trust it.
    """
    lines: list[str] = []
    if plan.is_full_retrain:
        lines.append(f"Training plan: full retrain of {len(plan.contexts_to_train)} context(s).")
        lines.extend(f"  {reason}" for reason in plan.global_reasons)
        return "\n".join(lines)

    lines.append("Training plan (selective):")
    if plan.carry_forward_from:
        lines.append(f"  carrying forward from version {plan.carry_forward_from}")
    if plan.dirty_commands:
        lines.append(f"  changed commands ({len(plan.dirty_commands)}):")
        lines.extend(f"    - {name}" for name in plan.dirty_commands)
    else:
        lines.append("  changed commands: none")

    lines.append(f"  retraining {len(plan.contexts_to_train)} context(s):")
    for context_name in plan.contexts_to_train:
        lines.append(f"    - {context_name}")
        lines.extend(f"        {reason}" for reason in plan.reasons.get(context_name, []))

    lines.append(
        f"  carrying forward {len(plan.contexts_carried_forward)} context(s): "
        + (", ".join(plan.contexts_carried_forward) or "none")
    )
    return "\n".join(lines)
