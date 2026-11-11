"""Selective retraining: deciding which contexts must be retrained when a command changes.

Retraining a single command is **not local**. "Only this command changed" is never
only one context, because two independent inheritance axes decide the blast radius:

``base`` — command inheritance, declared in ``_commands/context_inheritance_model.json``.
    ``CommandContextModel.commands(ctx)`` already merges every base context's commands
    into ``ctx``'s effective label space, so a changed command appears directly in the
    label space of every context that inherits it.

The parent axis (``context_hierarchy_model.json``) is retired (Phase 9 step 5). A head's
``wildcard`` class is the sparse "not here" class: one recorded sample of every foreign
command (design 2.50). So the command axis is the only one left, and it works like this:

* adding or removing a command dirties every head, because every head's foreign set changes;
* editing a command's files dirties the heads that LABEL it (its own heads), since a foreign
  head reads only the command's recorded sample;
* a head whose recorded sample for a foreign command no longer exists is dirty, and only that
  head, because it chooses that command's sample again.

Why the closure alone is not enough
-----------------------------------
The command-level closure above answers "which contexts are affected by a command
whose *file* changed". It does not answer "is this context's model still the model
this workflow would produce today", and those are different questions. Three change
classes move a context's label space without any command file changing:

* an edit to ``_commands/context_inheritance_model.json`` (``base``) changes which
  commands a context carries, and therefore which commands reach the *wildcard*
  classes of that context's descendants -- with no command file touched anywhere;
* an edit to a leftover ``context_hierarchy_model.json`` changes nothing here: the file
  is ignored (see ``CommandContextModel.load``);
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
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel, ValidationError

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.nlu_labels import (
    NON_ROUTABLE_LABELS,
    PARAMETER_VALUE_PLACEHOLDERS,
    WILDCARD_LABEL,
)
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
#
# 2: the not-here samples are recorded per head (design 2.50). A version written under 1 records
# no samples, so the first train after the change chooses and records them: one full retrain.
ARTIFACT_FORMAT_VERSION = 2

# Full-module digests are intentionally conservative. A helper or comments-only edit
# may trigger a full retrain; missing a helper edit could silently carry stale models.
TRAINER_SOURCE_MODULE = "fastworkflow.model_pipeline_training"
GENERATOR_SOURCE_MODULE = "fastworkflow.train.generate_synthetic"

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

    ``wildcard_sources`` describes the *escalation* class: under the sparse construction, every
    command outside this head (its seeds are the class's rows). The sources are recorded as
    names, and the fingerprints of those commands are compared elsewhere in the signature.

    ``expects_wildcard_label`` records whether ``WILDCARD_LABEL`` is expected to be a class at
    all: it is when there is any foreign command to sample. It is recorded for legibility
    rather than for safety, since the sources it is derived from are compared exactly.
    """

    context_name: str
    label_space: list[str] = []
    wildcard_sources: list[str] = []
    expects_wildcard_label: bool = False
    #: The reserved-class arm this context was trained under, always resolved (a shipped run
    #: records ``SHIPPED_ARM``, sparse_1). Switching arms changes it, which marks every head dirty.
    reserved_class_arm: Optional[str] = None
    #: The EXPERIMENT head layout ("flattened" or "scoped"; None for a shipped run). The two
    #: layouts train different label spaces and holdouts, so their artifacts never mix.
    head_scope: Optional[str] = None
    #: The sparse class's sample for each foreign command, ``{command: [seed text]}`` (design
    #: 2.50), chosen once and reused while the seed text still exists. None when this head
    #: records no samples (the internal CME) or the signature predates the recording.
    not_here_samples: Optional[dict[str, list[str]]] = None


def resolved_arm(reserved_class_arm: Optional[str]) -> str:
    """The arm the planner trains under: the explicit one, else the experiment override, else sparse_1."""
    # Lazy: model_pipeline_training imports torch, and the planner does not need it otherwise.
    from fastworkflow.model_pipeline_training import resolve_reserved_class_arm

    return resolve_reserved_class_arm(reserved_class_arm)


def sparse_arm(reserved_class_arm: Optional[str]) -> bool:
    """True when the arm in force builds the sparse "not here" class (Phase 9 step 1, step 4).

    ``None`` is the shipped construction, which is sparse, so a shipped run is sparse too. The
    internal CME's construction is ``"none"``, which is not sparse.
    """
    from fastworkflow.model_pipeline_training import arm_construction

    return arm_construction(resolved_arm(reserved_class_arm)) == "not_here"


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
    parameter_value_placeholders_sha256: Optional[str] = None
    #: The not-here construction record (``model_pipeline_training.not_here_construction``).
    #: A baseline without it, which is every artifact trained before Phase 9 step 4, differs
    #: from any current value, so the first train under the sparse construction retrains every
    #: head and never carries an old head forward.
    not_here: Optional[dict] = None
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
            "parameter_value_placeholders_sha256":
                self.parameter_value_placeholders_sha256,
            "not_here": self.not_here,
        }


class TrainingPlan(BaseModel):
    """The computed retraining closure, in a form a developer can audit.

    ``reasons`` is the point of this object. The developer must be able to see *why*
    each context was pulled in -- a changed command dirties every head, and each reason says so.

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
    for command_name in cmd_dir.get_execution_commands():
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
    *,
    structural_commands: Iterable[str] = (),
) -> dict[str, list[str]]:
    """The heads a set of changed commands dirties (Phase 9 step 6, design 2.50 and 3.9).

    A command that was added or removed (``structural_commands``) changes every head's foreign
    set, so it dirties every head. Any other changed command (its source or its seeds) dirties
    only the heads whose own label space holds it: a foreign head reads a command's recorded
    sample, and whether that sample moved is decided by the head's own signature diff, not here.
    Pure over plain data so the rule can be tested without building a workflow.
    ``context_commands`` must already have ``base`` inheritance merged in (which is what
    ``CommandContextModel.commands`` returns).

    Returns context name -> the list of reasons it must be retrained.
    """
    reasons: dict[str, list[str]] = {}
    if not dirty_commands:
        return reasons

    structural = sorted(set(structural_commands) & dirty_commands)
    if structural:
        changed = ", ".join(structural)
        for context_name in context_commands:
            reasons.setdefault(context_name, []).append(
                f"a command was added or removed ({changed}); every head is retrained "
                f"because every head's foreign set changed"
            )
        return reasons

    for context_name, own_commands in context_commands.items():
        if hit := sorted(own_commands & dirty_commands):
            reasons.setdefault(context_name, []).append(
                f"label space contains changed command(s): {', '.join(hit)}"
            )
    return reasons


def build_context_maps(
    workflow_folderpath: str,
    context_names: Iterable[str],
    head_scope: Optional[str] = None,
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Build the plain-data maps the closure and the signature consume.

    Returns ``(context_commands, unresolved)`` where ``unresolved`` maps a context name to the
    reasons its model could not be described. A context whose command list cannot be resolved
    falls back to ``crd.contexts`` and is reported, so the caller can force it to retrain.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    context_model = crd.context_model
    core_commands = set(crd.command_directory.core_command_names)
    scoped = head_scope == "scoped"
    if scoped:
        # Lazy: model_pipeline_training imports torch, and this module must not.
        from fastworkflow.model_pipeline_training import own_command_names

    context_commands: dict[str, set[str]] = {}
    unresolved: dict[str, list[str]] = {}

    # Lazy: model_pipeline_training imports torch, and this module must not.
    from fastworkflow.model_pipeline_training import planning_labels_in_layout

    for context_name in context_names:
        if scoped:
            # EXPERIMENT (Phase 9 step 3): under the scoped layout a head's label space is its
            # own "/" list, so the signature records exactly what the head trains.
            effective = own_command_names(crd, context_name)
        else:
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
        # Phase 10 step 1: under the production layout a planning command is a label of the heads
        # on its lookup path. The context model never lists them, so they are added here, where
        # the signature reads the label space.
        context_commands[context_name] = (
            effective | core_commands | planning_labels_in_layout(context_name, crd))

    return context_commands, unresolved


def contexts_for_training(workflow_folderpath: str) -> set[str]:
    """The contexts ``train()`` produces artifacts for, for *workflow_folderpath*.

    This lives here rather than inline in ``train()`` so that the planner and the
    trainer cannot disagree about the candidate set. They must not: a context the
    planner never considered is a context that is neither retrained nor carried
    forward, and ``publish_version`` removes the compatibility entry of any context
    a version does not contain -- so the disagreement would present as part of a
    workflow silently becoming untrained.

    The internal ``command_metadata_extraction`` and ``consultation`` contexts are
    excluded from an app workflow because their models live in those internal
    workflows, and the global wildcard context ``"*"`` is always included.
    """
    crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    consultation_path = fastworkflow.get_internal_workflow_path("consultation")
    internal_contexts: set[str] = set()
    for internal_path in (cme_path, consultation_path):
        if os.path.isdir(internal_path):
            internal_contexts |= set(
                fastworkflow.CommandContextModel.load(internal_path)._command_contexts.keys()
            )

    def _targets_internal_workflow(name: str, internal_path: str) -> bool:
        real = os.path.realpath(workflow_folderpath)
        resolved = os.path.realpath(internal_path)
        marker = os.path.join("_workflows", name)
        return (
            real == resolved
            or real.startswith(resolved + os.sep)
            or marker in workflow_folderpath
        )

    if _targets_internal_workflow("command_metadata_extraction", cme_path):
        cme_contexts = set(
            fastworkflow.CommandContextModel.load(cme_path)._command_contexts.keys()
        )
        # `ErrorCorrection` is excluded because the internal workflow declares it as a
        # context but trains no classifier for it.
        return set(cme_contexts) - {'ErrorCorrection'}
    if _targets_internal_workflow("consultation", consultation_path):
        return set(
            fastworkflow.CommandContextModel.load(consultation_path)._command_contexts.keys()
        )
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
    context_names: Iterable[str],
    reserved_class_arm: Optional[str] = None,
    universe: Optional[set[str]] = None,
    head_scope: Optional[str] = None,
    not_here_samples_by_context: Optional[dict[str, Optional[dict[str, list[str]]]]] = None,
) -> dict[str, ContextSignature]:
    """Describe each context's label space and wildcard-class inputs.

    Under a sparse arm the wildcard class samples every command outside the head, so its
    sources are *universe* minus the head's own labels. The internal CME (construction
    ``"none"``) has no wildcard class, so its sources are empty. ``not_here_samples_by_context``
    gives each head's recorded samples; a head not named there records none.
    """
    reserved_class_arm = resolved_arm(reserved_class_arm)
    sparse = sparse_arm(reserved_class_arm)
    if sparse and universe is None:
        universe = set().union(*context_commands.values()) if context_commands else set()
    signatures: dict[str, ContextSignature] = {}
    for context_name in context_names:
        own = context_commands.get(context_name, set())

        wildcard_sources: set[str] = set()
        if sparse:
            wildcard_sources = {
                command for command in (universe or set())
                if command not in own and command.split("/")[-1] != WILDCARD_LABEL
            }

        signatures[context_name] = ContextSignature(
            context_name=context_name,
            label_space=sorted(own),
            wildcard_sources=sorted(wildcard_sources),
            expects_wildcard_label=bool(wildcard_sources - own),
            reserved_class_arm=reserved_class_arm,
            head_scope=head_scope,
            not_here_samples=(
                (not_here_samples_by_context or {}).get(context_name)
                if sparse else None
            ),
        )
    return signatures


def compute_planning_fingerprints(workflow_folderpath: str) -> dict[str, CommandFingerprint]:
    """Fingerprints of the planning commands, which the sparse class samples.

    Keyed by a file's bytes, not by its parsed seeds: the parsed form is cached per process,
    so a seed edit made in this process would otherwise be invisible to the planner. The
    file hash also moves on a docstring edit, which is the conservative direction.
    """
    from fastworkflow.workflow_agent import _planning_command_files

    fingerprints: dict[str, CommandFingerprint] = {}
    root = os.path.join(workflow_folderpath, "_commands")
    for name, path in _planning_command_files(root):
        digest = _sha256_file(path)
        if digest is None:
            fingerprints[name] = _unresolvable(name, f"planning command file unreadable: {path}")
        else:
            fingerprints[name] = CommandFingerprint(
                command_name=name, source_path=path, source_sha256=digest,
                seed_utterances_sha256=None)
    return fingerprints


def compute_training_signature(
    workflow_folderpath: str,
    context_names: Iterable[str],
    seed: Optional[int] = None,
    reserved_class_arm: Optional[str] = None,
    head_scope: Optional[str] = None,
    recorded_samples: Optional[Mapping[str, Optional[Mapping[str, list[str]]]]] = None,
) -> tuple[TrainingSignature, dict[str, list[str]]]:
    """Build the signature this workflow would train under right now.

    ``recorded_samples`` is each head's not-here samples from the version being carried from
    (design 2.50). A recorded sample is kept while its seed text exists; ``None`` chooses every
    sample afresh (a first train, or ``--regenerate-utterances``).

    Returns ``(signature, unresolved)``; ``unresolved`` names contexts the context
    model could not describe, which the caller must retrain regardless of the diff.
    """
    contexts = sorted(context_names)
    context_commands, unresolved = build_context_maps(
        workflow_folderpath, contexts, head_scope=head_scope)

    from fastworkflow.model_pipeline_training import not_here_construction

    from fastworkflow.model_pipeline_training import arm_for_workflow

    reserved_class_arm = arm_for_workflow(workflow_folderpath, reserved_class_arm)
    command_fingerprints = compute_command_fingerprints(workflow_folderpath)
    universe: Optional[set[str]] = None
    not_here_samples_by_context: Optional[dict[str, Optional[dict[str, list[str]]]]] = None
    if sparse_arm(reserved_class_arm):
        planning = compute_planning_fingerprints(workflow_folderpath)
        command_fingerprints = {**command_fingerprints, **planning}
        universe = set().union(*context_commands.values()) | set(planning)
        # After the fingerprints, which hydrate every command the samples are read from.
        from fastworkflow.model_pipeline_training import not_here_samples_for_context

        crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
        not_here_samples_by_context = {
            context_name: not_here_samples_for_context(
                workflow_folderpath,
                crd,
                context_name,
                reserved_class_arm,
                recorded=(recorded_samples or {}).get(context_name),
            )
            for context_name in contexts
        }

    signature = TrainingSignature(
        format_version=ARTIFACT_FORMAT_VERSION,
        not_here=not_here_construction(reserved_class_arm, workflow_folderpath),
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
        # The bare-value literals are a code constant that becomes a real class in
        # EVERY context (`PARAMETER_VALUE_LABEL`). Editing the list changes every
        # context's training data without touching a single workflow file.
        parameter_value_placeholders_sha256=_sha256_strings(
            PARAMETER_VALUE_PLACEHOLDERS),
        command_fingerprints=command_fingerprints,
        contexts=compute_context_signatures(
            context_commands, contexts,
            reserved_class_arm=reserved_class_arm, universe=universe,
            head_scope=head_scope,
            not_here_samples_by_context=not_here_samples_by_context),
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


def command_utterance_fingerprint(fingerprint: CommandFingerprint) -> str:
    """Digest one command's training inputs for the version manifest.

    Collapses the two hashes ``CommandFingerprint`` keeps apart -- the file and the
    declared seed list -- into the single value AR3's per-context utterance-set
    fingerprints are hashed from. They are kept apart in the signature so a prose edit
    can be told from a data change *when deciding what to retrain*; a context's
    contribution is a set of training inputs, and either hash moving moves it.

    An unresolved fingerprint records a fixed marker rather than a fresh uuid, because
    the manifest's digests must be reproducible from the manifest. Its "this can never
    prove cleanliness" property is enforced at promotion instead, by refusing to carry
    a context forward when any command feeding it carries the marker.
    """
    if not fingerprint.resolved:
        return artifact_versioning.UNRESOLVED_FINGERPRINT
    payload = (
        f"{fingerprint.source_sha256 or ''}\x00"
        f"{fingerprint.seed_utterances_sha256 or ''}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def context_contribution_manifest_fields(
    signature: TrainingSignature, plan: TrainingPlan
) -> dict:
    """Record what AR3 needs to re-derive this run's closure from the version alone.

    ``ContextSignature`` names each context's wildcard sources, but only as command NAMES,
    and a name does not move when the command's file or seed list does. A source command
    whose *content* changed would then be invisible to every comparison built on those
    lists. Hashing each context's contribution over the per-command fingerprints closes
    that gap.

    Written into the manifest rather than beside ``training_signature.json`` because
    ``artifact_versioning`` is what runs the check, must not import this module (the
    dependency runs the other way), and must not import the routing registry at all --
    a postcondition that re-consults the planner is not independent of it.
    """
    fingerprints = {
        name: command_utterance_fingerprint(fingerprint)
        for name, fingerprint in sorted(signature.command_fingerprints.items())
    }
    carried = set(plan.contexts_carried_forward)

    def _arm_fields(context_signature: ContextSignature) -> dict:
        if context_signature.reserved_class_arm is None:
            return {}  # shipped runs: the manifest is unchanged
        from fastworkflow.model_pipeline_training import arm_construction

        not_here = sparse_arm(context_signature.reserved_class_arm)
        own_union = set()
        for other in signature.contexts.values():
            own_union |= set(other.label_space)
        planning_sources = sorted(
            set(context_signature.wildcard_sources) - own_union) if not_here else []
        fields = {
            "reserved_class_arm": context_signature.reserved_class_arm,
            "reserved_class_construction": arm_construction(
                resolved_arm(context_signature.reserved_class_arm)),
            "not_here_planning_sources": planning_sources,
        }
        if context_signature.not_here_samples is not None:
            fields[artifact_versioning.NOT_HERE_SAMPLES_KEY] = dict(
                sorted(context_signature.not_here_samples.items()))
        if context_signature.head_scope is not None:
            fields["head_scope"] = context_signature.head_scope
        return fields

    def _foreign_digest(context_signature: ContextSignature) -> str:
        # A sparse head reads one recorded sample per foreign command, so its digest is over those
        # samples: a foreign command whose other seeds changed does not move it (design 2.50).
        if context_signature.not_here_samples is not None:
            return artifact_versioning.not_here_samples_fingerprint(
                context_signature.not_here_samples)
        return artifact_versioning.utterance_set_fingerprint(
            fingerprints, context_signature.wildcard_sources
        )

    contributions = {
        context_name: {
            **_arm_fields(context_signature),
            # Always empty: no sparse or internal construction samples an ancestor. The key stays
            # so the promotion check (artifact_versioning) reads the same shape it always has.
            "ancestors": [],
            "label_space": sorted(context_signature.label_space),
            "wildcard_sources": sorted(context_signature.wildcard_sources),
            "expects_wildcard_label": context_signature.expects_wildcard_label,
            "own_utterances_sha256": artifact_versioning.utterance_set_fingerprint(
                fingerprints, context_signature.label_space
            ),
            "ancestor_utterances_sha256": _foreign_digest(context_signature),
            "carried_forward": context_name in carried,
        }
        for context_name, context_signature in sorted(signature.contexts.items())
    }

    fields = {
        artifact_versioning.CONTRIBUTION_FORMAT_KEY: (
            artifact_versioning.CONTRIBUTION_FORMAT_VERSION
        ),
        artifact_versioning.COMMAND_UTTERANCE_FINGERPRINTS_KEY: fingerprints,
        artifact_versioning.CONTEXT_CONTRIBUTIONS_KEY: contributions,
    }
    if plan.carry_forward_from:
        fields[artifact_versioning.CARRY_FORWARD_FROM_KEY] = plan.carry_forward_from
    return fields


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
    if previous.reserved_class_arm != current.reserved_class_arm:
        differences.append(
            f"reserved-class arm changed: {previous.reserved_class_arm} -> "
            f"{current.reserved_class_arm}")
    # None is today's layout, which is what "flattened" means, so the two compare equal.
    if (previous.head_scope or "flattened") != (current.head_scope or "flattened"):
        differences.append(
            f"head layout changed: {previous.head_scope or 'flattened'} -> "
            f"{current.head_scope or 'flattened'}")
    if previous.label_space != current.label_space:
        added = sorted(set(current.label_space) - set(previous.label_space))
        removed = sorted(set(previous.label_space) - set(current.label_space))
        differences.append(
            "label space changed"
            + (f"; added {', '.join(added)}" if added else "")
            + (f"; removed {', '.join(removed)}" if removed else "")
        )
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
    if previous.not_here_samples != current.not_here_samples:
        if previous.not_here_samples is None:
            differences.append(
                "no recorded not-here samples in the version being carried from")
        else:
            previous_samples = previous.not_here_samples
            current_samples = current.not_here_samples or {}
            moved = sorted(
                name for name in set(previous_samples) | set(current_samples)
                if previous_samples.get(name) != current_samples.get(name)
            )
            differences.append(
                "not-here samples changed for: " + ", ".join(moved))
    return differences


def _full_retrain(candidates: list[str], reason: str) -> TrainingPlan:
    return TrainingPlan(
        contexts_to_train=candidates,
        is_full_retrain=True,
        global_reasons=[reason],
        reasons={ctx: [reason] for ctx in candidates},
    )


def dirty_heads_from_signatures(
    previous: TrainingSignature,
    current: TrainingSignature,
    context_commands: dict[str, set[str]],
    *,
    only_commands: Iterable[str] = (),
) -> tuple[set[str], dict[str, list[str]]]:
    """The changed commands and every head's reasons to retrain, from two signatures alone.

    This is the planner's per-head decision, pure over the two signatures and the label maps, so a
    test can drive it on hand-built signatures. It has three parts:

    * the command closure: a changed or added or removed command dirties the heads its change
      reaches (``close_dirty_contexts``);
    * the sample diff: a head whose recorded not-here sample for some foreign command moved, or
      was never recorded, is dirty (``_diff_context_signature``); and
    * the context-model diff, which catches a label-space change with no command file touched.

    Returns ``(dirty command names, context name -> reasons)``.
    """
    dirty: set[str] = set(only_commands) | changed_commands(
        previous.command_fingerprints,
        current.command_fingerprints,
    )
    structural = set(previous.command_fingerprints) ^ set(current.command_fingerprints)
    reasons = close_dirty_contexts(dirty, context_commands, structural_commands=structural)
    for context_name in sorted(current.contexts):
        for difference in _diff_context_signature(
            previous.contexts.get(context_name),
            current.contexts[context_name],
        ):
            reasons.setdefault(context_name, []).append(difference)
    return dirty, reasons


def compute_training_plan(
    workflow_folderpath: str,
    candidate_contexts: Iterable[str],
    only_commands: Optional[Iterable[str]] = None,
    only_contexts: Optional[Iterable[str]] = None,
    changed_only: bool = True,
    seed: Optional[int] = None,
    carry_forward_from: Optional[str] = None,
    cache_mode: Optional[str] = None,
    reserved_class_arm: Optional[str] = None,
    head_scope: Optional[str] = None,
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
    from fastworkflow.model_pipeline_training import arm_for_workflow

    candidates = sorted(candidate_contexts)
    # The internal CME keeps its control construction whatever the caller passes (see arm_for_workflow).
    reserved_class_arm = arm_for_workflow(workflow_folderpath, reserved_class_arm)
    # The utterance-cache mode decides whether recorded samples may be reused (design 2.50):
    # --regenerate-utterances chooses every sample again.
    cache_mode = cache_mode or _resolve_cache_mode()

    previous_signature = load_training_signature(
        workflow_folderpath, carry_forward_from)
    recorded_samples = None
    if previous_signature is not None and cache_mode == _MODE_REUSE:
        recorded_samples = {
            context_name: context_signature.not_here_samples
            for context_name, context_signature in previous_signature.contexts.items()
        }

    current_signature, unresolved = compute_training_signature(
        workflow_folderpath, candidates, seed=seed,
        reserved_class_arm=reserved_class_arm, head_scope=head_scope,
        recorded_samples=recorded_samples)

    if not (only_commands or only_contexts or changed_only):
        return (
            _full_retrain(candidates, "full retrain (no selector given)"),
            current_signature,
        )

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
    # then be built on fresh text while a carried-forward context keeps the old text.
    # That is exactly the internally inconsistent version AR3 warns about, so the two
    # features are mutually exclusive rather than merely discouraged.
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

    context_commands, _ = build_context_maps(workflow_folderpath, candidates)
    dirty, reasons = dirty_heads_from_signatures(
        previous_signature, current_signature, context_commands,
        only_commands=only_commands or (),
    )

    # EXPERIMENT (Phase 9 step 3). Under either head layout the persona holdout is ONE split over
    # every command, so one changed command moves the training rows of every head, not only the
    # heads that own it. Carrying the others forward would keep models trained on text that the
    # current split no longer holds out. Retrain everything instead.
    if head_scope is not None and dirty:
        return (
            _full_retrain(
                candidates,
                f"full retrain (head layout {head_scope}: the persona holdout spans every "
                f"command, so a changed command moves every head's training rows)"),
            current_signature,
        )

    # An explicitly named context is retrained. Nothing else is: no head's wildcard class
    # is built from another head's utterances any more, so no descendant is pulled in.
    for context_name in only_contexts or []:
        reasons.setdefault(context_name, []).append("explicitly requested")

    for context_name in candidates:
        # A context the context model could not describe is a context whose closure
        # was computed on incomplete information.
        for reason in unresolved.get(context_name, []):
            reasons.setdefault(context_name, []).append(reason)

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
            workflow_folderpath, to_train, carried, context_commands
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
) -> set[str]:
    """Every command whose GENERATED utterances end up in *context_name*'s training data.

    Its own label space. The sparse wildcard class draws hand-written seeds of foreign
    commands, not cached generated text, so it adds nothing here.
    """
    return set(context_commands.get(context_name) or ())


def _generation_eligible_commands(workflow_folderpath: str) -> set[str]:
    """Commands synthetic generation would actually produce utterances for.

    The same two sources generation itself draws from, so a command that is never
    generated cannot be mistaken for one whose cache went missing. Returns the empty
    set if the directory cannot be read, which makes the caller's intersection empty
    and the guard silent -- an unreadable command directory is a problem for the
    planner to report, not for this check to turn into a full retrain.
    """
    try:
        cmd_dir = CommandDirectory.load(workflow_folderpath)
        names = set(cmd_dir.get_execution_commands()) | set(cmd_dir.core_command_names)
    except Exception:
        return set()

    # Reserved NLU labels appear among the core command names but generation skips
    # them, so they can never have a cache entry. `wildcard` is the one that bites:
    # it is in every workflow's core set, so leaving it in made this guard fire on
    # every selective run everywhere.
    return {
        name for name in names
        if name.rsplit("/", 1)[-1] not in NON_ROUTABLE_LABELS
    }


def _shared_commands_absent_from_cache(
    workflow_folderpath: str,
    to_train: Iterable[str],
    carried: Iterable[str],
    context_commands: dict[str, set[str]],
    generation_eligible: Optional[set[str]] = None,
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
        retrained_side |= _commands_contributing_to(context_name, context_commands)
    for context_name in carried:
        shared |= retrained_side & _commands_contributing_to(context_name, context_commands)

    # "No cache entry" only means "will be regenerated" for a command that has
    # something to generate. A command with no seed utterances -- set_root_context
    # in messaging_app_4, for instance -- is never generated and so is never cached,
    # and treating its permanent absence as a miss forced a full retrain on every
    # selective run of any workflow containing one.
    shared &= (
        _generation_eligible_commands(workflow_folderpath)
        if generation_eligible is None
        else generation_eligible
    )
    if not shared:
        return set()

    cache_root = os.path.join(
        workflow_folderpath, determinism.COMMAND_INFO_FOLDERNAME,
        utterance_cache.CACHE_DIRNAME,
    )
    if not os.path.isdir(cache_root):
        return shared

    # rsplit, not split: slugify deliberately preserves '.', so a dotted command name
    # produces a dotted stem and splitting on the FIRST dot would truncate it. The
    # trailing two components are always the variant key and the extension.
    cached_stems = {
        name.rsplit(".", 2)[0]
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
