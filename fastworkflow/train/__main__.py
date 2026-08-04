import argparse
import contextlib
import functools
import os
import json
import shutil
import time
from dotenv import dotenv_values
import importlib.util

from colorama import Fore, Style

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import python_utils
from fastworkflow import ModuleType
from fastworkflow.model_pipeline_training import (
    TrainingDataError,
    train,
    get_route_layer_filepath_model,
    set_active_artifact_version,
    GLOBAL_CONTEXT_FOLDER,
)
from fastworkflow.utils.generate_param_examples import generate_dspy_examples
from fastworkflow.command_directory import CommandDirectory, get_cached_command_directory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.train import (
    artifact_versioning,
    determinism,
    duplicate_detection,
    param_example_cache,
    personas,
    selective_training,
    training_report,
    utterance_cache,
)

# Cache the datasets availability check result
_DATASETS_AVAILABLE = None


def _datasets_available() -> bool:
    """Check if the datasets package is available in the environment.
    
    Returns:
        bool: True if datasets can be imported, False otherwise.
    """
    global _DATASETS_AVAILABLE
    if _DATASETS_AVAILABLE is None:
        _DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None
    return _DATASETS_AVAILABLE


def _validate_command_inputs(workflow_path: str) -> None:
    """Fail before LLM generation when command seeds describe duplicate capabilities."""
    duplicate_report = duplicate_detection.scan_workflow(workflow_path)
    duplicate_detection.write_report(workflow_path, duplicate_report)
    if duplicate_report.duplicates:
        print(duplicate_detection.format_report(duplicate_report))
        pairs = ", ".join(
            f"{finding.command_a} / {finding.command_b}"
            for finding in duplicate_report.duplicates
        )
        raise TrainingDataError(
            "Duplicate command capabilities must be resolved before training: "
            f"{pairs}"
        )
    if duplicate_report.overlapping:
        print(duplicate_detection.format_report(duplicate_report))

    crd = RoutingRegistry.get_definition(workflow_path)
    cmd_dir = crd.command_directory
    core_commands = set(cmd_dir.core_command_names)
    thin: list[tuple[str, int]] = []
    for command_name in sorted(cmd_dir.get_commands()):
        if (
            command_name in core_commands
            or command_name.split("/")[-1]
            in duplicate_detection.NON_CAPABILITY_LABELS
        ):
            continue
        metadata = cmd_dir.get_utterance_metadata(command_name)
        if metadata is None:
            continue
        seed_count = len(metadata.plain_utterances)
        if seed_count < training_report.DEFAULT_MIN_SEED_UTTERANCES:
            thin.append((command_name, seed_count))
    if thin:
        details = ", ".join(f"{name} ({count})" for name, count in thin)
        print(
            f"{Fore.YELLOW}Seed guidance: {len(thin)} command(s) have fewer than "
            f"{training_report.DEFAULT_MIN_SEED_UTTERANCES} hand-written utterances: "
            f"{details}. Eight is advisory, based on one workflow; training will "
            f"continue.{Style.RESET_ALL}"
        )


def _repair_noop_publication(
    workflow_path: str, current_version: str | None
) -> None:
    """Repair current-version reader paths before applying no-op retention."""
    if current_version is None:
        raise TrainingDataError(
            "Training plan is empty, but no current artifact version is available; "
            "cannot repair compatibility entries or apply retention."
        )

    artifact_versioning.publish_version(workflow_path, current_version)
    previous_version = artifact_versioning.read_manifest(
        workflow_path, current_version
    ).get("previous_version")
    artifact_versioning.retain_current_and_previous(
        workflow_path, previous_version
    )


def _require_publishable_training_report(report) -> None:
    """Refuse publication when the training-data safety gate cannot pass."""
    if report is None:
        raise TrainingDataError(
            "Training-data safety report could not be produced; refusing to publish "
            "models."
        )
    if report.has_blocking_problems:
        names = ", ".join(row.command_name for row in report.blocking_rows)
        raise TrainingDataError(
            "Training data is structurally incomplete; refusing to publish models for: "
            f"{names}"
        )


def _with_workflow_persona_source(train_func):
    """Install one workflow-specific persona source for each training invocation.

    Training recurses into child workflows, so teardown restores the source that was
    active before this invocation rather than always clearing it. The outermost call still
    tears down to None, while a parent resumes with its own source after a child finishes.
    """
    @functools.wraps(train_func)
    def wrapped(workflow_path: str, *args, **kwargs):
        previous_source = personas.get_persona_source()
        source = personas.persona_source_for_workflow(workflow_path)
        personas.set_persona_source(source)
        try:
            return train_func(workflow_path, *args, **kwargs)
        finally:
            personas.set_persona_source(previous_source)

    return wrapped


@_with_workflow_persona_source
def train_workflow(workflow_path: str, regenerate_utterances: bool = False):
    # Ensure context model is parsed so downstream helpers have contexts
    CommandContextModel.load(workflow_path)
    RoutingDefinition.build(workflow_path)
    # Ensure the command directory is persisted so that downstream helpers
    # (e.g. DSPy example generation) can read it from disk without first
    # needing to rebuild it in-memory.
    cmd_dir = CommandDirectory.load(workflow_path)
    cmd_dir.save()
    RoutingRegistry.get_definition(workflow_path, load_cached=False)

    #first, recursively train all child workflows
    workflows_dir = os.path.join(workflow_path, "_workflows")
    if os.path.isdir(workflows_dir):
        for child_workflow in os.listdir(workflows_dir):
            if "__pycache__" in child_workflow:
                continue
            child_workflow_path = os.path.join(workflows_dir, child_workflow)
            if os.path.isdir(child_workflow_path):
                print(f"{Fore.YELLOW}Training child workflow: {child_workflow_path}{Style.RESET_ALL}")
                train_workflow(
                    child_workflow_path,
                    regenerate_utterances=regenerate_utterances,
                )

    commands_dir = os.path.join(workflow_path, "_commands")
    if not os.path.isdir(commands_dir):
        logger.info(f"No _commands directory found at {workflow_path}, skipping training")
        return
    
    # Check if datasets package is available for training
    # Command directory and routing artifacts are generated above regardless
    if not _datasets_available():
        logger.warning(
            f"datasets package not found in environment. Skipping intent detection training "
            f"and DSPy few-shot parameter extraction for workflow: {workflow_path}. "
            f"Other artifacts such as command_directory.json and routing_definition.json have been generated successfully."
        )
        return

    _validate_command_inputs(workflow_path)
    
    # Bring any pre-versioning artifacts under version control BEFORE writing anything,
    # so this run cannot merge new work into a set that has no rollback point.
    if migrated := artifact_versioning.migrate_legacy_to_version(workflow_path):
        print(f"{Fore.YELLOW}Migrated existing artifacts into version "
              f"{migrated}{Style.RESET_ALL}")

    seed = determinism.get_training_seed()
    previous_version = artifact_versioning.resolve_current_version(workflow_path)
    cache_mode = (
        utterance_cache.MODE_REGENERATE
        if regenerate_utterances
        else utterance_cache.MODE_REUSE
    )

    # create a workflow and refresh the parameter-example artifacts before deciding that
    # the intent models are already current. Model-context fingerprints do not cover a
    # deleted or corrupt <command>_param_labeled.json, so an early no-op would otherwise
    # skip repairing runtime parameter extraction.
    workflow = fastworkflow.Workflow.create(
        workflow_path,
        workflow_id_str=f"train_{workflow_path}"
    )

    # The other LLM-driven generation path (bd fix-czb), installed the same way and for the
    # same reason as the utterance cache below: generate_dspy_examples is reached with only a
    # command name and a parameter model, so it cannot be handed a workflow path. Without it,
    # two runs at the same seed write different <command>_param_labeled.json -- measured 0/1
    # identical on hello_world and 0/5 on messaging_app_4; with it, 1/1 and 5/5.
    param_cache = param_example_cache.ParamExampleCache(
        workflow_path, mode=cache_mode)
    param_example_cache.set_param_example_cache(param_cache)
    try:
        _generate_dspy_examples_helper(workflow)
    finally:
        param_example_cache.set_param_example_cache(None)

    if param_cache.enabled:
        print(f"{Fore.CYAN}{param_cache.format_summary()}{Style.RESET_ALL}")

    try:
        plan, training_signature = selective_training.compute_training_plan(
            workflow_path,
            selective_training.contexts_for_training(workflow_path),
            changed_only=True,
            seed=seed,
            carry_forward_from=previous_version,
            cache_mode=cache_mode,
        )
    except Exception:
        workflow.close()
        raise
    if not plan.contexts_to_train:
        try:
            _repair_noop_publication(workflow_path, previous_version)
        finally:
            workflow.close()
        print(f"{Fore.GREEN}Training artifacts are already up to date.{Style.RESET_ALL}")
        return
    print(f"{Fore.CYAN}{selective_training.format_plan(plan)}{Style.RESET_ALL}")

    version_id = artifact_versioning.new_version_id()
    artifact_versioning.write_manifest(
        workflow_path,
        version_id,
        seed=seed,
        previous_version=previous_version,
    )

    # generate_diverse_utterances cannot return provenance (its signature is public API
    # called from user-authored command files), so it pushes into this recorder instead.
    recorder = determinism.ProvenanceRecorder(workflow_path)
    determinism.set_provenance_recorder(recorder)
    # Same reason, same shape: generation reaches the cache through a module-level handle
    # because it cannot be handed a workflow path. This is what makes two runs at the same
    # seed train on the same data -- seeding alone does not, because the LLM redraws every
    # run (measured: 0/5 commands identical at a fixed seed before this landed).
    cache = utterance_cache.UtteranceCache(workflow_path, mode=cache_mode)
    utterance_cache.set_utterance_cache(cache)
    set_active_artifact_version(workflow_path, version_id)
    # `train()` rewrites heldout_evaluation.json from only the contexts it trained, so a
    # selective run would silently shrink the report. Snapshot it first.
    previous_heldout = selective_training.capture_heldout_evaluation(workflow_path)
    previous_provenance = selective_training.capture_training_provenance(
        workflow_path, plan.carry_forward_from
    )
    started = time.monotonic()
    try:
        train(
            workflow,
            contexts_to_train=(
                None if plan.is_full_retrain else set(plan.contexts_to_train)
            ),
        )
    finally:
        set_active_artifact_version(workflow_path, None)
        determinism.set_provenance_recorder(None)
        utterance_cache.set_utterance_cache(None)
        workflow.close()

    if cache.enabled:
        print(f"{Fore.CYAN}{cache.format_summary()}{Style.RESET_ALL}")

    # Written twice, deliberately. The top-level copy is the stable path the per-command
    # training report (fix-551.4) reads. The copy inside the version is what makes a
    # version self-describing: without it the next train overwrites the top-level file and
    # an older version can no longer say which personas and seed produced its utterances,
    # which is most of what rolling back to it is worth.
    with contextlib.suppress(OSError):
        provenance_path = recorder.save()
        if (
            plan.contexts_carried_forward
            and not selective_training.merge_training_provenance(
                workflow_path, plan, previous_provenance
            )
        ):
            raise selective_training.SelectiveTrainingError(
                "Could not merge carried-forward training provenance."
            )
        shutil.copy2(
            provenance_path,
            artifact_versioning.version_dir(workflow_path, version_id)
            / determinism.PROVENANCE_FILENAME,
        )

    # Runs BEFORE publishing, and raises rather than degrading. A selective run's version
    # holds only the contexts it retrained; `publish_version` removes the compatibility
    # entry of every context its version lacks, so carrying the rest forward is not an
    # optimisation, it is what stops the run from un-training the workflow. Failing here
    # leaves the previous version current and complete.
    if carried := selective_training.carry_forward_contexts(
        workflow_path, plan, version_id
    ):
        print(f"{Fore.CYAN}Carried forward {len(carried)} context(s) from version "
              f"{plan.carry_forward_from}: {', '.join(carried)}{Style.RESET_ALL}")
    selective_training.merge_heldout_evaluation(
        workflow_path, plan, previous_heldout)

    # Written into the version, not to a fixed top-level path, so that rolling back with
    # `versions publish <old>` also rolls the selective-training baseline back to the one
    # describing the artifacts that rollback restored.
    selective_training.save_training_signature(
        workflow_path, version_id, training_signature)

    artifact_versioning.write_manifest(
        workflow_path,
        version_id,
        train_duration_seconds=time.monotonic() - started,
        contexts_retrained=sorted(plan.contexts_to_train),
        contexts_carried_forward=sorted(plan.contexts_carried_forward),
    )
    report = training_report.report_training_data(
        workflow_path, print_report=False, write=True)
    _require_publishable_training_report(report)
    # Only a SUCCESSFUL train becomes current. A failed one leaves the previous version
    # published and complete, with its own partial version beside it rather than on top
    # of it -- which is strictly stronger than the pre-versioning behaviour this replaces.
    artifact_versioning.publish_version(workflow_path, version_id)
    artifact_versioning.retain_current_and_previous(
        workflow_path, previous_version)
    print(f"{Fore.GREEN}Training complete.{Style.RESET_ALL}")

    # Only after training has successfully (re)generated artifacts do we prune
    # leftovers from commands/contexts that no longer exist. Running this *after*
    # the writes (rather than wiping up-front) means a failed train leaves the
    # previous, complete ___command_info intact and runnable.
    _prune_stale_artifacts(workflow_path)

def _generate_dspy_examples_helper(workflow):
    json_path=get_route_layer_filepath_model(workflow.folderpath,"command_directory.json")
    # json_path = "./examples/sample_workflow/___command_info/command_directory.json"
    commands = _get_commands_with_parameters(json_path)
    for command_name in commands.keys():
        command_metadata = commands[command_name]
        module_file_path = command_metadata["parameter_path"]
        if module := python_utils.get_module(
            module_file_path, workflow.folderpath
        ):
            module_class_name = command_metadata["parameters_class"]

            if "." in module_class_name:
                (outer, inner) = module_class_name.split(".")
                outer_cls = getattr(module, outer)
                fields = getattr(outer_cls, inner)
            else:
                fields = getattr(module, module_class_name)

            examples, rejected_examples = generate_dspy_examples(
            field_annotations=fields.model_fields,
            command_name=command_name,
            num_examples=15,
            validation_threshold=0.3  # You can adjust this threshold as needed
            )
            output_dir = os.path.join(workflow.folderpath, "___command_info")
            os.makedirs(output_dir, exist_ok=True)

            # Format the examples for JSON
            examples_data = {
                "command_name": command_name,
                "valid_examples": examples,
                "rejected_examples": rejected_examples
            }

            # Save to JSON file
            output_file = os.path.join(output_dir, f"{command_name}_param_labeled.json")
            with open(output_file, 'w') as f:
                json.dump(examples_data, f, indent=2)
        else:
            None

def _prune_stale_artifacts(workflow_path: str):
    """Remove orphaned per-command and per-context training artifacts.

    An artifact is "orphaned" only when the command or context it belongs to no
    longer exists in the freshly built routing definition / command directory.
    This is intentionally conservative: it never touches the JSON snapshots
    themselves, and it never removes a model folder whose context is still known
    (so checked-in / still-valid models such as ErrorCorrection are preserved).

    Because orphans are looked up by name/context at run time, they are harmless;
    this is purely a cleanliness pass and is safe to run after a successful train.
    """
    info_dir = os.path.join(workflow_path, "___command_info")
    if not os.path.isdir(info_dir):
        return

    # Expected per-command DSPy artifacts: one per command that has parameters.
    cmd_dir_json = os.path.join(info_dir, "command_directory.json")
    expected_param_files: set[str] = set()
    if os.path.isfile(cmd_dir_json):
        commands = _get_commands_with_parameters(cmd_dir_json)
        expected_param_files = {f"{name}_param_labeled.json" for name in commands}

    # Expected per-context model folders: every context known to routing, with
    # "*" mapped to the global folder. Folders for unknown contexts are orphans.
    crd = RoutingRegistry.get_definition(workflow_path)
    expected_ctx_folders = {GLOBAL_CONTEXT_FOLDER}
    for ctx_name in crd.contexts.keys():
        expected_ctx_folders.add(GLOBAL_CONTEXT_FOLDER if ctx_name == "*" else ctx_name)

    for entry in os.listdir(info_dir):
        full_path = os.path.join(info_dir, entry)
        if entry in artifact_versioning.RESERVED_TOPLEVEL_NAMES:
            # `versions/` holds every version's real bytes and `current` points at one of
            # them. Neither is ever an orphan, and neither may be pruned.
            continue
        if (
            entry.endswith("_param_labeled.json")
            and os.path.isfile(full_path)
            and entry not in expected_param_files
        ):
            with contextlib.suppress(OSError):
                os.remove(full_path)
                logger.info(f"Pruned orphaned param artifact: {full_path}")
        elif (
            os.path.isdir(full_path)
            and os.path.isfile(os.path.join(full_path, "threshold.json"))
            and entry not in expected_ctx_folders
        ):
            # Under versioning this entry is a compatibility pointer into a version.
            # shutil.rmtree REFUSES to follow a symlink, so without unrouting it first
            # this branch would silently stop cleaning orphans and leave the pointer
            # behind. Unrouting removes only the pointer; the version's bytes survive and
            # remain recoverable by republishing that version.
            if artifact_versioning.unroute_context(workflow_path, entry):
                logger.info(f"Unrouted orphaned context: {full_path}")
                continue
            with contextlib.suppress(OSError):
                shutil.rmtree(full_path)
                logger.info(f"Pruned orphaned context model folder: {full_path}")


def _get_commands_with_parameters(json_path):
    """
    Parse command_directory.json file and create a mapping between command names 
    and their parameter extraction signature module paths for commands that have
    a non-null command_parameters_class.
    
    Args:
        json_path: Path to the command_directory.json file
        
    Returns:
        dict: Dictionary mapping command names to parameter_extraction_signature_module_path
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        command_directory = json.load(f)
    
    # Extract the command metadata
    commands_metadata = command_directory.get("map_command_2_metadata", {})
    
    # Initialize result dictionary
    commands_with_parameters = {}
    
    # Iterate through each command entry
    for command_key, metadata in commands_metadata.items():
        # Check if command_parameters_class is not null
        if metadata.get("command_parameters_class") is not None:
            # We want to train on the full command path including the ContextFolder prefix
            command_name = command_key.split("/")[-1]
            
            # Get the parameter extraction module path
            param_extraction_path = metadata.get("parameter_extraction_signature_module_path")
            
            # Add to result dictionary
            commands_with_parameters[command_name] = {
                "parameter_path": param_extraction_path,
                "full_command_key": command_key,
                "parameters_class": metadata.get("command_parameters_class"),
                "input_class": metadata.get("input_for_param_extraction_class")
            }
    
    return commands_with_parameters

def is_fast_workflow_trained(fastworkflow_folderpath: str):
    # Check the artifacts for exactly the contexts that the CME trainer produces.
    cme_workflow_folderpath = os.path.join(
        fastworkflow_folderpath,
        '_workflows',
        'command_metadata_extraction',
    )
    try:
        trained_contexts = selective_training.contexts_for_training(
            cme_workflow_folderpath
        )
    except Exception:
        return False

    required_artifact_paths = []
    for context_name in trained_contexts:
        context_folder = (
            GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        )
        context_artifact_dir = os.path.join(
            cme_workflow_folderpath, "___command_info", context_folder
        )
        for artifact_name in selective_training.REQUIRED_CONTEXT_ARTIFACTS:
            artifact_path = os.path.join(context_artifact_dir, artifact_name)
            if not os.path.exists(artifact_path):
                return False
            required_artifact_paths.append(artifact_path)

    if not required_artifact_paths:
        return False

    oldest_model_mtime = min(
        os.path.getmtime(path) for path in required_artifact_paths
    )

    commands_path = os.path.join(
        cme_workflow_folderpath,
        "_commands",
    )

    for root, _, files in os.walk(commands_path):
        for file in files:
            if file.endswith(".pyc"):
                continue
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) > oldest_model_mtime:
                return False

    return True

def train_main(args):
    """Main function to train the workflow."""
    # Resolve the workflow path to absolute path to handle relative paths correctly
    workflow_path = os.path.abspath(args.workflow_folderpath)
    
    if not os.path.isdir(workflow_path):
        print(
            f"{Fore.RED}Error: The specified workflow path '{workflow_path}' is not a valid directory.{Style.RESET_ALL}"
        )
        exit(1)

    env_vars = {
        **dotenv_values(args.env_file_path),
        **dotenv_values(args.passwords_file_path)
    }
    if not env_vars.get("SPEEDDICT_FOLDERNAME"):
        print(f'Env file path: {args.env_file_path}')
        raise ValueError("SPEEDDICT_FOLDERNAME env var not found! Is the env file missing? or path is incorrect?")
    if not env_vars.get("LITELLM_API_KEY_SYNDATA_GEN"):
        print(f"LITELLM_API_KEY_SYNDATA_GEN password env var not found! OK if this is Bedrock. Otherwise, is the password env file missing or incorrect path? Path: {args.passwords_file_path}")

    fastworkflow.init(env_vars=env_vars)

    regenerate_utterances = getattr(args, "regenerate_utterances", False)
    # Check if fastworkflow has been trained, and train it if not. The regeneration
    # flag is forwarded when CME training is already required; it does not force
    # otherwise-unneeded CME retraining.
    fastworkflow_folderpath = fastworkflow.get_fastworkflow_package_path()
    if (
        "fastworkflow" not in workflow_path and
        not is_fast_workflow_trained(fastworkflow_folderpath)
    ):
        train_workflow(
            fastworkflow_folderpath,
            regenerate_utterances=regenerate_utterances,
        )

    train_workflow(
        workflow_path,
        regenerate_utterances=regenerate_utterances,
    )
    # Printed last so the actionable table is closest to the prompt. Already written to
    # disk by train_workflow, hence write=False.
    training_report.report_training_data(workflow_path, print_report=True, write=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the intent detection pipeline for a workflow"
    )
    parser.add_argument("workflow_folderpath", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    parser.add_argument("passwords_file_path", help="Path to the passwords file")
    parser.add_argument(
        "--regenerate-utterances",
        action="store_true",
        help="Ignore the generated-utterance cache and call the LLM again.",
    )
    args = parser.parse_args()
    train_main(args)