"""Phase 9 step 1 pilot: train one arm on a copy of the measured workflow, score it, time it.

EXPERIMENT DRIVER. It never writes to the source workflow: it copies the source into a
scratch directory and works only there. Run it with ``--dry-run`` first; the dry run prints
the plan and touches nothing. The source and the cache checkout are required: pass
``--source`` and ``--cache-from``, or set FW_EXPERIMENT_SOURCE and FW_EXPERIMENT_CACHE_FROM.

Steps::

    1  copy   source -> <scratch>/workflow, then the generated-utterance and
          parameter-example caches from --cache-from (only those two directories)
    2  block  LLM access for the run (see "How LLM calls are blocked")
    3  train  train_workflow(copy) under experiment_reserved_class_arm(arm), timed
    4  check  no command fell back to seeds-only (a fallback means an LLM call was attempted)
    5  cases  build the scoring cases from the exported persona holdout
    6  predict run the cases through the trained heads (arm checked against the artifacts)
    7  score  per-head metrics, Wilson intervals, printed and written as JSON
    8  edit   (--one-seed-edit only) change one seed of one command in the copy, retrain under
           the same arm, time it

How LLM calls are blocked
-------------------------
fastworkflow reads LLM settings through ``fastworkflow.get_env_var``, which takes the values
handed to ``fastworkflow.init`` before the process environment. The driver passes the
workflow's own env files plus overrides that replace every ``LITELLM_API_KEY_*`` with an
invalid key, and point ``LITELLM_PROXY_API_BASE`` (when set) at a closed local port. The
model strings are deliberately NOT changed: the utterance-cache key includes the model
string (``utterance_cache.py``), so changing it would miss the cache. Every real generation
site passes its key explicitly, so the invalid key is what the provider sees. Generation
retries a few times, then falls back to seeds and records ``fell_back`` in
``training_provenance.json``. Step 4 turns any fallback into a failure, so a blocked call
cannot pass silently.

``--one-seed-edit`` needs a real LLM call for the edited command, because a seed edit
changes that command's cache key. It therefore refuses to run unless
``--allow-llm-for-edit`` is given, and then the edit retrain runs unblocked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Optional

#: The arms this driver can train: every arm of RESERVED_CLASS_ARMS. ``control`` and ``ratio_0_5``
#: are RETIRED (Phase 9 step 5) and are refused with the trainer's reason, not offered here.
ARM_CHOICES = ("sparse_1", "sparse_8")  # tests check this against RESERVED_CLASS_ARMS
HEAD_SCOPE_CHOICES = ("flattened", "scoped")  # tests check this against HEAD_SCOPES
#: The arms any head layout may run under: the sparse not-here class (both layouts need it).
SCOPED_ARMS = ("sparse_1", "sparse_8")
#: The copy's folder name inside a scratch directory. Pruning and the probe use the same name.
WORKFLOW_DIRNAME = "workflow"
#: Environment variables that supply the required ``--source`` and ``--cache-from`` paths.
SOURCE_ENV = "FW_EXPERIMENT_SOURCE"
CACHE_FROM_ENV = "FW_EXPERIMENT_CACHE_FROM"
#: The cache checkout's caches, env and passwords are READ (copied or read); never written.
CACHE_DIRS = ("utterance_cache", "param_example_cache")
REPORT_FILENAME = "cache_report.json"
BLOCKED_KEY = "pilot-llm-blocked-no-valid-key"
CLOSED_PROXY_BASE = "http://127.0.0.1:9"  # nothing listens on port 9: connections are refused
LLM_KEY_VARS = (
    "LITELLM_API_KEY_SYNDATA_GEN", "LITELLM_API_KEY_PARAM_EXTRACTION", "LITELLM_API_KEY_AGENT",
    "LITELLM_API_KEY_PLANNER", "LITELLM_API_KEY_RESPONSE_GEN", "LITELLM_API_KEY_CONVERSATION_STORE",
    "LITELLM_API_KEY_COMMANDMETADATA_GEN", "LITELLM_PROXY_API_KEY",
)


class PilotFailure(RuntimeError):
    """A step failed in a way that invalidates the run."""


def copy_caches(cache_from: str, workflow_copy: str) -> list[str]:
    """Copy only the two cache directories from the owner's ___command_info. Nothing else."""
    source_info = os.path.join(cache_from, "___command_info")
    destination_info = os.path.join(workflow_copy, "___command_info")
    copied = []
    for name in CACHE_DIRS:
        source = os.path.join(source_info, name)
        if not os.path.isdir(source):
            raise PilotFailure(f"cache directory missing: {source}")
        shutil.copytree(source, os.path.join(destination_info, name), dirs_exist_ok=True)
        copied.append(name)
    return copied


def required_path(value: Optional[str], env_name: str, flag: str) -> str:
    """A required path: the command-line value, else the environment variable. Raises ValueError when neither is set."""
    chosen = value or os.environ.get(env_name)
    if not chosen:
        raise ValueError(f"{flag} is required: pass it on the command line or set {env_name}")
    return chosen


def copy_env(cache_from: str, scratch: str) -> str:
    """Copy the cache checkout's fastworkflow.env into the scratch area (never edited in place)."""
    source = os.path.join(cache_from, "fastworkflow.env")
    if not os.path.isfile(source):
        raise PilotFailure(f"no fastworkflow.env in {cache_from}; pass --env-file")
    destination_dir = os.path.join(scratch, "env")
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, "fastworkflow.env")
    shutil.copyfile(source, destination)
    return destination


def default_passwords(cache_from: str) -> Optional[str]:
    path = os.path.join(cache_from, "fastworkflow.passwords.env")
    return path if os.path.isfile(path) else None


def _worktree_execution_names(source: str) -> set[str]:
    """Qualified names of the workflow's execution commands, from the file layout (read only).

    A root file is named by its stem (global); a file in a context folder is "Context/stem".
    Planning commands (those with a PlanGenerator) and underscore-prefixed files are skipped.
    Both the bare stem and "*/stem" are returned for root files, since the global context
    may be spelled either way.
    """
    commands_dir = os.path.join(source, "_commands")
    names: set[str] = set()
    for root, dirs, files in os.walk(commands_dir):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for name in files:
            if not name.endswith(".py") or name.startswith((".", "_")):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                if "PlanGenerator" in handle.read():
                    continue
            stem = name[:-3]
            if os.path.abspath(root) == os.path.abspath(commands_dir):
                names.update({stem, f"*/{stem}"})
            else:
                names.add(f"{os.path.basename(root)}/{stem}")
    return names


def _core_command_names() -> set[str]:
    """The framework's core commands (IntentDetection/*), which every head trains on."""
    import fastworkflow

    folder = os.path.join(os.path.dirname(fastworkflow.__file__), "_workflows",
                          "command_metadata_extraction", "_commands", "IntentDetection")
    return {f"IntentDetection/{name[:-3]}" for name in os.listdir(folder)
            if name.endswith(".py") and not name.startswith(("_", "."))}


def expected_cache_hits(source: str, cache_from: str) -> dict:
    """Count the cache checkout's utterance-cache files whose command slug matches a worktree execution command.

    Matched by slug only. The variant key (seeds, generation dials, model string, generator
    source) is verified at training time, so these are the files that can hit, not a promise.
    """
    from fastworkflow.train.utterance_cache import slugify

    names = _worktree_execution_names(source)
    core = _core_command_names()
    slug_to_names: dict[str, set[str]] = {}
    for name in names | core:
        slug_to_names.setdefault(slugify(name), set()).add(name)
    core_slugs = {slugify(name) for name in core}
    cache_dir = os.path.join(cache_from, "___command_info", "utterance_cache")
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".json"))
    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for filename in files:
        slug = max((s for s in slug_to_names if filename.startswith(s + ".")), key=len, default=None)
        if slug is None:
            unmatched.append(filename)
        else:
            matched[slug] = matched.get(slug, 0) + 1
    without_file = sorted(slug for slug in slug_to_names if slug not in matched and slug not in core_slugs)
    param_dir = os.path.join(cache_from, "___command_info", "param_example_cache")
    return {
        "worktree_execution_commands": len([n for n in names if not n.startswith("*/")]),
        "cache_files": len(files),
        "core_commands_in_every_head": len(core),
        "core_command_files_matched": sum(c for slug, c in matched.items() if slug in core_slugs),
        "files_matched_to_worktree_commands": sum(c for slug, c in matched.items() if slug not in core_slugs),
        "distinct_commands_with_a_matching_file": len(matched),
        "commands_without_a_matching_file": len(without_file),
        "files_not_matching_any_worktree_command": len(unmatched),
        "param_example_cache_files": len(os.listdir(param_dir)) if os.path.isdir(param_dir) else 0,
        "sample_commands_without_file": without_file[:8],
    }


def prepare_registry(folder: str):
    """The trainer's own preamble (train_workflow's first lines): rebuild the routing artifacts
    in the COPY, then return the command directory. Writes only into the copy."""
    from fastworkflow.command_context_model import CommandContextModel
    from fastworkflow.command_directory import CommandDirectory
    from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry

    CommandContextModel.load(folder)
    RoutingDefinition.build(folder)
    CommandDirectory.load(folder).save()
    return RoutingRegistry.get_definition(folder, load_cached=False).command_directory


def utterance_command_names(cmd_dir) -> list[str]:
    """Every command the trainer asks utterances for: execution and core commands that take input.

    The trainer walks each context's label space, which holds these commands. Planning commands
    and the reserved wildcard are not labels, so they are excluded here as well.
    """
    from fastworkflow.model_pipeline_training import _requires_utterances
    from fastworkflow.nlu_labels import WILDCARD_LABEL, label_of

    names = set(cmd_dir.get_execution_commands()) | set(cmd_dir.core_command_names)
    out = []
    for name in sorted(names):
        if label_of(name) == WILDCARD_LABEL:
            continue
        cmd_dir.ensure_command_hydrated(name)
        if _requires_utterances(cmd_dir, name):
            out.append(name)
    return out


def cache_report(folder: str) -> dict:
    """Hits and misses per cache for the variant keys the trainer would ask for. No LLM call, no training.

    Keys come from the trainer's own functions: ``generate_synthetic.utterance_fingerprint`` and
    ``generate_param_examples.param_example_fingerprint``. Lookups use the caches' own ``lookup``,
    which only reads. A miss says what differs from the closest stored variant
    (``utterance_cache.describe_fingerprint_divergence``, the same helper the trainer logs with).

    Call estimates. An utterance miss makes ceil(personas / personas_per_batch) completions, one
    per batch, and a failed call is retried, which is not counted. The persona count is the
    configured one, so this is an upper bound when the persona pool is smaller. A parameter-example
    miss makes one completion.
    """
    import math

    import fastworkflow
    from fastworkflow.model_pipeline_training import get_route_layer_filepath_model
    from fastworkflow.train import generate_synthetic as gs
    from fastworkflow.train import param_example_cache as pc
    from fastworkflow.train import utterance_cache as uc
    from fastworkflow.train.__main__ import (
        PARAM_EXAMPLE_COUNT,
        PARAM_VALIDATION_THRESHOLD,
        _get_commands_with_parameters,
        parameter_model_of,
    )
    from fastworkflow.train.determinism import get_training_seed
    from fastworkflow.utils.generate_param_examples import extract_field_details, param_example_fingerprint

    cmd_dir = prepare_registry(folder)
    seed = get_training_seed()
    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    personas = gs._resolve_generation_count(None, "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS")
    per_persona = gs._resolve_generation_count(None, "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA")
    per_batch = gs._resolve_generation_count(None, "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH")
    batches = math.ceil(personas / per_batch) if personas and per_batch else 0

    utterance_cache = uc.UtteranceCache(folder, mode=uc.MODE_REUSE)
    utterance = {"commands": 0, "hits": 0, "misses": [], "uncacheable": []}
    for name in utterance_command_names(cmd_dir):
        seeds = list(cmd_dir.get_utterance_metadata(name).plain_utterances)
        fingerprint = gs.utterance_fingerprint(seeds, name, personas, per_persona, per_batch, model)
        utterance["commands"] += 1
        if utterance_cache.lookup(fingerprint, seed) is not None:
            utterance["hits"] += 1
            continue
        utterance["misses"].append({
            "command": name,
            "differs": uc.describe_fingerprint_divergence(utterance_cache.root, fingerprint)
                       or "no stored variant for this command",
            "estimated_llm_calls": batches,
        })

    param_cache = pc.ParamExampleCache(folder, mode=uc.MODE_REUSE)
    param = {"commands": 0, "hits": 0, "misses": [], "uncacheable": []}
    json_path = get_route_layer_filepath_model(folder, "command_directory.json")
    for name, metadata in sorted(_get_commands_with_parameters(json_path).items()):
        fields = parameter_model_of(folder, metadata)
        if fields is None:
            continue
        annotations = fields.model_fields
        fingerprint = param_example_fingerprint(
            annotations, name, PARAM_EXAMPLE_COUNT, PARAM_VALIDATION_THRESHOLD, model,
            field_details=extract_field_details(annotations), completion_fn=None)
        param["commands"] += 1
        if fingerprint is None:
            param["uncacheable"].append(name)
            continue
        if param_cache.lookup(fingerprint, seed) is not None:
            param["hits"] += 1
            continue
        param["misses"].append({
            "command": name,
            "differs": uc.describe_fingerprint_divergence(param_cache.root, fingerprint)
                       or "no stored variant for this command",
            "estimated_llm_calls": 1,
        })

    utterance["estimated_llm_calls"] = sum(m["estimated_llm_calls"] for m in utterance["misses"])
    param["estimated_llm_calls"] = sum(m["estimated_llm_calls"] for m in param["misses"])
    return {
        "seed": seed,
        "model": model,
        "utterance_settings": {"personas": personas, "utterances_per_persona": per_persona,
                               "personas_per_batch": per_batch},
        "utterance_cache": utterance,
        "param_example_cache": param,
        "estimated_llm_calls_total": utterance["estimated_llm_calls"] + param["estimated_llm_calls"],
        "notes": [
            "Keys are the trainer's own; a hit means the variant file holds this seed's entry.",
            "Utterance call counts are an upper bound when the persona pool is smaller than the configured count.",
            "Failed calls are retried by the generator; retries are not counted.",
            "The wildcard command's generator is LLM-free and is not counted.",
        ],
    }


def format_cache_report(report: dict) -> str:
    lines = [f"seed {report['seed']}  model {report['model']}  settings {report['utterance_settings']}"]
    for label, block in (("utterance_cache", report["utterance_cache"]),
                         ("param_example_cache", report["param_example_cache"])):
        lines.append(f"{label}: {block['commands']} commands, {block['hits']} hits, "
                     f"{len(block['misses'])} misses, {len(block['uncacheable'])} uncacheable, "
                     f"~{block['estimated_llm_calls']} LLM calls to fill")
        for miss in block["misses"]:
            lines.append(f"  miss {miss['command']}: {miss['differs']}")
    lines.append(f"total estimated LLM calls to fill: ~{report['estimated_llm_calls_total']}")
    return "\n".join(lines)


def fill_cache(folder: str) -> str:
    """Generate ONLY the missing cache entries, with the LLM allowed, into the copy. Stops before training.

    Runs the trainer's own generation path with the caches installed: a hit returns the stored
    entry without a call, and a miss calls the LLM and stores the result. Parameter examples are
    generated by the trainer's helper, which also writes <command>_param_labeled.json into the copy.
    Requires explicit approval (--allow-llm-for-fill), and the caller must have initialised fastworkflow with the
    unblocked env. Returns the cache counters as text.
    """
    import fastworkflow
    from fastworkflow.model_pipeline_training import _get_utterances
    from fastworkflow.train import param_example_cache as pc
    from fastworkflow.train import utterance_cache as uc
    from fastworkflow.train.__main__ import _generate_dspy_examples_helper

    cmd_dir = prepare_registry(folder)
    utterance_cache = uc.UtteranceCache(folder, mode=uc.MODE_REUSE)
    param_cache = pc.ParamExampleCache(folder, mode=uc.MODE_REUSE)
    uc.set_utterance_cache(utterance_cache)
    pc.set_param_example_cache(param_cache)
    workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"cache_fill_{os.path.basename(folder)}")
    try:
        for name in utterance_command_names(cmd_dir):
            _get_utterances(workflow, folder, cmd_dir, name)
        _generate_dspy_examples_helper(workflow)
    finally:
        uc.set_utterance_cache(None)
        pc.set_param_example_cache(None)
        workflow.close()
    return f"{utterance_cache.format_summary()}\n{param_cache.format_summary()}"


def plan_steps(args: argparse.Namespace, workflow_copy: str, scratch: str) -> list[str]:
    env_note = args.env_file or f"a copy of {args.cache_from}/fastworkflow.env"
    passwords = args.passwords_file or (default_passwords(args.cache_from) if args.cache_from else None)
    if args.cache_report or args.fill_cache_only:
        copy_step = (f"1 copy   {args.source} -> {workflow_copy} (read only source); then "
                     f"{', '.join(CACHE_DIRS)} from {args.cache_from}/___command_info")
        cache_steps = [
            f"2 env    {env_note}; passwords from {passwords or 'none'}",
        ]
        if args.cache_report:
            return [copy_step, "2 block  LLM keys invalid (no call is made)", *cache_steps,
                    f"3 report keys from the trainer's own functions, lookups only; "
                    f"{os.path.join(scratch, REPORT_FILENAME)}"]
        return [copy_step, "2 LLM    ALLOWED for this step (explicit approval): env unblocked",
                *cache_steps,
                "3 fill   generate ONLY the missing utterance and parameter-example entries into the copy",
                "4 stop   no training; the copy keeps the filled caches",
                f"5 report {os.path.join(scratch, REPORT_FILENAME)} after the fill"]
    steps = [
        f"1 copy   {args.source} -> {workflow_copy} (source is read only); then "
        f"{', '.join(CACHE_DIRS)} from {args.cache_from}/___command_info" if args.cache_from else
        f"1 copy   {args.source} -> {workflow_copy} (source is read only)",
        f"2 block  LLM keys overridden with an invalid key; closed proxy base if set; env from {env_note}; "
        f"passwords from {passwords or 'none'}",
        f"3 train  train_workflow({workflow_copy}) under arm {args.arm!r}, head scope "
        f"{args.head_scope or 'unset (today layout)'}, train seed "
        f"{args.train_seed if args.train_seed is not None else 'default (fixed)'}, wall clock timed",
        "4 check  no fell_back command in training_provenance.json",
        f"5 cases  {os.path.join(scratch, 'cases.json')} from ___command_info/not_here_holdout.json",
        f"6 predict {os.path.join(scratch, 'results.json')} (arm checked against the artifacts)",
        f"7 score  {os.path.join(scratch, 'summary.json')} (printed)",
    ]
    if args.one_seed_edit:
        steps.append(
            "8 edit   add one seed to one command in the copy, retrain under the same arm (timed); "
            + ("LLM allowed for this step" if args.allow_llm_for_edit else "REFUSED: needs --allow-llm-for-edit")
        )
    return steps


TRAINED_ARTIFACT_NAMES = ("versions", "current", "current.json", ".publication.lock", "global")


def _source_artifact_ignorer(source: str):
    """copytree ignore hook: inside the source's ___command_info, drop trained artifacts.

    Skipped: the versions/ tree, the current/global links, current.json, the publication
    lock, and every symlink or folder that points into versions/. The caches are not
    skipped here; copy_caches copies them from --cache-from afterwards. A fresh copy must
    look untrained apart from its caches.
    """
    info = os.path.join(os.path.abspath(source), "___command_info")
    versions = os.path.realpath(os.path.join(info, "versions"))

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {n for n in names if n == "__pycache__"}
        if os.path.abspath(directory) != info:
            return ignored
        for name in names:
            path = os.path.join(directory, name)
            if name in TRAINED_ARTIFACT_NAMES or os.path.islink(path):
                ignored.add(name)
            elif os.path.isdir(path) and os.path.realpath(path).startswith(versions + os.sep):
                ignored.add(name)
        return ignored

    return ignore


def copy_workflow(source: str, destination: str) -> str:
    """Copy the source workflow without its trained artifacts. Never writes to source."""
    if os.path.exists(destination):
        raise PilotFailure(f"{destination} already exists; choose an empty scratch directory")
    shutil.copytree(source, destination, ignore=_source_artifact_ignorer(source))
    return destination


def blocked_env(env: dict[str, str]) -> dict[str, str]:
    """The env the run uses: the workflow's values with every LLM key made invalid."""
    blocked = dict(env)
    for key in list(blocked):
        if key.startswith("LITELLM_API_KEY"):
            blocked[key] = BLOCKED_KEY
    for key in LLM_KEY_VARS:
        blocked[key] = BLOCKED_KEY
    if blocked.get("LITELLM_PROXY_API_BASE"):
        blocked["LITELLM_PROXY_API_BASE"] = CLOSED_PROXY_BASE
    blocked["LLM_CALLS_BLOCKED_BY_PILOT"] = "1"
    return blocked


def read_env(workflow_copy: str, env_file: str, passwords_file: Optional[str]) -> dict[str, str]:
    from dotenv import dotenv_values

    env_path = env_file if os.path.isabs(env_file) else os.path.join(workflow_copy, env_file)
    if not os.path.isfile(env_path):
        raise PilotFailure(f"env file not found: {env_path}")
    env = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    passwords = passwords_file or os.path.join(workflow_copy, "fastworkflow.passwords.env")
    if os.path.isfile(passwords):
        env.update({k: v for k, v in dotenv_values(passwords).items() if v is not None})
    return env


def fallen_back_commands(workflow_copy: str) -> list[str]:
    path = os.path.join(workflow_copy, "___command_info", "training_provenance.json")
    if not os.path.isfile(path):
        raise PilotFailure(f"no training_provenance.json after training: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    fell_back = [name for name, record in payload.get("commands", {}).items() if record.get("fell_back")]
    for context, commands in payload.get("context_training", {}).items():
        for command, record in commands.items():
            # context_training is keyed by context, then by the qualified command name
            if record.get("status") == "included_fallback" and command not in fell_back:
                fell_back.append(command)
    return sorted(fell_back)


_SEED_LIST = re.compile(r"plain_utterances\s*=\s*\[\s*\n?")


def add_one_seed(workflow_copy: str) -> tuple[str, str, str]:
    """Add one seed to the first execution command (sorted file order) in the COPY.

    Returns (file path, command name, new seed). Planning commands are skipped.
    """
    commands_dir = os.path.join(workflow_copy, "_commands")
    candidates = []
    for root, dirs, files in os.walk(commands_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        for name in sorted(files):
            if name.endswith(".py") and not name.startswith(("_", ".")):
                candidates.append(os.path.join(root, name))
    for path in candidates:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        if "PlanGenerator" in text or not _SEED_LIST.search(text):
            continue
        command = os.path.splitext(os.path.basename(path))[0]
        context = os.path.basename(os.path.dirname(path))
        if os.path.abspath(os.path.dirname(path)) != os.path.abspath(commands_dir):
            command = f"{context}/{command}"
        seed = f"pilot edit: please {command.split('/')[-1].replace('_', ' ')} right now"
        match = _SEED_LIST.search(text)
        edited = text[: match.end()] + f'        "{seed}",\n' + text[match.end():]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(edited)
        return path, command, seed
    raise PilotFailure("no execution command with a plain_utterances list found in the copy")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="not_here_pilot",
        description="Train one reserved-class arm on a copy of the measured workflow, score it and time it.",
    )
    parser.add_argument("--arm", default=None,
                        help=f"required to train: one of {', '.join(ARM_CHOICES)}; "
                             f"not used by --cache-report or --fill-cache-only")
    parser.add_argument("--head-scope", default=None, choices=HEAD_SCOPE_CHOICES,
                        help="head layout (Phase 9 step 3). Unset is today's layout. Either layout needs a "
                             "sparse arm (sparse_1 or sparse_8) and trains one global persona holdout")
    parser.add_argument("--cache-report", action="store_true",
                        help="copy, then report hits, misses and estimated LLM calls per cache; no LLM call, no training")
    parser.add_argument("--fill-cache-only", action="store_true",
                        help="copy, then generate only the missing cache entries with the LLM allowed; stops before training")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="model-training seed (weight init, shuffling, dropout, sparse reserved rows). "
                             "Never changes the persona holdout split, which stays on the fixed training seed")
    parser.add_argument("--allow-llm-for-fill", action="store_true",
                        help="required with --fill-cache-only (owner approval)")
    parser.add_argument("--scratch", required=True, help="empty directory for every output of the run")
    parser.add_argument("--env-file", default=None,
                        help="env file, relative to the copy or absolute. Default: a copy of the cache checkout's "
                             "fastworkflow.env from --cache-from (its model strings are the ones the cache "
                             "was generated with)")
    parser.add_argument("--passwords-file", default=None,
                        help="passwords env file. Default: the cache checkout's fastworkflow.passwords.env if it exists; "
                             "its keys are overridden for the run anyway")
    parser.add_argument("--source", default=None,
                        help=f"source workflow (read only). Required unless ${SOURCE_ENV} is set")
    parser.add_argument("--cache-from", default=None,
                        help="cache checkout whose ___command_info utterance_cache/ and param_example_cache/ "
                             f"are copied into the scratch copy (read only). Also the default env/passwords source. "
                             f"Required unless ${CACHE_FROM_ENV} is set")
    parser.add_argument("--one-seed-edit", action="store_true",
                        help="after scoring, edit one seed in the copy, retrain under the same arm, time it")
    parser.add_argument("--allow-llm-for-edit", action="store_true",
                        help="required with --one-seed-edit: the edited command must be regenerated by a real LLM")
    parser.add_argument("--dry-run", action="store_true", help="print the planned steps and exit")
    args = parser.parse_args(argv)

    try:
        args.source = required_path(args.source, SOURCE_ENV, "--source")
        args.cache_from = required_path(args.cache_from, CACHE_FROM_ENV, "--cache-from")
    except ValueError as error:
        parser.error(str(error))
    scratch = os.path.abspath(args.scratch)
    workflow_copy = os.path.join(scratch, WORKFLOW_DIRNAME)
    modes = [bool(args.cache_report), bool(args.fill_cache_only), bool(args.one_seed_edit)]
    if sum(modes) > 1:
        parser.error("choose one of --cache-report, --fill-cache-only, --one-seed-edit")
    if not (args.cache_report or args.fill_cache_only) and not args.arm:
        parser.error("--arm is required to train")
    if args.arm is not None and args.arm not in ARM_CHOICES:
        from fastworkflow.model_pipeline_training import RETIRED_ARMS

        if args.arm in RETIRED_ARMS:
            parser.error(f"--arm {args.arm!r} is retired: {RETIRED_ARMS[args.arm]}")
        parser.error(f"--arm {args.arm!r} is not an arm; choose one of {', '.join(ARM_CHOICES)}")
    if args.fill_cache_only and not args.allow_llm_for_fill:
        print("refused: --fill-cache-only needs --allow-llm-for-fill (explicit approval). It calls the LLM "
              "for every missing cache entry.", file=sys.stderr)
        return 2
    if args.head_scope is not None and args.arm not in SCOPED_ARMS:
        parser.error(f"--head-scope {args.head_scope} needs a sparse arm ({' or '.join(SCOPED_ARMS)}), "
                     f"not {args.arm!r}: the global persona holdout is drawn alongside the sparse class")
    if args.one_seed_edit and not args.allow_llm_for_edit:
        print("refused: --one-seed-edit needs --allow-llm-for-edit. A seed edit changes the cache key of "
              "the edited command, so its utterances must be regenerated by a real LLM; with the LLM blocked "
              "they would fall back to seeds and the timing would not measure an edit.", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Planned steps (dry run; nothing was copied, written or run):")
        for step in plan_steps(args, workflow_copy, scratch):
            print("  " + step)
        if args.cache_from:
            hits = expected_cache_hits(args.source, args.cache_from)
            print("Expected cache hits (slug match against the worktree's execution commands):")
            for key, value in hits.items():
                print(f"  {key}: {value}")
        return 0
    return _run(args, scratch, workflow_copy)


def _run(args: argparse.Namespace, scratch: str, workflow_copy: str) -> int:
    # Imported here so --help and --dry-run stay cheap.
    import fastworkflow
    from fastworkflow.model_pipeline_training import experiment_head_scope, experiment_reserved_class_arm
    from fastworkflow.train import score_not_here as scorer
    from fastworkflow.train import determinism
    from fastworkflow.train.__main__ import train_workflow

    os.makedirs(scratch, exist_ok=True)
    timings: dict[str, float] = {}

    copy_workflow(args.source, workflow_copy)
    if args.cache_from:
        copy_caches(args.cache_from, workflow_copy)
    env_file = args.env_file or copy_env(args.cache_from, scratch)
    passwords = args.passwords_file or (default_passwords(args.cache_from) if args.cache_from else None)
    env = read_env(workflow_copy, env_file, passwords)

    report_path = os.path.join(scratch, REPORT_FILENAME)
    if args.fill_cache_only:
        fastworkflow.init(env_vars=env)  # the only run that is allowed to reach the LLM
        counters = fill_cache(workflow_copy)
        print(counters)
        fastworkflow.init(env_vars=blocked_env(env))
        report = cache_report(workflow_copy)
        _dump(report_path, report)
        print(format_cache_report(report))
        return 0

    fastworkflow.init(env_vars=blocked_env(env))
    if args.cache_report:
        report = cache_report(workflow_copy)
        _dump(report_path, report)
        print(format_cache_report(report))
        return 0

    timings["train_seed"] = args.train_seed
    timings["head_scope"] = args.head_scope
    determinism.set_train_seed(args.train_seed)
    started = time.monotonic()
    with experiment_reserved_class_arm(args.arm), experiment_head_scope(args.head_scope):
        train_workflow(workflow_copy)
    timings["train_seconds"] = round(time.monotonic() - started, 2)

    fallen = fallen_back_commands(workflow_copy)
    if fallen:
        raise PilotFailure(f"LLM calls were attempted (fell back to seeds): {', '.join(fallen)}")

    cases_doc = {"schema": scorer.SCHEMA_VERSION, "cases": scorer.build_cases(scorer.load_holdout(workflow_copy))}
    _dump(os.path.join(scratch, "cases.json"), cases_doc)

    results = scorer.predict_cases(workflow_copy, cases_doc["cases"], args.arm)
    _dump(os.path.join(scratch, "results.json"), results)

    summary = scorer.summarize(results)
    _dump(os.path.join(scratch, "summary.json"), summary)
    print(scorer.format_summary(summary))

    if args.one_seed_edit:
        path, command, seed = add_one_seed(workflow_copy)
        fastworkflow.init(env_vars=env)  # unblocked: the edited command needs a real generation
        started = time.monotonic()
        with experiment_reserved_class_arm(args.arm), experiment_head_scope(args.head_scope):
            train_workflow(workflow_copy)
        timings["edit_retrain_seconds"] = round(time.monotonic() - started, 2)
        timings["edited_command"] = command
        timings["edited_seed"] = seed
        timings["edited_file"] = path
        timings["edit_fell_back"] = fallen_back_commands(workflow_copy)

    _dump(os.path.join(scratch, "timings.json"), timings)
    print(json.dumps(timings, indent=2))
    return 0


def _dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PilotFailure as failure:
        print(f"pilot failed: {failure}", file=sys.stderr)
        sys.exit(1)
