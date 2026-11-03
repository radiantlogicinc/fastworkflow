"""What a sub-task's run is shown: the full listing, or the command shortlist above a size (design 3.10, 10.15).

Production default: FULL, byte for byte as it is today. The shortlist is NOT the production default: the 10.7 gate
(the owner's choice) decides when the threshold rule becomes the default. The rule itself (``THRESHOLD`` arm, selected
by ``listing_arm``): a listing whose full text is at most ``FULL_LISTING_MAX_CHARS`` is the full listing; a larger one
is the COMMAND shortlist:

- the commands gathered for the sub-task, expanded with their producers, the commands that navigate to their
  context, and a planning command's callees, listed in full;
- the global commands as one line (3.9), unless a shortlisted command is global;
- every other context as one line.

Gathering (``live_gather``) uses what the workflow declares and what its trained heads answer: the names the sub-task
writes (exact names along the lookup path and the contexts reachable by descend edges, 3.9, 3.15), the labels the
per-context heads answer for the sub-task (``CommandRouter.predict_with_details``, local classifiers, no LLM call), the
planning labels among the heads' top-k (the unsure candidates, 3.9), and a planning-command match's distribution when
the caller has one (``planning_probabilities``). Without trained heads a context contributes nothing from them.

Selection (EXPERIMENT-ONLY): ``listing_arm`` selects ``FULL`` (always the full listing), ``COMMAND_SHORTLIST`` (always the
shortlist) or ``THRESHOLD`` (the rule above) for a ``with`` block. Its default, ``None``, is ``FULL``.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import fastworkflow
from fastworkflow import planning_commands
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.context_navigation import (
    collect_descend_edges, display_context_name, exact_name_scopes, live_context_chain,
    navigation_steps_to_context, plan_navigation)

logger = logging.getLogger(__name__)

#: A listing whose full text is at most this many characters is given whole (production policy). Set from the
#: offline measurement of Phase 10 step 4 (10.7): the library's full listing (6,516 characters from its Library
#: context) is whole, the no-hints tutorial's (6,353) too, and the app's (82,055) is shortlisted.
FULL_LISTING_MAX_CHARS = 7000

#: A planning command's unsure candidate: a probability at least this, below the match threshold. PROVISIONAL
#: (offline); the 10.7 measurement used this value.
UNSURE_CANDIDATE_FLOOR = 0.2
#: The match's option that means "no planning command fits"; never a candidate.
NONE_OPTION = "none"

#: The head labels that are not commands (a sub-task's run calls none of them).
NON_COMMAND_LABELS = ("parameter_value", "wildcard")

FULL = "full"
COMMAND_SHORTLIST = "command_shortlist"
THRESHOLD = "threshold"
ARMS = (FULL, COMMAND_SHORTLIST, THRESHOLD)

#: THE PRODUCTION LISTING ARM: the one line that changes production. FULL until the owner approves the threshold rule
#: after the live run (fix-3rsq.10.7 gate); set it to THRESHOLD to turn the size rule on (full up to
#: FULL_LISTING_MAX_CHARS, the trained-head shortlist above it). Both call sites and run_listing_text read it.
PRODUCTION_LISTING_ARM = FULL

#: EXPERIMENT-ONLY. The arm the run's listing uses inside a ``listing_arm`` block; None is the production default (FULL).
_EXPERIMENT_LISTING_ARM: str | None = None


@contextlib.contextmanager
def listing_arm(arm: str | None):
    """EXPERIMENT-ONLY: select a listing arm for the duration of a ``with`` block.

    ``FULL`` always gives the full listing; ``COMMAND_SHORTLIST`` always gives the shortlist; ``THRESHOLD`` gives the
    full listing up to ``FULL_LISTING_MAX_CHARS`` and the shortlist above it. ``None`` restores the production
    default, FULL. An unknown arm raises ValueError before anything changes.
    """
    global _EXPERIMENT_LISTING_ARM
    if arm is not None and arm not in ARMS:
        raise ValueError(f"unknown listing arm {arm!r}; expected one of {list(ARMS)}")
    previous = _EXPERIMENT_LISTING_ARM
    _EXPERIMENT_LISTING_ARM = arm
    try:
        yield
    finally:
        _EXPERIMENT_LISTING_ARM = previous


def selected_listing_arm() -> str | None:
    """The experiment's selected arm, or None when the experiment selects nothing."""
    return _EXPERIMENT_LISTING_ARM


def effective_listing_arm() -> str:
    """The arm the run uses: the experiment's when one is selected, else ``PRODUCTION_LISTING_ARM``."""
    return _EXPERIMENT_LISTING_ARM or PRODUCTION_LISTING_ARM


# ---------------------------------------------------------------------------
# Gathering (3.10, Gathering)
# ---------------------------------------------------------------------------

def planning_candidates(probabilities: Mapping[str, float],
                        threshold: float | None = None) -> tuple[str | None, tuple[str, ...]]:
    """``(confident, unsure)`` from a planning-command match's distribution over its options.

    ``confident`` is the option at or above the match threshold (``workflow_agent.PLANNING_COMMAND_MATCH_THRESHOLD``
    when None), or None. ``unsure`` are the other planning commands at or above ``UNSURE_CANDIDATE_FLOOR``,
    most probable first, ``none`` left out. Both go on the shortlist (3.10).
    """
    if threshold is None:
        from fastworkflow.workflow_agent import PLANNING_COMMAND_MATCH_THRESHOLD as threshold
    options = sorted(((name, float(p)) for name, p in probabilities.items() if name != NONE_OPTION),
                     key=lambda item: (-item[1], item[0]))
    confident = next((name for name, p in options if p >= threshold), None)
    unsure = tuple(name for name, p in options
                   if name != confident and p >= UNSURE_CANDIDATE_FLOOR)
    return confident, unsure


def _context_names(crd, context: str) -> list[str]:
    """The commands a context lists (execution and planning, its inherited ones too), ``wildcard`` left out."""
    names = [*crd.contexts.get(context, ()), *crd.get_planning_command_names(context)]
    return [name for name in names if name.split("/")[-1] != "wildcard"]


def _owner(qualified: str, crd, active: str) -> str:
    """The context whose section lists *qualified*: the active one when it lists it, else its prefix ('*' for none)."""
    if qualified in set(_context_names(crd, active)):
        return active
    return qualified.rpartition("/")[0] or "*"


def _mentioned_names(text: str, crd, contexts: Iterable[str]) -> set[str]:
    """Commands of *contexts* whose name the sub-task writes (exact names, as the exact-name lookup reads them)."""
    found: set[str] = set()
    for context in contexts:
        for qualified in _context_names(crd, context):
            leaf = qualified.split("/")[-1]
            if re.search(rf"(?<![\w]){re.escape(leaf)}(?![\w])", text):
                found.add(qualified)
    return found


def _reachable_contexts(folder: str, active: str) -> list[str]:
    """*active* and every context reachable from it by descend edges (``collect_descend_edges``), each once."""
    children: dict[str, list[str]] = {}
    for edge in collect_descend_edges(folder):
        children.setdefault(edge.source, []).append(edge.target)
    order, seen, queue = [active], {active}, deque([active])
    while queue:
        for target in children.get(queue.popleft(), ()):
            if target not in seen:
                seen.add(target)
                order.append(target)
                queue.append(target)
    return order


def gather_contexts(workflow) -> list[str]:
    """The contexts gathering visits: the lookup-path scopes, then every context reachable by descend edges."""
    active = workflow.current_command_context_name
    scopes = [scope.context_name for scope in exact_name_scopes(workflow)]
    return list(dict.fromkeys([*scopes, *_reachable_contexts(workflow.folderpath, active)]))


def gather(workflow, sub_task: str, planning_probabilities: Mapping[str, float] | None = None) -> frozenset[str]:
    """The candidates of a sub-task (3.10, Gathering): every scope on the lookup path and every reachable context,
    the commands whose names the sub-task writes, and the planning-command match's confident and unsure candidates.
    """
    folder = workflow.folderpath
    crd = fastworkflow.RoutingRegistry.get_definition(folder)
    found = _mentioned_names(sub_task, crd, gather_contexts(workflow))
    if planning_probabilities:
        confident, unsure = planning_candidates(planning_probabilities)
        found.update(name for name in (confident, *unsure) if name)
    return frozenset(found)


# ---------------------------------------------------------------------------
# Expansion (3.10, Expansion)
# ---------------------------------------------------------------------------

def _producers(resolution, qualified: str, known: set[str]) -> set[str]:
    """The commands a command's ``Input`` fields take their values from (its producer ``from_`` edges, 3.3)."""
    signature = resolution.signatures.get(qualified)
    if signature is None or not hasattr(signature, "Input"):
        return set()
    found: set[str] = set()
    for field_info in getattr(signature.Input, "model_fields", {}).values():
        extra = getattr(field_info, "json_schema_extra", None)
        if not isinstance(extra, dict):
            continue
        for ref in extra.get("available_from") or ():
            producer = str(ref).rpartition(".")[0]
            if producer in known:
                found.add(producer)
    return found


def _navigating(workflow, qualified_owner: str, active: str, edges) -> set[str]:
    """The descend-edge commands on the path from *active* to *qualified_owner* (``navigation_steps_to_context``)."""
    if qualified_owner == active:
        return set()
    try:
        path = plan_navigation(active, [qualified_owner], live_context_chain(workflow), edges).get(qualified_owner)
    except Exception:  # noqa: BLE001 - a navigation read never fails a listing; the line says no path
        logger.debug("navigation for the listing not computed", exc_info=True)
        return set()
    found: set[str] = set()
    here = active
    for step in path or ():
        match = next((e for e in edges if e.command == step.command and e.target == step.enters
                      and e.source == here), None) or next(
            (e for e in edges if e.command == step.command and e.target == step.enters), None)
        if match is not None:
            found.add(match.command if match.source in ("", "*") else f"{match.source}/{match.command}")
        here = step.enters
    return found


def expand(workflow, seeds: Iterable[str]) -> frozenset[str]:
    """The shortlist: *seeds*, each command's producers, the commands that navigate to its context, a planning
    command's callees and the active context's core commands; repeated until nothing new is added (3.10)."""
    from fastworkflow.workflow_agent import core_command_names

    folder = workflow.folderpath
    active = workflow.current_command_context_name
    crd = fastworkflow.RoutingRegistry.get_definition(folder)
    resolution = planning_commands.resolve_field_sources(folder)
    known: set[str] = set()
    for context in crd.contexts:
        known.update(_context_names(crd, context))
    edges = collect_descend_edges(folder)
    core = set(core_command_names())
    shortlist = {name for name in seeds}
    shortlist.update(name for name in _context_names(crd, active) if name in core)
    pending = deque(shortlist)
    while pending:
        current = pending.popleft()
        owner = _owner(current, crd, active)
        added = set(_producers(resolution, current, known))
        added |= _navigating(workflow, owner, active, edges)
        added |= set(resolution.callees.get(current, ()))
        for name in added:
            if name not in shortlist and name in known:
                shortlist.add(name)
                pending.append(name)
    return frozenset(shortlist)


def shortlist_for(workflow, sub_task: str, planning_probabilities: Mapping[str, float] | None = None) -> frozenset[str]:
    """Gathering by names only, then expansion (no heads read; the offline floor)."""
    return expand(workflow, gather(workflow, sub_task, planning_probabilities))


# ---------------------------------------------------------------------------
# Live gathering (3.9, 3.10): the trained heads of the gather contexts, read by the production path
# ---------------------------------------------------------------------------

def command_info_root_of(workflow) -> str:
    """The workflow's trained heads folder: ``___command_info``, one folder per context (``global`` for ``*``)."""
    return os.path.join(workflow.folderpath, "___command_info")


def head_answers(workflow, sub_task: str, command_info_root: str | None = None) -> dict[str, dict]:
    """Each gather context's trained head answers *sub_task*: the labels the run gets and the top-k labels and scores
    (``CommandRouter.predict_with_details``: local classifiers, no LLM call). A context with no trained head (no
    ``threshold.json`` etc. in its folder) answers nothing (3.9)."""
    from fastworkflow.model_pipeline_training import CommandRouter
    from fastworkflow.train.artifact_versioning import context_folder_name
    from fastworkflow.train.score_lookup import HEAD_MARKERS

    root = command_info_root or command_info_root_of(workflow)
    answers: dict[str, dict] = {}
    for context in gather_contexts(workflow):
        folder = os.path.join(root, context_folder_name(context))
        if not all(os.path.exists(os.path.join(folder, marker)) for marker in HEAD_MARKERS):   # models are folders
            continue
        labels, details = CommandRouter(folder).predict_with_details(sub_task)
        answers[context] = {
            "labels": [str(label) for label in labels],
            "topk_labels": [str(label) for label in details["topk_labels"]],
            "topk_scores": [float(score) for score in details["topk_scores"]],
        }
    return answers


def head_seeds(workflow, sub_task: str, command_info_root: str | None = None) -> tuple[frozenset[str], dict[str, float]]:
    """The commands the heads answer for *sub_task* (their labels, not ``parameter_value``/``wildcard``), and the
    planning commands among their top-k with the best score each got (the unsure candidates, 3.9)."""
    planning = frozenset(planning_commands.resolve_field_sources(workflow.folderpath).planning_commands)
    labels: set[str] = set()
    scores: dict[str, float] = {}
    for answer in head_answers(workflow, sub_task, command_info_root).values():
        labels.update(label for label in answer["labels"] if label not in NON_COMMAND_LABELS)
        for label, score in zip(answer["topk_labels"], answer["topk_scores"]):
            if label in planning:
                scores[label] = max(scores.get(label, 0.0), score)
    return frozenset(labels), scores


def live_gather(workflow, sub_task: str, planning_probabilities: Mapping[str, float] | None = None,
                command_info_root: str | None = None) -> frozenset[str]:
    """The seeds of a sub-task (3.10, Gathering) with the trained heads: the names the sub-task writes, the heads'
    command labels, and the planning candidates from the heads' planning top-k merged with *planning_probabilities*
    (a planning-command match's distribution, when the caller has one; the max per name)."""
    labels, head_planning = head_seeds(workflow, sub_task, command_info_root)
    merged = dict(head_planning)
    for name, probability in (planning_probabilities or {}).items():
        merged[name] = max(merged.get(name, 0.0), float(probability))
    crd = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    known = {name for context in crd.contexts for name in _context_names(crd, context)}
    seeds = set(gather(workflow, sub_task, merged or None))
    seeds.update(label for label in labels if label in known)
    return frozenset(seeds)


def shortlist_live(workflow, sub_task: str, planning_probabilities: Mapping[str, float] | None = None,
                   command_info_root: str | None = None) -> frozenset[str]:
    """The production shortlist of a sub-task: live gathering (heads and names), then expansion."""
    return expand(workflow, live_gather(workflow, sub_task, planning_probabilities, command_info_root))


# ---------------------------------------------------------------------------
# Listing (3.10, Listing)
# ---------------------------------------------------------------------------

def _one_line(workflow, active: str, context: str, on_path: bool, crd) -> str:
    """One context as one line, as the executor's list writes it (``get_other_contexts_text``)."""
    from fastworkflow.context_identity import context_class_for

    description = ""
    with contextlib.suppress(Exception):
        description = (inspect.getdoc(context_class_for(workflow, context)) or "").strip().split("\n")[0]
    if on_path:
        description = description or "its commands are on your lookup path"
        return (f"- {context}: {description.rstrip('.')}. Its commands are callable here by name; "
                "you do not enter it to run them.")
    if not description:
        added = sorted({name.split("/")[-1] for name in _context_names(crd, context)}
                       - {name.split("/")[-1] for name in _context_names(crd, active)})
        description = ("commands: " + ", ".join(added)) if added else ""
    steps = None
    with contextlib.suppress(Exception):
        steps = navigation_steps_to_context(workflow, active, context)
    return f"- {context}: {description.rstrip('.')}. To enter: {steps if steps else 'no declared navigation path'}"


def _global_line(crd, holding: set[str]) -> str:
    """The global scope as one line when no shortlisted command is global (3.9: global commands are callable from
    every context by name, so the run must see that they exist). Empty when the global scope is holding, or has none."""
    if "*" in holding:
        return ""
    # startup is not a command a sub-task's run calls: production's full listing never shows it, and the trainer
    # never labels it (model_pipeline_training.STARTUP_COMMAND). Only the global commands the agent can call are named.
    from fastworkflow.model_pipeline_training import STARTUP_COMMAND

    names = sorted({name for name in _context_names(crd, "*") if "/" not in name and name != STARTUP_COMMAND})
    if not names:
        return ""
    return ("Global commands (callable from anywhere by name, no navigation): "
            + ", ".join(names) + ".")


def _other_lines(workflow, active: str, crd, skip: set[str], on_path: set[str]) -> str:
    """One line per context not in *skip*, as the executor's list writes them. The global scope ``*`` is never a
    line there (production's list leaves it out; its commands are callable by name, 3.9)."""
    occupiable = CommandMetadataAPI._occupiable_context_names(workflow.folderpath)
    lines = [_one_line(workflow, active, context, context in on_path, crd)
             for context in sorted(crd.contexts)
             if context not in skip and context != "*"
             and (occupiable is None or context in occupiable or context in on_path)]
    if not lines:
        return ""
    return ("Other contexts (enter one to see and run its commands; a context whose commands are "
            "callable here needs no entering):\n" + "\n".join(lines))


def _active_section(workflow, names: Sequence[str] | None) -> str:
    folder = workflow.folderpath
    active = workflow.current_command_context_name
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    if names is None:
        return CommandMetadataAPI.get_command_display_text(
            subject_workflow_path=folder, cme_workflow_path=cme, active_context_name=active, with_templates=True)
    parts = [f"Commands available in the current context ({display_context_name(active)}), "
             "shortlisted for this sub-task:"]
    for qualified in names:
        if part := CommandMetadataAPI.get_command_display_text_for_command(
                subject_workflow_path=folder, cme_workflow_path=cme, active_context_name=active,
                qualified_command_name=qualified, with_template=True):
            parts.append(part)
    return "\n".join(parts)


def _scope_section(workflow, owner: str, qualified_names: Sequence[str]) -> str:
    """Commands of a scope on the lookup path, callable here by name (the production's section, 3.9)."""
    folder = workflow.folderpath
    active = workflow.current_command_context_name
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    where = "the global scope" if owner == "*" else f"the {owner} context"
    parts = [f"Commands available here from {where} (on your lookup path; run by name, no navigation):"]
    for qualified in qualified_names:
        if part := CommandMetadataAPI.format_callable_here_command(
                subject_workflow_path=folder, cme_workflow_path=cme, owning_context_name=owner,
                qualified_command_name=qualified, active_context_name=active):
            parts.append(part)
    return "\n".join(parts)


def _deferred_section(workflow, owner: str, qualified_names: Sequence[str]) -> str:
    """Commands of a context not yet entered, each with its navigation line (the production's section, 3.10)."""
    folder = workflow.folderpath
    active = workflow.current_command_context_name
    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    parts = [f"Commands available after entering the {owner} context (not callable until then):"]
    for qualified in qualified_names:
        if part := CommandMetadataAPI.format_deferred_planner_command(
                subject_workflow_path=folder, cme_workflow_path=cme, owning_context_name=owner,
                qualified_command_name=qualified, active_context_name=active, navigation_workflow=workflow):
            parts.append(part)
    return "\n".join(parts)


def _on_path(workflow) -> set[str]:
    return {scope.context_name for scope in exact_name_scopes(workflow)}


def command_shortlist_text(workflow, names: Iterable[str]) -> str:
    """The COMMAND_SHORTLIST listing: the shortlisted commands in full, a context with one of them as its section,
    every other context as one line."""
    folder = workflow.folderpath
    active = workflow.current_command_context_name
    crd = fastworkflow.RoutingRegistry.get_definition(folder)
    by_owner: dict[str, list[str]] = {}
    for name in sorted(names):
        by_owner.setdefault(_owner(name, crd, active), []).append(name)
    on_path = _on_path(workflow)
    sections: list[str] = []
    if active in by_owner:
        sections.append(_active_section(workflow, by_owner[active]))
    else:
        sections.append(_active_section(workflow, []))
    for owner in sorted(by_owner, key=lambda o: (o != active, o)):
        if owner == active:
            continue
        if owner in on_path:
            sections.append(_scope_section(workflow, owner, by_owner[owner]))
        else:
            sections.append(_deferred_section(workflow, owner, by_owner[owner]))
    skip = {active, *by_owner}
    if global_line := _global_line(crd, set(by_owner)):
        sections.append(global_line)
    if other := _other_lines(workflow, active, crd, skip, on_path):
        sections.append(other)
    return "\n\n".join(sections)


def run_listing_text(workflow, sub_task: str, *, arm: str | None = None,
                     planning_probabilities: Mapping[str, float] | None = None, full: str | None = None,
                     command_info_root: str | None = None) -> str:
    """The run's listing for *sub_task* (3.10, Listing).

    ``arm`` None is the selected arm, and FULL when none is selected (production). ``FULL`` gives *full* (the full
    listing the caller shows today; default ``planner_commands_text``). ``THRESHOLD`` gives *full* while it is at most
    ``FULL_LISTING_MAX_CHARS``, else the shortlist. ``COMMAND_SHORTLIST`` always gives the shortlist. The shortlist is
    gathered with the trained heads (``shortlist_live``).
    """
    from fastworkflow.workflow_agent import planner_commands_text

    arm = arm if arm is not None else effective_listing_arm()
    if arm not in ARMS:
        raise ValueError(f"unknown listing arm {arm!r}")
    if arm == COMMAND_SHORTLIST:
        return command_shortlist_text(
            workflow, shortlist_live(workflow, sub_task, planning_probabilities, command_info_root))
    if full is None:
        full = planner_commands_text(workflow)
    if arm == FULL or len(full) <= FULL_LISTING_MAX_CHARS:
        return full
    return command_shortlist_text(
        workflow, shortlist_live(workflow, sub_task, planning_probabilities, command_info_root))


# ---------------------------------------------------------------------------
# The argument check of a planning command the agent writes as a step (3.10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgumentCheck:
    """The outcome of checking a planned planning command's arguments: the guarded arguments, what the guards
    changed, and the refusal (None when the arguments pass)."""
    args: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    refusal: str | None = None

    @property
    def passed(self) -> bool:
        return self.refusal is None


def check_planned_arguments(command, raw: Mapping, sub_task: str,
                            earlier_outputs: Sequence[str] = ()) -> ArgumentCheck:
    """Check a planning command's arguments as the agent wrote them as a step (3.10, Arguments of a planned command).

    The request is the sub-task with the outputs of the turn's earlier steps, not the step text: a value an earlier
    step found is in the request and passes; a value in neither is refused (``check_arguments``'s ``in_request``
    rule for a required argument, or its default for an optional one).
    """
    request = "\n\n".join([sub_task, *earlier_outputs])
    try:
        args, notes = planning_commands.check_arguments(command, dict(raw), request)
    except planning_commands.PlanningCommandInputError as refused:
        return ArgumentCheck(refusal=str(refused))
    return ArgumentCheck(args=args, notes=tuple(notes))
