"""Score how a planning command is matched, four arms on one named test set (Phase 10 step 1).

EXPERIMENT TOOL. Four arms, one set of requests (``planning_test_set``), one scoring rule:

* ``heads_labelled`` (a): classifier heads with planning commands as labels. This is PRODUCTION
  (Phase 10 step 1 decision): a trained copy built with the default layout, or under
  ``model_pipeline_training.experiment_planning_heads("labelled")`` inside the sparse arm.
* ``heads_not_here`` (b): the baseline layout. Planning commands only in the sparse "not here"
  class: a trained copy built under ``experiment_planning_heads("not_here")`` inside the sparse arm.
  This is the BASELINE every report compares against.
* ``jev`` (c): Jev, through the real match path (``workflow_agent.match_planning_command``). LIVE.
* ``agent_plan`` (d): the agent choosing as it plans, with no matcher. LIVE. Measured as follows
  (decision, Phase 10 step 1c; evidence checked in the code, file:line as of this change):

  - The planner is the production text planner: the same signature and instructions
    (``workflow_agent.py:2215-2218`` ``TaskPlannerTextSignature``, ``PLANNER_INSTRUCTIONS``
    at ``workflow_agent.py:1696``), run as ``dspy.ChainOfThought`` under ``CommandsSystemPreludeAdapter``
    (``workflow_agent.py:43``), with the planner's listing as ``available_commands``. Its output is
    free text: ``next_steps`` names no command field, so the command is read from the steps.
  - A plan's steps are read the way production reads them: ``finish_check.parse_plan``
    (``observation_offloading/finish_check.py:161-190``; ``_classify`` at ``:193``, which takes the
    first declared command a step names), with the workflow's command names and the core (navigation)
    names as ``workflow_execution_context._workflow_command_names`` and
    ``workflow_agent.core_command_names`` give them (``workflow_execution_context.py:2743-2757``,
    ``workflow_agent.py:365-377``). Production's phase reader is the same call
    (``planning_phases.py:1040-1048``, ``:1547``, ``:1983``).
  - The command the agent is told to run first is the FIRST command step of its plan
    (``chosen_command``): the agent executes its plan in step order (``workflow_agent.py:2380-2385``
    "execute these next steps"). Its tool call is that step's command. The ReAct agent's own call
    needs an agent model call, so offline the plan step stands in for it; this is the planner's first
    command step, not a measured tool call.
  - Scoring is the same rule as the other arms (``case_verdict``): the chosen command is the top
    label. Plan length (numbered steps), the command steps, and the raw planner output are stored per
    case under ``plan``.
  - Design 2.44 (``docs/unified_command_architecture/2. Discussion.md:2502-2577``) gives three plans:
    the one call (the fast path), the call with plain steps before or after it for the parts the
    command does not cover, and an adapted copy. ``classify_plan`` reads them as ``single_call``,
    ``call_plus_steps``, ``no_call`` (the command is not one of the steps: an adapted copy or another
    command) and ``no_plan``. ``plan_2_44_summary`` reports covered requests (planning-command
    requests, whose whole sub-task the command covers) as single-call rates. Partly covered requests
    need a label the test set does not carry (``partly_covered`` with ``extra_commands``); with none,
    that count is reported as not measured.

(a) and (b) are offline: they run the trained heads' ``CommandRouter`` on a trained copy and read no
provider. (c) and (d) call a model provider. They refuse to run without ``--allow-live``, and
``--dry-run`` prints their call and token estimate without calling anything.

Per case the top label decides the verdict (``score_not_here`` conventions): a non-routable top label
(``wildcard``, ``parameter_value``) or no label is a decline; any other label is an accept.

Headline metrics, over the HELDOUT split only (training-split seeds are reported as in-sample):

* planning requests routed to their planning command (top label is that command);
* callee requests routed to their callee (not the planning command);
* wrong accepts: any request routed to a label other than its expected command.

Each rate has a 95% Wilson interval. Arms are paired on the same cases: an arm's rate minus the
baseline's has a paired Wald interval and an exact McNemar p-value.

Subcommands::

    python -m fastworkflow.train.score_planning_match predict --workflow W --arm heads_labelled --out R
        [--test-set T | --handwritten H] [--train-seconds S]
    python -m fastworkflow.train.score_planning_match live --arm jev --workflow W [--per-stratum 5]
        [--seed 0] [--dry-run] [--allow-live] --out R --run-id ID
        (a live run is refused if R exists or a lock for (arm, ID) is held; the lock is removed at the end)
    python -m fastworkflow.train.score_planning_match score --results R [--out S]
    python -m fastworkflow.train.score_planning_match report --baseline R0 --arm NAME=R [--arm ...] [--out M]

The recommended bars (the epic's notes) are shown in ``report`` as guidance. They are not gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import collections
from typing import Any, Mapping, Optional, Sequence

from fastworkflow.nlu_labels import is_non_routable, label_of
from fastworkflow.train import planning_test_set as pts
from fastworkflow.train import score_not_here as scorer

SCHEMA_VERSION = 1
Z_95 = scorer.Z_95

ARM_HEADS_LABELLED = "heads_labelled"
ARM_HEADS_NOT_HERE = "heads_not_here"
ARM_JEV = "jev"
ARM_AGENT_PLAN = "agent_plan"
ARMS: tuple[str, ...] = (ARM_HEADS_LABELLED, ARM_HEADS_NOT_HERE, ARM_JEV, ARM_AGENT_PLAN)
OFFLINE_ARMS: tuple[str, ...] = (ARM_HEADS_LABELLED, ARM_HEADS_NOT_HERE)
LIVE_ARMS: tuple[str, ...] = (ARM_JEV, ARM_AGENT_PLAN)
BASELINE_ARM = ARM_HEADS_NOT_HERE

#: The planning-head layout each offline arm needs in its trained copy's run metadata.
REQUIRED_PLANNING_HEADS = {ARM_HEADS_LABELLED: "labelled", ARM_HEADS_NOT_HERE: "not_here"}

#: The epic's recommended bars (guidance for the report, not gates; the owner decides).
BAR_PLANNING_LOWER_MIN = 0.0    # planning routed: lower bound of (arm - baseline) >= 0
BAR_CALLEE_LOWER_MIN = -0.02    # callee routed: lower bound of (arm - baseline) >= -0.02
BAR_WRONG_ACCEPT_UPPER_MAX = 0.02  # wrong accepts: upper bound of (arm - baseline) <= 0.02

#: Rough token sizes for the live estimate (characters per token). Printed as an estimate.
#: The trainer's export under an experiment arm (``model_pipeline_training.HOLDOUT_EXPORT_FILENAME``).
HOLDOUT_FILENAME = "not_here_holdout.json"
CHARS_PER_TOKEN = 4
# The planner runs as dspy.ChainOfThought (reasoning plus next_steps), so its output is larger than the
# bare plan: 400 is a rough allowance for both, not a measured figure.
AGENT_OUTPUT_TOKENS = 400
JEV_OUTPUT_TOKENS = 20
PER_STRATUM_LIBRARY = 5
PER_STRATUM_APP = 3


class ArmMismatchError(ValueError):
    """An offline arm was named against a trained copy built under a different planning-head layout."""


# ---------------------------------------------------------------- pure helpers

def _key(case: Mapping[str, Any]) -> tuple:
    return (case["head"], case["utterance"], case["expected"], case["kind"], case["split"])


def stratified_sample(cases: Sequence[Mapping[str, Any]], per_stratum: int, seed: int = 0) -> list[dict]:
    """Up to *per_stratum* cases per stratum, chosen deterministically (design 2.44, epic notes).

    A stratum is one expected command: a planning command's requests and each callee's requests are
    each one stratum. Within a stratum the cases are ranked by a SHA-256 of (seed, utterance,
    expected), so the choice is the same in every process and changes with the seed only. The
    result keeps the input order. Pure: it reads nothing but its arguments.
    """
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1")
    groups: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(case["expected"], []).append(index)
    chosen: set[int] = set()
    for expected in sorted(groups):
        ranked = sorted(
            groups[expected],
            key=lambda i: hashlib.sha256(
                f"{seed}\x00{cases[i]['utterance']}\x00{expected}".encode("utf-8")).hexdigest(),
        )
        chosen.update(ranked[:per_stratum])
    return [dict(cases[i]) for i in sorted(chosen)]


def case_verdict(top_label: Optional[str], expected: str) -> dict[str, bool]:
    """``{declined, routed, wrong_accept}`` for one case whose top label is *top_label*."""
    top = str(top_label or "")
    declined = (not top) or is_non_routable(top)
    routed = (not declined) and label_of(top) == label_of(expected)
    return {"declined": declined, "routed": routed, "wrong_accept": (not declined) and not routed}


#: Design 2.44's three plans, as ``classify_plan`` reads them (the expected command is the case's).
PLAN_SINGLE_CALL = "single_call"
PLAN_CALL_PLUS_STEPS = "call_plus_steps"
PLAN_NO_CALL = "no_call"
PLAN_NO_PLAN = "no_plan"
PLAN_CLASSES: tuple[str, ...] = (PLAN_SINGLE_CALL, PLAN_CALL_PLUS_STEPS, PLAN_NO_CALL, PLAN_NO_PLAN)


def _qualified(plain: str, names: Sequence[str], step_text: str = "") -> str:
    """The qualified name (``Head/command``) of a step's plain command name, from *names*.

    ``finish_check`` reads a step's command by its plain name. A plain name shared by two
    commands is resolved by the step's text (the qualified name that appears in it), else the first.
    """
    from fastworkflow.observation_offloading.finish_check import plain_command_name

    matches = [n for n in names if plain_command_name(n).lower() == plain.lower()]
    if not matches:
        return plain
    text = step_text.lower()
    named = [n for n in matches if n.lower() in text]
    return named[0] if named else matches[0]


def read_plan(raw_plan: str, names: Sequence[str], core: Sequence[str] = ()) -> dict[str, Any]:
    """The plan a planner wrote, read as production reads it (``finish_check.parse_plan``).

    Returns the raw text, one row per numbered step (number, text, kind, qualified command),
    the plan length (numbered steps), the command steps in order, and ``chosen_command``: the
    first command step's qualified name, the command the agent is told to run first ("" when
    no step names a command). Pure: it reads nothing but its arguments.
    """
    from fastworkflow.observation_offloading import finish_check

    text = str(raw_plan or "")
    parsed = finish_check.parse_plan(text, list(names), list(core))
    all_names = list(names) + list(core)
    rows = []
    for step in parsed:
        command = _qualified(step.command, all_names, step.text) if step.command else ""
        rows.append({"number": step.number, "text": step.text, "kind": step.kind, "command": command})
    command_steps = [r["command"] for r in rows if r["kind"] == finish_check.KIND_COMMAND and r["command"]]
    return {"raw": text, "steps": rows, "plan_length": len(rows), "command_steps": command_steps,
            "chosen_command": command_steps[0] if command_steps else ""}


def classify_plan(expected: str, plan: Mapping[str, Any]) -> str:
    """Design 2.44's class of *plan* for a request whose command is *expected*.

    ``single_call``: the plan is one step, the call to *expected*. ``call_plus_steps``: the call to
    *expected* and other steps. ``no_call``: *expected* is not one of its command steps. ``no_plan``:
    no steps.
    """
    if plan["plan_length"] == 0:
        return PLAN_NO_PLAN
    if expected not in plan["command_steps"]:
        return PLAN_NO_CALL
    if plan["plan_length"] == 1:
        return PLAN_SINGLE_CALL
    return PLAN_CALL_PLUS_STEPS


def right_extra_steps(case: Mapping[str, Any]) -> Optional[bool]:
    """For a partly covered case with labelled ``extra_commands``: whether the plan's other command
    steps are exactly those commands, each present. None when the case carries no such label."""
    extras = case.get("extra_commands")
    if extras is None or "plan" not in case:
        return None
    others = [c for c in case["plan"]["command_steps"] if c != case["expected"]]
    wanted = set(extras)
    return set(others) == wanted and bool(wanted)


def plan_2_44_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Design 2.44 on heldout planning-command requests of an agent_plan run (see the module docstring).

    *covered*: planning-command requests without a ``partly_covered`` label; the rate that the plan is
    the single call. *partly_covered*: labelled requests; the rate that the plan is the call with the
    right extra steps. Unlabelled partly covered requests are not measured and say so.
    """
    planned = [c for c in cases if _eligible(c) and c["kind"] == pts.KIND_PLANNING and "plan" in c]
    covered = [c for c in planned if not c.get("partly_covered")]
    partly = [c for c in planned if c.get("partly_covered")]
    classes: dict[str, int] = {k: 0 for k in PLAN_CLASSES}
    per_target: dict[str, dict[str, int]] = {}
    for case in covered:
        cls = classify_plan(case["expected"], case["plan"])
        classes[cls] += 1
        block = per_target.setdefault(case["expected"], {k: 0 for k in PLAN_CLASSES})
        block[cls] += 1
    single = classes[PLAN_SINGLE_CALL]
    labelled = [c for c in partly if right_extra_steps(c) is not None]
    right = sum(1 for c in labelled if right_extra_steps(c))
    partly_classes = {k: 0 for k in PLAN_CLASSES}
    for case in partly:
        partly_classes[classify_plan(case["expected"], case["plan"])] += 1
    return {
        "covered": {"n": len(covered), "single_call": scorer._rate_block(single, len(covered)),
                    "classes": classes, "per_target": dict(sorted(per_target.items()))},
        "partly_covered": {
            "n": len(partly), "labelled": len(labelled),
            "call_with_right_extra_steps": scorer._rate_block(right, len(labelled)),
            "classes": partly_classes,
            "note": ("not measured: no partly covered requests are labelled in this set"
                     if not partly else "labelled extra_commands only"),
        },
    }


#: Outcomes of a plan read for ALL of its steps (``plan_reach``), beside the first-step measure.
REACH_REACHES = "reaches"          # the plan names the expected command in some step
REACH_DELEGATES = "delegates"      # (callee requests) it names a planning command that calls the expected callee
REACH_MISROUTE = "misroute"        # it names commands, none of them the expected one or a covering planning command
REACH_NO_COMMAND = "no_command"    # it names no command (prose, a question, or no numbered step)
REACH_OUTCOMES: tuple[str, ...] = (REACH_REACHES, REACH_DELEGATES, REACH_MISROUTE, REACH_NO_COMMAND)

#: A no-command plan that asks the user something (a clarifying question) rather than describing steps.
_CLARIFY_RE = re.compile(r"\?|\b(clarify|ask|confirm|which|what|please|specify|need to know)\b", re.IGNORECASE)


def coverage_map(test_set: Mapping[str, Any]) -> dict[str, set[str]]:
    """callee -> the planning commands whose direct callees include it, from the test set's targets."""
    covers: dict[str, set[str]] = {}
    for name, block in test_set.get("targets", {}).items():
        if block.get("kind") != pts.KIND_PLANNING:
            continue
        for callee in block.get("callees", []):
            covers.setdefault(callee, set()).add(name)
    return covers


def _named_commands(plan: Mapping[str, Any]) -> list[str]:
    """The commands a plan names in ANY step that may run them: command and conditional steps
    (a conditional step still names its command). Skipped steps (negations, answers) name nothing
    that runs. A plan without step rows (Jev's one choice) reads its command steps."""
    from fastworkflow.observation_offloading import finish_check

    if plan.get("steps"):
        return [s["command"] for s in plan["steps"]
                if s["kind"] in (finish_check.KIND_COMMAND, finish_check.KIND_CONDITIONAL) and s["command"]]
    return list(plan["command_steps"])


def plan_reach(case: Mapping[str, Any], covers: Mapping[str, set[str]]) -> dict[str, Any]:
    """The reach measures for one case with a plan (all steps read, not only the first).

    ``reach``: one of REACH_OUTCOMES. For a planning request, ``reaches`` means the plan names its planning
    command in some step. For a callee request, ``reaches`` means it names the callee in some step; a plan
    that names no callee but a planning command covering it is ``delegates``. ``uses_planning`` is the
    planning-request measure (the plan names the command, any step). ``single_call`` is design 2.44's fast
    path (planning requests only). ``clarify`` (no-command plans only): a clarifying question, not a description.
    """
    plan = case["plan"]
    expected = case["expected"]
    named = _named_commands(plan)
    reaches = expected in named
    out: dict[str, Any] = {"uses_planning": None, "single_call": None}
    if case["kind"] == pts.KIND_PLANNING:
        out["uses_planning"] = reaches
        out["single_call"] = classify_plan(expected, plan) == PLAN_SINGLE_CALL
    if reaches:
        reach = REACH_REACHES
    elif case["kind"] == pts.KIND_CALLEE and any(c in covers.get(expected, set()) for c in named):
        reach = REACH_DELEGATES
    elif named:
        reach = REACH_MISROUTE
    else:
        reach = REACH_NO_COMMAND
    out["reach"] = reach
    out["clarify"] = None
    if reach == REACH_NO_COMMAND:
        out["clarify"] = bool(_CLARIFY_RE.search(plan.get("raw", "") or ""))
    return out


def jev_as_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    """Jev's choice as a one-step plan: its one planning command, or none (for the same measures)."""
    top = case.get("top_label") or ""
    return {"raw": "", "plan_length": 1 if top else 0, "command_steps": [top] if top else [],
            "steps": [], "chosen_command": top}


def reach_summary(cases: Sequence[Mapping[str, Any]], covers: Mapping[str, set[str]]) -> dict[str, Any]:
    """The reach measures over *cases* (each with a ``plan``), by kind, with per-command counts."""
    planning = [c for c in cases if c["kind"] == pts.KIND_PLANNING]
    callee = [c for c in cases if c["kind"] == pts.KIND_CALLEE]
    per_cmd: dict[str, collections.Counter] = {}
    rows = []
    for case in cases:
        measured = plan_reach(case, covers)
        rows.append(measured)
        per_cmd.setdefault(case["expected"], collections.Counter())[measured["reach"]] += 1
    p_rows = [r for c, r in zip(cases, rows) if c["kind"] == pts.KIND_PLANNING]
    c_rows = [r for c, r in zip(cases, rows) if c["kind"] == pts.KIND_CALLEE]
    counts = lambda rs: {o: sum(1 for r in rs if r["reach"] == o) for o in REACH_OUTCOMES}  # noqa: E731
    no_cmd = [r for r in rows if r["reach"] == REACH_NO_COMMAND]
    return {
        "planning": {"n": len(planning),
                     "uses_planning": scorer._rate_block(sum(1 for r in p_rows if r["uses_planning"]), len(planning)),
                     "single_call": scorer._rate_block(sum(1 for r in p_rows if r["single_call"]), len(planning)),
                     "outcomes": counts(p_rows)},
        "callee": {"n": len(callee),
                   "reaches": scorer._rate_block(sum(1 for r in c_rows if r["reach"] == REACH_REACHES), len(callee)),
                   "delegates": sum(1 for r in c_rows if r["reach"] == REACH_DELEGATES),
                   "outcomes": counts(c_rows)},
        "no_command": {"n": len(no_cmd), "clarifying_questions": sum(1 for r in no_cmd if r["clarify"]),
                       "descriptions": sum(1 for r in no_cmd if r["clarify"] is False)},
        "per_command": {k: dict(sorted(v.items())) for k, v in sorted(per_cmd.items())},
    }


# ---------------------------------------------------------------- lever 2: the agent gets the head's candidates

#: The hint block's first line (exact text, documented for the record). Lever 2 hands an unsure head's real
#: candidates to the sub-task's planner as this block, appended to the request; nothing else changes.
LEVER2_HEADER = "Intent detection suggests these commands for this sub-task (it is unsure, most likely first):"


def lever2_hint(candidates: Sequence[str], planning_templates: Mapping[str, Sequence[str]]) -> str:
    """The hint block for one sub-task. Format, one line per candidate, numbered in the order given::

        Intent detection suggests these commands for this sub-task (it is unsure, most likely first):
        1. Library/overdue_summary [planning command; template: step 1; step 2]
        2. Library/count_overdue [command]

    A planning command shows its template (the steps ``resolve_field_sources`` reads, joined by "; "), as
    the planner's own listing shows it. An empty candidate list gives "" (no hint, as in the r2 arm).
    """
    if not candidates:
        return ""
    lines = [LEVER2_HEADER]
    for index, name in enumerate(candidates, 1):
        if name in planning_templates:
            steps = "; ".join(str(step) for step in planning_templates[name]) or "(no template)"
            lines.append(f"{index}. {name} [planning command; template: {steps}]")
        else:
            lines.append(f"{index}. {name} [command]")
    return "\n".join(lines)


def _is_unsure_or_declined(head: Mapping[str, Any]) -> bool:
    """A head the lever-2 arm sends to the agent: not confident, or its top label is not a real command."""
    return (not head["confident"]) or not (bool(head["top_label"]) and not is_non_routable(head["top_label"]))


def lever2_cases(head_cases: Sequence[Mapping[str, Any]], analysis_keys: set, jev_keys: set) -> list[dict]:
    """The lever-2 cases: heads_labelled's stored cases on the shared analysis set (also in the jev sample)
    whose head is unsure or says 'not here'. Confident heads are not re-run."""
    out = []
    for case in head_cases:
        key = _key(case)
        if key in analysis_keys and key in jev_keys and _is_unsure_or_declined(case):
            out.append(dict(case))
    return out


def lever2_candidates(head: Mapping[str, Any]) -> list[str]:
    """The head's real top candidates (its topk, in the head's order; its top label first when real)."""
    names = []
    for label in [head.get("top_label")] + list(head.get("topk", [])):
        if label and not is_non_routable(label) and label not in names:
            names.append(label)
    return names


def paired_indicator_difference(arm_bits: Sequence[int], base_bits: Sequence[int]) -> dict[str, Any]:
    """``arm - base`` on one 0/1 indicator per case: rate difference, paired Wald 95% interval, McNemar.

    The formula is the one ``not_here_step1.paired_difference`` uses (the paired Wald interval over
    the discordant pairs). A positive difference means the arm routes more of these cases.
    """
    if len(arm_bits) != len(base_bits):
        raise ValueError("the arms score different numbers of cases")
    n = len(arm_bits)
    if n == 0:
        return {"n": 0, "arm_rate": None, "base_rate": None, "difference": None,
                "lower": None, "upper": None, "only_arm": 0, "only_base": 0, "p_two_sided": 1.0}
    only_arm = sum(1 for x, y in zip(arm_bits, base_bits) if x == 1 and y == 0)
    only_base = sum(1 for x, y in zip(arm_bits, base_bits) if x == 0 and y == 1)
    difference = (only_arm - only_base) / n
    variance = max(0.0, (only_arm + only_base - (only_arm - only_base) ** 2 / n) / (n * n))
    half = Z_95 * math.sqrt(variance)
    return {
        "n": n,
        "arm_rate": sum(arm_bits) / n,
        "base_rate": sum(base_bits) / n,
        "difference": difference,
        "lower": difference - half,
        "upper": difference + half,
        "only_arm": only_arm,
        "only_base": only_base,
        "p_two_sided": scorer.mcnemar_exact(only_arm, only_base),
    }


def _eligible(case: Mapping[str, Any]) -> bool:
    """A heldout case that no head was trained on (the trained-on flag is set by ``predict``)."""
    return case["split"] == pts.SPLIT_HELDOUT and not case.get("trained_on", False)


def _metric_bits(case: Mapping[str, Any]) -> dict[str, Optional[int]]:
    """The indicators one case contributes, ``None`` where the metric does not apply."""
    verdict = case_verdict(case.get("top_label"), case["expected"])
    return {
        "planning_routed": int(verdict["routed"]) if case["kind"] == pts.KIND_PLANNING else None,
        "callee_routed": int(verdict["routed"]) if case["kind"] == pts.KIND_CALLEE else None,
        "wrong_accepts": int(verdict["wrong_accept"]),
    }


METRICS = ("planning_routed", "callee_routed", "wrong_accepts")


def summarize(results: Mapping[str, Any]) -> dict[str, Any]:
    """Headline rates (heldout, not trained on), in-sample rates, and per-target routing for one arm."""
    cases = results["cases"]
    heldout = [c for c in cases if _eligible(c)]
    in_sample = [c for c in cases if c["split"] == pts.SPLIT_TRAIN]
    out: dict[str, Any] = {"schema": SCHEMA_VERSION, "arm": results.get("arm"),
                           "heldout_n": len(heldout), "in_sample_n": len(in_sample),
                           "trained_on_excluded": sum(1 for c in cases
                                                      if c["split"] == pts.SPLIT_HELDOUT and c.get("trained_on")),
                           "training_rows": results.get("training_rows") or {},
                           "train_seconds": results.get("train_seconds")}
    for metric in METRICS:
        bits = [b for b in (_metric_bits(c)[metric] for c in heldout) if b is not None]
        out[metric] = scorer._rate_block(sum(bits), len(bits))
        seeds = [b for b in (_metric_bits(c)[metric] for c in in_sample) if b is not None]
        out[f"in_sample_{metric}"] = scorer._rate_block(sum(seeds), len(seeds))
    per_target: dict[str, dict[str, int]] = {}
    for case in heldout:
        block = per_target.setdefault(case["expected"], {"n": 0, "routed": 0, "declined": 0, "wrong_accept": 0})
        verdict = case_verdict(case.get("top_label"), case["expected"])
        block["n"] += 1
        block["routed"] += int(verdict["routed"])
        block["declined"] += int(verdict["declined"])
        block["wrong_accept"] += int(verdict["wrong_accept"])
    out["per_target"] = dict(sorted(per_target.items()))
    if any("plan" in c for c in cases):
        out["plan_2_44"] = plan_2_44_summary(cases)
    return out


def compare_arms(arm: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """Paired comparison of *arm* against *base* on the cases both score (same cases, same order).

    A case is compared only when it is heldout and that no arm trained on it, so both arms are
    scored on exactly the same cases.
    """
    arm_cases, base_cases = arm["cases"], base["cases"]
    if [_key(c) for c in arm_cases] != [_key(c) for c in base_cases]:
        raise ValueError("the two results do not score the same cases in the same order")
    pairs = [(a, b) for a, b in zip(arm_cases, base_cases)
             if _eligible(a) and _eligible(b)]
    comparison: dict[str, Any] = {"schema": SCHEMA_VERSION, "arm": arm.get("arm"),
                                  "baseline": base.get("arm"), "paired_n": len(pairs), "metrics": {}}
    for metric in METRICS:
        arm_bits, base_bits = [], []
        for a, b in pairs:
            x, y = _metric_bits(a)[metric], _metric_bits(b)[metric]
            if x is None or y is None:
                continue
            arm_bits.append(x)
            base_bits.append(y)
        comparison["metrics"][metric] = paired_indicator_difference(arm_bits, base_bits)
    return comparison


def guidance(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """The epic's recommended bars against the baseline, as guidance: each bar's verdict and its numbers."""
    m = comparison["metrics"]
    planning, callee, accept = m["planning_routed"], m["callee_routed"], m["wrong_accepts"]
    checks = {
        "planning_routed_lower_at_least_0": planning["lower"] is not None and planning["lower"] >= BAR_PLANNING_LOWER_MIN,
        "callee_routed_lower_at_least_minus_2pts": callee["lower"] is not None and callee["lower"] >= BAR_CALLEE_LOWER_MIN,
        "wrong_accepts_upper_at_most_2pts": accept["upper"] is not None and accept["upper"] <= BAR_WRONG_ACCEPT_UPPER_MAX,
    }
    return {"checks": checks, "all_met": all(checks.values()),
            "note": "guidance only: the owner decides (epic fix-3rsq.10 owner policy)",
            "covered_plan_single_call": "live arms only (10.3): not measured offline"}


def report(baseline: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Every arm summarized and compared against *baseline*, with the guidance bars."""
    rows = {}
    base_summary = summarize(baseline)
    rows[baseline.get("arm") or BASELINE_ARM] = {"summary": base_summary, "comparison": None, "guidance": None}
    for name, results in arms.items():
        comparison = compare_arms(results, baseline)
        rows[name] = {"summary": summarize(results), "comparison": comparison, "guidance": guidance(comparison)}
    return {"schema": SCHEMA_VERSION, "baseline": baseline.get("arm") or BASELINE_ARM, "arms": rows}


def _fmt_rate(block: Mapping[str, Any]) -> str:
    if not block or not block["trials"]:
        return "n/a"
    lo, hi = block["wilson95"]
    return f"{block['count']}/{block['trials']} = {100 * block['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def _fmt_diff(block: Mapping[str, Any]) -> str:
    if block["n"] == 0:
        return "n/a"
    return (f"{100 * block['difference']:+.1f} pts [{100 * block['lower']:+.1f}, {100 * block['upper']:+.1f}] "
            f"only-arm {block['only_arm']} only-base {block['only_base']} p={block['p_two_sided']:.3g}")


def format_report(doc: Mapping[str, Any]) -> str:
    lines = [f"Planning-command match, arms against baseline {doc['baseline']} "
             "(heldout cases, Wilson 95% and paired 95% intervals)", ""]
    for name, row in doc["arms"].items():
        summary = row["summary"]
        lines.append(f"[{name}] heldout={summary['heldout_n']} in-sample={summary['in_sample_n']} "
                     f"trained-on excluded={summary['trained_on_excluded']} "
                     f"training rows={sum(summary['training_rows'].values()) if summary['training_rows'] else 'n/a'} "
                     f"train seconds={summary['train_seconds'] if summary['train_seconds'] is not None else 'n/a'}")
        lines.append(f"  planning routed: {_fmt_rate(summary['planning_routed'])}")
        lines.append(f"  callee routed:   {_fmt_rate(summary['callee_routed'])}")
        lines.append(f"  wrong accepts:   {_fmt_rate(summary['wrong_accepts'])}")
        if row["comparison"]:
            for metric in METRICS:
                lines.append(f"  vs baseline, {metric}: {_fmt_diff(row['comparison']['metrics'][metric])}")
            for check, passed in row["guidance"]["checks"].items():
                lines.append(f"  bar {check}: {'met' if passed else 'not met'}")
            lines.append(f"  ({row['guidance']['note']})")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------- offline: trained copies

def check_planning_arm(workflow_folderpath: str, arm: str) -> None:
    """Refuse an offline arm whose trained copy was built under a different planning-head layout."""
    from fastworkflow.train.determinism import ProvenanceRecorder

    if arm not in OFFLINE_ARMS:
        raise ValueError(f"{arm!r} is not an offline arm; offline arms are {list(OFFLINE_ARMS)}")
    recorded = ProvenanceRecorder.load_run_metadata(workflow_folderpath).get("planning_heads")
    wanted = REQUIRED_PLANNING_HEADS[arm]
    if recorded != wanted:
        raise ArmMismatchError(
            f"{workflow_folderpath}: arm {arm!r} needs planning_heads={wanted!r}, but the trained copy "
            f"records {recorded!r} (training_provenance.json). Train the copy under the matching layout.")


def _leak_flags(workflow_folderpath: str, cases: list[dict]) -> None:
    """Mark each case whose utterance was a training row of its head (the trainer's own export)."""
    path = os.path.join(workflow_folderpath, "___command_info", HOLDOUT_FILENAME)
    if not os.path.isfile(path):
        for case in cases:
            case["trained_on"] = False
        return
    heads = scorer._load_json(path).get("heads", {})
    for case in cases:
        head = heads.get(case["head"], {})
        case["trained_on"] = case["utterance"] in set(head.get("training_utterances", []))


def predict_arm(workflow_folderpath: str, test_set: Mapping[str, Any], arm: str,
                train_seconds: Optional[float] = None) -> dict[str, Any]:
    """Run every case of *test_set* through its head of a trained copy. Offline: local classifiers only."""
    check_planning_arm(workflow_folderpath, arm)
    cases = [dict(c) for c in test_set["cases"]]
    results = scorer.predict_cases(workflow_folderpath, cases, None)
    results["arm"] = arm
    results["train_seconds"] = train_seconds
    _leak_flags(workflow_folderpath, results["cases"])
    return results


# ---------------------------------------------------------------- live: (c) and (d)

def estimate(arm: str, cases: Sequence[Mapping[str, Any]], workflow_folderpath: str) -> dict[str, Any]:
    """Calls and rough tokens a live run of *arm* over *cases* would make. Reads the workflow; calls nothing."""
    listing = _listing_chars(workflow_folderpath)
    if arm == ARM_JEV:
        # The matcher's job-type text, rebuilt from the read-only loader: workflow_agent's own
        # listing goes through load_planning_commands, which writes the workflow's derived caches.
        job_types = _job_types_by_scope(workflow_folderpath)
        calls = 0
        in_chars = 0
        for case in cases:
            for scope in dict.fromkeys((case["head"], "")):
                text = job_types.get(scope, "")
                if not text:
                    continue
                calls += 1
                in_chars += len(text) + len(case["utterance"]) + 300
        out_tokens = calls * JEV_OUTPUT_TOKENS
        note = "one matcher call per case per scope that has planning commands (case scope, then root)"
    elif arm == ARM_AGENT_PLAN:
        from fastworkflow.workflow_agent import PLANNER_INSTRUCTIONS

        calls = len(cases)
        in_chars = sum(len(PLANNER_INSTRUCTIONS) + listing + len(c["utterance"]) for c in cases)
        out_tokens = calls * AGENT_OUTPUT_TOKENS
        note = "one planner call per case over the planner's listing (rough: listing size estimated)"
    else:
        raise ValueError(f"{arm!r} is not a live arm; live arms are {list(LIVE_ARMS)}")
    in_tokens = in_chars // CHARS_PER_TOKEN
    return {"arm": arm, "cases": len(cases), "calls": calls, "input_tokens": in_tokens,
            "output_tokens": out_tokens, "note": note,
            "token_estimate_accuracy": "rough: characters / 4"
            + ("; the planner listing size is estimated" if arm == ARM_AGENT_PLAN else "")}


def _planning_templates(workflow_folderpath: str) -> dict[str, tuple]:
    """Each planning command's template steps (what the planner's listing shows), by qualified name."""
    from fastworkflow.planning_commands import resolve_field_sources

    from fastworkflow.planning_commands import visible_planning_commands

    return {name: tuple(cmd.template) for name, cmd in visible_planning_commands(workflow_folderpath).items()}


def lever2_rows(head_cases: Sequence[Mapping[str, Any]], analysis_keys: set, jev_keys: set,
                workflow_folderpath: str) -> list[dict]:
    """The lever-2 request rows: the selected cases, each with its head's candidates and its hint block."""
    templates = _planning_templates(workflow_folderpath)
    rows = []
    for case in lever2_cases(head_cases, analysis_keys, jev_keys):
        candidates = lever2_candidates(case)
        rows.append({"expected": case["expected"], "utterance": case["utterance"], "kind": case["kind"],
                     "head": case["head"], "related": case["related"], "split": case["split"],
                     "source": case["source"], "head_top_label": case["top_label"],
                     "head_confident": case["confident"], "head_topk": list(case.get("topk", [])),
                     "candidates": candidates, "hint": lever2_hint(candidates, templates)})
    return rows


def run_lever2(rows: Sequence[Mapping[str, Any]], workflow_folderpath: str) -> dict[str, Any]:
    """The r2 agent plan with the hint block appended to each request. LIVE: only after --allow-live."""
    plans = _agent_plans(rows, workflow_folderpath, hints=[r["hint"] for r in rows])
    out_rows = []
    for row, plan in zip(rows, plans):
        out = dict(row)
        out["plan"] = plan
        out["top_label"] = plan["chosen_command"]
        out["topk"] = []
        out["trained_on"] = False
        out_rows.append(out)
    return {"schema": SCHEMA_VERSION, "arm": "lever2", "cases": out_rows, "training_rows": {}, "train_seconds": None}


# ---------------------------------------------------------------- listing arms: one planner call per request (10.7)

#: The two listing arms the planner-only run measures (run_listing's FULL and COMMAND_SHORTLIST, as production
#: builds them). One call per request: the production text planner over the arm's listing as available_commands.
LISTING_FULL = "full"
LISTING_SHORTLIST = "command_shortlist"
LISTING_ARMS: tuple[str, ...] = (LISTING_FULL, LISTING_SHORTLIST)
#: Gates of the approved run: calls and prompt tokens (the dry run must show both at or under these).
GATE_MAX_CALLS = 110
GATE_MAX_PROMPT_TOKENS = 1_300_000
#: Per-call prompt overhead outside the listing and the request (the prelude and adapter; the same allowance the
#: score_listing estimate uses), and the output allowance of the dry run (the live run counts the real output).
LISTING_OVERHEAD_TOKENS = 1500
LISTING_OUTPUT_ESTIMATE = 300


def listing_text(opened, utterance: str, arm: str) -> str:
    """The arm's listing for one request, built by run_listing as production builds it (no provider call: the
    shortlist gathers with the trained heads' local classifiers)."""
    from fastworkflow import run_listing

    if arm == LISTING_FULL:
        return run_listing.run_listing_text(opened.app, utterance, arm=run_listing.FULL)
    if arm == LISTING_SHORTLIST:
        return run_listing.run_listing_text(opened.app, utterance, arm=run_listing.COMMAND_SHORTLIST,
                                            command_info_root=opened.command_info_root)
    raise ValueError(f"{arm!r} is not a listing arm; listing arms are {list(LISTING_ARMS)}")


def listing_workflow_rows(label: str, source: str, testset: str, context_path: Optional[str], per_stratum: int,
                          arm: str, seed: int, scratch: str) -> dict[str, Any]:
    """One workflow's requests for one listing arm: a stratified sample of its held-out set (per target), each with
    its gold commands (score_listing's gold), the arm's listing, and the names the plan reader needs."""
    from pathlib import Path

    from fastworkflow.train import score_listing

    ts = pts.load_test_set(testset)
    callees = {name: list(t["callees"]) for name, t in ts.get("targets", {}).items() if "callees" in t}
    sample = stratified_sample(pts.heldout(ts), per_stratum, seed=seed)
    opened = score_listing.open_workflow(Path(source), label, Path(scratch), context_path=context_path)
    names, core = _plan_names(opened.app.folderpath)
    rows = []
    for case in sample:
        listing = listing_text(opened, case["utterance"], arm)
        rows.append({"workflow": label, "arm": arm, "utterance": case["utterance"], "kind": case["kind"],
                     "expected": case["expected"], "source": case.get("source", ""),
                     "gold": sorted(score_listing.gold_for(case["expected"], case["kind"], callees)),
                     "listing": listing, "listing_chars": len(listing)})
    return {"label": label, "arm": arm, "active": opened.root_context,
            "bases": {ctx: sorted(v) for ctx, v in opened.bases.items()},
            "names": names, "core": core, "rows": rows}


def listing_dry_run_totals(workflows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calls and prompt tokens the live run would make (the planner instructions, the listing, the request and the
    per-call overhead, at characters / 4), and the gate verdict."""
    from fastworkflow.workflow_agent import PLANNER_INSTRUCTIONS

    calls = 0
    prompt = 0
    per = {}
    for wf in workflows:
        wf_prompt = 0
        for row in wf["rows"]:
            calls += 1
            tokens = (len(PLANNER_INSTRUCTIONS) + row["listing_chars"] + len(row["utterance"])) // CHARS_PER_TOKEN
            wf_prompt += tokens + LISTING_OVERHEAD_TOKENS
        per[wf["label"]] = {"calls": len(wf["rows"]), "prompt_tokens": wf_prompt,
                            "output_tokens": len(wf["rows"]) * LISTING_OUTPUT_ESTIMATE}
        prompt += wf_prompt
    ok = calls <= GATE_MAX_CALLS and prompt <= GATE_MAX_PROMPT_TOKENS
    return {"calls": calls, "prompt_tokens": prompt, "per_workflow": per, "gate_ok": ok,
            "gates": {"max_calls": GATE_MAX_CALLS, "max_prompt_tokens": GATE_MAX_PROMPT_TOKENS}}


def _usage_of(history_entries: Sequence[Mapping[str, Any]]) -> tuple[int, int, bool]:
    """(prompt tokens, output tokens, any cache hit) of the new LM history entries (LiteLLM usage)."""
    prompt = output = 0
    cached = False
    for entry in history_entries:
        usage = entry.get("usage") or {}
        prompt += int(usage.get("prompt_tokens", 0) or 0)
        output += int(usage.get("completion_tokens", 0) or 0)
        cached = cached or bool(entry.get("cache_hit"))
    return prompt, output, cached


def run_listing_live(workflows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One planner call per request, each with its arm's listing. LIVE: only after --allow-live and the gates.

    Each call is retried at most once (call_with_one_retry). Stores the raw planner output, the read plan, the
    latency, and the LiteLLM usage of the call. A request whose two attempts fail is stored with no plan and the
    error, not dropped.
    """
    import time

    import dspy

    from fastworkflow.utils import dspy_utils
    from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
    from fastworkflow.workflow_agent import PLANNER_INSTRUCTIONS

    class TaskPlannerTextSignature(dspy.Signature):
        __doc__ = PLANNER_INSTRUCTIONS
        user_query: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    predictor = dspy.ChainOfThought(TaskPlannerTextSignature)
    out_workflows = []
    with dspy.context(lm=lm, adapter=CommandsSystemPreludeAdapter()):
        for wf in workflows:
            out_rows = []
            for row in wf["rows"]:
                before = len(lm.history)
                started = time.perf_counter()
                prediction, attempts, error = call_with_one_retry(
                    lambda row=row: predictor(user_query=row["utterance"], available_commands=row["listing"]))
                latency = time.perf_counter() - started
                prompt_tokens, output_tokens, cached = _usage_of(lm.history[before:])
                if prediction is None:
                    plan = read_plan("", wf["names"], wf["core"])
                    plan.update({"reasoning": "", "attempts": attempts, "error": error})
                else:
                    plan = read_plan(str(prediction.next_steps or ""), wf["names"], wf["core"])
                    plan.update({"reasoning": str(getattr(prediction, "reasoning", "") or ""),
                                 "attempts": attempts, "error": ""})
                out = {k: v for k, v in row.items() if k != "listing"}
                out.update({"listing": row["listing"], "plan": plan, "latency_s": round(latency, 3),
                            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                            "cache_hit": cached, "lm_calls": len(lm.history) - before})
                out_rows.append(out)
            out_workflows.append({"label": wf["label"], "arm": wf["arm"], "active": wf["active"],
                                  "bases": wf["bases"], "rows": out_rows})
    return {"schema": SCHEMA_VERSION, "kind": "listing_live", "workflows": out_workflows}


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def score_listing_run(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Per workflow: reach of ALL gold commands (any step), the gold planning command named, single call, plan
    length, clarifying-question plans, tokens and latency, listing size, and the cause of each miss (a gold command
    not on the listing = shortlist; gold on the listing but not reached = planner)."""
    from fastworkflow.train import score_listing

    report = {}
    for wf in doc["workflows"]:
        rows = wf["rows"]
        bases = {ctx: frozenset(v) for ctx, v in wf["bases"].items()}
        counts = collections.Counter()
        lengths, latencies, prompts, outputs, chars = [], [], [], [], []
        misses = collections.Counter()
        per_expected: dict[str, collections.Counter] = {}
        failures = 0
        for row in rows:
            plan = row["plan"]
            named = _named_commands(plan)
            gold = set(row["gold"])
            reach_all = gold <= set(named)
            counts["n"] += 1
            counts["reach_all_gold"] += reach_all
            if row["kind"] == pts.KIND_PLANNING:
                counts["planning_n"] += 1
                counts["names_gold_planning"] += row["expected"] in named
                counts["single_call"] += classify_plan(row["expected"], plan) == PLAN_SINGLE_CALL
            if not named:
                counts["no_command_named"] += 1
                counts["clarifying_question"] += bool(_CLARIFY_RE.search(plan.get("raw", "") or ""))
            if plan.get("error"):
                failures += 1
            lengths.append(plan["plan_length"])
            latencies.append(row["latency_s"])
            prompts.append(row["prompt_tokens"])
            outputs.append(row["output_tokens"])
            chars.append(row["listing_chars"])
            counts["lm_calls"] += row["lm_calls"]
            counts["cache_hits"] += bool(row["cache_hit"])
            if not reach_all:
                entries = score_listing.listed_entries(row["listing"], wf["active"])
                on_listing = score_listing.covered(frozenset(gold), entries, bases)
                misses["shortlist (gold not on the listing)" if not on_listing else "planner (gold on the listing, not used)"] += 1
            block = per_expected.setdefault(row["expected"], collections.Counter())
            block["n"] += 1
            block["reach_all"] += reach_all
        n = counts["n"]
        report[wf["label"]] = {
            "arm": wf["arm"], "n": n, "planning_n": counts["planning_n"],
            "reach_all_gold": counts["reach_all_gold"],
            "names_gold_planning": counts["names_gold_planning"],
            "single_call_of_planning": counts["single_call"],
            "plan_length_mean": round(sum(lengths) / n, 2) if n else None,
            "plan_length_max": max(lengths) if lengths else None,
            "no_command_named": counts["no_command_named"],
            "clarifying_question_no_command": counts["clarifying_question"],
            "prompt_tokens_total": sum(prompts), "output_tokens_total": sum(outputs),
            "prompt_tokens_mean": round(sum(prompts) / n, 1) if n else None,
            "output_tokens_mean": round(sum(outputs) / n, 1) if n else None,
            "latency_s_mean": round(sum(latencies) / n, 3) if n else None,
            "latency_s_p95": _pct(latencies, 0.95),
            "listing_chars_mean": round(sum(chars) / n, 1) if n else None,
            "listing_chars_p95": _pct(chars, 0.95),
            "listing_tokens_est_mean": round(sum(chars) / n / CHARS_PER_TOKEN, 1) if n else None,
            "lm_calls": counts["lm_calls"], "cache_hits": counts["cache_hits"], "failed_requests": failures,
            "misses": dict(misses),
            "per_expected": {k: {"n": v["n"], "reach_all": v["reach_all"]} for k, v in sorted(per_expected.items())},
        }
    return report


def _job_types_by_scope(workflow_folderpath: str) -> dict[str, str]:
    """The Jev matcher's job-type text per scope (``""`` for root), as ``workflow_agent._matcher_job_types``
    lays it out, read through ``resolve_field_sources`` (no write)."""
    from fastworkflow.planning_commands import resolve_field_sources
    from fastworkflow.workflow_agent import MATCHER_EXAMPLES

    resolution = resolve_field_sources(workflow_folderpath)
    by_scope: dict[str, list[str]] = {}
    from fastworkflow.planning_commands import visible_planning_commands

    for name, command in sorted(visible_planning_commands(workflow_folderpath).items()):
        scope = name.rpartition("/")[0]
        sig = resolution.signatures.get(name)
        shown = [" ".join(str(u).split()) for u in (getattr(sig, "plain_utterances", ()) or ())[:MATCHER_EXAMPLES]]
        shown = [u for u in shown if u]
        line = f"- {name}: {command.description}"
        if shown:
            line += " e.g. " + ", ".join(json.dumps(u, ensure_ascii=False) for u in shown)
        by_scope.setdefault(scope, []).append(line)
    return {scope: "\n".join(lines) for scope, lines in by_scope.items()}


def _listing_chars(workflow_folderpath: str) -> int:
    """Rough size of the planner's listing: every command's name and a description, plus each planning
    command's template. Estimated from the loader; the planner's real listing needs a live Workflow."""
    from fastworkflow.planning_commands import resolve_field_sources

    resolution = resolve_field_sources(workflow_folderpath)
    total = 0
    for name, sig in resolution.signatures.items():
        doc = (getattr(sig, "__doc__", None) or "").strip()
        total += len(name) + len(doc) + 40
    from fastworkflow.planning_commands import visible_planning_commands

    for planning in visible_planning_commands(workflow_folderpath).values():
        total += sum(len(str(step)) for step in planning.template)
    return total


def run_live(arm: str, cases: Sequence[Mapping[str, Any]], workflow_folderpath: str) -> dict[str, Any]:
    """Call the provider for each case and return results in the offline schema. LIVE: only after --allow-live."""
    rows = [dict(c) for c in cases]
    if arm == ARM_JEV:
        from fastworkflow.workflow_agent import match_planning_command

        for row in rows:
            matched = match_planning_command(row["utterance"], workflow_folderpath,
                                             scopes=tuple(dict.fromkeys((row["head"], ""))))
            row["top_label"] = matched[0] if matched else ""
            row["topk"] = []
    elif arm == ARM_AGENT_PLAN:
        plans = _agent_plans(rows, workflow_folderpath)
        for row, plan in zip(rows, plans):
            row["plan"] = plan
            row["top_label"] = plan["chosen_command"]
            row["topk"] = []
    else:
        raise ValueError(f"{arm!r} is not a live arm")
    return {"schema": SCHEMA_VERSION, "arm": arm, "cases": rows, "training_rows": {}, "train_seconds": None}


def _plan_names(workflow_folderpath: str) -> tuple[list[str], list[str]]:
    """The command names a plan step may name, and the core (navigation) names, as the runtime reads
    them: ``workflow_execution_context._workflow_command_names`` (routed commands, then planning
    commands) and ``workflow_agent.core_command_names``. Needs the workflow created (routing loaded)."""
    import fastworkflow
    from fastworkflow.planning_commands import resolve_field_sources
    from fastworkflow.workflow_agent import core_command_names

    routing = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    names = list(routing.routing_definition_map)
    from fastworkflow.planning_commands import visible_planning_commands

    names += list(visible_planning_commands(workflow_folderpath))
    return names, sorted(core_command_names())


def _agent_plans(rows: Sequence[Mapping[str, Any]], workflow_folderpath: str,
                 hints: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    """The plan the planner writes for each request, read as production reads it (``read_plan``).

    The planner is production's text planner: ``TaskPlannerTextSignature`` (``build_query_with_next_steps``)
    run as ``dspy.ChainOfThought`` under ``CommandsSystemPreludeAdapter``, with the planner's listing as
    ``available_commands``, on ``LLM_PLANNER``. No matcher and no template: the arm measures the choice
    the planner makes as it plans. The plan tool of design 2.52 is not built in this tree.
    """
    import dspy

    import fastworkflow
    from fastworkflow.utils import dspy_utils
    from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
    from fastworkflow.workflow_agent import PLANNER_INSTRUCTIONS, planner_commands_text

    class TaskPlannerTextSignature(dspy.Signature):
        __doc__ = PLANNER_INSTRUCTIONS
        user_query: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    workflow = fastworkflow.Workflow.create(workflow_folderpath, workflow_id_str="planning-match-scoring")
    listing = planner_commands_text(workflow)
    names, core = _plan_names(workflow_folderpath)
    lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    predictor = dspy.ChainOfThought(TaskPlannerTextSignature)
    plans = []
    with dspy.context(lm=lm, adapter=CommandsSystemPreludeAdapter()):
        for index, row in enumerate(rows):
            query = row["utterance"]
            if hints and hints[index]:
                query = f"{query}\n\n{hints[index]}"
            prediction, attempts, error = call_with_one_retry(
                lambda query=query: predictor(user_query=query, available_commands=listing))
            if prediction is None:
                plan = read_plan("", names, core)
                plan.update({"reasoning": "", "attempts": attempts, "error": error})
            else:
                plan = read_plan(str(prediction.next_steps or ""), names, core)
                plan.update({"reasoning": str(getattr(prediction, "reasoning", "") or ""),
                             "attempts": attempts, "error": ""})
            plan["hint"] = hints[index] if hints else ""
            plans.append(plan)
    return plans


def call_with_one_retry(call) -> tuple[Any, int, str]:
    """Run *call* once, and once more if it raises (a per-call transient error). Returns
    ``(prediction, attempts, error)``: the prediction is None when both attempts fail, and *error*
    names the last failure. A failed case is recorded as a case with no plan, not a crash."""
    error = ""
    for attempt in (1, 2):
        try:
            return call(), attempt, ""
        except Exception as exc:  # noqa: BLE001 - a provider or parse failure is recorded per case
            error = f"{type(exc).__name__}: {exc}"[:300]
    return None, 2, error


# ---------------------------------------------------------------- CLI

def _test_set(args) -> dict:
    if args.test_set:
        return pts.load_test_set(args.test_set)
    if not args.workflow:
        raise SystemExit("give --test-set, or --workflow to build the set in memory")
    return pts.build_test_set(args.workflow, per_command=args.per_command,
                              handwritten_path=args.handwritten, cache_root=args.cache_root)


def _write(path: Optional[str], payload: Mapping[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class LiveRunRefused(RuntimeError):
    """A live run would repeat one already made or already in progress (same arm and run id)."""


def live_lock_path(arm: str, out: str, run_id: str) -> str:
    """The lock file of one live run: ``.live_<arm>_<run_id>.lock`` beside its results file.

    Run ids are unique per results directory: the lock and the results file are both keyed on the
    arm and the run id, so a second launch of the same run in that directory is refused.
    """
    folder = os.path.dirname(os.path.abspath(out))
    return os.path.join(folder, f".live_{arm}_{run_id}.lock")


def live_run_problem(arm: str, out: str, run_id: str) -> Optional[str]:
    """Why a live run with this arm, results file and run id must not start, or None when it may."""
    if os.path.exists(out):
        return f"results file {out} already exists: this arm and run id already ran; use a new --run-id"
    lock = live_lock_path(arm, out, run_id)
    if os.path.exists(lock):
        return (f"lock {lock} exists: a run of arm {arm!r} with run id {run_id!r} is in progress or died. "
                "Remove the lock only after confirming no process holds it")
    return None


def claim_live_run(arm: str, out: str, run_id: str) -> str:
    """Create the run's lock exclusively (before any provider call) and return its path.

    Raises LiveRunRefused when the results file exists or the lock is already held: the check and the
    create are both done here, and ``O_EXCL`` makes the create fail for a second launch that races it.
    """
    problem = live_run_problem(arm, out, run_id)
    if problem:
        raise LiveRunRefused(problem)
    lock = live_lock_path(arm, out, run_id)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise LiveRunRefused(f"lock {lock} exists: a run of arm {arm!r} with run id {run_id!r} is in progress") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "arm": arm, "run_id": run_id, "out": os.path.abspath(out)}, handle)
    return lock


def release_live_run(lock: str) -> None:
    """Remove the run's lock (only the one this process claimed)."""
    if os.path.exists(lock):
        os.remove(lock)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="score_planning_match", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_set(p):
        p.add_argument("--test-set", default=None, help="a built test set (JSON)")
        p.add_argument("--workflow", default=None, help="build the set in memory from this workflow")
        p.add_argument("--handwritten", default=None)
        p.add_argument("--per-command", type=int, default=pts.DEFAULT_PER_COMMAND)
        p.add_argument("--cache-root", default=None)

    p_predict = sub.add_parser("predict", help="offline: score a trained copy (arms heads_labelled, heads_not_here)")
    p_predict.add_argument("--arm", required=True, choices=OFFLINE_ARMS)
    p_predict.add_argument("--trained", required=True, help="the trained workflow copy")
    p_predict.add_argument("--train-seconds", type=float, default=None)
    p_predict.add_argument("--out", required=True)
    add_set(p_predict)

    p_live = sub.add_parser("live", help="LIVE arms (jev, agent_plan): refuse without --allow-live")
    p_live.add_argument("--arm", required=True, choices=LIVE_ARMS)
    p_live.add_argument("--per-stratum", type=int, default=PER_STRATUM_LIBRARY)
    p_live.add_argument("--seed", type=int, default=0)
    p_live.add_argument("--dry-run", action="store_true", help="print the estimate; call nothing")
    p_live.add_argument("--allow-live", action="store_true", help="permit the provider calls")
    p_live.add_argument("--out", default=None, help="the results file (refused if it already exists)")
    p_live.add_argument("--run-id", default=None,
                        help="this run's identity: with --arm and --out it names the lock; a run id is used once")
    add_set(p_live)

    p_lever = sub.add_parser("lever2", help="LIVE lever 2: the agent plans with the unsure head's candidates "
                                           "(refused without --allow-live; same launcher and lock as live)")
    p_lever.add_argument("--workflow", required=True)
    p_lever.add_argument("--test-set", required=True)
    p_lever.add_argument("--heads-results", required=True, help="heads_labelled per-case results (one seed)")
    p_lever.add_argument("--analysis-keys", required=True, help="the shared analysis keys JSON")
    p_lever.add_argument("--jev-results", required=True, help="the jev sample results (which cases are shared)")
    p_lever.add_argument("--dry-run", action="store_true")
    p_lever.add_argument("--allow-live", action="store_true")
    p_lever.add_argument("--out", default=None)
    p_lever.add_argument("--run-id", default=None)
    p_list = sub.add_parser("listing-live", help="LIVE planner-only listing arms (one call per request); "
                                                 "dry run by default; refused without --allow-live")
    p_list.add_argument("--library-source", default=None)
    p_list.add_argument("--library-testset", default=None)
    p_list.add_argument("--library-context", default="fastworkflow.examples.library.application.library:Library")
    p_list.add_argument("--library-per-stratum", type=int, default=3)
    p_list.add_argument("--library-arm", choices=LISTING_ARMS, default=LISTING_FULL)
    p_list.add_argument("--app-source", default=None)
    p_list.add_argument("--app-testset", default=None)
    p_list.add_argument("--app-context", default=None)
    p_list.add_argument("--app-per-stratum", type=int, default=1)
    p_list.add_argument("--app-arm", choices=LISTING_ARMS, default=LISTING_SHORTLIST)
    p_list.add_argument("--seed", type=int, default=0)
    p_list.add_argument("--scratch", required=True)
    p_list.add_argument("--dry-run", action="store_true")
    p_list.add_argument("--allow-live", action="store_true")
    p_list.add_argument("--out", default=None)
    p_list.add_argument("--run-id", default=None)
    p_score = sub.add_parser("score", help="metrics for one results file")
    p_score.add_argument("--results", required=True)
    p_score.add_argument("--out", default=None)

    p_report = sub.add_parser("report", help="arms against the baseline, with the guidance bars")
    p_report.add_argument("--baseline", required=True, help="results of heads_not_here (the baseline)")
    p_report.add_argument("--arm", action="append", default=[], metavar="NAME=RESULTS")
    p_report.add_argument("--out", default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "predict":
        test_set = _test_set(args)
        try:
            results = predict_arm(args.trained, test_set, args.arm, train_seconds=args.train_seconds)
        except ArmMismatchError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        _write(args.out, results)
        summary = summarize(results)
        print(format_report({"baseline": args.arm, "arms": {args.arm: {"summary": summary,
                                                                        "comparison": None, "guidance": None}}}))
        print(f"\nwrote {len(results['cases'])} case results to {args.out}")
        return 0

    if args.command == "live":
        import fastworkflow

        fastworkflow.init({})
        test_set = _test_set(args)
        folder = args.workflow
        if not folder:
            parser.error("live arms need --workflow, the workflow folder the provider calls run on")
        sample = stratified_sample(pts.heldout(test_set), args.per_stratum, seed=args.seed)
        cases = routing_rows(sample)
        est = estimate(args.arm, cases, folder)
        print(f"{args.arm}: {len(cases)} heldout requests, {args.per_stratum} per stratum "
              f"(seed {args.seed}), workflow {test_set['workflow']}")
        print(f"estimate: {est['calls']} provider calls, ~{est['input_tokens']} input tokens, "
              f"~{est['output_tokens']} output tokens ({est['note']}; {est['token_estimate_accuracy']})")
        if args.dry_run:
            if args.out and args.run_id:
                problem = live_run_problem(args.arm, args.out, args.run_id)
                print(f"a live start now would be refused: {problem}" if problem
                      else f"a live start with --run-id {args.run_id} would be accepted")
            print("dry run: no call made")
            return 0
        if not args.allow_live:
            print(f"refused: arm {args.arm!r} is LIVE (it calls a model provider). Re-run with --allow-live "
                  "after the owner approves the estimate.", file=sys.stderr)
            return 2
        if not args.out or not args.run_id:
            print("refused: a live run needs --out (its results file) and --run-id (its identity)", file=sys.stderr)
            return 2
        try:
            lock = claim_live_run(args.arm, args.out, args.run_id)
        except LiveRunRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        try:
            results = run_live(args.arm, cases, folder)
            _write(args.out, results)
        finally:
            release_live_run(lock)
        print(f"wrote {len(results['cases'])} live case results to {args.out}")
        return 0

    if args.command == "lever2":
        import fastworkflow

        fastworkflow.init({})
        folder = args.workflow
        head_cases = _load(args.heads_results)["cases"]
        analysis_keys = {tuple(k) for k in _load(args.analysis_keys)["analysis_keys"]}
        jev_keys = {_key(c) for c in _load(args.jev_results)["cases"]}
        rows = lever2_rows(head_cases, analysis_keys, jev_keys, folder)
        shared = len(analysis_keys & jev_keys)
        print(f"lever2: {len(rows)} cases (head unsure or not-here, of the {shared} analysis cases shared with the jev sample); "
              f"hint blocks on {sum(1 for r in rows if r['hint'])}")
        estimated = [dict(r, utterance=r["utterance"] + ("\n\n" + r["hint"] if r["hint"] else "")) for r in rows]
        est = estimate(ARM_AGENT_PLAN, estimated, folder)
        print(f"estimate: {est['calls']} provider calls, ~{est['input_tokens']} input tokens, "
              f"~{est['output_tokens']} output tokens ({est['note']})")
        if args.dry_run:
            if args.out and args.run_id:
                problem = live_run_problem("lever2", args.out, args.run_id)
                print(f"a live start now would be refused: {problem}" if problem
                      else f"a live start with --run-id {args.run_id} would be accepted")
            print("dry run: no call made")
            return 0
        if not args.allow_live:
            print("refused: lever2 is LIVE (it calls a model provider). Re-run with --allow-live after approval.",
                  file=sys.stderr)
            return 2
        if not args.out or not args.run_id:
            print("refused: a live run needs --out and --run-id", file=sys.stderr)
            return 2
        try:
            lock = claim_live_run("lever2", args.out, args.run_id)
        except LiveRunRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        try:
            results = run_lever2(rows, folder)
            _write(args.out, results)
        finally:
            release_live_run(lock)
        print(f"wrote {len(results['cases'])} lever2 case results to {args.out}")
        return 0

    if args.command == "listing-live":
        import fastworkflow

        fastworkflow.init({})
        specs = []
        if args.library_source:
            specs.append(("library", args.library_source, args.library_testset, args.library_context,
                          args.library_per_stratum, args.library_arm))
        if args.app_source:
            specs.append(("app", args.app_source, args.app_testset, args.app_context,
                          args.app_per_stratum, args.app_arm))
        if not specs:
            parser.error("give --library-source and/or --app-source")
        workflows = [listing_workflow_rows(label, src, ts, ctx, k, arm, args.seed, args.scratch)
                     for label, src, ts, ctx, k, arm in specs]
        totals = listing_dry_run_totals(workflows)
        for wf in workflows:
            chars = sorted(r["listing_chars"] for r in wf["rows"])
            print(f"{wf['label']} arm={wf['arm']}: {len(wf['rows'])} requests; listing chars mean "
                  f"{sum(chars) / len(chars):.0f}, p95 {_pct(chars, 0.95)}")
        print(f"calls {totals['calls']} (gate {GATE_MAX_CALLS}); prompt tokens est {totals['prompt_tokens']} "
              f"(gate {GATE_MAX_PROMPT_TOKENS}); gate {'OK' if totals['gate_ok'] else 'FAILED'}")
        if args.dry_run:
            if args.out and args.run_id:
                problem = live_run_problem("listing_live", args.out, args.run_id)
                print(f"a live start now would be refused: {problem}" if problem
                      else f"a live start with --run-id {args.run_id} would be accepted")
            print("dry run: no call made")
            return 0 if totals["gate_ok"] else 2
        if not totals["gate_ok"]:
            print("refused: the dry run is over a gate; stop and report", file=sys.stderr)
            return 2
        if not args.allow_live:
            print("refused: the listing live run is LIVE (it calls a model provider). Re-run with --allow-live.",
                  file=sys.stderr)
            return 2
        if not args.out or not args.run_id:
            print("refused: a live run needs --out and --run-id", file=sys.stderr)
            return 2
        try:
            lock = claim_live_run("listing_live", args.out, args.run_id)
        except LiveRunRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        try:
            doc = run_listing_live(workflows)
            _write(args.out, doc)
        finally:
            release_live_run(lock)
        score = score_listing_run(doc)
        _write(args.out[:-5] + ".score.json" if args.out.endswith(".json") else args.out + ".score.json", score)
        print(json.dumps(score, indent=1, default=str))
        return 0

    if args.command == "score":
        summary = summarize(_load(args.results))
        _write(args.out, summary)
        print(json.dumps({k: summary[k] for k in METRICS}, indent=2))
        if "plan_2_44" in summary:
            print(json.dumps({"plan_2_44": summary["plan_2_44"]}, indent=2))
        return 0

    base = _load(args.baseline)
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"--arm takes NAME=RESULTS, got {spec!r}")
        arms[name] = _load(path)
    doc = report(base, arms)
    _write(args.out, doc)
    print(format_report(doc))
    return 0


def routing_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The rows a predict or live run scores: the set's fields the scorer reads, nothing else."""
    return [{k: c[k] for k in ("head", "utterance", "expected", "kind", "related", "split", "source")
             if k in c} for c in cases]


if __name__ == "__main__":
    sys.exit(main())
