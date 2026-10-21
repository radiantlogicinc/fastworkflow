#!/usr/bin/env python
"""replay_code_steps.py -- which stored phase steps code would run (fix-6llp).

READ-ONLY: the observability DB is opened with ``mode=ro``; nothing is run.

Every ``fw.skill.phase`` span in the DB carries the phase's numbered plan
(``steps``). Each plan is read the way the runner reads it before a phase run
(``skill_phases.plan_code_steps``), with the workflow's own command schemas:
a step is MECHANICAL (code runs it), PASSED over (an answer or negated step:
nothing to run) or OPEN (the agent's). Printed:

- the counts per reason;
- the phases mechanical end to end, and the steps in a leading mechanical run
  (where code runs before the agent);
- the mechanical steps that read only page 1 (a paged command, no paging value
  written): code hands over after one when more pages remain;
- every note code drops (text after the chain, from the first ``(`` or ``;``),
  with counts, for review: code never sends it.

Answer-only and skipped phases have no executor run and are left out.

Usage:
    replay_code_steps.py <workflow_folder> --db PATH [--experiment ID ...] [--json]

Cost: ~10 s (imports fastworkflow and the workflow's command modules).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import sys


def _plan_text(value):
    """A span's ``steps``: the text, or the value of a capped attribute envelope."""
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, str) else ""


def stored_plans(db, experiments=()):
    """``(span_id, skill, phase, item, plan_text)`` of each phase run with an executor."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        query = "SELECT span_id, attributes, experiment_id FROM spans WHERE name = 'fw.skill.phase'"
        rows = conn.execute(query + " ORDER BY trace_id, start_ns").fetchall()
    finally:
        conn.close()
    for span_id, attributes, experiment in rows:
        if experiments and experiment not in experiments:
            continue
        a = json.loads(attributes)
        plan = _plan_text(a.get("steps"))
        if a.get("skipped") or a.get("answer_only") or not plan.strip():
            continue
        yield span_id, a.get("skill"), a.get("phase"), a.get("item"), plan


def replay(folder, db, experiments=()):
    import fastworkflow
    from fastworkflow import skill_phases
    from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS

    env = os.path.join(folder, "fastworkflow.env")
    if os.path.isfile(env):
        from dotenv import dotenv_values

        fastworkflow.init(env_vars=dict(dotenv_values(env)))
    else:
        fastworkflow.init(env_vars={})
    from fastworkflow.workflow_agent import core_command_names

    routing = fastworkflow.RoutingRegistry.get_definition(folder)
    parameters = skill_phases.parameter_classes(routing)
    commands, core = list(routing.routing_definition_map), list(core_command_names())

    reasons = collections.Counter()
    notes = collections.Counter()
    page_one = collections.Counter()
    phases = steps = mechanical = all_mechanical = leading = 0
    for _span, _skill, _phase, _item, plan in stored_plans(db, experiments):
        read = skill_phases.plan_code_steps(plan, commands, core, parameters)
        phases += 1
        steps += len(read)
        open_seen = False
        for step, command, reason in read:
            kind = "mechanical" if command else "passed" if command == "" else "open"
            reasons[(kind, reason)] += 1
            if kind == "open":
                open_seen = True
            elif not open_seen:
                leading += 1
            if not command:
                continue
            mechanical += 1
            note = step.text.replace("`", " ").replace("*", " ")
            cut = re.search(r"[(;]", note[len(step.command):])
            if cut:
                notes[note[len(step.command) + cut.start():].strip()] += 1
            fields = {name for cls in parameters(step.command) if cls is not None
                      for name in cls.model_fields}
            if any(p in fields for p in PAGING_PARAMETERS) and not any(
                    p in step.arguments for p in PAGING_PARAMETERS):
                page_one[step.command] += 1
        all_mechanical += not open_seen
    return {
        "phases": phases, "steps": steps, "mechanical": mechanical,
        "phases_mechanical_end_to_end": all_mechanical, "steps_in_leading_run": leading,
        "reasons": {f"{kind}: {reason}": n for (kind, reason), n in reasons.most_common()},
        "page_one_steps": dict(page_one.most_common()),
        "dropped_notes": dict(notes.most_common()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow_folder")
    parser.add_argument("--db", required=True, help="the workflow's observability.sqlite3")
    parser.add_argument("--experiment", action="append", default=[],
                        help="only phase runs of this experiment id (repeatable)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    folder = os.path.abspath(args.workflow_folder)
    sys.path.insert(0, os.path.dirname(folder))
    result = replay(folder, args.db, set(args.experiment))
    if args.json:
        print(json.dumps(result, indent=1))
        return
    pct = lambda n, d: f"{n} of {d} ({100 * n // max(d, 1)}%)"  # noqa: E731
    print(f"phase runs: {result['phases']}; planned steps: {result['steps']}")
    print(f"mechanical steps: {pct(result['mechanical'], result['steps'])}")
    print(f"phases mechanical end to end: "
          f"{pct(result['phases_mechanical_end_to_end'], result['phases'])}")
    print(f"steps in a leading mechanical run: {pct(result['steps_in_leading_run'], result['steps'])}")
    print("\nper reason:")
    for reason, n in result["reasons"].items():
        print(f"  {n:6}  {reason}")
    print("\npage-1 runs (paged command, no paging value written; hand over if more pages):")
    for command, n in result["page_one_steps"].items():
        print(f"  {n:6}  {command}")
    print(f"\nnotes code drops ({sum(result['dropped_notes'].values())} steps):")
    for note, n in result["dropped_notes"].items():
        print(f"  {n:6}  {note[:150]}")


if __name__ == "__main__":
    main()
