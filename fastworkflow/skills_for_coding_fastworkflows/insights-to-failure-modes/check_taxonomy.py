#!/usr/bin/env python3
"""Validate a failure-mode taxonomy against the insight files it clusters.

    check_taxonomy.py <taxonomy.yaml> <execution_insights.md> <planning_insights.md>
                      [--exec-prefix E] [--plan-prefix P] [--forbid REGEX ...]

Insight files are fastworkflow distillation output: execution items as
"N. text" lines, planning items as "## N. text" headings. Items are numbered by
position, so the k-th execution item is <exec-prefix>k.

Errors (exit 1): an insight in no mode or in several; an id that does not
exist; a mode missing required fields; a compiled mode without check/hook; an
uncompiled mode without notes; an unknown signal or hook; a forbidden literal
(task data) inside a `check`. Warnings do not fail the run.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")

SIGNALS = {"trajectory", "+intent", "+user_text", "+question-judge", "none"}
HOOKS = {"before-side-effect", "after-observation", "before-finish"}
REASONS = (
    "contradicts-policy",
    "lexical-only",
    "no-deterministic-signal",
    "not-outcome-relevant",
    "needs-judge",
    "task-specific",
    "covered-by",
)
#: Literals that look like one task's data rather than a workflow-level check:
#: long digit runs (ids) and currency amounts. Extend with --forbid.
DEFAULT_FORBID = (r"\d{5,}", r"[$€£]\s?\d")


def count_items(path: str, planning: bool) -> int:
    pattern = re.compile(r"^##\s+\d+\." if planning else r"^\d+\.\s")
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if pattern.match(line))


def first_token(value) -> str:
    return str(value or "").strip().split()[0] if str(value or "").strip() else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("taxonomy")
    ap.add_argument("execution_insights")
    ap.add_argument("planning_insights")
    ap.add_argument("--exec-prefix", default="E")
    ap.add_argument("--plan-prefix", default="P")
    ap.add_argument("--forbid", action="append", default=[],
                    help="extra regex for task data that must not appear in `check`")
    args = ap.parse_args()

    with open(args.taxonomy, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    modes = doc.get("modes") or []
    n_exec = count_items(args.execution_insights, planning=False)
    n_plan = count_items(args.planning_insights, planning=True)
    expected = {f"{args.exec_prefix}{i}" for i in range(1, n_exec + 1)} | {
        f"{args.plan_prefix}{i}" for i in range(1, n_plan + 1)
    }
    forbid = [re.compile(p) for p in (*DEFAULT_FORBID, *args.forbid)]

    errors: list[str] = []
    warnings: list[str] = []
    where: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()

    for k, mode in enumerate(modes):
        mid = mode.get("id") or f"<mode #{k + 1}>"
        if mid in seen_ids:
            errors.append(f"{mid}: duplicate mode id")
        seen_ids.add(mid)
        for field in ("id", "name", "insights", "signal", "compiled"):
            if field not in mode:
                errors.append(f"{mid}: missing `{field}`")
        insights = mode.get("insights") or {}
        if not isinstance(insights, dict) or not {"execution", "planning"} <= set(insights):
            errors.append(f"{mid}: `insights` needs both `execution` and `planning` lists")
            insights = insights if isinstance(insights, dict) else {}
        for key in ("execution", "planning"):
            for iid in insights.get(key) or []:
                where[str(iid)].append(mid)
                if str(iid) not in expected:
                    errors.append(f"{mid}: insight {iid} does not exist in the corpus")
        signal = first_token(mode.get("signal"))
        if signal and signal not in SIGNALS:
            errors.append(f"{mid}: unknown signal {signal!r} (one of {sorted(SIGNALS)})")
        compiled = mode.get("compiled")
        if compiled not in (None, False):
            if not mode.get("check"):
                errors.append(f"{mid}: compiled as {compiled} but has no `check`")
            hooks = re.split(r"[,\s]+", str(mode.get("hook") or "").strip())
            hooks = [h for h in hooks if h]
            if not hooks:
                errors.append(f"{mid}: compiled as {compiled} but has no `hook`")
            for h in hooks:
                if h not in HOOKS:
                    errors.append(f"{mid}: unknown hook {h!r} (one of {sorted(HOOKS)})")
        elif compiled is False:
            notes = str(mode.get("notes") or "").strip()
            if not notes:
                errors.append(f"{mid}: not compiled but `notes` gives no reason")
            elif not notes.lower().startswith(REASONS):
                warnings.append(
                    f"{mid}: notes should start with a reason keyword ({', '.join(REASONS)})"
                )
        check = str(mode.get("check") or "")
        for pattern in forbid:
            if m := pattern.search(check):
                errors.append(
                    f"{mid}: `check` contains task data {m.group(0)!r} - keep checks workflow-level"
                )

    for iid in sorted(expected, key=lambda s: (s.rstrip("0123456789"), int(re.sub(r"\D", "", s)))):
        if not where.get(iid):
            errors.append(f"insight {iid} is in no mode (excluded ones need a mode with compiled: false)")
        elif len(where[iid]) > 1:
            errors.append(f"insight {iid} is in several modes: {', '.join(where[iid])}")

    print(f"{args.taxonomy}: {len(modes)} modes, {len(expected)} insights "
          f"({n_exec} execution, {n_plan} planning)")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
