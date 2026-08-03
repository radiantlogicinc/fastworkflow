#!/usr/bin/env python3
"""Score a workflow's intent benchmark and compare two runs case by case.

Why this exists
---------------
`fastworkflow train` already runs held-out evaluation and writes
`<workflow>/___command_info/heldout_evaluation.json`. That report is the right thing
to read for "how good is this run", but it is **not** enough to compare two runs:

* Its routing numbers are aggregate counts plus per-command counts. Per-command counts
  cannot reconstruct which individual cases flipped, so an exact McNemar test on the
  discordant pairs cannot be derived from two reports.
* Its routing cases come from the whole-persona holdout, which is re-drawn every run
  (and, until utterance persistence lands, drawn from freshly generated utterances).
  Two runs are therefore scored on different cases and cannot be paired at all.

This script closes both gaps by scoring a **fixed, developer-supplied benchmark file**
(`docs/intent_benchmark_format.md`) against a specific set of trained artifacts, and
recording the per-case verdict. Two such verdict files can then be compared pairwise.

It only reads artifacts. It never trains and never writes into a workflow directory.

Usage
-----
    score_benchmark.py score --workflow WF --benchmark WF/intent_benchmark.json \
        [--version VERSION_ID] [--context CTX ...] --out before.json

    score_benchmark.py compare --before before.json --after after.json

`--version current|previous|<id>` selects the trainer's bounded internal artifacts;
omit it to score current.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import fastworkflow
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.train import artifact_versioning, heldout_evaluation


def _resolve_version_id(workflow: str, requested: str | None) -> str | None:
    current = artifact_versioning.resolve_current_version(workflow)
    if requested in (None, "current"):
        return current
    if requested == "previous":
        if current is None:
            return None
        previous = artifact_versioning.read_manifest(
            workflow, current).get("previous_version")
        return str(previous) if previous else None
    return requested


def _resolve_context_dir(workflow: str, version: str | None, context: str) -> Path:
    """Return the artifact directory holding one context's trained models."""
    folder = artifact_versioning.context_folder_name(context)
    if version is None:
        # Legacy flat layout: ___command_info/<context>/
        return artifact_versioning.command_info_root(workflow) / folder
    return artifact_versioning.versions_root(workflow) / version / folder


def _case_key(case) -> str:
    """Stable identity for a benchmark case, used to pair two runs."""
    return f"{case.context}|{case.kind}|{heldout_evaluation.normalize_utterance(case.utterance)}"


def _score(args: argparse.Namespace) -> int:
    cases = heldout_evaluation.load_benchmark_file(args.benchmark)
    if not cases:
        print(f"No cases in {args.benchmark}", file=sys.stderr)
        return 1

    version_id = _resolve_version_id(args.workflow, args.version)
    contexts = args.contexts or sorted({case.context for case in cases})
    escalation_labels = {
        heldout_evaluation.normalize_label(label)
        for label in heldout_evaluation.DEFAULT_ESCALATION_LABELS
    }

    records = []
    for context in contexts:
        context_dir = _resolve_context_dir(args.workflow, version_id, context)
        if not (context_dir / "threshold.json").is_file():
            print(
                f"skipping context {context!r}: no threshold.json under {context_dir} "
                f"(untrained, or wrong --version)",
                file=sys.stderr,
            )
            continue

        router = CommandRouter(str(context_dir))
        for case in cases:
            if case.context != context:
                continue
            predictions = [
                heldout_evaluation.normalize_label(p)
                for p in (router.predict(case.utterance) or [])
            ]
            record = {
                "key": _case_key(case),
                "context": case.context,
                "kind": case.kind,
                "utterance": case.utterance,
                "predicted": predictions,
            }
            if case.kind == "routing":
                expected = heldout_evaluation.normalize_label(case.expected_label or "")
                record["expected"] = expected
                record["top1"] = bool(predictions) and predictions[0] == expected
                record["in_list"] = expected in predictions
            else:
                record["expected_ancestor_command"] = case.expected_ancestor_command
                # Only a LONE escalation label drives the parent-chain walk; an
                # escalation label inside a top-k list takes the ambiguity branch.
                record["escalated"] = (
                    len(predictions) == 1 and predictions[0] in escalation_labels
                )
            records.append(record)

    payload = {
        "workflow": str(args.workflow),
        "version": version_id,
        "benchmark": str(args.benchmark),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    routing = [r for r in records if r["kind"] == "routing"]
    escalation = [r for r in records if r["kind"] == "escalation"]
    print(f"wrote {args.out}: {len(records)} case(s)")
    if routing:
        top1 = sum(r["top1"] for r in routing)
        in_list = sum(r["in_list"] for r in routing)
        print(
            f"  routing    n={len(routing)}  top-1={top1}/{len(routing)} "
            f"({top1 / len(routing):.1%})  in-list={in_list}/{len(routing)} "
            f"({in_list / len(routing):.1%})"
        )
    if escalation:
        ok = sum(r["escalated"] for r in escalation)
        print(
            f"  escalation n={len(escalation)}  recall={ok}/{len(escalation)} "
            f"({ok / len(escalation):.1%})"
        )
    return 0


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b/c discordant pairs.

    Under the null each discordant pair is a fair coin, so the count is
    Binomial(b + c, 0.5). Matches
    `fastworkflow-proof-and-analysis-toolkit/scripts/passk_math.py mcnemar`.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _paired(before: list[dict], after: list[dict], field: str) -> tuple[int, int, list, list]:
    """Return (fixed, broken, fixed_cases, broken_cases) for one boolean verdict field."""
    after_by_key = {r["key"]: r for r in after}
    fixed, broken = [], []
    for record in before:
        other = after_by_key.get(record["key"])
        if other is None or field not in record or field not in other:
            continue
        if not record[field] and other[field]:
            fixed.append(record)
        elif record[field] and not other[field]:
            broken.append(record)
    return len(fixed), len(broken), fixed, broken


def _report_axis(name: str, before: list[dict], after: list[dict], field: str) -> None:
    paired_keys = {r["key"] for r in before} & {r["key"] for r in after}
    subset_before = [r for r in before if r["key"] in paired_keys and field in r]
    subset_after = [r for r in after if r["key"] in paired_keys and field in r]
    if not subset_before:
        return

    n = len(subset_before)
    before_rate = sum(r[field] for r in subset_before)
    after_rate = sum(r[field] for r in subset_after)
    fixed, broken, _, broken_cases = _paired(subset_before, subset_after, field)
    p = exact_mcnemar_p(broken, fixed)

    print(f"\n{name}  (n = {n} paired cases)")
    print(f"  before {before_rate}/{n} ({before_rate / n:.1%})   "
          f"after {after_rate}/{n} ({after_rate / n:.1%})")
    print(f"  fixed  {fixed}   broken {broken}   discordant {fixed + broken}   "
          f"exact McNemar two-sided p = {p:.3f}")
    if broken_cases:
        # A change that nets positive while breaking a coherent family of phrasings is
        # usually a regression wearing a disguise, so the broken list is always printed.
        print("  broken cases:")
        for record in broken_cases:
            print(f"    [{record['context']}] {record['utterance']!r} -> "
                  f"{record['predicted']}")


def _compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))["cases"]
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))["cases"]

    only_before = {r["key"] for r in before} - {r["key"] for r in after}
    only_after = {r["key"] for r in after} - {r["key"] for r in before}
    if only_before or only_after:
        print(
            f"WARNING: the two runs were scored on different cases "
            f"({len(only_before)} only in before, {len(only_after)} only in after). "
            f"Only the intersection is paired.",
            file=sys.stderr,
        )

    _report_axis("Routing top-1", before, after, "top1")
    _report_axis("Routing in-list", before, after, "in_list")
    _report_axis("Escalation", before, after, "escalated")
    print(
        "\nRouting and escalation are never blended into one number: they trade "
        "against each other (decision D2).\n"
        "Compare p against the noise floor you measured, not against 0.05 alone -- "
        "see the skill's Phase 0."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)

    p_score = sub.add_parser("score", help="Score a benchmark against one artifact set.")
    p_score.add_argument("--workflow", required=True, help="Workflow folder path.")
    p_score.add_argument("--benchmark", required=True, help="Benchmark JSON file.")
    p_score.add_argument("--version", default=None,
                         help="current, previous, or an internal artifact id; default current.")
    p_score.add_argument("--context", dest="contexts", action="append", default=None,
                         help="Limit to this context (repeatable).")
    p_score.add_argument("--out", required=True, help="Where to write per-case verdicts.")
    p_score.set_defaults(func=_score)

    p_cmp = sub.add_parser("compare", help="Paired comparison of two verdict files.")
    p_cmp.add_argument("--before", required=True)
    p_cmp.add_argument("--after", required=True)
    p_cmp.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    if args.action == "score":
        # The trainer's env is not loaded here; CommandRouter only reads files, but
        # get_env_var callers downstream expect fastworkflow to have been initialised.
        fastworkflow.init({})
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
