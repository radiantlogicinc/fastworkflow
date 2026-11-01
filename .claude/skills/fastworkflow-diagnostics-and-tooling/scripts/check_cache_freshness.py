#!/usr/bin/env python
"""check_cache_freshness.py — recompute the v2.22.1 source fingerprint and
compare it to the fingerprints stamped inside ___command_info/*.json.

Background (commit b5747df, v2.22.1): command_directory.json and
routing_definition.json are derived snapshots of the _commands/ tree. Each is
stamped with `source_fingerprint` = sha256 over the sorted list of
(absolute_path, size, mtime_ns) of every contributing source file:
  * <workflow>/_commands/**/*.py (recursively)
  * _commands/**/context_inheritance_model.json
  * the same for every base workflow named in workflow_inheritance_model.json
At load time fastworkflow trusts a snapshot ONLY if its stamp matches the
recomputed fingerprint; otherwise it silently rebuilds the JSON (cheap).
The trained MODELS (tinymodel.pth etc.) are NOT fingerprint-guarded — a stale
fingerprint tells you the JSONs will rebuild, and it is also your cue that the
command set may have drifted from what the classifier was trained on.

This script explains the verdict instead of just printing hashes:
  FRESH   — stamp == live fingerprint; snapshots will be trusted as-is
  STALE   — mismatch; next fastworkflow load rebuilds the JSON snapshots.
            Models are only retrained by an explicit `fastworkflow train`.
  LEGACY  — snapshot exists but has no source_fingerprint field (pre-v2.22.1);
            treated as not-fresh, forces rebuild
  MISSING — snapshot file absent (workflow never built, or info dir deleted)

Because the fingerprint embeds ABSOLUTE paths and mtime_ns, all of these
invalidate it even with byte-identical sources: moving the workflow directory,
a fresh git checkout, Docker COPY, `touch`. That is safe-by-design (a spurious
rebuild costs ~seconds, a trusted-stale snapshot costs an import error) but it
explains most "why did it rebuild?" surprises.

Usage:
    .venv/bin/python .claude/skills/fastworkflow-diagnostics-and-tooling/scripts/check_cache_freshness.py <workflow_dir> [--json]

READ-ONLY: never writes; avoids fastworkflow helpers that mkdir.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SNAPSHOTS = ("command_directory.json", "routing_definition.json")


def stamped_fingerprint(path: Path):
    """Return (verdict_if_unreadable_or_missing, fingerprint)."""
    if not path.is_file():
        return "MISSING", None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"UNREADABLE ({type(e).__name__})", None
    fp = data.get("source_fingerprint")
    return (None, fp) if fp else ("LEGACY", None)


def contributing_sources(workflow_dir: str):
    """Re-derive the file set the fingerprint covers, using the same package
    helpers the runtime uses (no reimplementation of the selection logic)."""
    from fastworkflow.command_directory import (
        _CONTEXT_MODEL_BASENAMES,
        _command_source_roots,
    )

    wf = Path(workflow_dir).resolve()
    entries = []
    for root in _command_source_roots(str(wf)):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and (p.suffix == ".py" or p.name in _CONTEXT_MODEL_BASENAMES):
                st = p.stat()
                entries.append((str(p), st.st_size, st.st_mtime_ns))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow_dir", help="path to the workflow folder (contains _commands/)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    wf = Path(args.workflow_dir).resolve()
    if not (wf / "_commands").is_dir():
        print(f"ERROR: {wf} has no _commands/ — not a fastworkflow workflow dir", file=sys.stderr)
        return 1

    # Import late: `import fastworkflow` takes ~5-10s.
    from fastworkflow.command_directory import compute_commands_source_fingerprint

    live = compute_commands_source_fingerprint(str(wf))
    sources = contributing_sources(str(wf))
    newest = sorted(sources, key=lambda t: t[2], reverse=True)[:5]

    info_dir = wf / "___command_info"
    result = {
        "workflow": str(wf),
        "live_fingerprint": live,
        "contributing_source_files": len(sources),
        "newest_sources": [
            {
                "path": p,
                "mtime": datetime.fromtimestamp(ns / 1e9).isoformat(timespec="seconds"),
            }
            for p, _sz, ns in newest
        ],
        "snapshots": {},
    }

    for name in SNAPSHOTS:
        verdict, fp = stamped_fingerprint(info_dir / name)
        if verdict is None:
            verdict = "FRESH" if fp == live else "STALE"
        result["snapshots"][name] = {"verdict": verdict, "stamped_fingerprint": fp}

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
        return 0

    print(f"Workflow: {result['workflow']}")
    print(f"Live fingerprint : {live}")
    print(f"Covers {len(sources)} source files; 5 most recently modified:")
    for e in result["newest_sources"]:
        print(f"  {e['mtime']}  {e['path']}")
    print()
    for name, s in result["snapshots"].items():
        print(f"{name}: {s['verdict']}")
        if s["stamped_fingerprint"]:
            print(f"  stamped: {s['stamped_fingerprint']}")
    verdicts = {s["verdict"] for s in result["snapshots"].values()}
    print()
    if verdicts == {"FRESH"}:
        print("Verdict: snapshots are trusted as-is; no rebuild on next load.")
    elif "MISSING" in verdicts:
        print(
            "Verdict: snapshot(s) missing — the workflow was never built/trained "
            "here, or ___command_info was deleted. First load will build the JSONs; "
            "intent models still require `fastworkflow train`."
        )
    else:
        print(
            "Verdict: snapshot(s) will be REBUILT on next fastworkflow load (cheap,\n"
            "automatic, seconds). If you also changed/added/removed commands since\n"
            "the last `fastworkflow train`, the intent classifiers are now trained\n"
            "on an outdated label set — retrain. If sources are byte-identical and\n"
            "this still says STALE, suspect moved paths or refreshed mtimes\n"
            "(fresh checkout, Docker COPY, touch): rebuild is spurious but harmless."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
