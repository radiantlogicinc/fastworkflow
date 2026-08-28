#!/usr/bin/env python
"""trace_turn.py — read a turn's command traces from the places fastworkflow
actually records them. READ-ONLY: opens SQLite in immutable mode, never writes.

Since Phase 7 the per-workflow `observability.sqlite3` is the single source of
truth for turns, spans, and artifacts, across ALL topologies (CLI included).
The cwd `action.jsonl` mirror it replaced is GONE — this script no longer reads
it, and a stale `action.jsonl` on disk is residue from a pre-Phase-7 build.

Where the data lives:

1. observability.sqlite3, under
   `$FASTWORKFLOW_STATE_ROOT/workflows/<workflow-id>/` (default state root
   `~/.local/state/fastworkflow`). Written by SQLiteTraceSink: turn records
   sync-first on the turn path, spans/artifacts on a background writer.
   `spans.trace_id` IS the `turns.turn_key`, so one join covers a whole turn.
   Persists across turns and processes — unlike action.jsonl, which only ever
   held the last turn.

   Span names: fw.turn (root), fw.planner.plan / fw.planner.replan,
   fw.agent.tool_call, fw.command.execute, fw.ask_user, fw.nlu.intent,
   fw.nlu.param_extraction, fw.llm.call, fw.train.*.

2. The in-memory action log: WorkflowExecutionContext.append_action_log()
   accumulates per-turn tool-call/ask_user records; the agent's final-answer
   synthesis and teacher/student distillation read it. It is per-turn and
   in-process only (cleared at turn start and between distillation passes) —
   grab it live via ctx.action_log; use this script for anything post-mortem.

3. Live CommandTraceEvent queue (chat_session.command_trace_queue /
   WorkflowExecutionContext.command_trace_queue): AGENT_TO_WORKFLOW and
   WORKFLOW_TO_AGENT events emitted around every tool call, terminated by a None
   sentinel per turn. The CLI drains it to render the dim yellow/green
   "Agent >"/"Workflow >" lines; the FastAPI turns engine drains it into the
   response's "traces" field — read it from the JSON body of
   /invoke_agent | /invoke_assistant | /initialize.

Usage:
    # List recent turns (newest first) from the observability DB
    trace_turn.py turns [--db PATH] [--channel ID] [--limit N]

    # Span tree + artifacts for one turn
    trace_turn.py turn <turn_key> [--db PATH]

    # Pretty-print the traces field of a saved server turn-response body
    #   e.g.: curl -s -X POST .../invoke_agent -H "Authorization: Bearer $T" \
    #              -H 'Content-Type: application/json' \
    #              -d '{"user_query": "..."}' > /tmp/turn.json
    trace_turn.py response /tmp/turn.json

    # Print the capture recipes (where to hook, in code) and exit
    trace_turn.py explain
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

EXPLAIN = """\
How to capture a turn's trace, per topology:

Any topology, post-mortem (preferred since Phase 7):
  trace_turn.py turns                 # find the turn_key
  trace_turn.py turn <turn_key>       # span tree + artifacts
  The DB is $FASTWORKFLOW_STATE_ROOT/workflows/<workflow-id>/observability.sqlite3
  (default root ~/.local/state/fastworkflow). It keeps every turn, so there is
  no longer a copy-it-before-the-next-turn race.

CLI (Topology A):
  * Run `fastworkflow run <wf> <env> <passwords>`; live traces render
    automatically while the spinner runs. CLI turns are recorded under a
    synthetic per-session channel id ("cli:<timestamp>"), so `turns --channel`
    can isolate one session.

In-process (tests / experiment harnesses, Topology B):
  ctx = fastworkflow.WorkflowExecutionContext(...)   # or via ChatSession
  out = ctx.process_turn(message)                    # TurnOutput
  out.turn_key                # developer handle for the logical turn; also the
                              # spans.trace_id in the observability DB
  out.status, out.success     # lifecycle vs all-commands-succeeded (orthogonal)
  out.command_outputs         # per-command CommandOutput provenance
  ctx.action_log              # list[dict], this turn's tool-call/ask_user records
  # Live events instead: inject a queue via
  #   ctx.set_transport_queues(command_trace_queue=queue.Queue())
  # then drain it until the None sentinel.

FastAPI server:
  POST /invoke_agent (or /invoke_assistant, /initialize) and read "traces",
  "turn_key", "status", "success", "command_outputs" from the JSON body.
  For a turn that deferred with 202, or whose "traces" came back empty, read the
  spans from the DB by turn_key instead.
"""


# ---------------------------------------------------------------------------
# DB discovery / access
# ---------------------------------------------------------------------------

def default_state_root() -> Path:
    return Path(
        os.environ.get("FASTWORKFLOW_STATE_ROOT", "~/.local/state/fastworkflow")
    ).expanduser()


def find_dbs() -> list[Path]:
    """Observability DBs under the state root, newest-modified first."""
    root = default_state_root()
    dbs = list(root.glob("workflows/*/observability.sqlite3"))
    return sorted(dbs, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_db(explicit: str | None) -> Path | None:
    """Pick the DB to read, printing why when the choice is ambiguous."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            print(f"No observability DB at {p}", file=sys.stderr)
            return None
        return p
    dbs = find_dbs()
    if not dbs:
        print(
            f"No observability.sqlite3 under {default_state_root()}/workflows/.\n"
            "Run a workflow first, or pass --db. If FASTWORKFLOW_STATE_ROOT was "
            "set for the run, set it here too.",
            file=sys.stderr,
        )
        return None
    if len(dbs) > 1:
        print(
            f"# {len(dbs)} workflows have observability DBs; using the most "
            f"recently written:\n#   {dbs[0]}\n"
            "# (pass --db to pick another)",
            file=sys.stderr,
        )
    return dbs[0]


def connect(path: Path) -> sqlite3.Connection:
    """Open read-only. `immutable=1` also lets us read a DB whose writer holds
    a lock, which matters when inspecting a live run."""
    conn = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# turns / spans
# ---------------------------------------------------------------------------

def cmd_turns(db: Path, channel: str | None, limit: int) -> int:
    with connect(db) as conn:
        sql = (
            "SELECT turn_key, channel_id, conversation_id, ordinal, status, "
            "success, user_message, answer, started_at, completed_at "
            "FROM turns"
        )
        params: list = []
        if channel:
            sql += " WHERE channel_id=?"
            params.append(channel)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    print(f"{db} — {len(rows)} turn(s), newest first:\n")
    if not rows:
        print("  (none — the workflow has not run, or --channel matched nothing)")
        return 0
    for r in rows:
        mark = "OK" if r["success"] else "FAIL"
        conv = (
            f"conv {r['conversation_id']}#{r['ordinal']}"
            if r["conversation_id"] is not None
            else "no conversation (CLI/one-off)"
        )
        print(f"{r['turn_key']}  [{r['status']}/{mark}]  {conv}")
        print(f"    channel : {r['channel_id'] or '(unbound)'}")
        print(f"    started : {r['started_at']}  completed: {r['completed_at']}")
        print(f"    message : {_clip(r['user_message'], 160)}")
        if r["answer"]:
            print(f"    answer  : {_clip(r['answer'], 160)}")
        print()
    return 0


def _clip(value, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


def _span_line(span: sqlite3.Row) -> str:
    if span["end_ns"] is not None:
        duration = f"{(span['end_ns'] - span['start_ns']) / 1e6:.1f}ms"
    else:
        duration = "unfinished"
    parts = [f"{span['name']} [{span['status']}, {duration}]"]
    if span["command_name"]:
        parts.append(f"command={span['command_name']}")
    if span["context"]:
        parts.append(f"context={span['context']}")
    return "  ".join(parts)


def _print_span_tree(spans: list[sqlite3.Row]) -> None:
    """Render parent→child. Orphans (parent written but missing, e.g. dropped by
    the background writer) are printed at the root so nothing is hidden."""
    by_parent: dict[str | None, list[sqlite3.Row]] = {}
    known = {s["span_id"] for s in spans}
    for s in spans:
        parent = s["parent_span_id"]
        if parent is not None and parent not in known:
            parent = None
        by_parent.setdefault(parent, []).append(s)

    def walk(parent: str | None, depth: int) -> None:
        for s in by_parent.get(parent, []):
            print(f"{'  ' * depth}- {_span_line(s)}")
            attributes = _load_json(s["attributes"])
            if attributes:
                for key, value in attributes.items():
                    print(f"{'  ' * depth}    {key}: {_clip(value, 300)}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)


def _load_json(raw):
    try:
        return json.loads(raw) if raw else None
    except (TypeError, json.JSONDecodeError):
        return None


def cmd_turn(db: Path, turn_key: str) -> int:
    with connect(db) as conn:
        turn = conn.execute(
            "SELECT * FROM turns WHERE turn_key=?", (turn_key,)
        ).fetchone()
        # spans.trace_id is the turn_key; spans can exist without a turn row if
        # the turn record write degraded, so query them either way.
        spans = conn.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns", (turn_key,)
        ).fetchall()
        artifacts = conn.execute(
            "SELECT artifact_id, key, content_type, size_bytes, error "
            "FROM artifacts WHERE turn_key=?",
            (turn_key,),
        ).fetchall()

    if turn is None and not spans:
        print(
            f"No turn or spans for turn_key {turn_key!r} in {db}.\n"
            "Run `trace_turn.py turns` to list what is recorded.",
            file=sys.stderr,
        )
        return 1

    if turn is None:
        print(
            "WARNING: spans exist but the turn record is missing — the turn-record "
            "write degraded, or the turn is still in flight.\n"
        )
    else:
        mark = "OK" if turn["success"] else "FAIL"
        print(f"turn_key : {turn['turn_key']}")
        print(f"status   : {turn['status']} / {mark}")
        if turn["failure_reason"]:
            print(f"failure  : {turn['failure_reason']}")
        conv = (
            f"conv {turn['conversation_id']}#{turn['ordinal']}"
            if turn["conversation_id"] is not None
            else "no conversation (CLI/one-off)"
        )
        print(f"channel  : {turn['channel_id'] or '(unbound)'}  {conv}")
        print(f"message  : {_clip(turn['user_message'], 400)}")
        if turn["refined_user_message"]:
            print(f"refined  : {_clip(turn['refined_user_message'], 400)}")
        if turn["answer"]:
            print(f"answer   : {_clip(turn['answer'], 400)}")
        if turn["suspended_ms"]:
            print(f"suspended: {turn['suspended_ms']}ms (ask_user)")
        print()

    print(f"spans ({len(spans)}):")
    if spans:
        _print_span_tree(spans)
    else:
        print("  (none — spans are written by a background writer and are "
              "best-effort; a deterministic turn also emits fewer)")

    if artifacts:
        print(f"\nartifacts ({len(artifacts)}):")
        for a in artifacts:
            detail = f"{a['content_type']}, {a['size_bytes']}B"
            if a["error"]:
                detail += f", ERROR: {a['error']}"
            print(f"  {a['key']}  ({detail})  id={a['artifact_id']}")
    return 0


# ---------------------------------------------------------------------------
# server response bodies
# ---------------------------------------------------------------------------

def cmd_response(path: str) -> int:
    p = Path(path)
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Cannot read {p} as JSON: {e}", file=sys.stderr)
        return 1
    for key in ("turn_key", "exec_state", "status", "success", "failure_reason"):
        if key in body:
            print(f"{key:15s}: {body[key]}")
    answer = body.get("answer")
    if answer:
        print(f"{'answer':15s}: {str(answer)[:300]}")
    traces = body.get("traces") or []
    print(f"\ntraces ({len(traces)}):")
    for i, t in enumerate(traces):
        direction = t.get("direction", "?")
        if direction == "agent_to_workflow":
            print(f"[{i}] Agent -> Workflow: {t.get('raw_command')}")
        else:
            ok = t.get("success")
            mark = "OK" if ok else ("FAIL" if ok is False else "?")
            print(
                f"[{i}] Workflow -> Agent [{mark}]: {t.get('command_name')}, "
                f"{t.get('parameters')}"
            )
            resp = str(t.get("response_text") or "")
            print(f"      {resp[:300]}")
    if not traces:
        print(
            "  (empty — deterministic non-agent turn, traces already drained by an\n"
            "   earlier poll of the same turn_key, or the turn deferred with 202.\n"
            "   Read the spans from the DB instead: trace_turn.py turn <turn_key>)"
        )
    outs = body.get("command_outputs") or []
    if outs:
        print(f"\ncommand_outputs ({len(outs)}):")
        for i, o in enumerate(outs):
            print(f"[{i}] {o.get('command_name') or '(unresolved)'} "
                  f"params={o.get('command_parameters')!r} "
                  f"duration_ms={o.get('duration_ms')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("turns", help="list recent turns from the observability DB")
    t.add_argument("--db", help="path to observability.sqlite3")
    t.add_argument("--channel", help="only turns for this channel_id")
    t.add_argument("--limit", type=int, default=20)

    o = sub.add_parser("turn", help="span tree + artifacts for one turn_key")
    o.add_argument("turn_key")
    o.add_argument("--db", help="path to observability.sqlite3")

    r = sub.add_parser("response", help="pretty-print a saved server turn-response JSON body")
    r.add_argument("path")
    sub.add_parser("explain", help="print the capture recipes and exit")
    args = ap.parse_args()

    if args.cmd == "turns":
        db = resolve_db(args.db)
        return 1 if db is None else cmd_turns(db, args.channel, args.limit)
    if args.cmd == "turn":
        db = resolve_db(args.db)
        return 1 if db is None else cmd_turn(db, args.turn_key)
    if args.cmd == "response":
        return cmd_response(args.path)
    print(EXPLAIN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
