"""Cross-process observability-store concurrency (WAL).

Ported from the legacy per-channel ConversationStore when the Phase-7
consolidation deleted that module (bead `fix-gxr`). The concern moved with the
data and got sharper: one `observability.sqlite3` per workflow now takes ALL the
conversation write traffic for every channel, where the legacy store gave each
channel its own file. Multi-process writers are supported on local filesystems
only — the WAL constraint the design states as the reason the state root must
not be NFS ([R8], observability design §3.2 durability class).

Each worker writes turn records the way the sink's synchronous path does, on its
own conversation, so this exercises concurrent `BEGIN IMMEDIATE` writers against
one DB rather than one writer per file.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import fastworkflow
from fastworkflow import TurnStatus
from fastworkflow import observability_store as obs

CHANNEL = "shared"


def _record(store: obs.ObservabilityStore, conv_id: int, summary: str) -> None:
    turn_result = fastworkflow.TurnResult(
        turn_output=fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="ok",
        ),
        channel_id=CHANNEL,
        conversation_id=conv_id,
        user_message="msg",
        conversation_summary=summary,
    )
    turn_row, artifact_rows = obs.serialize_turn_result(turn_result)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_turn_row(conn, turn_row, artifact_rows, obs.Redactor())
        conn.commit()


def _worker(db_path: str, conv_id: int, n: int, result_path: str) -> None:
    """Each process owns one conversation — concurrent writers on one DB."""
    try:
        store = obs.ObservabilityStore(db_path)
        for i in range(n):
            _record(store, conv_id, f"c{conv_id}-{i}")
            # Interleave reads while other processes write.
            assert store.count_usable_turns(CHANNEL, conv_id) == i + 1
        Path(result_path).write_text("ok", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        Path(result_path).write_text(f"fail:{exc!r}", encoding="utf-8")
        raise SystemExit(1) from exc


def test_observability_store_four_process_wal(tmp_path: Path):
    db_path = str(tmp_path / "observability.sqlite3")
    n_workers = 4
    n_ops = 40

    seed = obs.ObservabilityStore(db_path)
    conv_ids = [seed.mint_conversation_id(CHANNEL) for _ in range(n_workers)]

    result_paths = [str(tmp_path / f"r{i}.txt") for i in range(n_workers)]
    procs = [
        multiprocessing.Process(
            target=_worker, args=(db_path, conv_ids[i], n_ops, result_paths[i])
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, Path(result_paths[procs.index(p)]).read_text(encoding="utf-8")

    for rp in result_paths:
        assert Path(rp).read_text(encoding="utf-8") == "ok"

    store = obs.ObservabilityStore(db_path)
    for conv_id in conv_ids:
        assert store.count_usable_turns(CHANNEL, conv_id) == n_ops
        summaries = [
            entry["conversation summary"]
            for entry in store.get_memory_window(CHANNEL, conv_id, n_ops)
        ]
        assert summaries == [f"c{conv_id}-{i}" for i in range(n_ops)]


def test_concurrent_minting_never_issues_the_same_id_twice(tmp_path: Path):
    """The counter is the identity authority, so contention must not alias ids.

    A MAX-derived mint under concurrency hands two processes the same id, and
    both then write turns into what they each believe is their own conversation
    (ruling C2).
    """
    db_path = str(tmp_path / "observability.sqlite3")
    n_workers = 4
    n_ops = 25

    obs.ObservabilityStore(db_path)  # create the schema before forking
    result_paths = [str(tmp_path / f"m{i}.txt") for i in range(n_workers)]

    def _mint_worker(path: str, result_path: str, count: int) -> None:
        try:
            store = obs.ObservabilityStore(path)
            minted = [store.mint_conversation_id("minters") for _ in range(count)]
            Path(result_path).write_text(
                ",".join(str(i) for i in minted), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            Path(result_path).write_text(f"fail:{exc!r}", encoding="utf-8")
            raise SystemExit(1) from exc

    procs = [
        multiprocessing.Process(
            target=_mint_worker, args=(db_path, result_paths[i], n_ops)
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, Path(result_paths[procs.index(p)]).read_text(encoding="utf-8")

    all_ids: list[int] = []
    for rp in result_paths:
        text = Path(rp).read_text(encoding="utf-8")
        assert not text.startswith("fail:"), text
        all_ids.extend(int(part) for part in text.split(","))

    assert len(all_ids) == n_workers * n_ops
    assert len(set(all_ids)) == len(all_ids), "two processes were handed the same id"
