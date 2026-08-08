"""Cross-process ConversationStore concurrency (WAL)."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from fastworkflow.run_fastapi_mcp.conversation_store import ConversationStore


def _worker(base: str, channel: str, conv_id: int, n: int, result_path: str) -> None:
    """Each process owns one conversation id — exercises concurrent writers on one DB."""
    try:
        store = ConversationStore(channel, base)
        for i in range(n):
            store.append_conversation_turns(
                conv_id,
                [{"conversation summary": f"c{conv_id}-{i}", "i": i}],
            )
            # Interleave reads while other processes write.
            assert store.count_conversation_turns(conv_id) == i + 1
        Path(result_path).write_text("ok", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        Path(result_path).write_text(f"fail:{exc!r}", encoding="utf-8")
        raise SystemExit(1) from exc


def test_conversation_store_four_process_wal(tmp_path: Path):
    base = str(tmp_path / "conversations")
    channel = "shared"
    n_workers = 4
    n_ops = 40

    seed = ConversationStore(channel, base)
    conv_ids = [seed.reserve_next_conversation_id() for _ in range(n_workers)]

    result_paths = [str(tmp_path / f"r{i}.txt") for i in range(n_workers)]
    procs = [
        multiprocessing.Process(
            target=_worker, args=(base, channel, conv_ids[i], n_ops, result_paths[i])
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, Path(result_paths[procs.index(p)]).read_text(encoding="utf-8")

    for rp in result_paths:
        assert Path(rp).read_text(encoding="utf-8") == "ok"

    store = ConversationStore(channel, base)
    assert Path(store.db_path).suffix == ".sqlite3"
    for conv_id in conv_ids:
        assert store.count_conversation_turns(conv_id) == n_ops
        summaries = [
            t["conversation summary"] for t in store.get_conversation(conv_id)["turns"]
        ]
        assert summaries == [f"c{conv_id}-{i}" for i in range(n_ops)]
