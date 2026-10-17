"""Simple benchmark authoring and experiment identities for harness handoff.

Benchmark versions are immutable catalog files. Experiment registrations are
rows of the workflow's live DB (`experiment_registrations`) and can be created
before any evidence exists. A controller binds a registration when it declares
the attempts, never at UI creation.

An experiment enters its comparison group HERE, at creation, before any
evidence exists. There is one group per benchmark version (`fix-873a`), and a
group has at most one winner, which somebody sets or removes. The contest lives
in the control tables of the workflow's live DB (`observability/control.py`),
the same DB the runner later records the experiment into.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from fastworkflow.benchmark.catalog import (
    BenchmarkAlreadyExistsError,
    list_versions,
    load_version,
    replace_draft_version,
    set_details,
    write_version,
    _safe_segment,
)
from fastworkflow import state_paths
from fastworkflow.observability import control, selection
from fastworkflow.observability.store import (
    STORE_IDENTITY_DIAGNOSTIC,
    open_live_store,
    open_live_store_for_write,
)


class BenchmarkSetupConflict(ValueError):
    pass


class ExperimentDeleted(BenchmarkSetupConflict):
    """A deleted registration must never fall back to an unregistered run."""


# Runs per task. Bounded because the only thing between this number and n real
# executions is somebody's typing: 1000 is not a repeat count, it is a bill.
MIN_RUNS_PER_TASK = 1
MAX_RUNS_PER_TASK = 100


def validate_runs_per_task(value, *, field: str = "runs_per_task") -> int:
    """A positive, bounded, EXACT integer.

    `bool` is an `int` in Python and `int(2.9)` is 2, so the ordinary
    conversion would read `True` as one run and `2.9` as two — a repeat count
    nobody asked for, spent on real model calls. Decimal text is accepted
    because an HTTP query carries numbers as text; `"2.0"` is not.
    """
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{field} must be a whole number, not {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"{field} must be a whole number, not {value!r}")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be a whole number, not {value!r}")
    if value < MIN_RUNS_PER_TASK or value > MAX_RUNS_PER_TASK:
        raise ValueError(
            f"{field} must be between {MIN_RUNS_PER_TASK} and "
            f"{MAX_RUNS_PER_TASK}, not {value}"
        )
    return value


# ----------------------------------------------------------------------
# The workflow's contest, in its live DB
# ----------------------------------------------------------------------


def workflow_name_for(workflow_path):
    """The name a comparison group is scoped by.

    Registration and execution MUST agree on this string: it is half of the
    derived group id, so a mismatch does not raise anything — it quietly opens
    a second contest for the same benchmark version. One helper, called
    from both sides, is the cheapest way to make that impossible.
    """
    return os.path.basename(os.path.abspath(str(workflow_path)).rstrip("/\\"))


def _live_db_path(workflow_path):
    return state_paths.observability_db(str(workflow_path))


def workflow_control(workflow_path, *, write=False):
    """This workflow's contest, in its live DB's control tables.

    Never creates the DB. A read with no live DB answers every question empty,
    because showing a winner must not bring a database into existence for a
    workflow where nobody has recorded anything; a write refuses instead
    (`control.ControlUnavailable`).
    """
    return selection.SelectionControlStore(
        open_live_store(_live_db_path(workflow_path), write=write)
    )


def workflow_winner(workflow_path, experiment_id):
    """The current winner of the group this experiment is in, or None.

    Read-only: it never creates or writes the live DB, so inspecting a workflow
    where nobody has decided anything leaves it exactly as it was.
    """
    return workflow_control(workflow_path).winner_for_experiment(experiment_id)


def benchmark_winner(workflow_path, benchmark_id, benchmark_version):
    """The current winner of one benchmark version's contest, or None.

    Read-only, like `workflow_winner`. Each version is its own contest
    (`fix-873a`), so the page listing one version's experiments can mark the
    one its pointer names without opening each experiment.
    """
    benchmark_id = str(benchmark_id or "").strip()
    benchmark_version = str(benchmark_version or "").strip()
    if not benchmark_id or not benchmark_version:
        return None
    group = selection.comparison_group_identity({
        "workflow_name": workflow_name_for(workflow_path),
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
    })
    return workflow_control(workflow_path).current_winner(group["group_id"])


def save_benchmark(workflow_path, body):
    """Generate benchmark/version/new task identities; retain existing task IDs.

    Only tasks are frozen with a version. A save that leaves the latest
    version's tasks as they are updates the title (one per benchmark) and that
    version's description in place. Blank task rows are skipped. A latest
    version with no tasks is a draft: saving rewrites it in place instead of
    creating the next version.
    """
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    title, description = body.get("title"), body.get("description", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    raw = body.get("tasks", [])
    if not isinstance(raw, list):
        raise ValueError("tasks must be a list")
    benchmark_id = body.get("benchmark_id") or f"benchmark-{uuid.uuid4().hex}"
    _safe_segment(benchmark_id, "benchmark_id")
    versions = list_versions(workflow_path, benchmark_id)
    latest = versions[-1] if versions else None
    if body.get("expected_version") != latest:
        raise BenchmarkSetupConflict("benchmark changed; reopen before saving")
    prior = load_version(workflow_path, benchmark_id, latest) if latest else None
    if "description" not in body and prior is not None:
        description = prior["description"]
    known = {t["task_id"]: t for t in prior["tasks"]} if prior else {}
    tasks, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(
            item.get("prompt", ""), str
        ):
            raise ValueError("each task prompt must be text (or omitted)")
        task_id = item.get("task_id")
        if task_id is not None and task_id not in known:
            raise ValueError("new task IDs are assigned automatically")
        old = known.get(task_id, {})
        # A blank row is skipped; a payload-only task (no prompt) is kept.
        if not item.get("prompt", "").strip() and not old.get("payload"):
            continue
        task_id = task_id or f"task_{uuid.uuid4().hex}"
        if task_id in seen:
            raise ValueError("duplicate task ID")
        seen.add(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "prompt": item.get("prompt", ""),
                "description": old.get("description", ""),
                "payload": old.get("payload", {}),
            }
        )
    spec = {
        "benchmark_id": benchmark_id,
        "title": title.strip(),
        "description": description,
        "tasks": tasks,
    }
    if prior is not None and _task_key(tasks) == _task_key(prior["tasks"]):
        return update_benchmark_details(
            workflow_path, benchmark_id, latest, spec["title"], description
        )
    try:
        if prior is not None and not prior["tasks"]:
            version = latest
            replace_draft_version(
                workflow_path, {**spec, "version": version}, prior["digest_sha256"]
            )
        else:
            numbers = [int(v[1:]) for v in versions if re.fullmatch(r"v\d+", v)]
            version = f"v{max(numbers, default=0) + 1}"
            # `write_version` inserts exclusively, so of two saves against the
            # same `expected_version` exactly one writes the next version.
            write_version(workflow_path, {**spec, "version": version})
    except BenchmarkAlreadyExistsError as exc:
        raise BenchmarkSetupConflict("benchmark changed; reopen before saving") from exc
    return update_benchmark_details(
        workflow_path, benchmark_id, version, spec["title"], description
    )


def _task_key(tasks):
    return [(t["task_id"], t.get("prompt", ""), t.get("description", ""),
             t.get("payload", {})) for t in tasks]


def update_benchmark_details(workflow_path, benchmark_id, version, title, description):
    """Set the benchmark's title and one version's description; tasks untouched.

    Any version may be described, not only the latest. Returns that version
    as it now reads.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    load_version(workflow_path, benchmark_id, version)  # it must exist
    set_details(workflow_path, benchmark_id, title=title.strip(),
                version=version, description=description)
    return load_version(workflow_path, benchmark_id, version)


# ----------------------------------------------------------------------
# Registrations, in the live DB's `experiment_registrations`
# ----------------------------------------------------------------------

_SELECT_REGISTRATIONS = (
    "SELECT *, (SELECT value FROM diagnostics WHERE key=?) AS store_id "
    "FROM experiment_registrations "
)


def _record(row):
    """The registration as the API serves it; `store` is set once bound."""
    return {
        "experiment_id": row["experiment_id"],
        "benchmark_id": row["benchmark_id"],
        "benchmark_version": row["benchmark_version"],
        "benchmark_digest_sha256": row["benchmark_digest_sha256"],
        "description": row["description"],
        "task_ids": json.loads(row["task_ids_json"]),
        "runs_per_task": row["runs_per_task"],
        "user_simulator": bool(row["user_simulator"]),
        "created_at": row["created_at"],
        "store": {"store_id": row["store_id"]} if row["state"] == "bound" else None,
        "source_experiment_id": row["source_experiment_id"],
        "changed_fields": json.loads(row["changed_fields_json"]),
    }


def _live(row, experiment_id):
    if row is None:
        raise KeyError(experiment_id)
    if row["state"] == "deleted":
        raise ExperimentDeleted("This experiment was deleted. Create a new experiment.")
    return row


def _registrations(workflow_path, where, params):
    """Registration rows, read-only; none when the workflow has no live DB."""
    return control.rows(
        open_live_store(_live_db_path(workflow_path)),
        _SELECT_REGISTRATIONS + where,
        (STORE_IDENTITY_DIAGNOSTIC, *params),
    )


def _register(workflow_path, record):
    """Insert a new registration and join its contest, in one transaction.

    The one control write allowed to create the live DB: creating an
    experiment is a POST, and a workflow nobody has chatted with yet has no DB
    to register it in. Reads still never create one.
    """
    store = open_live_store_for_write(_live_db_path(workflow_path))
    with control.write(store) as conn:
        conn.execute(
            """INSERT INTO experiment_registrations
               (experiment_id, benchmark_id, benchmark_version,
                benchmark_digest_sha256, description, task_ids_json,
                runs_per_task, source_experiment_id, changed_fields_json,
                user_simulator, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["experiment_id"], record["benchmark_id"],
                record["benchmark_version"], record["benchmark_digest_sha256"],
                record["description"], json.dumps(record["task_ids"]),
                record["runs_per_task"], record["source_experiment_id"],
                json.dumps(record["changed_fields"]),
                int(record["user_simulator"]), record["created_at"],
            ),
        )
        selection.SelectionControlStore(store).join_in_txn(
            conn,
            record["experiment_id"],
            dict(record, workflow_name=workflow_name_for(workflow_path)),
        )
    return record


@contextmanager
def _registration_write(workflow_path, experiment_id):
    """One `BEGIN IMMEDIATE` on a live registration: (store, conn, row)."""
    db_path = _live_db_path(workflow_path)
    if not os.path.isfile(db_path):
        raise KeyError(experiment_id)
    store = open_live_store(db_path, write=True)
    with control.write(store) as conn:
        row = conn.execute(
            "SELECT *, NULL AS store_id FROM experiment_registrations "
            "WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        yield store, conn, _live(row, experiment_id)


def validate_user_simulator(value) -> bool:
    """A JSON boolean; anything else is refused rather than guessed at."""
    if not isinstance(value, bool):
        raise ValueError("user_simulator must be true or false")
    return value


def create_experiment(
    workflow_path, benchmark_id, version, description="", runs_per_task=1,
    user_simulator=False,
):
    """Mint an experiment identity; the description is the author's, optional.

    Creation stays one click: the description is free text the author may fill
    in later through `update_experiment_description`, for as long as the
    registration has not been handed to a runner. Copying the benchmark title
    in as a default would put a description on every experiment nobody wrote.

    `runs_per_task` is the existing declared-attempt count, surfaced at setup.
    `n > 1` needs no target, rubric, hypothesis or review gate: it is the same
    task and the same setup run n times, which the runner already does.

    `user_simulator` says a user simulator will drive every run, turning each
    task prompt into its own utterances (`fix-va9v`). It is fixed here: the
    runner takes it from the registration.

    The experiment enters its benchmark version's comparison group here, before
    any evidence exists. Nobody becomes the winner by being created. Nothing
    paid or side-effecting starts; this mints an identity.
    """
    manifest = load_version(workflow_path, benchmark_id, version)
    if not manifest["tasks"]:
        raise ValueError("add tasks to this benchmark before creating an experiment")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    runs_per_task = validate_runs_per_task(runs_per_task)
    user_simulator = validate_user_simulator(user_simulator)
    experiment_id = f"exp-{uuid.uuid4().hex}"
    record = {
        "experiment_id": experiment_id,
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "benchmark_digest_sha256": manifest["digest_sha256"],
        "description": description.strip(),
        "task_ids": [task["task_id"] for task in manifest["tasks"]],
        "runs_per_task": runs_per_task,
        "user_simulator": user_simulator,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "store": None,
        "source_experiment_id": None,
        "changed_fields": [],
    }
    return _register(workflow_path, record)


def duplicate_experiment(
    workflow_path, source_experiment_id, *, description=None, runs_per_task=None,
    user_simulator=None,
):
    """Run the same setup again, as a new experiment. The candidate action.

    The whole point is that nobody retypes anything: the benchmark, the pinned
    version and digest, the task set and the repeat count all come from the
    source experiment, and only what the caller explicitly passes differs. The
    identity is regenerated, because a second run of the same setup is a second
    experiment and sharing an id would make one overwrite the other's evidence.

    `source_experiment_id` and `changed_fields` are kept on the new record so a
    UI can show "from the current winner, with runs per task 1 → 3" without
    recomputing a diff, and so a reader months later can see what was varied.

    Registration only. No attempt is declared, no runner is started, nothing
    paid happens here.
    """
    source, _manifest = experiment_manifest(workflow_path, source_experiment_id)
    changed = []
    if description is None:
        description = source.get("description", "")
    else:
        if not isinstance(description, str):
            raise ValueError("description must be text")
        if description.strip() != source.get("description", ""):
            changed.append("description")
    if runs_per_task is None:
        runs_per_task = validate_runs_per_task(source.get("runs_per_task", 1))
    else:
        runs_per_task = validate_runs_per_task(runs_per_task)
        if runs_per_task != source.get("runs_per_task", 1):
            changed.append("runs_per_task")
    if user_simulator is None:
        user_simulator = bool(source.get("user_simulator"))
    else:
        user_simulator = validate_user_simulator(user_simulator)
        if user_simulator != bool(source.get("user_simulator")):
            changed.append("user_simulator")
    experiment_id = f"exp-{uuid.uuid4().hex}"
    # Copy the source wholesale, then overwrite exactly the fields that must
    # differ.
    record = dict(source)
    record.update(
        {
            "experiment_id": experiment_id,
            "description": description.strip(),
            "runs_per_task": runs_per_task,
            "user_simulator": user_simulator,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "store": None,
            "source_experiment_id": source["experiment_id"],
            "changed_fields": sorted(changed),
        }
    )
    return _register(workflow_path, record)


def update_experiment_description(workflow_path, experiment_id, description):
    """Edit the description while the registration is still the only record.

    Refused once a runner has bound the registration: from that point the
    description the run declared lives in its evidence store, and a
    registration edited afterwards would disagree with a store nothing here
    can write.
    """
    if not isinstance(description, str):
        raise ValueError("description must be text")
    with _registration_write(workflow_path, experiment_id) as (_store, conn, row):
        if row["state"] == "bound":
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner; its description "
                "is now part of the recorded run."
            )
        conn.execute(
            "UPDATE experiment_registrations SET description=? WHERE experiment_id=?",
            (description.strip(), experiment_id),
        )
    return dict(_record(row), description=description.strip())


def load_experiment(workflow_path, experiment_id):
    found = _registrations(workflow_path, "WHERE experiment_id=?", (experiment_id,))
    return _record(_live(found[0] if found else None, experiment_id))


def registered_experiments(workflow_path, benchmark_id=None):
    """Live registrations of one benchmark, or of every benchmark when None."""
    where, params = "WHERE state<>'deleted' ", ()
    if benchmark_id is not None:
        where, params = where + "AND benchmark_id=? ", (benchmark_id,)
    return [
        _record(row)
        for row in _registrations(workflow_path, where + "ORDER BY experiment_id", params)
    ]


def experiment_manifest(workflow_path, experiment_id):
    record = load_experiment(workflow_path, experiment_id)
    manifest = load_version(
        workflow_path, record["benchmark_id"], record["benchmark_version"]
    )
    if manifest["digest_sha256"] != record["benchmark_digest_sha256"]:
        raise BenchmarkSetupConflict(
            "benchmark contents no longer match this experiment's pinned version"
        )
    return record, manifest


def bind_experiment(workflow_path, experiment_id):
    """Mark a registration as handed to a runner; refused once deleted."""
    with _registration_write(workflow_path, experiment_id) as (_store, conn, row):
        if row["state"] == "registered":
            conn.execute(
                "UPDATE experiment_registrations SET state='bound', bound_at=? "
                "WHERE experiment_id=?",
                (datetime.now(timezone.utc).isoformat(), experiment_id),
            )


def delete_empty_experiment(workflow_path, experiment_id):
    """Remove an unused registration, serialized against a runner's binding.

    The tombstone (`state='deleted'`) prevents delayed runners from treating a
    deleted ID as a new, unregistered experiment. No evidence or benchmark
    version is touched. The registration leaves its contest in the same
    transaction; when it was the winner, the contest has no winner afterwards.
    """
    with _registration_write(workflow_path, experiment_id) as (store, conn, row):
        if row["state"] == "bound":
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner and cannot be deleted."
            )
        selection.SelectionControlStore(store).leave_in_txn(conn, experiment_id)
        conn.execute(
            "UPDATE experiment_registrations SET state='deleted', deleted_at=? "
            "WHERE experiment_id=?",
            (datetime.now(timezone.utc).isoformat(), experiment_id),
        )
    return _record(row)
