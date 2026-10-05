"""Simple benchmark authoring and experiment identities for harness handoff.

Benchmark versions are immutable catalog files. Experiment registrations are
rows of the workflow's live DB (`experiment_registrations`) and can be created
before any evidence exists. A controller binds a registration when it declares
the attempts, never at UI creation.

An experiment enters its comparison group HERE, at creation (`fix-9eg.17.1`),
before any evidence exists, which is what makes the first experiment of a group
its winner at the moment somebody decides to run it rather than at the moment a
runner happens to record it. The contest lives in the control tables of the
workflow's live DB (`observability/control.py`), the same DB the runner later
records the experiment into, so there is one winner per experiment.
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
    write_version,
    _safe_segment,
)
from fastworkflow import state_paths
from fastworkflow.observability import control, selection
from fastworkflow.observability.store import (
    STORE_IDENTITY_DIAGNOSTIC,
    ObservabilityStore,
    open_live_store,
)


class BenchmarkSetupConflict(ValueError):
    pass


class ExperimentDeleted(BenchmarkSetupConflict):
    """A deleted registration must never fall back to an unregistered run."""


class ExperimentSelected(BenchmarkSetupConflict):
    """The current winner is not deleted while others remain; select one first.

    Deleting an unused registration withdraws it from the contest
    (`delete_empty_experiment`), and the one thing that cannot be withdrawn
    while the group has other members is the experiment the contest currently
    names. The alternatives were worse: clearing the pointer files a "nobody
    won" nobody decided, and electing a successor decides a contest on the
    user's behalf during an unrelated deletion. So this is refused — a 409 at
    the HTTP edge — and the user promotes another experiment first.

    It applies to the automatic first winner too. Nobody chose it, but it is
    still the experiment every read reports right now.

    The group's ONLY member is not refused (`fix-65ik`): there is nobody to
    promote instead, so the refusal had no way out, and removing it leaves an
    empty contest rather than an undecided one. Its next experiment is elected
    automatically, the way the first one was.
    """


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
    a second contest, each with its own uncontested winner. One helper, called
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


def benchmark_winner(workflow_path, benchmark_id):
    """The current winner of this benchmark's contest, or None.

    Read-only, like `workflow_winner`. A contest is one group per benchmark
    lineage, across versions, so the page that lists every experiment of that
    lineage can mark the one the pointer names without opening each experiment.
    Inspecting a benchmark that has never entered a contest writes nothing.
    """
    benchmark_id = str(benchmark_id or "").strip()
    if not benchmark_id:
        return None
    control = workflow_control(workflow_path)
    name = workflow_name_for(workflow_path)
    groups = [
        group
        for group in control.list_groups()
        if group.get("group_kind") == selection.GROUP_BENCHMARK
        and group.get("benchmark_id") == benchmark_id
    ]
    named = [group for group in groups if group.get("workflow_name") == name]
    # A renamed workflow folder leaves the old name on the group. One such
    # group is still this contest; two, with neither matching, is ambiguous
    # and marking either experiment would be a guess.
    group = named[0] if named else (groups[0] if len(groups) == 1 else None)
    if group is None:
        return None
    return control.current_winner(str(group["group_id"]))


def is_sole_group_member(workflow_path, experiment_id):
    """True when this experiment is the only member of its comparison group.

    Read-only, like `workflow_winner`. It is the condition under which the
    current winner may be deleted (`ExperimentSelected`), asked separately so a
    screen can offer exactly what the deletion accepts.
    """
    control = workflow_control(workflow_path)
    group = control.group_for_experiment(experiment_id)
    if group is None:
        return False
    members = control.group_members(str(group["group_id"]))
    return [str(row["experiment_id"]) for row in members] == [experiment_id]


def save_benchmark(workflow_path, body):
    """Generate benchmark/version/new task identities; retain existing task IDs."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    title, description = body.get("title"), body.get("description", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(description, str):
        raise ValueError("description must be text")
    raw = body.get("tasks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("add at least one task")
    benchmark_id = body.get("benchmark_id") or f"benchmark-{uuid.uuid4().hex}"
    _safe_segment(benchmark_id, "benchmark_id")
    versions = list_versions(workflow_path, benchmark_id)
    latest = versions[-1] if versions else None
    if body.get("expected_version") != latest:
        raise BenchmarkSetupConflict("benchmark changed; reopen before saving")
    prior = load_version(workflow_path, benchmark_id, latest) if latest else None
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
        task_id = task_id or f"task_{uuid.uuid4().hex}"
        if task_id in seen:
            raise ValueError("duplicate task ID")
        seen.add(task_id)
        old = known.get(task_id, {})
        tasks.append(
            {
                "task_id": task_id,
                "prompt": item.get("prompt", ""),
                "description": old.get("description", ""),
                "payload": old.get("payload", {}),
            }
        )
    numbers = [int(v[1:]) for v in versions if re.fullmatch(r"v\d+", v)]
    version = f"v{max(numbers, default=0) + 1}"
    # `write_version` creates the file exclusively, so of two saves against the
    # same `expected_version` exactly one writes the next version.
    try:
        return write_version(
            workflow_path,
            {
                "benchmark_id": benchmark_id,
                "version": version,
                "title": title.strip(),
                "description": description,
                "tasks": tasks,
            },
        )
    except BenchmarkAlreadyExistsError as exc:
        raise BenchmarkSetupConflict("benchmark changed; reopen before saving") from exc


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
    db_path = _live_db_path(workflow_path)
    store = (
        open_live_store(db_path, write=True)
        if os.path.isfile(db_path)
        else ObservabilityStore(db_path)
    )
    with control.write(store) as conn:
        conn.execute(
            """INSERT INTO experiment_registrations
               (experiment_id, benchmark_id, benchmark_version,
                benchmark_digest_sha256, description, task_ids_json,
                runs_per_task, source_experiment_id, changed_fields_json,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["experiment_id"], record["benchmark_id"],
                record["benchmark_version"], record["benchmark_digest_sha256"],
                record["description"], json.dumps(record["task_ids"]),
                record["runs_per_task"], record["source_experiment_id"],
                json.dumps(record["changed_fields"]), record["created_at"],
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


def create_experiment(
    workflow_path, benchmark_id, version, description="", runs_per_task=1
):
    """Mint an experiment identity; the description is the author's, optional.

    Creation stays one click: the description is free text the author may fill
    in later through `update_experiment_description`, for as long as the
    registration has not been handed to a runner. Copying the benchmark title
    in as a default would put a description on every experiment nobody wrote.

    `runs_per_task` is the existing declared-attempt count, surfaced at setup.
    `n > 1` needs no target, rubric, hypothesis or review gate: it is the same
    task and the same setup run n times, which the runner already does.

    The experiment enters this workflow's comparison group here — before any
    evidence exists — so the first one becomes the group's winner the moment it
    is created. Nothing paid or side-effecting starts; this mints an identity.
    """
    manifest = load_version(workflow_path, benchmark_id, version)
    if not isinstance(description, str):
        raise ValueError("description must be text")
    runs_per_task = validate_runs_per_task(runs_per_task)
    experiment_id = f"exp-{uuid.uuid4().hex}"
    record = {
        "experiment_id": experiment_id,
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "benchmark_digest_sha256": manifest["digest_sha256"],
        "description": description.strip(),
        "task_ids": [task["task_id"] for task in manifest["tasks"]],
        "runs_per_task": runs_per_task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "store": None,
        "source_experiment_id": None,
        "changed_fields": [],
    }
    return _register(workflow_path, record)


def duplicate_experiment(
    workflow_path, source_experiment_id, *, description=None, runs_per_task=None
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
    experiment_id = f"exp-{uuid.uuid4().hex}"
    # Copy the source wholesale, then overwrite exactly the fields that must
    # differ.
    record = dict(source)
    record.update(
        {
            "experiment_id": experiment_id,
            "description": description.strip(),
            "runs_per_task": runs_per_task,
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
    version is touched.

    The registration also LEAVES THE CONTEST it joined at creation (`fix-jfy5`),
    in the same transaction: creation registers and may elect, so a deletion
    that only tombstoned would leave a group pointing at an experiment that no
    longer exists — reported as the winner, unresolvable, and impossible to
    duplicate. A refused deletion — the current winner of a group with other
    members, or a registration with evidence — leaves both the registration and
    the contest exactly as they were. Nothing here elects anybody: the winner
    can be the experiment being deleted only when it is the group's sole member
    (`fix-65ik`), and then there is nobody to succeed it — the group is left
    empty and winner-less, and its next experiment is elected on registration.
    """
    with _registration_write(workflow_path, experiment_id) as (store, conn, row):
        if row["state"] == "bound":
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner and cannot be deleted."
            )
        try:
            selection.SelectionControlStore(store).retire_in_txn(
                conn,
                experiment_id,
                rationale="its registration was deleted",
                allow_sole_winner=True,
            )
        except selection.SelectionRetirementRefused as exc:
            if exc.reason == "is_current_winner":
                raise ExperimentSelected(
                    "This experiment is the current winner of its comparison "
                    "group, and the group has other experiments in it. Deleting "
                    "it would leave them without a winner anybody chose, so it is "
                    "refused: promote one of the other experiments to winner "
                    "first, then delete this one."
                ) from exc
            raise BenchmarkSetupConflict(
                "This experiment is part of the recorded contest and cannot be deleted."
            ) from exc
        conn.execute(
            "UPDATE experiment_registrations SET state='deleted', deleted_at=? "
            "WHERE experiment_id=?",
            (datetime.now(timezone.utc).isoformat(), experiment_id),
        )
    return _record(row)
