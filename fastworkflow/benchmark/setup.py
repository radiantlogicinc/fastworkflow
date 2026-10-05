"""Simple benchmark authoring and experiment identities for harness handoff.

Benchmark versions are immutable catalog files. Experiment registrations can be
created before an execution store exists. A controller binds a registration to
its actual evidence store when it declares the attempts, never at UI creation.

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
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from fastworkflow.benchmark.catalog import (
    benchmarks_root,
    list_versions,
    load_version,
    write_version,
    _safe_segment,
)
from fastworkflow import state_paths
from fastworkflow.observability import selection
from fastworkflow.observability.store import open_live_store
from fastworkflow.utils.logging import logger


class BenchmarkSetupConflict(ValueError):
    pass


class ExperimentDeleted(BenchmarkSetupConflict):
    """A deleted registration must never fall back to an unregistered run."""


class ExperimentSelected(BenchmarkSetupConflict):
    """The current winner is not deleted while others remain; select one first.

    Deleting an unused registration withdraws it from the contest
    (`retire_experiment_selection`), and the one thing that cannot be withdrawn
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



@contextmanager
def _lock(workflow_path):
    # The project targets Unix; flock coordinates HTTP threads and harness processes.
    import fcntl

    root = benchmarks_root(workflow_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


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


def _reference_for(workflow_path, record):
    return selection.ExperimentReference(
        experiment_id=record["experiment_id"],
        workflow_name=workflow_name_for(workflow_path),
        benchmark_id=record.get("benchmark_id"),
        benchmark_version=record.get("benchmark_version"),
        benchmark_digest_sha256=record.get("benchmark_digest_sha256"),
        created_at=record.get("created_at"),
    )


def register_experiment_selection(workflow_path, record, *, allow_initial_winner=True):
    """Enter a registration in the workflow contest; the first one wins.

    Runs at creation, before the runner has recorded anything: an experiment
    is a decision to run something long before a runner records it, and the
    first experiment of a group should be its winner from that moment. Best
    effort by design — a live DB that cannot be written, or that this workflow
    does not have yet, must not stop somebody creating an experiment — so every
    problem is reported and logged rather than raised. The runner joins the
    same group when it records the experiment.

    `record` is the registration the CALLER holds, which is why this takes the
    setup lock and re-reads by id instead of trusting it. `create_experiment`
    writes the file, releases the lock and only then arrives here; a deletion
    in that gap has already withdrawn the experiment, and writing the caller's
    cached copy would put the tombstoned id straight back into the contest —
    and make it promotable again (`fix-jfy5`).
    """
    with _lock(workflow_path):
        return _register_selection_locked(
            workflow_path,
            record["experiment_id"],
            allow_initial_winner=allow_initial_winner,
        )


def _register_selection_locked(workflow_path, experiment_id, *, allow_initial_winner=True):
    """`register_experiment_selection` with the setup lock ALREADY held.

    Separate because the lock is a plain `flock` and is NOT reentrant: the
    deletion path is inside it when it needs to undo a withdrawal, and calling
    the public function from there would deadlock against itself rather than
    fail visibly.

    Returns None for a registration that is gone or tombstoned, which is the
    refusal that matters here: a deleted id must not come back as a member,
    because membership is what a later promotion is validated against.
    """
    try:
        record = load_experiment(workflow_path, experiment_id)
    except (KeyError, ExperimentDeleted, ValueError):
        return None
    try:
        return workflow_control(workflow_path, write=True).register_experiment_reference(
            _reference_for(workflow_path, record),
            allow_initial_winner=allow_initial_winner,
        )
    except Exception as exc:  # creation must not fail on a control problem
        logger.warning(
            f"observability: experiment {experiment_id!r} was created "
            f"but not registered for winner selection: {exc}"
        )
        return None


def retire_experiment_selection(workflow_path, experiment_id):
    """Take a registration that is being deleted OUT of the workflow contest.

    The other half of `register_experiment_selection`, and the thing whose
    absence was `fix-jfy5`: creation registers and may elect, deletion only
    tombstoned the JSON, so a group could be left pointing at an experiment
    that no longer exists — reported as the winner, unresolvable, and
    impossible to duplicate.

    NOT best effort, unlike registration, and that asymmetry is deliberate. A
    registration that fails to enter the contest is visibly winner-less; a
    deletion that fails to leave it is invisible, and the stale pointer it
    leaves behind is exactly this bug. So a control that exists and cannot be
    updated fails the deletion instead, with nothing changed. A workflow with
    no live DB at all has nothing to retire and nothing to fail: `None`.

    Raises `ExperimentSelected` (409) when the experiment is the group's
    current winner and other experiments are still in the group — see that
    class for why deletion does not get to move a winner pointer. The winner
    that is the group's ONLY member is withdrawn along with the pointer; the
    result's `winner_retired` says so, which is what the deletion's
    compensation needs to know.
    """
    if not os.path.isfile(_live_db_path(workflow_path)):
        return None
    control = workflow_control(workflow_path, write=True)
    try:
        return control.retire_experiment(
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
    with _lock(workflow_path):
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


def _registration_path(workflow_path, experiment_id):
    _safe_segment(experiment_id, "experiment_id")
    return benchmarks_root(workflow_path) / ".experiments" / f"{experiment_id}.json"


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
    with _lock(workflow_path):
        _atomic_json(_registration_path(workflow_path, experiment_id), record)
    register_experiment_selection(workflow_path, record)
    return record


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
    # differ: a setup field added later is inherited by duplicates without this
    # function having to learn about it.
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
    with _lock(workflow_path):
        _atomic_json(_registration_path(workflow_path, experiment_id), record)
    register_experiment_selection(workflow_path, record)
    return record


def update_experiment_description(workflow_path, experiment_id, description):
    """Edit the description while the registration is still the only record.

    Refused once a runner has bound the registration: from that point the
    description the run declared lives in its evidence store, and a
    registration edited afterwards would disagree with a store nothing here
    can write.
    """
    if not isinstance(description, str):
        raise ValueError("description must be text")
    with _lock(workflow_path):
        record = load_experiment(workflow_path, experiment_id)
        if record.get("store") is not None:
            raise BenchmarkSetupConflict(
                "This experiment has been handed to a runner; its description "
                "is now part of the recorded run."
            )
        record["description"] = description.strip()
        _atomic_json(_registration_path(workflow_path, experiment_id), record)
        return record


def load_experiment(workflow_path, experiment_id):
    path = _registration_path(workflow_path, experiment_id)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        if (path.parent / ".deleted" / path.name).is_file():
            raise ExperimentDeleted("This experiment was deleted. Create a new experiment.")
        raise KeyError(experiment_id)
    if record.get("experiment_id") != experiment_id:
        raise ValueError("experiment registration identity mismatch")
    return record


def registered_experiments(workflow_path, benchmark_id):
    root = benchmarks_root(workflow_path) / ".experiments"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            rows.append(load_experiment(workflow_path, path.stem))
        except (KeyError, ExperimentDeleted):
            continue  # A deletion can complete between enumeration and read.
    return [row for row in rows if row["benchmark_id"] == benchmark_id]


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


def bind_experiment(workflow_path, experiment_id, db_path, store_id):
    with _lock(workflow_path):
        record = load_experiment(workflow_path, experiment_id)
        target = {"db_path": os.path.abspath(db_path), "store_id": store_id}
        if record.get("store") not in (None, target):
            raise BenchmarkSetupConflict(
                "experiment is already bound to another evidence store"
            )
        record["store"] = target
        _atomic_json(_registration_path(workflow_path, experiment_id), record)


def delete_empty_experiment(workflow_path, experiment_id):
    """Remove an unused registration, serialized against a runner's store binding.

    A tombstone prevents delayed runners from treating a deleted ID as a new,
    unregistered experiment. No evidence database or benchmark version is touched.

    The registration also LEAVES THE CONTEST it joined at creation (`fix-jfy5`).
    Withdrawal comes FIRST because it is the only refusable step, so a refused
    deletion — the current winner of a group with other members, or a
    registration with evidence — leaves both the file and the contest exactly
    as they were. Nothing here elects anybody: the winner can be the experiment
    being deleted only when it is the group's sole member (`fix-65ik`), and
    then there is nobody to succeed it — the group is left empty and
    winner-less, and its next experiment is elected on registration.

    Two writes that are not one transaction, made safe by which one can fail
    and by the lock around both:

    - Withdraw, then tombstone. A crash between them leaves a live registration
      that is not a contest member; a rename that FAILS is compensated here, by
      re-registering the reference we just withdrew.
    """
    with _lock(workflow_path):
        return _delete_locked(workflow_path, experiment_id)


def _delete_locked(workflow_path, experiment_id):
    """The body of `delete_empty_experiment`, with the setup lock ALREADY held.

    Split out so the ordering it depends on can be tested against the real
    thing: a caller that holds the lock (a test reproducing a bootstrap racing
    a deletion) can run the actual deletion rather than a paraphrase of it.
    Everything about the policy lives here; the public function is the lock.
    """
    record = load_experiment(workflow_path, experiment_id)
    if record.get("store") is not None:
        raise BenchmarkSetupConflict(
            "This experiment has been handed to a runner and cannot be deleted."
        )
    retired = retire_experiment_selection(workflow_path, experiment_id) or {}
    was_winner = bool(retired.get("winner_retired"))
    path = _registration_path(workflow_path, experiment_id)
    deleted = path.parent / ".deleted" / path.name
    try:
        deleted.parent.mkdir(exist_ok=True)
        os.replace(path, deleted)
    except OSError:
        # The registration is still there, so the withdrawal above has to go
        # back: membership is what a later promotion is checked against, and an
        # experiment a user can still see must still be selectable.
        # Re-registering restores exactly what was removed — the member row
        # with no evidence source — and `allow_initial_winner=False` keeps it
        # from taking a pointer it did not hold a moment ago. A sole winner DID
        # hold the pointer a moment ago, and the withdrawal cleared it, so it
        # is re-registered with the election allowed: the group is empty but
        # for it, the election picks the oldest member, and the user gets back
        # the winner they had (with `initial`, `retire`, `initial` in the
        # append-only history, which is what happened). The LOCKED form,
        # because this lock is not reentrant and we are already inside it.
        restored = _register_selection_locked(
            workflow_path, experiment_id, allow_initial_winner=was_winner
        )
        if restored is None:
            # The repair failed, so the user keeps a registration that is no
            # longer in the contest. Said out loud, because the alternative is
            # an experiment that looks ordinary and silently cannot be
            # promoted.
            logger.error(
                f"observability: experiment {experiment_id!r} could not be "
                "tombstoned AND could not be put back into the selection "
                f"contest of {workflow_name_for(workflow_path)!r}. Its "
                "registration still exists but is not a member of the contest."
            )
        raise
    return record
