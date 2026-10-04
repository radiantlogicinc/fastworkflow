"""Simple benchmark authoring and experiment identities for harness handoff.

Benchmark versions are immutable catalog files. Experiment registrations can be
created before an execution store exists. A controller binds a registration to
its actual evidence store when it declares the attempts, never at UI creation.

This module also owns the workflow's ONE authoritative selection control
(`fix-9eg.17.1`, `fix-9eg.17.2`). Two things follow from "one":

- An experiment enters its comparison group HERE, at creation, with no
  evidence store yet, which is what makes the first experiment of a group its
  winner at the moment somebody decides to run it rather than at the moment a
  runner happens to open a database. A runner later binds its store to the
  same registration and writes into the same control file, so there is never a
  local winner disagreeing with a shared one for the same experiment.
- Every evidence store that joins is recorded by explicit authorization
  (`authorize_evidence_store`), together with the path it lives at. That map is
  what lets `ensure_selection_bootstrap` resolve EVERY known source before
  electing anybody — never by scanning the filesystem for databases.

The control file and that map live in the workflow's STATE dir, not in the
workflow folder: benchmarks are authored content, these are runtime facts about
this machine.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastworkflow.benchmark.catalog import (
    benchmarks_root,
    list_versions,
    load_version,
    write_version,
    _safe_segment,
)
from fastworkflow.observability import selection
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
# The workflow's one selection control
# ----------------------------------------------------------------------


def workflow_name_for(workflow_path):
    """The name a comparison group is scoped by.

    Registration and execution MUST agree on this string: it is half of the
    derived group id, so a mismatch does not raise anything — it quietly opens
    a second contest, each with its own uncontested winner. One helper, called
    from both sides, is the cheapest way to make that impossible.
    """
    return os.path.basename(os.path.abspath(str(workflow_path)).rstrip("/\\"))


def workflow_control_db_path(workflow_path):
    """The ONE control file for this workflow's winners and best runs.

    Under the workflow's state dir, beside its observability DB but never
    inside it: evidence is what happened, a selection is a judgement about it,
    and sealed evidence has to stay byte-identical while judgements keep being
    made. Distinct filename from the per-store sidecar, so the two can never be
    confused for one another.
    """
    from fastworkflow import state_paths

    return selection.shared_control_db_path_for(
        state_paths.workflow_state_dir(str(workflow_path))
    )


def _sources_path(workflow_path):
    """Beside the control file, in the workflow's STATE dir.

    Not under the workflow folder: a benchmark manifest is authored content a
    user keeps in their project, but "which evidence databases are in this
    contest" is runtime state of this machine. Putting it in `benchmarks/`
    would make a runner that merely starts write into the user's source tree —
    and into the repo, for the workflow folders the test suite runs against.
    """
    from fastworkflow import state_paths

    return (
        Path(state_paths.workflow_state_dir(str(workflow_path)))
        / "selection.sources.json"
    )


@contextmanager
def _sources_lock(workflow_path):
    """Serialize source-map updates across HTTP threads and harness processes."""
    import fcntl

    path = _sources_path(workflow_path).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def known_evidence_sources(workflow_path):
    """``{source_id: db_path}`` for every store explicitly authorized here.

    The only place a path comes from. `ensure_selection_bootstrap` resolves
    exactly these and nothing else: a control that went looking for databases
    near the ones it knew would make "who is in this contest" depend on the
    filesystem layout rather than on somebody having said so.
    """
    try:
        value = json.loads(_sources_path(workflow_path).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if isinstance(v, str)}


def _source_resolver(workflow_path):
    """Open each known store READ-ONLY, once, and remember the failures.

    Read-only because bootstrap is inspection: resolving a source to decide who
    the earliest experiment is must not create, migrate or write an evidence
    database. A store that cannot be opened resolves to None, which the control
    treats as "unknown", not "empty".
    """
    from fastworkflow.observability.store import ReadOnlyObservabilityStore

    paths = known_evidence_sources(workflow_path)
    cache = {}

    def resolve(source_id):
        if source_id in cache:
            return cache[source_id]
        db_path = paths.get(source_id)
        store = None
        if db_path and os.path.isfile(db_path):
            try:
                store = ReadOnlyObservabilityStore(db_path)
            except Exception as exc:  # an unopenable store is unknown, not empty
                logger.warning(
                    f"observability: evidence source {source_id!r} at {db_path!r} "
                    f"could not be opened for selection bootstrap: {exc}"
                )
                store = None
        if store is not None and store.store_identity() != source_id:
            # A DIFFERENT database at a remembered path: restored from the wrong
            # backup, or a fresh file created where the old one stood. Handing
            # it back would make the control raise its identity-mismatch error
            # on every read; returning None makes this source what it actually
            # is -- a known store of unknown content, which the existing
            # unresolved-source guard already refuses to elect around. The
            # mapping is left alone, so the real file returning to this path
            # resolves and recovers by itself.
            logger.warning(
                f"observability: evidence source {source_id!r} is remembered at "
                f"{db_path!r}, but the database there identifies as "
                f"{store.store_identity()!r}; it is treated as unreadable "
                "rather than answering for the source it is not"
            )
            store = None
        cache[source_id] = store
        return store

    return resolve


def open_workflow_control(workflow_path, *, create=True, sources=None):
    """Open this workflow's shared selection control.

    ``create=False`` is the read-only inspection path: showing a winner or a
    best run must not bring a control database into existence for a workflow
    where nobody has decided anything, because an empty file created by a GET
    is indistinguishable afterwards from one whose decisions were lost.
    """
    return selection.open_shared_control(
        workflow_control_db_path(workflow_path),
        sources=_source_resolver(workflow_path) if sources is None else sources,
        create=create,
    )


def authorize_evidence_store(workflow_path, store, db_path=None, *, label=None):
    """Explicitly admit one evidence store to this workflow's contest.

    Returns the source id, which is the store's own identity: derived, so two
    processes admitting the same store agree without coordinating, and unique,
    so the control's UNIQUE identity check and this id can never disagree.

    The path is recorded alongside it because authorization is the only moment
    anybody knows both facts at once. Without it a later session holds a
    control file full of source ids it cannot resolve, and — by the bootstrap
    contract — would correctly refuse to elect anyone, forever.
    """
    identity = store.store_identity()
    if not identity:
        raise BenchmarkSetupConflict(
            f"evidence store {getattr(store, 'db_path', '?')!r} has no store "
            "identity; a shared control will not admit a store it cannot name"
        )
    control = open_workflow_control(workflow_path)
    control.authorize_source(identity, store, label=label)
    path = os.path.abspath(db_path or store.db_path)
    with _sources_lock(workflow_path):
        known = known_evidence_sources(workflow_path)
        if known.get(identity) != path:
            known[identity] = path
            _atomic_json(_sources_path(workflow_path), known)
    return identity


def _reference_for(workflow_path, record, source_id=None):
    return selection.ExperimentReference(
        experiment_id=record["experiment_id"],
        workflow_name=workflow_name_for(workflow_path),
        benchmark_id=record.get("benchmark_id"),
        benchmark_version=record.get("benchmark_version"),
        benchmark_digest_sha256=record.get("benchmark_digest_sha256"),
        created_at=record.get("created_at"),
        source_id=source_id,
    )


def _declare_unreadable(workflow_path, control, store_id, path):
    """Record that this contest contains a store nobody can read today.

    Returning None alone is not enough, because the next caller is a different
    caller. Seeding knows the gap and registers without electing; the runner
    that starts an hour later calls straight into the control, sees a contest
    whose visible history is complete, and elects its brand-new experiment over
    the runs in the file that went missing. That is the same wrong winner the
    bootstrap contract exists to prevent, reached by the other door.

    Declaring the store makes the gap a fact the CONTROL holds, so every path
    into it -- registration, duplication, a starting runner -- meets the
    unresolved-source guard that is already there. The path is remembered too:
    recovery is then automatic, because the resolver can open it the moment it
    comes back.
    """
    if not store_id:
        return
    try:
        control.declare_bound_source(store_id)
    except selection.SelectionControlError as exc:
        logger.warning(
            f"observability: evidence store {store_id!r} named by a registration "
            f"could not be declared to the selection contest: {exc}"
        )
        return
    if not path:
        return
    with _sources_lock(workflow_path):
        known = known_evidence_sources(workflow_path)
        if known.get(store_id) != path:
            known[store_id] = path
            _atomic_json(_sources_path(workflow_path), known)


def _authorize_recorded_store(workflow_path, control, store_id, db_path):
    """Admit the store a REGISTRATION already names, or report it unavailable.

    `bind_experiment` writes `{"db_path", "store_id"}` into the registration
    when a runner takes it, which makes that pair an explicit authorization the
    user already performed — not a database this function went looking for. It
    is the only path considered, and it is checked: a file whose identity is
    not the recorded `store_id` is a different store at the same path and is
    refused rather than admitted under the old id.

    Returns the source id, or None when the store cannot be opened and
    identified right now — which makes the bootstrap incomplete rather than
    making the registration's history disappear.
    """
    from fastworkflow.observability.store import ReadOnlyObservabilityStore

    if not store_id:
        return None
    path = os.path.abspath(db_path) if db_path else None
    if control.source_for_identity(store_id) is not None:
        # Already in the contest. Whether its file is readable TODAY is the
        # resolver's question, and the control's own unresolved-source guard is
        # what answers it; re-opening it here would only duplicate that.
        return store_id
    if not path or not os.path.isfile(path):
        _declare_unreadable(workflow_path, control, store_id, path)
        return None
    try:
        store = ReadOnlyObservabilityStore(path)
        identity = store.store_identity()
    except Exception as exc:
        logger.warning(
            f"observability: evidence store {path!r} named by a registration "
            f"could not be opened for selection bootstrap: {exc}"
        )
        _declare_unreadable(workflow_path, control, store_id, path)
        return None
    if identity != store_id:
        logger.warning(
            f"observability: the evidence store at {path!r} identifies as "
            f"{identity!r}, but a registration names it {store_id!r}; it is not "
            "admitted to the selection contest under either id"
        )
        # The impostor's identity is NOT authorized -- it is a different store
        # and answers for nothing here. What IS recorded is that the store this
        # registration named is unreadable, because returning None alone only
        # stops THIS caller: the runner that starts an hour later opens the
        # control directly, sees a contest whose visible history is complete and
        # elects its brand-new experiment over the older runs in the database
        # that was replaced. The path travels with the declaration so the real
        # file coming back recovers automatically -- the resolver identity-checks
        # it, so the impostor cannot answer for the id in the meantime.
        _declare_unreadable(workflow_path, control, store_id, path)
        return None
    control.authorize_source(identity, store)
    with _sources_lock(workflow_path):
        known = known_evidence_sources(workflow_path)
        if known.get(identity) != path:
            known[identity] = path
            _atomic_json(_sources_path(workflow_path), known)
    return identity


def registration_records(workflow_path):
    """Every experiment registration of this workflow, OLDEST FIRST.

    Across all benchmarks, unlike `registered_experiments`, because the
    contest's question is "what did this workflow register, and when" — and the
    answer decides who the earliest experiment is. Sorted by `created_at` with
    the id breaking ties, so the order is the same in every process.
    """
    root = benchmarks_root(workflow_path) / ".experiments"
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        try:
            rows.append(load_experiment(workflow_path, path.stem))
        except (KeyError, ExperimentDeleted, ValueError):
            continue  # A deletion can complete between enumeration and read.
    rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row["experiment_id"])))
    return rows


def _admit_default_store(workflow_path, control):
    """Admit the store this workflow records into BY DEFAULT, if it exists.

    The gap this closes is an ordinary upgrade, not an exotic one. A workflow
    that has been running since before selection existed has its experiments in
    the one database `state_paths.observability_db` names for it, and nothing
    else: no `.experiments/*.json` registration named it, because registrations
    are the new thing. `_seed_registrations` therefore admits nothing, the
    contest starts empty, and the first experiment somebody registers after the
    upgrade becomes the winner over everything already recorded there.

    The path is the workflow's own canonical one, asked for by name. Nothing is
    searched for on disk, nothing is created -- a workflow that has never
    recorded anything has no file here and that is simply the answer -- and the
    store is opened READ-ONLY, because deciding who the earliest experiment is
    must not migrate or write evidence.

    Returns None when there is nothing to admit or it is already in the
    contest, and a description of the gap when the file is there but cannot be
    read: unreadable is not empty, and electing around it would be the same
    wrong winner by a different door.
    """
    from fastworkflow import state_paths
    from fastworkflow.observability.store import ReadOnlyObservabilityStore

    try:
        path = state_paths.observability_db(str(workflow_path))
    except Exception as exc:  # an unresolvable state root is not this call's job
        logger.warning(
            f"observability: the default evidence path for "
            f"{workflow_path!r} could not be resolved: {exc}"
        )
        return None
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    if path in set(known_evidence_sources(workflow_path).values()):
        # Already admitted -- by a runner, by a registration, or by a previous
        # bootstrap. Whether it is readable TODAY is the resolver's question.
        return None
    try:
        store = ReadOnlyObservabilityStore(path)
        identity = store.store_identity()
    except Exception as exc:
        logger.warning(
            f"observability: the default evidence store at {path!r} could not "
            f"be opened for selection bootstrap: {exc}"
        )
        return {"store_id": None, "db_path": path, "error": str(exc)}
    if not identity:
        return {"store_id": None, "db_path": path,
                "error": "the store reports no identity"}
    if control.source_for_identity(identity) is not None:
        return None
    try:
        control.authorize_source(identity, store, label="default evidence store")
    except selection.SelectionControlError as exc:
        logger.warning(
            f"observability: the default evidence store at {path!r} could not "
            f"be admitted to the selection contest: {exc}"
        )
        return {"store_id": identity, "db_path": path, "error": str(exc)}
    with _sources_lock(workflow_path):
        known = known_evidence_sources(workflow_path)
        if known.get(identity) != path:
            known[identity] = path
            _atomic_json(_sources_path(workflow_path), known)
    return None


def _seed_registrations(workflow_path, control, *, already_missing=()):
    """Put the registrations that predate this control INTO it, in order.

    The failure this exists to prevent: selection arrives in a workflow that
    has been registering and running experiments for weeks. Those experiments
    are recorded in `.experiments/*.json` and nowhere the control can see, so
    the next experiment somebody creates is the first one the control has ever
    heard of — and becomes the winner, outranking months of older work that was
    never in the contest to lose it.

    Seeding is safe to repeat: registering a known experiment adds no member
    row, and adds a decision only when its group still has no winner and the
    view is now whole -- the `initial` election a previous, incomplete
    bootstrap withheld, which goes to the group's OLDEST member rather than to
    whichever registration happens to be re-seeded first (`fix-kkod`). So this
    runs on every bootstrap rather than once behind a flag nobody can verify.

    Two passes, and the order is the point. The first admits every store the
    registrations name, so completeness is known BEFORE anybody is elected. The
    second registers oldest first, which is what makes the earliest
    registration — not whoever happens to be registering today — the one that
    takes an empty group's pointer. If the first pass could not admit some
    store, the second elects nobody at all and the next bootstrap retries.
    """
    records = registration_records(workflow_path)
    # A gap found before this pass counts against completeness here, or the
    # oldest registration would take an empty group's pointer while a store
    # nobody could read was still unaccounted for.
    missing = list(already_missing)
    sources = {}
    for record in records:
        bound = record.get("store") or {}
        if not bound.get("store_id"):
            continue
        source_id = _authorize_recorded_store(
            workflow_path, control, bound.get("store_id"), bound.get("db_path")
        )
        if source_id is None:
            missing.append(
                {
                    "experiment_id": record["experiment_id"],
                    "store_id": bound.get("store_id"),
                    "db_path": bound.get("db_path"),
                }
            )
        else:
            sources[record["experiment_id"]] = source_id

    seeded = []
    # Under the SETUP lock, re-reading each registration inside it. `records`
    # was enumerated before the lock, and `delete_empty_experiment` withdraws a
    # registration from the contest and tombstones it while holding this same
    # lock. Without the re-read, a bootstrap that read its list a moment
    # earlier would register the deleted id again — after its withdrawal — and
    # put a tombstone back in the contest, which is `fix-jfy5` by the back
    # door. A registration created since the enumeration is simply seeded on
    # the next bootstrap, the way it always was.
    with _lock(workflow_path):
        for stale in records:
            experiment_id = stale["experiment_id"]
            try:
                record = load_experiment(workflow_path, experiment_id)
            except (KeyError, ExperimentDeleted, ValueError):
                continue
            try:
                control.register_experiment_reference(
                    _reference_for(workflow_path, record, sources.get(experiment_id)),
                    allow_initial_winner=not missing,
                )
                seeded.append(experiment_id)
            except Exception as exc:
                logger.warning(
                    f"observability: registration {experiment_id!r} could "
                    f"not be seeded into the selection contest: {exc}"
                )
                missing.append({"experiment_id": experiment_id, "error": str(exc)})
    return {"seeded": seeded, "registrations_unavailable": missing}


def ensure_selection_bootstrap(workflow_path, *, create=False):
    """Adopt pre-existing history automatically, through what is KNOWN.

    Two kinds of history, both seeded here before anybody is elected:

    1. Registrations in `.experiments/`, which may predate this control file
       entirely and may never have run.
    2. Experiments recorded in the evidence stores those registrations name,
       plus any store a runner has authorized since.

    Called at registration and when a runner starts, so a workflow that already
    had experiments when selection was switched on elects its earliest one by
    itself. There is no adoption ceremony for a user to perform.

    What it will NOT do is elect from a partial view. If a registration names a
    store that cannot be opened, or an authorized source cannot be resolved,
    whether older runs of this lineage exist there is unknown, and the earliest
    experiment *visible* is not the earliest experiment. `complete` is then
    False, the report names what was missing, and the caller registers without
    initializing a winner; the next bootstrap retries and elects once the view
    is whole. That is what stops a runner holding only its own store from
    promoting its brand-new experiment over an older one in a database nobody
    opened.

    `create=False` is the inspection path: it will not bring a control database
    into existence for a workflow where nothing has been decided.
    """
    path = workflow_control_db_path(workflow_path)
    exists = selection.control_mode_of(path) is not None
    if not exists and not create:
        return {
            "status": "absent",
            "complete": False,
            "sources_skipped": [],
            "seeded": [],
            "registrations_unavailable": [],
        }
    try:
        control = open_workflow_control(workflow_path, create=create)
        # Before the registrations, because it is the history NO registration
        # names: what this workflow recorded by default before any of this
        # existed. Admitting it first means completeness is known before
        # anybody is elected, the same way the registration pass works.
        default_gap = _admit_default_store(workflow_path, control)
        seed = _seed_registrations(
            workflow_path, control,
            already_missing=() if default_gap is None else (default_gap,),
        )
    except (selection.SelectionControlError, OSError) as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "complete": False,
            "sources_skipped": [],
            "seeded": [],
            "registrations_unavailable": [],
        }
    known = [str(row["source_id"]) for row in control.list_sources()]
    if not known:
        # Registrations with no evidence anywhere yet: nothing to adopt, and
        # the references themselves already ordered the contest.
        report = {"sources_read": [], "sources_skipped": [], "complete": True,
                  "experiments_seen": 0, "experiments_adopted": 0,
                  "groups_without_winner": []}
    else:
        try:
            # A registration naming a store that could not be opened is a gap
            # `adopt_existing_experiments` cannot see for itself: that store was
            # never authorized, so it is not in anybody's skipped list, and the
            # oldest experiment still READABLE would be elected over the older
            # runs sitting in the file that went missing. Seeding knows; adoption
            # is told.
            report = control.adopt_existing_experiments(
                allow_initial_winner=not seed["registrations_unavailable"]
            )
        except selection.SelectionControlError as exc:
            # Not one authorized source could be opened. Every one of them is
            # skipped, and saying so is the difference between "retry later"
            # and "there is nothing here".
            report = {"error": str(exc), "complete": False, "sources_read": [],
                      "sources_skipped": sorted(known), "experiments_seen": 0,
                      "experiments_adopted": 0, "groups_without_winner": []}
    report.update(seed)
    report["complete"] = bool(
        report.get("complete") and not seed["registrations_unavailable"]
    )
    report["status"] = "adopted" if report["complete"] else "incomplete"
    if not report["complete"]:
        unreadable = [
            str(item.get("db_path") or item.get("experiment_id"))
            for item in seed["registrations_unavailable"]
        ] + [str(source) for source in report.get("sources_skipped", [])]
        logger.warning(
            "observability: selection bootstrap for "
            f"{workflow_name_for(workflow_path)!r} is incomplete; "
            f"{', '.join(unreadable) or 'an evidence source'} could not be "
            "read. No initial winner is elected until they can be."
        )
    return report


def register_experiment_selection(workflow_path, record, *, allow_initial_winner=True):
    """Enter a registration in the workflow contest; the first one wins.

    Runs at creation, with no evidence store: an experiment is a decision to
    run something long before a runner opens a database for it, and the first
    experiment of a group should be its winner from that moment. Best effort by
    design — a control file that cannot be written must not stop somebody
    creating an experiment — so every problem is reported and logged rather
    than raised.

    `allow_initial_winner=False` is what a caller passes when the bootstrap
    that ran just before it came back incomplete: register, but do not hand
    this brand-new experiment a title that some older, currently unreadable one
    may already hold.

    `record` is the registration the CALLER holds, which is why this takes the
    setup lock and re-reads by id instead of trusting it. `create_experiment`
    writes the file, releases the lock, runs a bootstrap and only then arrives
    here; a deletion in that gap has already withdrawn the experiment, and
    writing the caller's cached copy would put the tombstoned id straight back
    into the contest — and make it promotable again (`fix-jfy5`). Every setup
    path into the control goes through a lock-and-re-read for that reason:
    this one, and the bootstrap's `_seed_registrations`.
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
        control = open_workflow_control(workflow_path)
        return control.register_experiment_reference(
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
    no control file at all has nothing to retire and nothing to fail: `None`.

    Raises `ExperimentSelected` (409) when the experiment is the group's
    current winner and other experiments are still in the group — see that
    class for why deletion does not get to move a winner pointer. The winner
    that is the group's ONLY member is withdrawn along with the pointer; the
    result's `winner_retired` says so, which is what the deletion's
    compensation needs to know.
    """
    path = workflow_control_db_path(workflow_path)
    if selection.control_mode_of(path) is None:
        return None
    control = open_workflow_control(workflow_path, create=False)
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


def bind_runner_evidence(workflow_path, store, db_path=None):
    """A starting runner joins the workflow contest. Returns the control path.

    Three steps in one call because they are one event: the store is admitted
    (explicitly, by identity), the workflow's pre-existing history is adopted
    now that one more source can be resolved, and the caller is told where the
    single authoritative control lives so its experiment registers THERE.

    Returns None if any of that fails, and the caller must then record no
    winner at all rather than falling back to a private sidecar: two control
    files disagreeing about the same registered experiment is worse than one
    experiment whose winner was not recorded, because only the second is
    visibly missing.

    An INCOMPLETE bootstrap returns None for the same reason, and this is the
    third door into the wrong first winner (`fix-9eg.17.1`). The registration
    paths above already ask `ensure_selection_bootstrap` whether the view was
    whole and pass `allow_initial_winner=bootstrap["complete"]`; this one used
    to run the same bootstrap, discard its answer and hand back a path anyway,
    and the runner reads a path as "elect". Where the gap has a known identity
    the durable declaration catches it again inside the control, but a gap with
    no identity to declare — an unreadable DEFAULT evidence store, which is
    just this workflow's own history mid-outage — is invisible from in there,
    and the runner elected its brand-new experiment over it.

    The store is admitted BEFORE that check and stays admitted, so this costs
    nothing but the title: the runner's evidence is in the contest, the next
    bootstrap that finds the view whole adopts it along with the history that
    was missing, and the oldest experiment wins then. Refusing the path also
    keeps `create_experiment` from writing a private sidecar, because the
    runner passes `initialize_winner=False` when there is no shared control.
    """
    try:
        authorize_evidence_store(workflow_path, store, db_path)
        report = ensure_selection_bootstrap(workflow_path, create=True)
        if not report.get("complete"):
            logger.warning(
                "observability: evidence store "
                f"{os.path.abspath(db_path or store.db_path)!r} joined the "
                f"selection control of {workflow_name_for(workflow_path)!r}, "
                "but no winner is recorded for it: this workflow's history "
                "cannot be read in full right now. The next bootstrap that "
                "can read it elects the earliest experiment."
            )
            return None
        return workflow_control_db_path(workflow_path)
    except Exception as exc:
        logger.warning(
            "observability: evidence store "
            f"{os.path.abspath(db_path or store.db_path)!r} could not join the "
            f"selection control of {workflow_name_for(workflow_path)!r}: {exc}"
        )
        return None


def workflow_winner(workflow_path, experiment_id):
    """The current winner of the group this experiment is in, or None.

    Read-only: it never creates the control file, so inspecting a workflow
    where nobody has decided anything leaves it exactly as it was.
    """
    try:
        control = open_workflow_control(workflow_path, create=False)
    except selection.SelectionControlUnavailable:
        return None
    return control.winner_for_experiment(experiment_id)


def benchmark_winner(workflow_path, benchmark_id):
    """The current winner of this benchmark's contest, or None.

    Read-only, like `workflow_winner`. A contest is one group per benchmark
    lineage, across versions, so the page that lists every experiment of that
    lineage can mark the one the pointer names without opening each experiment.
    Inspecting a benchmark that has never entered a contest does not create
    the control file.
    """
    benchmark_id = str(benchmark_id or "").strip()
    if not benchmark_id:
        return None
    try:
        control = open_workflow_control(workflow_path, create=False)
    except selection.SelectionControlUnavailable:
        return None
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
    try:
        control = open_workflow_control(workflow_path, create=False)
    except selection.SelectionControlUnavailable:
        return False
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
    # Adopt any pre-existing history FIRST. Registering this brand-new
    # experiment into a group whose older runs are still unadopted would file a
    # judgement about history nobody made.
    bootstrap = ensure_selection_bootstrap(workflow_path, create=True)
    register_experiment_selection(
        workflow_path, record, allow_initial_winner=bootstrap["complete"]
    )
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
    bootstrap = ensure_selection_bootstrap(workflow_path, create=True)
    register_experiment_selection(
        workflow_path, record, allow_initial_winner=bootstrap["complete"]
    )
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
      that is not a contest member, which the next `ensure_selection_bootstrap`
      re-seeds (registration is idempotent, and a sole winner withdrawn this
      way is re-elected by it, being the oldest member of its group); a rename
      that FAILS is compensated here, by re-registering the reference we just
      withdrew.
    - `_seed_registrations` takes this same lock and re-reads the registrations
      inside it, so a bootstrap cannot be holding a records list from before
      the tombstone and re-register a deleted id after the withdrawal.
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
        if restored is not None and was_winner and restored.get("winner") is None:
            # Re-elected nobody: the view stopped being whole in between (an
            # evidence store became unreadable), so the election is withheld
            # like any other. The member is back and the next complete
            # bootstrap elects it; said out loud so nobody reads it as lost.
            logger.error(
                f"observability: experiment {experiment_id!r} could not be "
                "tombstoned, and was put back into the selection contest of "
                f"{workflow_name_for(workflow_path)!r} WITHOUT its winner "
                "title: the workflow's history cannot be read in full right "
                "now. The next bootstrap that can read it re-elects it."
            )
        if restored is None:
            # The repair failed, so the user keeps a registration that is no
            # longer in the contest. Said out loud, because the alternative is
            # an experiment that looks ordinary and silently cannot be
            # promoted; the next `ensure_selection_bootstrap` re-seeds it.
            logger.error(
                f"observability: experiment {experiment_id!r} could not be "
                "tombstoned AND could not be put back into the selection "
                f"contest of {workflow_name_for(workflow_path)!r}. Its "
                "registration still exists but is not a member until the next "
                "bootstrap re-seeds it."
            )
        raise
    return record
