"""Evidence-grade capture: a run mode, not a global change.

Architecture §12.0 delta 6 and requirements §12.4. The 3.2.0 observability store
is best-effort by design — spans ride a droppable queue, turn records ride a small
one, and retention prunes at thirty days or a gigabyte. That posture is right for
a developer debugging a workflow and wrong for a measured run, where a silently
dropped turn record means the reported score is computed over a denominator nobody
can reconstruct.

This module does not change that posture. It wraps a run in four checks and leaves
normal operation exactly as it was:

1. **Zero-drop assertion.** Writer-health counters are snapshotted before and
   after. A dropped turn record makes the run invalid, full stop. A dropped span
   does not — spans are best-effort — but it is reported with the turn keys it
   affected, so a reader knows which turns have incomplete detail instead of
   assuming all of them are whole.

2. **Prune suppression.** §12.4: "pruning shall not run mid-evaluation". The
   thirty-day horizon sounds like it could not possibly bite inside one run, but
   the size cap evicts oldest-first regardless of age, so a long or high-volume run
   can delete its own early spans while still recording its later ones.

3. **Archival.** The run's DB is copied into the bundle before retention can ever
   reach it, with a digest, via ``VACUUM INTO`` rather than a file copy — see
   `ObservabilityStore.archive_to` for why copying a WAL-mode database silently
   loses the end of the run.

4. **Provenance.** The `FW_OBS_*` values in effect, the capture profile and its
   policy version, the span-contract version, and the DB schema version. Trace
   evidence without these is uninterpretable rather than merely undocumented: the
   same workflow under the `evidence` and `debug` profiles produces records that
   differ in what they contain, not in what happened.

**What this module cannot promise.** It asserts that the *store* did not drop what
it was given. It cannot assert that everything which happened was emitted — an
emitter that was never called leaves no trace and no drop counter, so a run can be
zero-drop and still incomplete. `fix-ajv.9`'s unredacted write paths and the
un-migrated dispatch paths of §12.1.1 are both examples. Zero-drop is a floor.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastworkflow import state_paths, tracing
from fastworkflow.observability import capture_policy, store as observability_store
from fastworkflow.observability.provenance import ObservabilityProvenance
from fastworkflow.utils.logging import logger

# How often to re-read the persisted health row while waiting for it to advance.
_HEALTH_POLL_INTERVAL_S = 0.1


class EvidenceRunInvalid(RuntimeError):
    """A run that was required to be evidence-grade was not.

    Carries every problem found rather than the first, so an operator sees the
    whole picture in one pass — same reasoning as
    `runtime_manifest.ManifestConformanceError`.
    """

    def __init__(self, run_id: str, problems: tuple[str, ...]) -> None:
        self.run_id = run_id
        self.problems = problems
        detail = "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, start=1))
        super().__init__(f"evidence run {run_id} is not valid evidence:\n{detail}")


def capture_observability_provenance(
    *,
    dspy_history_enabled: Optional[bool] = None,
    evidence_grade: Optional[bool] = None,
) -> ObservabilityProvenance:
    """Snapshot the capture regime in effect (§12.4).

    Lives here rather than in `provenance` because it reads `observability_store`
    and `tracing`, neither of which is a §22 leaf.
    """
    config = observability_store.observability_config()
    return ObservabilityProvenance(
        # Recording has no switch, so the regime is always "on". The field
        # stays because every provenance record already written carries it;
        # whether a writer actually opened the store is checked separately,
        # from writer health.
        enabled=True,
        capture_profile=config[observability_store.CAPTURE_PROFILE_VAR],
        capture_policy_version=capture_policy.CAPTURE_POLICY_VERSION,
        span_contract_version=tracing.SPAN_CONTRACT_VERSION,
        span_contract_versions=tracing.span_contract_versions(),
        db_schema_version=observability_store.SCHEMA_VERSION,
        config=config,
        dspy_history_enabled=dspy_history_enabled,
        evidence_grade=evidence_grade,
    )


@dataclass
class EvidenceRun:
    """One measured run's evidence record. Built by `evidence_run()`."""

    run_id: str
    db_path: str
    provenance: ObservabilityProvenance
    started_at: datetime
    health_before: Optional[dict[str, Any]] = None
    health_after: Optional[dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    delta: Optional[observability_store.WriterHealthDelta] = None
    archive: Optional[dict[str, Any]] = None
    extra_problems: list[str] = field(default_factory=list)
    # Which process's counters this verdict rests on. True = this process holds
    # the sink and the numbers are its live counters; False = the writer is
    # somewhere else and the numbers come from the persisted diagnostics row,
    # which this run had to wait for rather than flush. Recorded because a
    # reader cannot otherwise tell a measured interval from an assumed one, and
    # the cross-process case is where the old code silently reported zeros
    # (fix-ajv.13).
    in_process: bool = True

    @property
    def valid(self) -> bool:
        """Whether this run may be reported as evidence.

        An unfinished run is not valid: `delta is None` means the interval was
        never closed, and treating that as a pass would let a crashed run be
        reported as clean.
        """
        return (
            self.delta is not None
            and self.delta.evidence_valid
            and not self.extra_problems
        )

    def problems(self) -> tuple[str, ...]:
        if self.delta is None:
            return ("the evidence run never completed, so nothing was verified",)
        return tuple(self.delta.problems()) + tuple(self.extra_problems)

    def as_record(self) -> dict[str, Any]:
        """The serializable summary a harness writes into its bundle."""
        return {
            "run_id": self.run_id,
            "db_path": self.db_path,
            "in_process": self.in_process,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "valid": self.valid,
            "problems": list(self.problems()),
            "observability": self.provenance.model_dump(mode="json"),
            "writer_health_before": self.health_before,
            "writer_health_after": self.health_after,
            "writer_health_delta": (
                self.delta.model_dump(mode="json") if self.delta else None
            ),
            "archive": self.archive,
        }


def _health_snapshot(
    db_path: str, sink: Optional[observability_store.SQLiteTraceSink]
) -> Optional[dict[str, Any]]:
    """The live counters if a sink is in this process, else the persisted row.

    Prefers the sink because the `diagnostics` row lags by up to a heartbeat: a
    run that read only the row could snapshot a drop that had already happened as
    though it had not. The row is the cross-process fallback, and `None` from both
    is reported as `incomparable` rather than as zero.

    The sink handed in here is now only ever one this process ALREADY had (see
    `existing_observability_sink`). It is never minted to satisfy this call, so
    "no sink" reliably means "this process is not the writer" instead of meaning
    "nobody has asked yet". fix-ajv.13.
    """
    if sink is not None and not sink._closed:
        return sink.health_snapshot()
    return _persisted_health(db_path)


def _persisted_health(db_path: str) -> Optional[dict[str, Any]]:
    """The `diagnostics` writer-health row, or None if it cannot be read."""
    try:
        return observability_store.ObservabilityStore(db_path).writer_health()
    except Exception as exc:
        logger.warning(f"Could not read writer health from {db_path}: {exc!r}")
        return None


def _await_health_refresh(
    db_path: str, since: Optional[str], settle_s: float
) -> Optional[dict[str, Any]]:
    """Poll the persisted health row until it advances past `since`.

    The cross-process half of fix-ajv.13. The writing process persists its
    counters on a heartbeat, so reading the row the instant the body returns can
    snapshot a moment BEFORE the run's last writes — and a drop during those
    writes would land outside the measured interval and be reported as clean.

    Returns the freshest row read. The caller decides what an unrefreshed row
    means; this function does not judge, so that "we waited and it never moved"
    stays distinguishable from "we never looked".
    """
    deadline = time.monotonic() + max(settle_s, 0.0)
    health = _persisted_health(db_path)
    while time.monotonic() < deadline:
        if health is not None and since is not None:
            if str(health.get("updated_at") or "") > since:
                return health
        elif health is not None and since is None:
            return health
        time.sleep(_HEALTH_POLL_INTERVAL_S)
        health = _persisted_health(db_path)
    return health


@contextlib.contextmanager
def evidence_run(
    workflow_folderpath: str,
    *,
    run_id: Optional[str] = None,
    archive_dir: Optional[str] = None,
    dspy_history_enabled: Optional[bool] = None,
    require_evidence_profile: bool = False,
    raise_on_invalid: bool = False,
    health_settle_s: float = 5.0,
):
    """Record a measured run, then verify nothing was silently lost.

    Yields an `EvidenceRun` that is populated on exit — `valid`, `problems()` and
    `archive` are only meaningful after the block ends.

    Pruning is suppressed for the block. On exit the sink is flushed, its counters
    are persisted so the archive carries the same verdict as the in-process delta,
    the health delta is computed, and the DB is archived if `archive_dir` is given.

    `raise_on_invalid` is opt-in rather than the default. A harness usually wants
    to record the verdict alongside its results and decide what to do with a
    partially-valid run itself; raising by default would discard the run's data to
    signal that the run's data is suspect.

    TOPOLOGY. This never constructs an observability sink. If this process is
    the writer it already has one and its live counters are used; otherwise the
    verdict is read from the persisted `diagnostics` row and `health_settle_s`
    bounds how long to wait for the writing process to publish a row covering
    the end of the run. A run that cannot read health from either source, or
    whose cross-process row never advances, is reported with a problem rather
    than as a clean interval — an unread counter is not a measurement of zero.
    See `EvidenceRun.in_process`. fix-ajv.13.

    A writer publishes a zero baseline row when it is CONSTRUCTED, not when it
    first counts something (`SQLiteTraceSink._publish_baseline_health`, fix-485),
    so a gate opened on a store whose server is up but idle reads a real
    `before` row instead of `None`. That is what makes the cross-process shape
    measurable at all: the "the writing process did not publish writer health
    during this run" check below compares stamps, so with no `before` stamp it
    cannot fire — a run with no baseline was never merely pessimistic, it was
    unmeasurable.

    CROSS-PROCESS PRUNING is a separate contract this cannot enforce from here:
    `suppress_pruning()` below is an in-process counter, so it withholds pruning
    in THIS process and not in the server that is actually writing. The only
    switch that crosses a process boundary is the `FW_OBS_SUPPRESS_PRUNE`
    environment variable, and it must be in the writer's environment BEFORE it
    starts, because `SQLiteTraceSink.__init__` prunes opportunistically. See
    fix-ajv.14 and `run_fastapi_mcp`'s startup report of the value in effect.

    `require_evidence_profile` additionally demands
    `FW_OBS_CAPTURE_PROFILE=evidence`. Off by default because the profile governs
    exposure, not completeness — a `debug`-profile run loses no evidence, it just
    keeps more than a tenant deployment should.

    Archiving and verification run even when the body raises, because a run that
    crashed is exactly when its evidence is most worth keeping.
    """
    # `evr-`, not `run-` (`fix-bn1` `[XR10]`). The distillation run id is also
    # spelled `run_id` and also starts with the literal `run-`, differing only in
    # body shape, so a log line reading `run-...` did not say which subsystem it
    # belonged to. That was a legibility defect until an `experiment_evidence_runs`
    # row held one beside a `distillation_runs` join, at which point `run_id`
    # became ambiguous in SQL. Safe to change: no DB row is keyed on it and no id
    # is derived from it.
    run_id = run_id or f"evr-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    db_path = state_paths.observability_db(workflow_folderpath)
    # PEEK, never construct. Constructing here is what made a cross-process run
    # certify itself: the harness got a sink of its own, both snapshots then read
    # THAT sink's counters, which are zero and stay zero because the server is
    # the process doing the dropping — so the delta was all zeros,
    # incomparable=False, evidence_valid=True, for a run that measured nothing.
    # It also started a second writer thread against the server's database and
    # ran an opportunistic prune on the way in. fix-ajv.13.
    sink = observability_store.existing_observability_sink(workflow_folderpath)
    in_process = sink is not None

    provenance = capture_observability_provenance(
        dspy_history_enabled=dspy_history_enabled, evidence_grade=True
    )
    run = EvidenceRun(
        run_id=run_id,
        db_path=db_path,
        provenance=provenance,
        started_at=datetime.now(timezone.utc),
        health_before=_health_snapshot(db_path, sink),
        in_process=in_process,
    )
    if not in_process and run.health_before is None:
        # No sink here and no persisted row there: nothing to compare against,
        # and the run must say so rather than report a clean interval. Zero
        # drops out of an unread counter is not a measurement.
        #
        # Since fix-485 a writer publishes a zero baseline row when it is
        # constructed, so reaching here means no writer has EVER opened this
        # store — the gate was opened before the server was, or against the
        # wrong path. It no longer means "the server is up but has not written
        # yet", which is the case that used to lose a whole run here.
        run.extra_problems.append(
            "no writer health is available: this process holds no observability "
            "sink and the database has no persisted writer-health row, so no "
            "writer has opened this store and no drop could have been detected "
            "by this run"
        )
    if require_evidence_profile and provenance.capture_profile != "evidence":
        run.extra_problems.append(
            f"capture profile is '{provenance.capture_profile}', not 'evidence'; "
            "unclassified fields were captured rather than withheld"
        )
    if dspy_history_enabled is False:
        run.extra_problems.append(
            "DSPy history is off, so this run has no token or cost evidence (§12.4)"
        )

    body_failed = False
    try:
        with observability_store.suppress_pruning():
            yield run
    except BaseException:
        # Tracked so the verification below cannot replace the body's exception
        # with its own. A run that crashed has a cause, and "the evidence is
        # incomplete" is a consequence of it — raising the consequence would hide
        # the cause and send the reader looking in the wrong place.
        body_failed = True
        raise
    finally:
        run.completed_at = datetime.now(timezone.utc)
        if sink is None:
            # Re-peek. A sink that did not exist when the block opened but does
            # now means THIS process became the writer during the run — the
            # ordinary in-process shape, where the harness opens the gate and
            # then starts the workflow that installs the sink. Its counters
            # began at zero when it was constructed, which was inside the
            # block, so every drop it counted belongs to this run and an empty
            # dict is the correct baseline: `health_delta` reads a missing key
            # as zero.
            #
            # Deliberately NOT the persisted row here. That row belongs to a
            # different writer instance, so subtracting it would either credit
            # this run with a predecessor's drops or, through max(0, ...), hide
            # this run's own. fix-ajv.13.
            appeared = observability_store.existing_observability_sink(
                workflow_folderpath
            )
            if appeared is not None:
                sink = appeared
                run.in_process = True
                run.health_before = {}
                # The problem recorded at open assumed nobody would write here.
                run.extra_problems = [
                    problem
                    for problem in run.extra_problems
                    if not problem.startswith("no writer health is available")
                ]
        if sink is not None and not sink._closed:
            # Flush first: an unflushed queue makes the "after" snapshot describe
            # a moment before the last turns were written, so a drop during that
            # final write would land outside the measured interval.
            # The bool is load-bearing, not a courtesy (`SQLiteTraceSink.flush`
            # says so in its own docstring). `_apply_batch`'s generic failure arm
            # rolls back the whole batch, counts ONE write_error and requeues
            # NOTHING — its comment says "the loss is total" — while
            # `records_dropped` stays 0. `WriterHealthDelta.evidence_valid` reads
            # only the drop counters, so without this check a run that lost an
            # entire batch of turn records, or whose writer thread died, reports
            # itself zero-drop and valid. False here is the only signal that the
            # durability barrier did not settle. fix-bn1 review round 2.
            if not sink.flush():
                run.extra_problems.append(
                    "the durability barrier did not settle: records enqueued "
                    "during this run may not have been committed, so this run "
                    "cannot claim that nothing was lost"
                )
            sink.persist_health()
            run.health_after = _health_snapshot(db_path, sink)
        else:
            # Cross-process: we cannot flush the writer, only wait for it to
            # publish. An unrefreshed row means the writing process never
            # persisted health inside this run's interval, so the "after"
            # numbers describe a moment before the run's last writes — the
            # window in which a drop is most likely and would go unseen.
            before_stamp = str((run.health_before or {}).get("updated_at") or "") or None
            run.health_after = _await_health_refresh(
                db_path, before_stamp, health_settle_s
            )
            after_stamp = str((run.health_after or {}).get("updated_at") or "") or None
            if run.health_after is None:
                run.extra_problems.append(
                    "the writing process published no writer-health row, so this "
                    "run's drop count could not be read at all"
                )
            elif before_stamp is not None and after_stamp == before_stamp:
                run.extra_problems.append(
                    f"the writing process did not publish writer health during this "
                    f"run (row still stamped {before_stamp} after waiting "
                    f"{health_settle_s}s), so the delta describes an interval that "
                    f"ended before the run did"
                )
        run.delta = observability_store.health_delta(run.health_before, run.health_after)

        if archive_dir is not None:
            # The sink is deliberately still open here (fix-7de). An in-process
            # run holds its writer for the whole block — `ExperimentRunner` takes
            # it BEFORE this gate opens, precisely so the verdict above rests on
            # live counters — and until fix-7de `archive_to` refused any store
            # whose writer was open, so every such run recorded "evidence
            # archival failed: WriterStillOpen" and produced no archive at all.
            # `archive_to` now holds that writer still for the snapshot instead
            # (`SQLiteTraceSink.quiesced`), which is also what keeps the writer's
            # heartbeat from landing between the archive's two digests. The
            # refusal survives for the case that earned it: a writer this process
            # cannot reach cannot be held still, and is still refused.
            try:
                target = Path(archive_dir) / f"{run_id}-observability.sqlite3"
                run.archive = observability_store.ObservabilityStore(db_path).archive_to(
                    str(target)
                )
            except Exception as exc:
                run.extra_problems.append(f"evidence archival failed: {exc!r}")
                logger.warning(f"Evidence archival failed for {run_id}: {exc!r}")

        if run.valid:
            logger.info(f"Evidence run {run_id}: valid, no evidence lost")
        else:
            for problem in run.problems():
                logger.warning(f"Evidence run {run_id}: {problem}")
            if raise_on_invalid and not body_failed:
                raise EvidenceRunInvalid(run_id, run.problems())
