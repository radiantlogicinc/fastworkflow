"""Run a task set as one measured object.

An **experiment** is a labelled set of tasks, each run one or more times, whose
turns can be found, scored and compared as a unit. Before `fix-bn1` no such
object existed: `evidence_run()` minted a `run_id` that lived only inside a
bundle JSON, and no column joined a turn back to the run it belonged to.

This module is the producer. `ObservabilityStore` owns the records (schema and
write-once/terminal enforcement); this owns *running* a task set into them.

Design: `docs/experiment_container_design.md`, rulings `[XR1]`–`[XR20]`.

**What a task set is.** Data. A list of `ExperimentTask(task_id, messages)`, or
anything that yields them. Nothing here knows what tau2 is, what a corpus is, or
what "passed" means — a grader decides that and its verdict is recorded with the
name of who decided (`[XR13]`). The 15-task hard set at k=3 is the motivating
case, not a special case.

**Four things this gets right that are easy to get wrong:**

1. **Pre-registration.** The experiment row, with its declared task and attempt
   counts, is written before the first task runs. A denominator recorded
   afterwards is one fitted to whatever survived, which is the failure a
   declaration exists to prevent.
2. **Independence.** One channel, one `Workflow`, one `WorkflowExecutionContext`
   per attempt (`[XR18]`). A shared channel would serialise the run and share
   one `ask_user` pending slot; a shared `workflow_id_str` would hand every
   attempt the same live `Workflow` object out of the registry.
3. **Determinism defeat.** Two caches would otherwise make repeated attempts
   identical and `pass^k` meaningless (`[XR16]`): DSPy's response cache and the
   NLU utterance cache. Both are turned off here, and what was done is recorded
   in the evidence segment.
4. **Never silently partial.** The store computes `complete`; this can only ask
   for it. An attempt that crashed halfway leaves `finished_at IS NULL`, which
   both blocks completion and is what `resume()` selects on.
"""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import fastworkflow
from fastworkflow.observability import evidence_run as evidence_run_module
from fastworkflow import state_paths
from fastworkflow.observability import store as observability_store
from fastworkflow.benchmark.catalog import load_version
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

# The env knobs `[XR16]` needs in the process before any attempt starts. Both
# are process-wide by nature: `fastworkflow.init()` installs one global env dict
# (`__init__.py:244-286`), so an arm that needs different values needs a
# different process.
LM_CACHE_VAR = "FW_LM_CACHE"
UTTERANCE_CACHE_SCOPE_VAR = "FW_UTTERANCE_CACHE_SCOPE"


class ExperimentAborted(RuntimeError):
    """The run could not be completed and the experiment was left `running`.

    Deliberately not the same thing as an invalid experiment. `running` means
    "resumable"; `invalid` means "do not score this, ever". A crash must not
    silently choose the second.
    """


class MissingExperimentLifecycleFeature(RuntimeError):
    """The target store was not installed for driver-neutral lifecycle writes."""


class BenchmarkPinDigestMismatch(ValueError):
    """A supplied benchmark digest does not match the workflow catalog file."""

    def __init__(
        self,
        benchmark_id: str,
        benchmark_version: str,
        expected_digest: str,
        supplied_digest: str,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.benchmark_version = benchmark_version
        self.expected_digest = expected_digest
        self.supplied_digest = supplied_digest
        super().__init__(
            f"benchmark pin digest for {benchmark_id}/{benchmark_version} "
            f"does not match the workflow catalog: expected "
            f"{expected_digest}, got {supplied_digest}"
        )


def experiment_store_readiness(db_path: str) -> dict[str, str]:
    """Describe an installed experiment store for an explicit controller handshake."""
    if not db_path or not os.path.isfile(db_path):
        raise MissingExperimentLifecycleFeature(
            f"{db_path!r} does not exist; the server must install the feature "
            "before readiness"
        )
    store = observability_store.ObservabilityStore(db_path, migrate=False)
    if (
        not store.has_feature(
            observability_store.FEATURE_EXPERIMENT_LIFECYCLE_V1
        )
        or not store.has_feature(
            observability_store.FEATURE_EXPERIMENT_DECLARATIONS_V1
        )
        or not store.experiment_declaration_schema_ready()
        or not store.has_feature(
            observability_store.FEATURE_EXPERIMENT_CLAIMS_V1
        )
        or not store.experiment_claim_schema_ready()
        or not store.has_feature(
            observability_store.FEATURE_EXPERIMENT_SEALING_V1
        )
        or not store.experiment_sealing_schema_ready()
    ):
        raise MissingExperimentLifecycleFeature(
            f"{db_path!r} does not advertise "
            f"{observability_store.FEATURE_EXPERIMENT_CLAIMS_V1!r}"
        )
    store_id = store.store_identity()
    regime = store.capture_regime()
    if store_id is None or regime is None:
        raise MissingExperimentLifecycleFeature(
            f"{db_path!r} has no installed store identity or capture regime"
        )
    return {
        "store_id": store_id,
        "resolved_path": os.path.realpath(db_path),
        "capture_profile": regime[0],
        "capture_policy_version": regime[1],
    }


@dataclass(frozen=True)
class ExperimentTask:
    """One task in a task set.

    `messages` are fed in order through `process_turn`. `startup_action` runs
    first when present, as its own logical turn — the shape
    `_create_user_runtime` uses.

    `task_id` must be stable across runs and across arms: it is the join key
    every score groups by, and it is never derived from the conversation topic
    (LLM-generated and channel-uniquified, so stable is the one thing it is not).
    """

    task_id: str
    messages: Sequence[str] = ()
    startup_action: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptRun:
    """What one attempt produced, handed to the grader."""

    task: ExperimentTask
    attempt: int
    channel_id: str
    conversation_id: Optional[int]
    turn_outputs: list = field(default_factory=list)
    error: Optional[BaseException] = None

    @property
    def turn_keys(self) -> list[str]:
        return [out.turn_key for out in self.turn_outputs if out is not None]

    @property
    def awaiting_user(self) -> bool:
        """Whether the attempt ended suspended on an unanswered question.

        Such an attempt is `incomplete`, not a fail: nobody answered it, so it
        never measured anything. `cancel_pending()` cannot be the remedy — it
        clears in-memory suspension and emits no turn record at all, so the
        stored row would stay `awaiting_user` forever.
        """
        if not self.turn_outputs:
            return False
        last = self.turn_outputs[-1]
        return getattr(getattr(last, "status", None), "value", None) == "awaiting_user"


@dataclass(frozen=True)
class AttemptBootstrap:
    registration_id: str
    secret: str
    channel_id: str
    expires_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "registration_id": self.registration_id,
            "secret": self.secret,
            "channel_id": self.channel_id,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class AttemptClaim:
    registration_id: str
    experiment_id: str
    task_id: str
    attempt: int
    source_key: str
    channel_id: str
    conversation_id: int
    epoch: int
    server_incarnation: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# A grader receives a finished AttemptRun and returns
# (outcome, outcome_source, reward, detail). Returning None defers to the
# built-in fallback below.
Grader = Callable[[AttemptRun], Optional[tuple]]


def derived_outcome(run: AttemptRun) -> tuple:
    """The fallback verdict, named for what it actually measures (`[XR13]`).

    **This is not a pass/fail judgement and must never be presented as one.**
    It reports whether any command in the attempt returned a failure code.

    `TurnOutput.success` is `all(command_output.success ...)` over per-command
    flags that default to True, and `all([])` is True. For a workflow whose
    commands never set `success=False`, this is true whenever the agent did not
    exhaust its iterations — regardless of whether the task was accomplished.
    Worse, the one thing that *does* flip it is a command that failed and the
    agent then recovered from, so an ordinarily successful trajectory that
    probed a wrong id and recovered reads as a failure.

    It is here because a harness with no grader needs *something*, and because
    recording `outcome_source='derived'` makes that choice visible in the
    record instead of letting a fallback masquerade as a measurement. Supply a
    real grader for any number you intend to report.
    """
    if run.error is not None:
        return ("error", "derived", None, {"error": repr(run.error)})
    if run.awaiting_user:
        return ("incomplete", "derived", None, {"reason": "awaiting_user"})
    if not run.turn_outputs:
        return ("incomplete", "derived", None, {"reason": "no turns"})
    clean = all(
        getattr(getattr(out, "status", None), "value", None) == "completed"
        and out.success
        for out in run.turn_outputs
    )
    return (
        "pass" if clean else "fail",
        "derived",
        None,
        {"predicate": "no_command_reported_failure"},
    )


def channel_for(experiment_id: str, task_id: str, attempt: int) -> str:
    """`[XR18]`: one channel per attempt, deterministic so a resume finds it."""
    return f"exp:{experiment_id}:{task_id}:{attempt}"


class ExperimentController:
    """Synchronous metadata-only experiment lifecycle controller.

    External controllers open an already-installed store with ``migrate=False``.
    Construction never acquires a trace sink, starts a writer, or prunes.
    """

    def __init__(
        self,
        db_path: str,
        expected_store_identity: str,
        *,
        migrate: bool = False,
        external: bool = True,
        capture_profile: Optional[str] = None,
        capture_policy_version: Optional[str] = None,
        workflow_folderpath: Optional[str] = None,
    ) -> None:
        if not db_path:
            raise ValueError("db_path is required")
        if not expected_store_identity:
            raise ValueError("expected_store_identity is required")
        if external and migrate:
            raise ValueError("external controllers must open with migrate=False")
        if external and not os.path.isfile(db_path):
            raise MissingExperimentLifecycleFeature(
                f"{db_path!r} does not exist; the server must install the "
                "feature before readiness"
            )
        self.db_path = db_path
        self.external = bool(external)
        # The folder this controller's runs come from, when the caller knows
        # it. Not stored in the DB (no schema change): it is the caller's own
        # fact about this machine, and it travels out again at seal time so a
        # workspace manifest can name the workflow whose benchmark catalogue
        # the run was pinned against.
        self.workflow_folderpath = (
            os.path.abspath(workflow_folderpath) if workflow_folderpath else None
        )
        self.store = observability_store.ObservabilityStore(
            db_path, migrate=migrate
        )
        if (
            not self.store.has_feature(
                observability_store.FEATURE_EXPERIMENT_LIFECYCLE_V1
            )
            or not self.store.has_feature(
                observability_store.FEATURE_EXPERIMENT_DECLARATIONS_V1
            )
            or not self.store.experiment_declaration_schema_ready()
            or not self.store.experiment_claim_schema_ready()
            or not self.store.experiment_sealing_schema_ready()
        ):
            raise MissingExperimentLifecycleFeature(
                f"{db_path!r} does not advertise the required experiment "
                "lifecycle and declaration features; "
                "the server must install the feature before readiness"
            )
        actual_store_identity = self.store.store_identity()
        if actual_store_identity != expected_store_identity:
            raise observability_store.StoreIdentityMismatch(
                f"expected store {expected_store_identity!r}, opened "
                f"{actual_store_identity!r} at {os.path.realpath(db_path)!r}"
            )
        self.store_identity = actual_store_identity
        target_regime = self.store.capture_regime()
        if target_regime is None:
            raise MissingExperimentLifecycleFeature(
                f"{db_path!r} has no installed capture regime"
            )
        target_profile, target_policy_version = target_regime
        incoming = (
            capture_profile or target_profile,
            capture_policy_version or target_policy_version,
        )
        if incoming != target_regime:
            raise observability_store.CaptureRegimeChanged(
                "<target-store>",
                f"{target_profile}/{target_policy_version}",
                f"{incoming[0]}/{incoming[1]}",
            )
        self.capture_profile, self.capture_policy_version = target_regime

    def _require_writer_drained(self) -> None:
        if not self.external:
            return
        if observability_store.sink_for_db_path(self.db_path) is not None:
            raise observability_store.WriterStillOpen(
                f"external evidence for {self.db_path!r} cannot be certified "
                "while its writer is open; call drain_before_certify() after "
                "stopping the owned server"
            )

    def drain_before_certify(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Drain a detectable owned writer and return final persisted health."""
        sink = observability_store.sink_for_db_path(self.db_path)
        if sink is not None:
            sink.close(timeout=timeout)
            if sink._writer.is_alive():
                raise observability_store.WriterStillOpen(
                    f"writer for {self.db_path!r} did not stop within {timeout}s"
                )
        health = self.store.writer_health()
        if health is None:
            raise RuntimeError(
                f"{self.db_path!r} has no final persisted writer health"
            )
        return health

    def register_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        source_key: str,
        channel_id: str,
        *,
        ttl_seconds: float = 300.0,
        recovery: Optional[dict[str, Any]] = None,
    ) -> AttemptBootstrap:
        return AttemptBootstrap(
            **self.store.register_attempt(
                experiment_id,
                task_id,
                attempt,
                source_key,
                channel_id,
                ttl_seconds=ttl_seconds,
                recovery=recovery,
            )
        )

    def claim_attempt(
        self,
        bootstrap: AttemptBootstrap,
        *,
        server_incarnation: str,
        lease_seconds: float = 300.0,
        runtime_snapshot: Optional[dict[str, Any]] = None,
    ) -> AttemptClaim:
        """Claim on behalf of a server, stamping its runtime snapshot (fix-qe2).

        ``runtime_snapshot`` is the claiming server's
        ``runtime_readiness_snapshot``. The FastAPI server takes it in-process
        at its own bind (`run_fastapi_mcp.utils._claim_registered_attempt`);
        a caller claiming from outside the server passes what it fetched from
        that server's ``/probes/readyz?runtime=true``, or None when it has no
        such answer. None is recorded as null, never as an invented snapshot.
        """
        return AttemptClaim(
            **self.store.claim_attempt(
                bootstrap.as_dict(),
                channel_id=bootstrap.channel_id,
                server_incarnation=server_incarnation,
                lease_seconds=lease_seconds,
                runtime_snapshot=runtime_snapshot,
            )
        )

    def bind_claim(
        self, ctx: WorkflowExecutionContext, claim: AttemptClaim
    ) -> None:
        """Bind the store-validated bootstrap result before constructing work."""
        ctx.bind_experiment_claim(claim.as_dict(), self.store)

    def create_experiment(
        self,
        experiment_id: str,
        description: str,
        *,
        declared_tasks: int,
        declared_attempts: int,
        declarations: Iterable[tuple[str, int, str]],
        required_evidence_segments: int = 0,
        arm: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        benchmark_version: Optional[str] = None,
        benchmark_digest_sha256: Optional[str] = None,
        workflow_folderpath: Optional[str] = None,
    ) -> None:
        # A UI-created identity pins the benchmark even when a driver supplies
        # only experiment/task IDs. Unregistered experiments keep their API.
        from fastworkflow.benchmark import setup as benchmark_setup

        folder = workflow_folderpath or self.workflow_folderpath
        registration = None
        if folder:
            try:
                registration, manifest = benchmark_setup.experiment_manifest(folder, experiment_id)
            except KeyError:
                pass
        if registration is not None:
            declarations = list(declarations)
            if {item[0] for item in declarations} != set(registration["task_ids"]):
                raise ValueError("declared task IDs must match the registered benchmark version")
            for supplied, key in ((benchmark_id, "benchmark_id"),
                                  (benchmark_version, "benchmark_version"),
                                  (benchmark_digest_sha256, "benchmark_digest_sha256")):
                if supplied is not None and supplied != registration[key]:
                    raise ValueError(f"{key} differs from the registered experiment")
            target = {"db_path": os.path.abspath(self.db_path), "store_id": self.store_identity}
            if registration.get("store") not in (None, target):
                raise ValueError("experiment is already bound to another evidence store")
            benchmark_id = registration["benchmark_id"]
            benchmark_version = registration["benchmark_version"]
            benchmark_digest_sha256 = registration["benchmark_digest_sha256"]
        if (
            benchmark_id is not None
            and benchmark_version is not None
            and benchmark_digest_sha256 is not None
            and workflow_folderpath is not None
        ):
            loaded = load_version(
                workflow_folderpath, benchmark_id, benchmark_version
            )
            if loaded["digest_sha256"] != benchmark_digest_sha256:
                raise BenchmarkPinDigestMismatch(
                    benchmark_id,
                    benchmark_version,
                    loaded["digest_sha256"],
                    benchmark_digest_sha256,
                )
        # Reserve the registration before writing evidence. Deletion uses the
        # same setup lock; a deleted ID is refused and a bound ID is protected.
        if registration is not None:
            benchmark_setup.bind_experiment(folder, experiment_id, self.db_path, self.store_identity)
        # Bind the evidence source for selection (`fix-9eg.17.1`/`.17.2`). The
        # experiment may already be registered in the workflow's contest — from
        # UI creation, before this store existed — and this is where its
        # evidence is attached to that registration. One control file for the
        # whole workflow: a private sidecar beside this DB would give the same
        # experiment two winners, one of which nobody is looking at.
        control_db_path = None
        if folder:
            control_db_path = benchmark_setup.bind_runner_evidence(
                folder, self.store, self.db_path
            )
        self.store.create_experiment(
            experiment_id,
            description,
            declared_tasks=declared_tasks,
            declared_attempts=declared_attempts,
            required_evidence_segments=required_evidence_segments,
            arm=arm,
            baseline_experiment_id=baseline_experiment_id,
            workflow_name=workflow_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            benchmark_digest_sha256=benchmark_digest_sha256,
            capture_profile=self.capture_profile,
            capture_policy_version=self.capture_policy_version,
            # No workflow folder means no workflow contest to join, and the
            # per-store sidecar stays the default. A folder we could not join
            # records nothing rather than a second, conflicting winner.
            initialize_winner=control_db_path is not None or not folder,
            selection_control_db_path=control_db_path,
        )
        self.store.declare_experiment_attempts(experiment_id, declarations)

    def start_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        channel_id: str,
        *,
        conversation_id: Optional[int] = None,
        source_attempt_key: Optional[dict[str, Any]] = None,
        source_key: Optional[str] = None,
    ) -> None:
        self.store.start_attempt(
            experiment_id,
            task_id,
            attempt,
            channel_id,
            conversation_id=conversation_id,
            source_attempt_key=source_attempt_key,
            source_key=source_key or channel_id,
        )

    def restart_attempt(
        self, experiment_id: str, task_id: str, attempt: int
    ) -> int:
        return self.store.restart_attempt(experiment_id, task_id, attempt)

    def terminalize_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        execution_status: str,
        conversation_id: Optional[int] = None,
    ) -> None:
        self.store.terminalize_attempt(
            experiment_id,
            task_id,
            attempt,
            execution_status=execution_status,
            conversation_id=conversation_id,
        )

    def record_outcome(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store.record_attempt_outcome(
            experiment_id,
            task_id,
            attempt,
            outcome=outcome,
            outcome_source=outcome_source,
            reward=reward,
            detail=detail,
        )

    def finish_attempt(
        self,
        experiment_id: str,
        task_id: str,
        attempt: int,
        *,
        outcome: str,
        outcome_source: str,
        reward: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
        conversation_id: Optional[int] = None,
        execution_status: str = "completed",
    ) -> None:
        """Compatibility operation for drivers with an immediate grader."""
        self.terminalize_attempt(
            experiment_id,
            task_id,
            attempt,
            execution_status=execution_status,
            conversation_id=conversation_id,
        )
        self.record_outcome(
            experiment_id,
            task_id,
            attempt,
            outcome=outcome,
            outcome_source=outcome_source,
            reward=reward,
            detail=detail,
        )

    def record_evidence_segment(
        self,
        experiment_id: str,
        seq: int,
        evidence_run_id: str,
        record: dict[str, Any],
        *,
        claim: Optional[AttemptClaim] = None,
    ) -> None:
        self._require_writer_drained()
        self.store.record_evidence_segment(
            experiment_id,
            seq,
            evidence_run_id,
            record,
            claim=claim.as_dict() if claim else None,
        )

    def complete_experiment(self, experiment_id: str) -> str:
        if self.external:
            self._require_writer_drained()
            return self.store.complete_external_capture(experiment_id)
        return self.store.complete_experiment(experiment_id)

    def seal_workspace_evidence(
        self,
        experiment_id: str,
        destination: str,
        *,
        workflow_folderpath: Optional[str] = None,
    ) -> dict[str, Any]:
        """Freeze captured data, then attach its digest as the sole handle.

        ORDER, and why it is this one (fix-tcg). `complete` is stamped on the
        source row BEFORE the snapshot. The snapshot is byte-immutable — 0444,
        sidecar-free, verified against a digest on every open — so anything
        written to the source afterwards exists only in the source, and the
        archive is the copy a reader opens. Sealing used to archive first, which
        is why the trial's sealed archive reported `capture_complete` about an
        experiment `workspace.json` presented as sealed.

        The digest cannot go the same way: the archive does not exist yet when
        the status is stamped, and a file cannot contain its own hash. It lands
        afterwards on the source row and, through this return value, in the
        manifest. Between the two writes the row is `complete` with no digest —
        an unfinished seal, which `experiment_scores` refuses to report on and
        which this method re-enters rather than rejects, so a seal that lost its
        archive to a full disk is retryable instead of terminal.

        `workflow_folderpath` rides out on the same return value, defaulting to
        the controller's own, for the manifest writer to record as the sealed
        workspace's `workflow_folderpath` (fix-zns). It is what makes a sealed
        archive able to say which workflow's benchmark catalogue its pin refers
        to; without it, a reader can see the pinned digest and has nothing to
        check it against. It is not written to the store: the experiments table
        holds `workflow_name`, not a folder, and this needs no schema change.
        """
        self._require_writer_drained()
        experiment = self.store.get_experiment(experiment_id)
        if experiment is None:
            raise observability_store.ExperimentNotFound(experiment_id)
        if experiment["workspace_archive_sha256"]:
            raise ValueError(
                f"experiment {experiment_id!r} is already sealed under "
                f"{experiment['workspace_archive_sha256']}; re-sealing is "
                "refused — one experiment names one immutable archive"
            )
        if experiment["status"] not in {"capture_complete", "complete"}:
            raise ValueError(
                f"experiment {experiment_id!r} is {experiment['status']!r}; "
                "workspace evidence can only seal capture_complete data"
            )
        # `archive_to` refuses while a live writer holds the DB, and it would do
        # so AFTER the promotion — leaving an unfinished seal for a condition
        # that was knowable beforehand. `_require_writer_drained` above only
        # checks this for an external controller, so check it here for every
        # caller and fail before touching the row. (fix-7de is the related
        # narrower race: a writer that opens BETWEEN this check and the
        # snapshot is still caught by `archive_to`, which is where the
        # authoritative check has to live.)
        if observability_store.sink_for_db_path(self.db_path) is not None:
            raise observability_store.WriterStillOpen(
                f"refusing to seal {self.db_path!r} while its writer is open"
            )
        self.store.begin_workspace_seal(experiment_id)
        # `quiesce_live_writer=False` keeps the seal's refusal unconditional
        # (fix-7de). `archive_to` will now hold a live writer still rather than
        # refuse it, which is right for an evidence run archiving its own DB
        # mid-flight — and wrong here, where the promotion above has already
        # declared the capture complete. A writer that appears between the two
        # is a contract violation, not a scheduling detail, and quiescing it
        # would seal a store somebody is still writing to under a status that
        # says nobody is.
        archive = self.store.archive_to(destination, quiesce_live_writer=False)
        status = self.store.record_workspace_archive(
            experiment_id,
            sha256=archive["sha256"],
            store_identity=archive["store_identity"],
        )
        archive["experiment_id"] = experiment_id
        archive["experiment_status"] = status
        folder = workflow_folderpath or self.workflow_folderpath
        archive["workflow_folderpath"] = (
            os.path.abspath(folder) if folder else None
        )
        return archive

    def invalidate_experiment(
        self,
        experiment_id: str,
        reason: str,
        detail: Optional[str] = None,
    ) -> str:
        return self.store.complete_experiment(
            experiment_id, force_invalid=reason, detail=detail
        )


class ExperimentHarness:
    """Runs a task set as one experiment.

    Usage::

        harness = ExperimentHarness(workflow_folderpath,
                                    description="tau2 hard set, insight #7")
        result = harness.run(tasks, attempts=3, grader=my_grader)

    One harness, one experiment. Reuse across experiments is not supported and
    is not wanted: the declaration is per-run.
    """

    def __init__(
        self,
        workflow_folderpath: str,
        *,
        description: str = "",
        arm: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        benchmark_version: Optional[str] = None,
        benchmark_digest_sha256: Optional[str] = None,
        run_as_agent: bool = True,
        max_workers: int = 4,
        archive_dir: Optional[str] = None,
        defeat_caches: bool = True,
        install_memory_policy: bool = False,
        runs_per_task: int = 1,
    ) -> None:
        self.workflow_folderpath = workflow_folderpath
        self.description = description
        self.arm = arm
        self.baseline_experiment_id = baseline_experiment_id
        self.benchmark_id = benchmark_id
        self.benchmark_version = benchmark_version
        self.benchmark_digest_sha256 = benchmark_digest_sha256
        self.experiment_id = experiment_id or f"exp-{uuid.uuid4().hex}"
        self.run_as_agent = run_as_agent
        self.max_workers = max(1, int(max_workers))
        self.archive_dir = archive_dir
        self.defeat_caches = defeat_caches
        self.install_memory_policy = install_memory_policy
        from fastworkflow.benchmark.setup import validate_runs_per_task

        self.runs_per_task = validate_runs_per_task(runs_per_task)
        self._db_path = state_paths.observability_db(workflow_folderpath)
        bootstrap_store = observability_store.ObservabilityStore(
            self._db_path, migrate=True
        )
        store_identity = bootstrap_store.store_identity()
        if store_identity is None:
            raise MissingExperimentLifecycleFeature(
                f"{self._db_path!r} has no installed store identity"
            )
        self._controller = ExperimentController(
            self._db_path,
            store_identity,
            migrate=False,
            external=False,
        )
        self._store = self._controller.store
        self._lock = threading.Lock()
        self._sink: Optional[observability_store.SQLiteTraceSink] = None

    @classmethod
    def from_benchmark_experiment(cls, workflow_folderpath: str, experiment_id: str, **kwargs):
        """Attach the UI's experiment ID; pass its task IDs to ``run``.

        Task prompts are optional in setup, so the harness supplies actual
        messages through ExperimentTask as usual. No model runs here.

        The registration's `runs_per_task` becomes this harness's default
        repeat count, so the number chosen at setup is the number that runs
        without the caller passing it again — the same value the UI and an
        agent both read.
        """
        from fastworkflow.benchmark.setup import experiment_manifest, validate_runs_per_task

        record, _ = experiment_manifest(workflow_folderpath, experiment_id)
        # The registration's description is a default, not an override: a runner
        # that names its own run wins over what setup recorded.
        kwargs.setdefault("description", record["description"])
        kwargs.setdefault(
            "runs_per_task", validate_runs_per_task(record.get("runs_per_task", 1))
        )
        return cls(workflow_folderpath, experiment_id=experiment_id,
                   benchmark_id=record["benchmark_id"],
                   benchmark_version=record["benchmark_version"],
                   benchmark_digest_sha256=record["benchmark_digest_sha256"], **kwargs)

    # -- the two things that must happen on the main thread ---------------

    def _prepare_process(self) -> dict[str, Any]:
        """Everything that must be done before any attempt thread is released.

        Returns what was done, for the evidence segment: a run whose cache
        posture is not recorded cannot be told apart later from one that forgot.
        """
        posture: dict[str, Any] = {"defeat_caches": self.defeat_caches}
        # One process-wide sink, taken BEFORE `evidence_run()` opens, so the
        # gate's `in_process` verdict rests on this process's live counters
        # rather than on a persisted row it had to wait for. Every attempt
        # shares it: one writer thread per DB is the [R7] contract, and N sinks
        # on one file would be N writer threads racing.
        self._sink = observability_store.get_observability_sink(
            self.workflow_folderpath
        )
        posture["sink"] = "in-process" if self._sink is not None else "none"
        if self.defeat_caches:
            # `[XR16]`, both contaminants. Set through `fastworkflow`'s env dict
            # rather than os.environ so `get_env_var` sees them; the process is
            # the experiment, so process-wide is the right scope.
            #
            # MERGED, never `fastworkflow.init({...})`. `init` REPLACES the
            # process env dict wholesale (`__init__.py:253`), so calling it with
            # two keys would delete `LLM_AGENT`, every `LITELLM_API_KEY_*`, and
            # every threshold the run needs -- and the failure would surface as
            # "DSPy Language Model not provided" from somewhere unrelated, long
            # after the harness had already been blamed and cleared.
            fastworkflow._env_vars[LM_CACHE_VAR] = "0"
            fastworkflow._env_vars[UTTERANCE_CACHE_SCOPE_VAR] = "workflow"
            posture[LM_CACHE_VAR] = "0"
            posture[UTTERANCE_CACHE_SCOPE_VAR] = "workflow"

        # The DSPy memory policy. A long agentic sweep in one process fills
        # DSPy's retention structures unbounded, and unlike the
        # `fastworkflow.chat_session` global this is a place where a bare-WEC
        # harness does NOT match FastAPI: `install_policy` is called only from
        # the server entrypoint.
        #
        # OPT-IN, not default, and this is deliberate. `install_policy` claims
        # DSPy's config-owner THREAD process-wide and has no uninstall, so a
        # library that called it on construction would poison every process that
        # ever built a harness -- including a test process, where the next
        # FastAPI app's lifespan then fails its own `claim_async_owner()`.
        # Observed exactly that way. An entry point running a real experiment
        # should pass `install_memory_policy=True` (or call `install_policy()`
        # itself before constructing the harness); an embedded or test caller
        # should not.
        #
        # Either way the answer is RECORDED, so a sweep that ran without the
        # policy is visible in its own evidence rather than silently different
        # from one that did.
        if self.install_memory_policy:
            try:
                from fastworkflow.run_fastapi_mcp import server_memory

                server_memory.install_policy()
                posture["dspy_memory_policy"] = "installed"
            except Exception as exc:  # pragma: no cover - optional extra
                posture["dspy_memory_policy"] = f"unavailable: {type(exc).__name__}"
                logger.warning(
                    f"Could not install the DSPy memory policy ({exc!r}); a long "
                    "experiment may accumulate DSPy history in this process"
                )
        else:
            posture["dspy_memory_policy"] = "not installed (caller opted out)"
        return posture

    def _warm_routers(self) -> None:
        """Build the NLU pipelines once, before N threads each build a copy.

        `CommandRouter._instances_cache` and `ModelPipeline._instances_cache`
        are unlocked dicts whose `__new__` publishes an UNinitialised instance
        and whose `__init__` sets `_initialised` last, so N cold attempts
        released together each run the full constructor: N concurrent
        TinyBERT+DistilBERT loads onto one shared object. An RSS spike, not
        corruption — but on a box that OOM-kills at 31 GB the distinction stops
        mattering.

        Constructing `CommandRouter` per `___command_info` subdirectory is what
        actually fills those caches; `RoutingRegistry.get_definition` fills a
        different one (`_definitions`) and touches neither. The loop mirrors
        `chat_session.py:217-241`, which is the only other place this is done.
        Also warms the CME workflow's routing definition, which the turn path
        builds and would otherwise have N threads racing to write.
        """
        try:
            from pathlib import Path

            from fastworkflow.command_routing import RoutingRegistry
            from fastworkflow.model_pipeline_training import CommandRouter

            RoutingRegistry.get_definition(self.workflow_folderpath)
            with contextlib.suppress(Exception):
                RoutingRegistry.get_definition(
                    fastworkflow.get_internal_workflow_path(
                        "command_metadata_extraction"
                    )
                )
            command_info_root = Path(self.workflow_folderpath) / "___command_info"
            if command_info_root.is_dir():
                for subdir in command_info_root.iterdir():
                    if subdir.is_dir():
                        with contextlib.suppress(Exception):
                            CommandRouter(str(subdir))
                # The global-context artefacts live in a pseudo-folder named '*'
                # in some workflows.
                with contextlib.suppress(Exception):
                    CommandRouter(str(command_info_root / "*"))
        except Exception as exc:
            logger.debug(f"Router warm-up skipped: {exc!r}")

    # -- running ----------------------------------------------------------

    def run(
        self,
        tasks: Iterable[ExperimentTask],
        *,
        attempts: Optional[int] = None,
        grader: Optional[Grader] = None,
    ) -> dict[str, Any]:
        """Run every task `attempts` times as one experiment.

        The experiment row is created FIRST, with its declaration, before any
        task executes. Everything else runs inside `evidence_run()`, so the run
        gets zero-drop assertion, prune suppression, archival and provenance —
        and an invalid verdict from it makes the experiment invalid.

        `attempts` defaults to this harness's `runs_per_task`, which
        `from_benchmark_experiment` took from the registration — so the repeat
        count chosen at setup is the one that runs. n repeats are n attempts of
        ONE experiment, with distinct attempt identities and channels: they are
        repeated samples of one setup, not n unrelated experiments. Planned,
        running, finished and failed attempts are all preserved; nothing reruns
        until n successes.
        """
        from fastworkflow.benchmark import setup as benchmark_setup
        from fastworkflow.benchmark.setup import validate_runs_per_task

        task_list = list(tasks)
        if not task_list:
            raise ValueError("a task set with no tasks is not an experiment")
        attempts = validate_runs_per_task(
            self.runs_per_task if attempts is None else attempts, field="attempts"
        )
        seen = {t.task_id for t in task_list}
        if len(seen) != len(task_list):
            raise ValueError(
                "task_ids must be unique within a task set: they are the join "
                "key every score groups by"
            )

        declarations = [
            (task.task_id, n, channel_for(self.experiment_id, task.task_id, n))
            for task in task_list
            for n in range(1, attempts + 1)
        ]
        self._controller.create_experiment(
            self.experiment_id,
            self.description,
            declared_tasks=len(task_list),
            declared_attempts=attempts,
            declarations=declarations,
            required_evidence_segments=1,
            arm=self.arm,
            baseline_experiment_id=self.baseline_experiment_id,
            # The same helper setup uses. A comparison group is derived from
            # this string, so registration and execution disagreeing about it
            # opens a second contest instead of raising anything.
            workflow_name=benchmark_setup.workflow_name_for(self.workflow_folderpath),
            benchmark_id=self.benchmark_id,
            benchmark_version=self.benchmark_version,
            benchmark_digest_sha256=self.benchmark_digest_sha256,
            workflow_folderpath=self.workflow_folderpath,
        )
        pairs = [(task, n) for task in task_list for n in range(1, attempts + 1)]
        return self._execute(pairs, grader, seq=1)

    def resume(
        self,
        tasks: Iterable[ExperimentTask],
        *,
        grader: Optional[Grader] = None,
    ) -> dict[str, Any]:
        """Re-run the attempts of a crashed run that never finished.

        Selects on `execution_finished_at IS NULL` — the execution completion
        marker — not on "has no terminal turn". An attempt that crashed after
        turn 3 of 10 has turns and would be skipped by the second test while
        remaining unfinished.

        Each selected attempt is cleared by `restart_attempt`, which deletes its
        conversations and turns in one transaction and bumps `restarts`. That
        deletion is deliberate: `idx_conv_experiment_attempt` is UNIQUE, so a
        second conversation under the same three labels is refused outright, and
        an abandoned partial trajectory is evidence of nothing.
        """
        experiment = self._store.get_experiment(self.experiment_id)
        if experiment is None:
            raise observability_store.ExperimentNotFound(self.experiment_id)
        if experiment["status"] != "running":
            raise ExperimentAborted(
                f"experiment {self.experiment_id} is {experiment['status']!r}; "
                "only a running experiment is resumable"
            )
        by_id = {t.task_id: t for t in tasks}
        pending = [
            row
            for row in self._store.experiment_attempt_rows(self.experiment_id)
            if row["execution_finished_at"] is None
        ]
        declared = int(experiment["declared_attempts"])
        started = {
            (row["task_id"], row["attempt"])
            for row in self._store.experiment_attempt_rows(self.experiment_id)
        }
        pairs: list[tuple] = []
        for row in pending:
            task = by_id.get(row["task_id"])
            if task is None:
                logger.warning(
                    f"Attempt {row['task_id']}#{row['attempt']} is unfinished but "
                    "its task is not in the supplied task set; it cannot be "
                    "resumed and the experiment cannot become complete"
                )
                continue
            self._controller.restart_attempt(
                self.experiment_id, row["task_id"], row["attempt"]
            )
            pairs.append((task, int(row["attempt"])))
        # Attempts that crashed before start_attempt ever ran leave no row at
        # all; they are found by absence, not by an open marker.
        for task in by_id.values():
            for n in range(1, declared + 1):
                if (task.task_id, n) not in started:
                    pairs.append((task, n))
        seq = len(experiment.get("evidence_runs") or []) + 1
        return self._execute(pairs, grader, seq=seq)

    def _execute(
        self, pairs: Sequence[tuple], grader: Optional[Grader], seq: int
    ) -> dict[str, Any]:
        posture = self._prepare_process()
        self._warm_routers()
        grade = grader or derived_outcome
        runs: list[AttemptRun] = []
        body_error: Optional[BaseException] = None

        with evidence_run_module.evidence_run(
            self.workflow_folderpath, archive_dir=self.archive_dir
        ) as evidence:
            try:
                if pairs:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                        # submit + as_completed, NOT pool.map: map's iterator
                        # raises on the first failed future, so an attempt that
                        # raised out of `_run_attempt` would discard every other
                        # attempt's result and leave `runs` empty -- and the
                        # evidence segment would then record `attempts_run: 0`
                        # for a run in which attempts demonstrably ran and wrote
                        # rows. A record whose whole purpose is honesty must not
                        # under-report itself on the crash path.
                        futures = [
                            pool.submit(self._run_attempt, task, n, grade)
                            for task, n in pairs
                        ]
                        for future in as_completed(futures):
                            try:
                                runs.append(future.result())
                            except BaseException as exc:  # noqa: BLE001
                                if body_error is None:
                                    body_error = exc
            except BaseException as exc:  # noqa: BLE001 - held, not re-raised here
                # Held rather than re-raised through the context manager, so the
                # evidence segment below is still recorded. `evidence_run` itself
                # archives on a raising body for the same reason: a run that
                # crashed is exactly when its evidence is most worth keeping, and
                # an experiment whose crash left no segment cannot be told apart
                # later from one that was never started.
                body_error = exc

        record = evidence.as_record()
        record["experiment"] = {
            "experiment_id": self.experiment_id,
            "cache_posture": posture,
            "attempts_run": len(runs),
            "aborted": None if body_error is None else repr(body_error),
        }
        self._controller.record_evidence_segment(
            self.experiment_id, seq, evidence.run_id, record
        )
        if body_error is not None:
            # Left `running`, not `invalid`: the run may be resumable, and only
            # a completed check may declare a verdict.
            raise ExperimentAborted(
                f"experiment {self.experiment_id} aborted: {body_error!r}"
            ) from body_error

        status = self._controller.complete_experiment(self.experiment_id)
        return {
            "experiment_id": self.experiment_id,
            "status": status,
            "evidence_run_id": evidence.run_id,
            "evidence_valid": evidence.valid,
            "evidence_problems": list(evidence.problems()),
            "attempts": len(runs),
            "scores": self._store.experiment_scores(self.experiment_id),
        }

    # -- one attempt ------------------------------------------------------

    def _run_attempt(
        self, task: ExperimentTask, attempt: int, grade: Grader
    ) -> AttemptRun:
        """Drive one attempt on its own channel, workflow and context.

        Every failure mode here is caught and recorded as an attempt outcome
        rather than raised: one task blowing up must not abort the other 44, and
        an attempt that errored is a data point, not a missing row.
        """
        channel_id = channel_for(self.experiment_id, task.task_id, attempt)
        run = AttemptRun(
            task=task, attempt=attempt, channel_id=channel_id, conversation_id=None
        )
        self._controller.start_attempt(
            self.experiment_id,
            task.task_id,
            attempt,
            channel_id,
            source_key=channel_id,
        )
        ctx: Optional[WorkflowExecutionContext] = None
        try:
            ctx = WorkflowExecutionContext(
                run_as_agent=self.run_as_agent,
                session_key=channel_id,
                trace_sink=self._sink,
            )
            ctx.bind_observability_identity(
                channel_id=channel_id,
                experiment_id=self.experiment_id,
                task_id=task.task_id,
                attempt=attempt,
            )
            # A unique workflow_id_str per attempt is not optional: a colliding
            # id returns the registry's existing live Workflow object and
            # overwrites its context, so every attempt would share one
            # application state.
            workflow = fastworkflow.Workflow.create(
                self.workflow_folderpath, workflow_id_str=channel_id
            )
            ctx.bind_app_workflow(workflow)

            if task.startup_action is not None:
                run.turn_outputs.append(ctx.process_action_turn(task.startup_action))
            for message in task.messages:
                run.turn_outputs.append(ctx.process_turn(message))
            run.conversation_id = ctx.observability_conversation_id
        except BaseException as exc:  # noqa: BLE001 - one attempt, not the run
            run.error = exc
            logger.warning(
                f"Attempt {task.task_id}#{attempt} raised {exc!r}; recorded as "
                "an errored attempt"
            )
        finally:
            if ctx is not None:
                if run.conversation_id is None:
                    run.conversation_id = ctx.observability_conversation_id
                with contextlib.suppress(Exception):
                    ctx.close()

        # A grader is caller code and may raise. One task's grader blowing up
        # must not abort the other 44, for the same reason the attempt body's
        # exceptions are caught: an attempt that could not be judged is a data
        # point, not a missing row. The failure is recorded as its own outcome
        # source so it is never mistaken for a real verdict.
        try:
            verdict = grade(run) or derived_outcome(run)
            outcome, source, reward, detail = verdict
        except BaseException as exc:  # noqa: BLE001
            logger.warning(
                f"Grader raised on {task.task_id}#{attempt}: {exc!r}; the attempt "
                "is recorded as incomplete, which blocks a headline score"
            )
            # `incomplete`, not `error`. `error` means the ATTEMPT failed,
            # which is a measurement of the agent and scores as a non-pass. A
            # grader that raised measured nothing about the agent at all, so
            # scoring it as a non-pass would silently attribute the judge's bug
            # to the thing being judged. `incomplete` blocks completion, which
            # forces the grader to be fixed and the attempt re-run.
            outcome, source, reward = "incomplete", "grader_error", None
            detail = {"grader_error": repr(exc)}
        with self._lock:
            self._controller.finish_attempt(
                self.experiment_id,
                task.task_id,
                attempt,
                outcome=outcome,
                outcome_source=source,
                reward=reward,
                detail=detail,
                conversation_id=run.conversation_id,
                execution_status="failed" if run.error is not None else "completed",
            )
        return run
