"""Sealed-workspace archive GETs for the selection HTTP surface.

Moved verbatim from ``run_chatbot.selection_api``. Callers keep importing
these names from ``selection_api``, which re-exports them.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from fastworkflow.observability import best_run as best_run_module
from fastworkflow.observability import comparison as comparison_module
from fastworkflow.observability import consistency as consistency_module
from fastworkflow.observability import selected_runs as selected_runs_module
from fastworkflow.observability import workspace as workspace_module
from fastworkflow.run_chatbot.selection_api import (
    EXPERIMENTS_PREFIX,
    VIEW_DIFFERENCES,
    VIEW_STEPS,
    WORKSPACE_SELECTED_RUNS_PARAMS,
    WORKSPACE_SELECTED_RUNS_VALIDATION_PARAMS,
    ApiError,
    _alignment_row,
    _consistency_bounds,
    _discovered_passes,
    _exact_int,
    _expected_members,
    _guard,
    _pass_request,
    _pass_scope,
    _population_baseline,
    _projection_payload,
    _refuse_unsupported,
    _require_comparable,
    _scalars,
    _segments,
    _selected_attempts,
    _selection_rule,
    _text,
    _view,
)

# ----------------------------------------------------------------------
# Sealed workspace: the evidence half, read-only
# ----------------------------------------------------------------------
#
# A workspace carries EVIDENCE. It does not carry the workflow's selection
# control, so there is no winner here, no best run and no pair review: those
# are the live machine's judgements and reporting them under an archive's name
# would attribute them to the archive. What an archive CAN answer is what was
# recorded -- attempts, one attempt's projection, and two of them compared --
# and refusing that would mean sealed evidence is the one place you cannot look
# at the evidence.
#
# Nothing below opens a control, creates a file, or resolves a path: stores are
# named by the manifest and opened through the workspace's own read-only
# registry.


class _WorkspaceNames:
    """The two names each archive of a workspace has, resolved on demand.

    The difference is load-bearing. A manifest calls a store whatever its author
    called it -- "alpha", "old-box" -- and that name is how every workspace
    ROUTE addresses it. An `ExecutionRef` names a store by the identity the
    database reports about itself, which is what a live reference already
    carries and what the feedback writer resolves a cross-store reference
    through (`ReadOnlyWorkspaceStoreRegistry.store_id_for_identity`). Publishing
    both, each under its own name, is what stops the two being swapped by
    whoever reads this next.

    Answered from the manifest's DECLARATION, through the registry's public
    `descriptor`, and only for the stores a request actually touches:
    `workspace.stores()` would answer the same question by re-hashing every
    archive the manifest names, and mapping one name onto another is not a
    reason to re-verify a file. The store that is actually read is still leased
    through `registry.open`, which verifies it.

    An identity claimed by more than one of the stores in play is REFUSED, never
    resolved. `store_id_for_identity` already refuses it on the write side for
    the same reason -- the right answer is to fix the manifest, not to pick one
    -- and picking one here would project one archive's evidence under another
    archive's reference, which is the failure a reader cannot see.
    """

    def __init__(self, workspace: Any) -> None:
        self._workspace = workspace
        self._identity: dict[str, Optional[str]] = {}
        self._claimed: dict[str, str] = {}

    def identity_of(self, manifest_store_id: str) -> Optional[str]:
        """The identity this manifest declares for one of its stores."""
        manifest_store_id = str(manifest_store_id)
        if manifest_store_id in self._identity:
            return self._identity[manifest_store_id]
        try:
            declared = self._workspace.registry.descriptor(
                manifest_store_id
            ).store_identity
        except workspace_module.UnknownWorkspaceStore as exc:
            raise ApiError(409, str(exc)) from exc
        declared = str(declared) if declared else None
        if declared is not None:
            owner = self._claimed.get(declared)
            if owner is not None and owner != manifest_store_id:
                raise ApiError(
                    409,
                    f"evidence identity {declared!r} is declared by more than "
                    f"one store of this workspace ({owner}, {manifest_store_id}); "
                    "a reference cannot name which archive it came from until "
                    "the manifest says",
                    ambiguous_identity=declared,
                )
            self._claimed[declared] = manifest_store_id
        self._identity[manifest_store_id] = declared
        return declared

    def ref_store_id(self, manifest_store_id: str) -> str:
        """What a reference to this archive names it by."""
        return self.identity_of(manifest_store_id) or str(manifest_store_id)

    def route_store_id(self, ref_store_id: str) -> str:
        """What a workspace ROUTE addressing the same archive is given.

        A manifest that declared no identity for a store keeps naming it the old
        way, and that store's references do too, so an id nothing claimed passes
        through -- which is what keeps such a workspace readable.
        """
        return self._claimed.get(str(ref_store_id), str(ref_store_id))


class _ManifestScopedReader:
    """Reads archived evidence addressed by evidence IDENTITY.

    The workspace addresses stores by manifest `store_id`; the references this
    module publishes name them by identity, so the two are translated once, here,
    rather than in every caller.
    """

    def __init__(self, workspace: Any, names: _WorkspaceNames) -> None:
        self._inner = comparison_module.WorkspaceExecutionReader(workspace)
        self._names = names

    def turn(self, store_id: str, turn_key: str) -> Optional[dict[str, Any]]:
        return self._inner.turn(self._names.route_store_id(store_id), turn_key)

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        return self._inner.trace(self._names.route_store_id(store_id), turn_key)


def _workspace_attempts(
    workspace: Any, experiment_id: str, task_id: str,
    names: Optional[_WorkspaceNames] = None,
    *,
    turn_refs_for: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Attempt rows of one task, each with the reference its turns support.

    `comparable` is the same fact it is live -- an attempt with no recorded
    turns has nothing to open -- stated here from the manifest's turn refs
    rather than inherited from a control that does not exist.

    `turn_refs_for` bounds which attempts have their turns read at all. An
    attempt outside it is reported with `evidence_state` "not_read" rather
    than "no_turns": what a run recorded is unknown to this row, and saying
    it recorded nothing would be a claim nobody made.
    """
    if names is None:
        names = _WorkspaceNames(workspace)
    rows: list[dict[str, Any]] = []
    for row in workspace.attempts(
        experiment_id, task_id=task_id, turn_refs_for=turn_refs_for
    ):
        read = "turn_refs" in row
        turn_refs = row.get("turn_refs") or []
        keys = tuple(str(ref["logical_turn_key"]) for ref in turn_refs)
        manifest_store_id = str(row["store_id"])
        # The reference names the store the way a reference names a store
        # everywhere else in this system: by the identity the database reports.
        # A live ref already does (an authorized source id IS the identity), and
        # the feedback writer resolves a paired reference by it. Carrying the
        # manifest's name here instead would make an archived pair's comment
        # unwritable and give it a different pair key from the comparison it
        # was written in.
        store_id = names.ref_store_id(manifest_store_id)
        # The reference names what the EVIDENCE records, which in an archive is
        # the segment's local experiment id -- a logical experiment stitched
        # from several runs has an id that exists only in the manifest, and a
        # reference carrying it would fail its own scope check against every
        # turn. The logical id is reported beside it, not inside it.
        local_id = str(row.get("local_experiment_id") or experiment_id)
        ref = (
            None
            if not read or not keys
            else comparison_module.ExecutionRef(
                store_id=store_id,
                turn_keys=keys,
                experiment_id=local_id,
                task_id=task_id,
                attempt=int(row["attempt"]),
                label=f"attempt {row['attempt']}",
            )
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "local_experiment_id": local_id,
                "task_id": task_id,
                "attempt": int(row["attempt"]),
                "label": f"attempt {row['attempt']}",
                "segment_id": row.get("segment_id"),
                "store_id": store_id,
                # The name every /api/workspace/* route addresses this archive
                # by. Published beside the identity so a client opening a turn
                # or scoping a write uses the right one of the two.
                "manifest_store_id": manifest_store_id,
                "execution_status": row.get("execution_status"),
                "execution_finished_at": row.get("execution_finished_at"),
                # The execution completion marker, carried out of the
                # archive rather than inferred from the status: a summary
                # over whole FINISHED runs has to be able to tell a run
                # that stopped from a run that ended.
                "finished": best_run_module.attempt_is_finished(row),
                "outcome": row.get("outcome"),
                "outcome_source": row.get("outcome_source"),
                "reward": row.get("reward"),
                "restarts": row.get("restarts"),
                "turn_count": len(keys) if read else None,
                "comparable": ref is not None,
                "evidence_state": (
                    ("recorded" if ref else "no_turns") if read else "not_read"
                ),
                "evidence_label": (
                    None if ref else "this attempt recorded no turns in this archive"
                ),
                "execution_ref": None if ref is None else ref.as_dict(),
                # No control, so no judgements: an archive has no best run and
                # no reference, and saying so beats showing an empty badge.
                "is_best": False,
                "is_reference": False,
            }
        )
    rows.sort(key=lambda row: (row["attempt"], str(row["segment_id"] or "")))
    return rows


def _with_turn_refs(
    workspace: Any, row: Mapping[str, Any]
) -> dict[str, Any]:
    """The same row, with the turns of THAT attempt read.

    Resolved from the row's own identity -- this archive, this local
    experiment, this task, this attempt -- rather than by looking an attempt
    NUMBER up again. Two segments can record an attempt 1 each, and a second
    search by number could read one archive's turns and hand them to the other
    archive's row, or return metadata from a run that is not the one whose
    population just passed its bounds.

    The metadata stays as it was first read, exactly as the live resolver
    keeps its own. Evidence that changes after that is what the optimistic
    validation is for; re-reading it here would only narrow the window while
    quietly mixing two reads into one answer.
    """
    keys = tuple(
        workspace.attempt_turn_keys(
            store_id=str(row["manifest_store_id"]),
            local_experiment_id=str(row["local_experiment_id"]),
            task_id=str(row["task_id"]),
            attempt=int(row["attempt"]),
        )
    )
    ref = (
        None
        if not keys
        else comparison_module.ExecutionRef(
            store_id=str(row["store_id"]),
            turn_keys=keys,
            experiment_id=str(row["local_experiment_id"]),
            task_id=str(row["task_id"]),
            attempt=int(row["attempt"]),
            label=str(row["label"]),
        )
    )
    resolved = dict(row)
    resolved.update(
        {
            "turn_count": len(keys),
            "comparable": ref is not None,
            "evidence_state": "recorded" if ref else "no_turns",
            "evidence_label": (
                None if ref else "this attempt recorded no turns in this archive"
            ),
            "execution_ref": None if ref is None else ref.as_dict(),
        }
    )
    return resolved


def _workspace_consistency(
    workspace: Any,
    names: _WorkspaceNames,
    reader: Any,
    experiment_id: str,
    task_id: str,
    scalars: Mapping[str, Any],
) -> dict[str, Any]:
    """The same consistency metrics, read out of a sealed archive.

    A human opening an archive and a coding agent reading this route get the
    figures the live route gives, computed by the same module from the same
    projections, so a number quoted from a comparison does not change meaning
    when the evidence is sealed. What an archive does not carry is the
    workflow's selection control, so there is no best run and therefore no
    reference rows -- which is stated rather than shown empty.

    Nothing is written. The derived vector cache is deliberately NOT used
    here: it is keyed to a live workflow's state directory, an archive has no
    such place of its own, and a read of sealed evidence that creates files is
    not a read. The cost is recomputing vectors per request, which is the
    right trade for a surface that must not touch what it measures.
    """
    max_runs, max_pairs = _consistency_bounds(scalars)
    rows = _workspace_attempts(workspace, experiment_id, task_id, names)
    _refuse_ambiguous_attempts(rows, experiment_id)
    embedder, unavailable = consistency_module.shared_embedder()

    def report(rows: list[dict[str, Any]], for_experiment: str) -> dict[str, Any]:
        runs, capped = consistency_module.collect_task_evidence(
            rows, reader, max_runs=max_runs
        )
        payload = consistency_module.task_consistency(
            experiment_id=for_experiment,
            task_id=task_id,
            runs=runs,
            best_attempt=None,
            embedder=embedder,
            embedding_unavailable=unavailable,
            cache=None,
            max_pairs=max_pairs,
            runs_capped=capped,
        )
        payload["sealed"] = True
        payload["reference"] = {
            "kind": "none",
            "usable": False,
            "reason": (
                "a sealed archive carries evidence, not the workflow's "
                "selection control, so it records no best run"
            ),
        }
        return payload

    payload = report(rows, experiment_id)
    compare_experiment = scalars.get("compare_experiment")
    if compare_experiment and compare_experiment != experiment_id:
        other_rows = _workspace_attempts(
            workspace, str(compare_experiment), task_id, names
        )
        _refuse_ambiguous_attempts(other_rows, str(compare_experiment))
        payload["comparison"] = consistency_module.compare_task_consistency(
            report(other_rows, str(compare_experiment)), payload
        )
        payload["compare_experiment_id"] = compare_experiment
    return payload


def _workspace_selected_rows(
    workspace: Any,
    names: "_WorkspaceNames",
    experiment_id: str,
    task_id: str,
    attempts: Optional[list[int]],
    segment_id: Optional[str],
    bounds: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """`(rows for the named attempts, every recorded attempt number, scope)`.

    `attempts=None` is the all-finished rule, resolved from the archive's own
    attempt metadata. Under that rule an attempt number recorded in two
    segments is refused for the whole population, because the population is
    then not one population; a named selection keeps refusing only the
    numbers it actually asked for.

    A sealed archive carries evidence, and summarizing the evidence of runs a
    reader names is a READ of it -- so it is answered here rather than refused
    with the decisions wording, which is about winners and best runs and has
    nothing to say about counting dispatches.

    Two archive-only ambiguities are refused, both about SCOPE and neither
    about decisions. A logical experiment stitched from several segments can
    record attempt 2 twice, and pooling two different runs under one number is
    exactly the error this whole route exists to avoid, so a duplicated
    attempt asks for `segment_id`. A selection whose members resolve to more
    than one archive is refused downstream by the coordinator, because one
    summary over two sources is not a summary of a source.
    """
    # METADATA first, and only metadata: which runs exist and which are
    # finished is recorded on the attempt rows themselves. Reading every
    # attempt's turns to answer it would read the runs this request is about
    # to refuse, or to leave out.
    rows = _workspace_attempts(
        workspace, experiment_id, task_id, names, turn_refs_for=(),
    )
    if segment_id:
        rows = [row for row in rows if str(row["segment_id"]) == segment_id]
    all_finished = attempts is None
    wanted = set() if all_finished else set(attempts)
    # The whole population, for the population report: one segment when named,
    # and an ambiguous attempt number left unstated rather than counted twice.
    seen: dict[int, dict[str, Any]] = {}
    duplicated: set[int] = set()
    for row in rows:
        attempt = int(row["attempt"])
        if attempt in seen:
            duplicated.add(attempt)
        seen.setdefault(attempt, row)
    scope = {
        "ambiguous": bool(duplicated),
        "duplicated": duplicated,
        "segment_id": segment_id,
        "recorded_attempts": sorted(seen),
        "finished_attempts": sorted(
            a for a, row in seen.items() if row.get("finished")
        ),
        "unfinished_attempts": sorted(
            a for a, row in seen.items() if not row.get("finished")
        ),
    }
    if all_finished:
        wanted = set(scope["finished_attempts"])
    # Ambiguity is decided on METADATA, before any turn is read: a request
    # that is about to be refused must not read the runs it refuses.
    ambiguous: dict[int, dict[str, Any]] = {}
    for row in rows:
        attempt = int(row["attempt"])
        if attempt not in wanted:
            continue
        if attempt in ambiguous:
            raise ApiError(
                409,
                f"attempt {attempt} is recorded in more than one segment of "
                "this archive, so a summary over it would pool two different "
                "runs under one number; name segment_id",
                segment_ids=sorted(
                    {
                        str(ambiguous[attempt].get("segment_id") or ""),
                        str(row.get("segment_id") or ""),
                    }
                ),
            )
        ambiguous[attempt] = row
    recorded = [int(row["attempt"]) for row in rows]
    if bounds is not None:
        # Every refusal this request can make -- ambiguity, the selection
        # bound, a foreign baseline -- happens HERE, before one turn is read.
        # A request that is about to be refused must not read the runs it is
        # refusing, and an over-cap task must not be scanned to say it is one.
        bounds(scope)
    # The rows already in hand, not another search: each member's turns are
    # read against the identity of the row that was chosen.
    return (
        [_with_turn_refs(workspace, ambiguous[attempt])
         for attempt in sorted(ambiguous)],
        recorded,
        scope,
    )


def _workspace_archive(
    workspace: Any,
    names: "_WorkspaceNames",
    experiment_id: str,
    segment_id: Optional[str],
) -> dict[str, Any]:
    """The ONE archive a population is a population of, from the manifest.

    Read from the logical experiment's declared segments rather than from the
    rows a task happens to have, so a task that recorded nothing is still
    bound to the archive it would have recorded into. A logical experiment
    stitched from two archives has no single source for a population and is
    refused here -- naming a `segment_id` is how a caller says which one it
    means -- because hashing such a population under a source of None would
    make two different archives' populations share one identity.
    """
    segments = workspace.segments(experiment_id)
    if segment_id:
        segments = [
            segment
            for segment in segments
            if str(segment["segment_id"]) == segment_id
        ]
        if not segments:
            raise ApiError(
                404,
                f"this workspace records no segment {segment_id!r} of "
                f"experiment {experiment_id!r}",
            )
    stores = sorted({str(segment["store_id"]) for segment in segments})
    if len(stores) != 1:
        raise ApiError(
            409,
            f"experiment {experiment_id!r} is stitched from more than one "
            "archive (" + ", ".join(stores) + "), so its runs of this task are "
            "not one population; name segment_id to say which archive you mean",
            store_ids=stores,
        )
    manifest_store_id = stores[0]
    return {
        "manifest_store_id": manifest_store_id,
        "store_id": names.ref_store_id(manifest_store_id),
        "segments": sorted(str(segment["segment_id"]) for segment in segments),
    }


def _workspace_selected_runs(
    workspace: Any,
    names: "_WorkspaceNames",
    reader: Any,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
    *,
    validate: bool,
) -> dict[str, Any]:
    allowed = (
        WORKSPACE_SELECTED_RUNS_VALIDATION_PARAMS
        if validate
        else WORKSPACE_SELECTED_RUNS_PARAMS
    )
    route = "selected-runs/validation" if validate else "selected-runs"
    baseline = _population_baseline(query) if validate else None
    rule = (
        selected_runs_module.SELECTION_EXPLICIT
        if validate
        else _selection_rule(query, route=route)
    )
    if rule == selected_runs_module.SELECTION_ALL_FINISHED:
        _refuse_unsupported(query, allowed, route=route)
        attempts, duplicates = None, 0
    else:
        attempts, duplicates = _selected_attempts(
            query, allowed=allowed, route=route,
            allow_empty=baseline is not None,
        )
    segment_id = (query.get("segment_id") or [None])[0]
    wants_population = (
        rule == selected_runs_module.SELECTION_ALL_FINISHED or baseline is not None
    )
    population: Optional[dict[str, Any]] = None

    def bounds(scope: dict[str, Any]) -> None:
        """Everything that can refuse this request, before any turn is read."""
        nonlocal population
        if not wants_population:
            return
        if scope["ambiguous"]:
            raise ApiError(
                409,
                "this archive records some attempt number of this task in "
                "more than one segment, so its runs of this task are not one "
                "population; name segment_id",
                segment_scoped=False,
            )
        archive = _workspace_archive(workspace, names, experiment_id, segment_id)
        population = selected_runs_module.build_task_population(
            selection_rule=(
                rule if baseline is None else baseline["selection_rule"]
            ),
            experiment_id=experiment_id,
            task_id=task_id,
            source_id=archive["manifest_store_id"],
            store_id=archive["store_id"],
            segment_id=segment_id,
            segments=archive["segments"],
            sealed=True,
            recorded_attempts=scope["recorded_attempts"],
            finished_attempts=scope["finished_attempts"],
            unfinished_attempts=scope["unfinished_attempts"],
        )
        if rule == selected_runs_module.SELECTION_ALL_FINISHED and (
            len(scope["finished_attempts"]) > selected_runs_module.MAX_SELECTED_RUNS
        ):
            selected_runs_module.refuse_over_limit(
                population, experiment_id=experiment_id, task_id=task_id
            )
        if baseline is not None:
            # A foreign or self-contradictory baseline is refused here too,
            # for the same reason: it costs the metadata read and nothing.
            selected_runs_module.population_drift(
                population=population,
                recorded_attempts=scope["recorded_attempts"],
                finished_attempts=scope["finished_attempts"],
                unfinished_attempts=scope["unfinished_attempts"],
                baseline=baseline,
            )

    selected, recorded, scope = _workspace_selected_rows(
        workspace, names, experiment_id, task_id, attempts, segment_id, bounds
    )
    requested = (
        scope["finished_attempts"]
        if rule == selected_runs_module.SELECTION_ALL_FINISHED
        else attempts
    )
    # Both of an archive's names, resolved once and used by both
    # routes: the evidence identity the references carry, and the
    # manifest's own name, which is what every workspace turn and
    # span route is addressed by.
    source_id = selected[0].get("manifest_store_id") if selected else None
    store_id = selected[0].get("store_id") if selected else None
    if validate:
        return selected_runs_module.validate_selection(
            experiment_id=experiment_id,
            task_id=task_id,
            source_id=source_id,
            store_id=store_id,
            requested=requested,
            recorded_attempts=recorded,
            candidate_rows=selected,
            reader=reader,
            expect=_text((query.get("expect") or [None])[0], "expect", required=False),
            expect_members=_expected_members(query),
            sealed=True,
            task_population=population,
            population_baseline=baseline,
            finished_attempts=scope["finished_attempts"],
            unfinished_attempts=scope["unfinished_attempts"],
        )
    return selected_runs_module.aggregate_selected_runs(
        experiment_id=experiment_id,
        task_id=task_id,
        source_id=source_id,
        store_id=store_id,
        requested=requested,
        duplicate_requests=duplicates,
        recorded_attempts=recorded,
        candidate_rows=selected,
        reader=reader,
        sealed=True,
        selection_rule=rule,
        task_population=population,
    )


def _refuse_ambiguous_attempts(
    rows: list[dict[str, Any]], experiment_id: str
) -> None:
    """Refuse a distribution over an attempt number that means two runs.

    A logical experiment stitched from several archive segments can record
    attempt 1 twice. Every figure here is keyed by attempt, so pooling them
    would silently average two different runs into one row -- and quietly
    dropping one would be worse. `/comparison` already refuses the same
    ambiguity by asking for a `segment_id`; there is no segment to ask for
    when the question is about the whole population.
    """
    seen: dict[int, str] = {}
    for row in rows:
        attempt = int(row["attempt"])
        segment = str(row.get("segment_id") or "")
        if attempt in seen:
            raise ApiError(
                409,
                f"attempt {attempt} of task is recorded in more than one "
                f"segment of this archive for {experiment_id!r}, so a "
                "consistency distribution over its attempts would pool two "
                "different runs under one number",
                segment_ids=sorted({seen[attempt], segment}),
            )
        seen[attempt] = segment


def _workspace_side(
    rows: list[dict[str, Any]],
    attempt: Optional[int],
    segment_id: Optional[str],
    *,
    which: str,
) -> dict[str, Any]:
    """One side of an archived comparison, named exactly.

    With no attempt given the side is the first comparable attempt in attempt
    order -- a viewing default, never a judgement, and labelled as such in the
    payload. One logical experiment can be stitched from several segments, so
    two segments can each hold an attempt 2; that is answered by naming
    `segment_id` rather than by picking one.
    """
    candidates = [
        row
        for row in rows
        if (attempt is None or row["attempt"] == attempt)
        and (segment_id is None or str(row["segment_id"]) == segment_id)
    ]
    if attempt is None:
        candidates = [row for row in candidates if row["comparable"]]
        if not candidates:
            raise ApiError(
                409,
                f"the {which} side has no comparable attempt: no attempt of this "
                "task recorded turns in this archive",
                comparable=False,
            )
        return candidates[0]
    if not candidates:
        raise ApiError(
            404,
            f"the {which} side names attempt {attempt}, which this archive does "
            "not record for this task",
        )
    if len(candidates) > 1:
        raise ApiError(
            409,
            f"attempt {attempt} is recorded in more than one segment of this "
            f"archive; name the {which} side's segment_id",
            segment_ids=[str(row["segment_id"]) for row in candidates],
        )
    return candidates[0]


def _workspace_get(
    workspace: Any, parts: list[str], query: dict[str, list[str]]
) -> tuple[int, dict[str, Any]]:
    experiment_id, rest = parts[0], parts[1:]
    if not (len(rest) >= 2 and rest[0] == "tasks"):
        raise ApiError(
            403,
            "a sealed workspace carries evidence, not the workflow's selection "
            "control: winners, best runs and pair review are live-workflow only",
            sealed=True,
        )
    task_id = rest[1]
    tail = rest[2:]
    names = _WorkspaceNames(workspace)
    reader = _ManifestScopedReader(workspace, names)
    scalars = _scalars(query)

    if tail == ["runs"]:
        rows = _workspace_attempts(workspace, experiment_id, task_id, names)
        return 200, {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "attempts": rows,
            "best_run": None,
            "reference": None,
            "sealed": True,
            "decisions_available": False,
        }

    if len(tail) >= 2 and tail[0] == "runs":
        attempt = _exact_int(tail[1], "attempt")
        rows = _workspace_attempts(workspace, experiment_id, task_id, names)
        run = _workspace_side(
            rows, attempt, scalars.get("segment_id"), which="requested"
        )
        ref = _require_comparable(run, "requested")
        pass_id, attribute = _pass_request(scalars, side="left")
        if tail[2:] == ["passes"]:
            recorded = _discovered_passes(reader, ref, attribute)
            return 200, {
                "run": run,
                "pass_attribute": attribute,
                "passes": [
                    {"pass_id": pid, "turn_keys": turns, "turn_count": len(turns)}
                    for pid, turns in sorted(recorded.items())
                ],
            }
        if tail[2:]:
            raise ApiError(404, "not found")
        selector = None
        omitted: list[str] = []
        if pass_id is not None:
            ref, selector, omitted = _pass_scope(
                reader, ref, pass_id, attribute, which="requested"
            )
        view = _view(query)
        projection = comparison_module.project_execution(
            ref, reader, pass_selector=selector
        )
        return 200, {
            "run": run,
            "projection": _projection_payload(
                projection, view, run.get("manifest_store_id")
            ),
            "pass_id": pass_id,
            "pass_attribute": attribute if pass_id else None,
            "pass_turns_omitted": omitted,
            "view": view,
            "sealed": True,
        }

    if tail == ["selected-runs"]:
        return 200, _workspace_selected_runs(
            workspace, names, reader, experiment_id, task_id, query,
            validate=False,
        )

    if tail == ["selected-runs", "validation"]:
        return 200, _workspace_selected_runs(
            workspace, names, reader, experiment_id, task_id, query,
            validate=True,
        )

    if tail == ["consistency"]:
        return 200, _workspace_consistency(
            workspace, names, reader, experiment_id, task_id, scalars
        )

    if tail == ["comparison"]:
        left_rows = _workspace_attempts(workspace, experiment_id, task_id, names)
        left = _workspace_side(
            left_rows,
            None if "left_attempt" not in scalars
            else _exact_int(scalars["left_attempt"], "left_attempt"),
            scalars.get("left_segment_id"),
            which="left",
        )
        right_experiment = scalars.get("right_experiment") or experiment_id
        right_rows = (
            left_rows
            if right_experiment == experiment_id
            else _workspace_attempts(
                workspace, right_experiment, task_id, names
            )
        )
        right = _workspace_side(
            right_rows,
            None if "right_attempt" not in scalars
            else _exact_int(scalars["right_attempt"], "right_attempt"),
            scalars.get("right_segment_id"),
            which="right",
        )
        left_ref = _require_comparable(left, "left")
        right_ref = _require_comparable(right, "right")
        left_pass, attribute = _pass_request(scalars, side="left")
        right_pass, _attribute = _pass_request(scalars, side="right")
        left_selector = right_selector = None
        left_omitted: list[str] = []
        right_omitted: list[str] = []
        if left_pass is not None:
            left_ref, left_selector, left_omitted = _pass_scope(
                reader, left_ref, left_pass, attribute, which="left"
            )
        if right_pass is not None:
            right_ref, right_selector, right_omitted = _pass_scope(
                reader, right_ref, right_pass, attribute, which="right"
            )
        view = _view(query)
        comparison = comparison_module.compare_executions(
            left_ref,
            right_ref,
            reader,
            left_pass=left_selector,
            right_pass=right_selector,
        )
        manifest_store_ids = {
            "left": left.get("manifest_store_id"),
            "right": right.get("manifest_store_id"),
        }
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "view": view,
            "left": _projection_payload(
                comparison.left, view, manifest_store_ids["left"]
            ),
            "right": _projection_payload(
                comparison.right, view, manifest_store_ids["right"]
            ),
            "left_run": left,
            "right_run": right,
            "summary": comparison.summary(),
            # The pair identity is the refs', so an archived pair and the same
            # pair read live agree -- but the archive records no review of it.
            "review_pair_key": comparison_module.review_pair_key(left_ref, right_ref),
            "difference_count": len(comparison.differences()),
            "pass_scope": {
                "attribute": attribute if (left_pass or right_pass) else None,
                "left_pass": left_pass,
                "right_pass": right_pass,
                "left_turns_omitted": left_omitted,
                "right_turns_omitted": right_omitted,
            },
            "sealed": True,
            "review_available": False,
        }
        if view in (VIEW_STEPS, VIEW_DIFFERENCES):
            pairs = (
                comparison.differences()
                if view == VIEW_DIFFERENCES
                else list(comparison.alignment.pairs)
            )
            payload["alignment"] = {
                "summary": comparison.alignment.summary(),
                "rows": [
                    _alignment_row(
                        comparison, pair, left_ref, right_ref, manifest_store_ids
                    )
                    for pair in pairs
                ],
            }
        return 200, payload

    raise ApiError(
        403,
        "a sealed workspace carries evidence, not the workflow's selection "
        "control: winners, best runs and pair review are live-workflow only",
        sealed=True,
    )


def handle_workspace_get(
    workspace: Any, path: str, query: dict[str, list[str]]
) -> Optional[tuple[int, dict[str, Any]]]:
    """Answer a GET against sealed evidence, or None when the path is not ours."""
    parts = _segments(path, EXPERIMENTS_PREFIX)
    if parts is None or len(parts) < 2:
        return None
    return _guard(lambda: _workspace_get(workspace, parts, query))
