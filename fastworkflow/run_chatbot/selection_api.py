"""HTTP surface for experiment winners, task best runs, comparison and pair review.

ONE contract, two clients. A person clicking "Use as best run" in the browser
and a coding agent posting the same JSON over HTTP reach the same functions in
`observability/selection.py`, `observability/best_run.py`,
`observability/comparison.py` and `observability/pair_review.py`. There is no
agent-only shortcut and no browser-only field: the wire shapes here ARE the
agent API, which is why they carry `expected_selection_id`, `ref_id` and
`review_pair_key` rather than screen state.

Why a separate module. `server.py` already carries the whole read layer; these
routes are four new surfaces with their own error vocabulary, and folding them
into the request handler would have put ~700 more lines into a 3900-line file.
The handlers below are plain functions of `(workflow_path, path, query/body)`
returning `(status, payload)`, so the same behavior is testable without a
socket AND over real HTTP -- both are exercised.

Discipline this module keeps, because the state it touches is judgements about
immutable evidence:

- **Reads never create a control.** Every GET reads the workflow's live DB
  read-only and answers empty where it or its control tables are absent.
  Control tables conjured by a page load are indistinguishable, afterwards,
  from ones whose decisions were lost.
- **Reads never move a pointer.** Nothing here writes on GET, including the
  automatic first-experiment winner: that is elected by `setup.create_experiment`
  at creation time, which is a write the user asked for.
- **References are derived from recorded evidence, never from the request.**
  A client names an experiment, a task and an attempt NUMBER; the store, the
  turn keys and the `ExecutionRef` come from `best_run`'s projection of the
  attempt rows. There is no route that accepts a turn key, a store path or a
  store id, so there is no path resolver to abuse and no forged scope to check.
- **Only the workflow's own evidence is readable.** `_reader` serves the live
  DB under its recorded identity; a ref naming any other store raises.
- **Pass scope is exposed, never fabricated.** `?left_pass=&right_pass=`
  compare two recorded passes -- including two passes of the SAME turn, which
  is the teacher/student case. The selectors are DISCOVERED from span
  attributes (`discover_pass_selectors`), never described by the request, and
  `.../runs/{n}/passes` reports what a run actually stamped. No producer
  stamps anything today (`fix-txxy`), so that route answers `[]` and naming a
  pass is a 404 -- the surface refuses rather than inventing a split.

Actor policy follows the one the feedback route already uses: a local,
token-holding client declares who it is (`actor`, `actor_kind`), and
`actor_kind` is restricted to the client kinds `FEEDBACK_PROVENANCES` defines.
`system` -- the kind the automatic first winner is recorded under -- is refused
from HTTP, so no client can forge a decision as the machine's own.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable, Mapping, Optional
from urllib.parse import unquote

from fastworkflow import state_paths
from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.observability import best_run as best_run_module
from fastworkflow.observability import command_summary as command_summary_module
from fastworkflow.observability import comparison as comparison_module
from fastworkflow.observability import consistency as consistency_module
from fastworkflow.observability import control as control_module
from fastworkflow.observability import pair_review as pair_review_module
from fastworkflow.observability import selected_runs as selected_runs_module
from fastworkflow.observability import selection
from fastworkflow.observability import workspace as workspace_module
from fastworkflow.observability.store import FEEDBACK_PROVENANCES

EXPERIMENTS_PREFIX = "/api/experiments/"
REGISTRATIONS_PREFIX = "/api/benchmark-experiments/"

# Which kinds a CLIENT may claim. `selection.SYSTEM_ACTOR_KIND` is deliberately
# absent: it marks the winner nobody chose, and a decision that can be claimed
# as automatic is a decision whose provenance means nothing.
CLIENT_ACTOR_KINDS = frozenset(FEEDBACK_PROVENANCES)

# Views of a projection, smallest first. The default is the product default --
# answers and artifacts -- because a full step alignment of a long attempt is
# megabytes, and a header render must not pay for a drill-down nobody opened.
VIEW_ANSWERS = "answers"
VIEW_STEPS = "steps"
VIEW_DIFFERENCES = "differences"
VIEWS = (VIEW_ANSWERS, VIEW_STEPS, VIEW_DIFFERENCES)

_MAX_PAGE = 500
_DEFAULT_PAGE = 100
_MAX_TEXT = 4000

# Where derived consistency vectors live: under the workflow's own state
# directory, beside its other derived caches and deliberately NOT inside any
# evidence database. Deleting it costs a recomputation and nothing else.
_CONSISTENCY_CACHE_DIRNAME = "consistency-vectors"

# The span attribute a pass-stamping producer records. `distillation.py` stamps
# it once per teacher/student pass (`fix-txxy`), so `discover_pass_selectors`
# now names both passes of a distilled turn; a turn that ran one pass, and every
# turn recorded before that, still answers `[]` and is compared whole. The
# routes resolve passes ONLY from recorded spans either way, so a client asking
# for a pass nothing recorded gets a refusal rather than a fabricated split.
DEFAULT_PASS_ATTRIBUTE = "fw.pass"


class ApiError(Exception):
    """A refusal with the status it should be reported under.

    Carries `payload` so a conflict can hand back what is currently true --
    a stale selection answers with the pointer that replaced it, which is what
    makes "re-read and decide again" a mechanical retry rather than advice.
    """

    def __init__(self, status: int, message: str, **payload: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.payload = payload

    def as_response(self) -> tuple[int, dict[str, Any]]:
        return self.status, {"error": self.message, **self.payload}


# ----------------------------------------------------------------------
# Parsing: exact values only
# ----------------------------------------------------------------------


def _exact_int(value: Any, field: str) -> int:
    """An attempt number, refusing everything that silently becomes one.

    Same rule as `best_run._exact_int`, repeated here because HTTP is where
    the loose values arrive: `bool` is an `int` in Python and `int("2.9")`
    would be a `ValueError` while `int(2.9)` is 2 -- a reference to somebody
    else's run, accepted without complaint.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        raise ApiError(400, f"{field} must be an exact integer, not {value!r}")
    return value


def _text(value: Any, field: str, *, required: bool = True) -> Optional[str]:
    if value is None:
        if required:
            raise ApiError(400, f"{field} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"{field} must be non-empty text")
    text = value.strip()
    if len(text) > _MAX_TEXT:
        raise ApiError(400, f"{field} must be at most {_MAX_TEXT} characters")
    return text


def _page(query: dict[str, list[str]]) -> tuple[int, int]:
    """`limit`/`offset`, clamped. Histories are append-only and unbounded."""
    def one(name: str, default: int) -> int:
        raw = (query.get(name) or [None])[0]
        if raw is None:
            return default
        return _exact_int(raw, name)

    limit = one("limit", _DEFAULT_PAGE)
    offset = one("offset", 0)
    if limit < 1 or offset < 0:
        raise ApiError(400, "limit must be at least 1 and offset must not be negative")
    return min(limit, _MAX_PAGE), offset


def _scalars(query: dict[str, list[str]]) -> dict[str, Any]:
    """First value of each query parameter, so one parser serves query and body."""
    return {name: values[0] for name, values in query.items() if values}


def _view(query: dict[str, list[str]]) -> str:
    view = (query.get("view") or [VIEW_ANSWERS])[0] or VIEW_ANSWERS
    if view not in VIEWS:
        raise ApiError(400, "view must be one of " + ", ".join(VIEWS))
    return view


def _refuse_unsupported(
    query: dict[str, list[str]], allowed: frozenset[str], *, route: str
) -> None:
    """Refuse a parameter this route does not implement, rather than drop it.

    Ignoring an unknown `right_experiment`, `pass_id` or `store_id` would
    answer a NARROWER question than the one asked and label the answer with
    the caller's own words for the wider one. Every scope this route does not
    support is therefore a refusal that names what it refused.
    """
    unsupported = sorted(name for name in query if name not in allowed)
    if not unsupported:
        return
    raise ApiError(
        400,
        f"{route} does not accept " + ", ".join(unsupported)
        + ". This route summarizes whole finished runs of the one experiment "
        "and task named in its path, from the source that experiment is "
        "authorized against: there is no pass scope, no second experiment, no "
        "client-named store and no default-store fallback. Accepted "
        "parameters: " + ", ".join(sorted(allowed)),
        refused="unsupported_parameter",
        unsupported=unsupported,
        accepted=sorted(allowed),
    )


def _selected_attempts(
    query: dict[str, list[str]],
    *,
    allowed: frozenset[str],
    route: str,
    allow_empty: bool = False,
) -> tuple[list[int], int]:
    """The runs a request names: exact integers, deduplicated and bounded.

    `allow_empty` is the metadata-only population check, which names no run on
    purpose and therefore re-projects none.
    """
    _refuse_unsupported(query, allowed, route=route)
    raw = list(query.get("attempt") or [])
    if len(raw) > selected_runs_module.MAX_ATTEMPT_PARAMS:
        raise ApiError(
            400,
            f"this request names {len(raw)} attempt parameters; at most "
            f"{selected_runs_module.MAX_ATTEMPT_PARAMS} are parsed",
            refused="too_many_parameters",
            max_parameters=selected_runs_module.MAX_ATTEMPT_PARAMS,
        )
    return selected_runs_module.bound_attempts(
        [_exact_int(value, "attempt") for value in raw], allow_empty=allow_empty
    )


def _expected_members(query: dict[str, list[str]]) -> dict[int, str]:
    """`?expect_member=3:mev-...` repeated, for a drill-down that revalidates
    only the run it is about to open."""
    expected: dict[int, str] = {}
    for value in (query.get("expect_member") or []):
        text = _text(value, "expect_member") or ""
        attempt, separator, digest = text.partition(":")
        if not separator or not digest.strip():
            raise ApiError(
                400,
                "expect_member must be '<attempt>:<evidence digest>', as the "
                "selection summary published it",
            )
        expected[_exact_int(attempt.strip(), "expect_member attempt")] = digest.strip()
    return expected


def _actor(body: dict[str, Any]) -> dict[str, str]:
    """Who is deciding, as the client declares itself.

    The existing local/token policy is unchanged: there is no per-user session
    to read an identity from, and inventing one here would be inventing an auth
    architecture. What IS enforced is that the claim is a client kind -- a
    request cannot file its decision as the system's automatic one.
    """
    actor = _text(body.get("actor"), "actor")
    actor_kind = _text(body.get("actor_kind"), "actor_kind")
    if actor_kind not in CLIENT_ACTOR_KINDS:
        raise ApiError(
            400,
            "actor_kind must be one of " + ", ".join(sorted(CLIENT_ACTOR_KINDS)),
        )
    provenance = _text(body.get("provenance"), "provenance", required=False)
    return {
        "actor": actor,
        "actor_kind": actor_kind,
        "provenance": provenance or actor_kind,
    }


def _reviewer(body: dict[str, Any]) -> dict[str, str]:
    reviewer = _text(body.get("reviewer"), "reviewer")
    kind = _text(body.get("reviewer_kind"), "reviewer_kind")
    if kind not in pair_review_module.REVIEWER_KINDS:
        raise ApiError(
            400,
            "reviewer_kind must be one of "
            + ", ".join(sorted(pair_review_module.REVIEWER_KINDS)),
        )
    return {"reviewer": reviewer, "reviewer_kind": kind}


# ----------------------------------------------------------------------
# Controls and evidence
# ----------------------------------------------------------------------


def _open_control(workflow_path: str, *, write: bool = False) -> selection.SelectionControlStore:
    return benchmark_setup.workflow_control(workflow_path, write=write)


def _registration(workflow_path: str, experiment_id: str) -> Optional[dict[str, Any]]:
    try:
        return benchmark_setup.load_experiment(workflow_path, experiment_id)
    except (KeyError, benchmark_setup.ExperimentDeleted, ValueError):
        return None


def _require_member(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
) -> dict[str, Any]:
    """The comparison group this experiment is in, or a refusal that says why.

    Three different facts, kept apart. An unknown id is 404. A registration
    that exists but never joined the contest (its best-effort registration
    failed) is 409 and says so, because rendering it as "no winner" would
    invite somebody to decide inside a contest they are not looking at. An
    experiment recorded before this build and never enrolled answers with the
    group it derives: a decision naming it enrols it (§2.3), a read does not.
    """
    group = control.group_for_experiment(experiment_id)
    if group is not None:
        return group
    recorded = None if control.store is None else control.store.get_experiment(experiment_id)
    if recorded is not None:
        return selection.comparison_group_identity(recorded)
    if _registration(workflow_path, experiment_id) is None:
        raise ApiError(404, f"unknown experiment {experiment_id!r}")
    raise ApiError(
        409,
        f"experiment {experiment_id!r} is registered but is not a member of "
        "any comparison group in this workflow's contest",
        registered=True,
    )


def _reader(control: selection.SelectionControlStore) -> comparison_module.StoreExecutionReader:
    """`ExecutionReader` over the live DB, under the identity its refs carry."""
    identity = None if control.store is None else control.store.store_identity()
    if identity is None:
        raise ApiError(409, "this workflow has no live evidence database to read runs from")
    return comparison_module.StoreExecutionReader(identity, control.store)


# ----------------------------------------------------------------------
# Resolving one side of a comparison from recorded evidence
# ----------------------------------------------------------------------


def _task_summary(
    control: selection.SelectionControlStore, experiment_id: str, task_id: str
) -> dict[str, Any]:
    return best_run_module.task_run_summary(control, experiment_id, task_id)


# ----------------------------------------------------------------------
# A bounded, explicitly selected set of whole runs (`fix-9eg.3.2.1`)
# ----------------------------------------------------------------------
#
# Deliberately NOT built on `_task_summary`: that projects every recorded
# attempt of the task, and a request naming three of forty runs must neither
# pay for the other thirty-seven nor publish them. `select_task_attempts`
# enumerates the attempt rows and projects the named ones only.

# `scope=all_finished` is the server-resolved rule (`fix-9eg.3.2.2.1`): the
# members are every finished run of this task, decided from the store's own
# attempt metadata rather than from a list a page built out of visible rows.
POPULATION_BASELINE_PARAMS = frozenset(
    {
        "population_scope",
        "population_rule",
        "expect_population",
        "baseline_member",
        "baseline_member_count",
        "baseline_unfinished",
        "baseline_unfinished_complete",
    }
)
SELECTED_RUNS_PARAMS = frozenset({"attempt", "scope"})
SELECTED_RUNS_VALIDATION_PARAMS = (
    frozenset({"attempt", "expect", "expect_member"}) | POPULATION_BASELINE_PARAMS
)
# One archive is one source. `segment_id` is not a wider scope: it is how a
# workspace names WHICH archive an attempt number belongs to when two hold it.
WORKSPACE_SELECTED_RUNS_PARAMS = frozenset({"attempt", "segment_id", "scope"})
WORKSPACE_SELECTED_RUNS_VALIDATION_PARAMS = (
    frozenset({"attempt", "segment_id", "expect", "expect_member"})
    | POPULATION_BASELINE_PARAMS
)


def _selection_rule(query: dict[str, list[str]], *, route: str) -> str:
    """`explicit` or `all_finished`, refusing the mixture.

    A rule and a list are two different questions. Answering a request that
    carries both would publish a figure over one population under the other's
    label, so it is refused rather than resolved by precedence.
    """
    raw = (query.get("scope") or [None])[0]
    if raw is None:
        return selected_runs_module.SELECTION_EXPLICIT
    rule = _text(raw, "scope")
    if rule != selected_runs_module.SELECTION_ALL_FINISHED:
        raise ApiError(
            400,
            f"{route} does not support scope={rule!r}. The only scope it "
            f"resolves is {selected_runs_module.SELECTION_ALL_FINISHED!r}, "
            "which summarizes every finished run of this task; name runs with "
            "?attempt= to summarize a chosen few",
            refused="unsupported_scope",
            accepted=[selected_runs_module.SELECTION_ALL_FINISHED],
        )
    if query.get("attempt"):
        raise ApiError(
            400,
            f"{route} was given both scope="
            f"{selected_runs_module.SELECTION_ALL_FINISHED!r} and named "
            "attempts. Those are two different populations and answering one "
            "under the other's label is the error this refuses; send one or "
            "the other",
            refused="mixed_scope",
        )
    return rule


def _flag(value: Any, field: str) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ApiError(400, f"{field} must be true or false")


_BASELINE_SINGLETONS = (
    "population_scope",
    "expect_population",
    "population_rule",
    "baseline_member_count",
    "baseline_unfinished_complete",
)


def _singleton(query: dict[str, list[str]], name: str) -> Optional[str]:
    """One value or none. Two of anything here is a malformed request.

    A repeated `expect_population` would otherwise be silently narrowed to
    whichever arrived first, and the answer would be about a baseline the
    caller did not entirely send.
    """
    values = query.get(name) or []
    if len(values) > 1:
        raise ApiError(400, f"{name} must be given at most once")
    return values[0] if values else None


def _population_baseline(
    query: dict[str, list[str]]
) -> Optional[dict[str, Any]]:
    """The population a caller was shown, as it may echo it back.

    Two opaque values and some attempt numbers, and none of them is believed
    as a description of anything: both opaque values are re-derived from
    authoritative metadata and COMPARED, and the lists are required to be
    consistent with the digest they arrive with. Nothing in a request can name
    a source, a store or an archive.

    An omitted list is never read as an empty one. `baseline_member_count` is
    required and may be 0, which is how "this population had no finished run"
    is stated; without it a client that simply forgot its members would be
    told every current member is new.
    """
    for name in _BASELINE_SINGLETONS:
        _singleton(query, name)
    scope = _singleton(query, "population_scope")
    expect = _singleton(query, "expect_population")
    members = list(query.get("baseline_member") or [])
    unfinished = list(query.get("baseline_unfinished") or [])
    complete = _singleton(query, "baseline_unfinished_complete")
    declared = _singleton(query, "baseline_member_count")
    rule = _singleton(query, "population_rule")
    if not any((scope, expect, members, unfinished, complete, rule, declared)):
        return None
    missing = [
        name
        for name, value in (
            ("population_scope", scope),
            ("expect_population", expect),
            ("baseline_member_count", declared),
            ("baseline_unfinished_complete", complete),
        )
        if value is None or value == ""
    ]
    if missing:
        raise ApiError(
            400,
            "a population baseline needs " + ", ".join(missing)
            + ", exactly as the summary published them; attempt numbers on "
            "their own are not bound to any scope, and an omitted list is not "
            "an empty one",
            refused="incomplete_population_baseline",
            missing=missing,
        )
    for name, values, cap in (
        ("baseline_member", members, selected_runs_module.MAX_SELECTED_RUNS),
        ("baseline_unfinished", unfinished,
         selected_runs_module.MAX_ATTEMPT_PARAMS),
    ):
        if len(values) > cap:
            raise ApiError(
                400,
                f"this request names {len(values)} {name} parameters; at most "
                f"{cap} are accepted",
                refused="too_many_parameters",
            )
    member_ids = [_exact_int(value, "baseline_member") for value in members]
    unfinished_ids = [
        _exact_int(value, "baseline_unfinished") for value in unfinished
    ]
    declared_count = _exact_int(declared, "baseline_member_count")
    if len(set(member_ids)) != len(member_ids):
        raise ApiError(
            400, "baseline_member names the same attempt twice",
            refused="inconsistent_population_baseline",
        )
    if len(set(unfinished_ids)) != len(unfinished_ids):
        raise ApiError(
            400, "baseline_unfinished names the same attempt twice",
            refused="inconsistent_population_baseline",
        )
    overlap = sorted(set(member_ids) & set(unfinished_ids))
    if overlap:
        raise ApiError(
            400,
            "baseline_member and baseline_unfinished both name attempt(s) "
            + ", ".join(str(attempt) for attempt in overlap)
            + "; a run was either finished or it was not",
            refused="inconsistent_population_baseline",
        )
    if declared_count != len(member_ids):
        raise ApiError(
            400,
            f"baseline_member_count says {declared_count} but "
            f"{len(member_ids)} baseline_member parameter(s) arrived; the "
            "count is what tells an empty population from a forgotten list",
            refused="inconsistent_population_baseline",
        )
    rule = selected_runs_module.SELECTION_ALL_FINISHED if rule is None else _text(
        rule, "population_rule"
    )
    if rule not in selected_runs_module.SELECTION_RULES:
        raise ApiError(
            400,
            "population_rule must be one of "
            + ", ".join(selected_runs_module.SELECTION_RULES),
        )
    return {
        "population_scope": _text(scope, "population_scope"),
        "expect_population": _text(expect, "expect_population"),
        "selection_rule": rule,
        "members": member_ids,
        "unfinished": unfinished_ids,
        "unfinished_complete": _flag(complete, "baseline_unfinished_complete"),
    }


def _planned_runs(
    workflow_path: str, experiment_id: str
) -> tuple[Optional[int], Optional[str]]:
    """How many runs were planned for ONE task, or nothing.

    The experiment registration's `runs_per_task` is a per-task figure and is
    the only plan reported here. An experiment-wide declared attempt total is
    not a task plan and is never substituted for one.
    """
    registration = _registration(workflow_path, experiment_id)
    planned = None if registration is None else registration.get("runs_per_task")
    if isinstance(planned, bool) or not isinstance(planned, int):
        return None, None
    return planned, "experiment_registration.runs_per_task"


def _selected_scope(
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    attempts: Optional[list[int]],
) -> dict[str, Any]:
    """One scoped metadata read, serving members AND population alike.

    `attempts=None` is the all-finished rule. Either way this enumerates the
    attempt rows once and projects only the members, so a population check
    costs no turn read and no unselected trace is ever touched.
    """
    resolved = best_run_module.select_task_attempts(
        control,
        experiment_id,
        task_id,
        attempts,
        max_selected=selected_runs_module.MAX_SELECTED_RUNS,
    )
    if not resolved["evidence_readable"]:
        raise ApiError(
            409,
            "this workflow's evidence is not readable right now, so these runs "
            "cannot be summarized",
        )
    return resolved


def _live_population(
    workflow_path: str,
    experiment_id: str,
    task_id: str,
    resolved: Mapping[str, Any],
    selection_rule: str,
) -> dict[str, Any]:
    """This task's runs, bound to the source the control authorized.

    Derived from the enumerated metadata and never from a member, so a task
    with no finished run at all is scope-bound exactly as tightly as a full
    one.
    """
    planned, planned_source = _planned_runs(workflow_path, experiment_id)
    return selected_runs_module.build_task_population(
        selection_rule=selection_rule,
        experiment_id=experiment_id,
        task_id=task_id,
        store_id=resolved["store_id"],
        recorded_attempts=resolved["recorded_attempts"],
        finished_attempts=resolved["finished_attempts"],
        unfinished_attempts=resolved["unfinished_attempts"],
        planned=planned,
        planned_source=planned_source,
    )


def _selected_runs_payload(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    rule = _selection_rule(query, route="selected-runs")
    if rule == selected_runs_module.SELECTION_ALL_FINISHED:
        _refuse_unsupported(query, SELECTED_RUNS_PARAMS, route="selected-runs")
        attempts, duplicates = None, 0
    else:
        attempts, duplicates = _selected_attempts(
            query, allowed=SELECTED_RUNS_PARAMS, route="selected-runs"
        )
    resolved = _selected_scope(control, experiment_id, task_id, attempts)
    # Only where the population is the question. An explicit selection did not
    # ask about the rest of the task, and publishing a task's whole finished
    # list beside it would put an unbounded array on a legacy payload.
    population = (
        _live_population(workflow_path, experiment_id, task_id, resolved, rule)
        if rule == selected_runs_module.SELECTION_ALL_FINISHED
        else None
    )
    if resolved["over_limit"]:
        selected_runs_module.refuse_over_limit(
            population, experiment_id=experiment_id, task_id=task_id
        )
    return selected_runs_module.aggregate_selected_runs(
        experiment_id=experiment_id,
        task_id=task_id,
        store_id=resolved["store_id"],
        requested=resolved["requested_attempts"],
        duplicate_requests=duplicates,
        recorded_attempts=resolved["recorded_attempts"],
        candidate_rows=resolved["selected"],
        reader=_reader(control),
        selection_rule=rule,
        task_population=population,
    )


def _selected_runs_validation(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    baseline = _population_baseline(query)
    attempts, _duplicates = _selected_attempts(
        query,
        allowed=SELECTED_RUNS_VALIDATION_PARAMS,
        route="selected-runs/validation",
        allow_empty=baseline is not None,
    )
    resolved = _selected_scope(control, experiment_id, task_id, attempts)
    population = (
        None
        if baseline is None
        else _live_population(
            workflow_path, experiment_id, task_id, resolved,
            baseline["selection_rule"],
        )
    )
    return selected_runs_module.validate_selection(
        experiment_id=experiment_id,
        task_id=task_id,
        # The same scope the aggregate was computed under: the result
        # digest is bound to it, so validating without it would
        # compare against a digest this route never published.
        store_id=resolved["store_id"],
        requested=attempts,
        recorded_attempts=resolved["recorded_attempts"],
        candidate_rows=resolved["selected"],
        reader=_reader(control),
        expect=_text((query.get("expect") or [None])[0], "expect", required=False),
        expect_members=_expected_members(query),
        task_population=population,
        population_baseline=baseline,
        finished_attempts=resolved["finished_attempts"],
        unfinished_attempts=resolved["unfinished_attempts"],
    )


def _side(
    summary: dict[str, Any], attempt: Optional[int], *, which: str
) -> dict[str, Any]:
    """The attempt projection a side names, or the pinned default.

    With no attempt given the side is the recorded best run, falling back to
    the `Reference` -- the first completed attempt, which is a viewing default
    and never a judgement. Naming an attempt that is not recorded is a 404 of
    the attempt, not of the task.
    """
    if attempt is None:
        best = summary.get("best_run") or {}
        run = best.get("run") or summary.get("reference")
        if run is None:
            raise ApiError(
                409,
                f"the {which} side has no pinned run: this task has neither a "
                "recorded best run nor a completed attempt to use as reference",
                comparable=False,
            )
        return run
    for row in summary["attempts"]:
        if row["attempt"] == attempt:
            return row
    raise ApiError(
        404,
        f"the {which} side names attempt {attempt} of task "
        f"{summary['task_id']!r}, which this experiment has not recorded",
    )


def _require_comparable(run: dict[str, Any], which: str) -> comparison_module.ExecutionRef:
    """The side's reference, or an explicit refusal that it has no evidence.

    A finished attempt with no recorded turns is a real state and is offered
    for SELECTION; it simply cannot be one half of a comparison, because there
    is nothing to open. Saying that plainly beats rendering an empty panel.
    """
    if not run.get("comparable") or not run.get("execution_ref"):
        raise ApiError(
            409,
            f"the {which} side (attempt {run.get('attempt')}) has no recorded "
            f"turns: {run.get('evidence_label') or 'there is nothing to compare'}",
            comparable=False,
            attempt=run.get("attempt"),
            evidence_state=run.get("evidence_state"),
        )
    return comparison_module.ExecutionRef.from_mapping(run["execution_ref"])


def _row_feedback_pair_key(
    left: comparison_module.ExecutionRef,
    right: comparison_module.ExecutionRef,
    anchors: dict[str, Any],
) -> Optional[str]:
    """The exact key a comment written from THIS row will carry, or None.

    It is the WHOLE comparison's `review_pair_key`. `feedback.FeedbackTarget`
    keeps the reference the reader was looking at and records the anchored turn
    in a field of its own, so `FeedbackAnchors.pair_key` hashes the same two
    references `comparison.review_pair_key` and `pair_review` do: two remarks
    written on different steps of the same two multi-turn attempts belong to
    ONE pair, and counting comments by this key cannot report an annotated pair
    as unannotated.

    None for a one-sided row: a comment there names one execution and feedback
    records no pair at all for it.
    """
    if not anchors.get("left") or not anchors.get("right"):
        return None
    return comparison_module.review_pair_key(left, right)


def _discovered_passes(
    reader: Any, ref: comparison_module.ExecutionRef, attribute_key: str
) -> dict[str, list[str]]:
    """`{pass_id: [turn keys that record it]}` for one reference.

    Discovery only, from recorded span attributes. A reference whose turns
    stamp nothing answers `{}`, which is the honest "this evidence records no
    pass identity" and is what every real attempt answers today.
    """
    found: dict[str, list[str]] = {}
    for turn_key in ref.turn_keys:
        spans = reader.trace(ref.store_id, turn_key)
        for selector in comparison_module.discover_pass_selectors(
            spans, attribute_key=attribute_key
        ):
            found.setdefault(selector.pass_id, []).append(turn_key)
    return found


def _pass_scope(
    reader: Any,
    ref: comparison_module.ExecutionRef,
    pass_id: str,
    attribute_key: str,
    *,
    which: str,
) -> tuple[comparison_module.ExecutionRef, comparison_module.PassSelector, list[str]]:
    """Narrow one side to a RECORDED pass, or refuse.

    Three things happen here and all three are the point.

    The selector is built by `discover_pass_selectors` from the turn's own
    spans, never from the request: a client names a pass, it does not describe
    one, so there is no way to assert membership for activity the span tree
    does not place in the pass.

    The reference is narrowed to the turns that actually record the pass, and
    the omitted ones are returned so the caller can SAY so. A multi-turn
    attempt where only the second turn ran two passes would otherwise be
    refused outright (`project_execution` resolves the selector against every
    named turn), and silently keeping all the turns would report the pass as
    having been absent from turns it was never in.

    The narrowed reference carries `pass_id`, so its `ref_id`, its feedback
    anchors and its `review_pair_key` are the PASS's -- distinct from the whole
    attempt's, which is what keeps a pass comparison from relabelling the
    comments written about the whole run.
    """
    recorded = _discovered_passes(reader, ref, attribute_key)
    turns = recorded.get(pass_id)
    if not turns:
        raise ApiError(
            404,
            f"the {which} side records no pass {pass_id!r} under attribute "
            f"{attribute_key!r}; recorded passes here: "
            + (", ".join(sorted(recorded)) or "none"),
            recorded_passes=sorted(recorded),
        )
    omitted = [key for key in ref.turn_keys if key not in turns]
    scoped = comparison_module.ExecutionRef(
        store_id=ref.store_id,
        turn_keys=tuple(key for key in ref.turn_keys if key in turns),
        experiment_id=ref.experiment_id,
        task_id=ref.task_id,
        attempt=ref.attempt,
        pass_id=pass_id,
        label=ref.label,
    )
    selector = comparison_module.PassSelector(
        pass_id=pass_id, attribute_key=attribute_key, attribute_value=pass_id
    )
    return scoped, selector, omitted


def _pass_request(
    source: dict[str, Any], *, side: str
) -> tuple[Optional[str], str]:
    """`(pass_id, attribute_key)` as a request states them."""
    raw = source.get(f"{side}_pass")
    pass_id = None if raw is None else _text(raw, f"{side}_pass")
    attribute = source.get("pass_attribute")
    key = DEFAULT_PASS_ATTRIBUTE if attribute is None else _text(
        attribute, "pass_attribute"
    )
    return pass_id, key


def _run_header(run: dict[str, Any]) -> dict[str, Any]:
    """What the pane's chrome shows about a side, status first and unlaundered."""
    return {
        "experiment_id": run.get("experiment_id"),
        "task_id": run.get("task_id"),
        "attempt": run.get("attempt"),
        "label": run.get("label"),
        "is_best": run.get("is_best", False),
        "is_reference": run.get("is_reference", False),
        "execution_status": run.get("execution_status"),
        "outcome": run.get("outcome"),
        "outcome_source": run.get("outcome_source"),
        "reward": run.get("reward"),
        "restarts": run.get("restarts"),
        "turn_count": run.get("turn_count"),
        "comparable": run.get("comparable"),
        "evidence_state": run.get("evidence_state"),
        "evidence_label": run.get("evidence_label"),
    }


def _projection_payload(
    projection: comparison_module.ExecutionProjection,
    view: str,
    manifest_store_id: Optional[str] = None,
) -> dict[str, Any]:
    """One side's wire shape, trimmed to the view that was asked for.

    `manifest_store_id` is set for sealed evidence only, where the manifest's
    name for an archive and the identity the reference carries are two different
    strings and a client needs to know which is which.
    """
    data = projection.as_dict()
    data["step_count"] = len(projection.steps)
    data["unassigned_step_count"] = len(projection.unassigned_steps)
    # Derived from the canonical projection BEFORE the view trims the steps it
    # was derived from, so the default answers view carries the same summary
    # the steps view does instead of a shorter one. Additive: a client that
    # does not know the key reads exactly what it read before.
    data["command_summary"] = command_summary_module.summarize_projection(projection)
    if manifest_store_id:
        data["manifest_store_id"] = manifest_store_id
    if view == VIEW_ANSWERS:
        data.pop("steps", None)
        data.pop("unassigned_steps", None)
    return data


# ----------------------------------------------------------------------
# Winner: read
# ----------------------------------------------------------------------


def _winner_payload(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
) -> dict[str, Any]:
    group = _require_member(workflow_path, control, experiment_id)
    group_id = str(group["group_id"])
    winner = control.current_winner(group_id)
    registration = _registration(workflow_path, experiment_id)
    return {
        "experiment_id": experiment_id,
        "group": group,
        "winner": winner,
        # Echo this into the next decision. A winner that moved while the page
        # was open answers the write with 409 instead of overwriting silently.
        "expected_selection_id": None if winner is None else winner["selection_id"],
        "is_winner": winner is not None and winner["experiment_id"] == experiment_id,
        "automatic": bool(winner and winner.get("automatic")),
        "members": control.group_members(group_id),
        "registration": registration,
        "runs_per_task": None if registration is None else registration.get("runs_per_task"),
        "control_exists": True,
    }


# ----------------------------------------------------------------------
# GET
# ----------------------------------------------------------------------


def _get_experiments(
    workflow_path: str, parts: list[str], query: dict[str, list[str]]
) -> tuple[int, dict[str, Any]]:
    experiment_id = parts[0]
    rest = parts[1:]
    control = _open_control(workflow_path)

    if rest == ["winner"]:
        return 200, _winner_payload(workflow_path, control, experiment_id)

    if rest == ["winner", "history"]:
        group = _require_member(workflow_path, control, experiment_id)
        limit, offset = _page(query)
        history = control.decision_history(
            str(group["group_id"]), limit=limit, offset=offset
        )
        return 200, {
            "experiment_id": experiment_id,
            "group_id": str(group["group_id"]),
            "history": history,
            "limit": limit,
            "offset": offset,
        }

    if len(rest) >= 2 and rest[0] == "tasks":
        return _get_task(workflow_path, control, experiment_id, rest[1], rest[2:], query)

    raise ApiError(404, "not found")


def _get_task(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    rest: list[str],
    query: dict[str, list[str]],
) -> tuple[int, dict[str, Any]]:
    _require_member(workflow_path, control, experiment_id)

    if rest == ["runs"]:
        return 200, _task_summary(control, experiment_id, task_id)

    if rest == ["runs", "history"]:
        limit, offset = _page(query)
        return 200, {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "history": best_run_module.best_run_history(
                control, experiment_id, task_id, limit=limit, offset=offset
            ),
            "limit": limit,
            "offset": offset,
        }

    if len(rest) == 3 and rest[0] == "runs" and rest[2] == "passes":
        attempt = _exact_int(rest[1], "attempt")
        summary = _task_summary(control, experiment_id, task_id)
        run = _side(summary, attempt, which="requested")
        ref = _require_comparable(run, "requested")
        _pass_id, attribute = _pass_request(_scalars(query), side="left")
        recorded = _discovered_passes(_reader(control), ref, attribute)
        return 200, {
            "run": _run_header(run),
            "pass_attribute": attribute,
            # `[]` for a turn nothing stamped, which is every turn that did not
            # run distillation and every turn recorded before `fix-txxy`: those
            # are compared whole rather than split on a guess.
            "passes": [
                {"pass_id": pass_id, "turn_keys": turns, "turn_count": len(turns)}
                for pass_id, turns in sorted(recorded.items())
            ],
        }

    if len(rest) == 2 and rest[0] == "runs":
        attempt = _exact_int(rest[1], "attempt")
        summary = _task_summary(control, experiment_id, task_id)
        run = _side(summary, attempt, which="requested")
        ref = _require_comparable(run, "requested")
        view = _view(query)
        reader = _reader(control)
        pass_id, attribute = _pass_request(_scalars(query), side="left")
        selector = None
        omitted: list[str] = []
        if pass_id is not None:
            ref, selector, omitted = _pass_scope(
                reader, ref, pass_id, attribute, which="requested"
            )
        projection = comparison_module.project_execution(
            ref, reader, pass_selector=selector
        )
        return 200, {
            "run": _run_header(run),
            "projection": _projection_payload(projection, view),
            "pass_id": pass_id,
            "pass_attribute": attribute if pass_id else None,
            "pass_turns_omitted": omitted,
            "view": view,
        }

    if rest == ["selected-runs"]:
        return 200, _selected_runs_payload(
            workflow_path, control, experiment_id, task_id, query
        )

    if rest == ["selected-runs", "validation"]:
        return 200, _selected_runs_validation(
            workflow_path, control, experiment_id, task_id, query
        )

    if rest == ["comparison"]:
        return 200, _comparison_payload(workflow_path, control, experiment_id, task_id, query)

    if rest == ["consistency"]:
        return 200, _consistency_payload(
            workflow_path, control, experiment_id, task_id, query
        )

    if rest == ["review-pairs"]:
        return 200, _review_pairs_payload(
            workflow_path, control, experiment_id, task_id, query
        )

    if rest == ["review-pairs", "history"]:
        pair_key = _text((query.get("pair_key") or [None])[0], "pair_key")
        limit, _offset = _page(query)
        history = pair_review_module.PairReviewStore(control.store).history(
            pair_key,
            reviewer=(query.get("reviewer") or [None])[0],
            limit=limit,
        )
        return 200, {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "pair_key": pair_key,
            "history": history,
            "limit": limit,
            "control_exists": control_module.present(control.store),
        }

    raise ApiError(404, "not found")


def _comparison_sides(
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Both sides, resolved from recorded attempts of ONE task.

    `right_experiment` is how winner-versus-candidate is expressed: the same
    task under another experiment of the same contest, whose evidence may live
    in a different authorized store. Nothing here accepts a turn key or a
    store, so neither side can name evidence the control has not admitted.
    """
    def attempt_of(name: str) -> Optional[int]:
        raw = (query.get(name) or [None])[0]
        return None if raw is None else _exact_int(raw, name)

    left_summary = _task_summary(control, experiment_id, task_id)
    left = _side(left_summary, attempt_of("left_attempt"), which="left")

    right_experiment = (query.get("right_experiment") or [None])[0] or experiment_id
    right_summary = (
        left_summary
        if right_experiment == experiment_id
        else _task_summary(control, right_experiment, task_id)
    )
    right = _side(right_summary, attempt_of("right_attempt"), which="right")
    return left, right


def _comparison_payload(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    right_experiment = (query.get("right_experiment") or [None])[0]
    if right_experiment and right_experiment != experiment_id:
        _require_member(workflow_path, control, right_experiment)
    left, right = _comparison_sides(control, experiment_id, task_id, query)
    left_ref = _require_comparable(left, "left")
    right_ref = _require_comparable(right, "right")
    view = _view(query)
    reader = _reader(control)
    scalars = _scalars(query)
    # Two passes of ONE turn is the case this exists for -- teacher and
    # student recorded side by side -- so each side resolves independently and
    # both may narrow to the same turn of the same attempt.
    left_pass, attribute = _pass_request(scalars, side="left")
    right_pass, _attribute = _pass_request(scalars, side="right")
    left_omitted: list[str] = []
    right_omitted: list[str] = []
    left_selector = right_selector = None
    if left_pass is not None:
        left_ref, left_selector, left_omitted = _pass_scope(
            reader, left_ref, left_pass, attribute, which="left"
        )
    if right_pass is not None:
        right_ref, right_selector, right_omitted = _pass_scope(
            reader, right_ref, right_pass, attribute, which="right"
        )
    comparison = comparison_module.compare_executions(
        left_ref,
        right_ref,
        reader,
        left_pass=left_selector,
        right_pass=right_selector,
    )
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "view": view,
        "left": _projection_payload(comparison.left, view),
        "right": _projection_payload(comparison.right, view),
        "left_run": _run_header(left),
        "right_run": _run_header(right),
        "summary": comparison.summary(),
        # Pass-aware, because the refs are: comments on the teacher pass do not
        # become comments on the whole attempt. ONE identity for this pair --
        # pair review records it under this key and a comment written from any
        # row of it carries the same one, so comment counts and review progress
        # are about the same thing without either client guessing.
        "review_pair_key": comparison_module.review_pair_key(left_ref, right_ref),
        "difference_count": len(comparison.differences()),
        "pass_scope": {
            "attribute": attribute if (left_pass or right_pass) else None,
            "left_pass": left_pass,
            "right_pass": right_pass,
            "left_turns_omitted": left_omitted,
            "right_turns_omitted": right_omitted,
        },
    }
    if view in (VIEW_STEPS, VIEW_DIFFERENCES):
        pairs = (
            comparison.differences()
            if view == VIEW_DIFFERENCES
            else list(comparison.alignment.pairs)
        )
        payload["alignment"] = {
            "summary": comparison.alignment.summary(),
            "rows": [_alignment_row(comparison, pair, left_ref, right_ref)
                     for pair in pairs],
        }
    return payload


def _alignment_row(
    comparison: comparison_module.ExecutionComparison,
    pair: comparison_module.AlignedPair,
    left_ref: comparison_module.ExecutionRef,
    right_ref: comparison_module.ExecutionRef,
    manifest_store_ids: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, Any]:
    """One compare-view row, carrying everything a comment on it needs.

    `anchors` is what the feedback API validates directly; the client adds only
    `target_label`. Each side carries its whole `ref` beside the anchored
    `turn_key`, which is the shape `feedback.FeedbackTarget.from_mapping`
    accepts, so a client posts the row's anchor VERBATIM and never rebuilds a
    reference from screen state. Handing over only the anchored turn would
    narrow the comment's scope to that turn and give it a different pair
    identity from the comparison it was written in.

    `feedback_pair_key` is the identity that comment will be stored under, so
    the row can show its own comments without a second round trip to work out
    which key to ask for.
    """
    anchors = comparison_module.anchors_for_pair(comparison, pair)
    for side, ref in (("left", left_ref), ("right", right_ref)):
        if not anchors.get(side):
            continue
        anchors[side]["ref"] = ref.as_dict()
        # Sealed evidence only. The ref names its archive by identity, which is
        # what a paired write resolves through; the manifest's own name for the
        # same archive is what a READ route is addressed by, and a client needs
        # both. Outside a workspace there is one name and this is absent.
        manifest_store_id = (manifest_store_ids or {}).get(side)
        if manifest_store_id:
            anchors[side]["manifest_store_id"] = manifest_store_id
    return dict(
        pair.as_dict(),
        anchors=anchors,
        feedback_pair_key=_row_feedback_pair_key(left_ref, right_ref, anchors),
    )


def _pair_rows(
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The pinned side and every pair it forms with the other recorded runs.

    The universe of pairs is the caller's question, so it is stated here and
    handed to the pair-review store rather than invented inside it: pinned-versus-each
    comparable attempt of this task, in attempt order, optionally against
    another experiment's attempts of the same task.
    """
    left_summary = _task_summary(control, experiment_id, task_id)
    raw_left = (query.get("left_attempt") or [None])[0]
    left = _side(
        left_summary,
        None if raw_left is None else _exact_int(raw_left, "left_attempt"),
        which="left",
    )
    left_ref = _require_comparable(left, "left")
    reader = _reader(control)
    scalars = _scalars(query)
    left_pass, attribute = _pass_request(scalars, side="left")
    right_pass, _attribute = _pass_request(scalars, side="right")
    if left_pass is not None:
        left_ref, _selector, _omitted = _pass_scope(
            reader, left_ref, left_pass, attribute, which="left"
        )

    right_experiment = (query.get("right_experiment") or [None])[0] or experiment_id
    right_summary = (
        left_summary
        if right_experiment == experiment_id
        else _task_summary(control, right_experiment, task_id)
    )
    rows = []
    for run in right_summary["attempts"]:
        if not run.get("comparable") or not run.get("execution_ref"):
            continue
        if (
            right_experiment == experiment_id
            and run["attempt"] == left["attempt"]
            and right_pass == left_pass
        ):
            continue
        right_ref = comparison_module.ExecutionRef.from_mapping(run["execution_ref"])
        if right_pass is not None:
            try:
                right_ref, _selector, _omitted = _pass_scope(
                    reader, right_ref, right_pass, attribute, which="right"
                )
            except ApiError:
                # An attempt that never recorded the pass is not a pair, it is
                # simply not in this universe. Omitting it beats offering a
                # row that would 404 the moment somebody opened it.
                continue
        rows.append(
            {
                "pair_key": comparison_module.review_pair_key(left_ref, right_ref),
                "right": _run_header(run),
                "right_pass": right_pass,
            }
        )
    return left, rows


def _review_pairs_payload(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    reviewer = _text((query.get("reviewer") or [None])[0], "reviewer")
    right_experiment = (query.get("right_experiment") or [None])[0]
    if right_experiment and right_experiment != experiment_id:
        _require_member(workflow_path, control, right_experiment)
    left, rows = _pair_rows(control, experiment_id, task_id, query)
    progress = pair_review_module.PairReviewStore(control.store).progress(
        reviewer, [row["pair_key"] for row in rows]
    )
    by_key = {state["pair_key"]: state for state in progress["states"]}
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "left_run": _run_header(left),
        # Zero comments does not imply unreviewed, and unreviewed does not
        # imply unseen: the state is the reviewer's own mark and nothing else.
        "pairs": [dict(row, state=by_key.get(row["pair_key"])) for row in rows],
        "progress": {key: progress[key] for key in
                     ("reviewer", "pairs", "reviewed", "remaining", "next_pair_key")},
        "control_exists": control_module.present(control.store),
    }


# ----------------------------------------------------------------------
# Consistency between the repeated runs of one task (`fix-9eg.17.5`)
# ----------------------------------------------------------------------
#
# A read of the SAME evidence the Runs and Compare views already show, summed
# up differently. It opens no control it would not otherwise open, writes
# nothing to any evidence store, and elects nobody: the payload is descriptive
# and says so in its own words.
#
# Derived vectors are cached under the workflow's state directory, never inside
# an evidence database, so this route cannot modify what it measures.


def _consistency_cache(workflow_path: str) -> consistency_module.VectorCache:
    return consistency_module.VectorCache(
        os.path.join(
            state_paths.workflow_state_dir(str(workflow_path)),
            _CONSISTENCY_CACHE_DIRNAME,
        )
    )


def _consistency_bounds(scalars: Mapping[str, Any]) -> tuple[int, int]:
    """`(max_runs, max_pairs)`, clamped. Unbounded work is a denial of service."""
    raw_runs = scalars.get("max_runs")
    raw_pairs = scalars.get("max_pairs")
    max_runs = (
        consistency_module.DEFAULT_MAX_RUNS
        if raw_runs is None
        else _exact_int(raw_runs, "max_runs")
    )
    max_pairs = (
        consistency_module.DEFAULT_MAX_PAIRS
        if raw_pairs is None
        else _exact_int(raw_pairs, "max_pairs")
    )
    if max_runs < 2 or max_pairs < 1:
        raise ApiError(
            400, "max_runs must be at least 2 and max_pairs must be at least 1"
        )
    return (
        min(max_runs, consistency_module.DEFAULT_MAX_RUNS),
        min(max_pairs, consistency_module.DEFAULT_MAX_PAIRS),
    )


def _consistency_report(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    *,
    reader: Any,
    embedder: Any,
    unavailable: Optional[Mapping[str, Any]],
    max_runs: int,
    max_pairs: int,
) -> dict[str, Any]:
    summary = _task_summary(control, experiment_id, task_id)
    runs, capped = consistency_module.collect_task_evidence(
        summary["attempts"], reader, max_runs=max_runs
    )
    best = summary.get("best_run") or {}
    return consistency_module.task_consistency(
        experiment_id=experiment_id,
        task_id=task_id,
        runs=runs,
        best_attempt=best.get("attempt"),
        embedder=embedder,
        embedding_unavailable=unavailable,
        cache=_consistency_cache(workflow_path),
        max_pairs=max_pairs,
        runs_capped=capped,
    )


def _consistency_payload(
    workflow_path: str,
    control: selection.SelectionControlStore,
    experiment_id: str,
    task_id: str,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    """This task's consistency, and optionally another experiment's beside it.

    `compare_experiment` is the across-experiments case: the SAME task under
    another member of the same contest. Both sides are measured by the same
    code in the same process here, so their metric identities agree -- and the
    comparison still checks, because a stored or a future cross-process report
    is the case where they will not.
    """
    scalars = _scalars(query)
    max_runs, max_pairs = _consistency_bounds(scalars)
    # One process-wide load. A page view must not re-read the weights, and a
    # machine without the model must not re-fail the lookup per request.
    embedder, unavailable = consistency_module.shared_embedder()
    reader = _reader(control)
    payload = _consistency_report(
        workflow_path,
        control,
        experiment_id,
        task_id,
        reader=reader,
        embedder=embedder,
        unavailable=unavailable,
        max_runs=max_runs,
        max_pairs=max_pairs,
    )

    compare_experiment = scalars.get("compare_experiment")
    if compare_experiment and compare_experiment != experiment_id:
        _require_member(workflow_path, control, compare_experiment)
        other = _consistency_report(
            workflow_path,
            control,
            compare_experiment,
            task_id,
            reader=reader,
            embedder=embedder,
            unavailable=unavailable,
            max_runs=max_runs,
            max_pairs=max_pairs,
        )
        # Baseline is the OTHER experiment and candidate is this one, so the
        # delta reads "what changed when we moved to the experiment being
        # looked at" rather than depending on which id is in the path.
        payload["comparison"] = consistency_module.compare_task_consistency(
            other, payload
        )
        payload["compare_experiment_id"] = compare_experiment
    return payload


# ----------------------------------------------------------------------
# Writes
# ----------------------------------------------------------------------


def _post_winner_decision(
    workflow_path: str, experiment_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    control = _open_control(workflow_path, write=True)
    group = _require_member(workflow_path, control, experiment_id)
    decision = _text(body.get("decision"), "decision")
    if decision not in selection.CLIENT_DECISIONS:
        raise ApiError(
            400,
            "decision must be one of " + ", ".join(sorted(selection.CLIENT_DECISIONS)),
        )
    who = _actor(body)
    candidate = body.get("candidate_experiment_id")
    if candidate is None and decision == selection.DECISION_PROMOTE:
        # Promotion names what it promotes; without it the route would have to
        # guess between "this experiment" and "whatever was on screen".
        candidate = experiment_id
    result = control.record_decision(
        str(group["group_id"]),
        decision,
        # Null when the reader saw no winner: a promotion into an empty
        # contest is still a compare-and-set, against "nobody".
        expected_selection_id=_text(
            body.get("expected_selection_id"), "expected_selection_id", required=False
        ),
        candidate_experiment_id=None if candidate is None else _text(
            candidate, "candidate_experiment_id"
        ),
        rationale=_text(body.get("rationale"), "rationale", required=False),
        **who,
    )
    # `record_decision` answers with the winner state carrying the decision
    # inside it. Split here so `decision` means the same thing on this route as
    # it does on the best-run routes: the record of what was just filed.
    return 201, {
        "decision": result["decision"],
        "winner": _winner_payload(workflow_path, control, experiment_id),
    }


def _post_best_run(
    workflow_path: str, experiment_id: str, task_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    control = _open_control(workflow_path, write=True)
    _require_member(workflow_path, control, experiment_id)
    if "attempt" not in body:
        raise ApiError(400, "attempt is required")
    who = _actor(body)
    result = best_run_module.select_best_run(
        control,
        experiment_id,
        task_id,
        _exact_int(body["attempt"], "attempt"),
        expected_selection_id=_text(
            body.get("expected_selection_id"), "expected_selection_id", required=False
        ),
        reason=_text(body.get("reason"), "reason", required=False),
        **who,
    )
    return 201, {"decision": result}


def _delete_best_run(
    workflow_path: str, experiment_id: str, task_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    control = _open_control(workflow_path, write=True)
    _require_member(workflow_path, control, experiment_id)
    who = _actor(body)
    result = best_run_module.clear_best_run(
        control,
        experiment_id,
        task_id,
        expected_selection_id=_text(
            body.get("expected_selection_id"), "expected_selection_id"
        ),
        reason=_text(body.get("reason"), "reason", required=False),
        **who,
    )
    return 200, {"decision": result}


def _post_undecided(
    workflow_path: str, experiment_id: str, task_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    control = _open_control(workflow_path, write=True)
    _require_member(workflow_path, control, experiment_id)
    who = _actor(body)
    candidate = body.get("candidate_attempt")
    result = best_run_module.leave_undecided(
        control,
        experiment_id,
        task_id,
        expected_selection_id=_text(
            body.get("expected_selection_id"), "expected_selection_id", required=False
        ),
        candidate_attempt=None if candidate is None else _exact_int(
            candidate, "candidate_attempt"
        ),
        reason=_text(body.get("reason"), "reason", required=False),
        **who,
    )
    return 201, {"decision": result}


def _post_review_pair(
    workflow_path: str, experiment_id: str, task_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    control = _open_control(workflow_path, write=True)
    _require_member(workflow_path, control, experiment_id)
    state = _text(body.get("state"), "state")
    if state not in (
        pair_review_module.STATE_REVIEWED,
        pair_review_module.STATE_NOT_REVIEWED,
    ):
        raise ApiError(
            400,
            "state must be "
            f"{pair_review_module.STATE_REVIEWED!r} or "
            f"{pair_review_module.STATE_NOT_REVIEWED!r}",
        )
    who = _reviewer(body)
    right_experiment = body.get("right_experiment")
    if right_experiment and right_experiment != experiment_id:
        _require_member(workflow_path, control, _text(right_experiment, "right_experiment"))
    # The same resolution the GET uses, so what was marked is exactly what was
    # shown: a pair is named by attempt numbers, never by a client-built ref.
    query: dict[str, list[str]] = {}
    for name, value in (
        ("left_attempt", body.get("left_attempt")),
        ("right_attempt", body.get("right_attempt")),
        ("right_experiment", right_experiment),
    ):
        if value is not None:
            query[name] = [str(value)]
    if "right_attempt" not in query:
        raise ApiError(400, "right_attempt is required")
    left, right = _comparison_sides(control, experiment_id, task_id, query)
    left_ref = _require_comparable(left, "left")
    right_ref = _require_comparable(right, "right")
    reader = _reader(control)
    left_pass, attribute = _pass_request(body, side="left")
    right_pass, _attribute = _pass_request(body, side="right")
    if left_pass is not None:
        left_ref, _selector, _omitted = _pass_scope(
            reader, left_ref, left_pass, attribute, which="left"
        )
    if right_pass is not None:
        right_ref, _selector, _omitted = _pass_scope(
            reader, right_ref, right_pass, attribute, which="right"
        )
    recorded = pair_review_module.PairReviewStore(control.store).set_state(
        left_ref,
        right_ref,
        state=state,
        note=_text(body.get("note"), "note", required=False),
        **who,
    )
    return 201, {
        "pair": recorded,
        "pair_key": comparison_module.review_pair_key(left_ref, right_ref),
        "left_run": _run_header(left),
        "right_run": _run_header(right),
        "left_pass": left_pass,
        "right_pass": right_pass,
    }


def _post_duplicate(
    workflow_path: str, source_experiment_id: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Run the same setup again. Registration only; nothing paid starts here."""
    runs_per_task = body.get("runs_per_task")
    record = benchmark_setup.duplicate_experiment(
        workflow_path,
        source_experiment_id,
        description=body.get("description"),
        runs_per_task=runs_per_task,
    )
    return 201, {
        "experiment": record,
        "source_experiment_id": source_experiment_id,
        "changed_fields": record.get("changed_fields", []),
    }


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def _segments(path: str, prefix: str) -> Optional[list[str]]:
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix):].strip("/")
    if not rest:
        return None
    return [unquote(part) for part in rest.split("/")]


def owns_write(path: str) -> bool:
    """Whether a POST/DELETE to this path belongs to this module.

    Used by the request handler's write allowlist, which refuses everything it
    does not name -- so a route added here without this returning True is
    simply refused rather than silently ungated.
    """
    parts = _segments(path, EXPERIMENTS_PREFIX)
    if parts is not None:
        rest = parts[1:]
        return (
            rest == ["winner", "decisions"]
            or (len(rest) >= 3 and rest[0] == "tasks" and rest[2] in
                ("best-run", "review-pairs"))
        )
    parts = _segments(path, REGISTRATIONS_PREFIX)
    return parts is not None and parts[-1:] == ["duplicate"]


def _guard(fn: Callable[[], tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
    """One translation from the domain vocabulary to HTTP, for every route.

    Each mapping is a distinct fact a client acts on differently: 404 means
    "not recorded", 409 means "the world moved or the evidence is unreadable",
    422 means "this attempt cannot be what you are asking it to be", and 400
    means the request itself was malformed.
    """
    try:
        return fn()
    except ApiError as exc:
        return exc.as_response()
    except selected_runs_module.SelectedRunsError as exc:
        # A refusal about the SELECTION itself: too many runs, none at
        # all, two sources, overlapping evidence. Each says which it was,
        # because a client retries a bound differently from a conflict.
        return exc.status, {
            "error": exc.message,
            "refused": exc.reason,
            **exc.payload,
        }
    except selection.StaleSelection as exc:
        return 409, {
            "error": str(exc),
            "stale": True,
            "expected_selection_id": exc.expected_selection_id,
            "current_selection_id": exc.current_selection_id,
            "current_experiment_id": exc.current_experiment_id,
            "scope_kind": exc.scope_kind,
            "scope_key": exc.scope_key,
        }
    except best_run_module.AttemptNotSelectable as exc:
        return 422, {
            "error": str(exc),
            "reason": exc.reason,
            "attempt": exc.attempt,
            "experiment_id": exc.experiment_id,
            "task_id": exc.task_id,
        }
    except best_run_module.NoBestRun as exc:
        return 409, {"error": str(exc)}
    except best_run_module.TaskBestUnavailable as exc:
        return 409, {"error": str(exc)}
    except selection.NoCurrentSelection as exc:
        return 409, {"error": str(exc)}
    except (selection.UnknownComparisonGroup, selection.ExperimentNotInGroup) as exc:
        return 404, {"error": str(exc.args[0] if exc.args else exc)}
    except (selection.SelectionControlError, control_module.ControlUnavailable) as exc:
        return 409, {"error": str(exc)}
    except workspace_module.UnknownLogicalExperiment as exc:
        return 404, {"error": f"unknown experiment: {exc.args[0] if exc.args else exc}"}
    except workspace_module.UnknownWorkspaceStore as exc:
        return 404, {"error": f"unknown store: {exc.args[0] if exc.args else exc}"}
    except workspace_module.WorkspaceBusyError as exc:
        return 503, {"error": str(exc)}
    except workspace_module.WorkspaceError as exc:
        return 409, {"error": str(exc)}
    except comparison_module.UnknownRecordedPass as exc:
        # The pass was resolved from spans and then failed to resolve against
        # some turn -- the evidence does not contain what was named.
        return 404, {"error": str(exc)}
    except comparison_module.ComparisonError as exc:
        return 400, {"error": str(exc)}
    except consistency_module.ConsistencyError as exc:
        # A consistency request that cannot be answered as asked -- two
        # different tasks, a metric asked of nothing. The unavailable-model
        # case is NOT here: that is a reported state of a 200 payload, because
        # the step-count metrics beside it are still real.
        return 400, {"error": str(exc)}
    except benchmark_setup.ExperimentDeleted as exc:
        return 404, {"error": str(exc)}
    except benchmark_setup.BenchmarkSetupConflict as exc:
        return 409, {"error": str(exc)}
    except sqlite3.OperationalError as exc:
        status = control_module.write_refusal_status(exc)
        if status is None:
            raise
        return status, {"error": str(exc)}
    except KeyError as exc:
        return 404, {"error": f"not found: {exc.args[0] if exc.args else exc}"}
    except (ValueError, TypeError) as exc:
        return 400, {"error": str(exc)}


def handle_get(
    workflow_path: str, path: str, query: dict[str, list[str]]
) -> Optional[tuple[int, dict[str, Any]]]:
    """Answer a GET, or None when the path belongs to somebody else."""
    parts = _segments(path, EXPERIMENTS_PREFIX)
    if parts is None or len(parts) < 2:
        return None
    return _guard(lambda: _get_experiments(workflow_path, parts, query))


def handle_post(
    workflow_path: str, path: str, body: Any
) -> Optional[tuple[int, dict[str, Any]]]:
    if not owns_write(path):
        return None
    if not isinstance(body, dict):
        return 400, {"error": "body must be a JSON object"}

    registration = _segments(path, REGISTRATIONS_PREFIX)
    if registration is not None:
        source = "/".join(registration[:-1])
        return _guard(lambda: _post_duplicate(workflow_path, source, body))

    parts = _segments(path, EXPERIMENTS_PREFIX) or []
    experiment_id, rest = parts[0], parts[1:]
    if rest == ["winner", "decisions"]:
        return _guard(lambda: _post_winner_decision(workflow_path, experiment_id, body))
    task_id = rest[1]
    if rest[2:] == ["best-run"]:
        return _guard(lambda: _post_best_run(workflow_path, experiment_id, task_id, body))
    if rest[2:] == ["best-run", "undecided"]:
        return _guard(lambda: _post_undecided(workflow_path, experiment_id, task_id, body))
    if rest[2:] == ["review-pairs"]:
        return _guard(
            lambda: _post_review_pair(workflow_path, experiment_id, task_id, body)
        )
    return 404, {"error": "not found"}


def handle_delete(
    workflow_path: str, path: str, body: Any
) -> Optional[tuple[int, dict[str, Any]]]:
    parts = _segments(path, EXPERIMENTS_PREFIX)
    if parts is None or len(parts) != 4 or parts[1] != "tasks" or parts[3] != "best-run":
        return None
    if not isinstance(body, dict):
        return 400, {"error": "body must be a JSON object"}
    experiment_id, task_id = parts[0], parts[2]
    return _guard(lambda: _delete_best_run(workflow_path, experiment_id, task_id, body))


__all__ = [
    "CLIENT_ACTOR_KINDS",
    "EXPERIMENTS_PREFIX",
    "REGISTRATIONS_PREFIX",
    "VIEWS",
    "VIEW_ANSWERS",
    "VIEW_DIFFERENCES",
    "VIEW_STEPS",
    "ApiError",
    "DEFAULT_PASS_ATTRIBUTE",
    "handle_delete",
    "handle_get",
    "handle_post",
    "handle_workspace_get",
    "owns_write",
]

# Names moved verbatim into selection_workspace. Re-exported lazily: that
# module imports the helpers above, so a module-level import here would cycle.
# This module does not call the moved names; callers still reach them here.
_SELECTION_WORKSPACE_REEXPORTS = (
    "_WorkspaceNames",
    "_ManifestScopedReader",
    "_workspace_attempts",
    "_with_turn_refs",
    "_workspace_consistency",
    "_workspace_selected_rows",
    "_workspace_archive",
    "_workspace_selected_runs",
    "_refuse_ambiguous_attempts",
    "_workspace_side",
    "_workspace_get",
    "handle_workspace_get",
)


def __getattr__(name: str):
    # PEP 562. The import stays inside this function because a top-level import
    # of selection_workspace would cycle: that module imports helpers from here.
    if name not in _SELECTION_WORKSPACE_REEXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(
        "fastworkflow.run_chatbot.selection_workspace"
    )
    for export in _SELECTION_WORKSPACE_REEXPORTS:
        globals()[export] = getattr(module, export)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SELECTION_WORKSPACE_REEXPORTS))
