"""Per-command training-data report and a minimum-row floor (spec R3b, bd fix-551.4).

What this exists to answer
--------------------------
"Which of my commands are starved?" — which, per finding F13, is the
highest-leverage question a workflow author can ask, because seed count is the
single largest measured input to routing accuracy. Before this module a training
run printed ``Generating utterances for context X, command Y ...`` with no count
and no verdict, tens of thousands of lines deep, and never validated that a
command ended up with a usable number of rows (F4).

Wave 1 (R3a, `fix-551.3`) fixed the *silent failure* underneath that: a
rate-limited command used to return ``[]`` and drop out of its context's
classifier entirely (F3); it now retries with backoff and degrades to
``[command_name] + seeds``. That makes the command weak instead of absent, and
records ``fell_back`` on its provenance. This module is the *visibility* half:
it turns those records into something a developer reads and acts on.

Where the numbers come from
---------------------------
Everything is derived from files already written under ``___command_info``. This
module computes nothing during training and never touches the training loop:

* ``training_provenance.json`` — one generation record per fully-qualified command,
  plus per-context labelled-row and explicit-skip records (written by
  `train.determinism.ProvenanceRecorder`).
* ``command_directory.json`` — ``core_command_names`` (which commands belong to
  the framework rather than to the developer). For legacy flat provenance only,
  ``input_for_param_extraction_class`` is the compatibility fallback for the explicit
  skip record that old training runs could not write.
* ``routing_definition.json`` — ``routing_definition_map``, i.e. which contexts
  each command is a label in.
* ``heldout_evaluation.json`` — optional. When R1's evaluation has run, its
  ``RoutingScore.per_command`` supplies the per-command held-out top-1 that R3b
  asks for. Absent, the column reads ``-``.

Reading from disk rather than from live objects is deliberate: the report is
available at the end of a training run and can be retained with its artifacts.

The two floors, and how much to trust them
------------------------------------------
See `DEFAULT_MIN_TRAINING_ROWS` and `DEFAULT_MIN_SEED_UTTERANCES`. One is derived
from a structural property of the training code; the other is an observation from
a single workflow. They are labelled differently in the output on purpose, and
only the structural floor can prevent publication.

`TrainingReport.has_blocking_problems` rejects structurally unusable data before
publication; fallback and seed-count findings remain advisory.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from fastworkflow.nlu_labels import (
    NON_ROUTABLE_LABELS,
    PARAMETER_VALUE_LABEL,
    WILDCARD_LABEL,
    label_of,
)
from fastworkflow.train.determinism import (
    COMMAND_INFO_FOLDERNAME,
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    ContextTrainingProvenance,
    ContextTrainingStatus,
    UtteranceProvenance,
)
from fastworkflow.train.selective_training import contexts_for_training
from fastworkflow.utils.logging import logger

REPORT_FILENAME: str = "training_report.txt"
REPORT_JSON_FILENAME: str = "training_report.json"
COMMAND_DIRECTORY_FILENAME: str = "command_directory.json"
ROUTING_DEFINITION_FILENAME: str = "routing_definition.json"

# Written by `train.heldout_evaluation.write_report`. Duplicated as a literal rather
# than imported because that module pulls in the scoring stack, and this one must stay
# importable (and cheap) from a CLI that only wants to print a table. The name is
# asserted to agree with heldout_evaluation.REPORT_FILENAME in the tests.
HELDOUT_REPORT_FILENAME: str = "heldout_evaluation.json"

#: **A default, not a measured constant.** The smallest per-command row count at which
#: the command is structurally guaranteed to participate in training.
#:
#: Derivation, from the training code rather than from any accuracy experiment:
#: `model_pipeline_training.train` uses a class-aware split that reserves at least
#: one row from every label for training and one for evaluation. A label therefore
#: needs two rows to participate in both sets.
#:
#: The consequence when it happens is not merely "this command trains badly". It is
#: already documented in-repo at `model_pipeline_training.py` (the comment above the
#: escalation-row block): a class absent from training cannot be learned, and the
#: ambiguous thresholds are computed from the evaluation split.
#:
#: What this floor is NOT: an accuracy threshold. Nothing in this evidence base
#: establishes a row count above which routing accuracy is acceptable. It is a floor
#: on *structural usability* only.
DEFAULT_MIN_TRAINING_ROWS: int = 2

#: **One workflow's observation, offered as a default.** Finding F13 records that on
#: a large reference workflow (160 commands, one hand-built 446-case benchmark) routing top-1
#: went 46.2% at 3.2 seeds/command, 70.4% at 8.0, and 73.8% at 9.3 — i.e. the return
#: on additional seeds flattens somewhere around eight.
#:
#: That curve comes from a single workflow, a single benchmark and a single domain,
#: and the spec says so explicitly. It is not a constant, it has no confidence
#: interval, and a different workflow may flatten somewhere else entirely. It is used
#: here only to decide when to *mention* that a command is thinly seeded — an
#: advisory that never contributes to `TrainingReport.has_blocking_problems`.
DEFAULT_MIN_SEED_UTTERANCES: int = 8

# How many healthy commands to name before collapsing the rest into a count. A report
# that enumerates 160 fine commands buries the two that are not.
HEALTHY_SAMPLE_LIMIT: int = 5

_TABLE_WIDTH: int = 78


class CommandKind(str, Enum):
    """Who owns a label, which decides who can act on a problem with it."""

    #: A command the workflow author wrote. Thin coverage here is theirs to fix.
    APPLICATION = "application"
    #: A framework command injected into every workflow (`core_command_names`):
    #: `IntentDetection/*`, `ErrorCorrection/*`. A developer cannot add seeds to
    #: these, so they are reported separately rather than mixed into the actionable
    #: list — but they are still reported, because a framework command that fell back
    #: degrades the same workflow.
    FRAMEWORK = "framework"
    #: A reserved NLU label (`wildcard`, `parameter_value`). **Not a command.**
    #: Reported in its own line so no reader can mistake `parameter_value` for an
    #: application command with thin coverage.
    RESERVED = "reserved"


class RowStatus(str, Enum):
    """Per-command verdict, most severe first in `_STATUS_SEVERITY`."""

    #: Generation degraded (R3a fallback). The command trained on its command name
    #: plus hand-written seeds plus whatever arrived before the failure.
    FELL_BACK = "fell_back"
    #: Fewer rows than the floor. May or may not have fallen back.
    BELOW_FLOOR = "below_floor"
    #: The command has `Signature.Input`, but its generator returned no rows in at
    #: least one context. This is an explicit trainer skip, not missing provenance.
    NO_UTTERANCES = "no_utterances"
    #: The command has no provenance record AND no context provenance showing it was
    #: trained: either it was never reached (a training run that died part-way) or
    #: provenance was lost. Distinct from EXCLUDED, which is by design, and from a
    #: command that supplies hand-written utterances without calling the generator -
    #: that one has nothing to record and is judged on its row count.
    MISSING = "missing"
    #: No `Signature.Input`, so `model_pipeline_training._requires_utterances` returns
    #: False and the command is deliberately not an intent-detection label at all. It
    #: is dispatched via `perform_action`. Not a defect — but worth stating, because
    #: "my command never routes" has this as a legitimate cause.
    EXCLUDED = "excluded"
    #: Above the floor, but fewer hand-written seeds than F13 suggests are useful.
    THIN_SEEDS = "thin_seeds"
    #: A reserved NLU label whose rows are fixed literals rather than generated text,
    #: so it has no provenance and no floor can apply. `wildcard`'s command file
    #: returns `PARAMETER_VALUE_PLACEHOLDERS` directly and never calls the generator,
    #: which is why an absent provenance record for it is normal rather than a gap.
    NOT_APPLICABLE = "not_applicable"
    OK = "ok"


_STATUS_SEVERITY: dict[RowStatus, int] = {
    RowStatus.FELL_BACK: 0,
    RowStatus.BELOW_FLOOR: 1,
    RowStatus.NO_UTTERANCES: 2,
    RowStatus.MISSING: 3,
    RowStatus.THIN_SEEDS: 4,
    RowStatus.EXCLUDED: 5,
    RowStatus.NOT_APPLICABLE: 6,
    RowStatus.OK: 7,
}

#: Statuses that make a training dataset structurally unusable. EXCLUDED and
#: THIN_SEEDS are deliberately absent: the first is a design choice, the second rests
#: on one workflow's curve and is too weak a basis for rejecting a run. FELL_BACK is
#: reported loudly but does not fail when enough rows remain.
BLOCKING_STATUSES: frozenset[RowStatus] = frozenset(
    {
        RowStatus.BELOW_FLOOR,
        RowStatus.NO_UTTERANCES,
        RowStatus.MISSING,
    }
)


class CommandRow(BaseModel):
    """One line of the report: what a command's training data actually looks like."""

    command_name: str
    kind: CommandKind
    status: RowStatus
    #: Contexts this command is a label in, from `routing_definition_map`. Empty when
    #: the routing definition could not be read.
    contexts: list[str] = Field(default_factory=list)
    #: Hand-written `plain_utterances` counted by the generator. None when unknown.
    seed_count: Optional[int] = None
    #: Utterances the LLM produced (excludes seeds and the command-name token).
    generated_count: Optional[int] = None
    #: Total labelled rows across every trained context. A shared generated set used
    #: in two contexts contributes twice here; see `rows_by_context` for the split.
    #: Legacy schema-v1 provenance has no context dimension, so its value remains the
    #: single generation's `final_count` and is accompanied by a compatibility problem.
    row_count: Optional[int] = None
    #: Actual labelled rows by trained context. Generation remains one record per
    #: fully-qualified command; repeated use through inheritance lives here.
    rows_by_context: dict[str, int] = Field(default_factory=dict)
    #: Contexts in which the trainer explicitly excluded this command, with the
    #: persisted reason (`no Signature.Input` or `generator returned no rows`).
    skipped_contexts: dict[str, str] = Field(default_factory=dict)
    fell_back: bool = False
    fallback_reason: Optional[str] = None
    #: Held-out cases for this command, from R1's evaluation. None when R1 has not run.
    heldout_total: Optional[int] = None
    heldout_top1_correct: Optional[int] = None

    @property
    def heldout_top1(self) -> Optional[float]:
        """Held-out top-1 accuracy, or None when there are no held-out cases."""
        if not self.heldout_total or self.heldout_top1_correct is None:
            return None
        return self.heldout_top1_correct / self.heldout_total

    @property
    def is_blocking(self) -> bool:
        return self.status in BLOCKING_STATUSES


class EscalationBudget(BaseModel):
    """Every number that decided one context's reserved escalation-class row count.

    "Why does this context's `wildcard` class have 87 rows?" is the question AR6 spent a
    spec section on, because the reference workflow's numbers could not be reconstructed
    after the fact. `model_pipeline_training._record_wildcard_context_training` writes
    these into provenance; this is the row that puts them in front of a person.
    """

    context_name: str
    #: False when the context has no non-local ancestor utterances, so no escalation class
    #: was emitted at all. Its `selected_rows` is 0 and every denominator is moot.
    included: bool
    #: Rows the escalation class actually trained on.
    selected_rows: int
    #: This context's own labelled rows, counted BEFORE escalation rows were appended.
    #: The budget is a multiplier on this, which is what bounds training cost.
    own_rows: Optional[int] = None
    #: Ancestor rows available before and after de-duplication against this context's own
    #: utterances. The gap between them is how much of the ancestor corpus means
    #: something here and therefore cannot train "ask my parent".
    raw_candidate_rows: Optional[int] = None
    deduplicated_candidate_rows: Optional[int] = None
    #: Rows the selection had to include regardless of budget (the base `wildcard`
    #: command's own utterances).
    always_include_rows: Optional[int] = None
    #: The bound `class_balance.reserved_class_budget` computed.
    selected_budget: Optional[int] = None
    #: Distinct ancestor (context, command) sources, EXCLUDING the core commands. Each
    #: must keep at least one row or it cannot be escalated to at all, so
    #: `class_balance.reserved_class_budget` takes the max of this and the cost ratio:
    #: coverage is a requirement, cost a preference. Core commands are excluded because
    #: they are a label in every context and so are never escalation targets -- which is
    #: also what stops this number moving when an unrelated context is renamed (bd
    #: fix-4ej); see `class_balance`'s module docstring for the derivation.
    coverage_floor: Optional[int] = None
    coverage_floor_applied: Optional[bool] = None
    reason: Optional[str] = None


class TrainingReport(BaseModel):
    """The whole report: rows, the floors they were judged against, and what broke."""

    workflow_folderpath: str
    generated_at: str
    min_rows: int
    min_seeds: int
    rows: list[CommandRow] = Field(default_factory=list)
    #: Per-context escalation-class budget decisions, newest run only. Empty for a
    #: workflow trained before these were recorded, and for one with no ancestor contexts.
    escalation_budgets: list[EscalationBudget] = Field(default_factory=list)
    #: Human-readable descriptions of anything that stopped the report being complete —
    #: a missing provenance file, unreadable JSON, a record that failed validation.
    #: Never an exception: a broken report must not mask the training result the
    #: developer spent hours waiting for.
    problems: list[str] = Field(default_factory=list)
    provenance_path: Optional[str] = None
    heldout_available: bool = False

    def of_kind(self, kind: CommandKind) -> list[CommandRow]:
        return [row for row in self.rows if row.kind == kind]

    def with_status(self, *statuses: RowStatus) -> list[CommandRow]:
        wanted = set(statuses)
        return [row for row in self.rows if row.status in wanted]

    @property
    def blocking_rows(self) -> list[CommandRow]:
        """Rows a developer should act on, worst first."""
        return [row for row in self.rows if row.is_blocking]

    @property
    def never_routed_rows(self) -> list[CommandRow]:
        """Rows the held-out evaluation scored and never once got right.

        Orthogonal to `RowStatus`, which describes training *data*. A command can have
        a perfectly healthy row count and still route nothing — the first hello_world
        run this report was developed against had exactly that shape
        (`IntentDetection/what_can_i_do`: 38 rows, 0 of 5 held-out cases correct), and
        summarising it as "healthy" hid the most actionable fact in the run.

        The criterion is "never correct", not "below some accuracy" — zero is the one
        boundary here that needs no invented constant. Advisory only: R1 labels its own
        persona holdout an in-generator holdout rather than a generalisation test, so a
        low score is a lead to follow, not a verdict.
        """
        return [
            row
            for row in self.rows
            if row.heldout_total and row.heldout_top1_correct == 0
        ]

    @property
    def has_blocking_problems(self) -> bool:
        """True when a command cannot participate in a valid training split.

        A transient LLM fallback remains visible but is not itself blocking when the
        command still has enough rows. Seed-count guidance is advisory because its
        evidence comes from one workflow.
        """
        return any(row.is_blocking for row in self.rows)

    def to_dict(self) -> dict:
        """JSON-serialisable form, for CI and for `--json` on a CLI subcommand."""
        payload = self.model_dump(mode="json")
        payload["summary"] = {
            "total_commands": len(self.rows),
            "blocking": len(self.blocking_rows),
            "fell_back": len(self.with_status(RowStatus.FELL_BACK)),
            "below_floor": len(self.with_status(RowStatus.BELOW_FLOOR)),
            "no_utterances": len(self.with_status(RowStatus.NO_UTTERANCES)),
            "missing_provenance": len(self.with_status(RowStatus.MISSING)),
            "thin_seeds": len(self.with_status(RowStatus.THIN_SEEDS)),
            "excluded_from_training": len(self.with_status(RowStatus.EXCLUDED)),
            "ok": len(self.with_status(RowStatus.OK)),
            "has_blocking_problems": self.has_blocking_problems,
            "escalation_contexts": len(self.escalation_budgets),
        }
        return payload


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def get_min_training_rows() -> int:
    """Return the structural row floor used by the class-aware split."""
    return DEFAULT_MIN_TRAINING_ROWS


def get_min_seed_utterances() -> int:
    """Return the fixed, advisory seed-count threshold."""
    return DEFAULT_MIN_SEED_UTTERANCES


# ---------------------------------------------------------------------
# Loading (every reader below degrades to a `problems` entry, never a raise)
# ---------------------------------------------------------------------

def command_info_dir(workflow_folderpath: str) -> Path:
    return Path(workflow_folderpath) / COMMAND_INFO_FOLDERNAME


def provenance_path(workflow_folderpath: str) -> Path:
    return command_info_dir(workflow_folderpath) / PROVENANCE_FILENAME


def _read_json(path: Path, problems: list[str], what: str) -> Optional[dict]:
    """Read a JSON object from *path*, appending a description of any failure.

    Returns None when the file is absent, unreadable, malformed, or is not an object.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        problems.append(f"{what} at {path} could not be read ({type(exc).__name__}: {exc})")
        return None
    if not isinstance(data, dict):
        problems.append(f"{what} at {path} is not a JSON object; ignoring it")
        return None
    return data


def load_provenance(
    workflow_folderpath: str, problems: list[str]
) -> dict[str, UtteranceProvenance]:
    """Load per-command provenance, tolerating a partial or corrupt file.

    `ProvenanceRecorder.load` is the happy path and raises on malformed content; a
    training run that crashed mid-write is exactly when this report is most wanted,
    so records are validated one at a time and the bad ones are described rather than
    thrown.
    """
    path = provenance_path(workflow_folderpath)
    if not path.is_file():
        problems.append(
            f"No training provenance at {path}. Either this workflow has never been "
            f"trained, or it was trained before per-command provenance existed. "
            f"Run `fastworkflow train` to produce it."
        )
        return {}

    payload = _read_json(path, problems, "Training provenance")
    if payload is None:
        return {}

    commands = payload.get("commands")
    if isinstance(commands, dict):
        schema_version = payload.get("schema_version")
        if schema_version != PROVENANCE_SCHEMA_VERSION:
            problems.append(
                f"Training provenance at {path} declares schema_version "
                f"{schema_version!r}; reading its command records with the "
                f"schema-{PROVENANCE_SCHEMA_VERSION} compatibility reader"
            )
        command_payload = commands
    else:
        # Schema v1 was a flat command map. It remains readable, but it cannot say how
        # many contexts reused a generated set or which commands the trainer skipped.
        command_payload = payload
        problems.append(
            f"Training provenance at {path} uses the legacy flat schema. Generation "
            f"records remain readable, but per-context row and skip counts are "
            f"unavailable until the next successful training run."
        )

    records: dict[str, UtteranceProvenance] = {}
    for name, raw in command_payload.items():
        if not isinstance(raw, dict):
            problems.append(f"Provenance entry for {name!r} is not an object; skipped")
            continue
        try:
            records[str(name)] = UtteranceProvenance(**raw)
        except (ValidationError, TypeError) as exc:
            problems.append(
                f"Provenance entry for {name!r} is unusable and was skipped "
                f"({type(exc).__name__})"
            )
    if not records:
        problems.append(f"Training provenance at {path} contained no usable records")
    return records


def load_context_training(
    workflow_folderpath: str, problems: list[str]
) -> dict[tuple[str, str], ContextTrainingProvenance]:
    """Load schema-v2 `(context, command)` records one at a time."""
    path = provenance_path(workflow_folderpath)
    payload = _read_json(path, problems, "Training provenance")
    if payload is None:
        return {}

    context_payload = payload.get("context_training")
    if context_payload is None:
        return {}
    if not isinstance(context_payload, dict):
        problems.append(
            f"Context training provenance at {path} is not an object; ignoring it"
        )
        return {}

    records: dict[tuple[str, str], ContextTrainingProvenance] = {}
    for context_name, command_map in context_payload.items():
        if not isinstance(command_map, dict):
            problems.append(
                f"Context provenance entry for {context_name!r} is not an object; "
                f"skipped"
            )
            continue
        for command_name, raw in command_map.items():
            if not isinstance(raw, dict):
                problems.append(
                    f"Context provenance entry for {context_name!r}/"
                    f"{command_name!r} is not an object; skipped"
                )
                continue
            try:
                record = ContextTrainingProvenance(**raw)
            except (ValidationError, TypeError) as exc:
                problems.append(
                    f"Context provenance entry for {context_name!r}/"
                    f"{command_name!r} is unusable and was skipped "
                    f"({type(exc).__name__})"
                )
                continue
            records[(record.context_name, record.command_name)] = record
    return records


def _load_command_directory(
    workflow_folderpath: str, problems: list[str]
) -> tuple[set[str], dict[str, bool]]:
    """Return (core command names, command -> participates in intent training).

    The second map mirrors `model_pipeline_training._requires_utterances`, which keys
    solely off `input_for_param_extraction_class`. A command without one is dispatched
    through `perform_action` and is deliberately not a classifier label.
    """
    data = _read_json(
        command_info_dir(workflow_folderpath) / COMMAND_DIRECTORY_FILENAME,
        problems,
        "Command directory",
    )
    if data is None:
        return set(), {}

    core = {str(name) for name in data.get("core_command_names") or []}
    metadata = data.get("map_command_2_metadata")
    requires: dict[str, bool] = {}
    if isinstance(metadata, dict):
        for name, entry in metadata.items():
            if isinstance(entry, dict):
                requires[str(name)] = bool(entry.get("input_for_param_extraction_class"))
    return core, requires


def _load_command_contexts(
    workflow_folderpath: str, problems: list[str]
) -> dict[str, list[str]]:
    """Return command -> the contexts it is a classifier label in."""
    data = _read_json(
        command_info_dir(workflow_folderpath) / ROUTING_DEFINITION_FILENAME,
        problems,
        "Routing definition",
    )
    if data is None:
        return {}
    mapping = data.get("routing_definition_map")
    if not isinstance(mapping, dict):
        return {}
    return {
        str(command): sorted(str(ctx) for ctx in contexts)
        for command, contexts in mapping.items()
        if isinstance(contexts, (list, tuple))
    }


def _load_heldout_per_command(
    workflow_folderpath: str, problems: list[str]
) -> dict[str, dict]:
    """Return command -> R1's per-command held-out counts, merged across contexts.

    A command is a label in several contexts and R1 scores each context separately,
    so the counts are summed. Absent entirely until R1's evaluation has run against
    this workflow, which is why every caller treats an empty result as normal rather
    than as a problem.
    """
    path = command_info_dir(workflow_folderpath) / HELDOUT_REPORT_FILENAME
    if not path.is_file():
        return {}
    data = _read_json(path, problems, "Held-out evaluation report")
    if data is None:
        return {}

    # `write_report` serialises the per-context `HeldoutReport` list under "contexts".
    # "reports" is accepted too so a rename on that side degrades to a missing column
    # rather than to a wrong one.
    entries = data.get("contexts") or data.get("reports") or []
    merged: dict[str, dict] = {}
    for report in entries:
        if not isinstance(report, dict):
            continue
        routing = report.get("routing")
        if not isinstance(routing, dict):
            continue
        per_command = routing.get("per_command")
        if not isinstance(per_command, dict):
            continue
        for command, counts in per_command.items():
            if not isinstance(counts, dict):
                continue
            slot = merged.setdefault(str(command), {"total": 0, "top1_correct": 0})
            slot["total"] += int(counts.get("total") or 0)
            slot["top1_correct"] += int(counts.get("top1_correct") or 0)
    return merged


# ---------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------

def _build_escalation_budgets(
    context_training: dict[tuple[str, str], ContextTrainingProvenance],
) -> list[EscalationBudget]:
    """Pull the escalation-class budget records out of the context-training provenance.

    Keyed on the reserved label rather than on a separate provenance block, because that
    is where the trainer already records them — the fields exist on every
    `ContextTrainingProvenance` and only the reserved labels populate them.
    """
    budgets: list[EscalationBudget] = []
    for (context_name, command_name), record in sorted(context_training.items()):
        if command_name != WILDCARD_LABEL:
            continue
        budgets.append(
            EscalationBudget(
                context_name=context_name,
                included=record.status
                in {
                    ContextTrainingStatus.INCLUDED,
                    ContextTrainingStatus.INCLUDED_FALLBACK,
                },
                selected_rows=record.row_count,
                own_rows=record.own_row_count,
                raw_candidate_rows=record.raw_candidate_count,
                deduplicated_candidate_rows=record.deduplicated_candidate_count,
                always_include_rows=record.always_include_count,
                selected_budget=record.selected_budget,
                coverage_floor=record.coverage_floor,
                coverage_floor_applied=record.coverage_floor_applied,
                reason=record.reason,
            )
        )
    return budgets


def _classify_kind(command_name: str, core_commands: set[str]) -> CommandKind:
    """Decide who owns *command_name*.

    Reserved beats framework: `wildcard` appears in `core_command_names` because it
    has a command file, but it is an NLU label rather than something a user can
    invoke, and reporting it as a framework command would put a non-command in a list
    of commands.
    """
    if label_of(command_name) in NON_ROUTABLE_LABELS:
        return CommandKind.RESERVED
    if command_name in core_commands:
        return CommandKind.FRAMEWORK
    return CommandKind.APPLICATION


def _classify_status(
    kind: CommandKind,
    row_count: Optional[int],
    rows_by_context: dict[str, int],
    seed_count: Optional[int],
    fell_back: bool,
    trains_as_label: Optional[bool],
    has_provenance: bool,
    context_records: list[ContextTrainingProvenance],
    min_rows: int,
    min_seeds: int,
) -> RowStatus:
    """Turn counts into a single verdict. Order is severity order.

    A reserved label is exempt from every floor *except* the fallback flag. Its rows
    are fixed literals or ancestor utterances rather than anything generated for it,
    so "no provenance" and "few seeds" are its normal state and flagging them would
    put a non-command at the top of a list of broken commands. A reserved label that
    genuinely fell back is still surfaced, because a degraded escalation class
    degrades the workflow like anything else.

    Absent generation provenance is only a defect when nothing else shows the command
    trained; see the comment on the `has_provenance` branch below.
    """
    if kind is CommandKind.RESERVED and not fell_back:
        return RowStatus.NOT_APPLICABLE
    context_statuses = {record.status for record in context_records}
    if ContextTrainingStatus.SKIPPED_NO_UTTERANCES in context_statuses:
        return RowStatus.NO_UTTERANCES
    included_statuses = {
        ContextTrainingStatus.INCLUDED,
        ContextTrainingStatus.INCLUDED_FALLBACK,
    }
    has_included_context = bool(context_statuses & included_statuses)
    if (
        ContextTrainingStatus.SKIPPED_NO_INPUT in context_statuses
        and not has_included_context
    ):
        return RowStatus.EXCLUDED
    if not has_provenance:
        if trains_as_label is False:
            return RowStatus.EXCLUDED
        if not has_included_context:
            # No generation record AND no sign the trainer ever reached this command:
            # the run died part-way or provenance was lost. That is what MISSING means.
            return RowStatus.MISSING
        # An `UtteranceProvenance` record describes *generation*, and only
        # `generate_diverse_utterances` writes one. A command whose `generate_utterances`
        # returns hand-written rows directly never calls the generator, so it has nothing
        # to record - exactly the reasoning `NOT_APPLICABLE` already applies to reserved
        # labels. The trainer's own context provenance proves this command trained, and
        # carries the row counts, so judge it on the floor like anything else instead of
        # blocking publication over a record that was never owed.
    if rows_by_context and any(
        context_rows < min_rows for context_rows in rows_by_context.values()
    ):
        return RowStatus.BELOW_FLOOR
    if not rows_by_context and row_count is not None and row_count < min_rows:
        return RowStatus.BELOW_FLOOR
    if fell_back:
        return RowStatus.FELL_BACK
    if seed_count is not None and seed_count < min_seeds:
        return RowStatus.THIN_SEEDS
    return RowStatus.OK


def build_report(
    workflow_folderpath: str,
    min_rows: Optional[int] = None,
    min_seeds: Optional[int] = None,
) -> TrainingReport:
    """Assemble the report for *workflow_folderpath*. Never raises on bad input.

    Every missing or malformed input becomes a `problems` entry and the rest of the
    report is produced from whatever remains, because the situations in which this is
    most valuable — a run that died half-way, a version restored from a backup — are
    exactly the situations in which the inputs are incomplete.
    """
    resolved_min_rows = get_min_training_rows() if min_rows is None else int(min_rows)
    resolved_min_seeds = get_min_seed_utterances() if min_seeds is None else int(min_seeds)

    problems: list[str] = []
    records = load_provenance(workflow_folderpath, problems)
    context_training = load_context_training(workflow_folderpath, problems)
    core_commands, requires_utterances = _load_command_directory(
        workflow_folderpath, problems
    )
    command_contexts = _load_command_contexts(workflow_folderpath, problems)
    heldout = _load_heldout_per_command(workflow_folderpath, problems)
    declared_contexts = {
        context_name
        for contexts in command_contexts.values()
        for context_name in contexts
    }
    try:
        trainable_contexts = set(contexts_for_training(workflow_folderpath))
    except Exception as exc:  # noqa: BLE001
        # Failing closed matters here: if the report cannot prove that a context is
        # intentionally excluded, its commands must remain subject to the publication
        # gate rather than being silently exempted.
        trainable_contexts = declared_contexts
        problems.append(
            "Could not determine the contexts produced by training; all declared "
            f"contexts remain subject to the publication gate: {exc}"
        )

    # Union so that a command present in only one source still gets a line: a command
    # that generated utterances but vanished from routing is as interesting as the
    # reverse, and silently reporting the intersection would hide both.
    context_commands = {command for _context, command in context_training}
    all_commands = (
        set(records) | context_commands | set(command_contexts) | set(requires_utterances)
    )

    rows: list[CommandRow] = []
    for command_name in sorted(all_commands):
        record = records.get(command_name)
        trains_as_label = requires_utterances.get(command_name)
        declared_command_contexts = command_contexts.get(command_name, [])
        if (
            trains_as_label is not False
            and declared_command_contexts
            and not set(declared_command_contexts) & trainable_contexts
        ):
            # Some workflows declare command-only contexts that deliberately have no
            # classifier. Their commands are real commands, not missing training data.
            trains_as_label = False
        heldout_counts = heldout.get(command_name) or {}
        kind = _classify_kind(command_name, core_commands)
        command_context_records = sorted(
            (
                context_record
                for (_context, recorded_command), context_record
                in context_training.items()
                if recorded_command == command_name
            ),
            key=lambda context_record: context_record.context_name,
        )
        rows_by_context = {
            context_record.context_name: context_record.row_count
            for context_record in command_context_records
            if context_record.status
            in {
                ContextTrainingStatus.INCLUDED,
                ContextTrainingStatus.INCLUDED_FALLBACK,
            }
        }
        skipped_contexts = {
            context_record.context_name: (
                context_record.reason or context_record.status.value
            )
            for context_record in command_context_records
            if context_record.status
            in {
                ContextTrainingStatus.SKIPPED_NO_INPUT,
                ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
            }
        }
        row_count = (
            sum(rows_by_context.values())
            if command_context_records
            else (record.final_count if record else None)
        )
        actual_contexts = (
            sorted({item.context_name for item in command_context_records})
            if command_context_records
            else command_contexts.get(command_name, [])
        )
        fell_back = bool(
            (record and record.fell_back)
            or any(
                item.status is ContextTrainingStatus.INCLUDED_FALLBACK
                for item in command_context_records
            )
        )
        status = _classify_status(
            kind=kind,
            row_count=row_count,
            rows_by_context=rows_by_context,
            seed_count=record.seed_utterance_count if record else None,
            fell_back=fell_back,
            trains_as_label=trains_as_label,
            has_provenance=record is not None,
            context_records=command_context_records,
            min_rows=resolved_min_rows,
            min_seeds=resolved_min_seeds,
        )
        rows.append(
            CommandRow(
                command_name=command_name,
                kind=kind,
                status=status,
                contexts=actual_contexts,
                seed_count=record.seed_utterance_count if record else None,
                generated_count=record.generated_count if record else None,
                row_count=row_count,
                rows_by_context=rows_by_context,
                skipped_contexts=skipped_contexts,
                fell_back=fell_back,
                fallback_reason=(
                    record.fallback_reason
                    if record and record.fallback_reason
                    else next(
                        (
                            item.reason
                            for item in command_context_records
                            if item.status
                            is ContextTrainingStatus.INCLUDED_FALLBACK
                            and item.reason
                        ),
                        None,
                    )
                ),
                heldout_total=heldout_counts.get("total"),
                heldout_top1_correct=heldout_counts.get("top1_correct"),
            )
        )

    rows.sort(key=lambda row: (_STATUS_SEVERITY[row.status], row.row_count or 0, row.command_name))

    path = provenance_path(workflow_folderpath)
    return TrainingReport(
        workflow_folderpath=str(workflow_folderpath),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        min_rows=resolved_min_rows,
        min_seeds=resolved_min_seeds,
        rows=rows,
        escalation_budgets=_build_escalation_budgets(context_training),
        problems=problems,
        provenance_path=str(path) if path.is_file() else None,
        heldout_available=bool(heldout),
    )


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def _fmt_count(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


def _fmt_heldout(row: CommandRow) -> str:
    top1 = row.heldout_top1
    if top1 is None:
        return "-"
    return f"{top1:.0%} ({row.heldout_top1_correct}/{row.heldout_total})"


def _name_column_width(rows: Iterable[CommandRow]) -> int:
    """One command-name column width for the whole report.

    Computed across every row rather than per group: three tables in one report that
    each pick their own width read as three unrelated tables.
    """
    longest = max((len(row.command_name) for row in rows), default=0)
    return max(28, min(46, longest + 1))


def _detail_lines(
    rows: Iterable[CommandRow], width: int, indent: str = "    "
) -> list[str]:
    """Render a group of rows as an aligned table.

    Column style was shared with `artifact_versioning.format_versions_table` so the
    two outputs a training run prints looked like one program's work. That function
    was deleted as test-only code (bd fix-k0i.50), so this is now the only place the
    style lives; anything rendering a version listing again should follow it.
    """
    rows = list(rows)
    if not rows:
        return []
    header = (
        f"{indent}{'COMMAND':{width}s} {'ROWS':>5s} {'SEEDS':>6s} {'GEN':>5s} "
        f"{'HELD-OUT':>14s}  CONTEXTS"
    )
    lines = [header]
    for row in rows:
        if row.rows_by_context:
            contexts = ", ".join(
                f"{context}:{count}"
                for context, count in sorted(row.rows_by_context.items())
            )
        elif row.skipped_contexts:
            contexts = ", ".join(
                f"{context}:skipped"
                for context in sorted(row.skipped_contexts)
            )
        else:
            contexts = ", ".join(row.contexts) if row.contexts else "-"
        lines.append(
            f"{indent}{row.command_name:{width}s} {_fmt_count(row.row_count):>5s} "
            f"{_fmt_count(row.seed_count):>6s} {_fmt_count(row.generated_count):>5s} "
            f"{_fmt_heldout(row):>14s}  {contexts}"
        )
        if row.fallback_reason:
            lines.append(f"{indent}  reason: {row.fallback_reason}")
        for context, reason in sorted(row.skipped_contexts.items()):
            lines.append(f"{indent}  skipped in {context}: {reason}")
    return lines


def _escalation_budget_lines(budgets: list[EscalationBudget]) -> list[str]:
    """Render the escalation-class budget table.

    Present even when nothing is wrong, because it is not a defect list: it is the
    artifact that answers "why does this context's wildcard class have this many rows"
    without a retained console log, which is the AR6 requirement.
    """
    if not budgets:
        return []
    width = max(28, min(46, max(len(b.context_name) for b in budgets) + 1))
    lines = [
        "",
        f"  ESCALATION BUDGET ({len(budgets)}) — rows given to the reserved "
        f"`{WILDCARD_LABEL}` class, and",
        "  the denominators that chose them. Budget bounds ancestor rows at this "
        "context's own",
        "  row count; the floor raises it when coverage needs more, so every ancestor "
        "(context,",
        "  command) source keeps at least one row. FLOOR excludes core commands: they "
        "are a",
        "  label here too, so they are never escalated to. SELECTED is what actually "
        "trained.",
        f"    {'CONTEXT':{width}s} {'SELECTED':>8s} {'OWN':>6s} {'BUDGET':>7s} "
        f"{'FLOOR':>6s} {'CAND(raw/dedup)':>17s} {'ALWAYS':>7s}",
    ]
    for budget in budgets:
        candidates = (
            f"{_fmt_count(budget.raw_candidate_rows)}/"
            f"{_fmt_count(budget.deduplicated_candidate_rows)}"
        )
        lines.append(
            f"    {budget.context_name:{width}s} {budget.selected_rows:>8d} "
            f"{_fmt_count(budget.own_rows):>6s} "
            f"{_fmt_count(budget.selected_budget):>7s} "
            f"{_fmt_count(budget.coverage_floor):>6s} {candidates:>17s} "
            f"{_fmt_count(budget.always_include_rows):>7s}"
        )
        if not budget.included:
            lines.append(
                f"      no escalation class: {budget.reason or 'not emitted here'}"
            )
        elif budget.coverage_floor_applied:
            lines.append(
                f"      coverage floor bound the budget: {budget.coverage_floor} "
                f"ancestor source(s) needed a row each, above this context's "
                f"{_fmt_count(budget.own_rows)} own row(s)"
            )
    return lines


def _summarize_healthy(rows: list[CommandRow]) -> str:
    """One line describing the commands that need no attention.

    Deliberately a summary and not a list: a report that enumerates 160 fine commands
    buries the two that are not, which is the failure mode F4 describes.
    """
    counts = [row.row_count for row in rows if row.row_count is not None]
    if not counts:
        return f"{len(rows)} command(s), row counts unknown"
    spread = f"{min(counts)}-{max(counts)} rows, median {int(statistics.median(counts))}"
    names = ", ".join(row.command_name for row in rows[:HEALTHY_SAMPLE_LIMIT])
    if len(rows) > HEALTHY_SAMPLE_LIMIT:
        names += f", +{len(rows) - HEALTHY_SAMPLE_LIMIT} more"
    return f"{len(rows)} command(s), {spread}  [{names}]"


def format_report(report: TrainingReport) -> str:
    """Render the report for a terminal.

    Structure is severity-ordered on purpose. A developer who reads only the first
    ten lines must still learn whether anything is wrong; a developer who reads none
    of it still sees the banner, which uses the same `!` rule as R3a's fallback
    announcement so the two are recognisably the same alarm.
    """
    workflow_name = os.path.basename(str(report.workflow_folderpath).rstrip("/")) or str(
        report.workflow_folderpath
    )
    rule = "=" * _TABLE_WIDTH
    lines = [rule, f"TRAINING DATA REPORT — {workflow_name}", rule]

    if not report.rows:
        lines.append("")
        lines.append("  No per-command training data could be read.")
        lines.extend(_format_problems(report))
        lines.append(rule)
        return "\n".join(lines)

    fell_back = report.with_status(RowStatus.FELL_BACK)
    below_floor = report.with_status(RowStatus.BELOW_FLOOR)
    no_utterances = report.with_status(RowStatus.NO_UTTERANCES)
    missing = report.with_status(RowStatus.MISSING)
    thin = report.with_status(RowStatus.THIN_SEEDS)
    excluded = report.with_status(RowStatus.EXCLUDED)
    healthy = report.with_status(RowStatus.OK)
    width = _name_column_width(report.rows)

    if report.has_blocking_problems:
        banner = "!" * _TABLE_WIDTH
        lines.extend([
            banner,
            f"!! {len(report.blocking_rows)} command(s) cannot participate in a valid "
            f"training split.",
            "!! Models will not be published until these structural problems are fixed.",
            banner,
        ])

    if fell_back:
        lines.append("")
        lines.append(
            f"  FELL BACK ({len(fell_back)}) — synthetic generation failed; these "
            f"trained on the"
        )
        lines.append(
            "  command name plus hand-written seeds plus whatever arrived first."
        )
        lines.extend(_detail_lines(fell_back, width))

    if below_floor:
        lines.append("")
        lines.append(
            f"  BELOW ROW FLOOR ({len(below_floor)}) — fewer than {report.min_rows} rows."
        )
        lines.append(
            "  The class-aware split needs at least one row for training and one for "
            "evaluation."
        )
        lines.append(
            "  A label below that floor cannot be learned and evaluated in the same run."
        )
        lines.extend(_detail_lines(below_floor, width))

    if missing:
        lines.append("")
        lines.append(
            f"  NO PROVENANCE ({len(missing)}) — these are classifier labels but no "
            f"utterance"
        )
        lines.append(
            "  generation was recorded. Expect a training run that did not complete."
        )
        lines.extend(_detail_lines(missing, width))

    if no_utterances:
        lines.append("")
        lines.append(
            f"  NO UTTERANCES ({len(no_utterances)}) — these commands define "
            f"Signature.Input,"
        )
        lines.append(
            "  but their utterance generators returned no rows in at least one context."
        )
        lines.extend(_detail_lines(no_utterances, width))

    if thin:
        lines.append("")
        lines.append(
            f"  THIN SEEDS ({len(thin)}, advisory) — fewer than {report.min_seeds} "
            f"hand-written seed"
        )
        lines.append(
            "  utterances. Seed count is the largest input to routing accuracy that "
            "has been"
        )
        lines.append(
            "  measured here: on ONE workflow, 3.2 seeds/command routed 46.2% of "
            "held-out"
        )
        lines.append(
            "  phrasings correctly, 8.0 routed 70.4%, and 9.3 routed 73.8% — returns "
            "flatten"
        )
        lines.append(
            f"  around eight. That is one workflow's curve, not a constant; "
            f"{report.min_seeds} is a default."
        )
        lines.extend(_detail_lines(thin, width))

    if never_routed := report.never_routed_rows:
        lines.append("")
        lines.append(
            f"  NEVER ROUTED ({len(never_routed)}, advisory) — the held-out evaluation "
            f"scored these"
        )
        lines.append(
            "  and got none of them right. Row count is not accuracy: a command can be "
            "well"
        )
        lines.append(
            "  supplied and still lose every case to a near-duplicate command or to a "
            "reserved"
        )
        lines.append(
            "  label. Treat as a lead, not a verdict — R1's holdout is an in-generator "
            "holdout."
        )
        lines.extend(_detail_lines(never_routed, width))

    lines.extend(_escalation_budget_lines(report.escalation_budgets))

    lines.append("")
    lines.append("  SUMMARY")
    if healthy:
        lines.append(f"    healthy          : {_summarize_healthy(healthy)}")
    else:
        lines.append("    healthy          : none")

    # Framework commands are counted separately because the developer's action on one
    # is different: they cannot add seed utterances to `IntentDetection/go_up`. They
    # are still counted rather than hidden — a framework command that fell back
    # degrades this workflow exactly as much as one of the developer's own.
    framework = report.of_kind(CommandKind.FRAMEWORK)
    framework_flagged = [
        row
        for row in framework
        if row.status not in (RowStatus.OK, RowStatus.NOT_APPLICABLE, RowStatus.EXCLUDED)
    ]
    if framework:
        state = (
            f"{len(framework_flagged)} flagged above"
            if framework_flagged
            else "none flagged"
        )
        lines.append(
            f"    framework        : {len(framework)} command(s), {state} — owned by "
            f"fastWorkflow,"
        )
        lines.append(
            "                       not by this workflow; seeds are not yours to add"
        )

    # Reserved labels get their own line, always, even when none of them produced a
    # provenance record. `parameter_value` never does — it is seven fixed literals, not
    # generated text — and a reader who does not see it named here has no way to tell
    # that it was considered rather than lost.
    reserved = report.of_kind(CommandKind.RESERVED)
    named = sorted(row.command_name for row in reserved)
    if PARAMETER_VALUE_LABEL not in named:
        named.append(f"{PARAMETER_VALUE_LABEL} (fixed literals, no provenance)")
    lines.append(
        f"    reserved labels  : {', '.join(named)}"
    )
    lines.append(
        "                       NLU labels, not commands — no seed or row floor "
        "applies"
    )

    if excluded:
        names = ", ".join(row.command_name for row in excluded[:HEALTHY_SAMPLE_LIMIT])
        if len(excluded) > HEALTHY_SAMPLE_LIMIT:
            names += f", +{len(excluded) - HEALTHY_SAMPLE_LIMIT} more"
        lines.append(
            f"    not intent-routed: {len(excluded)} command(s) have no "
            f"Signature.Input, so they are"
        )
        lines.append(
            f"                       dispatched via perform_action only [{names}]"
        )

    if report.heldout_available:
        lines.append(
            "    held-out         : per-command scores from the R1 evaluation report"
        )
    else:
        lines.append(
            "    held-out         : not available (no heldout_evaluation.json); "
            "run R1's evaluation"
        )

    lines.extend(_format_problems(report))

    lines.append("")
    lines.append(
        f"  Floors: rows >= {report.min_rows} (structural — one train and one "
        f"evaluation row per label),"
    )
    lines.append(
        f"  not from an accuracy measurement), seeds >= {report.min_seeds} (advisory — "
        f"one workflow's"
    )
    lines.append("  observation). These are trainer policy, not workflow configuration.")
    lines.append(
        "  Structural failures stop publication; advisory findings do not."
    )
    lines.append(rule)
    return "\n".join(lines)


def _format_problems(report: TrainingReport) -> list[str]:
    if not report.problems:
        return []
    lines = ["", f"  INCOMPLETE ({len(report.problems)}) — this report is missing data:"]
    lines.extend(f"    - {problem}" for problem in report.problems)
    return lines


# ---------------------------------------------------------------------
# Persistence and the one call the trainer makes
# ---------------------------------------------------------------------

def write_report(workflow_folderpath: str, report: TrainingReport) -> tuple[str, str]:
    """Write the text and JSON reports beside the provenance they came from.

    Returns `(text_path, json_path)`. Two formats because they have two readers: the
    text file is what a developer opens after the training output has scrolled away,
    the JSON is what a CI job or a convergence-loop script reads. Both are cheap and
    neither duplicates provenance, which records inputs rather than verdicts.
    """
    output_dir = command_info_dir(workflow_folderpath)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / REPORT_FILENAME
    text_path.write_text(format_report(report) + "\n", encoding="utf-8")

    json_path = output_dir / REPORT_JSON_FILENAME
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return str(text_path), str(json_path)


def report_training_data(
    workflow_folderpath: str,
    print_report: bool = True,
    write: bool = True,
) -> Optional[TrainingReport]:
    """Build, optionally print, and optionally persist the report. Never raises.

    This is the single entry point the trainer calls. It swallows everything, because
    it runs at the end of a run that has already produced its artifacts: a defect in
    the reporting code must never be what a developer sees instead of their trained
    models. On failure it logs and returns None.
    """
    try:
        report = build_report(workflow_folderpath)
        if print_report:
            print(format_report(report), flush=True)
        if write:
            text_path, _json_path = write_report(workflow_folderpath, report)
            if print_report:
                print(f"  Written to {text_path}", flush=True)
        return report
    except Exception as exc:  # noqa: BLE001 - reporting must not break training
        logger.error(
            f"Could not produce the per-command training report for "
            f"{workflow_folderpath}: {type(exc).__name__}: {exc}"
        )
        return None
