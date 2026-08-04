"""Deterministic training: global seeding plus per-command utterance provenance.

Two responsibilities, both prerequisites for the utterance cache (spec decision D6)
and for any paired comparison of two training runs:

1. **Seeding.** `seed_everything` seeds `random`, `numpy`, `torch` (+CUDA) and
   `transformers` from a single `TRAINING_SEED`. Before this module nothing in the
   package was seeded except `random_state=42` on the train/test split, which is why
   two identical runs disagreed on 20.6% of held-out routing cases.

2. **Provenance.** `UtteranceProvenance` records, per command, which seed and which
   PersonaHub rows produced its training utterances, whether generation fell back,
   and how many rows it ended up with. `generate_diverse_utterances` cannot return
   this — its signature is public API called from user-authored command files in
   every workflow — so it pushes into a process-wide `ProvenanceRecorder` that the
   trainer installs for the duration of a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.utils.logging import logger

# numpy / torch / transformers are declared dependencies, but seeding must not be the
# thing that makes an otherwise-working install fail, and this module is imported
# transitively by every command file via generate_synthetic. Degrade quietly instead.
try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
    _np = None

try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    _torch = None

try:
    from transformers import set_seed as _transformers_set_seed
except ImportError:  # pragma: no cover
    _transformers_set_seed = None


DEFAULT_TRAINING_SEED: int = 42

# Persona id used for hand-written seed utterances and for the command-name token.
# These are authored, not generated, so they belong to no PersonaHub row and must
# never be held out by a whole-persona evaluation split (R1/D1).
SEED_PERSONA_ID: str = "__seed__"

# Prefix for an utterance whose producing persona could not be resolved from the
# LLM's echoed persona name. The suffix lists every persona in the batch that
# produced it; a whole-persona holdout must treat the utterance as belonging to
# all of them.
UNRESOLVED_PERSONA_PREFIX: str = "__unresolved__:"

# Separator for a composite persona id, i.e. one utterance text produced by more
# than one persona.
PERSONA_ID_SEPARATOR: str = "+"

PROVENANCE_FILENAME: str = "training_provenance.json"
COMMAND_INFO_FOLDERNAME: str = "___command_info"
PROVENANCE_SCHEMA_VERSION: int = 2

# 2**31 - 1 keeps derived seeds inside the range accepted by numpy's legacy
# RandomState and by torch.manual_seed on every platform.
_SEED_MODULUS: int = 2**31 - 1


def get_training_seed() -> int:
    """Return fastWorkflow's fixed production training seed.

    Seed selection is part of the trainer implementation, not workflow configuration.
    Research code can still call ``seed_everything(seed)`` and
    ``select_persona_indices(..., seed, ...)`` explicitly without adding a knob to every
    user's environment file.
    """
    return DEFAULT_TRAINING_SEED


def seed_everything(seed: Optional[int] = None) -> int:
    """Seed `random`, numpy, torch (+CUDA when present) and transformers.

    Passing None reads the seed from the environment. Returns the seed actually
    used so callers can record it in provenance. Safe to call when CUDA is absent.
    """
    if seed is None:
        seed = get_training_seed()
    seed = int(seed)

    random.seed(seed)

    if _np is not None:
        _np.random.seed(seed % (2**32))

    if _torch is not None:
        _torch.manual_seed(seed)
        if getattr(_torch, "cuda", None) is not None and _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

    if _transformers_set_seed is not None:
        _transformers_set_seed(seed)

    return seed


def derived_seed(seed: int, *keys: str) -> int:
    """Derive a stable sub-seed from `(seed, *keys)`.

    Stable across processes and independent of how many random calls preceded it —
    that independence is the point: a sub-seed derived here gives the same value on
    every run regardless of the order commands are trained in, whereas the global
    `random` module does not.

    Uses `hashlib`, never the builtin `hash()`, which is salted per process.
    """
    normalized_seed = int(seed)
    payload = "\x00".join([f"{normalized_seed}", *[str(k) for k in keys]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


class UtteranceProvenance(BaseModel):
    """Everything needed to reproduce (or invalidate) one command's utterance set."""

    command_name: str
    seed: int
    persona_ids: list[str] = Field(default_factory=list)
    utterance_personas: dict[str, str] = Field(default_factory=dict)
    generator_config: dict = Field(default_factory=dict)
    fell_back: bool = False
    fallback_reason: Optional[str] = None
    seed_utterance_count: int = 0
    generated_count: int = 0
    final_count: int = 0


class ContextTrainingStatus(str, Enum):
    """How one fully-qualified command participated in one context model."""

    INCLUDED = "included"
    INCLUDED_FALLBACK = "included_fallback"
    SKIPPED_NO_INPUT = "skipped_no_input"
    SKIPPED_NO_UTTERANCES = "skipped_no_utterances"


def _normalize_count(value: int | float) -> int:
    """Convert a count to a non-negative integer."""
    return max(0, int(value))


def _normalize_optional_count(value: Optional[int | float]) -> Optional[int]:
    """Normalize a supplied count while preserving omission."""
    return None if value is None else _normalize_count(value)


class ContextTrainingProvenance(BaseModel):
    """Context-specific labelled-row use.

    Application and framework commands use the common fields. Reserved labels also
    populate the optional denominator and budget fields so their class share can be
    reconstructed without a retained console log.
    """

    context_name: str
    command_name: str
    status: ContextTrainingStatus
    row_count: int = 0
    reason: Optional[str] = None
    own_row_count: Optional[int] = None
    raw_candidate_count: Optional[int] = None
    deduplicated_candidate_count: Optional[int] = None
    always_include_count: Optional[int] = None
    selected_budget: Optional[int] = None
    coverage_floor: Optional[int] = None
    coverage_floor_applied: Optional[bool] = None


class ProvenanceRecorder:
    """Collects `UtteranceProvenance` for one training run and persists it.

    Written to `<workflow>/___command_info/training_provenance.json`. The per-command
    training report (fix-551.4) reads this back to surface fallen-back commands as
    table rows, which is the visibility that a lone `logger.error` never provided.

    Generation is keyed only by fully-qualified command because the generator receives
    no context and one generated set is deliberately reused wherever inheritance makes
    that command a label. Context training is a separate `(context, command)` dimension:
    it records repeated row use and explicit skips without inventing duplicate generation
    records.
    """

    def __init__(self, workflow_folderpath: str) -> None:
        self.workflow_folderpath = workflow_folderpath
        self._records: dict[str, UtteranceProvenance] = {}
        self._context_records: dict[
            tuple[str, str], ContextTrainingProvenance
        ] = {}
        self._lock = threading.Lock()

    @property
    def records(self) -> dict[str, UtteranceProvenance]:
        with self._lock:
            return dict(self._records)

    @property
    def context_records(
        self,
    ) -> dict[tuple[str, str], ContextTrainingProvenance]:
        with self._lock:
            return dict(self._context_records)

    def record(self, provenance: UtteranceProvenance) -> None:
        """Store `provenance`, keyed by command name.

        The trainer generates once per fully-qualified command. Direct callers can
        still invoke generation repeatedly, so last write wins, EXCEPT that an existing
        fallen-back record is never overwritten by a successful one — a command that
        was rate-limited even once produced degraded training data and must stay visible
        in the report.
        """
        with self._lock:
            existing = self._records.get(provenance.command_name)
            if existing is not None and existing.fell_back and not provenance.fell_back:
                return
            self._records[provenance.command_name] = provenance

    def record_context(
        self,
        *,
        context_name: str,
        command_name: str,
        status: ContextTrainingStatus,
        row_count: int | float = 0,
        reason: Optional[str] = None,
        own_row_count: Optional[int | float] = None,
        raw_candidate_count: Optional[int | float] = None,
        deduplicated_candidate_count: Optional[int | float] = None,
        always_include_count: Optional[int | float] = None,
        selected_budget: Optional[int | float] = None,
        coverage_floor: Optional[int | float] = None,
        coverage_floor_applied: Optional[bool] = None,
    ) -> None:
        """Record one command's inclusion or explicit skip in one context."""
        record = ContextTrainingProvenance(
            context_name=str(context_name),
            command_name=str(command_name),
            status=status,
            row_count=_normalize_count(row_count),
            reason=reason,
            own_row_count=_normalize_optional_count(own_row_count),
            raw_candidate_count=_normalize_optional_count(raw_candidate_count),
            deduplicated_candidate_count=_normalize_optional_count(
                deduplicated_candidate_count
            ),
            always_include_count=_normalize_optional_count(always_include_count),
            selected_budget=_normalize_optional_count(selected_budget),
            coverage_floor=_normalize_optional_count(coverage_floor),
            coverage_floor_applied=coverage_floor_applied,
        )
        with self._lock:
            self._context_records[(record.context_name, record.command_name)] = record

    def save(self) -> str:
        """Write the collected records and return the file path.

        Schema v2 uses an envelope so command-generation records and context-training
        records cannot be confused. `load` still accepts the legacy flat command map;
        the next successful save upgrades it without mutating old artifacts in place.
        """
        folder = os.path.join(self.workflow_folderpath, COMMAND_INFO_FOLDERNAME)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, PROVENANCE_FILENAME)
        with self._lock:
            payload = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "commands": {
                    name: record.model_dump()
                    for name, record in sorted(self._records.items())
                },
                "context_training": {
                    context_name: {
                        command_name: self._context_records[
                            (context_name, command_name)
                        ].model_dump(exclude_none=True)
                        for recorded_context, command_name in sorted(
                            self._context_records
                        )
                        if recorded_context == context_name
                    }
                    for context_name in sorted(
                        {context for context, _command in self._context_records}
                    )
                },
            }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, workflow_folderpath: str) -> dict[str, UtteranceProvenance]:
        """Load command-generation records from v2 or the legacy flat schema."""
        payload = cls._load_payload(workflow_folderpath)
        commands = cls._command_payload(payload)
        return {
            name: UtteranceProvenance(**record)
            for name, record in commands.items()
            if isinstance(record, dict)
        }

    @classmethod
    def load_context_records(
        cls, workflow_folderpath: str
    ) -> dict[tuple[str, str], ContextTrainingProvenance]:
        """Load v2 context records; legacy files correctly return an empty map."""
        payload = cls._load_payload(workflow_folderpath)
        context_payload = payload.get("context_training")
        if not isinstance(context_payload, dict):
            return {}

        records: dict[tuple[str, str], ContextTrainingProvenance] = {}
        for context_name, command_map in context_payload.items():
            if not isinstance(command_map, dict):
                continue
            for command_name, raw in command_map.items():
                if not isinstance(raw, dict):
                    continue
                record = ContextTrainingProvenance(**raw)
                records[(str(context_name), str(command_name))] = record
        return records

    @classmethod
    def _load_payload(cls, workflow_folderpath: str) -> dict:
        path = os.path.join(
            workflow_folderpath, COMMAND_INFO_FOLDERNAME, PROVENANCE_FILENAME
        )
        if not os.path.isfile(path):
            return {}
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _command_payload(payload: dict) -> dict:
        commands = payload.get("commands")
        if isinstance(commands, dict):
            return commands
        # Schema v1 was the command map itself. Do not rewrite it merely by reading:
        # immutable artifact versions must remain byte-identical.
        return payload

    def fallback_summary(self) -> list[UtteranceProvenance]:
        """Records whose generation degraded, worst-first by how little they got."""
        with self._lock:
            fallen = [r for r in self._records.values() if r.fell_back]
        return sorted(fallen, key=lambda r: (r.final_count, r.command_name))


_recorder_lock = threading.Lock()
_active_recorder: Optional[ProvenanceRecorder] = None


def set_provenance_recorder(recorder: Optional[ProvenanceRecorder]) -> None:
    """Install (or clear, with None) the recorder that generation pushes into."""
    global _active_recorder
    with _recorder_lock:
        _active_recorder = recorder


def get_provenance_recorder() -> Optional[ProvenanceRecorder]:
    """Return the installed recorder, or None when nothing is collecting."""
    with _recorder_lock:
        return _active_recorder


def record_provenance(provenance: UtteranceProvenance) -> None:
    """Push `provenance` into the installed recorder; a no-op when none is installed.

    Generation runs unchanged outside a training run (e.g. a workflow author calling
    `generate_diverse_utterances` directly), so the absence of a recorder is normal.
    """
    recorder = get_provenance_recorder()
    if recorder is None:
        return
    try:
        recorder.record(provenance)
    except Exception as exc:  # noqa: BLE001 - provenance must never break training
        logger.warning(
            f"Failed to record provenance for command "
            f"'{provenance.command_name}': {exc}"
        )
