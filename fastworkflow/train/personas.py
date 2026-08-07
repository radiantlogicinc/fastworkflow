"""App-supplied and domain-conditioned personas for synthetic generation (spec R9a / F12).

Why this module exists
----------------------
Synthetic training utterances are written by an LLM role-playing a persona drawn from
PersonaHub. The draw is uniform over the whole corpus and conditioned on nothing: a
retail workflow and an identity-governance workflow get the same population of
astrophysicists, sommeliers and marine biologists. Finding F12 records the consequence —
most sampled personas have no relationship to the vocabulary the application's users
actually employ.

This module lets an application say who its users are, in one of two ways:

* **Supply the personas.** Write ``personas.json`` in the workflow folder with a
  ``personas`` list. Those are used verbatim; PersonaHub is not consulted or downloaded.
* **Condition the draw.** Write ``personas.json`` with ``domain_keywords`` and no
  ``personas`` list. PersonaHub is filtered to rows mentioning those keywords, and the
  usual deterministic sample is taken from the survivors.

With no ``personas.json`` and no ``SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE`` the behaviour is
byte-identical to before this module existed: a deterministic sample over the full
PersonaHub corpus, with row indices as persona ids. ``test_personas.py`` pins that
equivalence against ``generate_synthetic.select_persona_indices`` directly rather than
asserting it in prose.

What is NOT claimed
-------------------
That domain-conditioned personas improve intent detection. The spec calls this
"plausibly a larger generalisation lever than anything in the training loop" and F12
labels it a hypothesis. It has not been measured in this repository, and per §11 (M4) it
currently cannot be measured from a single paired run: the utterances come from a live
LLM, so two runs at the same ``TRAINING_SEED`` produce different training data (0/5
commands matched). This module makes the experiment *possible*. It does not report a
result, and nothing here should be read as one.

Persona id discipline
---------------------
Persona ids are load-bearing beyond provenance. ``heldout_evaluation.expand_persona_id``
parses them to recover the atomic personas behind a composite id, and the whole-persona
holdout (decision D1) depends on that parse being correct. An app-supplied id containing
``PERSONA_ID_SEPARATOR``, or colliding with ``SEED_PERSONA_ID`` or
``UNRESOLVED_PERSONA_PREFIX``, would silently corrupt the holdout into the data leak D1
exists to prevent. ``validate_persona_id`` rejects all three, loudly, at load time.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.train.determinism import (
    PERSONA_ID_SEPARATOR,
    SEED_PERSONA_ID,
    UNRESOLVED_PERSONA_PREFIX,
    derived_seed,
    get_training_seed,
)
from fastworkflow.utils.logging import logger

#: File an application drops in its workflow folder to describe its users. Named to sit
#: alongside ``intent_benchmark.json`` (R1b), which is discovered the same way.
PERSONA_FILENAME: str = "personas.json"

#: Escape hatch for a persona file that lives outside the workflow folder, and the only
#: route that works with no integration at all (see ``default_persona_source``).
#:
#: Read with an empty-string code default, which follows the house convention: settable
#: from ``fastworkflow.env``, NOT from a shell ``export``, because a non-None code default
#: short-circuits the ``os.environ`` lookup in ``fastworkflow.get_env_var``. The empty
#: default also suppresses the "does not exist" warning that a ``None`` default logs on
#: every command of every run.
PERSONA_FILE_ENV_VAR: str = "SYNTHETIC_UTTERANCE_GEN_PERSONA_FILE"

PERSONA_SCHEMA_VERSION: int = 1

#: Every app-supplied persona id carries this prefix. PersonaHub ids are stringified row
#: indices, so an unnamespaced app id of ``"42"`` would be indistinguishable from
#: PersonaHub row 42 in a provenance record — and two runs whose persona source changed
#: would appear to have used the same personas.
APP_PERSONA_ID_PREFIX: str = "app:"

#: Source names recorded in provenance. Stable strings: a later run comparing two
#: provenance files keys on these.
SOURCE_PERSONAHUB: str = "personahub"
SOURCE_APP_SUPPLIED: str = "app_supplied"
SOURCE_DOMAIN_CONDITIONED: str = "domain_conditioned"

#: A domain-filtered pool smaller than this multiple of the per-command persona count
#: cannot supply a meaningfully different sample to each command — every command would
#: draw from nearly the same handful of rows, which reintroduces the near-duplication
#: that the whole-persona holdout exists to detect. Below it, the pool is topped up from
#: the unfiltered corpus and the fact is reported.
DEFAULT_MIN_POOL_MULTIPLE: int = 4

#: Where the per-command persona count comes from when a caller does not supply one.
#: `DomainConditionedPersonaSource` needs it to know how big a pool is big enough, and
#: the only production caller (`train/__main__.py`) has no reason to know about persona
#: pools, so the source reads the same setting `generate_synthetic` resolves
#: `num_personas` from. Read with a code default: an absent value must not log a warning
#: and must not make the pool floor collapse to the multiple alone (bd fix-k0i.38).
NUM_PERSONAS_ENV_VAR: str = "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS"

_WORD_RE = re.compile(r"[a-z0-9]+")


class PersonaConfigError(ValueError):
    """Raised when a ``personas.json`` cannot be used as written.

    Deliberately fatal rather than a warning-and-fallback. An application that ships a
    persona file has stated an intent about its training data; silently training on the
    generic PersonaHub draw instead would produce a model whose provenance record does
    not describe how it was built.
    """


def validate_persona_id(persona_id: str) -> str:
    """Return *persona_id* unchanged, or raise `PersonaConfigError`.

    The three rejected shapes are exactly the ones `expand_persona_id` would
    mis-parse. See this module's docstring for why that matters.
    """
    if not persona_id or not persona_id.strip():
        raise PersonaConfigError("Persona id must be a non-empty string.")
    if PERSONA_ID_SEPARATOR in persona_id:
        raise PersonaConfigError(
            f"Persona id {persona_id!r} contains {PERSONA_ID_SEPARATOR!r}, which joins "
            f"the contributors of a composite persona id. An id containing it would be "
            f"split into fragments by heldout_evaluation.expand_persona_id and the "
            f"whole-persona holdout would leak."
        )
    if persona_id == SEED_PERSONA_ID:
        raise PersonaConfigError(
            f"Persona id {persona_id!r} is reserved for hand-written seed utterances, "
            f"which are never held out."
        )
    if persona_id.startswith(UNRESOLVED_PERSONA_PREFIX):
        raise PersonaConfigError(
            f"Persona id {persona_id!r} starts with {UNRESOLVED_PERSONA_PREFIX!r}, which "
            f"marks an utterance whose producing persona could not be resolved."
        )
    return persona_id


class Persona(BaseModel):
    """One persona: a stable id and the description handed to the generator."""

    id: str
    text: str


class PersonaSelection(BaseModel):
    """The personas chosen for one command, plus everything provenance needs.

    ``personas`` and ``persona_ids`` are positionally aligned; the generation loop batches
    them together and maps an echoed persona name back through ``persona_ids``.
    """

    personas: list[str] = Field(default_factory=list)
    persona_ids: list[str] = Field(default_factory=list)
    #: One of the ``SOURCE_*`` constants.
    source: str = SOURCE_PERSONAHUB
    #: Identifies the persona *content*, not just the ids. Two runs can select the same
    #: ids from different persona files; R6's utterance cache must treat those as
    #: different inputs, and persona ids alone (decision D6) would not.
    fingerprint: str = ""
    #: Size of the pool the sample was drawn from. A shrinking pool is the single most
    #: useful diagnostic when domain conditioning starts producing repetitive utterances.
    pool_size: int = 0
    notes: list[str] = Field(default_factory=list)


class PersonaConfig(BaseModel):
    """Parsed ``personas.json``."""

    schema_version: int = PERSONA_SCHEMA_VERSION
    domain: Optional[str] = None
    domain_keywords: list[str] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    #: Where this came from, for error messages and provenance.
    origin: str = ""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _coerce_persona(entry, index: int) -> Persona:
    """Accept either a bare string or ``{"id": ..., "persona"|"text": ...}``."""
    if isinstance(entry, str):
        text = entry.strip()
        raw_id = str(index)
    elif isinstance(entry, dict):
        text = str(entry.get("persona") or entry.get("text") or "").strip()
        raw_id = str(entry.get("id") or index)
    else:
        raise PersonaConfigError(
            f"Persona at position {index} must be a string or an object, "
            f"found {type(entry).__name__}."
        )
    if not text:
        raise PersonaConfigError(
            f"Persona at position {index} has no description text."
        )
    return Persona(id=validate_persona_id(APP_PERSONA_ID_PREFIX + raw_id), text=text)


def load_persona_config(path: str) -> PersonaConfig:
    """Load and validate a persona file.

    Accepts the documented object form ``{"schema_version": 1, ...}`` or a bare list of
    personas, mirroring `heldout_evaluation.load_benchmark_file`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        raw = {"personas": raw}
    if not isinstance(raw, dict):
        raise PersonaConfigError(
            f"Persona file {path} must contain an object or a list of personas; "
            f"found {type(raw).__name__}."
        )

    version = raw.get("schema_version", PERSONA_SCHEMA_VERSION)
    if version != PERSONA_SCHEMA_VERSION:
        raise PersonaConfigError(
            f"Unsupported persona schema_version {version!r} in {path}; this build "
            f"understands {PERSONA_SCHEMA_VERSION}."
        )

    personas = [
        _coerce_persona(entry, index)
        for index, entry in enumerate(raw.get("personas") or [])
    ]

    seen: set[str] = set()
    for persona in personas:
        if persona.id in seen:
            raise PersonaConfigError(
                f"Duplicate persona id {persona.id!r} in {path}. Ids must be unique or "
                f"the provenance record cannot say which persona wrote what."
            )
        seen.add(persona.id)

    keywords = [str(k).strip() for k in (raw.get("domain_keywords") or []) if str(k).strip()]
    domain = raw.get("domain")
    if domain and not keywords:
        # A prose domain with no keyword list is still usable: its own words are the
        # keywords. This is the least-effort form of the feature and worth supporting.
        keywords = [w for w in _WORD_RE.findall(str(domain).lower()) if len(w) > 2]

    if not personas and not keywords:
        raise PersonaConfigError(
            f"Persona file {path} supplies neither a non-empty 'personas' list nor "
            f"'domain_keywords'/'domain'. Delete the file to use the default PersonaHub "
            f"draw; an empty file is more likely a mistake than an intent."
        )

    return PersonaConfig(
        schema_version=PERSONA_SCHEMA_VERSION,
        domain=str(domain) if domain else None,
        domain_keywords=keywords,
        personas=personas,
        origin=str(path),
    )


def persona_config_path(workflow_folderpath: str) -> str:
    """Return the path a workflow's persona file is discovered at."""
    return str(Path(workflow_folderpath) / PERSONA_FILENAME)


# ---------------------------------------------------------------------------
# Persona sources
# ---------------------------------------------------------------------------


def _resolve_num_personas_hint(num_personas_hint: Optional[int]) -> int:
    """How many personas a command will ask for, for pool-sizing purposes only.

    Falls back to `NUM_PERSONAS_ENV_VAR` so the hint is a real number in production
    rather than the 0 that made the pool floor collapse to `DEFAULT_MIN_POOL_MULTIPLE`
    on its own. Never raises: a malformed value degrades to "no hint", because a
    persona pool is not worth failing a training run over.
    """
    if num_personas_hint is not None:
        return max(0, int(num_personas_hint))
    try:
        configured = fastworkflow.get_env_var(NUM_PERSONAS_ENV_VAR, int, default=0)
    except (TypeError, ValueError):
        return 0
    return max(0, int(configured or 0))


class PersonaSource:
    """A pool of personas that a command can draw a deterministic sample from.

    Indexed rather than materialised: PersonaHub is ~200k rows and building a list of
    strings for it, per command, would dominate generation time. Subclasses expose
    ``pool_size()`` and ``persona_at(position)``; the sampling logic is shared and lives
    in ``select``.

    Sampling lives in ONE place on purpose. `generate_synthetic` digests
    ``PersonaSource.select`` into the utterance-cache key (bd fix-k0i.43) precisely so
    that an edit to how personas are drawn invalidates the cached utterances those
    personas wrote. A subclass that overrode ``select`` would slip out from under that
    digest, so a subclass with an unusual pool states the fact through
    ``priority_pool_size`` and lets the shared code act on it.
    """

    #: One of the ``SOURCE_*`` constants.
    name: str = SOURCE_PERSONAHUB

    def pool_size(self) -> int:
        raise NotImplementedError

    def persona_at(self, position: int) -> Persona:
        raise NotImplementedError

    def priority_pool_size(self) -> int:
        """How many leading pool positions must be exhausted before the rest.

        0 (the default) means the pool is one uniform population, which is what every
        source whose rows it actually wanted should say. A source that had to pad a
        thin pool with rows it did NOT want overrides this with the count of the rows
        it did — see `DomainConditionedPersonaSource.priority_pool_size`.
        """
        return 0

    def fingerprint(self) -> str:
        """Identify this source's *content*, without loading the corpus.

        Corpus-free by contract: the utterance cache (R6) fingerprints every command
        before deciding whether it needs to generate anything, and making that decision
        require a 200k-row download would undo the reason the cache exists.
        """
        raise NotImplementedError

    def notes(self) -> list[str]:
        """Anything the developer should know about this pool. Reported once per run."""
        return []

    def select(
        self, num_personas: int, seed: int, command_name: str
    ) -> PersonaSelection:
        """Draw *num_personas* personas for *command_name*, reproducibly.

        The draw is driven by a private ``random.Random`` seeded from
        ``(seed, command_name)`` — never the global ``random`` module, whose state
        depends on how many draws happened earlier in the process. This reproduces
        `generate_synthetic.select_persona_indices` exactly for the PersonaHub source,
        which is what keeps the no-persona-file path byte-identical to before.

        When ``priority_pool_size`` is non-zero the priority rows are drawn from first
        and the remainder of the request is filled from the rest of the pool. That is
        what keeps domain conditioning from being nullified by its own padding: a pool
        of 3 matched rows plus 13 generic ones sampled uniformly would put a matched
        persona in roughly one draw in five, whereas exhausting the matched rows first
        uses all 3 of them in every command (bd fix-k0i.38).
        """
        size = self.pool_size()
        if not size or not num_personas or num_personas <= 0:
            return PersonaSelection(
                source=self.name,
                fingerprint=self.fingerprint(),
                pool_size=size,
                notes=list(self.notes()),
            )
        rng = random.Random(derived_seed(seed, command_name))
        requested = min(num_personas, size)
        priority = min(max(0, self.priority_pool_size()), size)
        if 0 < priority < size:
            from_priority = min(requested, priority)
            positions = rng.sample(range(priority), from_priority)
            if from_priority < requested:
                positions += rng.sample(
                    range(priority, size), requested - from_priority
                )
        else:
            # The uniform draw, byte-identical to `select_persona_indices`. Every
            # source that has not padded its pool takes this branch, including the
            # default PersonaHub one — `test_personas` pins that equivalence.
            positions = rng.sample(range(size), requested)
        personas = [self.persona_at(position) for position in positions]
        return PersonaSelection(
            personas=[persona.text for persona in personas],
            persona_ids=[persona.id for persona in personas],
            source=self.name,
            fingerprint=self.fingerprint(),
            pool_size=size,
            notes=list(self.notes()),
        )


class PersonaHubSource(PersonaSource):
    """The default: a uniform draw over the whole PersonaHub corpus.

    Persona ids are stringified row indices, exactly as before this module existed, so
    provenance records written by older runs remain comparable with new ones.
    """

    name = SOURCE_PERSONAHUB

    def __init__(self, dataset_loader: Callable):
        self._dataset_loader = dataset_loader
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self._dataset_loader()
        return self._dataset

    def pool_size(self) -> int:
        return len(self.dataset)

    def persona_at(self, position: int) -> Persona:
        return Persona(id=str(position), text=self.dataset[position]["persona"])

    def fingerprint(self) -> str:
        # A constant, and deliberately not a function of the corpus: this is the
        # pre-existing default, and `active_persona_source_label` maps it to None so the
        # cache key of a workflow with no persona file is exactly what it was before R9a
        # existed. Hashing the corpus length here would invalidate every cached utterance
        # in every workflow the day PersonaHub gains a row.
        return SOURCE_PERSONAHUB


class AppPersonaSource(PersonaSource):
    """Personas supplied by the application. PersonaHub is never loaded."""

    name = SOURCE_APP_SUPPLIED

    def __init__(self, personas: list[Persona], origin: str = ""):
        if not personas:
            raise PersonaConfigError("An app-supplied persona source needs at least one persona.")
        self._personas = list(personas)
        self._origin = origin

    def pool_size(self) -> int:
        return len(self._personas)

    def persona_at(self, position: int) -> Persona:
        return self._personas[position]

    def fingerprint(self) -> str:
        # Hashes the persona TEXT, not just the ids. Editing a persona's description
        # while keeping its id must invalidate R6's utterance cache; ids alone would not.
        return _hash_parts(
            SOURCE_APP_SUPPLIED,
            *[f"{p.id}\x01{p.text}" for p in self._personas],
        )

    def notes(self) -> list[str]:
        notes = [
            f"Using {len(self._personas)} app-supplied persona(s)"
            + (f" from {self._origin}" if self._origin else "")
            + "; PersonaHub was not consulted."
        ]
        if len(self._personas) < DEFAULT_MIN_POOL_MULTIPLE:
            notes.append(
                f"Only {len(self._personas)} persona(s) supplied. Every command will draw "
                f"from nearly the same set, so its utterances will share a voice and the "
                f"whole-persona holdout (R1/D1) will have little to hold out."
            )
        return notes


class DomainConditionedPersonaSource(PersonaSource):
    """PersonaHub filtered to rows mentioning the application's domain keywords.

    Filtering is a whole-word match on the lowercased persona text, scanned once and
    cached. Multi-word keywords are matched as substrings. A row matching any keyword
    survives; the surviving row indices become the pool and keep their original PersonaHub
    ids, so a provenance record stays comparable with an unfiltered run.
    """

    name = SOURCE_DOMAIN_CONDITIONED

    def __init__(
        self,
        dataset_loader: Callable,
        keywords: Iterable[str],
        num_personas_hint: Optional[int] = None,
        min_pool_multiple: int = DEFAULT_MIN_POOL_MULTIPLE,
        origin: str = "",
    ):
        self._dataset_loader = dataset_loader
        self._keywords = sorted({str(k).strip().lower() for k in keywords if str(k).strip()})
        if not self._keywords:
            raise PersonaConfigError("Domain conditioning needs at least one keyword.")
        self._num_personas_hint = _resolve_num_personas_hint(num_personas_hint)
        self._min_pool_multiple = int(min_pool_multiple)
        self._origin = origin
        self._dataset = None
        self._rows: Optional[list[int]] = None
        #: How many of the leading entries in ``rows`` actually matched the keywords.
        #: Equal to ``len(rows)`` unless the pool had to be padded.
        self._matched_count: int = 0
        self._notes: list[str] = []

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self._dataset_loader()
        return self._dataset

    def _matches(self, text: str) -> bool:
        lowered = text.lower()
        words = set(_WORD_RE.findall(lowered))
        for keyword in self._keywords:
            if " " in keyword or "-" in keyword:
                if keyword in lowered:
                    return True
            elif keyword in words:
                return True
        return False

    @property
    def rows(self) -> list[int]:
        if self._rows is not None:
            return self._rows

        dataset = self.dataset
        total = len(dataset)
        matched = [
            index
            for index, row in enumerate(dataset)
            if self._matches(str(row.get("persona") or ""))
        ]

        wanted = max(
            self._num_personas_hint * self._min_pool_multiple,
            self._min_pool_multiple,
        )
        if not matched:
            self._notes.append(
                f"No PersonaHub persona matched domain keywords "
                f"{', '.join(self._keywords)}; falling back to the full corpus of {total} "
                f"personas. Domain conditioning had no effect on this run."
            )
            self._rows = list(range(total))
            self._matched_count = 0
            return self._rows

        if len(matched) < wanted:
            # Topping up rather than failing: too small a pool degrades diversity, but an
            # empty-handed abort would make a mistyped keyword stop a multi-hour training
            # run at the first command. The developer is told, in the same place fallen-back
            # generation is reported.
            #
            # Only the SHORTFALL is added, and it goes AFTER the matched rows so
            # `priority_pool_size` can make the shared sampler exhaust the matched rows
            # first. Appending every unmatched row instead — which is what this did —
            # made the pool the entire ~200k corpus and reduced a matched persona's
            # chance of being drawn to ~matched/200k, nullifying the conditioning at
            # exactly the moment it was thinnest and reporting it as merely "partial"
            # (bd fix-k0i.38).
            matched_set = set(matched)
            candidates = [index for index in range(total) if index not in matched_set]
            shortfall = wanted - len(matched)
            if shortfall < len(candidates):
                # Sampled rather than sliced, so the padding is spread across the corpus
                # instead of being the first N rows of it, and seeded from the keywords
                # rather than the training seed so the POOL is the same on every run at
                # every seed — a pool that moved per seed would make the utterance cache
                # miss for a reason no fingerprint input records.
                padding_rng = random.Random(
                    derived_seed(0, SOURCE_DOMAIN_CONDITIONED, *self._keywords)
                )
                candidates = sorted(padding_rng.sample(candidates, shortfall))
            self._notes.append(
                f"Only {len(matched)} of {total} PersonaHub personas matched domain "
                f"keywords {', '.join(self._keywords)}, fewer than the {wanted} needed for "
                f"a varied per-command draw. The pool was topped up to "
                f"{len(matched) + len(candidates)} with unfiltered personas; the "
                f"{len(matched)} matching one(s) are still drawn first, so a command "
                f"asking for more than {len(matched)} personas gets unconditioned ones "
                f"for the remainder. Add keywords or supply personas.json to fix it."
            )
            self._rows = matched + candidates
            self._matched_count = len(matched)
            return self._rows

        self._notes.append(
            f"Domain-conditioned persona pool: {len(matched)} of {total} PersonaHub "
            f"personas matched {', '.join(self._keywords)}"
            + (f" (from {self._origin})" if self._origin else "")
            + "."
        )
        self._rows = matched
        self._matched_count = len(matched)
        return self._rows

    def pool_size(self) -> int:
        return len(self.rows)

    def priority_pool_size(self) -> int:
        """The count of keyword-matching rows, which lead ``rows``.

        Equal to ``pool_size()`` when nothing was padded, which makes
        `PersonaSource.select` take its plain uniform branch. Only a padded pool
        reports a smaller number, and only then does the draw treat the pool as two
        tiers.
        """
        _ = self.rows
        return self._matched_count

    def persona_at(self, position: int) -> Persona:
        row = self.rows[position]
        return Persona(id=str(row), text=self.dataset[row]["persona"])

    def fingerprint(self) -> str:
        # Keywords only. The resulting pool size is a function of them plus the corpus,
        # and computing it here would force the download this method must not need.
        return _hash_parts(SOURCE_DOMAIN_CONDITIONED, *self._keywords)

    def notes(self) -> list[str]:
        # Touch `rows` so the filter has run and its note exists even if no command has
        # drawn from this source yet.
        _ = self.rows
        return list(self._notes)


def _hash_parts(*parts: str) -> str:
    payload = "\x00".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def persona_source_from_config(
    config: PersonaConfig,
    dataset_loader: Callable,
    num_personas_hint: Optional[int] = None,
) -> PersonaSource:
    """Build the source a parsed config asks for.

    An explicit ``personas`` list wins over ``domain_keywords``: supplying personas is the
    stronger statement, and honouring the keywords as well would silently mix generic
    PersonaHub rows into a set the developer curated.

    ``num_personas_hint`` is advisory and only sizes a domain-conditioned pool. None
    means "read the configured per-command persona count", which is what every
    production caller wants and none of them should have to know.
    """
    if config.personas:
        return AppPersonaSource(config.personas, origin=config.origin)
    return DomainConditionedPersonaSource(
        dataset_loader,
        config.domain_keywords,
        num_personas_hint=num_personas_hint,
        origin=config.origin,
    )


def persona_source_for_workflow(
    workflow_folderpath: str,
    dataset_loader: Optional[Callable] = None,
    num_personas_hint: Optional[int] = None,
) -> PersonaSource:
    """Return the persona source for *workflow_folderpath*.

    Discovers ``<workflow>/personas.json``; falls back to `PERSONA_FILE_ENV_VAR`; falls
    back to plain PersonaHub. Raises `PersonaConfigError` if a file exists but cannot be
    used — see that class for why this is not a warning.
    """
    loader = dataset_loader or _default_dataset_loader
    path = persona_config_path(workflow_folderpath)
    if not os.path.isfile(path):
        env_path = _persona_file_from_env()
        if not env_path:
            return PersonaHubSource(loader)
        path = env_path
    return persona_source_from_config(
        load_persona_config(path), loader, num_personas_hint=num_personas_hint
    )


def _persona_file_from_env() -> str:
    """Return the persona file path configured in the env file, or "" when unset/missing."""
    configured = fastworkflow.get_env_var(PERSONA_FILE_ENV_VAR, default="")
    configured = str(configured or "").strip()
    return configured if configured and os.path.isfile(configured) else ""


def _default_dataset_loader():
    """Load PersonaHub via the generator's own loader.

    Imported through the module object rather than at module import time because
    ``generate_synthetic`` imports *this* module: a top-level ``from ... import
    load_persona_dataset`` would be a circular import. The attribute lookup happens only
    when a PersonaHub-backed source actually needs the corpus, which an app-supplied
    source never does.
    """
    import fastworkflow.train.generate_synthetic as generate_synthetic  # noqa: PLC0415

    return generate_synthetic.load_persona_dataset()


# ---------------------------------------------------------------------------
# Process-wide installed source
# ---------------------------------------------------------------------------

_source_lock = threading.Lock()
_active_source: Optional[PersonaSource] = None
_notes_reported: set[str] = set()


def set_persona_source(source: Optional[PersonaSource]) -> None:
    """Install (or clear, with None) the source generation draws from.

    Process-wide for the same reason `ProvenanceRecorder` is: the call site is
    ``generate_diverse_utterances``, whose signature is public API invoked from
    user-authored command files in every workflow and cannot grow a parameter.
    """
    global _active_source
    with _source_lock:
        _active_source = source
        _notes_reported.clear()


def get_persona_source() -> Optional[PersonaSource]:
    """Return the installed source, or None when nothing is installed."""
    with _source_lock:
        return _active_source


def default_persona_source(dataset_loader: Callable) -> PersonaSource:
    """The source to use when nothing has been installed.

    Consults `PERSONA_FILE_ENV_VAR` so an application can condition personas with no
    integration at all, then falls back to plain PersonaHub — the pre-existing behaviour.
    """
    if env_path := _persona_file_from_env():
        return persona_source_from_config(load_persona_config(env_path), dataset_loader)
    return PersonaHubSource(dataset_loader)


def _report_notes(source_name: str, notes: list[str]) -> None:
    """Log each distinct note once per installed source.

    Generation calls this once per command; a domain-filtered pool would otherwise print
    the same line 160 times in a run whose log is already tqdm-flooded (F3's real cause).
    """
    for note in notes:
        key = f"{source_name}\x00{note}"
        with _source_lock:
            if key in _notes_reported:
                continue
            _notes_reported.add(key)
        logger.info(f"Persona source: {note}")


def active_persona_source_name() -> str:
    """Name the active persona source, without loading or downloading it.

    Distinct from `active_persona_source_label` in two ways that both matter to a caller
    recording provenance: it never returns None, and it carries no fingerprint. The label
    is a cache key, so it must be absent for the default draw (to avoid invalidating every
    existing entry) and must change whenever the persona CONTENT changes. This is a
    description of the run, so it must always be present and must NOT change when the
    content changes -- otherwise editing a persona file would alter provenance for reasons
    unrelated to what was generated.
    """
    source = get_persona_source() or default_persona_source(_default_dataset_loader)
    return source.name


def active_persona_source_label() -> Optional[str]:
    """Label the active persona source for the utterance cache fingerprint (R6 / D6).

    Returns None when the default PersonaHub draw is in effect, which is the whole point:
    the caller then keys the cache exactly as it did before this module existed, so no
    workflow's cached utterances are invalidated by R9a landing.

    A non-None label makes an app-supplied or domain-conditioned run a different cache
    variant. Without it, a workflow that adds a ``personas.json`` would keep serving
    utterances generated from generic PersonaHub personas out of the cache and the feature
    would appear to do nothing.

    Corpus-free, because it is computed before the cache decides whether to generate.
    """
    source = get_persona_source() or default_persona_source(_default_dataset_loader)
    if source.name == SOURCE_PERSONAHUB:
        return None
    return f"{source.name}#{source.fingerprint()}"


def persona_source_needs_corpus() -> bool:
    """Whether the active source will read PersonaHub, and so needs ``datasets``.

    False only for an app-supplied set. `generate_diverse_utterances_with_provenance`
    refuses to generate when ``datasets`` is unimportable, which is right for a
    PersonaHub draw and wrong for a workflow that wrote its own personas: that workflow
    needs no corpus, no download, and no optional dependency.
    """
    source = get_persona_source() or default_persona_source(_default_dataset_loader)
    return source.name != SOURCE_APP_SUPPLIED


def resolve_personas(
    num_personas: int,
    seed: Optional[int],
    command_name: str,
    dataset_loader: Callable,
) -> PersonaSelection:
    """Choose the personas for one command. This is the hook `generate_synthetic` calls.

    *dataset_loader* is a callable, not a loaded dataset, so an app-supplied persona set
    never triggers a PersonaHub download.
    """
    if seed is None:
        seed = get_training_seed()
    source = get_persona_source() or default_persona_source(dataset_loader)
    selection = source.select(num_personas, int(seed), command_name)
    _report_notes(source.name, selection.notes)
    return selection
