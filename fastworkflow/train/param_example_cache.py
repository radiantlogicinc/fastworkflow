"""Persist and reuse LLM-generated DSPy parameter examples (bd fix-czb, spec R2/R6).

Why this exists
---------------
R6 made the INTENT training data reproducible: two runs of `examples/hello_world` at
`TRAINING_SEED=42` went from 0/5 to 5/5 commands with identical utterance sets, with
`training_provenance.json` byte-identical. It covers exactly one of the two LLM calls
that training makes.

The other one is `fastworkflow.utils.generate_param_examples.generate_dspy_examples`.
It asks an LLM for ~15 `dspy.Example(...)` lines per parameterised command and writes
them to `___command_info/<command>_param_labeled.json`, which the runtime loads as
few-shot examples for parameter extraction. It runs at temperature 0.9, it takes no
seed, and no seed could reach it anyway. So "two runs at the same `TRAINING_SEED`
produce identical artifacts" — R2's acceptance criterion — stayed false with R6 alone.

The fix is the same shape as R6's, because the problem is the same shape: content-
address the generation configuration, and reuse the draw instead of redrawing.

Relationship to `utterance_cache`
---------------------------------
This is a sibling module, not a copy and not a generalisation of `UtteranceCache`.
The generic parts are IMPORTED from `utterance_cache` rather than duplicated:
`Fingerprint`, `source_digest`, `callable_identity`, `normalize_mode`, `slugify` and
`atomic_write_model`. What is NOT shared is the payload, and that is the whole reason
for a second module: an utterance entry is a list of strings with per-utterance
persona attribution; a parameter example entry is two lists of dicts (accepted and
rejected examples) for which persona attribution is meaningless. Forcing both into one
entry type would have meant a model with four fields that are always empty for one of
its two users.

Where the cache lives, and why
------------------------------
    <workflow>/___command_info/param_example_cache/<command-slug>.<variant-key>.json

Same four constraints as R6, decided the same way and for the same reasons:

* **Not inside `versions/<id>/`.** Versions are immutable snapshots (R4) and the
  cache is mutable and shared by all of them — automatic bounded retention must not
  silently delete the reuse that makes the next run reproducible.
* **Not in `~/.cache`.** That makes "did this workflow train reproducibly?" a
  property of one laptop, and it does not survive a container build.
* **Not committed by accident.** `___command_info*` is already in `.gitignore`.
* **Inspectable.** One JSON file per command variant, holding the examples verbatim.

`CACHE_DIRNAME` is listed in `artifact_versioning.RESERVED_TOPLEVEL_NAMES`, which is
what exempts it from `_prune_stale_artifacts`, from `publish_version`'s stale sweep
and from `migrate_legacy_to_version` — exactly as R6 did for `utterance_cache`.

Two-part key
------------
An entry is addressed by **(variant key, seed)**, as in R6. Note what the seed does
and does not mean here: the fixed trainer seed has no causal influence on this prompt at all
(unlike utterance generation, where persona selection is seed-derived). It is still
part of the address, for two reasons. First, a seed sweep — the standard way R1 and
the convergence loop measure run-to-run variance — must vary the whole pipeline; if
parameter examples were shared across seeds, the sweep would hold a nuisance variable
fixed by accident and understate the variance of "train this workflow". Second, it
keeps the two caches' semantics identical, so "is this run reproducible?" has one
answer rather than two. The cost is one extra LLM call per parameterised command when
the seed changes.

Modes
-----
`fastworkflow train --regenerate-utterances` passes `regenerate` directly to both
generated-data caches. Programmatic callers can still construct a cache with an explicit
mode. Historical cross-seed aggregation was intentionally removed from both caches because
it was unreachable through the shipped trainer. It was especially undefined here:
unioning parameter examples would change the SIZE of the few-shot set the runtime uses,
which is a behaviour change dressed up as a cache mode, and nothing has measured whether
more few-shot examples help or hurt.

Never cached: a generation that produced no usable examples. Freezing an empty draw
would turn one transient parse failure or truncated response into a command that
trains with no few-shot examples on every subsequent run, with nothing left to explain
why. Same ruling as R6, same reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from fastworkflow.train.determinism import COMMAND_INFO_FOLDERNAME
from fastworkflow.train.utterance_cache import (
    DEFAULT_CACHE_MODE,
    MODE_REGENERATE,
    MODE_REUSE,
    Fingerprint,
    atomic_write_model,
    normalize_mode,
    slugify,
)
from fastworkflow.utils.logging import logger

# Bumped by hand when the on-disk shape changes in a way older readers cannot
# interpret. A file with a different value is ignored (a miss), never migrated.
CACHE_FORMAT_VERSION: int = 1

CACHE_DIRNAME: str = "param_example_cache"
CACHE_README_FILENAME: str = "README.md"

_CACHE_README = """\
# Generated DSPy parameter-example cache — DELETING THIS COSTS MONEY AND REPRODUCIBILITY

Each file here holds the `dspy.Example(...)` draws that one command's parameter
extractor is few-shot prompted with, as produced by an LLM at temperature 0.9. They
are kept so that two training runs with the fixed trainer seed write the same
`<command>_param_labeled.json`. Without them, seeding does nothing at all for this
path: nothing in the request is seeded, and the sampling temperature is high on
purpose so the examples are diverse.

* One file per (command, generation-configuration) pair; the hex suffix is the
  fingerprint of that configuration.
* Inside, `entries` is keyed by the training seed.
* An entry is reused only when every fingerprint input matches — the command name,
  the parameter model's field annotations exactly as they are rendered into the
  prompt, how many examples were asked for, the validation threshold, the model, the
  temperature, and the text of the prompt-building and example-parsing code. Edit any
  of them and the command regenerates.
* A generation that yielded no usable examples is never written here.

Force regeneration with `fastworkflow train ... --regenerate-utterances`, which
refreshes both generated artifacts.
"""


def _utc_now() -> str:
    """UTC timestamp, used only as human-facing metadata on an entry."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_cache_mode() -> str:
    """Return the normal production mode.

    The CLI passes ``regenerate`` directly to the cache constructor when requested;
    cache policy is not workflow environment configuration.
    """
    return DEFAULT_CACHE_MODE


def _canonical_digest(payload: Any) -> str:
    """Stable short digest of any JSON-serialisable structure.

    `default=repr` so a `FieldInfo`, a type object or anything else pydantic hands us
    inside a field-details record still digests rather than raising. A repr that moves
    between library versions produces a different digest, which invalidates the entry
    — the safe direction, and in this case the correct one, because that same repr is
    what gets rendered into the prompt.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def compute_fingerprint(
    *,
    command_name: str,
    field_annotations_text: str,
    field_details: Any,
    num_examples: Optional[int],
    validation_threshold: Optional[float],
    temperature: Optional[float],
    model: Optional[str],
    api_base: Optional[str] = None,
    completion_backend: str = "",
    generator_source_digest: str = "",
) -> Fingerprint:
    """Build the variant key for one command's parameter-example configuration.

    Every argument is something that changes what the LLM would produce, or what would
    be written to disk from what it produced:

    * `field_annotations_text` — `str(ParameterModel.model_fields)`, interpolated
      verbatim into the prompt inside a ```python fence. **This is the input whose
      silent staleness would be worst**: a developer adds a field to a command's
      parameter model, and without this the runtime keeps few-shot prompting with
      examples that cannot possibly mention it.
    * `field_details` — the structured extraction of the same annotations, which is
      rendered into the prompt a second time as the "Fields to extract" section AND
      drives which parameters the validator looks for. Digested separately from the
      raw text because `extract_field_details` can change what it pulls out of an
      unchanged annotation.
    * `command_name` — interpolated into the prompt three times.
    * `num_examples` — asked for literally, three times.
    * `validation_threshold` — does not reach the LLM, but decides which examples land
      in `rejected_examples`, and that list is part of the cached artifact.
    * `temperature` — see the module docstring; kept explicit rather than left to the
      source digest so that a future decision to lower it invalidates every entry
      loudly instead of silently reusing high-temperature draws.
    * `model` / `api_base` — a different model, or the same model name pointed at a
      different proxy, is a different generator.
    * `completion_backend` — an injected completion function can never share an entry
      with the real one.
    * `generator_source_digest` — the prompt template and the parsing code.

    `max_tokens` is deliberately absent: it is a literal inside
    `generate_dspy_examples`, so the source digest already covers it. The API KEY is
    absent too — it identifies a caller, not a generator, and R6 excludes it for the
    same reason. The API base is hashed rather than stored because it can name an
    internal host and this file is meant to be readable and shareable.
    """
    api_base_digest = (
        hashlib.sha256(str(api_base).encode("utf-8")).hexdigest()[:12]
        if api_base
        else ""
    )
    inputs = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "command_name": str(command_name),
        "field_annotations_digest": _canonical_digest(str(field_annotations_text)),
        "field_details_digest": _canonical_digest(field_details),
        "num_examples": num_examples,
        "validation_threshold": validation_threshold,
        "temperature": temperature,
        "model": model,
        "api_base_digest": api_base_digest,
        "completion_backend": completion_backend,
        "generator_source_digest": generator_source_digest,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    variant_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return Fingerprint(
        command_name=str(command_name), variant_key=variant_key, inputs=inputs
    )


class ParamExampleCacheEntry(BaseModel):
    """One command's generated parameter examples at one seed.

    Holds both halves of what `generate_dspy_examples` returns, because both are
    written to `<command>_param_labeled.json` and a cache that reproduced only the
    accepted half would still leave that file differing between two runs.
    """

    seed: int
    created_at: str = ""
    valid_examples: list[dict] = []
    rejected_examples: list[dict] = []

    def is_usable(self) -> bool:
        """A hit must never hand back an empty example set.

        An entry that survives JSON parsing but carries no accepted examples is
        treated as a miss, so the caller regenerates instead of few-shot prompting
        with nothing — the degraded outcome, reached through the cache instead of
        through a truncated response.
        """
        return bool(self.valid_examples)


class ParamExampleCacheFile(BaseModel):
    """On-disk shape: one variant, every seed generated for it."""

    cache_format_version: int = CACHE_FORMAT_VERSION
    command_name: str
    variant_key: str
    fingerprint_inputs: dict = {}
    entries: dict[str, ParamExampleCacheEntry] = {}


class ParamExampleCache:
    """Reads and writes the per-workflow parameter-example cache.

    Every method degrades to "regenerate" rather than raising: a cache is an
    optimisation plus a determinism aid, and it must never be the reason a training
    run fails. Corrupt files are counted and logged, not repaired — the next
    successful generation overwrites them.
    """

    def __init__(self, workflow_folderpath: str, mode: Optional[str] = None) -> None:
        self.workflow_folderpath = workflow_folderpath
        self.mode = normalize_mode(mode) if mode is not None else resolve_cache_mode()
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "hit": 0,
            "miss": 0,
            "stored": 0,
            "unreadable": 0,
            "write_failed": 0,
        }

    # -- configuration -------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Always true: shipped cache policy is exact-seed reuse or regeneration."""
        return True

    @property
    def reads_enabled(self) -> bool:
        """`regenerate` still WRITES: it refreshes the cache rather than bypassing it.

        `reuse` is the only reading mode; see the module docstring for why no cross-seed
        union is defined on parameter examples.
        """
        return self.mode == MODE_REUSE

    # -- paths ---------------------------------------------------------

    @property
    def root(self) -> str:
        """`<workflow>/___command_info/param_example_cache`. Not created by reading."""
        return os.path.join(
            self.workflow_folderpath, COMMAND_INFO_FOLDERNAME, CACHE_DIRNAME
        )

    def entry_path(self, fingerprint: Fingerprint) -> str:
        """Path of the file holding every seed's entry for *fingerprint*."""
        stem = slugify(fingerprint.command_name)
        return os.path.join(self.root, f"{stem}.{fingerprint.variant_key}.json")

    # -- reading -------------------------------------------------------

    def _read_file(self, fingerprint: Fingerprint) -> Optional[ParamExampleCacheFile]:
        """Load a variant file, or None when absent, unreadable, or the wrong shape."""
        path = self.entry_path(fingerprint)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cached = ParamExampleCacheFile(**payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError,
                TypeError) as exc:
            self._bump("unreadable")
            logger.warning(
                f"Ignoring unreadable parameter-example cache entry {path}: {exc}. "
                f"Examples for '{fingerprint.command_name}' will be regenerated."
            )
            return None
        if cached.cache_format_version != CACHE_FORMAT_VERSION:
            logger.info(
                f"Parameter-example cache entry {path} is format v"
                f"{cached.cache_format_version}, this build reads v"
                f"{CACHE_FORMAT_VERSION}; regenerating."
            )
            return None
        if cached.variant_key != fingerprint.variant_key:
            # Only reachable if a file was renamed or hand-edited. The filename is
            # derived from the key, so disagreement means the content is not what the
            # name claims, and trusting it would be exactly the stale-reuse bug.
            logger.warning(
                f"Parameter-example cache entry {path} records variant key "
                f"{cached.variant_key!r} but is filed under "
                f"{fingerprint.variant_key!r}; ignoring it."
            )
            return None
        return cached

    def lookup(
        self, fingerprint: Fingerprint, seed: int
    ) -> Optional[ParamExampleCacheEntry]:
        """Return the entry for exactly (*fingerprint*, *seed*), or None."""
        if not self.reads_enabled:
            return None
        cached = self._read_file(fingerprint)
        if cached is None:
            self._bump("miss")
            return None
        entry = cached.entries.get(str(int(seed)))
        if entry is None or not entry.is_usable():
            self._bump("miss")
            return None
        self._bump("hit")
        return entry

    # -- writing -------------------------------------------------------

    def store(
        self,
        fingerprint: Fingerprint,
        seed: int,
        valid_examples: list[dict],
        rejected_examples: Optional[list[dict]] = None,
    ) -> bool:
        """Write (or replace) the entry for (*fingerprint*, *seed*). Never raises.

        Refuses to store an empty accepted set, so a degraded generation cannot poison
        the cache. Concurrency: the variant file is re-read immediately before writing
        and the other seeds' entries are merged forward, then a uniquely named
        temporary in the same directory is `os.replace`d over the target. `os.replace`
        is atomic on POSIX and Windows, so a concurrent reader sees either the old file
        or the new one, never a half-written one. Two writers racing on the SAME
        variant can still lose one update — the loser regenerates next run, which is
        the safe direction — but neither can corrupt the file.
        """
        if not self.enabled:
            return False
        cleaned = [example for example in (valid_examples or []) if example]
        if not cleaned:
            logger.debug(
                f"Not caching an empty parameter-example set for "
                f"'{fingerprint.command_name}'."
            )
            return False

        entry = ParamExampleCacheEntry(
            seed=int(seed),
            created_at=_utc_now(),
            valid_examples=cleaned,
            rejected_examples=list(rejected_examples or []),
        )

        path = self.entry_path(fingerprint)
        try:
            with self._lock:
                self._ensure_root()
                existing = self._read_file(fingerprint)
                entries = dict(existing.entries) if existing is not None else {}
                entries[str(int(seed))] = entry
                payload = ParamExampleCacheFile(
                    cache_format_version=CACHE_FORMAT_VERSION,
                    command_name=fingerprint.command_name,
                    variant_key=fingerprint.variant_key,
                    fingerprint_inputs=fingerprint.inputs,
                    entries=entries,
                )
                atomic_write_model(path, payload)
        except OSError as exc:
            self._bump("write_failed")
            logger.warning(
                f"Could not write the parameter-example cache entry for "
                f"'{fingerprint.command_name}' to {path}: {exc}. Training continues; "
                f"the next run will regenerate."
            )
            return False
        self._bump("stored")
        return True

    def _ensure_root(self) -> None:
        """Create the cache directory and drop the "this costs money" README."""
        os.makedirs(self.root, exist_ok=True)
        readme = os.path.join(self.root, CACHE_README_FILENAME)
        if not os.path.isfile(readme):
            with open(readme, "w", encoding="utf-8") as f:
                f.write(_CACHE_README)

    # -- reporting -----------------------------------------------------

    def _bump(self, counter: str) -> None:
        self._counters[counter] = self._counters.get(counter, 0) + 1

    @property
    def stats(self) -> dict[str, int]:
        """Counts of hits, misses, stores and failures for this run."""
        return dict(self._counters)

    def format_summary(self) -> str:
        """One line for the end of a training run.

        Deliberately NOT recorded in any artifact, for the same reason R6 keeps its
        summary out of `training_provenance.json`: whether a set of examples came from
        the cache or from the LLM is a property of the run, not of the data, and
        recording it would make two otherwise identical runs produce different files —
        defeating the byte-comparison this cache exists to make possible.
        """
        stats = self.stats
        return (
            f"Parameter-example cache ({self.mode}) at {self.root}: "
            f"{stats['hit']} reused, {stats['miss']} generated, "
            f"{stats['stored']} written"
            + (f", {stats['unreadable']} unreadable" if stats["unreadable"] else "")
            + (
                f", {stats['write_failed']} write failures"
                if stats["write_failed"]
                else ""
            )
        )


_active_cache_lock = threading.Lock()
_active_cache: Optional[ParamExampleCache] = None


def set_param_example_cache(cache: Optional[ParamExampleCache]) -> None:
    """Install (or clear, with None) the cache that generation consults.

    Same shape as `determinism.set_provenance_recorder` and
    `utterance_cache.set_utterance_cache`, and for the same reason:
    `generate_dspy_examples` is reached from a call site that passes only a command
    name and a parameter model, so it cannot be handed a workflow path. The trainer
    installs one for the duration of a run.
    """
    global _active_cache
    with _active_cache_lock:
        _active_cache = cache


def get_param_example_cache() -> Optional[ParamExampleCache]:
    """Return the installed cache, or None when nothing is caching.

    None is normal outside a training run — anything calling
    `generate_dspy_examples` directly gets the unchanged, uncached behaviour.
    """
    with _active_cache_lock:
        return _active_cache
