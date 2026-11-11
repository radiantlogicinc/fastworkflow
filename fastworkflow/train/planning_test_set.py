"""The named test set for planning commands and their callees (Phase 10 step 1, design 2.20).

EXPERIMENT TOOL. It reads a workflow's source and its utterance cache; it trains nothing and makes
no model call. For every planning command P of a workflow it names:

* requests that should route to P (``kind: planning``, ``expected: P``);
* for every execution command C that P's phases call (its callees, from the real steps, as
  ``planning_commands.resolve_field_sources(...).callees`` reads them), requests that should route
  to C and not to P (``kind: callee``, ``expected: C``, ``related: P``).

Each case records its source and its split:

* ``seed`` (split ``train``): the command's hand-written ``Signature.plain_utterances`` and the first
  line of its ``Signature`` docstring. The trainer reads these as rows, so they are reported as
  in-sample and kept out of the headline numbers.
* ``persona-holdout`` (split ``heldout``): utterances already generated in the utterance cache
  (``___command_info/utterance_cache``) whose persona the trainer's persona split holds out. No
  generation runs here; a workflow with no cache has none.
* ``hand-written`` (split ``heldout``): cases from a JSON file (``--handwritten``), for the
  requests the cache cannot supply.

Leakage: a held-out candidate that matches (casefold, whitespace collapsed, or the same dedupe ratio) any
training row of the trained version (``___command_info/training_provenance.json``, every command's
``utterance_personas``) is dropped and recorded, so the persona split's own assumptions never reach the set.

Per target, the heldout cases are filled to ``per_command`` in source order: persona-holdout first,
then hand-written. A target short of that is reported with its shortfall, not padded.

Subcommands::

    python -m fastworkflow.train.planning_test_set build --workflow W --out O [--handwritten H]
        [--per-command N] [--cache-root R]
    python -m fastworkflow.train.planning_test_set build --workflow W --dry-run

``--dry-run`` prints the counts and writes nothing.

Hand-written file: ``{"cases": [{"utterance": "...", "expected": "Book/loan_report"}, ...]}``.
``expected`` is a planning command or a callee of one; ``related`` (optional) names the planning
command a callee case is about.
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = 1

KIND_PLANNING = "planning"
KIND_CALLEE = "callee"
SOURCE_SEED = "seed"
SOURCE_PERSONA = "persona-holdout"
SOURCE_HANDWRITTEN = "hand-written"
SPLIT_TRAIN = "train"
SPLIT_HELDOUT = "heldout"
GLOBAL_HEAD = "*"

#: Heldout requests per target when ``--per-command`` is not given.
DEFAULT_PER_COMMAND = 8

#: Where the trainer keeps generated utterances, relative to the workflow folder.
CACHE_SUBDIR = os.path.join("___command_info", "utterance_cache")

#: A held-out request is dropped when its normalised text equals, or is at least this similar
#: (difflib ratio) to, a seed of any command or an already-kept held-out request.
DEDUPE_RATIO = 0.9


@dataclass
class Inventory:
    """A workflow's planning commands, their executable callees and their signatures."""

    workflow_folderpath: str
    planning: list[str]
    #: planning command -> its execution callees (sorted); planning callees are not here.
    callees: dict[str, list[str]]
    #: planning command -> callees that are themselves planning commands (not scored as callees).
    planning_callees: dict[str, list[str]]
    #: every qualified command name the workflow's signatures hold.
    signatures: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_callees(self) -> list[str]:
        return sorted({c for names in self.callees.values() for c in names})

    def callers_of(self, callee: str) -> list[str]:
        return sorted(p for p, names in self.callees.items() if callee in names)


def scope_of(name: str) -> str:
    """The head a command is scored through: its context, or ``*`` for a root command."""
    return name.rpartition("/")[0] or GLOBAL_HEAD


def inventory(workflow_folderpath: str) -> Inventory:
    """Read the workflow's planning commands and callees from the loader (no writes, no model)."""
    from fastworkflow.planning_commands import resolve_field_sources, visible_planning_commands

    resolution = resolve_field_sources(workflow_folderpath)
    planning = sorted(visible_planning_commands(workflow_folderpath))
    planning_set = set(planning)
    callees: dict[str, list[str]] = {}
    planning_callees: dict[str, list[str]] = {}
    for name in planning:
        found = sorted(resolution.callees.get(name, frozenset()))
        callees[name] = [c for c in found if c not in planning_set]
        planning_callees[name] = [c for c in found if c in planning_set]
    return Inventory(
        workflow_folderpath=workflow_folderpath,
        planning=planning,
        callees=callees,
        planning_callees=planning_callees,
        signatures=dict(resolution.signatures),
    )


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def normalise(text: Any) -> str:
    """Casefolded, whitespace collapsed: the form the dedupe compares."""
    return " ".join(str(text).casefold().split())


class HeldoutDeduper:
    """Drops a held-out candidate that duplicates a seed of any command, or an earlier kept held-out case.

    *seeds* is ``(text, owner)`` for every seed of every command in the workflow (plain utterances
    and the docstring's first line). A candidate is dropped when its normalised text equals one, or
    has a difflib ratio of at least ``DEDUPE_RATIO`` against one, the same test against kept
    held-out cases. Kept cases are checked in build order, so the first of a near-duplicate pair
    stays. Every drop is recorded with its reason in ``dropped``.
    """

    def __init__(self, seeds: Sequence[tuple[str, str]]):
        self._seeds = [(normalise(text), owner) for text, owner in seeds if normalise(text)]
        self._kept: list[tuple[str, str]] = []  # (normalised text, expected)
        self.dropped: list[dict] = []

    def check(self, utterance: str, expected: str, source: str) -> bool:
        """True when the candidate is kept (and remembered); False when it was dropped (and recorded)."""
        norm = normalise(utterance)
        reason = self._reason(norm)
        if reason is None:
            self._kept.append((norm, expected))
            return True
        self.dropped.append({"utterance": utterance, "expected": expected, "source": source, "reason": reason})
        return False

    def _reason(self, norm: str) -> Optional[str]:
        for text, owner in self._seeds:
            if norm == text:
                return f"equals a seed or trained row of {owner}"
        best_ratio, best_owner = 0.0, None
        for text, owner in self._seeds:
            ratio = difflib.SequenceMatcher(None, norm, text).ratio()
            if ratio >= DEDUPE_RATIO and ratio > best_ratio:
                best_ratio, best_owner = ratio, owner
        if best_owner is not None:
            return f"near a seed or trained row of {best_owner} (ratio {best_ratio:.2f})"
        for text, owner in self._kept:
            if norm == text:
                return f"equals a kept held-out request of {owner}"
        best_ratio, best_owner = 0.0, None
        for text, owner in self._kept:
            ratio = difflib.SequenceMatcher(None, norm, text).ratio()
            if ratio >= DEDUPE_RATIO and ratio > best_ratio:
                best_ratio, best_owner = ratio, owner
        if best_owner is not None:
            return f"near a kept held-out request of {best_owner} (ratio {best_ratio:.2f})"
        return None


def seed_cases_for(inv: Inventory, name: str) -> list[tuple[str, str]]:
    """``(utterance, origin)`` of a command's seed text: its plain utterances, then its docstring's first line."""
    signature = inv.signatures.get(name)
    out: list[tuple[str, str]] = []
    if signature is None:
        return out
    for text in getattr(signature, "plain_utterances", ()) or ():
        cleaned = _clean(text)
        if cleaned:
            out.append((cleaned, "plain_utterances"))
    doc = inspect.getdoc(signature) or ""
    first = _clean(doc.splitlines()[0]) if doc.strip() else ""
    if first:
        out.append((first, "docstring"))
    return out


def default_cache_root(workflow_folderpath: str) -> str:
    return os.path.join(workflow_folderpath, CACHE_SUBDIR)


def persona_holdout(cache_root: str, names: Sequence[str], seed: Optional[int] = None) -> tuple[dict[str, list[str]], list[str]]:
    """``({command: held-out generated utterances}, notes)`` from the utterance cache. Never generates.

    Every command in *names* that has a cache file contributes its newest variant's generated
    utterances, each with the persona that produced it. One persona split (``split_by_persona``,
    the trainer's seed) runs over all of them together, as the trainer's split does over a whole
    workflow; the held-out rows are the persona-holdout source. Per-persona batch alignment is not
    read here, so the split is reported as unaligned in the notes.
    """
    from fastworkflow.train import heldout_evaluation
    from fastworkflow.train.utterance_cache import stored_variants

    notes: list[str] = []
    if not os.path.isdir(cache_root):
        notes.append("no utterance cache: persona-holdout source is empty")
        return {}, notes
    records: list[heldout_evaluation.LabeledUtterance] = []
    for name in names:
        variants = stored_variants(cache_root, name)
        if not variants:
            continue
        newest_path, payload = max(variants, key=lambda item: os.path.getmtime(item[0]))
        seen: set[str] = set()
        for entry in (payload.get("entries") or {}).values():
            personas = entry.get("utterance_personas") or {}
            for utterance in entry.get("generated_utterances") or []:
                text = _clean(utterance)
                persona = personas.get(utterance)
                if not text or persona is None or text in seen:
                    continue
                seen.add(text)
                records.append(heldout_evaluation.LabeledUtterance(
                    utterance=text, label=name, persona=str(persona)))
    if not records:
        notes.append("utterance cache holds no persona-tagged rows for these commands")
        return {}, notes
    split = heldout_evaluation.split_by_persona(
        records, seed=heldout_evaluation.DEFAULT_SEED if seed is None else seed, persona_batches=None)
    notes.append("persona split is unaligned with generation batches (persona_batches not read)")
    notes.extend(f"persona split: {note}" for note in split.notes)
    held: dict[str, list[str]] = {}
    for record in split.heldout:
        held.setdefault(record.label, []).append(record.utterance)
    return {name: sorted(texts) for name, texts in held.items()}, notes


def read_handwritten(path: str) -> list[dict]:
    """The hand-written cases of a JSON file: ``[{"utterance", "expected", "related"?}]``."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected an object with a 'cases' list")
    out = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not _clean(row.get("utterance", "")) or not row.get("expected"):
            raise ValueError(f"{path}: case {index} needs 'utterance' and 'expected'")
        item = {"utterance": _clean(row["utterance"]), "expected": str(row["expected"])}
        if row.get("related"):
            item["related"] = str(row["related"])
        out.append(item)
    return out


def _check_handwritten(rows: list[dict], inv: Inventory) -> list[dict]:
    """Refuse a hand-written case whose expected command is not a planning command or a callee of one."""
    planning = set(inv.planning)
    callees = set(inv.all_callees)
    checked = []
    for row in rows:
        expected = row["expected"]
        if expected in planning:
            kind, related = KIND_PLANNING, expected
        elif expected in callees:
            kind = KIND_CALLEE
            related = row.get("related") or inv.callers_of(expected)[0]
            if related not in planning or expected not in inv.callees.get(related, []):
                raise ValueError(f"hand-written case {row['utterance']!r}: {related!r} does not call {expected!r}")
        else:
            raise ValueError(
                f"hand-written case {row['utterance']!r}: {expected!r} is neither a planning command "
                "nor a callee of one")
        checked.append({**row, "kind": kind, "related": related})
    return checked


def _case(utterance: str, expected: str, kind: str, related: str, source: str, split: str,
          origin: Optional[str] = None) -> dict:
    case = {"utterance": utterance, "expected": expected, "kind": kind, "related": related,
            "source": source, "split": split, "head": scope_of(expected)}
    if origin:
        case["origin"] = origin
    return case


def all_seed_texts(inv: Inventory) -> list[tuple[str, str]]:
    """``(text, owner)`` for every seed of every command the workflow's signatures hold, and every
    training row of the trained version (see ``trained_rows``)."""
    out: list[tuple[str, str]] = []
    for name in sorted(inv.signatures):
        for text, _origin in seed_cases_for(inv, name):
            out.append((text, name))
    out.extend(trained_rows(inv.workflow_folderpath))
    return out


def trained_rows(workflow_folderpath: str) -> list[tuple[str, str]]:
    """``(text, owner)`` for every utterance the trained version trained on, from its provenance.

    The trainer's own split is not re-derivable here: its training rows are every row of
    ``training_provenance.json`` (seeds and generated, for every command). A held-out case that
    matches one of them is leakage, whichever split the builder believed it came from. Empty when the
    workflow has not been trained.
    """
    path = os.path.join(workflow_folderpath, "___command_info", "training_provenance.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        commands = json.load(fh).get("commands") or {}
    out: list[tuple[str, str]] = []
    for name, entry in sorted(commands.items()):
        for utterance in entry.get("utterance_personas") or {}:
            out.append((utterance, name))
    return out


def _target_cases(inv: Inventory, name: str, kind: str, related: str, persona: Mapping[str, list[str]],
                  hand: Sequence[dict], per_command: int, dedupe: HeldoutDeduper) -> tuple[list[dict], dict]:
    cases: list[dict] = []
    seen: set[str] = set()
    in_sample = 0
    for utterance, origin in seed_cases_for(inv, name):
        if utterance in seen:
            continue
        seen.add(utterance)
        cases.append(_case(utterance, name, kind, related, SOURCE_SEED, SPLIT_TRAIN, origin))
        in_sample += 1
    heldout = 0
    for utterance in persona.get(name, []):
        if heldout >= per_command:
            break
        if not dedupe.check(utterance, name, SOURCE_PERSONA):
            continue
        if utterance in seen:
            continue
        seen.add(utterance)
        cases.append(_case(utterance, name, kind, related, SOURCE_PERSONA, SPLIT_HELDOUT))
        heldout += 1
    for row in hand:
        if heldout >= per_command:
            break
        if row["expected"] != name or row["utterance"] in seen:
            continue
        if not dedupe.check(row["utterance"], name, SOURCE_HANDWRITTEN):
            continue
        seen.add(row["utterance"])
        cases.append(_case(row["utterance"], name, kind, related, SOURCE_HANDWRITTEN, SPLIT_HELDOUT))
        heldout += 1
    summary = {"kind": kind, "related": related, "in_sample": in_sample, "heldout": heldout,
               "shortfall": max(0, per_command - heldout)}
    return cases, summary


def build_test_set(
    workflow_folderpath: str,
    *,
    per_command: int = DEFAULT_PER_COMMAND,
    handwritten_path: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> dict:
    """The named test set of one workflow (see the module docstring). Pure apart from the reads."""
    if per_command < 1:
        raise ValueError("per_command must be at least 1")
    inv = inventory(workflow_folderpath)
    hand = _check_handwritten(read_handwritten(handwritten_path), inv) if handwritten_path else []
    # The split runs over every command of the workflow, as the trainer's split does, so the
    # personas it holds out are the trainer's own (the same records, the same seed).
    names = sorted(inv.signatures)
    persona, notes = persona_holdout(cache_root or default_cache_root(workflow_folderpath), names)

    cases: list[dict] = []
    targets: dict[str, dict] = {}
    dedupe = HeldoutDeduper(all_seed_texts(inv))
    for name in inv.planning:
        found, summary = _target_cases(inv, name, KIND_PLANNING, name, persona, hand, per_command, dedupe)
        cases.extend(found)
        targets[name] = {**summary, "callees": inv.callees[name]}
    for callee in inv.all_callees:
        related = inv.callers_of(callee)[0]
        found, summary = _target_cases(inv, callee, KIND_CALLEE, related, persona, hand, per_command, dedupe)
        cases.extend(found)
        targets[callee] = {**summary, "callers": inv.callers_of(callee)}
    if dedupe.dropped:
        by_reason = Counter(d["reason"].split(" (")[0] for d in dedupe.dropped)
        notes.append(f"dedupe (ratio >= {DEDUPE_RATIO}, casefold, whitespace collapsed): dropped "
                     f"{len(dedupe.dropped)} held-out candidate(s) before filling: "
                     + "; ".join(f"{count} {reason}" for reason, count in sorted(by_reason.items())))
    for name in inv.planning:
        for other in inv.planning_callees[name]:
            notes.append(f"{name} calls planning command {other}: it is not scored as a callee")
    short = [name for name, block in targets.items() if block["shortfall"]]
    if short:
        notes.append(f"{len(short)} target(s) short of {per_command} heldout requests: {', '.join(short)}")
    return {
        "schema": SCHEMA_VERSION,
        "workflow": os.path.basename(os.path.normpath(workflow_folderpath)),
        "per_command": per_command,
        "planning_commands": inv.planning,
        "targets": targets,
        "cases": cases,
        "dropped": dedupe.dropped,
        "counts": count_cases(cases),
        "notes": notes,
    }


def count_cases(cases: Sequence[dict]) -> dict[str, Any]:
    """Counts by kind, source and split, and the heldout cases per kind."""
    return {
        "total": len(cases),
        "by_kind": dict(Counter(c["kind"] for c in cases)),
        "by_source": dict(Counter(c["source"] for c in cases)),
        "by_split": dict(Counter(c["split"] for c in cases)),
        "heldout_by_kind": dict(Counter(c["kind"] for c in cases if c["split"] == SPLIT_HELDOUT)),
    }


def heldout(test_set: Mapping[str, Any]) -> list[dict]:
    """The heldout cases of a test set (the ones the headline numbers are scored on)."""
    return [case for case in test_set["cases"] if case["split"] == SPLIT_HELDOUT]


def load_test_set(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path}: test-set schema {document.get('schema')!r}, expected {SCHEMA_VERSION}")
    return document


def format_counts(test_set: Mapping[str, Any]) -> str:
    lines = [f"Planning test set for {test_set['workflow']} (per command: {test_set['per_command']})", ""]
    lines.append(f"{'target':<32}{'kind':<10}{'in-sample':>10}{'heldout':>9}{'short':>7}  callers/callees")
    for name, block in sorted(test_set["targets"].items()):
        links = block.get("callees") if block["kind"] == KIND_PLANNING else block.get("callers")
        lines.append(f"{name:<32}{block['kind']:<10}{block['in_sample']:>10}{block['heldout']:>9}"
                     f"{block['shortfall']:>7}  {', '.join(links or [])}")
    counts = test_set["counts"]
    lines.append("")
    lines.append(f"cases: {counts['total']}  by kind {counts['by_kind']}  by source {counts['by_source']}")
    lines.append(f"heldout by kind: {counts['heldout_by_kind']}")
    for note in test_set["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _write(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="planning_test_set", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a workflow's planning test set")
    build.add_argument("--workflow", required=True, help="the workflow folder")
    build.add_argument("--out", default=None, help="where to write the set (JSON)")
    build.add_argument("--handwritten", default=None, help="hand-written cases (JSON)")
    build.add_argument("--per-command", type=int, default=DEFAULT_PER_COMMAND)
    build.add_argument("--cache-root", default=None, help="utterance cache (default: the workflow's)")
    build.add_argument("--dry-run", action="store_true", help="print the counts and write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    import fastworkflow

    fastworkflow.init({})
    test_set = build_test_set(
        args.workflow,
        per_command=args.per_command,
        handwritten_path=args.handwritten,
        cache_root=args.cache_root,
    )
    print(format_counts(test_set))
    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    _write(args.out, test_set)
    print(f"\nwrote {len(test_set['cases'])} cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
