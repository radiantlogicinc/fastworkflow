"""Offline measurement of the run-listing arms (Phase 10 step 4, gate 10.7; design 3.10, 2.40, 2.50).

EXPERIMENT TOOL. It makes no model call and trains nothing. For each workflow it copies the source folder to a
scratch directory, opens the copy as the agent's app workflow (the same way ``tests/test_run_listing.py`` does) and
builds, for every request of a named test set:

* the FULL listing (``workflow_agent.planner_commands_text``, today's run listing);
* the COMMAND_SHORTLIST listing (``run_listing.command_shortlist_text`` over ``run_listing.shortlist_for``), whichever
  ``FULL_LISTING_MAX_CHARS`` threshold is applied afterwards: the production policy gives the full listing whole while
  it is at most the threshold, so the threshold is applied here to the cached texts instead of by mutating the
  module constant. (The BY_CONTEXT arm was removed under fix-3rsq.10.15.)

Two inputs to gathering are bracketed: the planning-command match's distribution is ``None`` (no Jev), or the
request's gold planning command at 0.9 (what a perfect Jev would give; a callee request gets ``{"none": 0.9}``).

Metrics per (workflow, planning-distribution mode, arm):

* shortlist recall: the share of requests whose gold commands are ALL on the listing. "On the listing" is strict:
  the command has its own entry (a ``- name`` line in a section that owns it, or a global line). The one-line
  context summaries (``- Book: ... commands: a, b``) do not count. ``recall_named`` is the looser reading (the
  command's name appears anywhere in the text);
* listing size in characters and estimated tokens (chars / 4): mean and 95th percentile;
* whether FULL_LISTING_MAX_CHARS makes the full listing apply to the request.

Statistics: Wilson 95% intervals on every rate, and exact paired McNemar tests on recall between arms.

Threshold sweep: the same cached texts are reused for each candidate FULL_LISTING_MAX_CHARS; a request gets the
full listing when the full listing's characters are at most the candidate.

Live part: NOT here. ``estimate`` prints the calls and tokens a minimal end-to-end plan-success run would spend, per
arm, for a stratified sample. It is arithmetic on measured sizes; the owner approves the live run from its output.

Usage::

    python -m fastworkflow.train.score_listing offline --out DIR [--app-workflow P --app-testset T]
    python -m fastworkflow.train.score_listing sweep   --out DIR [--app-workflow P --app-testset T] [--thresholds ...]
    python -m fastworkflow.train.score_listing estimate --out DIR [--app-workflow P --app-testset T]

Without ``--app-*`` only the committed workflows run (library, the no-hints tutorial and todo copies). The app's
paths are arguments only; nothing here names it.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = 1
REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
TEST_SETS = TESTS / "planning_test_sets"

#: The planning-command match distribution the shortlist is gathered with (see the module docstring).
MODE_NONE = "none"
MODE_GOLD = "gold_0.9"
#: DIAGNOSTIC CEILING, not a deployable arm: gathering is given the gold commands as its seeds, so the recall it
#: reports is what expansion and listing lose on their own. Production gathering can only do worse than this.
MODE_ORACLE = "oracle_seeds"
MODES = (MODE_NONE, MODE_GOLD, MODE_ORACLE)
#: REALISTIC GATHERING (needs trained heads): the per-scope head answers (CommandRouter.predict_with_details, the
#: labels the run gets: the confident label, else the top-k) for every context gather visits, plus the exact-name
#: mentions and the planning candidates of the heads' planning-label top-k scores. Read from the source folder's
#: trained version; nothing is written. The planning scores are the heads', not Jev's (a different model).
MODE_HEADS = "heads"
#: The same, with the request's gold planning command added at 0.9 (what a perfect Jev would add).
MODE_HEADS_GOLD = "heads+gold_0.9"
HEAD_MODES = (MODE_HEADS, MODE_HEADS_GOLD)
#: THE PRODUCTION GATHER (run_listing.shortlist_live), with no planning distribution: the heads are read live from the
#: source's ___command_info (read only, the same models the production path loads). Its recall is the production
#: shortlist's recall without Jev; it must equal MODE_HEADS when the source's current version and ___command_info agree.
MODE_PRODUCTION = "production"
ALL_MODES = MODES + HEAD_MODES + (MODE_PRODUCTION,)
#: Labels a head can answer that are not commands.
NON_COMMAND_LABELS = ("parameter_value", "wildcard")
GOLD_PROBABILITY = 0.9
NONE_OPTION = "none"

#: The sweep of FULL_LISTING_MAX_CHARS candidates (inclusive).
DEFAULT_THRESHOLDS = tuple(range(3000, 12001, 500))

#: Stratified live sample: requests per planning command (and per callee of it), by workflow label.
LIVE_PER_COMMAND = {"library": 3, "app": 1}
SAMPLE_SEED = 1017

#: Heuristic for the live estimate (labelled as assumptions in its output; change them on the command line).
DEFAULT_CALLS_PER_REQUEST = 3
DEFAULT_OVERHEAD_TOKENS = 1500
DEFAULT_OUTPUT_TOKENS = 300

_COPY_IGNORE = shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__")

ARM_NAMES = ("full", "command_shortlist")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson score interval for a proportion (95% by default). ``(0.0, 0.0)`` when n is 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs: ``a_only`` = arm A hit and B missed, and the reverse."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def percentile95(values: Sequence[float]) -> float:
    """The nearest-rank 95th percentile (0.0 for no values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return float(ordered[max(0, rank - 1)])


# ---------------------------------------------------------------------------
# Reading a listing: which commands does it show?
# ---------------------------------------------------------------------------

_HEADERS = (
    (re.compile(r"^Commands available in the current context \("), "active"),
    (re.compile(r"^Commands available here from the global scope"), "*"),
    (re.compile(r"^Commands available here from the (\S+) context"), 1),
    (re.compile(r"^Commands available after entering the (\S+) context"), 1),
)
_GLOBAL_LINE = re.compile(r"^Global commands \(.*?\): (.*)$")
_OTHER_CONTEXTS = re.compile(r"^Other contexts\b")


def _qualify(owner: str, leaf: str) -> str:
    return leaf if owner in ("*", "") else f"{owner}/{leaf}"


_ENTRY = re.compile(r"^- ([A-Za-z_]\w*)\s*$")


def listed_entries(text: str, active: str) -> frozenset[str]:
    """The qualified commands the listing gives an entry to.

    An entry is a ``- name`` line (a bare identifier, no indent) in a section that owns the command: the active
    context's section, a scope section (global or on the lookup path), or a deferred context's section. The global
    line names its commands. A context summary line (``- Book: ...``) is not an entry. Other unindented lines are
    either a section header (matched by its text, an unknown header raises) or a continuation of a command's
    description, which is ignored: a description may wrap onto an unindented line.
    """
    entries: set[str] = set()
    owner: str | None = None
    for line in text.split("\n"):
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        if m := _ENTRY.match(line):
            if owner not in (None, "__other__"):
                entries.add(_qualify(owner, m.group(1)))
            continue
        if line.startswith("- "):
            continue  # a context summary line ("- Book: ...") or a wrapped description that starts with a dash
        matched = False
        for pattern, target in _HEADERS:
            m = pattern.match(line)
            if m:
                owner = active if target == "active" else target if isinstance(target, str) else m.group(target)
                matched = True
                break
        if matched:
            continue
        if m := _GLOBAL_LINE.match(line):
            entries.update(name.strip() for name in m.group(1).rstrip(".").split(", ") if name.strip())
            owner = None
            continue
        if _OTHER_CONTEXTS.match(line):
            owner = "__other__"
            continue
        if line.startswith(("Commands available", "Global commands", "Other contexts")):
            raise ValueError(f"an unrecognised listing header: {line[:80]!r}")
        # any other unindented text is a wrapped description line
    return frozenset(entries)


def named_in(text: str, qualified: Iterable[str]) -> frozenset[str]:
    """The qualified commands whose leaf name appears anywhere in the text (the looser reading)."""
    found = set()
    for name in qualified:
        leaf = name.split("/")[-1]
        if re.search(rf"(?<!\w){re.escape(leaf)}(?!\w)", text):
            found.add(name)
    return frozenset(found)


# ---------------------------------------------------------------------------
# Cases: requests with their gold commands
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    utterance: str
    kind: str          # "planning" (gold = the planning command and its callees) or "callee" (gold = the callee)
    expected: str      # the command the request is about
    gold: frozenset[str]
    source: str = ""


def gold_for(expected: str, kind: str, callees: Mapping[str, Sequence[str]]) -> frozenset[str]:
    if kind == "planning":
        return frozenset({expected, *callees.get(expected, ())})
    return frozenset({expected})


def load_test_set(path: Path, *, splits: Sequence[str] = ("heldout",)) -> list[Request]:
    """Requests from a planning test set file (the format ``fastworkflow.train.planning_test_set`` writes).

    Only the given splits are read; the headline numbers use ``heldout`` (train rows are in-sample).
    Gold: a planning case's expected command and its execution callees (``targets[P]["callees"]``); a callee
    case's expected command alone.
    """
    data = json.loads(Path(path).read_text())
    callees = {name: list(t["callees"]) for name, t in data.get("targets", {}).items() if "callees" in t}
    requests = []
    for case in data["cases"]:
        if case.get("split") not in splits:
            continue
        kind = case["kind"]
        requests.append(Request(
            utterance=case["utterance"], kind=kind, expected=case["expected"],
            gold=gold_for(case["expected"], kind, callees), source=case.get("source", "")))
    return requests


def _inventory_of(folder: str | Path):
    """The planning inventory of a workflow, computed on a scratch COPY (the loader can rewrite a workflow's derived
    snapshots, and other agents run on the same tree, so the repo's workflow folders are never touched)."""
    from fastworkflow.train import planning_test_set

    scratch = Path(tempfile.mkdtemp(prefix="score_listing_inv_"))
    copy = scratch / Path(folder).name
    shutil.copytree(folder, copy, ignore=_COPY_IGNORE, symlinks=True)
    return planning_test_set.inventory(str(copy))


def load_handwritten(path: Path, folder: str) -> list[Request]:
    """Requests from a hand-written file (``{"cases": [{"utterance", "expected"}]}``) on the workflow at *folder*.

    The kind and gold come from the workflow's own planning inventory: an expected planning command is a planning
    request with its callees; anything else is a callee-style request for that command alone.
    """
    from fastworkflow.train import planning_test_set

    inv = _inventory_of(folder)
    planning = set(inv.planning)
    callees = {name: list(inv.callees.get(name, [])) for name in inv.planning}
    requests = []
    for case in json.loads(Path(path).read_text())["cases"]:
        kind = "planning" if case["expected"] in planning else "callee"
        requests.append(Request(utterance=case["utterance"], kind=kind, expected=case["expected"],
                                gold=gold_for(case["expected"], kind, callees), source="hand-written"))
    return requests


def planning_distribution(mode: str, request: Request) -> dict[str, float] | None:
    """The planning-command match's distribution under *mode* (None, or the gold command at 0.9)."""
    if mode in (MODE_NONE, MODE_ORACLE):
        return None
    if mode == MODE_GOLD:
        return {request.expected: GOLD_PROBABILITY} if request.kind == "planning" else {NONE_OPTION: GOLD_PROBABILITY}
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Opening a workflow copy (as tests/test_run_listing.py does)
# ---------------------------------------------------------------------------

@dataclass
class OpenedWorkflow:
    label: str
    app: Any
    ctx: Any
    root_context: str
    #: context -> the contexts it inherits from (transitively), from the workflow's inheritance declaration
    bases: Mapping[str, frozenset[str]] = None  # type: ignore[assignment]
    #: the qualified planning command names
    planning: frozenset[str] = frozenset()
    #: head name -> utterance -> the head's answer (None when the workflow has no trained heads)
    heads: Mapping[str, Mapping[str, dict]] | None = None
    #: the SOURCE workflow's ___command_info (read only; the production gather reads its heads), None when absent
    command_info_root: str | None = None

    def close(self) -> None:
        """Nothing to release: the listing builders need no execution context (it would also require the internal
        workflow's construction record, which a stale tree fails before any listing is built)."""


def inheritance_bases(folder: Path) -> dict[str, frozenset[str]]:
    """Each context's transitive base contexts (``_commands/context_inheritance_model.json``). A base's commands are
    listed under each context that inherits them (Book and Member list Record's commands)."""
    path = Path(folder) / "_commands" / "context_inheritance_model.json"
    raw = json.loads(path.read_text()) if path.is_file() else {}

    def walk(name: str, seen: frozenset[str]) -> set[str]:
        out: set[str] = set()
        for base in raw.get(name, {}).get("base", []):
            if base not in seen:
                out.add(base)
                out |= walk(base, seen | {base})
        return out

    return {name: frozenset(walk(name, frozenset({name}))) for name in raw}


def covered(gold: frozenset[str], entries: frozenset[str], bases: Mapping[str, frozenset[str]]) -> bool:
    """Every gold command has an entry on the listing, or an entry with its leaf name in its own context or in a
    context that inherits it. The second reading is what the agent can call: after the context is entered, the
    inherited command is listed under the inheriting context (Book/show_notes for Record/show_notes)."""
    by_leaf: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        ctx, _, leaf = entry.rpartition("/")
        by_leaf[leaf].append(ctx)
    for name in gold:
        if name in entries:
            continue
        ctx, _, leaf = name.rpartition("/")
        if not any(c == ctx or ctx in bases.get(c, ()) for c in by_leaf.get(leaf, ())):
            return False
    return True


def open_workflow(source: Path, label: str, scratch: Path, *, context_path: str | None = None) -> OpenedWorkflow:
    """Copy *source* into *scratch* and open the copy as the agent's app workflow.

    ``context_path`` is ``module:Class`` for a root context object the workflow needs entered (the library's Library
    context); otherwise the workflow's own root is used.
    """
    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry

    folder = scratch / label
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(source, folder, ignore=_COPY_IGNORE, symlinks=True)
    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"listing-{label}")
    ctx = None
    if context_path:
        module_name, _, class_name = context_path.partition(":")
        root = getattr(importlib.import_module(module_name), class_name)()
        app.root_command_context = root
        app.current_command_context = root
    from fastworkflow import planning_commands

    planning = frozenset(planning_commands.visible_planning_commands(str(folder)))
    info = Path(source) / "___command_info"
    return OpenedWorkflow(label=label, app=app, ctx=ctx, root_context=app.current_command_context_name,
                          bases=inheritance_bases(folder), planning=planning,
                          command_info_root=str(info) if info.is_dir() else None)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Built:
    """The listings one request gets from one mode, before the threshold is applied."""
    full: str
    command_shortlist: str
    shortlist: tuple[str, ...]


def gather_contexts(opened: OpenedWorkflow) -> list[str]:
    """The contexts gather visits: the lookup-path scopes, then every context reachable by descend edges."""
    from fastworkflow.context_navigation import exact_name_scopes
    from fastworkflow.run_listing import _reachable_contexts

    scopes = [scope.context_name for scope in exact_name_scopes(opened.app)]
    return list(dict.fromkeys([*scopes, *_reachable_contexts(opened.app.folderpath, opened.app.current_command_context_name)]))


def head_seeds(opened: OpenedWorkflow, utterance: str) -> tuple[set[str], dict[str, float]]:
    """The labels the trained heads answer for *utterance* on the gather contexts, and the heads' planning-label
    scores (the top-k each head returned). A context with no trained head contributes nothing."""
    labels: set[str] = set()
    planning_scores: dict[str, float] = {}
    for context in gather_contexts(opened):
        # discover_heads keys the global head by GLOBAL ("*"), the same name the gather contexts use.
        answer = (opened.heads or {}).get(context, {}).get(utterance)
        if answer is None:
            continue
        labels.update(label for label in answer["labels"] if label not in NON_COMMAND_LABELS)
        for label, score in zip(answer["topk_labels"], answer["topk_scores"]):
            if label in opened.planning:
                planning_scores[label] = max(planning_scores.get(label, 0.0), float(score))
    return labels, planning_scores


def build_listings(opened: OpenedWorkflow, request: Request, mode: str) -> Built:
    from fastworkflow import run_listing
    from fastworkflow import workflow_agent

    app = opened.app
    if mode in HEAD_MODES:
        if opened.heads is None:
            raise ValueError(f"mode {mode!r} needs the workflow's trained heads")
        from fastworkflow.run_listing import gather, expand, planning_candidates
        labels, planning_scores = head_seeds(opened, request.utterance)
        if mode == MODE_HEADS_GOLD:
            for name, probability in (planning_distribution(MODE_GOLD, request) or {}).items():
                planning_scores[name] = max(planning_scores.get(name, 0.0), probability)
        seeds = set(gather(app, request.utterance)) | labels
        confident, unsure = planning_candidates(planning_scores)
        seeds.update(name for name in (confident, *unsure) if name)
        names = expand(app, seeds)
    elif mode == MODE_PRODUCTION:
        if opened.command_info_root is None:
            raise ValueError(f"mode {mode!r} needs the source workflow's ___command_info (not found for {opened.label})")
        names = run_listing.shortlist_live(app, request.utterance, None, command_info_root=opened.command_info_root)
    elif mode == MODE_ORACLE:
        names = run_listing.expand(app, request.gold)
    else:
        names = run_listing.shortlist_for(app, request.utterance, planning_distribution(mode, request))
    return Built(
        full=workflow_agent.planner_commands_text(app),
        command_shortlist=run_listing.command_shortlist_text(app, names),
        shortlist=tuple(sorted(names)),
    )


def arm_text(built: Built, arm: str, full_chars: int, threshold: int) -> str:
    """The listing an arm shows at a threshold: the full listing whole when it is at most the threshold."""
    if arm == "full" or full_chars <= threshold:
        return built.full
    return built.command_shortlist


def tokens_of(chars: int) -> float:
    return chars / 4.0


@dataclass
class RowResult:
    index: int
    kind: str
    mode: str
    arm: str
    #: the primary recall: every gold command is listed, or reachable through inheritance (what the agent can call)
    hit_entry: bool
    #: the strict reading: every gold command has its own entry in a section of its own context
    hit_strict: bool
    hit_named: bool
    chars: int
    full_applies: bool


def measure_requests(opened: OpenedWorkflow, requests: Sequence[Request], threshold: int,
                     cache: dict | None = None, modes: Sequence[str] = MODES) -> tuple[list[RowResult], dict[str, Any]]:
    """One row per (request, mode, arm) at *threshold*, plus the per-request built texts (cached in *cache*)."""
    cache = {} if cache is None else cache
    full_entries = listed_entries(workflow_full(opened), opened.root_context)
    rows: list[RowResult] = []
    for index, request in enumerate(requests):
        for mode in modes:
            key = (opened.label, request.utterance, mode)
            if key not in cache:
                cache[key] = build_listings(opened, request, mode)
            built = cache[key]
            full_chars = len(built.full)
            for arm in ARM_NAMES:
                text = arm_text(built, arm, full_chars, threshold)
                entries = full_entries if arm == "full" else listed_entries(text, opened.root_context)
                gold = request.gold
                rows.append(RowResult(
                    index=index, kind=request.kind, mode=mode, arm=arm,
                    hit_entry=covered(gold, entries, opened.bases),
                    hit_strict=gold <= entries,
                    hit_named=gold <= named_in(text, gold),
                    chars=len(text), full_applies=full_chars <= threshold))
    return rows, cache


def workflow_full(opened: OpenedWorkflow) -> str:
    from fastworkflow import workflow_agent

    return workflow_agent.planner_commands_text(opened.app)


def summarise(rows: Sequence[RowResult], requests: Sequence[Request]) -> dict[str, Any]:
    """Per mode and arm: recall (primary and strict) with Wilson intervals, size, and McNemar between arms."""
    out: dict[str, Any] = {}
    present = [mode for mode in ALL_MODES if any(r.mode == mode for r in rows)]
    for mode in present:
        out[mode] = {}
        per_arm: dict[str, list[RowResult]] = {arm: [] for arm in ARM_NAMES}
        for row in rows:
            if row.mode == mode:
                per_arm[row.arm].append(row)
        for arm in ARM_NAMES:
            arm_rows = sorted(per_arm[arm], key=lambda r: r.index)
            n = len(arm_rows)
            hits = sum(r.hit_entry for r in arm_rows)
            lo, hi = wilson(hits, n)
            strict = sum(r.hit_strict for r in arm_rows)
            slo, shi = wilson(strict, n)
            named = sum(r.hit_named for r in arm_rows)
            nlo, nhi = wilson(named, n)
            chars = [r.chars for r in arm_rows]
            kinds = {}
            for kind in ("planning", "callee"):
                sub = [r for r in arm_rows if r.kind == kind]
                k = sum(r.hit_entry for r in sub)
                klo, khi = wilson(k, len(sub))
                kinds[kind] = {"n": len(sub), "hits": k, "recall": _rate(k, len(sub)), "ci95": [klo, khi]}
            out[mode][arm] = {
                "n": n,
                "hits": hits,
                "recall": _rate(hits, n),
                "ci95": [lo, hi],
                "recall_strict": _rate(strict, n),
                "recall_strict_ci95": [slo, shi],
                "recall_named": _rate(named, n),
                "recall_named_ci95": [nlo, nhi],
                "chars_mean": _mean(chars),
                "chars_p95": percentile95(chars),
                "tokens_mean": _mean([tokens_of(c) for c in chars]),
                "tokens_p95": percentile95([tokens_of(c) for c in chars]),
                "full_applies_share": _rate(sum(r.full_applies for r in arm_rows), n),
                "by_kind": kinds,
            }
        out[mode]["mcnemar"] = {}
        for a, b in (("full", "command_shortlist"),):
            ra = {r.index: r.hit_entry for r in per_arm[a]}
            rb = {r.index: r.hit_entry for r in per_arm[b]}
            a_only = sum(1 for i in ra if ra[i] and not rb.get(i, False))
            b_only = sum(1 for i in ra if rb.get(i, False) and not ra[i])
            out[mode]["mcnemar"][f"{a}_vs_{b}"] = {
                "a_hit_b_missed": a_only, "b_hit_a_missed": b_only,
                "p_two_sided": mcnemar_exact(a_only, b_only)}
    return out


def source_head_answers(source_folder: Path, utterances: Sequence[str]) -> dict[str, dict[str, dict]] | None:
    """The trained heads' answers to each utterance, read from the SOURCE folder's current version (read only).

    Each head is loaded, asked every utterance, and released before the next (two BERT models per head). None when
    the workflow has no trained version: such a workflow has no realistic gathering to measure.
    """
    import gc

    from fastworkflow.model_pipeline_training import CommandRouter, ModelPipeline
    from fastworkflow.train import artifact_versioning
    from fastworkflow.train.score_lookup import discover_heads

    version = artifact_versioning.resolve_current_version(str(source_folder))
    if version is None:
        return None
    base = str(artifact_versioning.version_dir(str(source_folder), version))
    answers: dict[str, dict[str, dict]] = {}
    for head, folder in sorted(discover_heads(base).items()):
        router = CommandRouter(folder)
        answers[head] = {}
        for utterance in dict.fromkeys(utterances):
            labels, details = router.predict_with_details(utterance)
            answers[head][utterance] = {
                "labels": [str(label) for label in labels],
                "topk_labels": details["topk_labels"],
                "topk_scores": details["topk_scores"],
                "confident": details["confident"],
            }
        CommandRouter._instances_cache.clear()
        ModelPipeline._instances_cache.clear()
        gc.collect()
    return answers


def _rate(k: int, n: int) -> float | None:
    return None if n == 0 else k / n


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


# ---------------------------------------------------------------------------
# The threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(opened: OpenedWorkflow, requests: Sequence[Request], thresholds: Sequence[int],
                     cache: dict, contexts: Iterable[str]) -> dict[str, Any]:
    """For each candidate threshold: which contexts get the full listing, and each arm's recall on the requests.

    The FULL listing's size is measured with the workflow entered in each context (a context that cannot be
    entered without an instance argument is reported as None). The requests' recall is measured in the root only.
    """
    full_chars = {name: context_full_chars(opened, name) for name in contexts}
    result = {"contexts_full_chars": full_chars, "thresholds": {}}
    # the head modes (and production) when the source has trained heads: the same modes run_spec measures
    modes = ALL_MODES if opened.heads is not None else MODES
    for threshold in thresholds:
        rows, _ = measure_requests(opened, requests, threshold, cache, modes=modes)
        summary = summarise(rows, requests)
        result["thresholds"][str(threshold)] = {
            "full_applies_by_context": {name: (None if chars is None else chars <= threshold)
                                        for name, chars in full_chars.items()},
            "recall": {mode: {arm: summary[mode][arm]["recall"] for arm in ARM_NAMES} for mode in modes},
            "full_applies_share": {mode: summary[mode]["full"]["full_applies_share"] for mode in modes},
        }
    return result


def context_full_chars(opened: OpenedWorkflow, name: str) -> int | None:
    """The FULL listing's characters with the workflow entered in context *name* (None if it cannot be entered)."""
    from fastworkflow import workflow_agent
    from fastworkflow.context_identity import context_class_for

    app = opened.app
    previous = app.current_command_context
    try:
        if name == opened.root_context:
            return len(workflow_agent.planner_commands_text(app))
        cls = context_class_for(app, name)
        if cls is None:
            return None
        try:
            instance = cls()
        except Exception:  # noqa: BLE001 - a context that needs an instance argument is left out, and says so
            return None
        app.current_command_context = instance
        return len(workflow_agent.planner_commands_text(app))
    except Exception:  # noqa: BLE001 - a listing that cannot be built in a context is reported as None
        return None
    finally:
        app.current_command_context = previous


# ---------------------------------------------------------------------------
# The live-run estimate (arithmetic only; no model call)
# ---------------------------------------------------------------------------

def stratified_sample(requests: Sequence[Request], per_command: int, callees: Mapping[str, Sequence[str]],
                      seed: int = SAMPLE_SEED) -> list[Request]:
    """About *per_command* planning requests per planning command, and as many callee requests for each of its
    callees (a callee sampled once even when several planning commands call it). Deterministic for a seed."""
    rng = random.Random(seed)
    by_expected: dict[str, list[Request]] = defaultdict(list)
    for request in requests:
        by_expected[request.expected].append(request)
    chosen: list[Request] = []
    planning = sorted({r.expected for r in requests if r.kind == "planning"})
    done_callees: set[str] = set()
    for command in planning:
        pool = sorted((r for r in by_expected[command] if r.kind == "planning"), key=lambda r: r.utterance)
        chosen.extend(rng.sample(pool, min(per_command, len(pool))))
        for callee in sorted(callees.get(command, ())):
            if callee in done_callees:
                continue
            done_callees.add(callee)
            cpool = sorted((r for r in by_expected[callee] if r.kind == "callee"), key=lambda r: r.utterance)
            chosen.extend(rng.sample(cpool, min(per_command, len(cpool))))
    return chosen


def estimate_live(label: str, sample_sizes: Mapping[str, Mapping[str, float]], n_requests: int, *,
                  calls_per_request: int, overhead_tokens: int, output_tokens: int,
                  arms_to_price: Sequence[str] = ARM_NAMES) -> dict[str, Any]:
    """Calls and tokens of a minimal end-to-end plan-success run for each arm (the sizes are measured means)."""
    arms = {}
    for arm in arms_to_price:
        listing_tokens = sample_sizes[arm]["tokens_mean"]
        prompt_per_call = overhead_tokens + listing_tokens
        calls = n_requests * calls_per_request
        arms[arm] = {
            "requests": n_requests,
            "calls": calls,
            "prompt_tokens_per_call_mean": round(prompt_per_call, 1),
            "prompt_tokens_total": round(calls * prompt_per_call),
            "output_tokens_total": calls * output_tokens,
            "listing_tokens_mean": round(listing_tokens, 1),
        }
    return {"workflow": label, "arms": arms,
            "assumptions": {"calls_per_request": calls_per_request, "overhead_tokens": overhead_tokens,
                            "output_tokens_per_call": output_tokens}}


# ---------------------------------------------------------------------------
# Workflows the scorer runs on
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    label: str
    folder: Path
    testset: Path
    handwritten: bool
    context_path: str | None = None


def committed_specs() -> list[Spec]:
    return [
        Spec("library", REPO / "fastworkflow" / "examples" / "library", TEST_SETS / "library.json", False,
             context_path="fastworkflow.examples.library.application.library:Library"),
        Spec("tutorial_no_hints", TESTS / "planning_tutorial_workflow", TEST_SETS / "tutorial_no_hints.json", True),
        Spec("todo_no_hints", TESTS / "todo_list_workflow", TEST_SETS / "todo_no_hints.json", True),
    ]


def app_spec(folder: Path, testset: Path) -> Spec:
    return Spec("app", folder, testset, False)


def run_spec(spec: Spec, scratch: Path, thresholds: Sequence[int] | None, *, requests: Sequence[Request] | None = None,
             threshold: int | None = None) -> dict[str, Any]:
    """Measure one workflow: summary at the production threshold, and optionally the sweep."""
    import fastworkflow.run_listing as run_listing

    reqs = list(requests) if requests is not None else (
        load_handwritten(spec.testset, str(spec.folder)) if spec.handwritten else load_test_set(spec.testset))
    opened = open_workflow(spec.folder, spec.label, scratch, context_path=spec.context_path)
    try:
        limit = run_listing.FULL_LISTING_MAX_CHARS if threshold is None else threshold
        cache: dict = {}
        heads = source_head_answers(spec.folder, [r.utterance for r in reqs])
        opened.heads = heads
        modes = ALL_MODES if heads is not None else MODES
        rows, _ = measure_requests(opened, reqs, limit, cache, modes=modes)
        result: dict[str, Any] = {
            "workflow": spec.label,
            "requests": len(reqs),
            "by_kind": {k: sum(1 for r in reqs if r.kind == k) for k in ("planning", "callee")},
            "threshold": limit,
            "full_listing_chars": len(workflow_full(opened)),
            "trained_heads": sorted(heads) if heads is not None else None,
            "summary": summarise(rows, reqs),
        }
        if thresholds:
            from fastworkflow.command_routing import RoutingRegistry

            crd = RoutingRegistry.get_definition(opened.app.folderpath)
            contexts = {name: None for name in sorted(crd.contexts) if name != "*"}
            contexts.setdefault(opened.root_context, None)
            result["sweep"] = sweep_thresholds(opened, reqs, thresholds, cache, contexts)
        return result
    finally:
        opened.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("offline", "sweep", "estimate"))
    parser.add_argument("--out", required=True, help="output directory (JSON); use the scratchpad, not the repo")
    parser.add_argument("--app-workflow", help="the app's workflow folder (read; copied before use)")
    parser.add_argument("--app-testset", help="the app's planning test set (read only)")
    parser.add_argument("--thresholds", help="comma-separated candidates for sweep (default 3000..12000 step 500)")
    parser.add_argument("--only", help="comma-separated labels to run (library, tutorial_no_hints, todo_no_hints, app)")
    parser.add_argument("--per-command-library", type=int, default=LIVE_PER_COMMAND["library"],
                        help="estimate: planning requests per planning command (and per callee)")
    parser.add_argument("--per-command-app", type=int, default=LIVE_PER_COMMAND["app"])
    parser.add_argument("--app-arms", default="command_shortlist",
                        help="estimate: arms priced for the app (full is ~11M prompt tokens; add it to price it)")
    parser.add_argument("--calls-per-request", type=int, default=DEFAULT_CALLS_PER_REQUEST)
    parser.add_argument("--overhead-tokens", type=int, default=DEFAULT_OVERHEAD_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    args = parser.parse_args(argv)

    specs = committed_specs()
    if args.app_workflow and args.app_testset:
        specs.append(app_spec(Path(args.app_workflow), Path(args.app_testset)))
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in specs if s.label in wanted]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = None
    if args.command == "sweep":
        thresholds = [int(t) for t in args.thresholds.split(",")] if args.thresholds else list(DEFAULT_THRESHOLDS)

    scratch = Path(tempfile.mkdtemp(prefix="score_listing_"))
    try:
        if args.command in ("offline", "sweep"):
            results = [run_spec(spec, scratch, thresholds) for spec in specs]
            path = out / f"{args.command}.json"
            path.write_text(json.dumps({"schema": SCHEMA, "results": results}, indent=2, sort_keys=True))
            print(f"wrote {path}")
        else:
            estimates = []
            for spec in specs:
                from fastworkflow.train import planning_test_set

                if spec.handwritten:
                    reqs = load_handwritten(spec.testset, str(spec.folder))
                    inv = _inventory_of(spec.folder)
                    callees = {name: list(inv.callees.get(name, [])) for name in inv.planning}
                else:
                    reqs = load_test_set(spec.testset)
                    callees = {name: list(t["callees"]) for name, t in json.loads(
                        Path(spec.testset).read_text()).get("targets", {}).items() if "callees" in t}
                per = args.per_command_app if spec.label == "app" else args.per_command_library
                arms_to_price = tuple(a for a in args.app_arms.split(",") if a) if spec.label == "app" else ARM_NAMES
                sample = stratified_sample(reqs, per, callees)
                if not sample:
                    print(f"{spec.label}: no planning command in the test set; nothing to sample, no estimate")
                    continue
                opened = open_workflow(spec.folder, spec.label, scratch, context_path=spec.context_path)
                try:
                    rows, _ = measure_requests(opened, sample, run_listing_threshold(), {}, modes=(MODE_GOLD,))
                    summary = summarise(rows, sample)
                finally:
                    opened.close()
                sizes = {arm: summary[MODE_GOLD][arm] for arm in ARM_NAMES}
                estimates.append(estimate_live(
                    spec.label, sizes, len(sample), calls_per_request=args.calls_per_request,
                    overhead_tokens=args.overhead_tokens, output_tokens=args.output_tokens,
                    arms_to_price=arms_to_price)
                    | {"sample_by_kind": {k: sum(1 for r in sample if r.kind == k) for k in ("planning", "callee")},
                       "per_command": per})
            totals = {}
            for arm in ARM_NAMES:
                priced = [e["arms"][arm] for e in estimates if arm in e["arms"]]
                totals[arm] = {
                    "requests": sum(a["requests"] for a in priced),
                    "calls": sum(a["calls"] for a in priced),
                    "prompt_tokens_total": sum(a["prompt_tokens_total"] for a in priced),
                    "output_tokens_total": sum(a["output_tokens_total"] for a in priced),
                }
            path = out / "estimate.json"
            path.write_text(json.dumps({"schema": SCHEMA, "live": False, "estimates": estimates, "totals": totals},
                                       indent=2, sort_keys=True))
            print_estimate(estimates)
            print("\nTOTALS per arm (only the priced workflows):")
            for arm in ARM_NAMES:
                t = totals[arm]
                if t["calls"]:
                    print(f"  {arm:<18} requests {t['requests']:>4}  calls {t['calls']:>5}  "
                          f"prompt tokens {t['prompt_tokens_total']:>12,}  output tokens {t['output_tokens_total']:>10,}")
            print(f"wrote {path}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def run_listing_threshold() -> int:
    from fastworkflow.run_listing import FULL_LISTING_MAX_CHARS

    return FULL_LISTING_MAX_CHARS


def print_estimate(estimates: Sequence[Mapping[str, Any]]) -> None:
    print("DRY RUN: no model is called. Sizes are measured; calls and tokens are estimates.")
    for est in estimates:
        a = est["assumptions"]
        print(f"\n{est['workflow']}: sample {est['sample_by_kind']} ({est['per_command']} per command), "
              f"assumptions: {a['calls_per_request']} calls/request, {a['overhead_tokens']} overhead tokens/call, "
              f"{a['output_tokens_per_call']} output tokens/call")
        print(f"  {'arm':<18}{'requests':>9}{'calls':>8}{'listing tok':>13}{'prompt tok':>13}{'output tok':>12}")
        for arm in ARM_NAMES:
            if arm not in est["arms"]:
                print(f"  {arm:<18} not priced (see --app-arms)")
                continue
            row = est["arms"][arm]
            print(f"  {arm:<18}{row['requests']:>9}{row['calls']:>8}{row['listing_tokens_mean']:>13}"
                  f"{row['prompt_tokens_total']:>13,}{row['output_tokens_total']:>12,}")


if __name__ == "__main__":
    sys.exit(main())
