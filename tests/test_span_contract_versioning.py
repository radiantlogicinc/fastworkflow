"""Per-emitter span attribute contracts (bead fix-ajv.8, arch §12.0 delta 5).

`tracing.SPAN_CONTRACT_VERSION` can say that something in the taxonomy moved
between two runs. It cannot say what. `tracing.SPAN_CONTRACTS` declares a version
and an attribute-key set per emitter so a reader who notices the aggregate moved
can find the emitter, and so a stored span describes its own contract without a
lookup table.

Declared numbers are worth nothing unless something forces them to track the
code, so most of this file is one AST pass and the checks it feeds:

**Coverage.** Every span name any emitter opens must have a contract. Discovered
by scanning the whole package for `start_span` and for `tracing.Span(name=...)`
reconstructions, so a new emitter anywhere fails here rather than being noticed
later by a reader with a gap in their data.

**Attribute agreement.** The declared key set must equal the keys the emitters
actually write. Discovery follows the keys into the bags they are written to:
`nlu_trace` and `diagnostics` are filled statements (and, for `diagnostics`,
modules) away from the emission site, `**tracing.capture_attributes(...)` is a
shared projection, and a scan that looked only at the dict literal beside
`attributes=` would see about half of the real contract and quietly bless the
rest. Adding a key therefore forces an edit to the declaration, which puts the
version under the cursor — the closest a static check gets to "you cannot add a
key without deciding about the version", and strictly better than the
hand-maintained list it replaces, which had no way to notice a bag.

**Producer agreement.** `fw.agent.tool_call` is opened from THREE sites. One
version for that name is a lie unless the three write the same keys, so the rule
is enforced rather than asserted in a comment: every function that *opens* a span
of a given name must write the same set. Functions that RECONSTRUCT a span (the
`Span(...)` rebuilds that close `fw.turn` and `fw.ask_user` across a suspension)
are exempt, because they are a later phase of one span rather than an alternative
producer of it — their keys join the union without having to match the open.

Both scans have a self-test against a known-bad sample: a structural check nobody
has watched reject anything is decoration
(`test_no_capture_control_flow.py::test_the_scan_would_actually_catch_a_read`).

The integration half runs the real todo_list_workflow through the real
WorkflowExecutionContext into a real SQLite sink, because the static half proves
what the source says and not what a span ends up carrying.
"""

from __future__ import annotations

import ast
import json
import sqlite3
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.observability import store as obs
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observability.evidence_run import capture_observability_provenance
from fastworkflow.observability.provenance import ObservabilityProvenance
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

PACKAGE_ROOT = Path(fastworkflow.__file__).resolve().parent

# `tracing` defines the emission machinery, so its own `Span(...)` construction is
# the mechanism rather than an emitter. It is still searched for the shared
# attribute projections (`capture_attributes`), which three emitters call.
MECHANISM = "tracing.py"


# ----------------------------------------------------------------------
# The AST pass
# ----------------------------------------------------------------------


def _package_trees() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        with suppress(SyntaxError, UnicodeDecodeError):
            trees[str(path.relative_to(PACKAGE_ROOT))] = ast.parse(
                path.read_text(encoding="utf-8")
            )
    return trees


TREES = _package_trees()


def _called_name(node: ast.AST) -> str | None:
    """The bare name of a called function, ignoring how it was qualified."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _span_names(node: ast.AST) -> set[str]:
    """Span-name constants an expression can evaluate to.

    Resolved through the `tracing` module rather than by matching text, so
    renaming a constant cannot silently empty the scan. An `IfExp` contributes
    both branches: `build_query_with_next_steps` opens `.replan` or `.plan` from
    one call.
    """
    if isinstance(node, ast.IfExp):
        return _span_names(node.body) | _span_names(node.orelse)
    if isinstance(node, ast.Attribute):
        value = getattr(tracing, node.attr, None)
    elif isinstance(node, ast.Name):
        value = getattr(tracing, node.id, None)
    elif isinstance(node, ast.Constant):
        value = node.value
    else:
        return set()
    return {value} if isinstance(value, str) and value.startswith("fw.") else set()


def _attribute_key(node: ast.AST) -> str | None:
    """One attribute key, written as a literal or as a `tracing.ATTR_*` constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    name = _called_name(node)
    if name and name.startswith("ATTR_"):
        value = getattr(tracing, name, None)
        if isinstance(value, str):
            return value
    return None


def _function_defs(name: str, scope: dict[str, ast.Module]):
    for tree in scope.values():
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                yield node


def _handoff_bodies(bag: str, home_tree: ast.Module) -> list[ast.AST]:
    """Functions the emitting module hands *bag* to, by keyword.

    `validate_parameters(..., diagnostics=diagnostics)` fills the parameter-
    extraction span's bag from `utils/signatures.py`. Following the handoff by
    keyword keeps that reachable without searching the package for every
    assignment to a common name, which would sweep in unrelated dicts.
    """
    handed: set[str] = set()
    for node in ast.walk(home_tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == bag
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == bag
                and (called := _called_name(node.func))
            ):
                handed.add(called)
    return [
        body for name in sorted(handed) for body in _function_defs(name, TREES)
    ]


class _Unresolved(Exception):
    """The scan met an expression it cannot read keys out of.

    Raised rather than skipped: a scan that silently gives up reports a contract
    it never checked, which is the failure mode this whole file exists to avoid.
    """


def _dict_keys(node: ast.AST, home: str, seen: set) -> set[str]:
    """Keys of a dict-valued expression, following names and helper calls."""
    keys: set[str] = set()
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:  # **expansion
                keys |= _dict_keys(value, home, seen)
                continue
            resolved = _attribute_key(key)
            if resolved is None:
                raise _Unresolved(f"attribute key at line {key.lineno} of {home}")
            keys.add(resolved)
    elif isinstance(node, ast.Name):
        keys |= _bag_keys(node.id, home, seen)
    elif isinstance(node, ast.Call):
        name = _called_name(node.func)
        if name == "dict" and node.args:
            keys |= _dict_keys(node.args[0], home, seen)
        elif name:
            keys |= _returned_keys(name, home, seen)
        else:
            raise _Unresolved(f"call at line {node.lineno} of {home}")
    else:
        raise _Unresolved(f"{type(node).__name__} at line {node.lineno} of {home}")
    return keys


def _bag_keys(bag: str, home: str, seen: set) -> set[str]:
    """Keys written into a dict *variable* anywhere it is filled."""
    if ("bag", bag) in seen:
        return set()
    seen.add(("bag", bag))
    home_tree = TREES[home]
    keys: set[str] = set()
    for body in (home_tree, *_handoff_bodies(bag, home_tree)):
        for node in ast.walk(body):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == bag
                    ):
                        resolved = _attribute_key(target.slice)
                        if resolved is None:
                            raise _Unresolved(
                                f"{bag}[...] at line {target.lineno} of {home}"
                            )
                        keys.add(resolved)
                    elif (
                        isinstance(target, ast.Name)
                        and target.id == bag
                        and node.value is not None
                    ):
                        keys |= _dict_keys(node.value, home, seen)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == bag
                and node.args
            ):
                if node.func.attr == "update":
                    keys |= _dict_keys(node.args[0], home, seen)
                elif node.func.attr == "setdefault":
                    resolved = _attribute_key(node.args[0])
                    if resolved is None:
                        raise _Unresolved(
                            f"{bag}.setdefault at line {node.lineno} of {home}"
                        )
                    keys.add(resolved)
    return keys


def _returned_keys(function_name: str, home: str, seen: set) -> set[str]:
    """Keys of the dicts a helper returns.

    Searched in the emitting module and in `tracing`, which is where the
    projections shared across emitters live.
    """
    if ("function", function_name) in seen:
        return set()
    seen.add(("function", function_name))
    scope = {home: TREES[home], MECHANISM: TREES[MECHANISM]}
    keys: set[str] = set()
    found = False
    for body in _function_defs(function_name, scope):
        found = True
        for node in ast.walk(body):
            if isinstance(node, ast.Return) and node.value is not None:
                keys |= _dict_keys(node.value, home, seen)
    if not found:
        raise _Unresolved(f"helper {function_name}() called from {home}")
    return keys


def _parent_table(tree: ast.Module) -> dict:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _scan_emission_sites() -> list[dict]:
    """One entry per function that opens, reconstructs, or closes a span."""
    sites: list[dict] = []
    for relative_path, tree in TREES.items():
        if relative_path == MECHANISM:
            continue
        parents = _parent_table(tree)
        module_names: set[str] = set()
        calls: list[tuple[ast.Call, str, set[str]]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node.func)
            if name == "start_span":
                opened = _span_names(node.args[1]) if len(node.args) > 1 else set()
                if not opened:
                    raise _Unresolved(
                        f"start_span at line {node.lineno} of {relative_path} "
                        "opens a span whose name this scan cannot resolve"
                    )
                module_names |= opened
                calls.append((node, "open", opened))
            elif name == "Span":
                rebuilt: set[str] = set()
                for keyword in node.keywords:
                    if keyword.arg == "name":
                        rebuilt = _span_names(keyword.value)
                # A `Span(name=<something dynamic>)` is a copy, not an emitter:
                # `observability_store.emit_span` snapshots what it was handed.
                if not rebuilt:
                    continue
                module_names |= rebuilt
                calls.append((node, "rebuild", rebuilt))
            elif name == "end_span":
                calls.append((node, "close", set()))
        if not calls:
            continue

        def enclosing_function(node: ast.AST):
            current = parents.get(node)
            while current is not None and not isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                current = parents.get(current)
            return current

        grouped: dict[int, dict] = {}
        for node, role, names in calls:
            function = enclosing_function(node)
            entry = grouped.setdefault(
                id(function),
                {"function": function, "opens": set(), "rebuilds": set(), "calls": []},
            )
            if role == "open":
                entry["opens"] |= names
            elif role == "rebuild":
                entry["rebuilds"] |= names
            entry["calls"].append(node)

        for entry in grouped.values():
            names = entry["opens"] | entry["rebuilds"]
            if not names:
                # A function that only closes a span it was handed. The module's
                # own span names answer it when there is exactly one; more than
                # one and the site is genuinely ambiguous, which is a scan gap
                # rather than something to guess at.
                if len(module_names) != 1:
                    raise _Unresolved(
                        f"end_span in {relative_path}::"
                        f"{entry['function'].name if entry['function'] else '<module>'}"
                        " cannot be attributed to a span name"
                    )
                names = set(module_names)
            keys: set[str] = set()
            for node in entry["calls"]:
                for keyword in node.keywords:
                    if keyword.arg == "attributes":
                        keys |= _dict_keys(keyword.value, relative_path, set())
            sites.append(
                {
                    "file": relative_path,
                    "function": (
                        entry["function"].name if entry["function"] else "<module>"
                    ),
                    "opens": frozenset(entry["opens"]),
                    "names": frozenset(names),
                    "keys": frozenset(keys),
                }
            )
    return sites


SITES = _scan_emission_sites()


def _emitted_attributes() -> dict[str, set[str]]:
    discovered: dict[str, set[str]] = {}
    for site in SITES:
        for name in site["names"]:
            discovered.setdefault(name, set()).update(site["keys"])
    return discovered


# ----------------------------------------------------------------------
# The scan is real
# ----------------------------------------------------------------------


def test_the_scan_found_the_emitters_that_obviously_exist():
    """A scan that found nothing would pass every check below.

    These four are named because each is a different discovery shape: an opener
    with a literal name, an opener reached through a conditional, a closer with no
    name at its own call site, and a reconstruction.
    """
    found = {(site["file"], site["function"]) for site in SITES}
    for expected in (
        ("command_executor.py", "invoke_command"),
        ("workflow_agent.py", "build_query_with_next_steps"),
        ("utils/dspy_logger.py", "on_lm_end"),
        ("workflow_execution_context.py", "_close_ask_user_span"),
    ):
        assert expected in found, f"the scan missed {expected}"

    assert len(_emitted_attributes()) >= 10


def test_the_scan_reaches_keys_written_far_from_the_emission_site():
    """The bags, specifically — the half a dict-literal scan cannot see.

    `escalation_outcome` is written into `nlu_trace` inside `_predict_impl`;
    `db_lookup` is written into `diagnostics` from `utils/signatures.py`, a
    different module entirely; `context_before` arrives through
    `**tracing.capture_attributes(...)`. If any of the three stopped being
    discovered, the corresponding declaration would silently become unchecked.
    """
    discovered = _emitted_attributes()
    assert "escalation_outcome" in discovered[tracing.SPAN_NLU_INTENT]
    assert "db_lookup" in discovered[tracing.SPAN_NLU_PARAM_EXTRACTION]
    assert tracing.ATTR_CONTEXT_BEFORE in discovered[tracing.SPAN_AGENT_TOOL_CALL]


# ----------------------------------------------------------------------
# Coverage: every emitter declares a version
# ----------------------------------------------------------------------


def test_every_emitted_span_name_declares_a_contract():
    """Arch §12.0 delta 5. A span nobody versioned is a span nobody can compare."""
    undeclared = sorted(set(_emitted_attributes()) - set(tracing.SPAN_CONTRACTS))
    assert undeclared == [], (
        "these span names are emitted with no entry in tracing.SPAN_CONTRACTS: "
        f"{undeclared}. Declare the emitter's attribute keys and a version."
    )


def test_no_contract_is_declared_for_a_span_nobody_emits():
    """The other direction: a version for a dead emitter is a claim about
    evidence that does not exist."""
    orphans = sorted(set(tracing.SPAN_CONTRACTS) - set(_emitted_attributes()))
    assert orphans == []


def test_the_reserved_train_prefix_has_no_emitter_and_no_contract():
    """`fw.train.*` is reserved so the schema needs no migration when training
    starts emitting. Nothing opens one today, so versioning it would declare a
    contract no code has ever kept."""
    assert tracing.SPAN_TRAIN_PREFIX not in tracing.SPAN_CONTRACTS
    emitted = set(_emitted_attributes())
    assert not [name for name in emitted if name.startswith(tracing.SPAN_TRAIN_PREFIX)]


def test_every_declared_version_is_a_positive_integer():
    for name, contract in tracing.SPAN_CONTRACTS.items():
        assert isinstance(contract.version, int), name
        assert contract.version >= 1, name
        assert contract.attributes, f"{name} declares no attributes at all"


# ----------------------------------------------------------------------
# Attribute agreement
# ----------------------------------------------------------------------


def test_declared_attributes_match_what_the_emitters_write():
    """The check that makes the version numbers mean something.

    A key added to an emitter and not to its declaration fails here; so does a
    declared key no emitter writes, which is how a contract rots into fiction
    after a key is removed.
    """
    discovered = _emitted_attributes()
    problems = []
    for name, contract in sorted(tracing.SPAN_CONTRACTS.items()):
        written = discovered.get(name, set())
        if added := sorted(written - set(contract.attributes)):
            problems.append(f"{name}: emitted but undeclared {added}")
        if dropped := sorted(set(contract.attributes) - written):
            problems.append(f"{name}: declared but never emitted {dropped}")
    assert problems == [], (
        "tracing.SPAN_CONTRACTS disagrees with the emitters:\n  "
        + "\n  ".join(problems)
        + "\nUpdate the declaration, and bump that emitter's version while you "
        "are in it — a reader joining a run recorded before the change to one "
        "recorded after has no other way to notice."
    )


def test_the_envelope_key_belongs_to_no_emitter():
    """`span_contract_version` is written by the machinery, not by an emitter, so
    it is not part of the contract it describes — an emitter that declared it
    would be versioning its own version stamp."""
    assert tracing.ATTR_SPAN_CONTRACT_VERSION in tracing.ENVELOPE_ATTRIBUTES
    for name, contract in tracing.SPAN_CONTRACTS.items():
        assert tracing.ATTR_SPAN_CONTRACT_VERSION not in contract.attributes, name


# ----------------------------------------------------------------------
# Producer agreement — what makes ONE version for fw.agent.tool_call honest
# ----------------------------------------------------------------------


def test_all_producers_of_one_span_name_write_the_same_attributes():
    """Three sites open `fw.agent.tool_call`, and they share one version.

    Until fix-ajv.3's capture reached the third (`workflow_agent`), that single
    version described three different attribute sets — a reader would have had to
    know which site wrote a span before knowing what to expect from it. Enforced
    here rather than described in a comment, so a fourth producer, or a key added
    to one of the three, fails instead of quietly re-opening the problem.
    """
    producers: dict[str, list[dict]] = {}
    for site in SITES:
        for name in site["opens"]:
            producers.setdefault(name, []).append(site)

    problems = []
    for name, sites in sorted(producers.items()):
        distinct = {site["keys"] for site in sites}
        if len(distinct) > 1:
            detail = "; ".join(
                f"{site['file']}::{site['function']} writes {sorted(site['keys'])}"
                for site in sites
            )
            problems.append(f"{name}: {detail}")
    assert problems == [], (
        "these span names are opened by sites that write different attributes, "
        "so one contract version cannot describe them:\n  " + "\n  ".join(problems)
    )


def test_fw_agent_tool_call_really_is_opened_from_three_sites():
    """The test above is only interesting while the trio exists."""
    openers = sorted(
        (site["file"], site["function"])
        for site in SITES
        if tracing.SPAN_AGENT_TOOL_CALL in site["opens"]
    )
    assert openers == [
        ("workflow_agent.py", "_execute_workflow_query"),
        ("workflow_execution_context.py", "_process_action"),
        ("workflow_execution_context.py", "_process_message"),
    ]


def test_a_reconstruction_is_not_held_to_the_producer_rule():
    """`fw.turn` and `fw.ask_user` close in a function that rebuilds the span to
    survive a process boundary, and a close legitimately writes keys the open did
    not. Treating the rebuild as a rival producer would make the rule above
    unsatisfiable for both."""
    rebuilders = {
        (site["file"], site["function"])
        for site in SITES
        if site["names"] and not site["opens"]
    }
    assert ("workflow_execution_context.py", "_close_ask_user_span") in rebuilders
    assert ("workflow_execution_context.py", "_finalize_turn_trace") in rebuilders


# ----------------------------------------------------------------------
# The scans can fail
# ----------------------------------------------------------------------


def test_the_key_scan_catches_a_key_added_to_a_bag():
    """The shape a real regression takes: not a literal beside `attributes=`, but
    one more line writing into the bag several statements away."""
    module = ast.parse(
        "def emit(host):\n"
        "    bag = {'known': 1}\n"
        "    bag['sneaked_in'] = 2\n"
        "    tracing.end_span(host, span, attributes=bag)\n"
    )
    TREES["<sample>"] = module
    try:
        keys = _dict_keys(
            module.body[0].body[-1].value.keywords[0].value, "<sample>", set()
        )
    finally:
        del TREES["<sample>"]
    assert keys == {"known", "sneaked_in"}


def test_the_site_scan_catches_a_new_emitter():
    module = ast.parse(
        "def emit(host):\n"
        "    tracing.start_span(host, tracing.SPAN_TURN, attributes={'k': 1})\n"
    )
    TREES["<sample>"] = module
    try:
        sites = [site for site in _scan_emission_sites() if site["file"] == "<sample>"]
    finally:
        del TREES["<sample>"]
    assert len(sites) == 1
    assert sites[0]["opens"] == frozenset({tracing.SPAN_TURN})
    assert sites[0]["keys"] == frozenset({"k"})


def test_an_unreadable_emission_site_is_reported_rather_than_skipped():
    """A scan that shrugs at what it cannot parse reports a contract it never
    checked."""
    module = ast.parse("def emit(host, chosen):\n    start_span(host, chosen)\n")
    TREES["<sample>"] = module
    try:
        with pytest.raises(_Unresolved):
            _scan_emission_sites()
    finally:
        del TREES["<sample>"]


# ----------------------------------------------------------------------
# Provenance (arch §12.0 delta 6, §12.4)
# ----------------------------------------------------------------------


def test_provenance_carries_the_per_emitter_map():
    provenance = capture_observability_provenance()
    assert provenance.span_contract_versions == tracing.span_contract_versions()
    assert provenance.span_contract_versions[tracing.SPAN_AGENT_TOOL_CALL] >= 1


def test_the_aggregate_is_kept_beside_the_map_not_replaced_by_it():
    """A run-to-run comparison asks "did anything move" and reads one number; the
    map answers "which emitter" once that comparison fails. Dropping the aggregate
    would force every comparison to diff a map to answer a yes/no question, and
    would break every existing reader of the field."""
    provenance = capture_observability_provenance()
    assert provenance.span_contract_version == tracing.SPAN_CONTRACT_VERSION
    assert isinstance(provenance.span_contract_version, int)
    assert provenance.span_contract_versions


def test_the_aggregate_number_moves_only_on_purpose():
    """Every other check here compares against `tracing.SPAN_CONTRACT_VERSION`
    itself, so a bump -- or an accidental revert -- passes them all. Pinning the
    literal makes a change to the aggregate a deliberate edit to this test, next
    to the history comment in `tracing.py` that says what moved."""
    assert tracing.SPAN_CONTRACT_VERSION == 12


def test_a_provenance_record_written_before_the_map_still_validates():
    """`ObservabilityProvenance` is frozen with `extra="forbid"`, so the new field
    had to be optional: a required one would reject every already-serialized
    record and every caller that predates it."""
    legacy = {
        "enabled": True,
        "span_contract_version": 2,
        "db_schema_version": obs.SCHEMA_VERSION,
    }
    restored = ObservabilityProvenance(**legacy)
    assert restored.span_contract_versions == {}
    assert restored.span_contract_version == 2


def test_the_map_survives_the_json_round_trip_a_bundle_takes():
    dumped = json.loads(
        json.dumps(capture_observability_provenance().model_dump(mode="json"))
    )
    assert dumped["span_contract_versions"] == tracing.span_contract_versions()


# ----------------------------------------------------------------------
# Integration: what a span actually carries
# ----------------------------------------------------------------------

LIST_COMMAND = "TodoListManager/list_todo_lists"
CREATE_COMMAND = "TodoListManager/create_todo_list"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    """A real TraceSink implementation — the pluggable seam the design defines."""

    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


def _make_ctx(todo_workflow_path: str, tmp_path, sink) -> WorkflowExecutionContext:
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"contract-{uuid.uuid4().hex}"
    )
    context = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    context.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    return context


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str = LIST_COMMAND, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _nesting_cme_hop(monkeypatch, app_workflow, nested_command: str = LIST_COMMAND):
    """Stand in for the untrained CME wildcard hop, keeping the nesting real.

    Same stand-in as tests/test_command_call_id.py: the real hop resolves the
    user's text into an Action and calls `perform_action` again, so this keeps the
    real dispatch underneath and only replaces the NLU the test workflows cannot
    run.
    """
    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, workflow, action):
        command_output = real_perform_action(app_workflow, _action(nested_command))
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = nested_command
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))


def test_every_span_a_real_turn_emits_is_self_describing(ctx, sink):
    """A stored span has to say which contract it was written under, or a reader
    joining two runs has to know when each was recorded to interpret either."""
    ctx.process_action_turn(_action())

    assert sink.spans
    for span in sink.spans:
        assert span.attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == (
            tracing.SPAN_CONTRACTS[span.name].version
        )


def test_a_real_turn_writes_no_attribute_outside_the_declared_contract(ctx, sink):
    """The static scan proves what the source says; this proves what arrives.

    A key that reaches a span through a path the AST pass cannot see would slip
    past `test_declared_attributes_match_what_the_emitters_write` and show up
    here.
    """
    ctx.process_action_turn(_action(CREATE_COMMAND, description="groceries"))

    problems = []
    for span in sink.spans:
        allowed = tracing.SPAN_CONTRACTS[span.name].attributes | (
            tracing.ENVELOPE_ATTRIBUTES
        )
        if undeclared := sorted(set(span.attributes) - allowed):
            problems.append(f"{span.name}: {undeclared}")
    assert problems == []


def test_the_prose_path_stamps_its_two_spans_too(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """`fw.command.execute` and the `fw.agent.tool_call` around it are versioned
    independently, so a change to one does not implicate the other."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, context.app_workflow)
        context.process_turn("list my todo lists")

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
        assert execute.attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == (
            tracing.SPAN_CONTRACTS[tracing.SPAN_COMMAND_EXECUTE].version
        )
        assert tool_call.attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == (
            tracing.SPAN_CONTRACTS[tracing.SPAN_AGENT_TOOL_CALL].version
        )
    finally:
        with suppress(Exception):
            context.close()


def test_the_version_survives_the_store_and_is_readable_back(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """"Self-describing" means in the DATABASE, which is why the version is an
    attribute and not a `Span` field: `spans` has fixed columns, and a field the
    store has no column for is dropped at the boundary that matters."""
    db_path = str(tmp_path / "observability.sqlite3")
    store_sink = obs.SQLiteTraceSink(db_path)
    context = _make_ctx(todo_workflow_path, tmp_path, store_sink)
    try:
        turn = context.process_action_turn(_action())
    finally:
        with suppress(Exception):
            context.close()
        store_sink.close()  # drains pending writes

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT name, attributes FROM spans WHERE trace_id=?",
                (turn.turn_key,),
            )
        ]
    finally:
        connection.close()

    assert rows
    for row in rows:
        attributes = json.loads(row["attributes"])
        assert attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == (
            tracing.SPAN_CONTRACTS[row["name"]].version
        )


def test_re_emitting_a_span_does_not_move_its_version(ctx, sink):
    """The store upserts on span_id ([R2]/[R6]), so a span emitted at open and
    again at close must not disagree with itself about its contract."""
    ctx.process_action_turn(_action())
    span = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
    stamped = span.attributes[tracing.ATTR_SPAN_CONTRACT_VERSION]

    tracing.end_span(ctx, span, attributes={"response_text": "replayed"})

    replays = [s for s in sink.spans if s.span_id == span.span_id]
    assert len(replays) == 2
    assert {
        s.attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] for s in replays
    } == {stamped}


def test_stamping_costs_nothing_when_nothing_is_recording(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """The stamp lives in the emit funnel, which a turn with no sink never
    reaches — so this is additive recording rather than work every turn pays for.
    The no-op sink is passed explicitly: a context given no sink at all opens
    its workflow's own observability sink."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink=tracing.NoOpTraceSink())
    try:
        turn = context.process_action_turn(_action())
        assert turn.success
        assert isinstance(context.trace_sink, tracing.NoOpTraceSink)
    finally:
        with suppress(Exception):
            context.close()
