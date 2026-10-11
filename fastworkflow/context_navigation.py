"""The command sequence that reaches a context the caller is not in.

The agent tool refuses a command name that belongs to some other context. Naming
that context is not enough: ``go_up`` and ``reset_context`` only climb, and the
command the caller just attempted often lives *down* from here (``find_item``
is reached from global by ``open_item_explorer``). This module turns the declarations
the workflow already made into that sequence, and names the parameters a step
cannot fill in.

Nothing here navigates. A misnamed command must not move the workflow.
"""
from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.entry_declarations import declared_entry_commands, declared_occupiable

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")

#: ``go_up`` is tried before ``reset_context`` when both cost the same, because
#: ``go_up`` is the move that lands on the live parent. ``reset_context`` still
#: wins when it is shorter (two climbs versus one reset).
_KIND_ORDER = {"ascend": 0, "descend": 1, "reset": 2}


def display_context_name(context: str) -> str:
    return "global" if context == "*" else context


@dataclass(frozen=True)
class DescendEdge:
    """One command that enters ``target`` when run from ``source``."""

    source: str
    command: str
    target: str
    needs: tuple[str, ...] = ()
    required_to_enter: tuple[str, ...] = ()


@dataclass(frozen=True)
class NavStep:
    """One command on a path, and why it may not be runnable as written."""

    command: str
    enters: str
    needs: tuple[str, ...] = ()
    required_to_enter: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Move:
    command: str
    target: str
    needs: tuple[str, ...]
    required_to_enter: tuple[str, ...]
    kind: str


def live_context_chain(workflow) -> tuple[str, ...]:
    """The contexts ``go_up`` will actually visit, current first, global last.

    The hierarchy file lists every parent a context *may* have. ``go_up`` follows
    the object in hand, which has one parent. A path that climbed a parent the
    live object does not have would send the caller the wrong way.
    """
    current = getattr(workflow, "current_command_context_name", "*")
    if current == "*":
        return ("*",)
    names = [current]
    seen = {current}
    obj = getattr(workflow, "current_command_context", None)
    root = getattr(workflow, "root_command_context", None)
    while obj is not None and obj is not root:
        try:
            parent = workflow.get_parent(obj)
        except Exception:
            logger.debug("parent walk stopped at %r", names[-1], exc_info=True)
            break
        if parent is None or parent is root or parent is obj:
            break
        parent_name = type(parent).__name__
        if parent_name in seen:
            break
        seen.add(parent_name)
        names.append(parent_name)
        obj = parent
    if names[-1] != "*":
        names.append("*")
    return tuple(names)


def plan_navigation(
    current: str,
    targets: list[str],
    live_chain: tuple[str, ...],
    edges: tuple[DescendEdge, ...] | list[DescendEdge],
) -> dict[str, tuple[NavStep, ...] | None]:
    """Shortest command sequence from ``current`` to each target.

    Cost is the number of commands, then the number of parameters those commands
    cannot fill in, then the number of ``reset_context`` steps. An empty tuple
    means the caller is already there. ``None`` means nothing declared reaches it.
    """
    if not live_chain or live_chain[0] != current:
        live_chain = (current, "*") if current != "*" else ("*",)

    graph: dict[str, list[_Move]] = {}

    def add(source: str, move: _Move) -> None:
        graph.setdefault(source, []).append(move)

    for here, parent in zip(live_chain, live_chain[1:]):
        add(here, _Move("go_up", parent, (), (), "ascend"))

    nodes = set(live_chain)
    for edge in edges:
        nodes.add(edge.source)
        nodes.add(edge.target)
        if edge.source == edge.target:
            continue
        add(edge.source, _Move(
            edge.command, edge.target, edge.needs, edge.required_to_enter, "descend"))
    for node in nodes:
        if node != "*":
            add(node, _Move("reset_context", "*", (), (), "reset"))

    for source in graph:
        graph[source].sort(key=lambda move: (
            len(move.needs) + len(move.required_to_enter),
            _KIND_ORDER[move.kind],
            move.command,
        ))

    best: dict[str, tuple[int, int, int]] = {current: (0, 0, 0)}
    prev: dict[str, tuple[str, _Move] | None] = {current: None}
    heap: list[tuple[int, int, int, str]] = [(0, 0, 0, current)]
    while heap:
        steps, caveats, resets, node = heapq.heappop(heap)
        if (steps, caveats, resets) != best.get(node):
            continue
        for move in graph.get(node, ()):
            cand = (
                steps + 1,
                caveats + len(move.needs) + len(move.required_to_enter),
                resets + (1 if move.kind == "reset" else 0),
            )
            if move.target not in best or cand < best[move.target]:
                best[move.target] = cand
                prev[move.target] = (node, move)
                heapq.heappush(heap, (*cand, move.target))

    planned: dict[str, tuple[NavStep, ...] | None] = {}
    for target in targets:
        if target not in best:
            planned[target] = None
            continue
        steps_rev: list[NavStep] = []
        node = target
        while prev[node] is not None:
            source, move = prev[node]
            steps_rev.append(NavStep(
                move.command, move.target, move.needs, move.required_to_enter))
            node = source
        steps_rev.reverse()
        planned[target] = tuple(steps_rev)
    return planned


def navigation_steps_to_context(workflow, current: str, target: str) -> str | None:
    """The command sequence to reach *target* from *current*, or None if undeclared.

    Returns an empty string when *current* is already *target*. Returns ``None``
    when no path exists.
    """
    if current == target:
        return ""
    try:
        chain = live_context_chain(workflow)
        edges = collect_descend_edges(workflow.folderpath)
        path = plan_navigation(current, [target], chain, edges).get(target)
    except Exception:
        logger.debug("navigation steps not computed", exc_info=True)
        return None
    if path is None:
        return None
    if not path:
        return ""
    return ", then ".join(_format_step(step) for step in path)


def format_how_to_enter_context(workflow, current: str, target: str) -> str:
    """Prescriptive steps to reach *target* from the workflow's current context."""
    current_label = display_context_name(current)
    target_label = display_context_name(target)
    steps = navigation_steps_to_context(workflow, current, target)
    if steps is None:
        return (
            f"How to enter {target_label} from {current_label}: "
            "no declared navigation path."
        )
    if steps == "":
        return f"You are already in {target_label}."
    return f"How to enter {target_label} from {current_label}: {steps}."


def render_unavailable_command(
    token: str,
    current: str,
    homes: list[str],
    paths: dict[str, tuple[NavStep, ...] | None],
) -> str:
    """The refusal the agent tool raises."""
    current_label = display_context_name(current)
    home_labels = ", ".join(repr(display_context_name(home)) for home in homes)
    text = (
        f"Command {token!r} is not available in the current context "
        f"{current_label!r}. It is available in: {home_labels}."
    )
    sentences = []
    for home in homes:
        label = display_context_name(home)
        path = paths.get(home)
        if path is None:
            sentences.append(
                f"No declared navigation path from {current_label!r} to {label!r}."
            )
        elif not path:
            sentences.append(
                f"{label!r} is the current context. Retry {token!r}."
            )
        else:
            rendered = ", then ".join(_format_step(step) for step in path)
            sentences.append(
                f"From here to {label!r}: {rendered}. Then retry {token!r}."
            )
    return text + " " + " ".join(sentences)


def unavailable_command_message(
    workflow, token: str, current: str, homes: list[str],
) -> str:
    """Path from ``workflow``'s current context to each context that has ``token``."""
    try:
        chain = live_context_chain(workflow)
        edges = collect_descend_edges(workflow.folderpath)
        paths = plan_navigation(current, homes, chain, edges)
    except Exception:
        logger.debug("navigation path not computed", exc_info=True)
        paths = {home: None for home in homes}
    return render_unavailable_command(token, current, homes, paths)


def enters_current_context(workflow, token: str, current: str) -> bool:
    """Whether ``token`` is a command that enters ``current``, the context the agent is in."""
    try:
        edges = collect_descend_edges(workflow.folderpath)
    except Exception:
        logger.debug("descend edges unread", exc_info=True)
        return False
    return any(edge.target == current
               and edge.command.split("/")[-1].lower() == token.lower()
               for edge in edges)


def render_already_in_context(token: str, current: str, workflow) -> str:
    """The refusal for a command that enters the context the agent is already in.

    Told only "not available here, go_up", a weak model goes up and enters the
    same context again, over and over. Say it is already there, and what it can
    run there instead.
    """
    from fastworkflow.context_identity import context_clause_for

    label = display_context_name(current)
    instance = context_clause_for(workflow).partition(" ")[2]
    try:
        names = RoutingRegistry.get_definition(workflow.folderpath).get_command_names(current)
        commands = sorted({name.split("/")[-1] for name in names} - {"wildcard"})
    except Exception:
        logger.debug("commands of %r unread", current, exc_info=True)
        commands = []
    text = f"You are already in context {label!r}" + (f" ({instance})" if instance else "")
    text += f": {token!r} is how you enter it."
    if commands:
        text += f" To work on it, run one of its commands: {', '.join(commands)}."
    text += f" To open a different one, go_up first, then retry {token!r}."
    return text


def collect_descend_edges(workflow_folderpath: str) -> tuple[DescendEdge, ...]:
    """Descend commands declared by ``descends_to`` and by ``enter_command``.

    ``enter_command`` supplies the parameter names ``descends_to`` does not
    (``open_account_by_uid <account_uid>``). It is an edge into its context unless
    the command's own ``descends_to`` names a different context.
    """
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
        contexts: dict[str, list[str]] = dict(routing.contexts)
    except Exception:
        logger.debug("routing definition unread", exc_info=True)
        return ()

    effects, occupiable = _navigation_effects(workflow_folderpath)
    raw: dict[tuple[str, str, str], _RawEdge] = {}

    for command_key, effect in effects.items():
        sources = _contexts_for_command_key(command_key, contexts)
        gate = effect.when_parameter_present
        for target in effect.declared_targets():
            if not _enterable(target, occupiable):
                continue
            for source in sources:
                if source == target:
                    continue
                shown = _shown_command(source, command_key, contexts)
                _put(raw, _RawEdge(source, shown, command_key, target, gate, ()))

    for context_name, names in contexts.items():
        if not _enterable(context_name, occupiable):
            continue
        for declaration in declared_entry_commands(workflow_folderpath, context_name):
            token, placeholders = _parse_enter(declaration)
            if not token:
                continue
            effect = _effect_for_token(token, effects, contexts)
            if effect is not None and context_name not in effect.declared_targets():
                continue
            for source, routing_name in _sources_for_token(token, contexts):
                if source == context_name:
                    continue
                shown = _shown_command(source, routing_name, contexts)
                _put(raw, _RawEdge(
                    source, shown, routing_name, context_name, None, placeholders))

    schema: dict[str, tuple[str, ...] | None] = {}
    edges: dict[tuple[str, str, str], DescendEdge] = {}
    for record in raw.values():
        if record.routing_name not in schema:
            schema[record.routing_name] = _required_parameters(routing, record.routing_name)
        needs, required = _classify(
            schema[record.routing_name], record.placeholders, record.gate)
        key = (record.source, record.command, record.target)
        edge = DescendEdge(record.source, record.command, record.target, needs, required)
        existing = edges.get(key)
        edges[key] = edge if existing is None else _merge_edge(existing, edge)
    return tuple(edges.values())


@dataclass(frozen=True)
class _RawEdge:
    source: str
    command: str
    routing_name: str
    target: str
    gate: str | None
    placeholders: tuple[str, ...]


def _put(raw: dict[tuple[str, str, str], _RawEdge], record: _RawEdge) -> None:
    key = (record.source, record.command, record.target)
    existing = raw.get(key)
    if existing is None:
        raw[key] = record
        return
    raw[key] = _RawEdge(
        record.source,
        record.command,
        existing.routing_name or record.routing_name,
        record.target,
        existing.gate or record.gate,
        _union(existing.placeholders, record.placeholders),
    )


@dataclass(frozen=True)
class _DeclaredDescend:
    """A command's ``descends_to`` attribute."""

    target: str
    when_parameter_present: str | None = None

    def declared_targets(self) -> tuple[str, ...]:
        return (self.target,)


def _declared_descend(routing, command_name: str) -> _DeclaredDescend | None:
    """The ``descends_to`` a command declares on its response-generator class, or None.

    The class is read through the same routing lookup as the other command
    facts. ``descend_parameter`` is the parameter whose presence makes the command
    descend; without it the command always descends.
    """
    try:
        module = routing.get_command_class(
            command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
    except Exception:
        logger.debug("descend declaration unread for %r", command_name, exc_info=True)
        return None
    target = getattr(module, "descends_to", None)
    if not isinstance(target, str) or not target.strip():
        return None
    gate = getattr(module, "descend_parameter", None)
    gate = gate.strip() if isinstance(gate, str) and gate.strip() else None
    return _DeclaredDescend(target.strip(), gate)


def _navigation_effects(workflow_folderpath: str):
    """The ``descends_to`` commands and the ``occupiable`` contexts of a workflow.

    Both are read from the workflow's own classes: ``occupiable`` on a context's
    callback class, ``descends_to`` on a command's response-generator class. A
    context or command that declares nothing is absent from the result.
    """
    effects: dict = {}
    occupiable: dict = {}
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
    except Exception:
        logger.debug("routing definition unread", exc_info=True)
        return effects, occupiable
    for context_name, names in routing.contexts.items():
        declared = declared_occupiable(workflow_folderpath, context_name)
        if declared is not None:
            occupiable[context_name] = declared
        for name in names:
            descend = _declared_descend(routing, name)
            if descend is not None:
                effects[name] = descend
    return effects, occupiable


def _enterable(context_name: str, occupiable: dict[str, bool | None]) -> bool:
    """A context declared ``occupiable = False`` is a mixin, not a place."""
    return occupiable.get(context_name) is not False


def _contexts_for_command_key(
    command_key: str, contexts: dict[str, list[str]],
) -> tuple[str, ...]:
    """Contexts whose command list contains this routing name."""
    return tuple(name for name, names in contexts.items() if command_key in names)


def _sources_for_token(
    token: str, contexts: dict[str, list[str]],
) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for context_name, names in contexts.items():
        for name in names:
            if name.split("/")[-1] == token:
                found.append((context_name, name))
                break
    return tuple(found)


def _effect_for_token(token: str, effects: dict, contexts: dict[str, list[str]]):
    for command_key, effect in effects.items():
        if command_key.split("/")[-1] != token:
            continue
        if _contexts_for_command_key(command_key, contexts):
            return effect
    return None


def _shown_command(
    source: str, routing_name: str, contexts: dict[str, list[str]],
) -> str:
    """Short name, unless this context has two commands with that tail."""
    short = routing_name.split("/")[-1]
    collisions = [
        name for name in contexts.get(source, ())
        if name.split("/")[-1] == short
    ]
    if len(collisions) <= 1:
        return short
    if routing_name in collisions:
        return routing_name
    return collisions[0]


def _parse_enter(declaration: str) -> tuple[str, tuple[str, ...]]:
    parts = declaration.split(None, 1)
    if not parts:
        return "", ()
    return parts[0], tuple(_PLACEHOLDER.findall(declaration))


def _required_parameters(routing, command_name: str) -> tuple[str, ...] | None:
    """Input fields with no default. ``None`` when the signature cannot be read.

    ``None`` keeps the declaration's placeholders. An empty tuple means the
    signature was read and the command can be invoked with no arguments, which
    is a different fact from not having looked.
    """
    try:
        model = routing.get_command_class(
            command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:
        logger.debug("parameters unread for %r", command_name, exc_info=True)
        return None
    fields = getattr(model, "model_fields", None)
    if not fields:
        return ()
    required = []
    for name, field in fields.items():
        is_required = getattr(field, "is_required", None)
        if callable(is_required) and is_required():
            required.append(name)
    return tuple(required)


def _classify(
    schema_required: tuple[str, ...] | None,
    placeholders: tuple[str, ...],
    gate: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split parameters into 'must pass to run' and 'must pass to enter'.

    A required input is ``needs``. ``when_parameter_present`` on an optional
    field is the second kind: the command runs without it and stays put.
    Placeholders on ``enter_command`` join that second kind when the schema
    does not already call them required, so a signature that failed to import
    still names the value.
    """
    needs = tuple(schema_required or ())
    required: list[str] = []
    if gate and gate not in needs:
        required.append(gate)
    for name in placeholders:
        if name in needs or name in required:
            continue
        required.append(name)
    return needs, tuple(required)


def _union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for name in (*left, *right):
        if name not in seen:
            seen.add(name)
            merged.append(name)
    return tuple(merged)


def _merge_edge(left: DescendEdge, right: DescendEdge) -> DescendEdge:
    needs = _union(left.needs, right.needs)
    required = tuple(
        name for name in _union(left.required_to_enter, right.required_to_enter)
        if name not in needs
    )
    return DescendEdge(left.source, left.command, left.target, needs, required)


def _format_step(step: NavStep) -> str:
    holes = list(step.needs)
    holes.extend(name for name in step.required_to_enter if name not in step.needs)
    text = step.command
    if holes:
        text += "".join(f" <{name}>" for name in holes)
    notes = [f"needs {name}" for name in step.needs]
    if step.required_to_enter:
        target = display_context_name(step.enters)
        joined = ", ".join(step.required_to_enter)
        notes.append(f"{joined} required to enter {target}")
    if notes:
        text += " (" + "; ".join(notes) + ")"
    return text
