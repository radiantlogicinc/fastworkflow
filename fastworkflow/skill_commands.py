"""Skills as parameterized commands.

A skill (``_skills/<name>/SKILL.md``) may declare typed parameters in front matter.
The planner then fills the parameters from the request (one typed dspy call) and
code expands the skill into its numbered command steps; a skill without front matter
is plain text, exactly as before (see ``workflow_agent.load_skills``).

Format::

    ---
    params:
      name:
        type: str
        strip: group, unit
        desc: the record's name exactly as the request writes it
      kind:
        type: enum[group, unit]
        desc: what the request is about
      person:
        type: str
        default: ""
        desc: one named person to check for; empty for everyone
      include_risk:
        type: bool
        desc: the request asks how risky it is
      count:
        type: int
        default: 2
        desc: how many members to open
    maps:
      find:
        by: kind
        values:
          group: find_group
          unit: find_unit
    ---
    Show one named group or unit. One job per record.
    {find} query={name}
    [if person] list_members filter={person}
    [if not person] list_members
    [if include_risk] assess_risk
    [repeat count]
    open member {n}
    [end]

- Front matter sits between two ``---`` lines at the very top. It is a small,
  strict, indentation-based subset of YAML: ``key: value`` lines and ``key:`` blocks;
  a value is the rest of the line, or a JSON string when written in double quotes
  (``""`` is the empty string). Top-level keys: ``params`` (required) and ``maps``.
- A parameter has ``type`` (``str``, ``bool``, ``int`` or ``enum[a, b, ...]``), and
  optionally ``default``, ``desc``, ``requires`` (a condition: when it is false the
  argument is reset to its default) and, for a ``str``, ``strip`` (words removed from
  either end of the argument, with a possessive ``'s`` before a trailing one: a kind
  word the model appends to a name, "HR feed" -> "HR"). A parameter without a default
  is required. A ``bool`` always defaults to false (a ``default: true`` fails the load):
  the model sets it true only when the request asks for that part.
- A map turns the argument of one ``enum``, ``bool`` or ``int`` parameter (``by``) into
  text: ``values`` maps each argument to its text, ``default`` covers the rest; an
  argument with neither renders as itself. Map texts may hold placeholders.
- The body's first line is the description (what the skill matcher sees); every
  other non-empty line is one step. ``{param}`` or ``{map}`` is replaced by its value
  (``{{`` and ``}}`` are literal braces). A step may start with a condition
  ``[if <cond>]``: ``<cond>`` combines ``param``, ``param == "value"``, ``param != "value"``
  (an int compares with a number) with ``not``, ``and``, ``or`` and parentheses; a bare
  str is true when non-empty, an int when non-zero, an enum must be compared.
  ``[repeat <int param>]`` or ``[repeat <int param> - <k>]`` on its own line opens a block
  closed by ``[end]``; its steps run that many times (at most ``REPEAT_LIMIT``), and
  ``{n}`` in them numbers the iterations ``k+1 .. <param>``. Blocks do not nest.
- ``@<skill>`` lines are included as plain text (no placeholders or markers are read in
  them); a skill with front matter cannot be included.

Any malformed skill fails the load with a ``SkillFormatError`` naming the file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Literal, Union

#: The most iterations one ``[repeat]`` block may expand to.
REPEAT_LIMIT = 20
#: Names a parameter or map may not take (they are signature fields or the repeat index).
RESERVED_NAMES = frozenset({"user_query", "available_commands", "steps_before", "steps_after",
                            "reasoning", "n"})

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{(" + _NAME + r")\}|[{}]")
_COMMAND_LINE = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*$", re.M)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SkillFormatError(ValueError):
    """A malformed SKILL.md (front matter, condition, repeat block or placeholder)."""


class SkillCallError(ValueError):
    """A skill call that cannot be used (bad argument, failed guard); the planner falls back."""


# ---------------------------------------------------------------- the parsed skill

@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "bool" | "int" | "enum"
    values: tuple[str, ...] = ()
    default: object = None
    required: bool = False
    desc: str = ""
    requires: "Cond | None" = None
    strip: tuple[str, ...] = ()

    def type_text(self) -> str:
        return "|".join(self.values) if self.type == "enum" else self.type


@dataclass(frozen=True)
class ValueMap:
    name: str
    by: str
    values: tuple[tuple[object, str], ...]
    default: str | None = None

    def text_for(self, value) -> str | None:
        for key, text in self.values:
            if key == value:
                return text
        return self.default


@dataclass(frozen=True)
class Cond:
    source: str
    tree: tuple


@dataclass(frozen=True)
class Step:
    text: str
    cond: Cond | None = None
    literal: bool = False


@dataclass(frozen=True)
class Repeat:
    count: str
    offset: int
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class SkillCommand:
    """A skill with parameters: its signature, maps and step templates."""

    name: str
    description: str
    params: tuple[Param, ...]
    maps: tuple[ValueMap, ...]
    body: tuple[Union[Step, Repeat], ...]
    path: str = field(default="", compare=False)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def value_map(self, name: str) -> ValueMap | None:
        return next((m for m in self.maps if m.name == name), None)

    def signature_text(self) -> str:
        """``name(p: type = default, ...)`` with the description, parameter docs and steps."""
        parts = []
        for p in self.params:
            s = f"{p.name}: {p.type_text()}"
            if not p.required:
                s += f" = {json.dumps(p.default)}"
            parts.append(s)
        lines = [f"{self.name}({', '.join(parts)})", f"  {self.description}"]
        lines += [f"  - {p.name}: {p.desc}" for p in self.params]
        lines.append("  expands to:")
        lines += [f"    {i}. {s}" for i, s in enumerate(template_steps(self), 1)]
        return "\n".join(lines)


# ---------------------------------------------------------------- front matter

def _scalar(text: str, where: str):
    text = text.strip()
    if text.startswith('"'):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SkillFormatError(f"{where}: bad quoted value {text!r} ({exc.msg})") from None
        if not isinstance(value, str):
            raise SkillFormatError(f"{where}: bad quoted value {text!r}")
        return value
    return text


def parse_front_matter(lines: list[tuple[int, str]], where: str) -> dict:
    """The nested mapping of indented ``key: value`` / ``key:`` lines (see the module doc)."""
    root: dict = {}
    # (indent of the block's key, the block, indent of its children or None)
    stack: list[list] = [[-1, root, None]]
    for lineno, raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lead = raw[:len(raw) - len(raw.lstrip())]
        if "\t" in lead:
            raise SkillFormatError(f"{where} line {lineno}: indent with spaces, not tabs")
        indent = len(lead)
        m = re.fullmatch(r"([A-Za-z0-9_][\w-]*)\s*:(?:\s+(.*?))?\s*", raw.strip())
        if m is None:
            raise SkillFormatError(f"{where} line {lineno}: expected 'key: value' or 'key:', got {raw.strip()!r}")
        while indent <= stack[-1][0]:
            stack.pop()
        top = stack[-1]
        if top[2] is None:
            top[2] = indent
        elif indent != top[2]:
            raise SkillFormatError(f"{where} line {lineno}: inconsistent indentation")
        key, value = m.group(1), m.group(2)
        if key in top[1]:
            raise SkillFormatError(f"{where} line {lineno}: duplicate key {key!r}")
        if value is None or value == "":
            child: dict = {}
            top[1][key] = child
            stack.append([indent, child, None])
        else:
            top[1][key] = _scalar(value, f"{where} line {lineno}")
    return root


def _need_block(value, what: str, where: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise SkillFormatError(f"{where}: {what} must be a non-empty block")
    return value


def _check_keys(block: dict, allowed: set[str], what: str, where: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise SkillFormatError(f"{where}: {what} has unknown key(s) {', '.join(unknown)}")


def _parse_param(name: str, spec, where: str, conds: list) -> Param:
    if not re.fullmatch(_NAME, name) or name in RESERVED_NAMES:
        raise SkillFormatError(f"{where}: bad parameter name {name!r}")
    if not isinstance(spec, dict):
        raise SkillFormatError(f"{where}: parameter {name!r} must be a block with a type")
    _check_keys(spec, {"type", "default", "desc", "requires", "strip"}, f"parameter {name!r}", where)
    type_text = spec.get("type")
    if not isinstance(type_text, str):
        raise SkillFormatError(f"{where}: parameter {name!r} needs a type")
    values: tuple[str, ...] = ()
    m = re.fullmatch(r"enum\[(.*)\]", type_text.strip())
    if m:
        values = tuple(v.strip() for v in m.group(1).split(","))
        if not values or any(not re.fullmatch(r"[\w-]+", v) for v in values) or len(set(values)) != len(values):
            raise SkillFormatError(f"{where}: parameter {name!r} has a bad enum {type_text!r}")
        kind = "enum"
    elif type_text.strip() in ("str", "bool", "int"):
        kind = type_text.strip()
    else:
        raise SkillFormatError(f"{where}: parameter {name!r} has unknown type {type_text!r} "
                               "(str, bool, int or enum[...])")
    desc = spec.get("desc", "")
    if not isinstance(desc, str):
        raise SkillFormatError(f"{where}: parameter {name!r} desc must be text")
    raw_default = spec.get("default")
    if isinstance(raw_default, dict):
        raise SkillFormatError(f"{where}: parameter {name!r} default must be a value")
    required = raw_default is None and kind != "bool"
    default: object = None
    if kind == "bool":
        if raw_default not in (None, "false"):
            raise SkillFormatError(f"{where}: bool parameter {name!r} must default to false "
                                   "(the model sets it true only when the request asks)")
        default = False
    elif raw_default is not None:
        if kind == "int":
            if not re.fullmatch(r"-?\d+", raw_default):
                raise SkillFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not an int")
            default = int(raw_default)
        elif kind == "enum":
            if raw_default not in values:
                raise SkillFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not one of its values")
            default = raw_default
        else:
            default = raw_default
    strip: tuple[str, ...] = ()
    if "strip" in spec:
        if kind != "str" or not isinstance(spec["strip"], str):
            raise SkillFormatError(f"{where}: only a str parameter takes strip words ({name!r})")
        strip = tuple(w.strip() for w in spec["strip"].split(",") if w.strip())
    requires = spec.get("requires")
    if requires is not None and not isinstance(requires, str):
        raise SkillFormatError(f"{where}: parameter {name!r} requires must be a condition")
    param = Param(name=name, type=kind, values=values, default=default, required=required,
                  desc=desc, strip=strip)
    if requires:
        conds.append((name, requires))
    return param


def _map_key(param: Param, key: str, where: str):
    if param.type == "enum":
        if key not in param.values:
            raise SkillFormatError(f"{where}: map key {key!r} is not a value of {param.name!r}")
        return key
    if param.type == "bool":
        if key not in ("true", "false"):
            raise SkillFormatError(f"{where}: map key {key!r} of bool {param.name!r} must be true or false")
        return key == "true"
    if not re.fullmatch(r"-?\d+", key):
        raise SkillFormatError(f"{where}: map key {key!r} of int {param.name!r} is not an int")
    return int(key)


def _parse_map(name: str, spec, params: dict[str, Param], where: str) -> ValueMap:
    if not re.fullmatch(_NAME, name) or name in RESERVED_NAMES or name in params:
        raise SkillFormatError(f"{where}: bad map name {name!r} (a parameter or reserved name?)")
    if not isinstance(spec, dict):
        raise SkillFormatError(f"{where}: map {name!r} must be a block with by and values")
    _check_keys(spec, {"by", "values", "default"}, f"map {name!r}", where)
    by = spec.get("by")
    if by not in params or params[by].type == "str":
        raise SkillFormatError(f"{where}: map {name!r} must be by an enum, bool or int parameter")
    values = _need_block(spec.get("values"), f"map {name!r} values", where)
    entries = []
    for key, text in values.items():
        if not isinstance(text, str):
            raise SkillFormatError(f"{where}: map {name!r} value for {key!r} must be text")
        entries.append((_map_key(params[by], key, where), text))
    default = spec.get("default")
    if default is not None and not isinstance(default, str):
        raise SkillFormatError(f"{where}: map {name!r} default must be text")
    return ValueMap(name=name, by=by, values=tuple(entries), default=default)


# ---------------------------------------------------------------- conditions

_COND_TOKEN = re.compile(r'\s*(?:(\()|(\))|(==|!=)|("(?:[^"\\]|\\.)*")|(-?\d+)|(' + _NAME + r'))')


def _tokens(source: str, where: str) -> list[tuple[str, object]]:
    out, i = [], 0
    while i < len(source):
        if source[i:].strip() == "":
            break
        m = _COND_TOKEN.match(source, i)
        if m is None:
            raise SkillFormatError(f"{where}: cannot read condition {source!r} at {source[i:]!r}")
        lp, rp, op, string, number, ident = m.groups()
        if lp:
            out.append(("(", None))
        elif rp:
            out.append((")", None))
        elif op:
            out.append(("op", op))
        elif string:
            out.append(("str", json.loads(string)))
        elif number:
            out.append(("int", int(number)))
        elif ident in ("and", "or", "not"):
            out.append((ident, None))
        else:
            out.append(("name", ident))
        i = m.end()
    return out


def parse_condition(source: str, params: dict[str, Param], where: str) -> Cond:
    tokens = _tokens(source, where)
    pos = 0

    def peek():
        return tokens[pos][0] if pos < len(tokens) else None

    def take(kind):
        nonlocal pos
        if peek() != kind:
            raise SkillFormatError(f"{where}: bad condition {source!r}")
        pos += 1
        return tokens[pos - 1][1]

    def disjunction():
        node = conjunction()
        while peek() == "or":
            take("or")
            node = ("or", node, conjunction())
        return node

    def conjunction():
        node = negation()
        while peek() == "and":
            take("and")
            node = ("and", node, negation())
        return node

    def negation():
        if peek() == "not":
            take("not")
            return ("not", negation())
        if peek() == "(":
            take("(")
            node = disjunction()
            take(")")
            return node
        name = take("name")
        param = params.get(name)
        if param is None:
            raise SkillFormatError(f"{where}: condition {source!r} names unknown parameter {name!r}")
        if peek() != "op":
            if param.type == "enum":
                raise SkillFormatError(f"{where}: compare enum {name!r} with == or != in {source!r}")
            return ("var", name)
        op = take("op")
        if param.type == "bool":
            raise SkillFormatError(f"{where}: test bool {name!r} as '{name}' or 'not {name}' in {source!r}")
        if param.type == "int":
            value = take("int")
        else:
            value = take("str")
            if param.type == "enum" and value not in param.values:
                raise SkillFormatError(f"{where}: {value!r} is not a value of {name!r} in {source!r}")
        return ("eq" if op == "==" else "ne", name, value)

    if not tokens:
        raise SkillFormatError(f"{where}: empty condition")
    tree = disjunction()
    if pos != len(tokens):
        raise SkillFormatError(f"{where}: bad condition {source!r}")
    return Cond(source=source.strip(), tree=tree)


def evaluate(cond: Cond | None, env: dict) -> bool:
    def ev(node) -> bool:
        op = node[0]
        if op == "var":
            return bool(env[node[1]])
        if op == "eq":
            return env[node[1]] == node[2]
        if op == "ne":
            return env[node[1]] != node[2]
        if op == "not":
            return not ev(node[1])
        if op == "and":
            return ev(node[1]) and ev(node[2])
        return ev(node[1]) or ev(node[2])
    return True if cond is None else ev(cond.tree)


def _condition_words(cond: Cond, command: "SkillCommand") -> str:
    def is_str(node) -> bool:
        param = command.param(node[1]) if node[0] == "var" else None
        return param is not None and param.type == "str"

    def words(node) -> str:
        op = node[0]
        if op == "var":
            return f"{node[1]} is given" if is_str(node) else node[1]
        if op in ("eq", "ne"):
            return f"{node[1]} is {'' if op == 'eq' else 'not '}{node[2]}"
        if op == "not":
            return f"{node[1][1]} is empty" if is_str(node[1]) else f"not {words(node[1])}"
        return f"({words(node[1])} {op} {words(node[2])})"
    text = words(cond.tree)
    return text[1:-1] if text.startswith("(") and text.endswith(")") else text


# ---------------------------------------------------------------- body

def _placeholders(text: str, names: set[str], where: str) -> None:
    for m in _PLACEHOLDER.finditer(text):
        token = m.group(0)
        if token in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise SkillFormatError(f"{where}: stray brace in {text!r} (write {{{{ or }}}} for a literal one)")
        if m.group(1) not in names:
            raise SkillFormatError(f"{where}: unknown placeholder {{{m.group(1)}}} in {text!r}")


def _parse_body(items: list[tuple[str, bool]], params: dict[str, Param], map_names: set[str],
                where: str) -> tuple[Union[Step, Repeat], ...]:
    body: list = []
    block: list | None = None
    block_head: tuple[str, int] | None = None
    names = set(params) | map_names
    for text, literal in items:
        if literal:
            (block if block is not None else body).append(Step(text=text, literal=True))
            continue
        if text.startswith("[repeat") or text.startswith("[end") or text.startswith("[if"):
            m_rep = re.fullmatch(r"\[repeat\s+(" + _NAME + r")(?:\s*-\s*(\d+))?\s*\]", text)
            m_end = re.fullmatch(r"\[end\]", text)
            m_if = re.fullmatch(r"\[if\s+([^\]]+)\]\s*(.*)", text)
            if m_rep:
                if block is not None:
                    raise SkillFormatError(f"{where}: [repeat] blocks do not nest ({text!r})")
                param = params.get(m_rep.group(1))
                if param is None or param.type != "int":
                    raise SkillFormatError(f"{where}: {text!r} must repeat an int parameter")
                block, block_head = [], (param.name, int(m_rep.group(2) or 0))
                continue
            if m_end:
                if block is None:
                    raise SkillFormatError(f"{where}: [end] without [repeat]")
                if not block:
                    raise SkillFormatError(f"{where}: empty [repeat] block")
                body.append(Repeat(count=block_head[0], offset=block_head[1], steps=tuple(block)))
                block, block_head = None, None
                continue
            if m_if and m_if.group(2).strip():
                cond = parse_condition(m_if.group(1), params, where)
                step_text = m_if.group(2).strip()
                _placeholders(step_text, names | ({"n"} if block is not None else set()), where)
                (block if block is not None else body).append(Step(text=step_text, cond=cond))
                continue
            raise SkillFormatError(f"{where}: malformed marker line {text!r}")
        _placeholders(text, names | ({"n"} if block is not None else set()), where)
        (block if block is not None else body).append(Step(text=text))
    if block is not None:
        raise SkillFormatError(f"{where}: [repeat] without [end]")
    if not body:
        raise SkillFormatError(f"{where}: a skill with params needs steps")
    return tuple(body)


def has_front_matter(lines: list[str]) -> bool:
    return bool(lines) and lines[0].strip() == "---"


def split_front_matter(raw_lines: list[str], where: str) -> tuple[list[tuple[int, str]], list[str]]:
    """``(front matter lines with numbers, body lines)`` of a file that starts with ``---``."""
    for i in range(1, len(raw_lines)):
        if raw_lines[i].strip() == "---":
            front = [(n + 1, raw_lines[n].rstrip("\n")) for n in range(1, i)]
            return front, raw_lines[i + 1:]
    raise SkillFormatError(f"{where}: front matter has no closing '---'")


def build_command(name: str, description: str, front: list[tuple[int, str]],
                  items: list[tuple[str, bool]], where: str) -> SkillCommand:
    """The SkillCommand of a skill with front matter; ``items`` are its steps, (text, literal)."""
    spec = parse_front_matter(front, where)
    _check_keys(spec, {"params", "maps"}, "front matter", where)
    raw_params = _need_block(spec.get("params"), "params", where)
    conds: list[tuple[str, str]] = []
    params = {n: _parse_param(n, s, where, conds) for n, s in raw_params.items()}
    raw_maps = spec.get("maps", {})
    if not isinstance(raw_maps, dict):
        raise SkillFormatError(f"{where}: maps must be a block")
    maps = {n: _parse_map(n, s, params, where) for n, s in raw_maps.items()}
    names = set(params) | set(maps)
    for m in maps.values():
        for text in [t for _k, t in m.values] + ([m.default] if m.default is not None else []):
            _placeholders(text, names | {"n"}, where)
    _check_map_cycles(maps, where)
    for n, source in conds:
        params[n] = replace(params[n], requires=parse_condition(source, params, where))
    body = _parse_body(items, params, set(maps), where)
    return SkillCommand(name=name, description=description, params=tuple(params.values()),
                        maps=tuple(maps.values()), body=body, path=where)


def _check_map_cycles(maps: dict[str, ValueMap], where: str) -> None:
    def refs(m: ValueMap) -> set[str]:
        texts = [t for _k, t in m.values] + ([m.default] if m.default is not None else [])
        return {g for t in texts for g in re.findall(r"(?<!\{)\{(" + _NAME + r")\}", t) if g in maps}

    def visit(name, chain):
        if name in chain:
            raise SkillFormatError(f"{where}: maps refer to each other in a cycle ({' -> '.join(chain + (name,))})")
        for r in refs(maps[name]):
            visit(r, chain + (name,))
    for name in maps:
        visit(name, ())


# ---------------------------------------------------------------- expansion

def _text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render(command: SkillCommand, text: str, env: dict) -> str:
    def sub(m):
        token = m.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = m.group(1)
        if name in env:
            return _text(env[name])
        vmap = command.value_map(name)
        value = vmap.text_for(env[vmap.by])
        return _render(command, value, env) if value is not None else _text(env[vmap.by])
    return _PLACEHOLDER.sub(sub, text)


def expand(command: SkillCommand, args: dict) -> list[str]:
    """The skill's command steps for ``args`` (complete, checked arguments), unnumbered.

    Values are substituted, steps whose condition is false are dropped, repeat blocks are
    unrolled. Raises SkillCallError when a repeat would exceed REPEAT_LIMIT.
    """
    env = dict(args)
    out: list[str] = []

    def emit(step: Step, env: dict) -> None:
        if step.literal:
            out.append(step.text)
        elif evaluate(step.cond, env):
            out.append(_render(command, step.text, env))

    for node in command.body:
        if isinstance(node, Repeat):
            times = int(env[node.count]) - node.offset
            if times > REPEAT_LIMIT:
                raise SkillCallError(f"{node.count}={env[node.count]} repeats more than {REPEAT_LIMIT} times")
            for i in range(max(times, 0)):
                for step in node.steps:
                    emit(step, {**env, "n": node.offset + 1 + i})
        else:
            emit(node, env)
    return out


def number(steps: list[str]) -> str:
    """Steps as the planner writes them: a numbered list, one per line."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def template_steps(command: SkillCommand) -> tuple[str, ...]:
    """The skill's steps as plain text, for the planner template (``plan_template_for``).

    Placeholders read ``<param>``; a map reads its texts joined with "or" (``<find_a or
    find_b, by kind>``), a map by a bool its true text (``<text, when flag>``); a condition is
    a trailing "(only when ...)"; a repeat block is one line.
    """
    def show(text: str) -> str:
        def sub(m):
            token = m.group(0)
            if token in ("{{", "}}"):
                return token[0]
            name = m.group(1)
            if name == "n":
                return "<n>"
            vmap = command.value_map(name)
            if vmap is None:
                return f"<{name}>"
            if command.param(vmap.by).type == "bool":
                shown = show(vmap.text_for(True) or "").strip(" ;,")
                return f" <{shown}, when {vmap.by}>" if shown else ""
            texts = []
            for _key, value in vmap.values + (((None, vmap.default),) if vmap.default is not None else ()):
                shown = show(value).strip()
                if shown and shown not in texts:
                    texts.append(shown)
            return "<" + " or ".join(texts) + f", by {vmap.by}>" if texts else ""
        return _PLACEHOLDER.sub(sub, text)

    def line(step: Step) -> str:
        if step.literal:
            return step.text
        text = show(step.text)
        return f"{text} (only when {_condition_words(step.cond, command)})" if step.cond else text

    out = []
    for node in command.body:
        if isinstance(node, Repeat):
            count = node.count + (f" - {node.offset}" if node.offset else "")
            out.append(f"repeat {count} times (n = {node.offset + 1} to {node.count}): "
                       + "; then ".join(line(s) for s in node.steps))
        else:
            out.append(line(node))
    return tuple(out)


# ---------------------------------------------------------------- arguments and guards

def _clean_str(value) -> str:
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.rstrip(" .;,")


def _strip_words(text: str, words: tuple[str, ...]) -> str:
    if not words:
        return text
    lowered = {w.lower() for w in words}
    tokens = text.split()
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        if tokens[0].lower() in lowered:
            tokens, changed = tokens[1:], True
        elif tokens[-1].lower() in lowered:
            tokens, changed = tokens[:-1], True
            if tokens and re.search(r"['’]s$", tokens[-1]):
                tokens[-1] = tokens[-1][:-2]
    return " ".join(tokens).strip()


def _empty(value) -> bool:
    return value is None or str(value).strip().strip("\"'`").lower() in ("", "none", "null")


def check_arguments(command: SkillCommand, raw: dict, request: str) -> tuple[dict, list[str]]:
    """The guarded, complete arguments of a skill call; ``(args, notes)``.

    - a missing or empty argument takes its default; a required one fails the call;
    - an enum must be one of its values, an int an integer, a bool true/false;
    - a str argument loses surrounding quotes and its strip words, and must then be a
      verbatim (case-insensitive) substring of ``request``; it takes the request's own
      spelling. One that is not: an optional argument takes its default, a required one
      fails the call (the planner falls back to the template path);
    - an optional str argument equal to another str argument takes its default;
    - an argument whose ``requires`` condition is false takes its default.

    ``notes`` says what a guard changed. Raises SkillCallError.
    """
    notes: list[str] = []
    args: dict = {}
    lowered = request.lower()
    for p in command.params:
        value = raw.get(p.name)
        if _empty(value):
            if p.required:
                raise SkillCallError(f"missing required argument {p.name}")
            args[p.name] = p.default
            continue
        if p.type == "bool":
            if isinstance(value, str):
                if value.strip().lower() not in ("true", "false"):
                    raise SkillCallError(f"{p.name}={value!r} is not true or false")
                value = value.strip().lower() == "true"
            args[p.name] = bool(value)
        elif p.type == "int":
            try:
                args[p.name] = int(str(value).strip())
            except ValueError:
                raise SkillCallError(f"{p.name}={value!r} is not an integer") from None
        elif p.type == "enum":
            text = _clean_str(value).lower()
            if text not in p.values:
                raise SkillCallError(f"{p.name}={value!r} is not one of {', '.join(p.values)}")
            args[p.name] = text
        else:
            text = _strip_words(_clean_str(value), p.strip)
            at = lowered.find(text.lower()) if text else -1
            if at < 0:
                if p.required:
                    raise SkillCallError(f"{p.name}={value!r} is not written in the request")
                notes.append(f"{p.name}={value!r} is not in the request: default")
                args[p.name] = p.default
                continue
            exact = request[at:at + len(text)]
            if exact != str(value):
                notes.append(f"{p.name}={value!r} -> {exact!r}")
            args[p.name] = exact
    strs = [p for p in command.params if p.type == "str"]
    for p in strs:
        if p.required or args[p.name] == p.default:
            continue
        if any(o is not p and str(args[o.name]).lower() == str(args[p.name]).lower() for o in strs):
            notes.append(f"{p.name}={args[p.name]!r} repeats another argument: default")
            args[p.name] = p.default
    for p in command.params:
        if p.requires is not None and args[p.name] != p.default and not evaluate(p.requires, args):
            notes.append(f"{p.name}={args[p.name]!r} needs {p.requires.source}: default")
            args[p.name] = p.default
    return args, notes


def command_names(available_commands: str) -> frozenset[str]:
    """Command names of the planner's catalog text (its ``- <name>`` lines)."""
    return frozenset(_COMMAND_LINE.findall(available_commands or ""))


def filter_extra_steps(extra, expanded: list[str], names: frozenset[str], skill_name: str
                       ) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` of the model's steps_before/steps_after entries.

    Dropped: empty entries, entries naming the skill itself (the expansion already runs it),
    entries naming no available command, and entries whose every command the expansion
    already runs. With no command names known only the first two rules apply.
    """
    expanded_cmds = {w for s in expanded for w in _IDENT.findall(s) if w in names}
    kept, dropped = [], []
    for entry in extra or []:
        text = str(entry).strip()
        if not text:
            continue
        cmds = {w for w in _IDENT.findall(text) if w in names}
        if skill_name in text or (names and (not cmds or cmds <= expanded_cmds)):
            dropped.append(text)
        else:
            kept.append(text)
    return kept, dropped


# ---------------------------------------------------------------- the planner call

def _field_desc(p: Param) -> str:
    desc = p.desc or p.name
    if p.type == "bool":
        return f"{desc}. true only when the request explicitly asks for this; otherwise false"
    if p.type == "str":
        desc += ". Copied verbatim from the request"
    return desc + (" (required)" if p.required else f" (default {json.dumps(p.default)})")


def typed_signature(command: SkillCommand):
    """The skill's dspy signature: user_query in; one typed field per parameter, plus
    steps_before / steps_after, out."""
    import dspy

    fields: dict = {"user_query": (str, dspy.InputField())}
    for p in command.params:
        kind = {"str": str, "bool": bool, "int": int}.get(p.type)
        if p.type == "enum":
            kind = Literal[p.values]
        fields[p.name] = (kind, dspy.OutputField(desc=_field_desc(p)))
    fields["steps_before"] = (list[str], dspy.OutputField(
        desc="plain command steps to run before the skill, only for parts of the request it does not cover; usually []"))
    fields["steps_after"] = (list[str], dspy.OutputField(
        desc="plain command steps to run after the skill, only for parts of the request it does not cover; usually []"))
    instructions = (
        "Plan the user_query by calling one skill command and, only if needed, plain command steps.\n"
        "Fill the skill's parameters from the request: names and values exactly as the request writes them "
        "(never add a word the request does not attach to the name); a boolean is true only when the request "
        "explicitly asks for that part, otherwise false; an enum is what the request is about; leave a "
        "parameter empty to take its default.\n"
        "Code expands the skill into its command steps. steps_before / steps_after are only for parts of the "
        "request the expansion does not cover, each a short sentence naming one available command; never "
        "repeat a command the expansion already runs; usually both are empty.\n\n"
        f"Skill command:\n{command.signature_text()}")
    return dspy.make_signature(fields, instructions)


@dataclass
class SkillCallPlan:
    plan: str
    args: dict
    notes: list[str]
    prediction: object


def plan_skill_call(command: SkillCommand, request: str, available_commands: str) -> SkillCallPlan:
    """Plan ``request`` as a call of ``command``: the typed call, the guards, the expansion.

    Runs under the caller's dspy context (the planner LM and adapter); ``available_commands``
    reaches the adapter as extra input, as for the text planner. Raises on any failure
    (the caller falls back to the template path).
    """
    import dspy

    prediction = dspy.ChainOfThought(typed_signature(command))(
        user_query=request, available_commands=available_commands)
    raw = {p.name: getattr(prediction, p.name, None) for p in command.params}
    args, notes = check_arguments(command, raw, request)
    steps = expand(command, args)
    names = command_names(available_commands)
    before, dropped_before = filter_extra_steps(getattr(prediction, "steps_before", None), steps, names, command.name)
    after, dropped_after = filter_extra_steps(getattr(prediction, "steps_after", None), steps, names, command.name)
    notes += [f"dropped extra step {s!r}" for s in dropped_before + dropped_after]
    return SkillCallPlan(plan=number(before + steps + after), args=args, notes=notes, prediction=prediction)
