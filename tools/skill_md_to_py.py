#!/usr/bin/env python
"""Convert a workflow's ``_skills/<name>/SKILL.md`` files to ``skill.py`` skill files.

A development tool: it lives outside the ``fastworkflow`` package and is not installed with
it. Run it with the interpreter that has fastworkflow installed::

    python tools/skill_md_to_py.py PATH... [--write [--force]] [--check] [--remove-md]

``PATH`` is a SKILL.md, a skill folder (``_skills/<name>``), or a ``_skills`` folder (every
``<name>/SKILL.md`` under it).

- Without ``--write`` the generated ``skill.py`` of each SKILL.md is printed.
- ``--write`` writes ``skill.py`` next to each SKILL.md (an existing ``skill.py`` is kept
  unless ``--force``).
- ``--check`` loads every skill of each ``_skills`` folder twice, as SKILL.md and as the
  generated ``skill.py`` (in a temporary copy), and compares them: the description, the
  parameters, the template text, and the expanded steps over a grid of arguments
  (``argument_grid``). Any difference is printed and the exit status is 1.
- ``--remove-md`` (with ``--write`` and ``--check``) deletes each SKILL.md once its
  ``skill.py`` is written and the check passed.

Every construct the SKILL.md parser accepts converts: typed parameters with their options
(``Field(json_schema_extra=...)``), maps (dicts, or functions when a text holds a
placeholder), ``{param}`` placeholders (f-strings), ``[if cond]`` (``if``), ``[repeat p - k]``
(``for n in repeat(input.p, k)``), ``@<skill>`` includes (an ``"@<skill>"`` line, expanded by
the loader) and plain skills without params (``steps()`` returning fixed lines). The
template text the planner falls back to is written as ``Signature.template``, so it stays
what the SKILL.md gave.
"""
from __future__ import annotations

import argparse
import itertools
import json
import keyword
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

from fastworkflow import skill_commands as sc

GRID_LIMIT = 4096
_INCLUDE = re.compile(r"@(\S+)")


# ---------------------------------------------------------------- reading a SKILL.md

def read_skill_md(text: str, name: str, where: str = "SKILL.md"):
    """``(description, steps, command or None)`` of a SKILL.md's text.

    ``command`` (a skill with params) is parsed with each ``@<skill>`` line kept as a literal
    step, so the generated file includes the skill by name instead of copying its steps.
    """
    raw_lines = text.splitlines(keepends=True)
    front = None
    if sc.has_front_matter(raw_lines):
        front, raw_lines = sc.split_front_matter(raw_lines, where)
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        raise sc.SkillFormatError(f"{where}: no description line")
    description, steps = lines[0], lines[1:]
    if front is None:
        return description, steps, None
    items = [(s, bool(_INCLUDE.fullmatch(s))) for s in steps]
    return description, steps, sc.build_command(name, description, front, items, where)


# ---------------------------------------------------------------- Python source helpers

def _lit(text: str) -> str:
    """A Python string literal of ``text``: double-quoted, or single-quoted when that saves
    escaping a double quote."""
    if '"' in text and "'" not in text and json.dumps(text, ensure_ascii=False) == \
            '"' + text.replace('"', '\\"') + '"':
        return "'" + text + "'"
    return json.dumps(text, ensure_ascii=False)


def _value(value, indent: int) -> str:
    """A Python literal of a str/bool/int or a (nested) dict/list of them, one item per line."""
    pad = " " * indent
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _lit(value)
    if isinstance(value, dict):
        one_line = "{" + ", ".join(f"{_lit(k)}: {_value(v, indent)}" for k, v in value.items()) + "}"
        if "\n" not in one_line and len(one_line) + indent < 80:
            return one_line
        inner = "".join(f"{pad}    {_lit(k)}: {_value(v, indent + 4)},\n" for k, v in value.items())
        return "{\n" + inner + pad + "}"
    if isinstance(value, (list, tuple)):
        one_line = "[" + ", ".join(_value(v, indent) for v in value) + "]"
        if len(one_line) + indent < 90:
            return one_line
        inner = "".join(f"{pad}    {_value(v, indent + 4)},\n" for v in value)
        return "[\n" + inner + pad + "]"
    raise TypeError(value)


def _docstring(text: str, indent: int) -> str:
    pad = " " * indent
    body = text.replace("\\", "\\\\").replace('"', '\\"')
    if " ".join(text.split()) == text and len(body) + indent + 6 > 96:
        wrapped = textwrap.wrap(body, width=92 - indent, break_long_words=False, break_on_hyphens=False)
        return f'{pad}"""' + f"\n{pad}".join(wrapped) + f'\n{pad}"""'
    return f'{pad}"""{body}"""'


class _Source:
    """Python expressions for a skill's placeholders, conditions and maps."""

    def __init__(self, command: sc.SkillCommand):
        self.command = command
        self.maps = {m.name: m for m in command.maps}
        self.plain = {m.name for m in command.maps if not any(self._holds_placeholder(t) for t in self._texts(m))}

    @staticmethod
    def _texts(vmap: sc.ValueMap) -> list[str]:
        return [t for _k, t in vmap.values] + ([vmap.default] if vmap.default is not None else [])

    @staticmethod
    def _holds_placeholder(text: str) -> bool:
        return any(m.group(1) for m in sc._PLACEHOLDER.finditer(text))

    @staticmethod
    def _unescape(text: str) -> str:
        return text.replace("{{", "{").replace("}}", "}")

    def needs_n(self, name: str, seen: frozenset = frozenset()) -> bool:
        if name == "n":
            return True
        vmap = self.maps.get(name)
        if vmap is None or name in seen:
            return False
        refs = {m.group(1) for t in self._texts(vmap) for m in sc._PLACEHOLDER.finditer(t) if m.group(1)}
        return any(self.needs_n(r, seen | {name}) for r in refs)

    def const_name(self, name: str) -> str:
        return name.upper()

    def func_name(self, name: str) -> str:
        return f"{name}_text"

    def as_text(self, param: sc.Param) -> str:
        """The expression of a parameter's value as text, as the SKILL.md renders it."""
        if param.type == "bool":
            return f"'true' if input.{param.name} else 'false'"
        if param.type == "int":
            return f"str(input.{param.name})"
        return f"input.{param.name}"

    def expr(self, name: str, in_loop: bool) -> str:
        if name == "n":
            return "n"
        param = self.command.param(name)
        if param is not None:
            return self.as_text(param) if param.type == "bool" else f"input.{name}"
        vmap = self.maps[name]
        if name not in self.plain:
            call_n = ", n" if in_loop and self.needs_n(name) else ""
            return f"{self.func_name(name)}(input{call_n})"
        by = self.command.param(vmap.by)
        keys = {k for k, _ in vmap.values}
        total = (by.type == "enum" and keys >= set(by.values)) or (by.type == "bool" and keys >= {True, False})
        if total:
            return f"{self.const_name(name)}[input.{by.name}]"
        fallback = f"{self.const_name(name)}_DEFAULT" if vmap.default is not None else self.as_text(by)
        if " " in fallback:
            fallback = f"({fallback})"
        return f"{self.const_name(name)}.get(input.{by.name}, {fallback})"

    def string(self, text: str, in_loop: bool) -> str:
        """A Python string (an f-string when it holds a placeholder) rendering a step template."""
        if not self._holds_placeholder(text):
            return _lit(self._unescape(text))
        whole = sc._PLACEHOLDER.fullmatch(text)
        if whole and whole.group(1) and whole.group(1) != "n":
            param = self.command.param(whole.group(1))
            if param is None or param.type == "str":   # a map's text or a str: already text
                return self.expr(whole.group(1), in_loop)
        parts = []
        pos = 0
        for m in sc._PLACEHOLDER.finditer(text):
            parts.append(self._literal_part(text[pos:m.start()]))
            token = m.group(0)
            parts.append("{{" if token == "{{" else "}}" if token == "}}" else "{" + self.expr(m.group(1), in_loop) + "}")
            pos = m.end()
        parts.append(self._literal_part(text[pos:]))
        return 'f"' + "".join(parts) + '"'

    @staticmethod
    def _literal_part(text: str) -> str:
        body = json.dumps(text, ensure_ascii=False)[1:-1]
        return body.replace("{", "{{").replace("}", "}}")

    def cond(self, node) -> str:
        op = node[0]
        if op == "var":
            return f"input.{node[1]}"
        if op in ("eq", "ne"):
            value = node[2] if isinstance(node[2], int) and not isinstance(node[2], bool) else _lit(node[2])
            return f"input.{node[1]} {'==' if op == 'eq' else '!='} {value}"
        if op == "not":
            inner = self.cond(node[1])
            return f"not ({inner})" if node[1][0] in ("and", "or") else f"not {inner}"
        left, right = self.cond(node[1]), self.cond(node[2])
        if op == "and":
            left = f"({left})" if node[1][0] == "or" else left
            right = f"({right})" if node[2][0] == "or" else right
        return f"{left} {op} {right}"

    def key_test(self, by: sc.Param, key) -> str:
        if by.type == "bool":
            return f"input.{by.name}" if key else f"not input.{by.name}"
        return f"input.{by.name} == {key if by.type == 'int' else _lit(key)}"


# ---------------------------------------------------------------- generating skill.py

def _check_names(command: sc.SkillCommand, where: str) -> None:
    from pydantic import BaseModel

    for p in command.params:
        if keyword.iskeyword(p.name) or p.name.startswith("model_") or hasattr(BaseModel, p.name):
            raise sc.SkillFormatError(f"{where}: parameter {p.name!r} cannot be a pydantic field name; "
                                      "rename it in the SKILL.md first")


def _field(p: sc.Param) -> str:
    if p.type == "enum":
        annotation = "Literal[" + ", ".join(_lit(v) for v in p.values) + "]"
    else:
        annotation = p.type
    args = []
    if not p.required:
        args.append(f"default={_value(p.default, 12)}")
    if p.desc:
        args.append(f"description={_lit(p.desc)}")
    extra: dict = {}
    if p.strip:
        extra["strip"] = list(p.strip)
    for key in ("subject", "verify", "in_request"):
        if getattr(p, key):
            extra[key] = True
    if p.means:
        extra["means"] = dict(p.means)
    if p.requires is not None:
        extra["requires"] = p.requires.source
    if extra:
        args.append(f"json_schema_extra={_value(extra, 12)}")
    head = f"        {p.name}: {annotation} = Field("
    one_line = head + ", ".join(args) + ")"
    if "\n" not in one_line and len(one_line) <= 100:
        return one_line
    return head + "\n" + "".join(f"            {a},\n" for a in args) + "        )"


def _map_source(src: _Source, vmap: sc.ValueMap) -> list[str]:
    by = src.command.param(vmap.by)
    if vmap.name in src.plain:
        out = [f"{src.const_name(vmap.name)} = {_value({k: src._unescape(t) for k, t in vmap.values}, 0)}"
               if by.type == "enum" else
               f"{src.const_name(vmap.name)} = " + "{" + ", ".join(
                   f"{('True' if k else 'False') if by.type == 'bool' else k}: {_lit(src._unescape(t))}"
                   for k, t in vmap.values) + "}"]
        if vmap.default is not None:
            out.append(f"{src.const_name(vmap.name)}_DEFAULT = {_lit(src._unescape(vmap.default))}")
        return out
    n_arg = ", n: int = 0" if src.needs_n(vmap.name) else ""
    out = [f'def {src.func_name(vmap.name)}(input: "Signature.Input"{n_arg}) -> str:',
           f'    """The {vmap.name} text, by {vmap.by}."""']
    for key, text in vmap.values:
        out.append(f"    if {src.key_test(by, key)}:")
        out.append(f"        return {src.string(text, True)}")
    if vmap.default is not None:
        out.append(f"    return {src.string(vmap.default, True)}")
    else:
        out.append(f"    return {src.as_text(by)}")
    return out


def _step_lines(src: _Source, body, indent: int, in_loop: bool) -> list[str]:
    pad = " " * indent
    out: list[str] = []
    pending: list[str] = []   # plain appends waiting to be written together

    def flush():
        if len(pending) == 1:
            out.append(f"{pad}out.append({pending[0]})")
        elif pending:
            out.append(f"{pad}out += [")
            out.extend(f"{pad}    {s}," for s in pending)
            out.append(f"{pad}]")
        pending.clear()

    i = 0
    nodes = list(body)
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, sc.Repeat):
            flush()
            offset = f", {node.offset}" if node.offset else ""
            out.append(f"{pad}for n in repeat(input.{node.count}{offset}):")
            out.extend(_step_lines(src, node.steps, indent + 4, True))
            i += 1
            continue
        if node.literal:
            pending.append(_lit(node.text))
            i += 1
            continue
        if node.cond is None:
            pending.append(src.string(node.text, in_loop))
            i += 1
            continue
        flush()
        group = [node]
        while (i + len(group) < len(nodes) and isinstance(nodes[i + len(group)], sc.Step)
               and not nodes[i + len(group)].literal and nodes[i + len(group)].cond == node.cond):
            group.append(nodes[i + len(group)])
        out.append(f"{pad}if {src.cond(node.cond.tree)}:")
        if len(group) == 1:
            out.append(f"{pad}    out.append({src.string(node.text, in_loop)})")
        else:
            out.append(f"{pad}    out += [")
            out.extend(f"{pad}        {src.string(s.text, in_loop)}," for s in group)
            out.append(f"{pad}    ]")
        i += len(group)
    flush()
    return out


def convert(text: str, name: str, where: str = "SKILL.md") -> str:
    """The ``skill.py`` source of a SKILL.md's ``text`` (skill ``name``)."""
    description, steps, command = read_skill_md(text, name, where)
    header = f'"""The {name} skill (converted from SKILL.md)."""\n'
    if command is None:
        body = "".join(f"        {_lit(s)},\n" for s in steps)
        return (header + "\n\nclass Signature:\n" + _docstring(description, 4) + "\n\n\n"
                "def steps() -> list[str]:\n    return [\n" + body + "    ]\n")
    _check_names(command, where)
    src = _Source(command)
    has_repeat = any(isinstance(node, sc.Repeat) for node in command.body)
    imports = []
    if any(p.type == "enum" for p in command.params):
        imports.append("from typing import Literal\n\n")
    imports.append("from pydantic import BaseModel, Field\n")
    if has_repeat:
        imports.append("\nfrom fastworkflow.skill_commands import repeat\n")
    parts = [header, "\n", "".join(imports)]
    previous_function = True
    for vmap in command.maps:
        function = vmap.name not in src.plain
        parts.append("\n\n" if function or previous_function else "\n")
        parts.append("\n".join(_map_source(src, vmap)) + "\n")
        previous_function = function
    parts.append("\n\nclass Signature:\n" + _docstring(description, 4) + "\n\n    class Input(BaseModel):\n")
    parts.append("\n".join(_field(p) for p in command.params) + "\n")
    template = list(sc.template_steps(command))
    parts.append("\n    template = " + _value(template, 4) + "\n")
    parts.append('\n\ndef steps(input: Signature.Input) -> list[str]:\n    out: list[str] = []\n')
    parts.append("\n".join(_step_lines(src, command.body, 4, False)) + "\n    return out\n")
    return "".join(parts)


# ---------------------------------------------------------------- checking a conversion

def argument_grid(command: sc.SkillCommand, limit: int = GRID_LIMIT, seed: int = 0) -> list[dict]:
    """Argument sets covering each parameter's values: every enum value, both bools, ints from
    0 to 3 with the default and the map keys, and for a str its default and a sample holding
    quotes and braces. More than ``limit`` combinations are sampled (deterministically)."""
    choices = []
    for p in command.params:
        if p.type == "enum":
            values = list(p.values)
        elif p.type == "bool":
            values = [False, True]
        elif p.type == "int":
            keys = {k for m in command.maps if m.by == p.name for k, _ in m.values}
            values = sorted({0, 1, 2, 3, p.default or 0} | keys)
        else:
            sample = f'{p.name} "x" {{1}} 4421'
            values = [sample] if p.required else list(dict.fromkeys([p.default, "", sample]))
        choices.append([(p.name, v) for v in values])
    total = 1
    for c in choices:
        total *= len(c)
    if total <= limit:
        return [dict(combo) for combo in itertools.product(*choices)]
    rng = random.Random(seed)
    grid = [dict(rng.choice(c) for c in choices) for _ in range(limit)]
    for c in choices:            # every single value at least once
        for pair in c:
            grid.append({**dict(rng.choice(o) for o in choices), pair[0]: pair[1]})
    return grid


def _outcome(command: sc.SkillCommand, args: dict):
    try:
        return tuple(sc.expand(command, args))
    except sc.SkillCallError as exc:
        return ("error", type(exc).__name__)


def compare(old: sc.SkillCommand, new: sc.SkillCommand, grid: list[dict] | None = None,
            limit: int = 20) -> list[str]:
    """What differs between two parsed forms of one skill (empty when they agree)."""
    problems = []
    if old.description != new.description:
        problems.append(f"description: {old.description!r} != {new.description!r}")
    if old.params != new.params:
        for a, b in itertools.zip_longest(old.params, new.params):
            if a != b:
                problems.append(f"param: {a} != {b}")
    if sc.template_steps(old) != sc.template_steps(new):
        problems.append(f"template: {sc.template_steps(old)} != {sc.template_steps(new)}")
    if old.signature_text() != new.signature_text():
        problems.append("signature text differs")
    for args in grid if grid is not None else argument_grid(old):
        a, b = _outcome(old, args), _outcome(new, args)
        if a != b:
            problems.append(f"steps for {args}: {a} != {b}")
            if len(problems) >= limit:
                break
    return problems


def compare_folder(skills_dir: Path, workdir: str | None = None) -> dict[str, list[str]]:
    """``{skill: problems}`` for every skill of ``skills_dir`` (a workflow's ``_skills``),
    loaded as it is (SKILL.md) and with each SKILL.md converted (in temporary copies, under
    ``workdir`` when given)."""
    from fastworkflow import workflow_agent

    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        old_wf, new_wf = Path(tmp) / "skills_md_form", Path(tmp) / "skills_py_form"
        shutil.copytree(skills_dir, old_wf / "_skills")
        shutil.copytree(skills_dir, new_wf / "_skills")
        for md in sorted((new_wf / "_skills").glob(f"*/{sc.LEGACY_SKILL_FILE}")):
            (md.parent / sc.SKILL_FILE).write_text(convert(md.read_text(encoding="utf-8"), md.parent.name, str(md)),
                                                   encoding="utf-8")
            md.unlink()
        old = {s[0]: s for s in workflow_agent._parsed_skills(str(old_wf))}
        new = {s[0]: s for s in workflow_agent._parsed_skills(str(new_wf))}
    report: dict[str, list[str]] = {}
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            report[name] = [f"only in the {'SKILL.md' if name in old else 'skill.py'} form"]
            continue
        (_, d1, s1, c1), (_, d2, s2, c2) = old[name], new[name]
        if (c1 is None) != (c2 is None):
            report[name] = ["one form has params, the other not"]
        elif c1 is None:
            report[name] = ([] if (d1, s1) == (d2, s2) else [f"plain skill: {(d1, s1)} != {(d2, s2)}"])
        else:
            report[name] = compare(c1, c2)
    return report


# ---------------------------------------------------------------- command line

def _skill_files(paths: list[str]) -> list[Path]:
    found = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif (path / sc.LEGACY_SKILL_FILE).is_file():
            found.append(path / sc.LEGACY_SKILL_FILE)
        elif path.is_dir():
            found.extend(sorted(path.glob(f"*/{sc.LEGACY_SKILL_FILE}")))
        else:
            raise SystemExit(f"{raw}: no SKILL.md here")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="SKILL.md files, skill folders or _skills folders")
    parser.add_argument("--write", action="store_true", help="write skill.py next to each SKILL.md")
    parser.add_argument("--force", action="store_true", help="overwrite an existing skill.py")
    parser.add_argument("--check", action="store_true", help="compare each converted skill with its SKILL.md")
    parser.add_argument("--remove-md", action="store_true", help="delete each SKILL.md after --write and --check pass")
    args = parser.parse_args(argv)
    files = _skill_files(args.paths)
    status = 0
    if args.check:
        for skills_dir in sorted({f.parent.parent for f in files}):
            for name, problems in compare_folder(skills_dir).items():
                print(f"{skills_dir / name}: {'ok' if not problems else 'DIFFERS'}")
                for problem in problems:
                    print(f"    {problem}")
                status = status or (1 if problems else 0)
    for md in files:
        source = convert(md.read_text(encoding="utf-8"), md.parent.name, str(md))
        if not args.write:
            if not args.check:
                print(f"# ---- {md.parent / sc.SKILL_FILE}\n{source}")
            continue
        target = md.parent / sc.SKILL_FILE
        if target.exists() and not args.force:
            print(f"{target}: exists, kept (--force overwrites)", file=sys.stderr)
            continue
        target.write_text(source, encoding="utf-8")
        print(f"wrote {target}")
        if args.remove_md and args.check and status == 0:
            os.remove(md)
            print(f"removed {md}")
    return status


if __name__ == "__main__":
    sys.exit(main())
