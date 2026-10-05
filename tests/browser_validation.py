"""Run the browser checks instead of letting them skip themselves.

Every DOM test in this suite starts with the same three lines: read
``TEST_JSDOM_ROOT``, and skip when it is unset. That is the right default for
``pytest`` on a machine with no jsdom — but it also means a run that was
supposed to prove the UI works can report success having executed no browser
code at all, which is how "relevant browser interaction checks passed" ends up
meaning nothing.

This module is the other mode: prerequisites are checked up front and reported
as failures rather than skips, the DOM tests are discovered rather than listed
(so a new one is covered the day it is written), and a run in which a required
check skipped is a failed run. It adds no framework — it selects and reports on
the existing jsdom tests, which ``pytest`` still runs.

Usage, from the repository root with the venv active::

    TEST_JSDOM_ROOT=/path/to/jsdom-install python -m tests.browser_validation

``TEST_JSDOM_ROOT`` is a directory containing ``node_modules/jsdom`` — the
harness scripts require it from there by path, so the location is the caller's
to configure and is never assumed here. Narrow the run by passing test files:

    TEST_JSDOM_ROOT=... python -m tests.browser_validation tests/test_x.py
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

JSDOM_ROOT_VAR = "TEST_JSDOM_ROOT"
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


class BrowserPrerequisiteMissing(RuntimeError):
    """A browser check cannot run, and saying so is the point."""


@dataclass
class ValidationReport:
    required: list[str]
    passed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    pytest_exit_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        """Success needs every required check to have actually executed."""
        return bool(
            self.required
            and self.pytest_exit_code == 0
            and not self.skipped
            and not self.failed
            and not self.missing
        )

    def summary(self) -> str:
        lines = [
            f"required browser checks: {len(self.required)}",
            f"executed and passed:     {len(self.passed)}",
        ]
        for label, nodes in (
            ("skipped", self.skipped),
            ("failed", self.failed),
            ("never reported", self.missing),
        ):
            if nodes:
                lines.append(f"{label}: {len(nodes)}")
                lines.extend(f"  - {node}" for node in nodes)
        if not self.required:
            lines.append(
                "no browser checks were discovered at all, which is itself a "
                "failure: this gate exists to run them"
            )
        return "\n".join(lines)


def _function_defs(body: Iterable[ast.stmt]) -> dict[str, ast.stmt]:
    return {
        node.name: node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node: ast.stmt, receivers: frozenset[str]) -> set[str]:
    """Names this function calls, within the scopes discovery can resolve.

    Plain calls are the enclosing module's or the function's own scope.
    Attribute calls count only when the receiver is one this scope can be sure
    about -- ``self``, ``cls``, or the class itself -- so that a method calling
    a sibling helper is followed while ``something_else.run()`` is not mistaken
    for a local one.
    """
    names: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        function = call.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in receivers
        ):
            names.add(function.attr)
    return names


def _reaching(
    defs: dict[str, ast.stmt],
    source: str,
    inherited: frozenset[str],
    receivers: frozenset[str],
) -> set[str]:
    """Which functions in one scope reach jsdom, directly or via a helper.

    Several DOM tests call the harness through a helper rather than naming the
    variable themselves, so direct matching would report coverage of a smaller
    suite than exists -- which is the failure this whole module is about. The
    fixpoint keeps discovery honest as more helpers appear.
    """
    reaching = {
        name
        for name, node in defs.items()
        if JSDOM_ROOT_VAR in (ast.get_source_segment(source, node) or "")
    }
    growing = True
    while growing:
        growing = False
        for name, node in defs.items():
            if name in reaching:
                continue
            if _called_names(node, receivers) & (reaching | inherited):
                reaching.add(name)
                growing = True
    return reaching


def _collected_class(node: ast.stmt) -> bool:
    """Whether pytest would collect this class at all.

    Discovery must not require a node id pytest will never run: that would
    read as "never reported" and fail the gate for a check that does not
    exist. The two rules that matter here are pytest's default
    ``python_classes = Test*`` and its refusal to collect a class with an
    ``__init__``.
    """
    return (
        isinstance(node, ast.ClassDef)
        and node.name.startswith("Test")
        and "__init__" not in _function_defs(node.body)
    )


def _class_tails(
    body: Iterable[ast.stmt],
    source: str,
    module_reaching: frozenset[str],
    prefix: str,
) -> list[str]:
    """Node-id tails for browser checks written as methods of a test class.

    A DOM test nested in a class used to be invisible here, and invisible is
    the one thing this gate cannot afford: the test would skip itself on a
    machine without jsdom and the run would report nothing missing (fix-17mu).
    Classes nest, so this recurses, and each level addresses its own methods.
    """
    tails: list[str] = []
    for node in body:
        if not _collected_class(node):
            continue
        scope = f"{prefix}{node.name}::"
        defs = _function_defs(node.body)
        reaching = _reaching(
            defs,
            source,
            module_reaching,
            frozenset({"self", "cls", node.name}),
        )
        tails.extend(
            f"{scope}{name}" for name in sorted(reaching) if name.startswith("test_")
        )
        # A nested class's methods see the module's helpers, not the outer
        # class's: an inner class is not an instance of the outer one.
        tails.extend(_class_tails(node.body, source, module_reaching, scope))
    return tails


def _browser_check_tails(tree: ast.Module, source: str) -> list[str]:
    """Every node-id tail in one module that drives a real DOM."""
    module_reaching = _reaching(
        _function_defs(tree.body), source, frozenset(), frozenset()
    )
    tails = [name for name in module_reaching if name.startswith("test_")]
    tails.extend(_class_tails(tree.body, source, frozenset(module_reaching), ""))
    return sorted(tails)


def discover_required_dom_tests(
    paths: Optional[Iterable[Path]] = None,
) -> list[str]:
    """Node ids of every test that drives a real DOM, newest ones included."""
    candidates: list[Path] = []
    for path in paths or [TESTS_DIR]:
        path = Path(path)
        if path.is_dir():
            candidates.extend(sorted(path.glob("test_*.py")))
        elif path.is_file():
            candidates.append(path)

    node_ids: list[str] = []
    for candidate in candidates:
        source = candidate.read_text(encoding="utf-8", errors="ignore")
        if JSDOM_ROOT_VAR not in source:
            continue
        tree = ast.parse(source)
        try:
            relative = candidate.resolve().relative_to(REPO_ROOT)
        except ValueError:
            relative = candidate
        node_ids.extend(
            f"{relative}::{tail}" for tail in _browser_check_tails(tree, source)
        )
    return node_ids


def check_prerequisites(jsdom_root: Optional[str] = None) -> str:
    """Refuse to pretend. Returns the configured jsdom root."""
    root = jsdom_root if jsdom_root is not None else os.environ.get(JSDOM_ROOT_VAR)
    if not root:
        raise BrowserPrerequisiteMissing(
            f"{JSDOM_ROOT_VAR} is not set. The DOM tests skip themselves "
            "without it, so this validation has nothing to run. Point it at a "
            "directory containing node_modules/jsdom."
        )
    if shutil.which("node") is None:
        raise BrowserPrerequisiteMissing(
            "node is not on PATH; the jsdom harness scripts cannot run."
        )
    module = Path(root).expanduser() / "node_modules" / "jsdom"
    if not module.is_dir():
        raise BrowserPrerequisiteMissing(
            f"{JSDOM_ROOT_VAR}={root!r} has no node_modules/jsdom. The harness "
            "requires jsdom from that path; install it there (offline: copy an "
            "existing install) rather than pointing at a different directory."
        )
    probe = subprocess.run(
        ["node", "-e", f"require({str(module)!r})"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise BrowserPrerequisiteMissing(
            f"jsdom under {JSDOM_ROOT_VAR}={root!r} is not loadable by node:\n"
            f"{probe.stderr.strip()}"
        )
    return str(Path(root).expanduser())


def _split_junit_classname(dotted: str) -> tuple[Optional[Path], tuple[str, ...]]:
    """A JUnit ``classname`` back into its module path and enclosing classes.

    JUnit has one dotted field for both, so ``tests.test_x.TestThing`` has to
    be cut in the right place; cutting it wrong turns a class-nested check into
    a node id naming a file that does not exist, which reads as "never
    reported". The cut is made by finding the module on disk, and when the
    report came from a run rooted elsewhere, by falling back on the naming
    pytest itself relies on: modules are ``test_*.py``, collected classes are
    ``Test*``.
    """
    parts = [part for part in dotted.split(".") if part]
    for cut in range(len(parts), 0, -1):
        candidate = Path(*parts[:cut]).with_suffix(".py")
        if (REPO_ROOT / candidate).is_file():
            return candidate, tuple(parts[cut:])
    cut = next(
        (index for index, part in enumerate(parts) if part[:1].isupper()),
        len(parts),
    )
    if cut == 0:
        return None, ()
    return Path(*parts[:cut]).with_suffix(".py"), tuple(parts[cut:])


def read_junit_outcomes(report_path: Path) -> dict[str, str]:
    """Outcome per node id from a pytest JUnit report.

    Parametrised cases report one entry each; they are folded back onto the
    function that generated them, and a function counts as executed only when
    every one of its cases did.
    """
    outcomes: dict[str, str] = {}
    for case in ElementTree.parse(report_path).getroot().iter("testcase"):
        path, classes = _split_junit_classname(case.get("classname") or "")
        name = (case.get("name") or "").split("[", 1)[0]
        if path is None or not name:
            continue
        node = f"{path}::{'::'.join((*classes, name))}"
        if any(child.tag == "skipped" for child in case):
            outcome = "skipped"
        elif any(child.tag in {"failure", "error"} for child in case):
            outcome = "failed"
        else:
            outcome = "passed"
        # Worst outcome wins across a function's parametrisations.
        ranking = {"passed": 0, "skipped": 1, "failed": 2}
        if ranking[outcome] >= ranking.get(outcomes.get(node, "passed"), 0):
            outcomes[node] = outcome
    return outcomes


def outcome_for(node_id: str, outcomes: dict[str, str]) -> Optional[str]:
    """Find a node's outcome despite pytest's rootdir-relative reporting.

    A JUnit `classname` is relative to whatever pytest chose as its rootdir,
    which is not always this repository, so the recorded key can be a shorter
    path than the node id that asked for it. Test file names are unique within
    `tests/`, so matching on the file name and the rest of the node id is exact
    here -- the rest being the function, or the class path and the method.
    """
    path, _, name = node_id.partition("::")
    for key, outcome in outcomes.items():
        key_path, _, key_name = key.partition("::")
        if key_name == name and (
            key_path == path or Path(key_path).name == Path(path).name
        ):
            return outcome
    return None


def validate(
    paths: Optional[Sequence[Path]] = None,
    *,
    jsdom_root: Optional[str] = None,
    extra_pytest_args: Sequence[str] = (),
) -> ValidationReport:
    """Run the discovered browser checks and report what actually executed."""
    root = check_prerequisites(jsdom_root)
    required = discover_required_dom_tests(paths)
    report = ValidationReport(required=required)
    if not required:
        return report

    environment = dict(os.environ, **{JSDOM_ROOT_VAR: root})
    with tempfile.TemporaryDirectory(prefix="fw-browser-validation-") as scratch:
        junit = Path(scratch) / "browser-validation.xml"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *required,
                "-p",
                "no:randomly",
                "-q",
                f"--junitxml={junit}",
                *extra_pytest_args,
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        report.pytest_exit_code = completed.returncode
        outcomes = read_junit_outcomes(junit) if junit.exists() else {}

    for node in required:
        outcome = outcome_for(node, outcomes)
        if outcome == "passed":
            report.passed.append(node)
        elif outcome == "skipped":
            report.skipped.append(node)
        elif outcome == "failed":
            report.failed.append(node)
        else:
            report.missing.append(node)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the jsdom browser checks and fail when any required check "
            "did not execute."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="test files or directories to validate (default: tests/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the required browser checks without running them",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="pytest_args",
        help="extra argument forwarded to pytest (repeatable)",
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        for node in discover_required_dom_tests(arguments.paths or None):
            print(node)
        return 0

    try:
        report = validate(
            arguments.paths or None, extra_pytest_args=arguments.pytest_args
        )
    except BrowserPrerequisiteMissing as missing:
        print(f"browser validation cannot run: {missing}", file=sys.stderr)
        return 2

    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
