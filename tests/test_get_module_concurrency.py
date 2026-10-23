"""Regression tests: command modules that import other command modules load safely.

``python_utils.get_module`` loads a command file by path under its package name.  A
composite command file can import another command file by that same package name, so
the same file may be loaded both by ``get_module`` and by Python's ordinary import,
from several threads at once.  These tests build a throwaway workflow whose command
modules import each other and check that:

* a module is executed once and shared, never replaced by a fresh copy mid-import;
* a thread importing a module never sees it half executed;
* a module whose load fails is not left behind in ``sys.modules`` to poison later imports.
"""

import sys
import textwrap
import threading
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from fastworkflow.utils import python_utils

GATE = "_fw_test_import_gate"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    """A workflow folder with a uniquely named package, cleaned out of sys.modules after."""
    name = f"wf_{uuid.uuid4().hex[:10]}"
    wf = tmp_path / name
    gate = types.ModuleType(GATE)
    gate.executions = {}
    gate.hooks = {}
    gate.lock = threading.Lock()

    def on_exec(module_name):
        with gate.lock:
            count = gate.executions.get(module_name, 0) + 1
            gate.executions[module_name] = count
        hook = gate.hooks.get(module_name.rsplit(".", 1)[-1])
        if hook:
            hook(count)

    gate.on_exec = on_exec
    monkeypatch.setitem(sys.modules, GATE, gate)
    monkeypatch.setattr(sys, "path", list(sys.path))
    python_utils.get_module.cache_clear()
    yield name, wf, gate
    python_utils.get_module.cache_clear()
    for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[mod]


def _base_and_composite(wf):
    _write(wf / "_commands" / "Alpha" / "base.py", f"""
        import {GATE} as gate
        gate.on_exec(__name__)

        def helper():
            return "helper"

        class ResponseGenerator:
            pass
    """)
    _write(wf / "_commands" / "Beta" / "composite.py", f"""
        import {GATE} as gate
        from ..Alpha.base import helper, ResponseGenerator as BaseGenerator
        gate.on_exec(__name__)

        class ResponseGenerator:
            pass
    """)
    return wf / "_commands" / "Alpha" / "base.py", wf / "_commands" / "Beta" / "composite.py"


def test_concurrent_load_never_exposes_a_half_executed_module(workflow):
    """Forces the interleaving from the field.

    Thread 1 loads the composite, whose ``from ..Alpha.base import helper`` starts an
    ordinary import of ``base`` and pauses inside it.  Thread 2 then loads ``base`` by
    path.  Thread 2 must wait for and reuse thread 1's module; it must not swap a fresh,
    still-executing copy into ``sys.modules`` that thread 1's import then returns.
    """
    name, wf, gate = workflow
    base_path, composite_path = _base_and_composite(wf)
    first_exec_paused = threading.Event()
    second_exec_started = threading.Event()
    release_second = threading.Event()

    def base_hook(count):
        if count == 1:
            first_exec_paused.set()
            # Before the fix a second execution starts here; after it, none does.
            second_exec_started.wait(timeout=1.0)
        else:
            second_exec_started.set()
            release_second.wait(timeout=10.0)

    gate.hooks["base"] = base_hook
    results, errors = {}, {}

    def load(key, path):
        try:
            results[key] = python_utils.get_module(str(path), str(wf))
        except Exception as e:  # noqa: BLE001 - recorded for the assertion
            errors[key] = e

    t1 = threading.Thread(target=load, args=("composite", composite_path))
    t1.start()
    assert first_exec_paused.wait(timeout=10.0)
    t2 = threading.Thread(target=load, args=("base", base_path))
    t2.start()
    t1.join(timeout=15.0)
    release_second.set()
    t2.join(timeout=15.0)
    assert not t1.is_alive() and not t2.is_alive()

    assert errors == {}
    base_name = f"{name}._commands.Alpha.base"
    assert gate.executions[base_name] == 1
    assert results["base"] is sys.modules[base_name]
    assert results["composite"].helper is results["base"].helper
    assert results["composite"].BaseGenerator is results["base"].ResponseGenerator


def test_failed_load_does_not_poison_later_imports(workflow):
    """A module whose first execution fails is removed from sys.modules, so the next
    import of it, ordinary or by path, executes it again instead of finding the broken
    half-module and failing with ``cannot import name``."""
    name, wf, gate = workflow
    base_path, composite_path = _base_and_composite(wf)

    def base_hook(count):
        if count == 1:
            raise RuntimeError("transient failure while loading")

    gate.hooks["base"] = base_hook
    with pytest.raises(ImportError):
        python_utils.get_module(str(base_path), str(wf))
    assert f"{name}._commands.Alpha.base" not in sys.modules

    composite = python_utils.get_module(str(composite_path), str(wf))
    assert composite.helper() == "helper"
    assert python_utils.get_module(str(base_path), str(wf)).helper is composite.helper


def test_cross_importing_modules_load_from_many_threads(workflow):
    """Many threads load a web of cross-importing command modules, repeatedly, with the
    loader cache cleared between rounds (as a routing rebuild does)."""
    name, wf, gate = workflow
    leaves = []
    for i in range(6):
        path = wf / "_commands" / f"Ctx{i}" / f"leaf{i}.py"
        _write(path, f"""
            import time
            import {GATE} as gate
            gate.on_exec(__name__)
            time.sleep(0.002)

            def section():
                return {i}

            class ResponseGenerator:
                pass
        """)
        leaves.append(path)
    composites = []
    for j in range(6):
        imports = "\n".join(
            f"from ..Ctx{i}.leaf{i} import section as s{i}, ResponseGenerator as G{i}"
            for i in range(6) if (i + j) % 2 == 0 or i == j)
        path = wf / "_commands" / f"Mix{j}" / f"composite{j}.py"
        _write(path, (
            f"import time\nimport {GATE} as gate\n{imports}\n"
            "gate.on_exec(__name__)\ntime.sleep(0.002)\n\n"
            "class ResponseGenerator:\n    pass\n"))
        composites.append(path)
    _write(wf / "_commands" / "Top" / "top.py", f"""
        import {GATE} as gate
        from ..Mix0.composite0 import ResponseGenerator as C0
        from ..Mix3.composite3 import ResponseGenerator as C3
        from ..Ctx5.leaf5 import section
        gate.on_exec(__name__)

        class ResponseGenerator:
            pass
    """)
    paths = [str(p) for p in leaves + composites] + [str(wf / "_commands" / "Top" / "top.py")]

    for round_no in range(8):
        python_utils.get_module.cache_clear()
        order = paths[round_no % len(paths):] + paths[:round_no % len(paths)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(python_utils.get_module, p, str(wf))
                       for p in order * 3 + list(reversed(order)) * 3]
            modules = [f.result() for f in futures]
        assert all(hasattr(m, "ResponseGenerator") for m in modules)

    # Every module ran exactly once and every importer shares the same objects.
    assert all(count == 1 for count in gate.executions.values()), gate.executions
    for i, leaf in enumerate(leaves):
        mod = python_utils.get_module(str(leaf), str(wf))
        assert mod is sys.modules[f"{name}._commands.Ctx{i}.leaf{i}"]
    top = python_utils.get_module(paths[-1], str(wf))
    assert top.section is python_utils.get_module(str(leaves[5]), str(wf)).section


def test_edited_module_is_reexecuted_after_cache_clear(workflow):
    """A command file edited on disk is picked up again once the loader cache is
    cleared, as before; an unchanged one is reused."""
    name, wf, gate = workflow
    path = wf / "_commands" / "Alpha" / "edited.py"
    _write(path, f"""
        import {GATE} as gate
        gate.on_exec(__name__)
        VALUE = 1
    """)
    mod = python_utils.get_module(str(path), str(wf))
    assert mod.VALUE == 1
    python_utils.get_module.cache_clear()
    assert python_utils.get_module(str(path), str(wf)) is mod
    assert gate.executions[f"{name}._commands.Alpha.edited"] == 1

    import os
    _write(path, f"""
        import {GATE} as gate
        gate.on_exec(__name__)
        VALUE = 2
    """)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    python_utils.get_module.cache_clear()
    again = python_utils.get_module(str(path), str(wf))
    assert again.VALUE == 2
    assert gate.executions[f"{name}._commands.Alpha.edited"] == 2
