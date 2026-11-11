import hashlib
import inspect
import os
import importlib
import importlib.machinery
import re
import importlib.util
import sys
import threading
import types
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import typing
from typing import Any, Optional, Union, get_args, get_origin

from fastworkflow.utils.logging import logger

# Serialises loads where Python's per-module import lock is unavailable.
_FALLBACK_LOAD_LOCK = threading.RLock()
# Dotted name -> mtime_ns of the file when it was last executed or first reused.
_LOADED_MTIMES: dict[str, int] = {}
# Separates a workflow folder's basename from the digest of its resolved path in a module namespace.
_NAMESPACE_SEP = "__"
_NAMESPACE_LOCK = threading.RLock()
_NAMESPACE_DIGEST = re.compile(r"\b(\w+?)__[0-9a-f]{8}(?=\.)")


def workflow_namespace(folder: str) -> str:
    """The top-level package name under which a workflow folder's command modules are loaded.

    It is the folder's basename plus a short digest of its resolved path, so two folders with the
    same basename (two copies of ``library``) get different module names and never share modules or
    relative imports. The namespace package is registered in ``sys.modules`` with the folder as its
    ``__path__``, so ``from ..application import x`` inside a command file resolves inside that folder.
    """
    real = os.path.realpath(folder)
    digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:8]
    basename = os.path.basename(real).replace(".", "_") or "workflow"
    namespace = f"{basename}{_NAMESPACE_SEP}{digest}"
    with _NAMESPACE_LOCK:
        package = sys.modules.get(namespace)
        if package is None or list(getattr(package, "__path__", ())) != [real]:
            package = types.ModuleType(namespace)
            package.__path__ = [real]
            package.__package__ = namespace
            spec = importlib.machinery.ModuleSpec(namespace, None, is_package=True)
            spec.submodule_search_locations = [real]
            package.__spec__ = spec
            sys.modules[namespace] = package
    return namespace


def legacy_module_names(text: str) -> str:
    """*text* with each workflow namespace written back to the bare basename it had before the
    namespace existed (``library__3fa2b1c4.x`` -> ``library.x``).

    Applied where a module name is digested into a cache key, so a key written before the namespace
    still matches and a trained or cached artifact is not regenerated for that reason alone.
    """
    return _NAMESPACE_DIGEST.sub(lambda match: match.group(1), text)


def type_text(annotation: Any) -> str:
    """An annotation as user-facing text: a class by its plain name (its module is left out, so a
    workflow folder's namespace never reaches a preview or an error), a generic by its arguments
    (``list[Book]``, ``Book | None``)."""
    if annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    if isinstance(annotation, str):
        return annotation
    if inspect.isclass(annotation) and not get_args(annotation):
        return annotation.__name__
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is typing.Annotated:
        return type_text(args[0])
    if origin in (Union, types.UnionType):
        return " | ".join(type_text(arg) for arg in args)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(arg) for arg in args)}]"
    if origin is None:
        return re.sub(r"\btyping\.", "", repr(annotation))
    name = getattr(origin, "__name__", None) or str(origin)
    if not args:
        return name
    return f"{name}[{', '.join(type_text(arg) for arg in args)}]"


@contextmanager
def _module_import_lock(name: str):
    """Hold the same per-module lock Python's own ``import`` statement takes for
    ``name``, so a load by path and an ordinary import of the same module (for
    example a command module importing another command module) never execute it
    at the same time, in any thread.  The lock is reentrant per thread and takes
    part in importlib's deadlock detection."""
    manager = getattr(getattr(importlib, "_bootstrap", None), "_ModuleLockManager", None)
    if manager is None:  # pragma: no cover - CPython has always provided it
        with _FALLBACK_LOAD_LOCK:
            yield
        return
    with manager(name):
        yield


def _same_file(module: Any, abs_module_path: str) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return os.path.samefile(module_file, abs_module_path)
    except OSError:
        return os.path.normcase(os.path.abspath(module_file)) == os.path.normcase(abs_module_path)


def _mtime_ns(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _load_module_once(name: str, abs_module_path: str) -> Any:
    """Load ``abs_module_path`` as module ``name``, thread-safely and idempotently.

    * A module already in ``sys.modules`` under ``name`` from the same file (loaded
      earlier by path or by an ordinary import) is reused, not executed again;
      it is re-executed in place only when the file changed on disk since then.
    * While a fresh module executes it is marked initialising, as importlib does,
      so a concurrent ordinary import waits for it instead of using it half done.
    * A module whose execution fails is removed from ``sys.modules`` (the earlier
      entry, if any, is restored), so later imports of it are not poisoned.
    """
    with _module_import_lock(name):
        existing = sys.modules.get(name)
        if existing is not None and _same_file(existing, abs_module_path):
            current = _mtime_ns(abs_module_path)
            recorded = _LOADED_MTIMES.setdefault(name, current)
            if current is None or recorded is None or current == recorded:
                return existing
            # Edited on disk: re-execute in the same module object so code that
            # already holds it keeps a fully populated module throughout.
            spec = importlib.util.spec_from_file_location(name, abs_module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for {abs_module_path}")
            spec.loader.exec_module(existing)
            _LOADED_MTIMES[name] = current
            return existing

        spec = importlib.util.spec_from_file_location(name, abs_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {abs_module_path}")
        module = importlib.util.module_from_spec(spec)
        mtime = _mtime_ns(abs_module_path)
        # Add to sys.modules before executing to support relative and circular
        # imports; mark it initialising so other threads wait on the lock.
        sys.modules[name] = module
        spec._initializing = True
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(name) is module:
                if existing is not None:
                    sys.modules[name] = existing
                else:
                    del sys.modules[name]
            raise
        finally:
            spec._initializing = False
        _LOADED_MTIMES[name] = mtime
        # Like importlib, return what the module left in sys.modules.
        return sys.modules.get(name, module)


# Normalize arguments so logically identical calls share the same cache key
@lru_cache(maxsize=128)
def get_module(module_path: str, search_root: Optional[str] = None) -> Any:
    """
    Dynamically import a module from a file path. Ensures that a module
    is imported under a consistent, canonical name to avoid pickling issues
    and to allow relative imports within the loaded module.

    Args:
        module_path: The absolute or relative path to the module file.
        search_root: Optional. The root directory of the project/workflow
                     that contains the module. This helps in constructing
                     a proper package name. If None, the module's parent
                     directory is used as a simple anchor.

    Returns:
        The imported module or None if import fails.
    """
    if not module_path:
        return None

    try:
        # Normalise to absolute paths
        abs_module_path = os.path.abspath(module_path)

        # Determine project root used for building the importable dotted path.
        #
        # * When `search_root` is provided it usually points to **the workflow
        #   folder itself** (e.g. ``.../examples/hello_world``).  The modules we
        #   load live *inside* that folder (e.g. ``_commands/startup.py``).
        #   We **include** the workflow folder name as the first package segment
        #   so that intra-workflow relative imports (like
        #   ``from ..application.todo_manager import ...``) resolve correctly.
        #   Therefore we treat the *parent* of ``search_root`` as the project
        #   root when computing the dotted path – this yields an import path
        #   such as ``hello_world._commands.startup`` instead of
        #   ``_commands.startup``.
        #
        # * If `search_root` is not provided we fall back to the repository
        #   root (two levels up from this util file) which preserves existing
        #   behaviour for library code.
        if search_root:
            project_root = os.path.abspath(os.path.join(search_root, os.pardir))
            # Ensure parent dir is on sys.path so that the workflow folder is
            # importable as a top-level package (e.g. ``import hello_world``).
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            # Also ensure the workflow folder itself is on sys.path to match
            # fastworkflow.Workflow behavior and enable absolute imports
            search_root_abs = os.path.abspath(search_root)
            if search_root_abs not in sys.path:
                sys.path.insert(0, search_root_abs)
        else:
            # Fallback: repository root (two levels up from this util file)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Accept module paths that are either within the workflow-specific
        # `project_root` (parent of ``search_root``) *or* within the main
        # FastWorkflow package itself.  The latter is required for loading
        # internal helper workflows that live under
        # ``fastworkflow/_workflows``.

        fw_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        if abs_module_path.startswith(fw_pkg_root):
            relative_base = fw_pkg_root
        elif abs_module_path.startswith(project_root):
            relative_base = project_root
        else:
            raise ImportError(
                f"Module {abs_module_path} is outside of permitted roots: {project_root} or {fw_pkg_root}")

        # Build import path relative to project root
        if relative_base == project_root and search_root:
            folder = os.path.abspath(search_root)
            inside = os.path.relpath(abs_module_path, folder)
            if not inside.startswith(os.pardir) and not os.path.isabs(inside):
                # A workflow file: named under the folder's own namespace (see workflow_namespace).
                namespace = workflow_namespace(folder)
                inside = inside.rsplit(".py", 1)[0]
                dotted = inside.replace(os.sep, ".")
                return _load_module_once(f"{namespace}.{dotted}", abs_module_path)

        relative_path = os.path.relpath(abs_module_path, relative_base)
        module_pythonic_path = relative_path.replace(os.sep, ".").rsplit(".py", 1)[0]
        if relative_base == fw_pkg_root:
            pkg_name = Path(fw_pkg_root).name
            module_pythonic_path = f"{pkg_name}.{module_pythonic_path}"

        return _load_module_once(module_pythonic_path, abs_module_path)

    except Exception as e:
        logger.critical(f"Could not import module from path: {module_path}. Error: {e}")
        raise ImportError(
            f"Could not import module from path: {module_path}. Error: {e}") from e

def get_module_import_path(file_path: str, source_dir: str) -> str:
    """
    Given a Python file path and the source directory, return the correct Python import path.
    Handles nested packages, __init__.py, platform differences, and validates identifiers.
    Raises ValueError if the file is not within the source directory or if the import path is invalid.
    """
    file_path = os.path.abspath(file_path)
    source_dir = os.path.abspath(source_dir)

    if not file_path.startswith(source_dir):
        raise ValueError(f"{file_path} is not inside {source_dir}")

    rel_path = os.path.relpath(file_path, source_dir)
    # Remove .py extension
    if rel_path.endswith('.py'):
        rel_path = rel_path[:-3]
    # Remove __init__ if present
    if rel_path.endswith('__init__'):
        rel_path = rel_path[:-9]
        if rel_path.endswith(os.sep):
            rel_path = rel_path[:-1]
    # Convert path separators to dots
    module_path_str = rel_path.replace(os.sep, '.')
    # Remove leading/trailing dots
    module_path_str = module_path_str.strip('.')
    # Validate segments
    if module_path_str: # Ensure not empty before splitting
        for segment in module_path_str.split('.'):
            if segment and not segment.isidentifier(): # allow empty segment if rel_path was just '__init__' -> ''
                raise ValueError(f"Invalid Python identifier in import path: {segment}")
    return module_path_str

def extract_custom_types_from_annotation(annotation: str) -> set:
    """
    Recursively extract custom type names from a type annotation string.
    Handles cases like 'List[Foo]', 'Optional[Bar]', 'Dict[str, Baz]', etc.
    Returns a set of type names that are not built-in/standard types.
    """
    if not annotation or not isinstance(annotation, str):
        return set()
    # Remove common wrappers
    wrappers = [
        'List', 'Dict', 'Optional', 'Union', 'Set', 'Tuple', 'Sequence', 'Mapping', 'Iterable', 'Literal', 'Any', 'str', 'int', 'float', 'bool', 'None', 'bytes', 'object'
    ]
    # Find all identifiers (words starting with a letter or underscore)
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", annotation))
    # Remove wrappers and built-ins
    return {t for t in tokens if t not in wrappers}

def find_module_dependencies(class_info) -> set:
    """
    Analyze a ClassInfo object for dependencies on other custom types/classes.
    Considers base classes, method parameter/return types, and property types.
    Returns a set of custom type names (as strings).
    """
    dependencies = {
        base
        for base in getattr(class_info, 'bases', [])
        if base not in {'object'}
    }
    # 2. Methods: parameter and return type annotations
    for method in getattr(class_info, 'methods', []):
        for param in getattr(method, 'parameters', []):
            annotation = param.get('annotation')
            dependencies.update(extract_custom_types_from_annotation(annotation))
        if getattr(method, 'return_annotation', None):
            dependencies.update(extract_custom_types_from_annotation(method.return_annotation))
    # 3. Properties: type annotations
    for prop in getattr(class_info, 'properties', []):
        if getattr(prop, 'type_annotation', None):
            dependencies.update(extract_custom_types_from_annotation(prop.type_annotation))
    # Remove built-in/standard types (redundant, but safe)
    builtins = {'str', 'int', 'float', 'bool', 'list', 'dict', 'Any', 'None', 'bytes', 'object'}
    return {d for d in dependencies if d not in builtins}