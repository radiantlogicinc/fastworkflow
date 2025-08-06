import os
import importlib
import re
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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

        if not (abs_module_path.startswith(project_root) or abs_module_path.startswith(fw_pkg_root)):
            raise ImportError(
                f"Module {abs_module_path} is outside of permitted roots: {project_root} or {fw_pkg_root}")

        # Build import path relative to project root
        relative_path = os.path.relpath(abs_module_path, project_root)
        module_pythonic_path = relative_path.replace(os.sep, ".").rsplit(".py", 1)[0]

        # Use spec_from_file_location for dynamic loading from file paths
        spec = importlib.util.spec_from_file_location(module_pythonic_path, abs_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {abs_module_path}")
        
        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules before executing to support relative imports
        sys.modules[module_pythonic_path] = module
        spec.loader.exec_module(module)
        return module

    except ImportError as e:
        # re-raise with clearer context
        raise ImportError(
            f"Could not import module from path: {module_path}. Error: {e}") from e
    except Exception:
        return None

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