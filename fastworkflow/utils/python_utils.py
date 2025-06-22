import os
import importlib
import re
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# Add lru_cache to avoid repeated imports of the same module
@lru_cache(maxsize=128)
def get_module(module_path: str, search_root: Optional[str] = None) -> Any:
    """
    Dynamically import a module from a file path.

    Args:
        module_path: The path to the module file.
        search_root: Optional root directory to search for the module.

    Returns:
        The imported module or None if import fails.
    """
    try:
        if search_root:
            # If search_root is provided, make module_path relative to it
            root_path = Path(search_root)
            module_path_obj = Path(module_path)
            
            # If module_path is already relative to search_root, use it as is
            if module_path_obj.is_relative_to(root_path):
                module_path = str(module_path_obj)
            else:
                # Try to make module_path relative to search_root
                try:
                    rel_path = module_path_obj.relative_to(root_path)
                    module_path = str(rel_path)
                except ValueError:
                    # If module_path is not relative to search_root, use it as is
                    pass

        # ------------------------------------------------------------------
        # Determine an importable module name so that relative imports inside
        # the dynamically-loaded file work.  When ``search_root`` is supplied
        # we interpret it as the root of the workflow being loaded (e.g.
        # ``examples/retail_workflow``).  Any Python files beneath that root
        # should behave as if they live in a real package structure rooted at
        # the directory *above* ``search_root``.  That gives command modules a
        # ``__package__`` like ``examples.retail_workflow._commands`` so that
        # statements such as ``from ..retail_data import foo`` resolve.
        #
        # If anything goes wrong we fall back to the original anonymous name
        # to avoid breaking unrelated callers.
        # ------------------------------------------------------------------
        module_name = None
        if search_root:
            try:
                root_path = Path(search_root).resolve()
                file_path = Path(module_path).resolve()

                # Special handling for internal workflows in _workflows directory
                if '_workflows' in root_path.parts:
                    # For internal workflows, use fastworkflow._workflows.workflow_name as the package prefix
                    # Find the position of _workflows in the path
                    workflows_idx = root_path.parts.index('_workflows')
                    if workflows_idx >= 0:
                        # Get the workflow name (directory after _workflows)
                        if workflows_idx + 1 < len(root_path.parts):
                            workflow_name = root_path.parts[workflows_idx + 1]
                            # Build a package name like 'fastworkflow._workflows.command_metadata_extraction'
                            package_prefix = f"fastworkflow._workflows.{workflow_name}"
                            
                            # Get the relative path from the workflow root
                            rel_path = file_path.relative_to(root_path)
                            # Convert to module path
                            rel_module = ".".join(rel_path.with_suffix("").parts)
                            
                            # Combine to get the full module name
                            module_name = f"{package_prefix}.{rel_module}" if rel_module else package_prefix
                            
                            # Ensure fastworkflow is in sys.path
                            fw_path = str(Path(root_path).parents[workflows_idx])
                            if fw_path not in sys.path:
                                sys.path.insert(0, fw_path)
                else:
                    # Standard case for regular workflows (unchanged)
                    # The importable package should start one directory above the
                    # workflow root so that the workflow directory itself becomes
                    # a package component (e.g. ``examples.retail_workflow``).
                    anchor_for_package = root_path.parent

                    rel_path_from_anchor = file_path.relative_to(anchor_for_package)
                    # Strip the .py suffix and convert path separators to dots.
                    module_name = ".".join(rel_path_from_anchor.with_suffix("").parts)

                    # Ensure the anchor directory is on sys.path so the package
                    # hierarchy can be resolved by Python's import machinery. We
                    # prepend to honour relative-import expectations.
                    anchor_str = str(anchor_for_package)
                    if anchor_str not in sys.path:
                        sys.path.insert(0, anchor_str)
            except Exception as e:
                # Fall back to anonymous hashed name below.
                module_name = None

        # Fall back to a unique anonymous name if we could not build a package name
        if not module_name:
            module_name = f"dynamic_module_{hash(module_path)}"
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        # print(f"Error importing module {module_path}: {e}")
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
    module_path = rel_path.replace(os.sep, '.')
    # Remove leading/trailing dots
    module_path = module_path.strip('.')
    # Validate segments
    for segment in module_path.split('.'):
        if segment and not segment.isidentifier():
            raise ValueError(f"Invalid Python identifier in import path: {segment}")
    return module_path

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
    dependencies = set()
    # 1. Inheritance (base classes)
    for base in getattr(class_info, 'bases', []):
        if base not in {'object'}:
            dependencies.add(base)
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
    dependencies = {d for d in dependencies if d not in builtins}
    return dependencies