import os
import importlib
import re

def get_module(module_file_path: str, workflow_folderpath: str):
    if not module_file_path:
        return None

    # Get absolute paths to ensure consistency
    abs_module_path = os.path.abspath(module_file_path)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Determine the pythonic path relative to the project root
    if not abs_module_path.startswith(project_root):
        raise ImportError(f"Module {abs_module_path} is outside of project root {project_root}")
        
    relative_path = os.path.relpath(abs_module_path, project_root)
    module_pythonic_path = relative_path.replace(os.sep, ".").rsplit(".py", 1)[0]
    
    try:
        return importlib.import_module(module_pythonic_path)
    except ImportError as e:
        raise ImportError(f"Could not import module from path: {module_pythonic_path}") from e

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