import os
from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.utils.python_utils import find_module_dependencies

def generate_import_statements(class_info, source_dir, class_name_to_module_path=None):
    """
    Generate import statements for a command file based on class_info and its dependencies.
    - Always imports typing and pydantic basics
    - Imports the main class using relative import: from ..<leaf_of_source_dir>.<class file name> import <Class Name>
    - For each dependency, imports using the same pattern if possible
    Args:
        class_info: ClassInfo object
        source_dir: root of the source tree
        class_name_to_module_path: Optional dict mapping class names to module import paths
    Returns:
        str: import block
    """
    imports = [
        "from typing import List, Dict, Any, Optional, Union",
        "from pydantic import BaseModel, Field"
    ]
    # Use only the last component of source_dir for import
    leaf_src = os.path.basename(os.path.normpath(source_dir))
    # Main class import
    class_file = os.path.splitext(os.path.basename(class_info.module_path))[0]
    imports.append(f"from ...{leaf_src}.{class_file} import {class_info.name}")
    # Dependency imports
    dependencies = find_module_dependencies(class_info)
    for dep in sorted(dependencies):
        dep_module_path = None
        if class_name_to_module_path and dep in class_name_to_module_path:
            dep_module_path = class_name_to_module_path[dep]
        elif hasattr(class_info, 'dependency_module_paths') and dep in class_info.dependency_module_paths:
            dep_module_path = class_info.dependency_module_paths[dep]
        if dep_module_path:
            dep_file = os.path.splitext(os.path.basename(dep_module_path))[0]
            imports.append(f"from ...{leaf_src}.{dep_file} import {dep}")
        else:
            imports.append(f"# You may need to import {dep}")
    # Ensure uniqueness of imports before joining
    unique_imports = list(dict.fromkeys(imports)) # Preserves order for Python 3.7+
    return "\n".join(unique_imports) 