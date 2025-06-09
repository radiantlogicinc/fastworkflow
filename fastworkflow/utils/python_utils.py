import os
import importlib

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