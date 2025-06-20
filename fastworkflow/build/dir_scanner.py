import os
from typing import List, Set

EXCLUDE_DIRS: Set[str] = {'venv', '__pycache__', '.git', 'build', 'dist'}

def find_python_files(root_dir: str, exclude_dirs: Set[str] = EXCLUDE_DIRS, verbose: bool = False) -> List[str]:
    """Recursively find all readable Python files in root_dir, excluding specified directories."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # Exclude unwanted directories in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r'):
                        pass  # Just to check readability
                    python_files.append(file_path)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Skipping unreadable file {file_path}: {e}")
    return python_files 