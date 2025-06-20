from typing import Dict, Set, Optional, List, Tuple
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.command_import_utils import generate_import_statements
from fastworkflow.build.command_dependency_resolver import resolve_command_dependencies
from fastworkflow.utils.python_utils import find_module_dependencies, get_module_import_path
from fastworkflow.utils.logging import logger

class DependencyManager:
    """
    Manages command dependencies and import generation for a set of analyzed classes.
    Provides utilities for import blocks, dependency graph, cycle detection, and diagnostics.
    """
    def __init__(self, classes: Dict[str, ClassInfo], source_dir: str):
        self.classes = classes
        self.source_dir = source_dir
        self.graph = resolve_command_dependencies(classes)
        # Build a mapping from command_id to (class_info, method/property)
        self.command_map = {}
        for class_info in classes.values():
            for method in class_info.methods:
                cmd_id = f"{class_info.name}.{method.name}"
                self.command_map[cmd_id] = (class_info, method)
            for prop in class_info.properties:
                cmd_id = f"{class_info.name}.get_{prop.name}"
                self.command_map[cmd_id] = (class_info, prop)

    def get_imports_for_command(self, command_id: str, class_name_to_module_path: Optional[Dict[str, str]] = None) -> str:
        """
        Return the import block for a given command_id.
        """
        if command_id not in self.command_map:
            raise ValueError(f"Unknown command_id: {command_id}")
        class_info, _ = self.command_map[command_id]
        return generate_import_statements(class_info, self.source_dir, class_name_to_module_path)

    def get_command_dependencies(self, command_id: str) -> Set[str]:
        """
        Return the set of command_ids that the given command depends on.
        """
        return self.graph.get(command_id, set())

    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Return the full dependency graph.
        """
        return self.graph

    def check_circular_dependencies(self) -> List[List[str]]:
        """
        Return a list of cycles (each as a list of command_ids) if any exist.
        Uses DFS for cycle detection.
        """
        visited = set()
        stack = []
        cycles = []
        def dfs(node, path):
            if node in path:
                idx = path.index(node)
                cycles.append(path[idx:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            for dep in self.graph.get(node, []):
                dfs(dep, path + [node])
        for node in self.graph:
            dfs(node, [])
        # Remove duplicate cycles (cycles with same set of nodes)
        unique = []
        seen = set()
        for cycle in cycles:
            key = tuple(sorted(cycle))
            if key not in seen:
                unique.append(cycle)
                seen.add(key)
        return unique

    def diagnostics(self) -> str:
        """
        Return a summary of dependency issues (cycles, orphan commands, etc.).
        """
        cycles = self.check_circular_dependencies()
        orphans = [cmd for cmd, deps in self.graph.items() if not deps]
        msg = []
        if cycles:
            msg.append(f"Circular dependencies detected: {cycles}")
        else:
            msg.append("No circular dependencies detected.")
        msg.append(f"Orphan commands (no dependencies): {orphans}")
        return "\n".join(msg) 