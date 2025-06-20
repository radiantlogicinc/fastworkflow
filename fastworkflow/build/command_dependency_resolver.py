from typing import Dict, Set
from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.utils.python_utils import find_module_dependencies

def resolve_command_dependencies(classes: Dict[str, ClassInfo]) -> Dict[str, Set[str]]:
    """
    Build a dependency graph for commands.
    Each key is a command id (ClassName.method_name, ClassName.get_properties, ClassName.set_properties),
    and the value is a set of command ids it depends on (if any).
    Only dependencies that correspond to other commands are included.
    """
    class_map = {c.name: c for c in classes.values()}
    command_ids = set() # This set might not be strictly necessary if graph keys cover all commands

    graph = {}

    for class_name, class_info in classes.items():
        # Methods
        for method in class_info.methods:
            cmd_id = f"{class_name}.{method.name}"
            command_ids.add(cmd_id)
            deps = set()
            # Assuming find_module_dependencies gives class names the current class_info depends on
            for dep_class_name in find_module_dependencies(class_info):
                if dep_class_name in class_map:
                    dep_class_info = class_map[dep_class_name]
                    for m in dep_class_info.methods:
                        deps.add(f"{dep_class_name}.{m.name}")
                    if dep_class_info.properties:
                        deps.add(f"{dep_class_name}.get_properties")
                    if dep_class_info.all_settable_properties:
                        deps.add(f"{dep_class_name}.set_properties")
            graph[cmd_id] = deps

        # Get Properties command
        if class_info.properties:
            cmd_id = f"{class_name}.get_properties"
            command_ids.add(cmd_id)
            deps = set()
            for dep_class_name in find_module_dependencies(class_info):
                if dep_class_name in class_map:
                    dep_class_info = class_map[dep_class_name]
                    for m in dep_class_info.methods:
                        deps.add(f"{dep_class_name}.{m.name}")
                    if dep_class_info.properties:
                        deps.add(f"{dep_class_name}.get_properties")
                    if dep_class_info.all_settable_properties:
                        deps.add(f"{dep_class_name}.set_properties")
            graph[cmd_id] = deps
        
        # Set Properties command
        if class_info.all_settable_properties:
            cmd_id = f"{class_name}.set_properties"
            command_ids.add(cmd_id)
            deps = set()
            for dep_class_name in find_module_dependencies(class_info):
                if dep_class_name in class_map:
                    dep_class_info = class_map[dep_class_name]
                    for m in dep_class_info.methods:
                        deps.add(f"{dep_class_name}.{m.name}")
                    if dep_class_info.properties:
                        deps.add(f"{dep_class_name}.get_properties")
                    if dep_class_info.all_settable_properties:
                        deps.add(f"{dep_class_name}.set_properties")
            graph[cmd_id] = deps
            
    return graph 