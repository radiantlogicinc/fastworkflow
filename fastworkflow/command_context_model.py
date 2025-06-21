from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.utils import python_utils

"""Utility for loading and traversing the single-file workflow command context model.

The canonical representation lives in a JSON file named ``context_inheritance_model.json``
inside the workflow _commands folder (e.g. ``examples/sample_workflow/_commands/context_inheritance_model.json``).

The ``context_inheritance_model.json`` contains a dictionary whose keys represent command context names
and values represent context definitions. Its primary role is to define inheritance 
relationships between contexts.

Schema (see Cursor rule ``context-model-design``):
------------------------------------------------
Context := {  # As defined in context_inheritance_model.json
    "base": list[str]      # Contexts whose commands are inherited. Cannot be empty or missing.
}

Validation Rules:
-----------------------------------------------
- For any context defined in ``context_inheritance_model.json``:
    - The "base" key MUST be present and be a list of strings. It cannot be empty or missing.
- An empty or missing ``context_inheritance_model.json`` file is valid.
-----------------------------------------------
This loader performs:
1. Validation of ``context_inheritance_model.json`` against the rules above.
2. Discovery of contexts and their direct commands from the ``_commands/`` filesystem structure.
3. Merging of filesystem-discovered commands with JSON-defined inheritance.
4. Cycle detection for the ``base`` context inheritance graph.
5. Resolution of effective command lists per context (own commands from filesystem + inherited commands from JSON).
"""


class CommandContextModelError(Exception):
    """Base class for errors in this module."""


class CommandContextModelValidationError(CommandContextModelError):
    """Raised when the context model fails validation."""


@dataclass
class CommandContextModel:
    """Represents the command context model for a workflow."""

    _workflow_path: str
    _command_contexts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    _resolved_commands: dict[str, list[str]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        self._resolve_inheritance()

    @classmethod
    def _discover_contexts_from_filesystem(
        cls, workflow_path: Path
    ) -> dict[str, dict[str, list[str]]]:
        """
        Scans the _commands directory to discover contexts (subdirectories)
        and their commands (Python files within those subdirectories).
        Returns a dictionary in the same format as context_inheritance_model.json contents.
        """
        discovered_contexts = {}
        commands_root_dir = workflow_path / "_commands"

        if not commands_root_dir.is_dir():
            return discovered_contexts

        # First, collect top-level commands (directly in _commands/)
        top_level_commands = []
        for command_file in commands_root_dir.glob("*.py"):
            if (
                command_file.is_file()
                and command_file.suffix == ".py"
                and not command_file.name.startswith("_")
                and command_file.name != "__init__.py"
            ):
                command_name_str = command_file.stem
                top_level_commands.append(command_name_str)
        
        # Add top-level commands to '*' context if any were found
        if top_level_commands:
            discovered_contexts["*"] = {"/": sorted(top_level_commands)}

        # Then collect commands in subdirectories (contexts)
        for item in commands_root_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):  # Treat directories as contexts
                context_name = item.name
                context_commands = []
                
                # Discover commands in this context directory using qualified names
                for command_file in item.glob("*.py"):
                    if (
                        command_file.is_file()
                        and command_file.suffix == ".py"
                        and not command_file.name.startswith("_")
                        and command_file.name != "__init__.py"
                    ):
                        command_name_str = command_file.stem
                        # Use qualified name format: "ContextName/command_name"
                        qualified_command_name = f"{context_name}/{command_name_str}"
                        context_commands.append(qualified_command_name)
                
                # Add context even if it has no .py files, making it discoverable for 'base' inheritance.
                discovered_contexts[context_name] = {"/": sorted(list(set(context_commands)))}
        
        return discovered_contexts

    @classmethod
    def load(cls, workflow_path: str | Path) -> CommandContextModel:
        """Loads and validates the command context model from a workflow path,
        augmenting with contexts and commands discovered from the _commands directory."""
        workflow_path_obj = Path(workflow_path)
        json_model_path = workflow_path_obj / "_commands/context_inheritance_model.json"

        # 1. Load from JSON file (if it exists)
        raw_contexts_from_json = {}
        if json_model_path.is_file():
            try:
                with json_model_path.open("r") as f:
                    raw_contexts_from_json = json.load(f)
            except json.JSONDecodeError as e:
                raise CommandContextModelValidationError(
                    f"Invalid JSON in {json_model_path}: {e}"
                ) from e
            if not isinstance(raw_contexts_from_json, dict):
                raise CommandContextModelValidationError(
                    f"Context model root in {json_model_path} must be a dictionary, "
                    f"but found {type(raw_contexts_from_json)}"
                )

            # Validate structure of each entry in raw_contexts_from_json
            for context_name_in_json, context_def_in_json in raw_contexts_from_json.items():
                if "/" in context_def_in_json:
                    raise CommandContextModelValidationError(
                        f"Context '{context_name_in_json}' in {json_model_path.name} "
                        f"must not contain a '/' key. Direct commands are discovered from the filesystem. "
                        f"Only 'base' key is allowed for context definitions in this file."
                    )
                # If a context is defined in JSON, it must be for defining 'base'
                if "base" not in context_def_in_json:
                    raise CommandContextModelValidationError(
                        f"Context '{context_name_in_json}' in {json_model_path.name} "
                        f"is missing the required 'base' key. Context definitions in this file are for inheritance only."
                    )
                if not context_def_in_json["base"] or \
                       not isinstance(context_def_in_json["base"], list) or \
                       not all(isinstance(b, str) for b in context_def_in_json["base"]):
                    raise CommandContextModelValidationError(
                        f"Key 'base' for context '{context_name_in_json}' in {json_model_path.name} "
                        f"must be a list of context name strings."
                    )

        # 2. Use CommandDirectory to get filesystem-derived commands
        cmd_dir = CommandDirectory.load(workflow_path)

        # Reconstruct the equivalent of discovered_contexts_from_fs using cmd_dir
        fs_derived_direct_commands = {}

        # Process all qualified command names from CommandDirectory
        for qualified_cmd_name in cmd_dir.map_command_2_metadata.keys():
            if "/" in qualified_cmd_name:
                # Extract context part from qualified name (e.g., "Core" from "Core/abort")
                context_part, _ = qualified_cmd_name.split("/", 1)
                if context_part not in fs_derived_direct_commands:
                    fs_derived_direct_commands[context_part] = {"/": []}
                fs_derived_direct_commands[context_part]["/"].append(qualified_cmd_name)
            else:  # Global command (e.g., "wildcard")
                if "*" not in fs_derived_direct_commands:
                    fs_derived_direct_commands["*"] = {"/": []}
                fs_derived_direct_commands["*"]["/"].append(qualified_cmd_name)

        # Sort command lists for consistency
        for context_data in fs_derived_direct_commands.values():
            context_data["/"] = sorted(list(set(context_data["/"])))

        # 3. Merge JSON-defined inheritance with Filesystem-derived direct commands
        merged_contexts = {}
        # Combine context names from JSON and those with actual commands found by CommandDirectory
        all_context_names = set(raw_contexts_from_json.keys()) | set(fs_derived_direct_commands.keys())

        for name in all_context_names:
            json_def = raw_contexts_from_json.get(name, {})  # Contains only 'base' if present
            fs_def = fs_derived_direct_commands.get(name, {})  # Contains only '/' from CommandDirectory data

            current_context_definition: dict[str, list[str]] = {}

            # 'base' comes ONLY from JSON
            if "base" in json_def:
                current_context_definition["base"] = json_def["base"]

            # '/' commands come ONLY from CommandDirectory's data for this context 'name'
            current_context_definition["/"] = fs_def["/"] if "/" in fs_def else []
            # Validate that each context has either:
            # 1. Direct commands from filesystem ('/' key with non-empty list)
            # 2. OR is used as a base in JSON ('base' key)
            if not current_context_definition.get("/") and "base" not in current_context_definition:
                # This is a context with no commands and no inheritance role
                raise CommandContextModelValidationError(
                    f"Context '{name}' has no commands and is not used as a base for inheritance. "
                    f"Each context must either contain command files or be used as a base."
                )

            merged_contexts[name] = current_context_definition

        return cls(_workflow_path=str(workflow_path_obj), _command_contexts=merged_contexts)

    def _resolve_inheritance(self) -> None:
        """Resolves command inheritance and detects cycles."""
        # The core of this is a topological sort.
        # We can do this with a simple depth-first traversal.
        for context_name in self._command_contexts:
            self.commands(context_name)

    def commands(self, context_name: str, visiting: set[str] | None = None) -> list[str]:
        """Returns the effective list of commands for a given command context, including inherited ones."""
        if context_name in self._resolved_commands:
            return self._resolved_commands[context_name]

        if context_name not in self._command_contexts:
            raise CommandContextModelValidationError(f"Context '{context_name}' not found in model.")

        if visiting is None:
            visiting = set()

        if context_name in visiting:
            raise CommandContextModelValidationError(
                f"Inheritance cycle detected: {' -> '.join(list(visiting) + [context_name])}"
            )

        visiting.add(context_name)

        context_def = self._command_contexts[context_name]
        own_commands_list = context_def.get("/") or []  # Qualified names from FS

        # Track which simple command names have been added and their qualified names
        # This mapping helps us track which command has been added for each simple name
        simple_to_qualified = {}

        # 1. Add own commands (highest precedence)
        # Own commands are typically already sorted by _discover_contexts_from_filesystem
        # but iterating explicitly maintains their relative order if any.
        for own_cmd_qualified in own_commands_list:
            own_cmd_simple = own_cmd_qualified.split('/')[-1]
            simple_to_qualified[own_cmd_simple] = own_cmd_qualified

        # 2. Add commands from base contexts (in order of definition in the "base" list)
        base_contexts_list = context_def.get("base") or []
        for base_context_name in base_contexts_list: # Process in defined order for precedence
            # Pass a copy of 'visiting' for the recursive call
            inherited_commands_qualified_list = self.commands(base_context_name, visiting.copy()) 

            for inherited_cmd_qualified in inherited_commands_qualified_list:
                inherited_cmd_simple = inherited_cmd_qualified.split('/')[-1]
                # Only add if we haven't seen this simple name yet (derived context overrides base)
                if inherited_cmd_simple not in simple_to_qualified:
                    simple_to_qualified[inherited_cmd_simple] = inherited_cmd_qualified

        final_effective_commands_list = sorted(simple_to_qualified.values())
        self._resolved_commands[context_name] = final_effective_commands_list

        visiting.remove(context_name) # Remove after processing and caching for this context_name

        return final_effective_commands_list

    # ---------------------------------------------------------------------
    # Context callback class resolution
    # ---------------------------------------------------------------------

    def get_context_class(self, context_name: str, module_type: fastworkflow.ModuleType):
        """Retrieve the callback class implementation for a given context.

        Currently only supports ModuleType.CONTEXT_CLASS. Returns None if not found.
        """

        if module_type != fastworkflow.ModuleType.CONTEXT_CLASS:
            raise ValueError(
                "CommandContextModel.get_context_class only supports ModuleType.CONTEXT_CLASS"
            )

        # Load command directory for metadata
        cmd_dir = CommandDirectory.load(self._workflow_path)

        try:
            context_metadata = cmd_dir.map_context_2_metadata[context_name]
        except KeyError:
            return None

        if module := python_utils.get_module(
            context_metadata.context_module_path,
            context_metadata.workflow_folderpath or self._workflow_path,
        ):
            return getattr(module, context_metadata.context_class, None)
        else:
            return None
