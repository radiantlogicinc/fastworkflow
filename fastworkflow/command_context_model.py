from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
import os
from pathlib import Path

import fastworkflow
from fastworkflow.command_directory import get_cached_command_directory
from fastworkflow.utils import python_utils
from fastworkflow.workflow_inheritance_model import WorkflowInheritanceModel

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
    _resolved_ancestors: dict[str, list[str]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        # Inheritance is resolved lazily on first `commands()` call to
        # avoid upfront DFS across all contexts during cold-start.
        pass

    @classmethod
    def load(cls, workflow_path: str | Path) -> CommandContextModel:
        """Loads and validates the command context model from a workflow path,
        augmenting with contexts and commands discovered from the _commands directory."""
        workflow_path_obj = Path(workflow_path)
        
        # 1. Load command inheritance from JSON file (if it exists)
        raw_contexts_from_json = cls._load_inheritance_json(workflow_path_obj)
        
        # 2. Use cached CommandDirectory to get filesystem-derived commands
        cmd_dir = get_cached_command_directory(str(workflow_path_obj))
        fs_derived_direct_commands = {}
        for qualified_cmd_name in cmd_dir.get_execution_commands():
            if "/" in qualified_cmd_name:
                context_part, _ = qualified_cmd_name.split("/", 1)
                if context_part not in fs_derived_direct_commands:
                    fs_derived_direct_commands[context_part] = {"/": []}
                fs_derived_direct_commands[context_part]["/"].append(qualified_cmd_name)
            else:
                if "*" not in fs_derived_direct_commands:
                    fs_derived_direct_commands["*"] = {"/": []}
                fs_derived_direct_commands["*"]["/"].append(qualified_cmd_name)
        fs_derived_direct_commands.pop('Core', None)
        for context_data in fs_derived_direct_commands.values():
            context_data["/"] = sorted(list(set(context_data["/"])))

        # 3. Merge JSON-defined inheritance with Filesystem-derived direct commands
        merged_contexts = {}
        all_context_names = set(raw_contexts_from_json.keys()) | set(fs_derived_direct_commands.keys())
        for name in all_context_names:
            json_def = raw_contexts_from_json.get(name, {})
            fs_def = fs_derived_direct_commands.get(name, {})
            current_context_definition: dict[str, list[str]] = {}
            if "base" in json_def:
                current_context_definition["base"] = json_def["base"]
            current_context_definition["/"] = fs_def.get("/", [])
            if not current_context_definition.get("/") and "base" not in current_context_definition:
                raise CommandContextModelValidationError(
                    f"Context '{name}' has no commands and is not used as a base for inheritance."
                )
            merged_contexts[name] = current_context_definition

        # Create the instance with the merged command contexts
        instance = cls(_workflow_path=str(workflow_path_obj), _command_contexts=merged_contexts)

        # 4. Load and resolve the context hierarchy (for parent relationships)
        hierarchy_data = instance._load_context_hierarchy()
        instance._resolve_ancestry(hierarchy_data)
        
        return instance

    @classmethod
    def _load_inheritance_json(cls, workflow_path: str | Path) -> dict[str, dict[str, list[str]]]:
        """Load and validate ``_commands/context_inheritance_model.json`` without importing commands."""
        workflow_path_obj = Path(workflow_path)
        json_model_path = workflow_path_obj / "_commands/context_inheritance_model.json"
        if not json_model_path.is_file():
            return {}
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
        for context_name_in_json, context_def_in_json in raw_contexts_from_json.items():
            if not isinstance(context_def_in_json, dict):
                raise CommandContextModelValidationError(
                    f"Definition for context '{context_name_in_json}' in {json_model_path.name} "
                    f"must be a dictionary, but found {type(context_def_in_json)}"
                )
            if "/" in context_def_in_json:
                raise CommandContextModelValidationError(
                    f"Context '{context_name_in_json}' in {json_model_path.name} "
                    f"must not contain a '/' key. Only 'base' key is allowed."
                )
            if "base" not in context_def_in_json:
                raise CommandContextModelValidationError(
                    f"Context '{context_name_in_json}' in {json_model_path.name} "
                    f"is missing the required 'base' key."
                )
            if not isinstance(context_def_in_json["base"], list) or \
                   not all(isinstance(b, str) for b in context_def_in_json["base"]):
                raise CommandContextModelValidationError(
                    f"Key 'base' for context '{context_name_in_json}' in {json_model_path.name} "
                    f"must be a list of context name strings."
                )
        return raw_contexts_from_json

    @classmethod
    def _validate_context_inheritance(
        cls,
        context_name: str,
        contexts: dict[str, dict[str, list[str]]],
        valid_contexts: set[str] | None = None,
        visiting: set[str] | None = None,
    ) -> None:
        """Validate that *context_name* and its transitive ``base`` contexts exist in
        *valid_contexts* (defaults to ``set(contexts)``) and contain no inheritance cycle."""
        allowed = set(contexts) if valid_contexts is None else valid_contexts
        if context_name not in allowed:
            # A context must be defined either by having command files
            # or by being part of a command inheritance structure. If it's not
            # in _command_contexts, it's an unknown/invalid context.
            raise CommandContextModelValidationError(f"Context '{context_name}' not found in model.")

        if visiting is None:
            visiting = set()

        if context_name in visiting:
            raise CommandContextModelValidationError(
                f"Inheritance cycle detected: {' -> '.join(list(visiting) + [context_name])}"
            )

        next_visiting = visiting.copy()
        next_visiting.add(context_name)
        for base_context_name in contexts.get(context_name, {}).get("base") or []:
            cls._validate_context_inheritance(
                base_context_name, contexts, allowed, next_visiting
            )

    @staticmethod
    def _ordered_bases_from_map(
        context_name: str,
        contexts: dict[str, dict[str, list[str]]],
    ) -> list[str]:
        """Depth-first declared base contexts for *context_name*, each once, from *contexts*."""
        ordered: list[str] = []
        seen: set[str] = {context_name}

        def walk(name: str) -> None:
            for base in contexts.get(name, {}).get("base") or []:
                if base in seen:
                    continue
                seen.add(base)
                ordered.append(base)
                walk(base)

        walk(context_name)
        return ordered

    @classmethod
    def load_base_contexts(
        cls,
        workflow_path: str | Path,
        known_contexts: set[str] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Load and validate ``_commands/context_inheritance_model.json`` and return
        ``{context: (base, ...)}`` in depth-first declared order without importing any command
        or creating ``___command_info``.

        Validates the JSON schema, verifies that every referenced base context exists
        (either in the JSON or in *known_contexts*, discovered from ``_commands/`` folders
        when *known_contexts* is not provided), and rejects inheritance cycles with
        :class:`CommandContextModelValidationError`.
        """
        workflow_path_obj = Path(workflow_path)
        raw_contexts = cls._load_inheritance_json(workflow_path_obj)
        if not raw_contexts:
            return {}

        if known_contexts is None:
            known: set[str] = set()
            inh = WorkflowInheritanceModel.load(str(workflow_path_obj))
            base_wf_paths = [Path(p) for p in inh.resolve_base_paths(str(workflow_path_obj))]
            for wf_dir in (*base_wf_paths, workflow_path_obj):
                cmds_dir = wf_dir / "_commands"
                if cmds_dir.is_dir():
                    for entry in cmds_dir.iterdir():
                        if entry.is_dir() and not entry.name.startswith(("_", ".")):
                            known.add(entry.name)
        else:
            known = set(known_contexts)

        valid_contexts = set(raw_contexts.keys()) | known
        for ctx_name in raw_contexts:
            cls._validate_context_inheritance(ctx_name, raw_contexts, valid_contexts)

        all_contexts = (set(raw_contexts.keys()) | known) - {"*", ""}
        return {
            ctx: bases
            for ctx in sorted(all_contexts)
            if (bases := tuple(cls._ordered_bases_from_map(ctx, raw_contexts)))
        }

    def _load_context_hierarchy(self) -> dict:
        """Loads and validates the context hierarchy model from the workflow path."""
        hierarchy_model_path = Path(self._workflow_path) / "context_hierarchy_model.json"
        if not hierarchy_model_path.is_file():
            return {}

        with hierarchy_model_path.open("r") as f:
            hierarchy_data = json.load(f)

        return hierarchy_data

    def _resolve_ancestry(self, hierarchy: dict[str, dict[str, list[str]]]) -> None:
        """Resolves the complete ancestry for each context and detects cycles."""
        all_contexts = set(self._command_contexts.keys()) | set(hierarchy.keys())
        for context_name in all_contexts:
            if context_name not in self._resolved_ancestors:
                self.get_ancestor_contexts(context_name, _hierarchy=hierarchy)

    def get_ancestor_contexts(
        self, 
        context_name: str, 
        visiting: set[str] | None = None,
        _hierarchy: dict[str, dict[str, list[str]]] | None = None
    ) -> list[str]:
        """
        Returns the effective list of ancestor contexts for a given context.
        This includes the full parent chain up to the root.
        """
        if context_name in self._resolved_ancestors:
            return self._resolved_ancestors[context_name]

        if _hierarchy is None:
            _hierarchy = self._load_context_hierarchy()

        if visiting is None:
            visiting = set()

        if context_name in visiting:
            raise CommandContextModelValidationError(
                f"Context hierarchy cycle detected: {' -> '.join(list(visiting) + [context_name])}"
            )

        visiting.add(context_name)

        context_def = _hierarchy.get(context_name, {})
        parent_contexts = context_def.get("parent", [])

        if not isinstance(parent_contexts, list) or not all(isinstance(p, str) for p in parent_contexts):
            raise CommandContextModelValidationError(
                f"Key 'parent' for context '{context_name}' in context_hierarchy_model.json "
                f"must be a list of context name strings."
            )

        all_ancestors = set()
        for parent in parent_contexts:
            all_ancestors.add(parent)
            grandparents = self.get_ancestor_contexts(parent, visiting.copy(), _hierarchy)
            all_ancestors.update(grandparents)

        final_ancestors = sorted(list(all_ancestors))
        self._resolved_ancestors[context_name] = final_ancestors

        visiting.remove(context_name)

        return final_ancestors

    def _resolve_inheritance(self) -> None:
        """Resolves command inheritance and detects cycles."""
        for context_name in self._command_contexts:
            self._validate_context_inheritance(context_name, self._command_contexts)
            self.commands(context_name)

    def commands(self, context_name: str, visiting: set[str] | None = None) -> list[str]:
        """Returns the effective list of commands for a given command context, including inherited ones."""
        if context_name in self._resolved_commands:
            return self._resolved_commands[context_name]

        self._validate_context_inheritance(
            context_name, self._command_contexts, visiting=visiting
        )
        if visiting is None:
            visiting = set()
        visiting.add(context_name)

        context_def = self._command_contexts.get(context_name, {})
        own_commands_list = context_def.get("/") or []

        simple_to_qualified = {}

        for own_cmd_qualified in own_commands_list:
            own_cmd_simple = own_cmd_qualified.split('/')[-1]
            simple_to_qualified[own_cmd_simple] = own_cmd_qualified

        base_contexts_list = context_def.get("base") or []
        for base_context_name in base_contexts_list:
            inherited_commands_qualified_list = self.commands(base_context_name, visiting.copy()) 

            for inherited_cmd_qualified in inherited_commands_qualified_list:
                inherited_cmd_simple = inherited_cmd_qualified.split('/')[-1]
                if inherited_cmd_simple not in simple_to_qualified:
                    simple_to_qualified[inherited_cmd_simple] = inherited_cmd_qualified

        final_effective_commands_list = sorted(simple_to_qualified.values())
        self._resolved_commands[context_name] = final_effective_commands_list

        visiting.remove(context_name)

        return final_effective_commands_list

    def inherited_base_contexts(self, context_name: str) -> set[str]:
        """Every context whose commands *context_name* inherits, transitively.

        Unknown contexts (including the global ``*``) inherit nothing. Cycles
        are rejected when the model loads, so the walk terminates.
        """
        bases: set[str] = set()
        pending = list(self._command_contexts.get(context_name, {}).get("base") or [])
        while pending:
            base = pending.pop()
            if base not in bases:
                bases.add(base)
                pending.extend(self._command_contexts.get(base, {}).get("base") or [])
        return bases

    def ordered_base_contexts(self, context_name: str) -> list[str]:
        """Every context whose commands *context_name* inherits, transitively,
        depth-first in declared ``base`` order, each once.

        Matches the order :meth:`commands` resolves inherited commands in.
        Unknown contexts (including the global ``*``) inherit nothing and give
        ``[]``. Cycles are rejected when the model loads, so the walk
        terminates.
        """
        return self._ordered_bases_from_map(context_name, self._command_contexts)

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
        cmd_dir = get_cached_command_directory(self._workflow_path)

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


# ---------------------------------------------------------------------
# Workflow info helper (module-level)
# ---------------------------------------------------------------------

def get_workflow_info(chat_session_or_path: fastworkflow.ChatSession | str) -> dict[str, Any]:
    """
    Return workflow-level metadata for the active workflow in the given chat session or workflow path.

    Shape:
    {
      "workflow_name": str,
      "available_contexts": list[str],
      "context_inheritance_model": dict,
      "context_hierarchy_model": dict
    }

    Notes:
    - available_contexts are derived from CommandContextModel.load(workflow_path)
      so they reflect filesystem + inheritance.
    - current context and purpose/description are intentionally omitted.
    - Raw JSON for inheritance (from _commands/context_inheritance_model.json)
      and hierarchy (from context_hierarchy_model.json) is returned verbatim when present.
    """
    if isinstance(chat_session_or_path, str):
        workflow_path_str = chat_session_or_path
    else:
        workflow_path_str = chat_session_or_path.get_active_workflow().folderpath
    workflow_name = os.path.basename(os.path.abspath(workflow_path_str))

    available_contexts: list[str]
    inheritance_json: dict[str, Any] = {}
    hierarchy_json: dict[str, Any] = {}

    try:
        # Derive contexts via consolidated model
        context_model = CommandContextModel.load(workflow_path_str)
        # Do not normalize names beyond keeping what the model exposes
        available_contexts = sorted([key for key in context_model._command_contexts.keys() if key != 'IntentDetection'])
    except Exception:
        available_contexts = ["/"]

    # Load raw context inheritance model (if present)
    try:
        inheritance_path = Path(workflow_path_str) / "_commands" / "context_inheritance_model.json"
        if inheritance_path.is_file():
            inheritance_json = json.loads(inheritance_path.read_text())
    except Exception:
        inheritance_json = {}

    # Load raw context hierarchy model (if present)
    try:
        hierarchy_path = Path(workflow_path_str) / "context_hierarchy_model.json"
        if hierarchy_path.is_file():
            hierarchy_json = json.loads(hierarchy_path.read_text())
    except Exception:
        hierarchy_json = {}

    return {
        "workflow_name": workflow_name,
        "available_contexts": available_contexts,
        "context_inheritance_model": inheritance_json,
        "context_hierarchy_model": hierarchy_json,
    }
