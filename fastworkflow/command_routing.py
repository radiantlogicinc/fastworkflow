"""Command routing system for FastWorkflow.

This module consolidates three previously separate concerns:
1. Command discovery and context mapping (was command_router.py)
2. Full routing definition with inheritance, core commands, and class loading (was command_routing_definition.py)
3. Utterance helpers for commands (was utterance_definition.py)

It exposes two primary classes:
- RoutingDefinition: The main class that handles command routing logic
- RoutingRegistry: A singleton registry that caches RoutingDefinition instances
"""


from __future__ import annotations

import contextlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, ClassVar

from pydantic import BaseModel, ConfigDict

from fastworkflow import ModuleType
from fastworkflow.command_directory import CommandDirectory, UtteranceMetadata, get_cached_command_directory
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger


class RoutingDefinition(BaseModel):
    """
    Defines the available commands for each context in a workflow.
    
    This class consolidates the functionality previously spread across:
    - Simple context-to-commands and command-to-contexts mapping
    - Full routing with inheritance, core commands
    - Access to command utterances
    
    It builds these mappings from the workflow's context model and command directory.
    """
    workflow_folderpath: str
    command_directory: CommandDirectory
    context_model: CommandContextModel
    
    # Resolved command lists for each context (with inheritance applied)
    contexts: dict[str, list[str]]
    
    # Simple mappings for quick lookups (previously in simple router)
    # Maps context → commands (set)
    command_directory_map: Dict[str, Set[str]] = {}
    # Maps command → contexts (set)
    routing_definition_map: Dict[str, Set[str]] = {}

    # Single, process-wide cache of imported command classes
    _GLOBAL_COMMAND_CLASS_CACHE: ClassVar[dict[str, Type[Any]]] = {}

    # ------------------------------------------------------------------
    # Command Router functionality
    # ------------------------------------------------------------------
    
    def get_commands_for_context(self, context_name: str) -> Set[str]:
        """Return the set of commands for *context_name* (empty set if unknown)."""
        return self.command_directory_map.get(context_name, set())

    def get_contexts_for_command(self, command_name: str) -> Set[str]:
        """Return the set of contexts that can execute *command_name*."""
        return self.routing_definition_map.get(command_name, set())

    # ------------------------------------------------------------------
    # Command Routing Definition functionality
    # ------------------------------------------------------------------
    
    def get_command_names(self, context: str) -> list[str]:
        """Returns the list of command names available in the given context."""
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found in the workflow.")
        commands = self.contexts[context]
        if 'wildcard' in commands:
            commands = [cmd for cmd in commands if cmd != 'wildcard']
        return commands

    def get_command_class(self, command_name: str, module_type: ModuleType) -> Optional[Type[Any]]:
        """
        Retrieves a command's implementation class from the cache or loads it.
        The context is no longer needed for lookup, as command names are unique.
        """
        cache_key = f"{command_name}:{module_type.name}"

        cls = type(self)

        # First check the **global** cache shared by every RoutingDefinition
        if cache_key in cls._GLOBAL_COMMAND_CLASS_CACHE:
            return cls._GLOBAL_COMMAND_CLASS_CACHE[cache_key]

        # Not cached yet – load it now.
        result = self._load_command_class(command_name, module_type)

        # Store in both caches so *all* future RoutingDefinition objects benefit.
        cls._GLOBAL_COMMAND_CLASS_CACHE[cache_key] = result
        return result

    def _load_command_class(self, command_name: str, module_type: ModuleType) -> Optional[Type[Any]]:
        """Loads a command's class from its source file."""
        try:
            # Lazily hydrate metadata so that Signature-related fields are available
            self.command_directory.ensure_command_hydrated(command_name)

            command_metadata = self.command_directory.get_command_metadata(command_name)
        except KeyError:
            # This is the expected path for commands in the context model that have no .py file.
            return None

        if module_type == ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS:
            module_file_path = command_metadata.parameter_extraction_signature_module_path
            module_class_name = command_metadata.input_for_param_extraction_class
        elif module_type == ModuleType.COMMAND_PARAMETERS_CLASS:
            module_file_path = command_metadata.parameter_extraction_signature_module_path
            module_class_name = command_metadata.command_parameters_class
        elif module_type == ModuleType.RESPONSE_GENERATION_INFERENCE:
            module_file_path = command_metadata.response_generation_module_path
            module_class_name = command_metadata.response_generation_class_name
        else:
            raise ValueError(f"Invalid module type '{module_type}'")

        # If the command does not define the requested module/class (e.g. no Signature
        # for parameter-extraction), bail out early.  Returning ``None`` signals to
        # callers that the implementation is not available, which higher-level logic
        # already handles gracefully.
        if not module_file_path or not module_class_name:
            return None

        # Use the cached module loader
        module = python_utils.get_module(str(module_file_path), self.workflow_folderpath)

        # Handle nested classes like 'Signature.Input'
        if '.' in module_class_name:
            parts = module_class_name.split('.')
            cls_obj = module
            for part in parts:
                cls_obj = getattr(cls_obj, part, None)
            return cls_obj
        
        return getattr(module, module_class_name, None)

    # ------------------------------------------------------------------
    # Utterance Definition functionality
    # ------------------------------------------------------------------
    
    def get_command_utterances(self, command_name: str) -> UtteranceMetadata:
        """Gets the utterance metadata for a single, specific command."""
        if utterance_metadata := self.command_directory.get_utterance_metadata(
            command_name
        ):
            return utterance_metadata

        raise KeyError(
            f"Could not find utterance metadata for command '{command_name}'. "
            "It might be missing from the _commands directory."
        )

    def get_sample_utterances(self, command_context: str) -> list[str]:
        """Gets a sample utterance for each command in the given context."""
        command_names = self.get_command_names(command_context)
        sample_utterances = []
        for command_name in command_names:
            command_utterances = self.get_command_utterances(command_name)
            if not command_utterances:
                continue

            if command_utterances.template_utterances:
                sample_utterances.append(f"{command_name}: {command_utterances.template_utterances[0]}")
            elif command_utterances.plain_utterances:
                sample_utterances.append(f"{command_name}: {command_utterances.plain_utterances[0]}")
        
        return sample_utterances

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save the routing definition to JSON, excluding the command_directory"""
        save_path = f"{CommandDirectory.get_commandinfo_folderpath(self.workflow_folderpath)}/routing_definition.json"
        
        # Create a dict without command_directory
        save_dict = self.model_dump(exclude={'command_directory'})
        
        # Convert sets to lists for JSON serialization
        if 'command_directory_map' in save_dict:
            save_dict['command_directory_map'] = {k: list(v) for k, v in save_dict['command_directory_map'].items()}
        if 'routing_definition_map' in save_dict:
            save_dict['routing_definition_map'] = {k: list(v) for k, v in save_dict['routing_definition_map'].items()}
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=4)

    @classmethod
    def load(cls, workflow_folderpath):
        """Load the routing definition from JSON"""
        load_path = f"{CommandDirectory.get_commandinfo_folderpath(workflow_folderpath)}/routing_definition.json"
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["workflow_folderpath"] = workflow_folderpath

        # Use cached command directory
        command_directory = get_cached_command_directory(workflow_folderpath)
        data["command_directory"] = command_directory

        return cls.model_validate(data)

    @classmethod
    def build(cls, workflow_folderpath: str) -> "RoutingDefinition":
        """
        Builds the routing definition by loading the context model
        and discovering commands from the filesystem.
        
        The command context model now uses qualified names for commands in context subdirectories,
        in the format 'ContextName/command_name'. This method ensures these qualified names
        are properly handled when building the routing definition.
        """       
        # Validate that the workflow directory exists and has a _commands folder
        commands_dir = Path(workflow_folderpath) / "_commands"
        if not commands_dir.is_dir():
            raise RuntimeError(f"Workflow path '{workflow_folderpath}' does not contain '_commands' directory")
            
        # Use cached command directory to avoid repeated filesystem scanning
        command_directory = get_cached_command_directory(workflow_folderpath)
        context_model = CommandContextModel.load(workflow_folderpath)

        # Use dynamically discovered core commands
        core_commands = command_directory.core_command_names
        logger.debug(f"Core commands: {str(core_commands)}, Workflow folderpath {workflow_folderpath}")

        resolved_contexts = {}
        for context_name in context_model._command_contexts:
            # Get commands for this context from the context model
            context_commands = context_model.commands(context_name)

            # Add core commands to every context
            resolved_contexts[context_name] = sorted(set(context_commands) | set(core_commands))

        # Ensure global context '*' also exists with core commands (if not in model)
        if '*' not in resolved_contexts:
            resolved_contexts['*'] = sorted(core_commands)
        routing_definition = cls(
            workflow_folderpath=workflow_folderpath,
            command_directory=command_directory,
            context_model=context_model,
            contexts=resolved_contexts,
        )

        # ------------------------------------------------------------
        # Eager module pre-import
        # ------------------------------------------------------------
        # Import every command implementation we know about right now so
        # that later look-ups hit the in-process cache and do not have to
        # contend for the importlib module lock at request time.  This
        # increases build() cost slightly but removes ~300 ms of latency
        # from the first real user message.
        all_command_names: list[str] = command_directory.get_commands()

        for cmd_name in all_command_names:
            # The vast majority of run-time look-ups are for the inference
            # implementation.  Loading the other two module types is cheap
            # so we bring them all in here to fully populate the cache.
            for _mtype in (
                ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
                ModuleType.COMMAND_PARAMETERS_CLASS,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            ):
                # We purposefully ignore load failures (e.g., commands that
                # do not define a Signature); higher-level logic already
                # handles missing implementations.
                with contextlib.suppress(Exception):
                    routing_definition.get_command_class(cmd_name, _mtype)

        # Build the simple mappings for quick lookups
        routing_definition._build_simple_mappings()

        # Only save if we were able to build successfully
        if resolved_contexts != {"*": []}:
            routing_definition.save()
        elif not workflow_folderpath.endswith('fastworkflow'):
            logger.error(f"Could not build command_routing_definition.json for workflow folderpath {workflow_folderpath}")

        return routing_definition

    def scan(self, use_cache=True):
        """
        Legacy method for backward compatibility with tests.
        In the consolidated code, scanning is done during build().
        
        Args:
            use_cache: Whether to use cached command directory (ignored, included for API compatibility)
            
        Returns:
            self: Returns self for method chaining
            
        Raises:
            RuntimeError: If the workflow path lacks a _commands folder
        """
        # Check if _commands directory exists
        commands_dir = Path(self.workflow_folderpath) / "_commands"
        if not commands_dir.is_dir():
            raise RuntimeError(f"Workflow path '{self.workflow_folderpath}' does not contain '_commands' directory")
            
        self._build_simple_mappings()
        return self

    def _build_simple_mappings(self):
        """Build the simple mappings from the contexts data."""
        self.command_directory_map = {}
        self.routing_definition_map = {}
        
        # Build command_directory_map: context -> set of commands
        for context, commands in self.contexts.items():
            self.command_directory_map[context] = set(commands)
        
        # Build routing_definition_map: command -> set of contexts
        for context, commands in self.contexts.items():
            for command in commands:
                if command not in self.routing_definition_map:
                    self.routing_definition_map[command] = set()
                self.routing_definition_map[command].add(context)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RoutingRegistry:
    """
    A registry that holds a single, active RoutingDefinition per workflow.
    It builds the definition on-demand the first time it's requested for a workflow.
    """
    _definitions: dict[str, RoutingDefinition] = {}

    @classmethod
    def get_definition(cls, workflow_folderpath: str, load_cached: bool = True) -> RoutingDefinition:
        """
        Gets the routing definition for a workflow.
        If it doesn't exist, it will be built and cached.
        """
        workflow_folderpath = str(Path(workflow_folderpath).resolve())

        if load_cached:
            if workflow_folderpath in cls._definitions:
                return cls._definitions[workflow_folderpath]

            # Attempt to load; if missing, build and save
            try:
                definition = RoutingDefinition.load(workflow_folderpath)
            except Exception:
                definition = RoutingDefinition.build(workflow_folderpath)
                definition.save()
            cls._definitions[workflow_folderpath] = definition
            return definition
        
        # build fresh definition and persist via .save()
        cls._definitions[workflow_folderpath] = RoutingDefinition.build(workflow_folderpath)
        return cls._definitions[workflow_folderpath]

    @classmethod
    def clear_registry(cls):
        """Clears the registry. Useful for testing."""
        cls._definitions.clear()
        
        # Clear the global command class cache to ensure fresh command loading
        RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()
        
        # Also clear the CommandDirectory cache to ensure fresh data on reload
        import fastworkflow.command_directory
        if hasattr(fastworkflow.command_directory.get_cached_command_directory, 'cache_clear'):
            fastworkflow.command_directory.get_cached_command_directory.cache_clear()
        
        # Clear the python_utils module import cache
        import fastworkflow.utils.python_utils
        if hasattr(fastworkflow.utils.python_utils.get_module, 'cache_clear'):
            fastworkflow.utils.python_utils.get_module.cache_clear() 