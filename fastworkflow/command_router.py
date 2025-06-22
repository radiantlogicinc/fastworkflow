from __future__ import annotations

"""Command router that performs a depth-1 scan of a `_commands` directory.

It produces two structures:

• `command_directory`: ``dict[str, set[str]]`` mapping *context name* → set of
  command names available in that context (``'*'`` represents the *global*
  context — commands placed directly under the `_commands/` root).

• `routing_definition`: ``dict[str, set[str]]`` mapping *command name* → set of
  contexts that can handle that command.

The scan is **depth-1**: only files directly inside `_commands/` and one level
of sub-directories are considered. Sub-directories whose name starts with an
underscore are skipped (allowing helpers such as `__pycache__`).
"""

from pathlib import Path
from typing import Dict, Set, Optional

from fastworkflow.command_directory import CommandDirectory, get_cached_command_directory

__all__ = ["CommandRouter"]


class CommandRouter:  # noqa: D101 – simple data container
    def __init__(self, workflow_path: str | Path, commands_root: Optional[str | Path] = None) -> None:
        """
        Initialize a CommandRouter for the given workflow path.
        
        Args:
            workflow_path: Path to the workflow directory
            commands_root: Optional path to the commands directory (defaults to "_commands" under workflow_path)
        """
        self.workflow_path = str(workflow_path)
        self.commands_root = Path(commands_root) if commands_root else Path(workflow_path) / "_commands"
        # Maps context → commands (set)
        self.command_directory: Dict[str, Set[str]] = {"*": set()}
        # Maps command → contexts (set)
        self.routing_definition: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def scan(self, use_cache: bool = True) -> None:
        """
        Populate `command_directory` and `routing_definition` using the CommandDirectory.
        
        Args:
            use_cache: Whether to use the cached CommandDirectory. Set to False for tests.
        """
        self.command_directory = {"*": set()}
        self.routing_definition = {}

        # Get the CommandDirectory, using cache if requested
        if use_cache:
            cmd_dir = get_cached_command_directory(self.workflow_path)
        else:
            cmd_dir = CommandDirectory.load(self.workflow_path)
        
        # Process all qualified command names from CommandDirectory
        for qualified_cmd_name in cmd_dir.map_command_2_metadata.keys():
            if "/" in qualified_cmd_name:
                # Extract context part from qualified name (e.g., "Core" from "Core/abort")
                context_part, cmd_name = qualified_cmd_name.split("/", 1)
                self._add_mapping(context_part, cmd_name)
            else:  # Global command (e.g., "wildcard")
                self._add_mapping("*", qualified_cmd_name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_commands_for_context(self, context_name: str) -> Set[str]:
        """Return the set of commands for *context_name* (empty set if unknown)."""
        return self.command_directory.get(context_name, set())

    def get_contexts_for_command(self, command_name: str) -> Set[str]:
        """Return the set of contexts that can execute *command_name*."""
        return self.routing_definition.get(command_name, set())

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _add_mapping(self, context: str, command: str) -> None:
        self.command_directory.setdefault(context, set()).add(command)
        self.routing_definition.setdefault(command, set()).add(context) 