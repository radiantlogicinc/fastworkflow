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
from typing import Dict, Set

__all__ = ["CommandRouter"]


class CommandRouter:  # noqa: D101 – simple data container
    def __init__(self, commands_root: str | Path = "_commands") -> None:
        self.commands_root = Path(commands_root)
        # Maps context → commands (set)
        self.command_directory: Dict[str, Set[str]] = {"*": set()}
        # Maps command → contexts (set)
        self.routing_definition: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Populate `command_directory` and `routing_definition`."""
        self.command_directory = {"*": set()}
        self.routing_definition = {}

        if not self.commands_root.exists():
            # Nothing to scan – keep only global context entry
            return

        for item in self.commands_root.iterdir():
            if item.is_file() and item.suffix == ".py":
                # Global command
                cmd_name = item.stem
                self._add_mapping("*", cmd_name)
            elif item.is_dir() and not item.name.startswith("_"):
                context_name = item.name
                self.command_directory.setdefault(context_name, set())
                # Depth-1 scan: look only at files directly inside the folder
                for cmd_file in item.iterdir():
                    if cmd_file.is_file() and cmd_file.suffix == ".py":
                        cmd_name = cmd_file.stem
                        self._add_mapping(context_name, cmd_name)

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