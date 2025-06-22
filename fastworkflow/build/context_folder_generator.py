from __future__ import annotations

"""Utility to generate context folders based on the command context model.

This module provides functionality to create the directory structure required
for context-aware commands, based on the inheritance relationships defined
in the context_inheritance_model.json file.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from fastworkflow.context_model_loader import ContextModelLoader
from fastworkflow.utils.logging import logger
from fastworkflow.utils.context_utils import get_context_names

__all__ = ["ContextFolderGenerator"]


class ContextFolderGenerator:
    """Generates context folders based on the command context model."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "_commands/context_inheritance_model.json"
    ) -> None:
        """Initialize the context folder generator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "_commands/context_inheritance_model.json"
        """
        self.commands_root = Path(commands_root)
        self.model_path = Path(model_path)
        self._model_data: Optional[Dict[str, Any]] = None

    def load_context_model(self) -> Dict[str, Any]:
        """Load the context model using the ContextModelLoader.

        Returns:
            Dict[str, Any]: The parsed context model
        
        Raises:
            Exception: If the context model cannot be loaded
        """
        if self._model_data is not None:
            return self._model_data  # Return cached model if available

        try:
            loader = ContextModelLoader(self.model_path)
            self._model_data = loader.load()
            return self._model_data
        except Exception as e:
            logger.error(f"Error loading context model: {e}")
            raise

    def generate_folders(self) -> Dict[str, Path]:
        """Generate context folders based on the model.
        
        Creates a folder for each context in the model,
        except for the global "*" context. Also creates a _<ContextName>.py file
        in each context folder with a Context class and get_parent method.
        
        Returns:
            Dict[str, Path]: Mapping of context names to their folder paths
        """
        # Load context model
        context_model = self.load_context_model()

        # Ensure root commands directory exists
        self.commands_root.mkdir(exist_ok=True, parents=True)

        # Create context folders
        contexts = get_context_names(context_model)
        if '*' in contexts:
            contexts.remove('*')  # Global context doesn't need a folder

        created_folders = {}

        for context in contexts:
            context_dir = self.commands_root / context
            context_dir.mkdir(exist_ok=True)
            logger.debug(f"Created context folder: {context_dir}")
            created_folders[context] = context_dir

            # Create _<ContextName>.py file if it doesn't exist
            handler_file = context_dir / f"_{context}.py"
            if not handler_file.exists():
                # Determine parent type based on inheritance model
                parent_type = "None"
                parent_import = ""
                
                # Get base classes for this context
                base_classes = context_model.get(context, {}).get('base', [])
                if base_classes:
                    # Use first base class as parent type
                    parent_type = base_classes[0]
                    parent_import = f"from ...application.{parent_type.lower()} import {parent_type}"
                
                # Create the handler file content
                handler_content = f"""from typing import Optional
from ...application.{context.lower()} import {context}
{parent_import}

class Context:
    @classmethod
    def get_parent(cls, command_context_object: {context}) -> Optional[{parent_type}]:
        return getattr(command_context_object, 'parent', None)
"""

                # Write the file
                handler_file.write_text(handler_content)
                logger.debug(f"Created context handler file: {handler_file}")

        return created_folders 