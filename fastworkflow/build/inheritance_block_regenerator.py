from __future__ import annotations

"""Utility to regenerate only the inheritance block in the command context model.

This module provides functionality to update the inheritance relationships in
the context_inheritance_model.json file.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Set, Optional, List

from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.utils.logging import logger
from fastworkflow.utils.context_utils import get_context_names

__all__ = ["InheritanceBlockRegenerator"]


class InheritanceBlockRegenerator:
    """Regenerates only the inheritance block in the command context model."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "_commands/context_inheritance_model.json"
    ) -> None:
        """Initialize the inheritance block regenerator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "context_inheritance_model.json"
        """
        self.commands_root = Path(commands_root)
        self.model_path = Path(model_path)
        self._model_data: Optional[Dict[str, Any]] = None

    def scan_contexts(self) -> Set[str]:
        """Scan the commands directory to identify contexts.
        
        Returns:
            Set[str]: Set of context names found in the directory structure
        """
        contexts = set()
        
        if not self.commands_root.exists():
            logger.warning(f"Commands root directory {self.commands_root} does not exist")
            return contexts
        
        try:
            for item in self.commands_root.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    contexts.add(item.name)
                    logger.debug(f"Found context: {item.name}")
        except PermissionError:
            logger.error(f"Permission denied when scanning {self.commands_root}")
        except Exception as e:
            logger.error(f"Error scanning contexts: {e}")
        
        return contexts
        
    def load_existing_model(self) -> Dict[str, Any]:
        """Load the existing context model from file.
        
        Returns:
            Dict[str, Any]: The loaded context model, or a default model if the file
                doesn't exist or is invalid
        """
        if self._model_data is not None:
            return self._model_data  # Return cached model if available
        
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    self._model_data = json.load(f)
                    logger.debug(f"Loaded existing model from {self.model_path}")
                    return self._model_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model file {self.model_path}: {e}")
        except PermissionError:
            logger.error(f"Permission denied when reading {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model file: {e}")
        
        # Return default model if file doesn't exist or is invalid
        self._model_data = {}
        return self._model_data
    
    def regenerate_inheritance(self, classes: Optional[Dict[str, ClassInfo]] = None) -> Dict[str, Any]:
        """Regenerate the inheritance block based on directory structure and class information.
        
        Args:
            classes: Optional dictionary mapping class names to ClassInfo objects from AST analysis.
                    If provided, inheritance relationships will be based on this data.
        
        Returns:
            Dict[str, Any]: The updated context model with regenerated inheritance relationships
                           in a flat structure (no "inheritance" wrapper key)
        """
        # Scan directory structure to identify contexts
        contexts = self.scan_contexts()
        
        # Build inheritance map - now directly as the top-level structure
        inheritance_map: Dict[str, Dict[str, List[str]]] = {}
        
        if classes:
            # Use class information to build inheritance relationships
            all_class_names = set(classes.keys())
            
            # Include all classes in the inheritance block, not just those with base classes
            for class_name, class_info in classes.items():
                # Only include base classes that are also in the analyzed set
                base_contexts = [b for b in class_info.bases if b in all_class_names]
                inheritance_map[class_name] = {"base": base_contexts}
        else:
            # Without class info, we can't determine inheritance relationships,
            # so we'll just create empty entries for all contexts found in the directory
            inheritance_map = {context: {"base": []} for context in contexts}
        
        # Load existing model to check for any existing entries we should preserve
        existing_model = self.load_existing_model()
        
        # Preserve any existing entries not derived from class analysis
        for context, data in existing_model.items():
            if context not in inheritance_map:
                inheritance_map[context] = data
        
        # Write updated model back to file
        self.write_model(inheritance_map)
        
        return inheritance_map
    
    def write_model(self, model: Dict[str, Any]) -> None:
        """Write the updated model to file.
        
        Args:
            model: The model data to write (flat structure, no inheritance wrappers)
        """
        try:
            # Ensure parent directories exist
            self.model_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2)
                logger.debug(f"Updated model written to {self.model_path}")
        except PermissionError:
            logger.error(f"Permission denied when writing to {self.model_path}")
        except Exception as e:
            logger.error(f"Error writing model file: {e}") 