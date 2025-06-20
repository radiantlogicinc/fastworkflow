from __future__ import annotations

"""Utility to regenerate only the inheritance block in the command context model.

This module provides functionality to update the inheritance relationships in
the command_context_model.json file while preserving the aggregation block,
which is maintained manually by developers.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Set, Optional, List

from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.utils.logging import logger

__all__ = ["InheritanceBlockRegenerator"]


class InheritanceBlockRegenerator:
    """Regenerates only the inheritance block in the command context model."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "command_context_model.json"
    ) -> None:
        """Initialize the inheritance block regenerator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "command_context_model.json"
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
        self._model_data = {"inheritance": {"*": {"base": []}}, "aggregation": {}}
        return self._model_data
    
    def regenerate_inheritance(self, classes: Optional[Dict[str, ClassInfo]] = None) -> Dict[str, Any]:
        """Regenerate the inheritance block based on directory structure and class information.
        
        Args:
            classes: Optional dictionary mapping class names to ClassInfo objects from AST analysis.
                    If provided, inheritance relationships will be based on this data.
        
        Returns:
            Dict[str, Any]: The updated context model with regenerated inheritance block
        """
        # Scan directory structure to identify contexts
        contexts = self.scan_contexts()
        
        # Build inheritance map
        inheritance: Dict[str, Dict[str, List[str]]] = {}
        
        if classes:
            # Use class information to build inheritance relationships
            all_class_names = set(classes.keys())
            
            for class_name, class_info in classes.items():
                # Only include base classes that are also in the analyzed set
                base_contexts = [b for b in class_info.bases if b in all_class_names]
                
                # Only include classes that have base classes to keep the file small
                if base_contexts:
                    inheritance[class_name] = {"base": base_contexts}
        else:
            # Without class info, we can't determine inheritance relationships,
            # so we'll just create empty entries for all contexts found in the directory
            inheritance = {context: {"base": []} for context in contexts}
        
        # Always include global context
        inheritance['*'] = {"base": []}
        
        # Load existing model to preserve aggregation
        existing_model = self.load_existing_model()
        
        # Create new model with updated inheritance and preserved aggregation
        new_model = {
            "inheritance": inheritance,
            "aggregation": existing_model.get("aggregation", {})
        }
        
        # Write updated model back to file
        self.write_model(new_model)
        
        return new_model
    
    def write_model(self, model: Dict[str, Any]) -> None:
        """Write the updated model to file.
        
        Args:
            model: The model data to write
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