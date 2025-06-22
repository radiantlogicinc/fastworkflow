from __future__ import annotations

"""Utility to generate navigator stub files with context expansion logic.

This module provides functionality to create skeleton navigator files for
different contexts, including appropriate context expansion logic based on
inheritance relationships.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple

from fastworkflow.context_model_loader import ContextModelLoader
from fastworkflow.utils.logging import logger
from fastworkflow.utils.context_utils import get_context_names

__all__ = ["NavigatorStubGenerator"]


class NavigatorStubGenerator:
    """Generates navigator stub files with context expansion logic."""

    def __init__(
        self, 
        navigators_root: str | Path = "navigators", 
        model_path: str | Path = "_commands/context_inheritance_model.json"
    ) -> None:
        """Initialize the navigator stub generator.

        Args:
            navigators_root: Path to the navigators directory, defaults to "navigators"
            model_path: Path to the command context model JSON file, defaults to
                "context_inheritance_model.json"
        """
        self.navigators_root = Path(navigators_root)
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
            # Return a minimal default model
            return {}

    def get_navigator_file_path(self, context: str) -> Path:
        """Get the file path for a navigator for a specific context.
        
        Args:
            context: The context name
            
        Returns:
            Path: The file path for the navigator
        """
        return self.navigators_root / f"{context.lower()}_navigator.py"

    def check_file_exists(self, file_path: Path) -> Tuple[bool, str]:
        """Check if a file exists and determine its status.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple[bool, str]: (exists, reason)
                - exists: True if the file exists, False otherwise
                - reason: Description of the file status
        """
        if not file_path.exists():
            return False, "File does not exist"
        
        if not file_path.is_file():
            return True, "Path exists but is not a file"
        
        if file_path.stat().st_size == 0:
            return True, "File exists but is empty"
        
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.strip():
                return True, "File exists but contains only whitespace"
            
            # Check if it's a Python file with actual content
            if content.strip() and file_path.suffix == '.py':
                return True, "File exists with content"
        except Exception as e:
            return True, f"File exists but could not be read: {e}"
        
        return True, "File exists"

    def generate_navigator_stub(self, context: str, force: bool = False) -> Optional[Path]:
        """Generate a navigator stub file for a specific context.
        
        Args:
            context: The context name (should not be '*' for global context)
            force: If True, overwrite existing files
            
        Returns:
            Optional[Path]: Path to the generated file, or None if the file already exists and force is False
        """
        if context == '*':
            logger.debug("Skipping navigator generation for global context '*'")
            return None
            
        # Determine file path
        file_path = self.get_navigator_file_path(context)
        
        # Check if file exists
        exists, reason = self.check_file_exists(file_path)
        
        if exists and not force:
            logger.debug(f"Navigator file already exists: {file_path} - {reason}")
            return None
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get parent contexts
        parent_contexts = self.get_parent_contexts(context)
        
        # Generate stub content
        stub_content = self._generate_stub_content(
            context, 
            parent_contexts["inheritance"], 
        )
        
        # Write stub file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(stub_content)
            logger.debug(f"Generated navigator stub: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error writing navigator stub file {file_path}: {e}")
            return None

    def generate_navigator_stubs(self, force: bool = False) -> Dict[str, Path]:
        """Generate navigator stub files for all contexts in the model.
        
        Args:
            force: If True, overwrite existing files
            
        Returns:
            Dict[str, Path]: Dictionary mapping context names to generated file paths
        """
        model = self.load_context_model()
        contexts = get_context_names(model)
        if '*' in contexts:
            contexts.remove('*')  # Global context doesn't need a navigator
        
        generated_files = {}
        for context in contexts:
            file_path = self.generate_navigator_stub(context, force)
            if file_path:
                generated_files[context] = file_path
        
        return generated_files

    def _generate_stub_content(
        self, 
        context: str, 
        base_contexts: List[str]
    ) -> str:
        """Generate the content for a navigator stub file.
        
        Args:
            context: The context name
            base_contexts: List of base contexts (inheritance)
            
        Returns:
            str: The generated stub content
        """
        # Basic stub template
        stub = f'''"""Navigator for {context} context.

This module provides navigation functionality for {context} objects,
allowing movement between contexts based on inheritance
relationships.
"""

from typing import Optional

from fastworkflow.session import WorkflowSnapshot
from fastworkflow.context import ContextExpander


class {context}Navigator(ContextExpander):
    """Navigator for {context} context.
    
    Implements context delegation for {context} objects.
    """
    
    def move_to_parent_context(self, snapshot: WorkflowSnapshot) -> None:
        """Move from {context} to parent context.
        
        Args:
            snapshot: The workflow snapshot to modify
        """
        current_obj = snapshot.current_context_object
        
        # Ensure we're in the correct context
        if current_obj is None or current_obj.__class__.__name__ != '{context}':
            # Reset to global if context is incorrect
            snapshot.clear_context()
            return
'''
        
        # NOTE: container context support was removed.  Keep an empty list so
        # template variables still resolve without raising NameError.
        container_contexts: list[str] = []

        # Add container context navigation if available (currently disabled)
        if container_contexts:
            for container in container_contexts:
                if container == '*':
                    # Container is global context
                    stub += f'''
        # No need to check for container - reset to global context
        snapshot.clear_context()
        return
'''
                else:
                    # Container is another context
                    stub += f'''
        # Try to navigate to container context: {container}
        try:
            # BEGIN TODO #
            # Get the container object from the current context object
            # Example: container_obj = current_obj.{container.lower()}
            container_obj = None  # Replace with actual container object retrieval
            # END TODO ###
            
            if container_obj is not None:
                snapshot.set_context(container_obj)
                return
        except Exception as e:
            # Log the error but continue to other navigation options
            pass
'''
        
        # Add inheritance-based navigation if available
        if base_contexts:
            for base in base_contexts:
                if base == '*':
                    # Base is global context
                    stub += f'''
        # Base context is global - reset to global context
        snapshot.clear_context()
        return
'''
                else:
                    # Base is another context
                    stub += f'''
        # Try to navigate to base context: {base}
        try:
            # BEGIN TODO #
            # Get the base object from the current context object
            # This might be the same object cast as its base type, or a separate object
            # Example: base_obj = current_obj  # If same object with different type
            # Example: base_obj = current_obj.{base.lower()}  # If separate object
            base_obj = None  # Replace with actual base object retrieval
            # END TODO ###
            
            if base_obj is not None:
                snapshot.set_context(base_obj)
                return
        except Exception as e:
            # Log the error but continue to other navigation options
            pass
'''
        
        # Add fallback to global if no specific navigation was successful
        stub += '''
        # Fallback to global context if no specific navigation was successful
        snapshot.clear_context()
'''
        
        return stub 

    # -----------------------------------------------------------------
    # Helper utilities (public so tests can call them directly)
    # -----------------------------------------------------------------

    def get_parent_contexts(self, context: str) -> dict:
        """Return the list of immediate base contexts for *context*.

        The loader works with the *flat* context-model schema where each
        context maps to a dict that contains a ``base`` list.  We expose a
        stable structure that older tests can also use (a dict with the
        key ``"inheritance"``) so callers only need to look at
        ``result["inheritance"]``.
        """
        model = self.load_context_model()
        bases: list[str] = model.get(context, {}).get("base", [])
        return {"inheritance": bases} 