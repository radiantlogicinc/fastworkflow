from __future__ import annotations

"""Utility to generate command stub files with ContextExpander implementations.

This module provides functionality to create skeleton command files for both
global and context-specific commands, and generates separate '_fastworkflow_handlers.py'
files containing ContextExpander implementations for contexts with container relationships.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

from fastworkflow.context_model_loader import ContextModelLoader
from fastworkflow.utils.logging import logger
from fastworkflow.utils.context_utils import get_context_names

__all__ = ["CommandStubGenerator"]


class CommandStubGenerator:
    """Generates command stub files and ContextExpander implementations."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "_commands/context_inheritance_model.json"
    ) -> None:
        """Initialize the command stub generator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "context_inheritance_model.json"
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
            # Return a minimal default model
            return {}

    def get_command_file_path(self, context: str, command_name: str) -> Path:
        """Get the file path for a command in a specific context.
        
        Args:
            context: The context name (use '*' for global context)
            command_name: The name of the command
            
        Returns:
            Path: The file path for the command
        """
        if context == '*':
            return self.commands_root / f"{command_name}.py"
        else:
            return self.commands_root / context / f"{command_name}.py"

    def get_handlers_file_path(self, context: str) -> Path:
        """Get the file path for the _fastworkflow_handlers.py file in a specific context.
        
        Args:
            context: The context name (should not be '*' for global context)
            
        Returns:
            Path: The file path for the handlers file
        """
        if context == '*':
            # Global context doesn't have a handlers file
            return self.commands_root / "_fastworkflow_handlers.py"
        else:
            return self.commands_root / context / "_fastworkflow_handlers.py"

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

    def generate_command_stub(self, context: str, command_name: str, force: bool = False) -> Optional[Path]:
        """Generate a command stub file for a specific context and command.
        
        Args:
            context: The context name (use '*' for global context)
            command_name: The name of the command
            force: If True, overwrite existing files
            
        Returns:
            Optional[Path]: Path to the generated file, or None if the file already exists and force is False
        """
        # Determine file path
        file_path = self.get_command_file_path(context, command_name)
        
        # Check if file exists
        exists, reason = self.check_file_exists(file_path)
        
        if exists and not force:
            logger.debug(f"Command file already exists: {file_path} - {reason}")
            return None
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate stub content
        stub_content = self._generate_command_stub_content(context, command_name)
        
        # Write stub file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(stub_content)
            logger.debug(f"Generated command stub: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error writing command stub file {file_path}: {e}")
            return None

    def generate_command_stubs_for_context(self, context: str, command_names: List[str], force: bool = False) -> List[Path]:
        """Generate command stub files for multiple commands in a context.
        
        Args:
            context: The context name (use '*' for global context)
            command_names: List of command names to generate
            force: If True, overwrite existing files
            
        Returns:
            List[Path]: List of paths to the generated files
        """
        generated_files = []

        # Generate command stubs
        for command_name in command_names:
            if file_path := self.generate_command_stub(
                context, command_name, force
            ):
                generated_files.append(file_path)

        if handlers_path := self.generate_handlers_file(context, force):
            generated_files.append(handlers_path)

        return generated_files

    def generate_command_stubs_for_all_contexts(self, command_names: Dict[str, List[str]], force: bool = False) -> Dict[str, List[Path]]:
        """Generate command stub files for multiple commands in multiple contexts.
        
        Args:
            command_names: Dictionary mapping context names to lists of command names
            force: If True, overwrite existing files
            
        Returns:
            Dict[str, List[Path]]: Dictionary mapping context names to lists of generated file paths
        """
        generated_files = {}
        
        for context, commands in command_names.items():
            context_files = self.generate_command_stubs_for_context(context, commands, force)
            if context_files:
                generated_files[context] = context_files
        
        return generated_files

    def generate_all_handlers_files(self, force: bool = False) -> Dict[str, Path]:
        """Generate _fastworkflow_handlers.py files for all contexts that have container contexts.
        
        Args:
            force: If True, overwrite existing files
            
        Returns:
            Dict[str, Path]: Dictionary mapping context names to generated file paths
        """
        generated_files = {}
        
        # Get all contexts with container contexts
        contexts = self.get_contexts_with_containers()
        
        # Generate handlers files
        for context in contexts:
            if context != '*':  # Skip global context
                file_path = self.generate_handlers_file(context, force)
                if file_path:
                    generated_files[context] = file_path
        
        return generated_files

    def _generate_command_stub_content(self, context: str, command_name: str) -> str:
        """Generate the content for a command stub file.
        
        Args:
            context: The context name
            command_name: The name of the command
            
        Returns:
            str: The generated stub content
        """
        # Format command name for display (capitalize first letter)
        display_command = command_name.replace('_', ' ').capitalize()
        
        # Basic stub template - using regular string with .format() to avoid f-string issues
        stub = """\"\"\"Command to {display_command_lower} in the {context} context.\"\"\"

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict

from fastworkflow.session import WorkflowSnapshot


class Signature:
    \"\"\"Execute {display_command} command in {context} context.\"\"\"
    
    class Input(BaseModel):
        \"\"\"Parameters for {display_command}.\"\"\"
        # TODO: Define command parameters here
        
        model_config = ConfigDict(extra="forbid")
    
    class Output(BaseModel):
        \"\"\"Result of {display_command}.\"\"\"
        # TODO: Define command output here
        result: str
        
    # Utterance examples for intent detection
    plain_utterances = [
        # TODO: Add example utterances
        "{command_name}",
    ]
    
    template_utterances = []
    
    @staticmethod
    def generate_utterances():
        \"\"\"Generate additional utterances dynamically.\"\"\"
        return []
    
    @staticmethod
    def process_extracted_parameters(snapshot: WorkflowSnapshot, parameters: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Process extracted parameters before command execution.\"\"\"
        # TODO: Add parameter processing logic if needed
        return parameters


class ResponseGenerator:
    \"\"\"Generates response for the command.\"\"\"
    
    def __call__(self, session, command, input_obj=None):
        \"\"\"Execute the command and generate a response.
        
        Args:
            session: The workflow session
            command: The original command text
            input_obj: The parsed input parameters
            
        Returns:
            The command output
        \"\"\"
        from fastworkflow import CommandOutput, CommandResponse
        
        # TODO: Implement command logic
        result = "Executed {command_name} command"
        
        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response=result,
                    artifacts={{"result": result}}
                )
            ]
        )
""".format(
            display_command_lower=display_command.lower(),
            display_command=display_command,
            context=context,
            command_name=command_name
        )
        
        return stub

    def _generate_handlers_stub_content(self, context: str, container_contexts: List[str]) -> str:
        """Generate the content for a _fastworkflow_handlers.py file.
        
        Args:
            context: The context name
            container_contexts: List of container context names
            
        Returns:
            str: The generated stub content
        """
        # Start with the common header - using regular string with .format() to avoid f-string issues
        stub = """\"\"\"Handlers for {context} context.

This file contains ContextExpander implementation for {context} context.
\"\"\"

from fastworkflow.session import WorkflowSnapshot
from fastworkflow.context import ContextExpander


class ContextExpander(ContextExpander):
    \"\"\"Implements context delegation for {context} context.\"\"\"
    
    def move_to_parent_context(self, snapshot: WorkflowSnapshot):
        \"\"\"Move from {context} to parent context.
        
        Args:
            snapshot: The workflow snapshot to modify
        \"\"\"
        # BEGIN TODO #
""".format(context=context)
        
        # Add container context navigation
        if container_contexts:
            if len(container_contexts) == 1:
                # Single container
                container = container_contexts[0]
                if container == '*':
                    # Container is global context
                    stub += """        # Current context object's parent is the global context
        snapshot.clear_context()  # Reset to global context
"""
                else:
                    # Container is another context
                    stub += """        # Get parent object of type '{container}' from current context object
        # TODO: Replace this line with actual parent object retrieval code
        # parent_obj = snapshot.current_context_object.parent  # Example
        parent_obj = None  # Placeholder - replace with actual parent retrieval
        snapshot.set_context(parent_obj)
""".format(container=container)
            else:
                # Multiple containers - provide options
                stub += """        # Multiple possible container contexts detected. Choose the appropriate one:
"""
                for container in container_contexts:
                    if container == '*':
                        stub += """        # Option: Reset to global context
        # snapshot.clear_context()
"""
                    else:
                        stub += """        # Option: Get parent object of type '{container}'
        # TODO: Replace this with actual parent object retrieval code
        # parent_obj = snapshot.current_context_object.parent  # Example
        # snapshot.set_context(parent_obj)
""".format(container=container)
                stub += """        # For now, default to resetting to global context
        snapshot.clear_context()
"""
        else:
            # No container contexts (this shouldn't happen based on our checks)
            stub += """        # No container contexts defined - reset to global context
        snapshot.clear_context()
"""
        
        stub += """        # END TODO ###
"""
        
        return stub 