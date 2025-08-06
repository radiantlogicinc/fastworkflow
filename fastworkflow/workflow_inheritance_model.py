"""
Module for handling workflow inheritance model configuration.

This module provides functionality to parse and validate workflow_inheritance_model.json files
that declare which base workflows a workflow extends.
"""

import json
import os
import importlib
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class WorkflowInheritanceModelError(Exception):
    """Base class for errors in this module."""
    pass


class WorkflowInheritanceModelValidationError(WorkflowInheritanceModelError):
    """Raised when the workflow inheritance model fails validation."""
    pass


class WorkflowInheritanceModel(BaseModel):
    """
    Model for workflow_inheritance_model.json configuration.
    
    Defines which base workflows this workflow inherits from.
    Base workflows are processed in order, with later entries having higher precedence.
    The local workflow has the highest precedence of all.
    """
    
    base: List[str] = Field(
        default_factory=list,
        description="List of base workflows to inherit from, in order of precedence"
    )
    
    @field_validator("base")
    @classmethod
    def validate_base_entries(cls, v: List[str]) -> List[str]:
        """Validate that base entries are non-empty strings."""
        if not isinstance(v, list):
            raise ValueError("base must be a list")
        
        for i, entry in enumerate(v):
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"base[{i}] must be a non-empty string")
        
        return v
    
    @classmethod
    def load(cls, workflow_folderpath: str) -> "WorkflowInheritanceModel":
        """
        Load workflow inheritance model from workflow_inheritance_model.json.
        
        Args:
            workflow_folderpath: Path to the workflow folder
            
        Returns:
            WorkflowInheritanceModel instance (empty if file doesn't exist)
            
        Raises:
            WorkflowInheritanceModelValidationError: If the file exists but is invalid
        """
        inheritance_file = Path(workflow_folderpath) / "workflow_inheritance_model.json"
        
        if not inheritance_file.exists():
            # Return empty model if file doesn't exist (no inheritance)
            return cls()
        
        try:
            with open(inheritance_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return cls.model_validate(data)
            
        except json.JSONDecodeError as e:
            raise WorkflowInheritanceModelValidationError(
                f"Invalid JSON in workflow_inheritance_model.json: {e}"
            ) from e
        except Exception as e:
            raise WorkflowInheritanceModelValidationError(
                f"Error loading workflow_inheritance_model.json: {e}"
            ) from e
    
    def resolve_base_paths(self, workflow_folderpath: str) -> List[str]:
        """
        Resolve base workflow entries to actual filesystem paths.
        
        Args:
            workflow_folderpath: Path to the current workflow folder
            
        Returns:
            List of resolved paths to base workflow folders
            
        Raises:
            WorkflowInheritanceModelValidationError: If a base workflow cannot be resolved
        """
        resolved_paths = []
        
        for base_entry in self.base:
            try:
                resolved_path = self._resolve_single_base(base_entry, workflow_folderpath)
                resolved_paths.append(resolved_path)
            except Exception as e:
                raise WorkflowInheritanceModelValidationError(
                    f"Cannot resolve base workflow '{base_entry}': {e}"
                ) from e
        
        return resolved_paths
    
    def _resolve_single_base(self, base_entry: str, workflow_folderpath: str) -> str:
        """
        Resolve a single base workflow entry to a filesystem path.
        
        Args:
            base_entry: The base workflow entry (package path or file path)
            workflow_folderpath: Path to the current workflow folder
            
        Returns:
            Resolved path to the base workflow folder
            
        Raises:
            WorkflowInheritanceModelValidationError: If the base workflow cannot be resolved
        """
        base_entry = base_entry.strip()
        
        # Check if it's a filesystem path (relative or absolute)
        if base_entry.startswith('./') or base_entry.startswith('../') or os.path.isabs(base_entry):
            return self._resolve_filesystem_path(base_entry, workflow_folderpath)
        
        # Otherwise, treat it as a Python package import path
        return self._resolve_package_path(base_entry)
    
    def _resolve_filesystem_path(self, path: str, workflow_folderpath: str) -> str:
        """
        Resolve a filesystem path (relative or absolute) to an absolute path.
        
        Args:
            path: The filesystem path
            workflow_folderpath: Path to the current workflow folder (for relative resolution)
            
        Returns:
            Absolute path to the base workflow folder
            
        Raises:
            WorkflowInheritanceModelValidationError: If the path doesn't exist or lacks _commands
        """
        if os.path.isabs(path):
            resolved_path = path
        else:
            # Resolve relative to the current workflow folder
            resolved_path = os.path.abspath(os.path.join(workflow_folderpath, path))
        
        if not os.path.isdir(resolved_path):
            raise WorkflowInheritanceModelValidationError(
                f"Filesystem path does not exist: {resolved_path}"
            )
        
        commands_dir = os.path.join(resolved_path, "_commands")
        if not os.path.isdir(commands_dir):
            raise WorkflowInheritanceModelValidationError(
                f"Base workflow at {resolved_path} does not contain a _commands directory"
            )
        
        return resolved_path
    
    def _resolve_package_path(self, package_path: str) -> str:
        """
        Resolve a Python package import path to a filesystem path.
        
        Args:
            package_path: The Python package path (e.g., 'fastworkflow.examples.simple_workflow_template')
            
        Returns:
            Absolute path to the base workflow folder
            
        Raises:
            WorkflowInheritanceModelValidationError: If the package cannot be imported or lacks _commands
        """
        try:
            module = importlib.import_module(package_path)
            
            # Handle both regular packages and namespace packages
            if hasattr(module, '__file__') and module.__file__ is not None:
                # Regular package with __init__.py
                package_dir = os.path.dirname(module.__file__)
            elif hasattr(module, '__path__') and module.__path__:
                # Namespace package - use the first path in __path__
                package_dir = next(iter(module.__path__))
            else:
                raise WorkflowInheritanceModelValidationError(
                    f"Package {package_path} has no accessible path and cannot be used as a base workflow"
                )
            
            # Check if it contains _commands directory
            commands_dir = os.path.join(package_dir, "_commands")
            if not os.path.isdir(commands_dir):
                raise WorkflowInheritanceModelValidationError(
                    f"Package {package_path} does not contain a _commands directory"
                )
            
            return package_dir
            
        except ImportError as e:
            raise WorkflowInheritanceModelValidationError(
                f"Cannot import package {package_path}: {e}"
            ) from e
