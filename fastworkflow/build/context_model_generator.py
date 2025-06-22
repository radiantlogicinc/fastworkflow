import os
import json
from typing import Dict, Any, Optional
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.command_file_template import generate_utterances
from fastworkflow.build.inheritance_block_regenerator import InheritanceBlockRegenerator
import ast
from fastworkflow.utils.logging import logger

def generate_context_model(classes: Dict[str, ClassInfo], output_dir: str, file_name: str = "_commands/context_inheritance_model.json") -> Dict[str, Any]:
    """Generate the simplified command context model JSON.

    New schema (v3):
    {
      "ClassA": {"base": ["Base1"]},
      "ClassB": {"base": []}
    }

    • Per-class command lists are no longer included because command files are organised on disk in `_commands/<ClassName>/`.
    • The context model is now a flat structure with context classes at the top level.
    • Each context has a "base" list containing its base classes.
    """
    logger.debug(f"Called generate_context_model with output_dir={output_dir}, file_name={file_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Use InheritanceBlockRegenerator to handle the model update
    model_path = os.path.join(output_dir, file_name)
    regenerator = InheritanceBlockRegenerator(
        commands_root=os.path.join(output_dir, "_commands"),
        model_path=model_path
    )

    return regenerator.regenerate_inheritance(classes) 