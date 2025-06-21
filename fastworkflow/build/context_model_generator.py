import os
import json
from typing import Dict, Any, Optional
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.command_file_template import generate_utterances
from fastworkflow.build.inheritance_block_regenerator import InheritanceBlockRegenerator
import ast
from fastworkflow.utils.logging import logger

def generate_context_model(classes: Dict[str, ClassInfo], output_dir: str, file_name: str = "_commands/context_inheritance_model.json", aggregation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate the simplified command context model JSON.

    New schema (v2):
    {
      "inheritance": {
        "ClassA": {"base": ["Base1"]},
        "*": {"base": []}
      },
      "aggregation": { ... }  # OPTIONAL – only if deterministically derived
    }

    • Per-class command lists are no longer included because command files are organised on disk in `_commands/<ClassName>/`.
    • A special "*" entry is kept inside the inheritance map to preserve prior behaviour.
    • The aggregation block is preserved if it exists in the previous model.
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