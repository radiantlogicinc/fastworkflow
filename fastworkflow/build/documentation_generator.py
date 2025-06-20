import os
import json
import ast
from typing import List, Dict, Any, Optional, Tuple

def collect_command_files_and_context_model(
    output_dir: str
) -> Tuple[List[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Collect all command files and load the context model from the output directory.
    Returns:
        - List of command file paths (excluding __init__.py)
        - Parsed context model dict (or None if error)
        - Error message (or None if no error)
    """
    command_files = []
    error = None
    # List all .py files (excluding __init__.py)
    for root, _, files in os.walk(output_dir):
        command_files.extend(
            os.path.join(root, file)
            for file in files
            if file.endswith('.py') and file != '__init__.py'
        )
    # Load context model
    context_model_path = os.path.join(output_dir, 'command_context_model.json')
    context_model = None
    if not os.path.exists(context_model_path):
        error = f"Context model file not found at {context_model_path}"
    else:
        try:
            with open(context_model_path, 'r', encoding='utf-8') as f:
                context_model = json.load(f)
        except json.JSONDecodeError as e:
            error = f"Invalid JSON in context model: {str(e)}"
        except Exception as e:
            error = f"Error reading context model: {str(e)}"
    return command_files, context_model, error 

def extract_command_metadata(command_files: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts metadata from each command file:
      - command_name
      - file_path
      - plain_utterances (list of strings)
      - input_model (optional)
      - output_model (optional)
      - docstring (optional)
      - errors (list of error messages)
    Returns a list of metadata dicts.
    """
    metadata_list = []

    for file_path in command_files:
        meta = {
            "command_name": None,
            "file_path": file_path,
            "plain_utterances": [],
            "input_model": None,
            "output_model": None,
            "docstring": None,
            "errors": []
        }
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            meta["command_name"] = (
                file_path.split("/")[-1].replace(".py", "")
            )

            # Module docstring
            meta["docstring"] = ast.get_docstring(tree)

            # Find plain_utterances assignment
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "plain_utterances"
                        ):
                            if isinstance(node.value, ast.List):
                                values = []
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Str):
                                        values.append(elt.s)
                                    elif hasattr(ast, "Constant") and isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        values.append(elt.value)
                                meta["plain_utterances"] = values
                            else:
                                meta["errors"].append("plain_utterances is not a list")
                # Find Input/Output model classes
                if isinstance(node, ast.ClassDef):
                    if node.name == "Input":
                        meta["input_model"] = "Input"
                    elif node.name == "Output":
                        meta["output_model"] = "Output"
                    # Optionally, check for subclasses of BaseModel
                    for base in node.bases:
                        if (
                            isinstance(base, ast.Name)
                            and base.id == "BaseModel"
                        ):
                            if "Input" in node.name and not meta["input_model"]:
                                meta["input_model"] = node.name
                            elif "Output" in node.name and not meta["output_model"]:
                                meta["output_model"] = node.name

        except Exception as e:
            meta["errors"].append(f"Failed to parse file: {e}")

        metadata_list.append(meta)

    return metadata_list 

def generate_readme_content(
    command_metadata: list,
    context_model: dict,
    source_dir: str
) -> str:
    """
    Build the README.md content as a single string.
    For each context, include commands from its own '/' key and recursively from all base classes.
    Inherited commands are indicated in the README.
    """
    def get_all_commands(context, visited=None):
        if visited is None:
            visited = set()
        if context in visited:
            return set(), set()  # Prevent cycles
        visited.add(context)
        data = context_model.get(context, {})
        own_cmds = set(data.get('/', []) or [])
        inherited_cmds = set()
        for base in data.get('base', []) or []:
            base_own, base_inherited = get_all_commands(base, visited)
            inherited_cmds.update(base_own)
            inherited_cmds.update(base_inherited)
        return own_cmds, inherited_cmds

    # --- Overview Section ---
    readme = [
        "# FastWorkflow Commands",
        "",
        f"This directory contains FastWorkflow command files generated from the Python application in `{source_dir}`.",
        "",
        "## Overview",
        "",
        "The generated code includes:",
        "- Command files for all public methods and properties in the application",
        "- A command context model mapping classes to commands",
        "",
        "## Usage",
        "",
        "These commands can be used with FastWorkflow's orchestration and agent frameworks to enable chat-based and programmatic interaction with the application.",
        "",
    ]

    # --- Available Commands Section ---
    readme.append("## Available Commands\n")
    for context, data in context_model.items():
        context_name = "Global Commands" if context == "*" else f"{context} Context"
        readme.append(f"### {context_name}\n")
        own_cmds, inherited_cmds = get_all_commands(context)
        if own_cmds or inherited_cmds:
            # List own commands first
            for cmd_name in sorted(own_cmds):
                meta = next((m for m in command_metadata if m['command_name'] == cmd_name), None)
                readme.append(f"- **{cmd_name}**")
                if meta:
                    if meta['plain_utterances']:
                        readme.append("  - Example utterances:")
                        for utt in meta['plain_utterances']:
                            readme.append(f"    - `{utt}`")
                    if meta['input_model']:
                        readme.append(f"  - Input model: `{meta['input_model']}`")
                    if meta['output_model']:
                        readme.append(f"  - Output model: `{meta['output_model']}`")
                    if meta['docstring']:
                        readme.append(f"  - Description: {meta['docstring']}")
                    if meta['errors']:
                        readme.append(f"  - [!] Metadata extraction errors: {meta['errors']}")
                else:
                    readme.append(f"  - (metadata not found)")
            # List inherited commands
            for cmd_name in sorted(inherited_cmds - own_cmds):
                meta = next((m for m in command_metadata if m['command_name'] == cmd_name), None)
                readme.append(f"- **{cmd_name}** (inherited)")
                if meta:
                    if meta['plain_utterances']:
                        readme.append("  - Example utterances:")
                        for utt in meta['plain_utterances']:
                            readme.append(f"    - `{utt}`")
                    if meta['input_model']:
                        readme.append(f"  - Input model: `{meta['input_model']}`")
                    if meta['output_model']:
                        readme.append(f"  - Output model: `{meta['output_model']}`")
                    if meta['docstring']:
                        readme.append(f"  - Description: {meta['docstring']}")
                    if meta['errors']:
                        readme.append(f"  - [!] Metadata extraction errors: {meta['errors']}")
                else:
                    readme.append(f"  - (metadata not found)")
        else:
            readme.append("No commands in this context.\n")
        base_classes = data.get('base')
        if base_classes:
            readme.append(f"  - Base classes: {', '.join(base_classes)}")
        readme.append("")

    # --- Context Model Section ---
    readme.append("## Context Model\n")
    readme.append("The `command_context_model.json` file maps application classes to command contexts, organizing commands by their class.\n")
    readme.append("Structure example:\n")
    readme.append("```json\n{\n  \"context_name\": {\n    \"/\": [\"command1\", \"command2\", ...],\n    \"base\": [\"BaseClass1\", ...]\n  },\n  ...\n}\n```\n")
    readme.append("### Contexts and Commands\n")
    for context, data in context_model.items():
        context_name = "Global Context (*)" if context == "*" else f"{context} Context"
        readme.append(f"#### {context_name}")
        own_cmds, inherited_cmds = get_all_commands(context)
        if own_cmds:
            readme.append("Commands:")
            for cmd in sorted(own_cmds):
                readme.append(f"- `{cmd}`")
        if inherited_cmds:
            readme.append("Inherited commands:")
            for cmd in sorted(inherited_cmds - own_cmds):
                readme.append(f"- `{cmd}`")
        if not own_cmds and not inherited_cmds:
            readme.append("No commands in this context.")
        base_classes = data.get('base')
        if base_classes:
            readme.append(f"Base classes: {', '.join(base_classes)}")
        readme.append("")

    # --- Extending and Testing Section ---
    readme.append("## Extending\n")
    readme.append("To add new commands:\n")
    readme.append("1. Add new public methods or properties to your application classes\n2. Run the FastWorkflow build tool again to regenerate the command files and documentation\n")
    readme.append("## Testing\n")
    readme.append("You can test these commands using FastWorkflow's MCP server and agent interfaces.\n")

    # Join all lines into a single string
    return "\n".join(line for line in readme if line is not None)

def write_readme_file(output_dir: str, content: str) -> bool:
    """
    Write the given content to README.md in the output directory.
    Overwrites any existing README.md. Returns True on success, False on error.
    """
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing README.md: {e}")
        return False 