import os
import json
import ast
from typing import List, Dict, Any, Optional, Tuple

def collect_command_files_and_context_model(
    output_dir: str
) -> Tuple[List[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Collect all command files and load the context model from the output directory.
    
    Args:
        output_dir: Path to the directory containing command files and context model
                   (should be the _commands directory)
    
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
    context_model_path = os.path.join(output_dir, 'context_inheritance_model.json')
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
    For each context, discover commands from the file system instead of relying on the '/' key.
    Inherited commands are indicated in the README.
    
    Args:
        command_metadata: List of command metadata dictionaries
        context_model: Context model dictionary (flat structure with context classes at top level)
        source_dir: Path to the source directory
        
    Returns:
        str: README.md content
    """
    # Group command metadata by context
    commands_by_context = {}
    for meta in command_metadata:
        # Extract context from file path: _commands/ContextName/command.py
        parts = meta['file_path'].split(os.path.sep)
        if len(parts) >= 2:
            # Find the part that comes after "_commands"
            try:
                cmd_idx = parts.index("_commands")
                if cmd_idx + 1 < len(parts):
                    context = parts[cmd_idx + 1]
                    # Skip files directly in _commands (like startup.py)
                    if context.endswith(".py"):
                        continue
                    # Skip special files like _TodoItem.py
                    if os.path.basename(meta['file_path']).startswith("_"):
                        continue
                    if context not in commands_by_context:
                        commands_by_context[context] = []
                    commands_by_context[context].append(meta)
            except ValueError:
                # "_commands" not found in path
                continue

    # Helper function to get inherited commands
    def get_inherited_commands(context, visited=None):
        if visited is None:
            visited = set()
        if context in visited:
            return set()  # Prevent cycles
        visited.add(context)

        # Get base classes from context model - now directly at top level
        base_classes = []
        if context in context_model:
            base_classes = context_model[context].get('base', [])

        # Collect commands from all base classes
        inherited_cmds = set()
        for base in base_classes:
            # Add commands from this base class
            if base in commands_by_context:
                inherited_cmds.update(meta['command_name'] for meta in commands_by_context[base])
            # Add commands inherited by this base class
            inherited_cmds.update(get_inherited_commands(base, visited))

        return inherited_cmds

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

    # Add global commands first if they exist
    global_commands = []
    for meta in command_metadata:
        # Check if it's a file directly in _commands and not a special file
        parts = meta['file_path'].split(os.path.sep)
        if "_commands" in parts:
            cmd_idx = parts.index("_commands")
            if cmd_idx + 1 < len(parts) and parts[cmd_idx + 1].endswith(".py") and (not os.path.basename(meta['file_path']).startswith("_") and \
                               os.path.basename(meta['file_path']) != "startup.py" and \
                               os.path.basename(meta['file_path']) != "__init__.py"):
                global_commands.append(meta)

    if global_commands:
        readme.append("### Global Commands\n")
        for meta in sorted(global_commands, key=lambda m: m['command_name']):
            readme.append(f"- **{meta['command_name']}**")
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
        readme.append("")

    # Process each context from the context model
    for context in sorted(context_model.keys()):
        # Skip any special entries that might still be in the model
        if context == "*":
            continue

        context_name = f"{context} Context"
        readme.append(f"### {context_name}\n")

        # Get own commands
        own_commands = commands_by_context.get(context, [])
        own_command_names = {meta['command_name'] for meta in own_commands}

        # Get inherited commands
        inherited_command_names = get_inherited_commands(context)

        if own_commands or inherited_command_names:
            # List own commands first
            for meta in sorted(own_commands, key=lambda m: m['command_name']):
                readme.append(f"- **{meta['command_name']}**")
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

            # List inherited commands
            for cmd_name in sorted(inherited_command_names - own_command_names):
                # Find metadata for this command
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
                    readme.append("  - (metadata not found)")
        else:
            readme.append("No commands in this context.\n")

        # Show base classes
        base_classes = context_model.get(context, {}).get('base', [])
        if base_classes:
            readme.append(f"  - Base classes: {', '.join(base_classes)}")
        readme.append("")

    # --- Context Model Section ---
    readme.append("## Context Model\n")
    readme.append("The `context_inheritance_model.json` file maps application classes to command contexts, organizing commands by their class.\n")
    readme.append("Structure example:\n")
    readme.append("```json\n{\n  \"ClassA\": {\"base\": [\"BaseClass1\", ...]},\n  \"ClassB\": {\"base\": []}\n}\n```\n")
    readme.append("### Contexts and Commands\n")

    # Process each context again for the context model section
    for context in sorted(context_model.keys()):
        # Skip any special entries that might still be in the model
        if context == "*":
            continue
            
        context_name = f"{context} Context"
        readme.append(f"#### {context_name}")

        # Get own commands
        own_command_names = {meta['command_name'] for meta in commands_by_context.get(context, [])}

        # Get inherited commands
        inherited_command_names = get_inherited_commands(context)

        if own_command_names:
            readme.append("Commands:")
            for cmd in sorted(own_command_names):
                readme.append(f"- `{cmd}`")

        if inherited_command_names:
            readme.append("Inherited commands:")
            for cmd in sorted(inherited_command_names - own_command_names):
                readme.append(f"- `{cmd}`")

        if not own_command_names and not inherited_command_names:
            readme.append("No commands in this context.")

        # Show base classes
        base_classes = context_model.get(context, {}).get('base', [])
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
    
    Args:
        output_dir: Directory to write README.md to (should be the _commands directory)
        content: Content to write to README.md
        
    Returns:
        bool: True on success, False on error
    """
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing README.md: {e}")
        return False 