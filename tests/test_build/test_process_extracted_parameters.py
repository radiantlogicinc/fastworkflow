import os
import tempfile
import ast
from pathlib import Path

import pytest

from fastworkflow.build.command_file_template import create_command_file
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def todo_item_class():
    """Create a TodoItem class with methods for testing."""
    class_info = ClassInfo('TodoItem', 'application/todo_item.py')
    
    # Add a method that will be used for testing
    class_info.methods.append(MethodInfo(
        name="complete",
        parameters=[],
        docstring="Mark the todo item as complete.",
        return_annotation="bool"
    ))
    
    return class_info


def test_validate_extracted_parameters_is_absent_unless_the_command_writes_one(temp_dir, todo_item_class):
    """The generator emits no trivial validate_extracted_parameters stub (fix-3rsq.one-hook): a command
    with no check has no hook, and the framework accepts it. The hook, when a command writes one, is a
    @staticmethod taking (workflow, command, input) and returning None (fix-3rsq.one-hook)."""
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=todo_item_class.methods[0],
        output_dir=output_dir,
        file_name="complete.py",
        source_dir="."
    )

    with open(file_path, 'r') as f:
        file_content = f.read()
    tree = ast.parse(file_content)

    signature_class = next(node for node in ast.walk(tree)
                           if isinstance(node, ast.ClassDef) and node.name == 'Signature')
    hooks = [node for node in signature_class.body
             if isinstance(node, ast.FunctionDef) and node.name == 'validate_extracted_parameters']
    assert hooks == [], "the generator must not emit a trivial validate_extracted_parameters stub"
    assert "validate_extracted_parameters" not in file_content

    # Compiled with the file's absolute imports, the Signature class has no hook for the framework to call.
    imports = [node for node in tree.body
               if isinstance(node, ast.Import) or (isinstance(node, ast.ImportFrom) and node.level == 0)]
    namespace = {}
    exec(compile(ast.Module(body=imports + [signature_class], type_ignores=[]), file_path, 'exec'), namespace)
    assert not hasattr(namespace['Signature'], 'validate_extracted_parameters')


def test_generated_signature_leaves_utterances_to_the_framework_default(temp_dir, todo_item_class):
    """A generated Signature defines no generate_utterances and no template_utterances: the
    framework's default (command_directory.default_generate_utterances) gives the utterances."""
    output_dir = temp_dir / "TodoItem"
    output_dir.mkdir(exist_ok=True)
    file_path = create_command_file(
        class_info=todo_item_class,
        method_info=todo_item_class.methods[0],
        output_dir=output_dir,
        file_name="complete.py",
        source_dir="."
    )

    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    signature_class = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'Signature'),
        None)
    assert signature_class is not None, "Could not find Signature class"

    defined = {node.name for node in signature_class.body if isinstance(node, ast.FunctionDef)}
    assigned = {t.id for node in signature_class.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert "generate_utterances" not in defined, "generated Signature must not define generate_utterances"
    assert "template_utterances" not in assigned, "generated Signature must not define template_utterances"
    assert "plain_utterances" in assigned
