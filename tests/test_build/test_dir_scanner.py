import os
import pytest
from fastworkflow.build.dir_scanner import find_python_files, EXCLUDE_DIRS
from fastworkflow.utils.python_utils import get_module_import_path

# Define the root directory for examples
EXAMPLES_ROOT = './tests/todo_list_workflow/application'


def test_find_python_files_basic():
    files = find_python_files(EXAMPLES_ROOT)
    # Should find all .py files in the directory and subdirectories
    assert isinstance(files, list)
    assert all(f.endswith('.py') for f in files)
    # Should find at least the main files (adjust as needed for your project)
    expected_files = ['todo_list.py', 'todo_item.py', 'todo_manager.py']
    found_files = [os.path.basename(f) for f in files]
    for fname in expected_files:
        assert fname in found_files


def test_find_python_files_excludes_dirs(tmp_path):
    # Create a fake excluded directory with a .py file
    exclude_dir = tmp_path / 'venv'
    exclude_dir.mkdir()
    py_file = exclude_dir / 'should_not_find.py'
    py_file.write_text('print("hello")')
    # Also add a valid file in root
    valid_file = tmp_path / 'main.py'
    valid_file.write_text('print("main")')
    files = find_python_files(str(tmp_path))
    found_files = [os.path.basename(f) for f in files]
    assert 'main.py' in found_files
    assert 'should_not_find.py' not in found_files


def test_find_python_files_unreadable(tmp_path):
    # Create a .py file and make it unreadable
    unreadable = tmp_path / 'bad.py'
    unreadable.write_text('print("bad")')
    unreadable.chmod(0o000)
    try:
        files = find_python_files(str(tmp_path), verbose=True)
        assert 'bad.py' not in [os.path.basename(f) for f in files]
    finally:
        unreadable.chmod(0o644)


def make_file(tmp_path, rel_path):
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("# test file")
    return str(file_path)


def test_module_path_flat(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = make_file(src, "foo.py")
    assert get_module_import_path(f, src) == "foo"


def test_module_path_nested(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = make_file(src, "bar/baz.py")
    assert get_module_import_path(f, src) == "bar.baz"


def test_module_path_init(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = make_file(src, "bar/__init__.py")
    assert get_module_import_path(f, src) == "bar"
    f2 = make_file(src, "bar/baz/__init__.py")
    assert get_module_import_path(f2, src) == "bar.baz"


def test_module_path_outside_src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = make_file(tmp_path, "other.py")
    with pytest.raises(ValueError):
        get_module_import_path(f, src)


def test_module_path_invalid_identifier(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = make_file(src, "123abc.py")
    with pytest.raises(ValueError):
        get_module_import_path(f, src)


def test_find_python_files():
    """Test that the function finds Python files in the examples directory."""
    files = find_python_files(EXAMPLES_ROOT)
    assert len(files) > 0
    assert all(f.endswith('.py') for f in files) 