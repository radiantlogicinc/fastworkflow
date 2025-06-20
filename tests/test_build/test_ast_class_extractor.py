import os
import pytest
from fastworkflow.build.ast_class_extractor import analyze_python_file
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo

EXAMPLES_ROOT = './examples/todo_list/application'


def test_analyze_python_file_todo_list():
    file_path = os.path.join(EXAMPLES_ROOT, 'todo_list.py')
    classes = analyze_python_file(file_path)
    assert isinstance(classes, dict)
    assert 'TodoList' in classes
    todo_list = classes['TodoList']
    assert isinstance(todo_list, ClassInfo)
    # Should have at least one public method
    assert any(isinstance(m, MethodInfo) for m in todo_list.methods)
    # Check that private methods are not included
    assert all(not m.name.startswith('_') for m in todo_list.methods)
    # Check that method signatures are extracted
    for m in todo_list.methods:
        assert isinstance(m.parameters, list)
        for p in m.parameters:
            assert 'name' in p


def test_analyze_python_file_todo_item():
    file_path = os.path.join(EXAMPLES_ROOT, 'todo_item.py')
    classes = analyze_python_file(file_path)
    assert 'TodoItem' in classes
    todo_item = classes['TodoItem']
    assert isinstance(todo_item, ClassInfo)
    # Check that private methods are not included
    assert all(not m.name.startswith('_') for m in todo_item.methods)
    # Check that method signatures are extracted (if any methods exist)
    for m in todo_item.methods:
        assert isinstance(m.parameters, list)
        for p in m.parameters:
            assert 'name' in p
    # Check for docstrings (if any methods exist)
    for m in todo_item.methods:
        assert hasattr(m, 'docstring')


def test_analyze_python_file_nested_class(tmp_path):
    # Create a file with a nested class
    code = '''
class Outer:
    def foo(self): pass
    class Inner:
        def bar(self): pass
'''
    file_path = tmp_path / 'nested.py'
    file_path.write_text(code)
    classes = analyze_python_file(str(file_path))
    assert 'Outer' in classes
    outer = classes['Outer']
    assert any(nc.name == 'Inner' for nc in outer.nested_classes)
    inner = next(nc for nc in outer.nested_classes if nc.name == 'Inner')
    assert any(m.name == 'bar' for m in inner.methods)


def test_analyze_python_file_properties(tmp_path):
    code = '''
class MyClass:
    @property
    def foo(self) -> int:
        """Foo property docstring."""
        return 42

    @foo.setter
    def foo(self, value: int):
        pass

    def bar(self):
        """A regular method."""
        pass
'''
    file_path = tmp_path / 'props.py'
    file_path.write_text(code)
    classes = analyze_python_file(str(file_path))
    assert 'MyClass' in classes
    myclass = classes['MyClass']
    # Should detect one property
    assert len(myclass.properties) == 1
    prop = myclass.properties[0]
    assert prop.name == 'foo'
    assert prop.docstring == 'Foo property docstring.'
    assert prop.type_annotation == 'int'
    # Should detect two methods: the property setter and bar
    method_names = {m.name for m in myclass.methods}
    assert method_names == {'foo', 'bar'}


def test_google_style_docstring_parsing(tmp_path):
    code = '''
class MyClass:
    """MyClass summary.
    
    Args:
        x (int): The x value.
        y (str): The y value.
    Returns:
        MyClass: An instance.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def foo(self):
        """Foo property.
        Returns:
            int: The foo value.
        """
        return 42

    def bar(self, a, b):
        """Bar method summary.
        
        Args:
            a (int): First arg.
            b (str): Second arg.
        Returns:
            bool: True if success.
        """
        return True
'''
    file_path = tmp_path / 'docstrings.py'
    file_path.write_text(code)
    classes = analyze_python_file(str(file_path))
    assert 'MyClass' in classes
    myclass = classes['MyClass']
    # Class docstring
    parsed = myclass.docstring_parsed
    assert parsed['summary'] == 'MyClass summary.'
    assert parsed['params'][0]['name'] == 'x'
    assert parsed['params'][0]['type'] == 'int'
    assert parsed['params'][0]['desc'] == 'The x value.'
    assert parsed['returns']['type'] == 'MyClass'
    # Property docstring
    prop = myclass.properties[0]
    prop_parsed = prop.docstring_parsed
    assert prop_parsed['summary'] == 'Foo property.'
    assert prop_parsed['returns']['type'] == 'int'
    # Method docstring
    bar = next(m for m in myclass.methods if m.name == 'bar')
    bar_parsed = bar.docstring_parsed
    assert bar_parsed['summary'] == 'Bar method summary.'
    assert bar_parsed['params'][0]['name'] == 'a'
    assert bar_parsed['params'][0]['type'] == 'int'
    assert bar_parsed['params'][0]['desc'] == 'First arg.'
    assert bar_parsed['returns']['type'] == 'bool'


def test_class_variable_type_annotations(tmp_path):
    code = '''
from typing import List, Optional

class MyClass:
    x: int
    y: Optional[str]
    z: List[int]
    def __init__(self):
        self.x = 1
        self.y = None
        self.z = []
'''
    file_path = tmp_path / 'classvars.py'
    file_path.write_text(code)
    classes = analyze_python_file(str(file_path))
    assert 'MyClass' in classes
    myclass = classes['MyClass']
    prop_types = {p.name: p.type_annotation for p in myclass.properties}
    assert prop_types['x'] == 'int'
    assert prop_types['y'] == 'Optional[str]'
    assert prop_types['z'] == 'List[int]'


def test_extracts_base_classes(tmp_path):
    code = '''
class Base:
    pass
class Derived(Base):
    pass
'''
    file_path = tmp_path / 'inheritance.py'
    file_path.write_text(code)
    classes = analyze_python_file(str(file_path))
    assert 'Derived' in classes
    derived = classes['Derived']
    assert derived.bases == ['Base'], f"Expected ['Base'], got {derived.bases}"
    assert 'Base' in classes
    base = classes['Base']
    assert base.bases == [] 