import pytest
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
from fastworkflow.build.command_import_utils import generate_import_statements
from fastworkflow.utils.python_utils import find_module_dependencies, extract_custom_types_from_annotation

def test_method_info():
    params = [{'name': 'x', 'annotation': 'int'}, {'name': 'y', 'annotation': 'str'}]
    m = MethodInfo('foo', params, docstring='A method', return_annotation='bool', decorators=['staticmethod'])
    assert m.name == 'foo'
    assert m.parameters == params
    assert m.docstring == 'A method'
    assert m.return_annotation == 'bool'
    assert m.decorators == ['staticmethod']
    assert 'foo' in repr(m)
    d = m.to_dict()
    assert d['name'] == 'foo'
    assert d['parameters'] == params
    assert d['docstring'] == 'A method'
    assert d['return_annotation'] == 'bool'
    assert d['decorators'] == ['staticmethod']

def test_property_info():
    p = PropertyInfo('bar', docstring='A property', type_annotation='int')
    assert p.name == 'bar'
    assert p.docstring == 'A property'
    assert p.type_annotation == 'int'
    assert 'bar' in repr(p)
    d = p.to_dict()
    assert d['name'] == 'bar'
    assert d['docstring'] == 'A property'
    assert d['type_annotation'] == 'int'

def test_class_info():
    c = ClassInfo('Baz', 'baz.py', docstring='A class', bases=['Base'])
    assert c.name == 'Baz'
    assert c.module_path == 'baz.py'
    assert c.docstring == 'A class'
    assert c.bases == ['Base']
    assert c.methods == []
    assert c.properties == []
    assert c.nested_classes == []
    assert 'Baz' in repr(c)
    c.methods.append(MethodInfo('foo', []))
    c.properties.append(PropertyInfo('bar'))
    c.nested_classes.append(ClassInfo('Inner', 'baz.py'))
    d = c.to_dict()
    assert d['name'] == 'Baz'
    assert d['module_path'] == 'baz.py'
    assert d['docstring'] == 'A class'
    assert d['bases'] == ['Base']
    assert isinstance(d['methods'], list)
    assert isinstance(d['properties'], list)
    assert isinstance(d['nested_classes'], list)

def test_edge_cases():
    # Empty/None fields
    m = MethodInfo('foo', [])
    p = PropertyInfo('bar')
    c = ClassInfo('Baz', 'baz.py')
    assert m.to_dict()['docstring'] is None
    assert p.to_dict()['type_annotation'] is None
    assert c.to_dict()['docstring'] is None

def test_extract_custom_types_from_annotation_basic():
    assert extract_custom_types_from_annotation('Foo') == {'Foo'}
    assert extract_custom_types_from_annotation('List[Foo]') == {'Foo'}
    assert extract_custom_types_from_annotation('Optional[Bar]') == {'Bar'}
    assert extract_custom_types_from_annotation('Dict[str, Baz]') == {'Baz'}
    assert extract_custom_types_from_annotation('Union[Foo, Bar]') == {'Foo', 'Bar'}
    assert extract_custom_types_from_annotation('List[Dict[str, Bar]]') == {'Bar'}
    assert extract_custom_types_from_annotation('int') == set()
    assert extract_custom_types_from_annotation('List[int]') == set()
    assert extract_custom_types_from_annotation('') == set()

def test_find_module_dependencies_all():
    # Class with base, method param, return, and property
    c = ClassInfo('MyClass', 'foo.py', bases=['Base', 'object'])
    c.methods.append(MethodInfo('do', [
        {'name': 'x', 'annotation': 'Foo'},
        {'name': 'y', 'annotation': 'List[Bar]'},
        {'name': 'z', 'annotation': 'int'}
    ], return_annotation='Baz'))
    c.properties.append(PropertyInfo('prop', type_annotation='Optional[Qux]'))
    deps = find_module_dependencies(c)
    assert deps == {'Base', 'Foo', 'Bar', 'Baz', 'Qux'}

def test_find_module_dependencies_nested_types():
    c = ClassInfo('C', 'c.py', bases=[])
    c.methods.append(MethodInfo('f', [
        {'name': 'a', 'annotation': 'List[Dict[str, Foo]]'},
        {'name': 'b', 'annotation': 'Optional[Bar]'}
    ], return_annotation='Union[Baz, None]'))
    c.properties.append(PropertyInfo('p', type_annotation='List[Qux]'))
    deps = find_module_dependencies(c)
    assert deps == {'Foo', 'Bar', 'Baz', 'Qux'}

def test_find_module_dependencies_no_custom():
    c = ClassInfo('Simple', 's.py', bases=['object'])
    c.methods.append(MethodInfo('f', [
        {'name': 'a', 'annotation': 'int'}
    ], return_annotation='str'))
    c.properties.append(PropertyInfo('p', type_annotation='float'))
    deps = find_module_dependencies(c)
    assert deps == set()

@pytest.mark.skip(reason="Skipping import statement validation as per user request")
def test_generate_import_statements_basic(tmp_path):
    from fastworkflow.utils.python_utils import get_module_import_path
    src = tmp_path / "src"
    src.mkdir()
    f = src / "foo.py"
    f.write_text("# test file")
    c = ClassInfo('Foo', str(f))
    imports = generate_import_statements(c, src)
    assert f"from {get_module_import_path(str(f), src)} import Foo" in imports
    assert "from typing import List" in imports
    assert "from pydantic import BaseModel" in imports

@pytest.mark.skip(reason="Skipping import statement validation as per user request")
def test_generate_import_statements_with_deps(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = src / "foo.py"
    f.write_text("# test file")
    c = ClassInfo('Foo', str(f), bases=['Bar', 'Baz'])
    # No mapping: should add comments
    imports = generate_import_statements(c, src)
    assert "# You may need to import Bar" in imports
    assert "# You may need to import Baz" in imports
    # With mapping: should import
    mapping = {'Bar': 'myapp.bar', 'Baz': 'myapp.baz'}
    imports2 = generate_import_statements(c, src, mapping)
    assert "from myapp.bar import Bar" in imports2
    assert "from myapp.baz import Baz" in imports2
    # Should not have the comment if mapping is present
    assert "# You may need to import Bar" not in imports2
    assert "# You may need to import Baz" not in imports2 