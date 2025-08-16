from fastworkflow.build.class_analysis_structures import MethodInfo, PropertyInfo
from fastworkflow.build.pydantic_model_generator import generate_input_model_code, generate_output_model_code, generate_property_output_model_code

def test_generate_input_model_code_with_annotations():
    method = MethodInfo('foo', [
        {'name': 'x', 'annotation': 'int'},
        {'name': 'y', 'annotation': 'str'}
    ], docstring='Foo method.')
    code = generate_input_model_code('MyClass', method)
    assert 'x: int' in code
    assert 'y: str' in code
    assert 'class MyClassFooInput' in code

def test_generate_input_model_code_with_docstring_types():
    method = MethodInfo('bar', [
        {'name': 'a'},
        {'name': 'b'}
    ], docstring='Bar method.')
    method.docstring_parsed = {
        'params': [
            {'name': 'a', 'type': 'float', 'desc': 'A param.'},
            {'name': 'b', 'type': 'bool', 'desc': 'B param.'}
        ]
    }
    code = generate_input_model_code('MyClass', method)
    assert 'a: float' in code
    assert 'b: bool' in code
    # Description comments now added by GenAI postprocessor
    # assert 'A param.' in code
    # assert 'B param.' in code

def test_generate_input_model_code_with_missing_types():
    method = MethodInfo('baz', [{'name': 'z'}], docstring='Baz method.')
    code = generate_input_model_code('MyClass', method)
    assert 'z: Any' in code

def test_generate_output_model_code():
    method = MethodInfo('foo', [], docstring='Foo method.', return_annotation='str')
    code = generate_output_model_code('MyClass', method)
    assert 'result: str' in code
    assert 'class MyClassFooOutput' in code

def test_generate_output_model_code_with_docstring():
    method = MethodInfo('bar', [], docstring='Bar method.')
    method.docstring_parsed = {'returns': {'type': 'int', 'desc': 'Return value.'}}
    code = generate_output_model_code('MyClass', method)
    assert 'result: int' in code
    # Description comments now added by GenAI postprocessor
    # assert 'Return value.' in code

def test_generate_property_output_model_code():
    prop = PropertyInfo('value', docstring='Value property.', type_annotation='float')
    prop.docstring_parsed = {'returns': {'type': 'float', 'desc': 'The value.'}}
    code = generate_property_output_model_code('MyClass', prop)
    assert 'value: float' in code
    # Description comments now added by GenAI postprocessor
    # assert 'The value.' in code 