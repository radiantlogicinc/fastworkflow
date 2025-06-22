from typing import List, Dict, Optional, Any
from dataclasses import field

class MethodInfo:
    def __init__(self, name: str, parameters: List[Dict[str, Any]], docstring: Optional[str] = None, return_annotation: Optional[str] = None, decorators: Optional[List[str]] = None, docstring_parsed: Optional[Dict] = None):
        self.name = name
        self.parameters = parameters
        self.docstring = docstring
        self.return_annotation = return_annotation
        self.decorators = decorators or []
        self.docstring_parsed = docstring_parsed

    def __repr__(self):
        return f"<MethodInfo {self.name}({', '.join([p['name'] for p in self.parameters])})>"

    def to_dict(self):
        return {
            'name': self.name,
            'parameters': self.parameters,
            'docstring': self.docstring,
            'return_annotation': self.return_annotation,
            'decorators': self.decorators,
            'docstring_parsed': self.docstring_parsed,
        }

class FunctionInfo:
    def __init__(self, name: str, module_path: str, parameters: List[Dict[str, Any]], docstring: Optional[str] = None, return_annotation: Optional[str] = None, decorators: Optional[List[str]] = None, docstring_parsed: Optional[Dict] = None):
        self.name = name
        self.module_path = module_path
        self.parameters = parameters
        self.docstring = docstring
        self.return_annotation = return_annotation
        self.decorators = decorators or []
        self.docstring_parsed = docstring_parsed

    def __repr__(self):
        return f"<FunctionInfo {self.name}({', '.join([p['name'] for p in self.parameters])})>"

    def to_dict(self):
        return {
            'name': self.name,
            'module_path': self.module_path,
            'parameters': self.parameters,
            'docstring': self.docstring,
            'return_annotation': self.return_annotation,
            'decorators': self.decorators,
            'docstring_parsed': self.docstring_parsed,
        }

class PropertyInfo:
    def __init__(self, name: str, docstring: Optional[str] = None, type_annotation: Optional[str] = None, docstring_parsed: Optional[Dict] = None):
        self.name = name
        self.docstring = docstring
        self.type_annotation = type_annotation
        self.docstring_parsed = docstring_parsed

    def __repr__(self):
        return f"<PropertyInfo {self.name}>"

    def to_dict(self):
        return {
            'name': self.name,
            'docstring': self.docstring,
            'type_annotation': self.type_annotation,
            'docstring_parsed': self.docstring_parsed,
        }

class ClassInfo:
    def __init__(self, name: str, module_path: str, docstring: Optional[str] = None, bases: Optional[List[str]] = None, docstring_parsed: Optional[Dict] = None):
        self.name = name
        self.module_path = module_path
        self.docstring = docstring
        self.bases = bases or []
        self.methods: List[MethodInfo] = []
        self.properties: List[PropertyInfo] = []
        self.nested_classes: List['ClassInfo'] = []
        self.docstring_parsed = docstring_parsed if docstring_parsed is not None else {}
        self._property_setters: Dict[str, 'ast.FunctionDef'] = {}
        self.all_settable_properties: List[PropertyInfo] = []

    def __repr__(self):
        return f"<ClassInfo {self.name} in {self.module_path}>"

    def to_dict(self):
        return {
            'name': self.name,
            'module_path': self.module_path,
            'docstring': self.docstring,
            'bases': self.bases,
            'methods': [m.to_dict() for m in self.methods],
            'properties': [p.to_dict() for p in self.properties],
            'nested_classes': [c.to_dict() for c in self.nested_classes],
            '_property_setters': {k: str(v) for k, v in self._property_setters.items()},
            'docstring_parsed': self.docstring_parsed,
            'all_settable_properties': [p.to_dict() for p in self.all_settable_properties],
        } 