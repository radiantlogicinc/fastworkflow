from typing import List, Any
from fastworkflow.build.class_analysis_structures import MethodInfo, PropertyInfo

def generate_input_model_code(class_name: str, method_info: MethodInfo) -> str:
    """Generate Pydantic input model code for a method."""
    model_name = f"{class_name}{method_info.name.capitalize()}Input"
    fields = []
    doc_params = {p['name']: p for p in (method_info.docstring_parsed.get('params', []) if method_info.docstring_parsed else [])}
    for param in method_info.parameters:
        name = param['name']
        typ = param.get('annotation') or doc_params.get(name, {}).get('type') or 'Any'
        desc = doc_params.get(name, {}).get('desc', f"Parameter {name}")
        # If type is missing or ambiguous, use Any
        if not typ:
            typ = 'Any'
        fields.append(f"    {name}: {typ} = Field(description=\"{desc}\")")
    if not fields:
        fields.append("    pass")
    return f"class {model_name}(BaseModel):\n" + "\n".join(fields)

def generate_output_model_code(class_name: str, method_info: MethodInfo) -> str:
    """Generate Pydantic output model code for a method."""
    model_name = f"{class_name}{method_info.name.capitalize()}Output"
    doc_return = (method_info.docstring_parsed.get('returns') if method_info.docstring_parsed else None)
    typ = method_info.return_annotation or (doc_return['type'] if doc_return else None) or 'Any'
    desc = (doc_return['desc'] if doc_return else "Result of the method call")
    return f"class {model_name}(BaseModel):\n    result: {typ} = Field(description=\"{desc}\")"

def generate_property_output_model_code(class_name: str, prop_info: PropertyInfo) -> str:
    """Generate Pydantic output model code for a property getter."""
    model_name = f"{class_name}Get_{prop_info.name.capitalize()}Output"
    typ = prop_info.type_annotation or 'Any'
    desc = (prop_info.docstring_parsed.get('returns', {}).get('desc') if prop_info.docstring_parsed else None) or f"Value of {prop_info.name}"
    return f"class {model_name}(BaseModel):\n    value: {typ} = Field(description=\"{desc}\")" 