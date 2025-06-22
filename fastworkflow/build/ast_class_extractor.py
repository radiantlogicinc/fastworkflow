import ast
from typing import Dict, Tuple
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo, FunctionInfo

try:
    from ast import unparse as ast_unparse
except ImportError:
    def ast_unparse(node):
        # Fallback for Python <3.9
        return str(node)

def parse_google_docstring(docstring: str) -> dict:
    """Parse a Google-style docstring into structured data."""
    if not docstring:
        return {}
    import re
    lines = docstring.strip().splitlines()
    summary = lines[0].strip() if lines else ''
    params = []
    returns = None
    in_args = False
    in_returns = False
    param_pattern = re.compile(r'^\s*(\w+) \(([^)]+)\): (.+)$')
    arg_pattern = re.compile(r'^\s*(\w+): (.+)$')
    return_pattern = re.compile(r'^\s*([^:]+): (.+)$')
    for i, line in enumerate(lines[1:]):
        l = line.strip()
        if l.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            in_returns = False
            continue
        if l.lower() in ("returns:", "return:"):
            in_args = False
            in_returns = True
            continue
        if in_args:
            m = param_pattern.match(l) or arg_pattern.match(l)
            if m:
                if len(m.groups()) == 3:
                    params.append({'name': m.group(1), 'type': m.group(2), 'desc': m.group(3)})
                elif len(m.groups()) == 2:
                    params.append({'name': m.group(1), 'type': None, 'desc': m.group(2)})
        elif in_returns:
            m = return_pattern.match(l)
            if m:
                returns = {'type': m.group(1).strip(), 'desc': m.group(2).strip()}
    return {
        'summary': summary,
        'params': params,
        'returns': returns
    }

def analyze_python_file(file_path: str) -> Tuple[Dict[str, ClassInfo], Dict[str, FunctionInfo]]:
    with open(file_path, 'r') as f:
        node = ast.parse(f.read(), filename=file_path)
    classes = {}
    functions = {}

    def extract_class(class_node, parent_path=None):
        class_name = class_node.name
        module_path = parent_path or file_path
        class_doc = ast.get_docstring(class_node)
        # Extract base class names
        bases = [ast_unparse(base) for base in class_node.bases]
        class_info = ClassInfo(class_name, module_path, docstring=class_doc, bases=bases, docstring_parsed=parse_google_docstring(class_doc))
        # Methods and Properties
        seen_properties = set(p.name for p in class_info.properties)
        property_setters = {}
        for item in class_node.body:
            # Class variable with type annotation
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                var_name = item.target.id
                if var_name not in seen_properties:
                    type_annotation = ast_unparse(item.annotation) if item.annotation else None
                    property_info = PropertyInfo(var_name, docstring=None, type_annotation=type_annotation)
                    class_info.properties.append(property_info)
                    seen_properties.add(var_name)
            # Methods and Properties
            if isinstance(item, ast.FunctionDef):
                # Detect @property and @<property>.setter
                is_property = False
                is_setter = False
                for decorator in item.decorator_list:
                    # @property
                    if isinstance(decorator, ast.Name) and decorator.id == 'property':
                        is_property = True
                        break
                    # @x.setter
                    if isinstance(decorator, ast.Attribute) and decorator.attr == 'setter':
                        is_setter = True
                        setter_for = decorator.value.id if isinstance(decorator.value, ast.Name) else None
                        if setter_for:
                            property_setters[setter_for] = item
                        break # Found a relevant decorator for setter
                    # @x.deleter (future)
                    if isinstance(decorator, ast.Attribute) and decorator.attr == 'deleter':
                        # Could be handled in future
                        pass
                
                if is_property:
                    prop_name = item.name
                    if prop_name in seen_properties:
                        continue  # Only add the first occurrence
                    seen_properties.add(prop_name)
                    docstring = ast.get_docstring(item)
                    type_annotation = ast_unparse(item.returns) if item.returns else None
                    property_info = PropertyInfo(prop_name, docstring, type_annotation, docstring_parsed=parse_google_docstring(docstring))
                    class_info.properties.append(property_info)
                    continue  # Do not add to methods if it's a property getter

                # If it's a setter, still treat it as a public method so that
                # analyses (and legacy tests) include the setter alongside
                # other methods.
                if is_setter:
                    # fall through to regular public-method processing
                    pass

                # Regular method (neither a property getter nor a property setter)
                if not item.name.startswith('_'):
                    params = []
                    for arg in item.args.args:
                        if arg.arg != 'self':
                            param_info = {'name': arg.arg}
                            param_info['annotation'] = ast_unparse(arg.annotation) if arg.annotation else None
                            params.append(param_info)
                    docstring = ast.get_docstring(item)
                    return_annotation = ast_unparse(item.returns) if item.returns else None
                    decorators = [ast_unparse(d) for d in item.decorator_list]
                    method_info = MethodInfo(item.name, params, docstring, return_annotation, decorators, docstring_parsed=parse_google_docstring(docstring))
                    class_info.methods.append(method_info)
        # Nested classes
        for item in class_node.body:
            if isinstance(item, ast.ClassDef):
                nested = extract_class(item, parent_path=module_path)
                class_info.nested_classes.append(nested)
        # Attach property setters info to class_info for later use
        class_info._property_setters = property_setters
        return class_info

    # Extract top-level functions
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            # Skip private functions
            if item.name.startswith('_'):
                continue
                
            # Extract function parameters
            params = []
            for arg in item.args.args:
                param_info = {'name': arg.arg}
                param_info['annotation'] = ast_unparse(arg.annotation) if arg.annotation else None
                params.append(param_info)
                
            # Extract docstring and return annotation
            docstring = ast.get_docstring(item)
            return_annotation = ast_unparse(item.returns) if item.returns else None
            decorators = [ast_unparse(d) for d in item.decorator_list]
            
            # Create FunctionInfo
            function_info = FunctionInfo(
                name=item.name,
                module_path=file_path,
                parameters=params,
                docstring=docstring,
                return_annotation=return_annotation,
                decorators=decorators,
                docstring_parsed=parse_google_docstring(docstring)
            )
            
            # Add to functions dict
            functions[item.name] = function_info

    # Extract classes
    for item in ast.walk(node):
        if isinstance(item, ast.ClassDef) and item in node.body:  # Only top-level classes
            class_info = extract_class(item)
            classes[class_info.name] = class_info
            
    return classes, functions

def resolve_inherited_properties(all_my_classes: Dict[str, ClassInfo]) -> None:
    """Update each ClassInfo object in all_my_classes to include inherited properties."""
    for class_name in list(all_my_classes.keys()): # Iterate over a copy of keys if modifying dict during iteration, though here we modify objects
        if class_name not in all_my_classes: # Should not happen if keys are from the dict itself
            continue
        
        class_info = all_my_classes[class_name]
        
        # Get MRO for the class (excluding the class itself for finding *inherited* properties)
        # A proper MRO calculation is complex. For simplicity, we'll do a depth-first traversal of bases.
        # This assumes 'object' and other common builtins are not in all_my_classes or are handled.
        # ClassInfo.bases contains immediate parent names.
        
        mro = []
        visited_bases = set()
        
        def get_mro_recursive(current_c_name):
            if current_c_name in visited_bases or current_c_name not in all_my_classes:
                return
            visited_bases.add(current_c_name)
            # Add parents first (depth-first)
            parent_class_info = all_my_classes[current_c_name]
            for base_n in parent_class_info.bases:
                get_mro_recursive(base_n)
            # Add current class to MRO list after its parents
            if current_c_name not in mro:
                 mro.append(current_c_name)

        # Build MRO starting from current class, then reverse to get typical MRO order (self, then parents)
        temp_mro_build_list = []
        
        def build_mro_for_class(c_name_build):
            if c_name_build not in all_my_classes or c_name_build in temp_mro_build_list:
                return
            
            base_class_infos_for_mro = all_my_classes[c_name_build].bases
            for base_c_name_mro in base_class_infos_for_mro:
                build_mro_for_class(base_c_name_mro)
            
            if c_name_build not in temp_mro_build_list:
                temp_mro_build_list.append(c_name_build)

        build_mro_for_class(class_name)
        # The temp_mro_build_list is in C3-like order (class, then linearized parents)
        # For finding inherited properties, we want to iterate from furthest ancestor to direct parent.
        # So, we reverse it, and skip the class itself (first element after reverse).
        
        ordered_bases_for_inheritance = list(reversed(temp_mro_build_list[:-1]))

        current_property_names = {p.name for p in class_info.properties}

        for base_class_name_to_inherit_from in ordered_bases_for_inheritance:
            if base_class_name_to_inherit_from in all_my_classes:
                base_class_info_to_inherit = all_my_classes[base_class_name_to_inherit_from]
                for prop_to_inherit in base_class_info_to_inherit.properties:
                    if prop_to_inherit.name not in current_property_names:
                        class_info.properties.append(prop_to_inherit) 
                        current_property_names.add(prop_to_inherit.name)
        
        # Phase 2: Resolve all_settable_properties, respecting MRO for overrides
        # temp_mro_build_list is already in C3-like order: class, then linearized parents.
        # We iterate through this MRO. The first class in MRO that defines a setter for a property wins.
        
        added_settable_prop_names = set()
        # The MRO (temp_mro_build_list) is e.g., [Derived, Base1, Base2, object-like-base]
        # When looking for setters for 'Derived', we check Derived._property_setters first, then Base1._property_setters etc.
        for mro_class_name_for_setter in temp_mro_build_list:
            if mro_class_name_for_setter not in all_my_classes:
                continue
            mro_class_info_for_setter = all_my_classes[mro_class_name_for_setter]
            
            # Iterate direct setters of this class in the MRO
            for prop_name, _setter_ast_node in mro_class_info_for_setter._property_setters.items():
                if prop_name not in added_settable_prop_names:
                    # Found the highest-priority setter for this prop_name for the original class_info
                    # Now, get its type from the fully resolved class_info.properties list
                    prop_type = 'Any' # Default type
                    for p_info in class_info.properties: # class_info is the original class we are populating
                        if p_info.name == prop_name:
                            prop_type = p_info.type_annotation
                            break
                    
                    # Create a PropertyInfo for this settable property
                    # Docstring for setters can be generic or extracted from setter_ast_node if needed later
                    settable_prop_info = PropertyInfo(
                        name=prop_name, 
                        type_annotation=prop_type, 
                        docstring=f"Settable property {prop_name}."
                        # docstring_parsed could be populated if setter docstrings are parsed
                    )
                    class_info.all_settable_properties.append(settable_prop_info)
                    added_settable_prop_names.add(prop_name) 