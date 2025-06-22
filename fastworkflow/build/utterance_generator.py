def generate_utterances(class_name: str, method_or_property_name: str, parameters: list, is_property: bool = False, is_function: bool = False) -> list:
    """Generate natural language utterances for a command."""
    name_lc = method_or_property_name.lower()
    utterances = []
    
    if is_function:
        # Function utterances
        base = f"{name_lc.replace('_', ' ')}"
        # Direct invocation
        utterances.append(f"{base}")
        # Parameterized invocation
        if parameters:
            param_str = ' '.join([f'{{{p["name"]}}}' for p in parameters])
            utterances.append(f"{base} {param_str}")
            utterances.append(f"call {name_lc} with {param_str}")
    elif is_property:
        # Property getter utterances
        class_name_lc = class_name.lower()
        utterances.append(f"Get {name_lc} of {class_name_lc}")
        utterances.append(f"Retrieve {class_name_lc} {name_lc}")
        utterances.append(f"Show {class_name_lc} {name_lc}")
    else:
        # Method utterances
        class_name_lc = class_name.lower()
        base = f"{name_lc.replace('_', ' ')} {class_name_lc}" if not name_lc.startswith(('get', 'set', 'update', 'delete')) else f"{name_lc.replace('_', ' ')} {class_name_lc}"
        # Direct invocation
        utterances.append(f"{base}")
        utterances.append(f"Call {name_lc} on {class_name_lc}")
        # Parameterized invocation
        if parameters:
            param_str = ' '.join([f'{{{p["name"]}}}' for p in parameters])
            utterances.append(f"{base} {param_str}")
            utterances.append(f"Call {name_lc} on {class_name_lc} with {param_str}")
    return utterances 