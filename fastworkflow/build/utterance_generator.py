def generate_utterances(class_name: str, method_or_property_name: str, parameters: list, is_property: bool = False, is_function: bool = False) -> list:
    """Generate natural language utterances for a command."""
    # Return empty list - GenAI postprocessor will generate utterances
    return [] 