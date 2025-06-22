from fastworkflow.command_context_model import CommandContextModel

def list_context_names(workflow_folderpath: str, include_default: bool = False) -> list[str]:
    """Return context names defined in context_inheritance_model.json.
    If include_default is False the '*' context is filtered out.
    """
    model = CommandContextModel.load(workflow_folderpath)
    names = list(model._command_contexts.keys())
    if not include_default and "*" in names:
        names.remove("*")
    return names 

def get_context_names(model: dict) -> set[str]:
    """Extract context names from a context model.
    
    Args:
        model: The context model dictionary
        
    Returns:
        Set of context names
    """
    return set(model.keys()) 