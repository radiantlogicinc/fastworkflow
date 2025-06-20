from fastworkflow.command_context_model import CommandContextModel

def list_context_names(workflow_folderpath: str, include_default: bool = False) -> list[str]:
    """Return context names defined in command_context_model.json.
    If include_default is False the '*' context is filtered out.
    """
    model = CommandContextModel.load(workflow_folderpath)
    names = list(model._command_contexts.keys())
    if not include_default and "*" in names:
        names.remove("*")
    return names 