import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any

from sentence_transformers import SentenceTransformer, util as st_util

# import dspy  # type: ignore

# import fastworkflow  # For env configuration when using DSPy
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.utils import python_utils
from fastworkflow.command_metadata_api import CommandMetadataAPI


@dataclass
class ParamMeta:
    name: str
    type_str: str
    description: str
    examples: List[str]


@dataclass
class CommandParams:
    inputs: List[ParamMeta]
    outputs: List[ParamMeta]


# def _serialize_type_str(t: Any) -> str:
#     try:
#         # Basic serialization for common typing and classes
#         return t.__name__ if hasattr(t, "__name__") else str(t)
#     except Exception:
#         return str(t)


def _collect_command_params(workflow_path: str) -> Dict[str, CommandParams]:
    params_map = CommandMetadataAPI.get_params_for_all_commands(workflow_path)

    results: Dict[str, CommandParams] = {}
    for qualified_name, io in params_map.items():
        inputs = [
            ParamMeta(
                name=p.get("name", ""),
                type_str=str(p.get("type_str", "")),
                description=str(p.get("description", "")),
                examples=list(p.get("examples", []) or []),
            )
            for p in io.get("inputs", [])
        ]
        outputs = [
            ParamMeta(
                name=p.get("name", ""),
                type_str=str(p.get("type_str", "")),
                description=str(p.get("description", "")),
                examples=list(p.get("examples", []) or []),
            )
            for p in io.get("outputs", [])
        ]
        # Include all discovered commands; context overlap is handled later
        results[qualified_name] = CommandParams(inputs=inputs, outputs=outputs)

    return results


def _exact_match(out_param: ParamMeta, in_param: ParamMeta) -> bool:
    return out_param.name.lower() == in_param.name.lower() and out_param.type_str == in_param.type_str


def _semantic_match(out_param: ParamMeta, in_param: ParamMeta, threshold: float = 0.85) -> bool:
    # Sentence Transformers cosine similarity between parameter texts
    def to_text(p: ParamMeta) -> str:
        parts = [
            f"name: {p.name}",
            f"type: {p.type_str}",
            f"description: {p.description}",
            f"examples: {'|'.join(map(str, p.examples))}",
        ]
        return " | ".join(parts).lower()

    out_param_text = to_text(out_param)
    in_param_text = to_text(in_param)
    if not out_param_text or not in_param_text:
        return False

    # Lazy-load model and cache embeddings to avoid repeated work
    global _st_model  # type: ignore
    global _embedding_cache  # type: ignore
    if '_st_model' not in globals():
        _st_model = None  # type: ignore
    if '_embedding_cache' not in globals():
        _embedding_cache = {}  # type: ignore

    if _st_model is None:
        _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # type: ignore

    if out_param_text not in _embedding_cache:
        _embedding_cache[out_param_text] = _st_model.encode(
            out_param_text, convert_to_tensor=True, normalize_embeddings=True
        )
    if in_param_text not in _embedding_cache:
        _embedding_cache[in_param_text] = _st_model.encode(
            in_param_text, convert_to_tensor=True, normalize_embeddings=True
        )

    emb_out = _embedding_cache[out_param_text]
    emb_in = _embedding_cache[in_param_text]
    sim = st_util.cos_sim(emb_out, emb_in)  # 1x1 tensor
    return float(sim.item()) >= threshold


# ----------------------------------------------------------------------------
# LLM-based matching using DSPy
# ----------------------------------------------------------------------------

# Lazy singletons
# _llm_initialized: bool = False  # type: ignore
# _llm_module: Optional["CommandDependencyModule"] = None  # type: ignore


# def _initialize_dspy_llm_if_needed() -> None:
#     """Initialize DSPy LM once using FastWorkflow environment.

#     Controlled by env vars:
#       - LLM_COMMAND_METADATA_GEN (model id for LiteLLM via DSPy)
#       - LITELLM_API_KEY_COMMANDMETADATA_GEN (API key)
#     """
#     global _llm_initialized, _llm_module
#     if _llm_initialized:
#         return

#     model = fastworkflow.get_env_var("LLM_COMMAND_METADATA_GEN")
#     api_key = fastworkflow.get_env_var("LITELLM_API_KEY_COMMANDMETADATA_GEN")
#     lm = dspy.LM(model=model, api_key=api_key, max_tokens=1000)

#     # Define signature and module only if dspy is available
#     class CommandDependencySignature(dspy.Signature):  # type: ignore
#         """Analyze if two commands have a dependency relationship.

#         There is a dependency relationship if and only if the outputs from one command can be used directly as inputs of the other.
#         Tip for figuring out dependency direction: Commands with hard-to-remember inputs (such as id) typically depend on commands with easy to remember inputs (such as name, email).
#         """

#         cmd_x_name: str = dspy.InputField(desc="Name of command X")
#         cmd_x_inputs: str = dspy.InputField(desc="Input parameters of command X (name:type)")
#         cmd_x_outputs: str = dspy.InputField(desc="Output parameters of command X (name:type)")
        
#         cmd_y_name: str = dspy.InputField(desc="Name of command Y")
#         cmd_y_inputs: str = dspy.InputField(desc="Input parameters of command Y (name:type)")
#         cmd_y_outputs: str = dspy.InputField(desc="Output parameters of command Y (name:type)")

#         has_dependency: bool = dspy.OutputField(
#             desc="True if there's a dependency between the commands"
#         )
#         direction: str = dspy.OutputField(
#             desc="Direction: 'x_depends_on_y', 'y_depends_on_x', or 'none'"
#         )

#     class CommandDependencyModule(dspy.Module):  # type: ignore
#         def __init__(self):
#             super().__init__()
#             self.generate = dspy.ChainOfThought(CommandDependencySignature)

#         def forward(
#             self,
#             cmd_x_name: str,
#             cmd_x_inputs: str,
#             cmd_x_outputs: str,
#             cmd_y_name: str,
#             cmd_y_inputs: str,
#             cmd_y_outputs: str,
#         ) -> tuple[bool, str]:
#             with dspy.context(lm=lm):
#                 prediction = self.generate(
#                     cmd_x_name=cmd_x_name,
#                     cmd_x_inputs=cmd_x_inputs,
#                     cmd_x_outputs=cmd_x_outputs,
#                     cmd_y_name=cmd_y_name,
#                     cmd_y_inputs=cmd_y_inputs,
#                     cmd_y_outputs=cmd_y_outputs,
#                 )
#                 return prediction.has_dependency, prediction.direction

#     _llm_module = CommandDependencyModule()
#     _llm_initialized = True


# def _llm_command_dependency(
#     cmd_x_name: str, 
#     cmd_x_params: CommandParams,
#     cmd_y_name: str,
#     cmd_y_params: CommandParams
# ) -> Optional[str]:
#     """Check if two commands have a dependency using LLM.
    
#     Returns:
#         - "x_to_y" if Y depends on X (X's outputs feed Y's inputs)
#         - "y_to_x" if X depends on Y (Y's outputs feed X's inputs)  
#         - None if no dependency
#     """
#     _initialize_dspy_llm_if_needed()

#     # Format parameters for LLM
#     x_inputs = ", ".join([f"{p.name}:{p.type_str}" for p in cmd_x_params.inputs])
#     x_outputs = ", ".join([f"{p.name}:{p.type_str}" for p in cmd_x_params.outputs])
#     y_inputs = ", ".join([f"{p.name}:{p.type_str}" for p in cmd_y_params.inputs])
#     y_outputs = ", ".join([f"{p.name}:{p.type_str}" for p in cmd_y_params.outputs])

#     has_dep, direction = _llm_module(
#         cmd_x_name=cmd_x_name,
#         cmd_x_inputs=x_inputs or "none",
#         cmd_x_outputs=x_outputs or "none",
#         cmd_y_name=cmd_y_name,
#         cmd_y_inputs=y_inputs or "none",
#         cmd_y_outputs=y_outputs or "none",
#     )

#     if not has_dep or direction == "none":
#         return None
#     if direction == "x_depends_on_y":
#         return "y_to_x"  # Y's outputs -> X's inputs
#     return "x_to_y" if direction == "y_depends_on_x" else None


def _contexts_overlap(routing: RoutingDefinition, cmd_x: str, cmd_y: str) -> bool:
    cx = routing.get_contexts_for_command(cmd_x)
    cy = routing.get_contexts_for_command(cmd_y)
    return False if not cx or not cy else bool(cx & cy)


def _check_param_dependencies(
    outputs: List[ParamMeta], 
    inputs: List[ParamMeta],
    semantic_threshold: float,
    exact_only: bool
) -> bool:
    """Check if any output parameter can satisfy any input parameter."""
    if not outputs or not inputs:
        return False
    
    for out_param in outputs:
        for in_param in inputs:
            # Check exact match
            if _exact_match(out_param, in_param):
                return True
            # Check semantic match if not in exact_only mode
            if not exact_only and _semantic_match(out_param, in_param, semantic_threshold):
                return True
    return False


def generate_dependency_graph(workflow_path: str) -> str:
    """
    Build the parameter dependency graph and persist as JSON in command_dependency_graph.json.

    Returns the path to the generated JSON file.
    """
    # Use default values
    semantic_threshold = 0.85
    exact_only = False
    params_by_command = _collect_command_params(workflow_path)
    routing = RoutingDefinition.build(workflow_path)

    # Exclude core commands and wildcard
    excluded_commands = {
        "IntentDetection/go_up",
        "IntentDetection/reset_context", 
        "IntentDetection/what_can_i_do",
        "IntentDetection/what_is_current_context",
        "wildcard"
    }
    
    # Filter out excluded commands
    filtered_params = {k: v for k, v in params_by_command.items() if k not in excluded_commands}
    
    nodes = sorted(list(filtered_params.keys()))
    edges: List[Dict[str, Any]] = []

    # Process pairs only once, checking both directions efficiently
    for i, cmd_x in enumerate(nodes):
        for cmd_y in nodes[i+1:]:  # Only check each pair once
            if not _contexts_overlap(routing, cmd_x, cmd_y):
                continue

            # Print on-going progress status since this is a long-running operation
            print(f"Checking {cmd_x} <-> {cmd_y}")

            x_params = filtered_params[cmd_x]
            y_params = filtered_params[cmd_y]
            
            # Check both directions efficiently using helper function
            x_to_y_match = _check_param_dependencies(
                x_params.outputs, y_params.inputs, semantic_threshold, exact_only
            )
            y_to_x_match = _check_param_dependencies(
                y_params.outputs, x_params.inputs, semantic_threshold, exact_only
            )
            
            # Add edges for matches found
            if x_to_y_match:
                edges.append({"from": cmd_y, "to": cmd_x})
            if y_to_x_match:
                edges.append({"from": cmd_x, "to": cmd_y})
            
            # If no exact/semantic match found, try LLM (which also checks both directions)
            # if not x_to_y_match and not y_to_x_match and not exact_only:
            #     llm_direction = _llm_command_dependency(cmd_x, x_params, cmd_y, y_params)
            #     if llm_direction == "x_to_y":
            #         edges.append({"from": cmd_y, "to": cmd_x})
            #     elif llm_direction == "y_to_x":
            #         edges.append({"from": cmd_x, "to": cmd_y})

    out_path = os.path.join(workflow_path, "command_dependency_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    return out_path


def _load_graph(graph_path: str) -> Dict[str, Any]:
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"nodes": [], "edges": []}


def get_dependency_suggestions(
    graph_path: str,
    y_qualified_name: str,
    missing_input_param: str,
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """
    Recursively resolves dependencies, returning a list of plans for resolving the missing param.
    Note: The simplified graph no longer tracks which specific params match, so this function
    returns all possible dependency paths without filtering by parameter.
    """
    graph = _load_graph(graph_path)
    edges = graph.get("edges", [])

    # Build adjacency: from -> list of to nodes
    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    def recurse(node: str, depth: int) -> List[Dict[str, Any]]:
        if depth > max_depth:
            return []
        plans: List[Dict[str, Any]] = []
        for neighbor in adj.get(node, []):
            sub_plans = recurse(neighbor, depth + 1)
            plans.append({
                "command": neighbor,
                "sub_plans": sub_plans,
            })
        return plans

    dependency_plans = recurse(y_qualified_name, 0)
    # Prefer shallower trees
    dependency_plans.sort(key=lambda p: len(p.get("sub_plans", [])))
    return dependency_plans
